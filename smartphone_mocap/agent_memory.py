"""Transactional session state. Large evidence stays in referenced files."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sqlite3
import time


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class AgentMemory:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS requests (
                    session TEXT, id TEXT, prompt TEXT, mode TEXT, state TEXT,
                    snapshot TEXT, result TEXT, created_ns INTEGER, PRIMARY KEY(session,id));
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, session TEXT, request TEXT, kind TEXT,
                    payload TEXT, created_ns INTEGER);
                CREATE TABLE IF NOT EXISTS properties (
                    session TEXT, object_id TEXT, version INTEGER, payload TEXT,
                    request TEXT, valid INTEGER DEFAULT 1, PRIMARY KEY(session,object_id,version));
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY, session TEXT, object_id TEXT, field TEXT,
                    value TEXT, scope TEXT, quote TEXT, request TEXT, created_ns INTEGER);
                CREATE TABLE IF NOT EXISTS settings (
                    session TEXT, key TEXT, payload TEXT, PRIMARY KEY(session,key));
                CREATE TABLE IF NOT EXISTS actions (
                    id TEXT PRIMARY KEY, session TEXT, request TEXT, payload TEXT,
                    state TEXT, result TEXT);
            ''')
            if 'conversation_id' not in {r['name'] for r in db.execute('PRAGMA table_info(requests)')}:
                db.execute("ALTER TABLE requests ADD COLUMN conversation_id TEXT DEFAULT 'default'")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def claim(self, session, request, prompt, mode, snapshot):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            prior = db.execute('SELECT * FROM requests WHERE session=? AND id=?', (session,request)).fetchone()
            if prior:
                if prior['prompt'] != prompt:
                    raise ValueError('Request ID already belongs to another prompt')
                return dict(prior)
            running = db.execute("SELECT id FROM requests WHERE session=? AND state='running'", (session,)).fetchone()
            if running:
                return {'busy': True, 'id': running['id']}
            db.execute('INSERT INTO requests(session,id,prompt,mode,state,snapshot,result,created_ns,conversation_id) VALUES(?,?,?,?,?,?,?,?,?)',
                       (session,request,prompt,mode,'running',encode(snapshot),None,time.time_ns(),snapshot.get('conversation_id','default')))
        return None

    def event(self, session, request, kind, payload):
        with self.connect() as db:
            db.execute('INSERT INTO events(session,request,kind,payload,created_ns) VALUES(?,?,?,?,?)',
                       (session,request,kind,encode(payload),time.time_ns()))

    def save_properties(self, session, request, decisions):
        versions = {}
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for object_id, value in decisions.items():
                value = {k:v for k,v in value.items() if k != "property_version"}
                prior = db.execute('SELECT version,payload,valid FROM properties WHERE session=? AND object_id=? ORDER BY version DESC LIMIT 1', (session,object_id)).fetchone()
                if prior and prior['valid'] and json.loads(prior['payload']) == value:
                    versions[object_id] = prior['version']
                    continue
                if prior and prior['valid']:
                    old = json.loads(prior['payload'])
                    transient = {'xyz','rotation_wxyz','geometry_digest'}
                    if {k:v for k,v in old.items() if k not in transient} == {k:v for k,v in value.items() if k not in transient}:
                        versions[object_id] = prior["version"]
                        continue
                version = prior['version']+1 if prior else 1
                db.execute('INSERT INTO properties VALUES(?,?,?,?,?,1)', (session,object_id,version,encode(value),request))
                versions[object_id] = version
        return versions

    def set_table(self, session, value):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO settings VALUES(?,?,?)', (session,'table_material',encode(value)))

    def fact(self, session, request, object_id, field, value, quote, scope):
        with self.connect() as db:
            db.execute('INSERT INTO facts(session,object_id,field,value,scope,quote,request,created_ns) VALUES(?,?,?,?,?,?,?,?)',
                       (session,object_id,field,encode(value),scope,quote,request,time.time_ns()))
            if scope == 'object':
                latest = db.execute('SELECT payload FROM properties WHERE session=? AND object_id=? ORDER BY version DESC LIMIT 1', (session,object_id)).fetchone()
                unchanged_material = (field == 'material' and latest and
                    str(json.loads(latest['payload']).get('material','')).strip().casefold() == str(value).strip().casefold())
                if not unchanged_material:
                    db.execute('UPDATE properties SET valid=0 WHERE session=? AND object_id=?', (session,object_id))

    def hydrate(self, session, snapshot):
        result = copy.deepcopy(snapshot)
        state = result.setdefault('twin_state', {})
        with self.connect() as db:
            rows = db.execute('SELECT p.* FROM properties p JOIN (SELECT object_id,MAX(version) v FROM properties WHERE session=? GROUP BY object_id) q ON p.object_id=q.object_id AND p.version=q.v WHERE p.session=?', (session,session)).fetchall()
            properties = state.setdefault('physics_properties', {})
            for row in rows:
                if row['valid']:
                    properties[row['object_id']] = {**json.loads(row['payload']), 'property_version':row['version']}
                else:
                    properties.pop(row['object_id'], None)
            table = db.execute('SELECT payload FROM settings WHERE session=? AND key=?',(session,'table_material')).fetchone()
            if table:
                state['physics_table_material'] = json.loads(table['payload'])
            facts = db.execute("SELECT * FROM facts WHERE session=? AND scope='object' ORDER BY id",(session,)).fetchall()
            latest = {(r['object_id'],r['field']): dict(r) for r in facts}
            valid_ids = {r['object_id'] for r in rows if r['valid']}
            for object_id, _ in latest:
                if object_id not in valid_ids:
                    properties.pop(object_id, None)
            state['object_facts'] = [{**r,'value':json.loads(r['value'])} for r in latest.values()]
        return result

    def history(self, session, query='', limit=6, conversation_id=None):
        with self.connect() as db:
            rows = db.execute("SELECT id,prompt,result,state FROM requests WHERE session=? AND result IS NOT NULL AND instr(prompt,?)>0 AND (? IS NULL OR conversation_id=?) ORDER BY created_ns DESC LIMIT ?", (session,query,conversation_id,conversation_id,min(20,max(1,limit)))).fetchall()
        return [{'request_id':r['id'],'prompt':r['prompt'],'reply':json.loads(r['result']).get('reply',''),'state':r['state']} for r in reversed(rows)]

    def pending_experiment(self, session, conversation_id):
        """Only the latest completed turn in this conversation may be continued."""
        with self.connect() as db:
            row = db.execute("SELECT result FROM requests WHERE session=? AND conversation_id=? AND result IS NOT NULL ORDER BY created_ns DESC LIMIT 1",
                             (session, conversation_id)).fetchone()
        result = json.loads(row['result']) if row else {}
        return result.get('pending_experiment') if result.get('task_state') == 'needs_clarification' else None

    def finish(self, session, request, result):
        value = copy.deepcopy(result)
        with self.connect() as db:
            for index, action in enumerate(value.get('actions', [])):
                action_id = f'{session}:{request}:{index}'
                action['action_id'] = action_id
                db.execute('INSERT INTO actions VALUES(?,?,?,?,?,NULL)', (action_id,session,request,encode(action),'queued'))
            db.execute('UPDATE requests SET state=?,result=? WHERE session=? AND id=?',
                       ('failed' if value.get('provider_error') else 'planned' if value.get('actions') else 'answered',encode(value),session,request))
        return value

    def acknowledge(self, session, action_id, ok, detail):
        with self.connect() as db:
            cursor = db.execute("UPDATE actions SET state=?,result=? WHERE session=? AND id=? AND state='queued'", ('applied' if ok else 'failed',encode(detail),session,action_id))
            changed = cursor.rowcount == 1
            if changed and not ok:
                db.execute("UPDATE actions SET state='blocked',result=? WHERE session=? AND request=(SELECT request FROM actions WHERE id=?) AND state='queued' AND rowid>(SELECT rowid FROM actions WHERE id=?)",
                           (encode({'reason':'Earlier action failed'}),session,action_id,action_id))
            return changed

    def backend_context(self, session, request):
        with self.connect() as db:
            row = db.execute('SELECT id,mode,state,prompt FROM requests WHERE session=? AND id=?',(session,request)).fetchone()
            actions = db.execute('SELECT id,state,result FROM actions WHERE session=? AND request=?',(session,request)).fetchall()
        states = [a['state'] for a in actions]
        execution = ('failed' if any(s in {'failed','blocked'} for s in states) else
                     'applied' if states and all(s == 'applied' for s in states) else
                     'pending' if states else 'no_actions')
        return {'request':dict(row) if row else None, 'execution':execution,
                'actions':[{**dict(a),'result':json.loads(a['result']) if a['result'] else None} for a in actions]}

    def recover_interrupted(self):
        # Single service process owns this DB; a restart never replays actions.
        with self.connect() as db:
            db.execute("UPDATE requests SET state='interrupted',result=? WHERE state='running'",
                       (encode({'reply':'서비스 재시작으로 요청이 중단됐습니다. 준비된 물성은 보존되어 있습니다.', 'actions':[], 'provider_error':'service restarted'}),))

    def tool_result(self, session, request, name, limit=2):
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM events WHERE session=? AND request=? AND kind='tool' ORDER BY id DESC",(session,request)).fetchall()
        return [json.loads(r['payload']) for r in rows if json.loads(r['payload']).get('name') == name][:limit]
