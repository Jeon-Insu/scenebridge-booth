from __future__ import annotations

import argparse
import json
import time
import uuid
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
from typing import Any

from .agent_control import read_json, atomic_write_json
from .scene_agent import respond
from .agent_memory import AgentMemory
from .physics_scenario import simulation_requested
from .asset_studio import AssetStudio


class TabletopAgentHandler(BaseHTTPRequestHandler):
    server_version = "GISTTabletopAgent/1.0"

    def _write(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        url = urlsplit(self.path)
        if url.path in {"/asset-studio", "/v1/assets", "/v1/assets/image", "/v1/assets/file"}:
            try:
                studio = self.server.asset_studio
                if url.path == "/v1/assets":
                    self._write(200, studio.listing()); return
                values = parse_qs(url.query)
                if url.path == "/asset-studio":
                    path = Path(__file__).parent / "web/asset_studio.html"
                    mime = "text/html; charset=utf-8"
                elif url.path == "/v1/assets/image":
                    path = studio.image_path(values['snapshot'][0], values['camera'][0]); mime = 'image/jpeg'
                else:
                    folder = studio.path(values['id'][0]); path = (folder / values['name'][0]).resolve()
                    if path.parent != folder or path.suffix != '.png': raise ValueError('Invalid image')
                    mime = 'image/png'
                body = path.read_bytes()
                self.send_response(200); self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
            except (OSError, ValueError, KeyError) as exc:
                self._write(400, {'ok':False, 'error':str(exc)})
            return
        if self.path.rstrip("/") == "/health":
            self._write(
                200,
                {
                    "ok": True,
                    "service": "gist.tabletop_agent",
                    "configured_provider": self.server.agent_config.get("provider", "auto"),
                },
            )
            return
        url = urlsplit(self.path)
        if url.path == "/v1/agent/task":
            values = parse_qs(url.query)
            self._write(200, self.server.memory.backend_context(values.get("session_id",[""])[0], values.get("request_id",[""])[0]))
            return
        self._write(404, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.rstrip('/') == '/v1/assets/command':
            try:
                origin = self.headers.get('Origin')
                if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                    raise ValueError('Cross-origin writes are not allowed')
                if not self.headers.get('Content-Type','').startswith('application/json'):
                    raise ValueError('JSON required')
                length = int(self.headers.get('Content-Length',0))
                if not 0 < length <= 1_000_000: raise ValueError('Invalid request size')
                result = self.server.asset_studio.command(json.loads(self.rfile.read(length)))
                self._write(200, {'ok':True,'result':result})
            except Exception as exc:
                self._write(400, {'ok':False,'error':str(exc)})
            return
        if self.path.rstrip("/") == "/v1/agent/ack":
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                changed = self.server.memory.acknowledge(str(payload["session_id"]), str(payload["action_id"]),
                                                       payload["ok"] is True, payload.get("detail"))
                self._write(200, {"ok":changed})
            except (ValueError, KeyError, TypeError) as exc:
                self._write(400, {"ok":False,"error":str(exc)})
            return
        if self.path.rstrip("/") != "/v1/agent/respond":
            self._write(404, {"ok": False, "error": "Not found"})
            return
        session_id = request_id = None
        claimed = False
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            prompt = str(request.get("prompt", "")).strip()
            snapshot = request.get("snapshot")
            if snapshot is None and request.get("snapshot_path"):
                snapshot = read_json(str(request["snapshot_path"]))
            if not prompt:
                raise ValueError("prompt is required")
            if not isinstance(snapshot, dict):
                raise ValueError("snapshot or snapshot_path must contain a JSON object")
            snapshot = dict(snapshot)
            snapshot['registered_asset_candidates'] = [
                {k:d.get(k) for k in ('id','label','snapshot_id','status','desired','position_source')}
                for d in self.server.asset_studio.listing()['records']
            ]
            snapshot["conversation_id"] = str(request.get("conversation_id") or snapshot.get("conversation_id") or "default")
            conversation = request.get("conversation")
            request_id = str(request.get("request_id") or uuid.uuid4().hex)
            session_id = str(request.get("session_id") or snapshot.get("session_id") or "legacy-"+str(snapshot.get("snapshot_id",request_id)))
            mode = "simulation" if simulation_requested(prompt) else "qa"
            previous = self.server.memory.claim(session_id,request_id,prompt,mode,snapshot)
            if previous and previous.get('busy'):
                self._write(409, {'ok':False,'error':'이 세션의 이전 요청이 아직 처리 중입니다.','request_id':previous['id']})
                return
            if previous:
                result = json.loads(previous['result']) if previous['result'] else {"reply":"동일한 요청이 처리 중입니다."}
                self._write(200, {"ok":True, **result, "actions":[], "duplicate_request":True})
                return
            claimed = True
            self.server.memory.event(session_id,request_id,"conversation_input",conversation or [])
            def progress(event):
                progress_path = self.server.agent_config.get("progress_path")
                if progress_path and request_id:
                    atomic_write_json(progress_path, {**event, "request_id": request_id, "updated_at_ns": time.time_ns()})
            result = respond(
                prompt,
                snapshot,
                self.server.agent_config,
                conversation=conversation if isinstance(conversation, list) else None,
                progress=progress, memory=self.server.memory, session_id=session_id, request_id=request_id,
            )
            result = self.server.memory.finish(session_id,request_id,result)
            progress({"phase": "complete", "evidence_trace": result.get("evidence_trace", [])})
            self._write(200, {"ok": True, **result})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            if claimed:
                self.server.memory.finish(session_id,request_id,{'reply':str(exc),'provider_error':str(exc),'actions':[]})
            self._write(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            if claimed:
                self.server.memory.finish(session_id,request_id,{'reply':'요청 처리 중 오류가 발생했습니다. 저장된 상태는 유지됩니다.','provider_error':str(exc),'actions':[]})
            self._write(500, {'ok':False,'error':str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"Agent HTTP: {format % args}")


class TabletopAgentServer(ThreadingHTTPServer):
    def __init__(self, address, config: dict[str, Any]):
        super().__init__(address, TabletopAgentHandler)
        self.agent_config = config
        self.memory = AgentMemory(config.get("memory_db_path", "outputs/tabletop_agent/state.sqlite3"))
        self.memory.recover_interrupted()
        self.asset_studio = AssetStudio(config.get('asset_studio_root'))


def run_agent_service(config_path: str | Path) -> None:
    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    key_file = Path(__file__).resolve().parents[1] / '.secrets/meshy_api_key'
    if key_file.is_file() and not os.environ.get('MESHY_API_KEY'):
        os.environ['MESHY_API_KEY'] = key_file.read_text().strip()
    host = str(config.get("listen_host", "127.0.0.1"))
    port = int(config.get("listen_port", 7080))
    server = TabletopAgentServer((host, port), config)
    print(
        f"Tabletop agent service listening on http://{host}:{port}; "
        f"provider={config.get('provider', 'auto')}"
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local tabletop scene agent service")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("smartphone_mocap/config/tabletop_agent.json"),
    )
    args = parser.parse_args()
    run_agent_service(args.config)


if __name__ == "__main__":
    main()
