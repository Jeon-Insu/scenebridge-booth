"""Bounded model working context; the full transcript stays in SQLite."""
import copy
import json
import re


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k,v in value.items() if k not in {
            'source_contact_geometry','points_local_m','usd_world_transform','geometry_digest'}}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def consume_images(messages):
    # Called only after successful inference, so images get one complete read.
    for message in messages:
        if isinstance(message.get('content'), list):
            message['content'] = [({'type':'text','text':'[Previously supplied image retained in evidence files; inspect_images can retrieve it again.]'}
                                   if part.get('type') == 'image_url' else part) for part in message['content']]


def recent_history(conversation, max_chars=5000):
    result, used = [], 0
    for item in reversed(conversation or []):
        role = str(item.get('role','')).lower()
        if role not in {'you','agent','user','assistant'}:
            continue
        content = str(item.get('text', item.get('content','')))
        if used+len(content) > max_chars:
            continue
        result.append({'role':'user' if role in {'you','user'} else 'assistant','content':content})
        used += len(content)
        if len(result) >= 6:
            break
    return list(reversed(result))


def context_error_budget(detail):
    total = re.search(r'maximum context length of (\d+)', detail)
    used = re.search(r'(\d+) tokens from the input', detail)
    if total and used:
        return int(total[1])-int(used[1])-64
    return None


def post_with_budget(post, endpoint, payload, timeout):
    """Keep evidence intact; retry an explicit context rejection with less output.

    Never retries timeouts or a successful response, so tools cannot run twice.
    Server token accounting includes the actual vision/template tokens.
    """
    import urllib.error
    try:
        return post(endpoint,payload,timeout)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        detail = raw.decode('utf-8',errors='replace')
        budget = context_error_budget(detail)
        if exc.code == 400 and budget is not None and 256 <= budget < payload['max_tokens']:
            retry = copy.deepcopy(payload)
            retry['max_tokens'] = budget
            return post(endpoint,retry,timeout)
        # Restore body for the caller's error reporting.
        import io
        raise urllib.error.HTTPError(exc.url, exc.code, exc.msg, exc.headers, io.BytesIO(raw)) from exc


def preflight_budget(post, endpoint, payload, timeout, *, history_count=0, image_tokens=1024):
    """Count chat/template/tool tokens on the local server before generation.

    Some tokenizer endpoints count image markers rather than vision patches;
    reserve a configurable amount per image and keep exact rejection handling.
    The original question, current tool pairs and requested images are preserved.
    """
    def count():
        request = {k:payload[k] for k in ('model','messages','tools','tool_choice','chat_template_kwargs')}
        result = post(endpoint.rsplit('/',1)[0].replace('/chat','')+'/tokenize', request, min(timeout,10))
        tokens = int(result['count'])
        limit = int(result['max_model_len'])
        images = sum(part.get('type') == 'image_url' for m in payload['messages']
                     if isinstance(m.get('content'),list) for part in m['content'])
        return tokens, limit, images*image_tokens
    tokens, limit, reserve = count()
    removed = 0
    if tokens+reserve+payload['max_tokens']+64 > limit and history_count:
        del payload['messages'][2:2+history_count]
        removed = history_count
        tokens, limit, reserve = count()
    available = limit-tokens-reserve-64
    if available < 256:
        error = ValueError('Current evidence exceeds the context budget; select fewer camera views or objects. No actions applied.')
        error.context_budget = {'tokenizer_input_tokens':tokens, 'image_reserve_tokens':reserve,
                                'max_model_len':limit, 'available_output_tokens':available,
                                'history_removed':removed, 'rejected':True}
        raise error
    payload['max_tokens'] = min(payload['max_tokens'], available)
    return {'tokenizer_input_tokens':tokens,'image_reserve_tokens':reserve,
            'max_model_len':limit,'output_tokens':payload['max_tokens'],'history_removed':removed}


def is_hypothetical(prompt):
    text = re.sub(r'가정(?:한|하는|했던)?\s*(?:것|게|상황)?[이가은는]?\s*(?:아니\w*|말고)', '', prompt)
    return bool(re.search(r'가정(?:해|하|한|했|으로)|만약|what[ -]?if|assum(?:e|ing)|hypothet|suppose', text, re.I))
