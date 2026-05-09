"""Structured error log dump used when SGLang returns 4xx/5xx."""

import json
from datetime import datetime
from pathlib import Path

_REQ_TRUNC = 8000
_RESP_TRUNC = 4000


def dump_error(
    log_path: Path,
    request_body: bytes,
    status_code: int,
    response_body: bytes,
    path: str,
) -> None:
    """Append a structured error block to log_path. Best-effort: any failure
    inside this function is swallowed."""
    try:
        ts = datetime.now().isoformat()
        req_text = _format(request_body, _REQ_TRUNC)
        resp_text = _format(response_body, _RESP_TRUNC)
        with log_path.open("a") as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"[{ts}] ERROR DUMP: SGLang returned HTTP {status_code}\n")
            f.write(f"Path: {path}\n")
            f.write("--- Request Body ---\n")
            f.write(req_text)
            f.write(f"\n--- Response (HTTP {status_code}) ---\n")
            f.write(resp_text)
            f.write(f"\n{'=' * 60}\n")
    except Exception:
        pass


def _format(raw: bytes, limit: int) -> str:
    text = raw.decode("utf-8", errors="replace")[:limit]
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)[:limit]
    except (json.JSONDecodeError, ValueError):
        return text
