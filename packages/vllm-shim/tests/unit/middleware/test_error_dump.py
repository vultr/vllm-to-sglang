"""Tests for the structured error-dump helper."""

from pathlib import Path

from vllm_shim.middleware.error_dump import dump_error


def test_writes_request_and_response_to_log(tmp_path: Path) -> None:
    log = tmp_path / "shim.log"
    dump_error(
        log_path=log,
        backend_name="sglang",
        request_body=b'{"model":"m"}',
        status_code=422,
        response_body=b'{"error":"bad"}',
        path="/v1/chat/completions",
    )
    text = log.read_text()
    assert "HTTP 422" in text
    assert "sglang returned" in text
    assert "/v1/chat/completions" in text
    assert '"model":' in text or '"model":"m"' in text


def test_handles_invalid_json_gracefully(tmp_path: Path) -> None:
    log = tmp_path / "shim.log"
    dump_error(
        log_path=log,
        backend_name="trtllm",
        request_body=b"not json",
        status_code=500,
        response_body=b"plain text",
        path="/v1/x",
    )
    text = log.read_text()
    assert "HTTP 500" in text
    assert "trtllm returned" in text
    assert "plain text" in text
