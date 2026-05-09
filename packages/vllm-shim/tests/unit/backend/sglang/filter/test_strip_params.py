"""Tests for the StripVLLMParams request filter."""

import json

from vllm_shim.backend.sglang.filter.strip_params import StripVLLMParams


def test_applies_to_chat_completions_post() -> None:
    f = StripVLLMParams()
    assert f.applies_to("POST", "/v1/chat/completions")


def test_does_not_apply_to_other_paths() -> None:
    f = StripVLLMParams()
    assert not f.applies_to("POST", "/v1/embeddings")
    assert not f.applies_to("GET", "/v1/chat/completions")


def test_strips_known_params() -> None:
    body = json.dumps({"model": "m", "logprobs": True, "top_logprobs": 5}).encode()
    out = StripVLLMParams().transform(body)
    data = json.loads(out)
    assert data == {"model": "m"}


def test_strips_chat_template_kwargs() -> None:
    body = json.dumps({"model": "m", "chat_template_kwargs": {"k": "v"}}).encode()
    out = StripVLLMParams().transform(body)
    assert "chat_template_kwargs" not in json.loads(out)


def test_strips_guided_json_and_guided_regex() -> None:
    body = json.dumps({"guided_json": {}, "guided_regex": "x"}).encode()
    out = StripVLLMParams().transform(body)
    data = json.loads(out)
    assert "guided_json" not in data
    assert "guided_regex" not in data


def test_passes_invalid_json_through() -> None:
    body = b"not json"
    assert StripVLLMParams().transform(body) == body
