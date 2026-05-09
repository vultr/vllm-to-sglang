"""Tests for the FixToolSchemas request filter."""

import json

from vllm_shim.backend.sglang.filter.fix_schema import FixToolSchemas


def test_applies_to_chat_completions_post() -> None:
    assert FixToolSchemas().applies_to("POST", "/v1/chat/completions")


def test_fixes_properties_array_to_object() -> None:
    body = json.dumps({
        "tools": [{
            "function": {
                "name": "f",
                "parameters": {"type": "object", "properties": []},
            }
        }]
    }).encode()
    out = json.loads(FixToolSchemas().transform(body))
    assert out["tools"][0]["function"]["parameters"]["properties"] == {}


def test_replaces_non_dict_parameters() -> None:
    body = json.dumps({"tools": [{"function": {"name": "f", "parameters": []}}]}).encode()
    out = json.loads(FixToolSchemas().transform(body))
    params = out["tools"][0]["function"]["parameters"]
    assert params == {"type": "object", "properties": {}}


def test_recurses_into_nested_items() -> None:
    body = json.dumps({
        "tools": [{
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "list": {"type": "array", "items": {"properties": []}}
                    },
                },
            }
        }]
    }).encode()
    out = json.loads(FixToolSchemas().transform(body))
    items = out["tools"][0]["function"]["parameters"]["properties"]["list"]["items"]
    assert items["properties"] == {}


def test_drops_non_list_required() -> None:
    body = json.dumps({
        "tools": [{
            "function": {
                "name": "f",
                "parameters": {"type": "object", "properties": {}, "required": "name"},
            }
        }]
    }).encode()
    out = json.loads(FixToolSchemas().transform(body))
    assert "required" not in out["tools"][0]["function"]["parameters"]


def test_passes_through_when_no_tools() -> None:
    body = b'{"model":"m"}'
    assert FixToolSchemas().transform(body) == body


def test_passes_through_invalid_json() -> None:
    body = b"not json"
    assert FixToolSchemas().transform(body) == body
