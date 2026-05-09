"""FixToolSchemas filter: repair JSON-Schema fragments SGLang's strict parser would reject."""

import json
from typing import Any

from vllm_shim.backend.base.filter import RequestFilter


class FixToolSchemas(RequestFilter):
    """Repairs broken JSON-Schema bodies that some vLLM clients send,
    where SGLang's strict parser would reject them."""

    def applies_to(self, method: str, path: str) -> bool:
        return method == "POST" and "chat/completions" in path

    def transform(self, body: bytes) -> bytes:
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return body
        if not isinstance(data, dict):
            return body
        tools = data.get("tools")
        if not isinstance(tools, list):
            return body

        changed = False
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            func = tool.get("function")
            if not isinstance(func, dict):
                continue
            if not isinstance(func.get("parameters"), dict):
                func["parameters"] = {"type": "object", "properties": {}}
                changed = True
            if self._fix_schema(func["parameters"]):
                changed = True

        return json.dumps(data).encode() if changed else body

    @classmethod
    def _fix_schema(cls, schema: dict[str, Any]) -> bool:
        fixed = False
        if "properties" in schema and not isinstance(schema["properties"], dict):
            schema["properties"] = {}
            fixed = True
        if "required" in schema and not isinstance(schema["required"], list):
            del schema["required"]
            fixed = True
        if isinstance(schema.get("properties"), dict):
            for val in schema["properties"].values():
                if isinstance(val, dict) and cls._fix_schema(val):
                    fixed = True
        if isinstance(schema.get("items"), dict) and cls._fix_schema(schema["items"]):
            fixed = True
        for key in ("anyOf", "allOf", "oneOf"):
            seq = schema.get(key)
            if isinstance(seq, list):
                for item in seq:
                    if isinstance(item, dict) and cls._fix_schema(item):
                        fixed = True
        if isinstance(schema.get("additionalProperties"), dict) and cls._fix_schema(
            schema["additionalProperties"]
        ):
            fixed = True
        return fixed
