"""StripVLLMParams filter: drop vLLM-only request keys that SGLang rejects."""

import json

from vllm_shim.backend.base.filter import RequestFilter

STRIP_PARAMS: frozenset[str] = frozenset(
    {"logprobs", "top_logprobs", "chat_template_kwargs", "guided_json", "guided_regex"}
)


class StripVLLMParams(RequestFilter):
    """Removes vLLM-only request parameters that SGLang's parsers reject."""

    def applies_to(self, method: str, path: str) -> bool:
        return method == "POST" and "chat/completions" in path

    def transform(self, body: bytes) -> bytes:
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return body
        if not isinstance(data, dict):
            return body
        changed = False
        for key in STRIP_PARAMS:
            if key in data:
                del data[key]
                changed = True
        return json.dumps(data).encode() if changed else body
