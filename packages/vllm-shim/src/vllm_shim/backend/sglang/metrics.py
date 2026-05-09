"""SGLang Prometheus exposition translator: renames series, derives kv-cache usage,
synthesizes vLLM-only series."""

import re

from vllm_shim.backend._shared import translate_prom_line, vllm_synthesized_tail
from vllm_shim.backend.base.metrics import MetricsTranslator

SGLANG_TO_VLLM: dict[str, str] = {
    "sglang:num_running_reqs": "vllm:num_requests_running",
    "sglang:num_queue_reqs": "vllm:num_requests_waiting",
    "sglang:cache_hit_rate": "vllm:gpu_prefix_cache_hit_rate",
    "sglang:e2e_request_latency_seconds": "vllm:e2e_request_latency_seconds",
    "sglang:inter_token_latency_seconds": "vllm:request_time_per_output_token_seconds",
    "sglang:time_to_first_token_seconds": "vllm:time_to_first_token_seconds",
    "sglang:prompt_tokens_total": "vllm:prompt_tokens_total",
    "sglang:generation_tokens_total": "vllm:generation_tokens_total",
    "sglang:num_requests_total": "vllm:request_success_total",
    "sglang:num_aborted_requests_total": "vllm:request_aborted_total",
    "sglang:cached_tokens_total": "vllm:prompt_tokens_cached_total",
}

_RE_FIRST_PASS = re.compile(r"^(\w[\w:]*)(\{[^}]*\})?\s+(.+)$")


class SGLangMetricsTranslator(MetricsTranslator):
    """Translates SGLang's Prometheus output into vLLM-named series."""

    def translate(self, prom_text: str) -> str:
        # Two-pass walk: first to collect inputs for kv_cache_usage_perc, then to rewrite lines.
        used: dict[str, float] = {}
        capacity: dict[str, float] = {}
        for line in prom_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _RE_FIRST_PASS.match(line)
            if not m:
                continue
            name, labels, value_str = m.group(1), m.group(2) or "", m.group(3)
            if name in ("sglang:num_used_tokens", "sglang:max_total_num_tokens"):
                try:
                    val = float(value_str.split()[0])
                except (ValueError, IndexError):
                    continue
                if name == "sglang:num_used_tokens":
                    used[labels] = val
                else:
                    capacity[labels] = val

        output: list[str] = []
        for line in prom_text.splitlines():
            output.extend(translate_prom_line(line, SGLANG_TO_VLLM))

        if used and capacity:
            output.append("# HELP vllm:kv_cache_usage_perc KV cache usage percentage")
            output.append("# TYPE vllm:kv_cache_usage_perc gauge")
            for lbl in sorted(set(used) | set(capacity)):
                u = used.get(lbl, 0.0)
                c = capacity.get(lbl, 0.0)
                pct = (u / c * 100.0) if c > 0 else 0.0
                output.append(f"vllm:kv_cache_usage_perc{lbl} {pct:.4f}")

        output.extend(vllm_synthesized_tail())

        return "\n".join(output) + "\n"
