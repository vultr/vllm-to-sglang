"""TRT-LLM Prometheus exposition translator: renames series, derives kv-cache usage,
synthesizes vLLM-only series."""

import re

from vllm_shim.backend._shared import translate_prom_line, vllm_synthesized_tail
from vllm_shim.backend.base.metrics import MetricsTranslator

TRTLLM_TO_VLLM: dict[str, str] = {
    "trtllm_request_success_total": "vllm:request_success_total",
    "trtllm_e2e_request_latency_seconds": "vllm:e2e_request_latency_seconds",
    "trtllm_time_to_first_token_seconds": "vllm:time_to_first_token_seconds",
    "trtllm_request_queue_time_seconds": "vllm:request_queue_time_seconds",
    "trtllm_kv_cache_hit_rate": "vllm:gpu_prefix_cache_hit_rate",
}

_RE_FIRST_PASS = re.compile(r"^(\w[\w:]*)(\{[^}]*\})?\s+(.+)$")


class TRTLLMMetricsTranslator(MetricsTranslator):
    """Translates TRT-LLM's Prometheus output (trtllm_* prefix) into vLLM-named series."""

    def translate(self, prom_text: str) -> str:
        # First pass: collect kv_cache_utilization values per label set.
        utilization: dict[str, float] = {}
        for line in prom_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _RE_FIRST_PASS.match(line)
            if not m:
                continue
            name, labels, value_str = m.group(1), m.group(2) or "", m.group(3)
            if name == "trtllm_kv_cache_utilization":
                try:
                    utilization[labels] = float(value_str.split()[0])
                except (ValueError, IndexError):
                    continue

        output: list[str] = []
        for line in prom_text.splitlines():
            output.extend(translate_prom_line(line, TRTLLM_TO_VLLM))

        if utilization:
            output.append("# HELP vllm:kv_cache_usage_perc KV cache usage percentage")
            output.append("# TYPE vllm:kv_cache_usage_perc gauge")
            for lbl, ratio in sorted(utilization.items()):
                pct = ratio * 100.0
                output.append(f"vllm:kv_cache_usage_perc{lbl} {pct:.4f}")

        output.extend(vllm_synthesized_tail())

        return "\n".join(output) + "\n"
