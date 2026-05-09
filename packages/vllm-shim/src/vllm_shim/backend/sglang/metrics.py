import re

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

_RE_METRIC_LINE = re.compile(r"^(#\s+(?:HELP|TYPE)\s+)?(\w[\w:]*)(.*)")
_RE_SAMPLE_LINE = re.compile(r"^(\w[\w:]*)(\{[^}]*\})?\s+(.+)$")


class SGLangMetricsTranslator(MetricsTranslator):
    def translate(self, prom_text: str) -> str:
        used: dict[str, float] = {}
        capacity: dict[str, float] = {}
        for line in prom_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _RE_SAMPLE_LINE.match(line)
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
            output.extend(self._translate_line(line))

        if used and capacity:
            output.append("# HELP vllm:kv_cache_usage_perc KV cache usage percentage")
            output.append("# TYPE vllm:kv_cache_usage_perc gauge")
            for lbl in sorted(set(used) | set(capacity)):
                u = used.get(lbl, 0.0)
                c = capacity.get(lbl, 0.0)
                pct = (u / c * 100.0) if c > 0 else 0.0
                output.append(f"vllm:kv_cache_usage_perc{lbl} {pct:.4f}")

        output.append("# HELP vllm:healthy_pods_total Number of healthy vLLM pods")
        output.append("# TYPE vllm:healthy_pods_total gauge")
        output.append('vllm:healthy_pods_total{endpoint="default"} 1')

        output.append("# HELP vllm:num_requests_swapped Number of swapped requests")
        output.append("# TYPE vllm:num_requests_swapped gauge")
        output.append("vllm:num_requests_swapped 0")

        return "\n".join(output) + "\n"

    @staticmethod
    def _translate_line(line: str) -> list[str]:
        m = _RE_SAMPLE_LINE.match(line)
        if m:
            name, labels, value = m.group(1), m.group(2) or "", m.group(3)
            vllm_name = SGLANG_TO_VLLM.get(name)
            if vllm_name:
                return [f"{vllm_name}{labels} {value}"]
            for suffix in ("_bucket", "_sum", "_count"):
                if name.endswith(suffix):
                    base = SGLANG_TO_VLLM.get(name[: -len(suffix)])
                    if base:
                        return [f"{base}{suffix}{labels} {value}"]
            return [line]

        m = _RE_METRIC_LINE.match(line)
        if m:
            prefix, name, rest = m.group(1) or "", m.group(2), m.group(3)
            vllm_name = SGLANG_TO_VLLM.get(name)
            if vllm_name:
                return [f"{prefix}{vllm_name}{rest}"]
        return [line]
