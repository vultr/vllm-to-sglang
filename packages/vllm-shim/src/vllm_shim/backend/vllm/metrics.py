"""vLLM MetricsTranslator: identity pass-through.

vLLM emits ``vllm:*``-prefixed Prometheus series natively, which is exactly
what the middleware serves to operators. No renames, no derived series.

Deliberately does NOT append ``vllm_synthesized_tail()``. Whether vLLM emits
``vllm:num_requests_swapped`` depends on the release (older builds did; v0.21
does not); appending our own constant-zero copy would risk duplicating the
series on any release that does emit it. ``vllm:healthy_pods_total`` is
genuinely missing from vLLM's native output; if Production Stack dashboards
complain, the right fix is either an upstream PR or selective tail synthesis
here.
"""

from vllm_shim.backend.base.metrics import MetricsTranslator


class VLLMMetricsTranslator(MetricsTranslator):
    """Identity translator: returns the backend's exposition unchanged."""

    def translate(self, prom_text: str) -> str:
        return prom_text
