# Metrics translation

The middleware exposes `/metrics` on the public listener. It scrapes SGLang's native `/metrics` (Prometheus exposition format), rewrites it into vLLM-named series, and serves the result. Existing dashboards and alerts built against vLLM continue to work unchanged.

The pipeline lives in two files:

- `MetricsHandler` (`packages/vllm-shim/src/vllm_shim/middleware/handler/metrics.py`): fetches, caches, and serves.
- `SGLangMetricsTranslator` (`packages/vllm-shim/src/vllm_shim/backend/sglang/metrics.py`): the pure text-to-text translation.

## Translation pipeline

For each `/metrics` request:

1. Cache check (1-second TTL). On hit, return the cached translation.
2. `httpx.AsyncClient.get(SGLang/metrics, timeout=10s)`. On connection error or timeout, return a 503 with the error body.
3. On non-200 from SGLang, forward the response unchanged.
4. On 200, run `SGLangMetricsTranslator.translate(text)` and cache the result.

The 1-second cache exists because Prometheus scrape intervals overlap with k8s readiness probes, and SGLang's `/metrics` is non-trivial to compute. Concurrent scrapes share one fetch.

## What the translator does

The translator passes through the SGLang exposition format line-by-line and applies three transformations:

### 1. Rename known series

A static map (`SGLANG_TO_VLLM` in the translator module) rewrites series names:

| SGLang name | vLLM name |
|---|---|
| `sglang:num_running_reqs` | `vllm:num_requests_running` |
| `sglang:num_queue_reqs` | `vllm:num_requests_waiting` |
| `sglang:cache_hit_rate` | `vllm:gpu_prefix_cache_hit_rate` |
| `sglang:e2e_request_latency_seconds` | `vllm:e2e_request_latency_seconds` |
| `sglang:inter_token_latency_seconds` | `vllm:request_time_per_output_token_seconds` |
| `sglang:time_to_first_token_seconds` | `vllm:time_to_first_token_seconds` |
| `sglang:prompt_tokens_total` | `vllm:prompt_tokens_total` |
| `sglang:generation_tokens_total` | `vllm:generation_tokens_total` |
| `sglang:num_requests_total` | `vllm:request_success_total` |
| `sglang:num_aborted_requests_total` | `vllm:request_aborted_total` |
| `sglang:cached_tokens_total` | `vllm:prompt_tokens_cached_total` |

Both the `# HELP` / `# TYPE` lines and the sample lines are rewritten. Unknown series pass through with their `sglang:` prefix intact.

### 2. Histogram suffix handling

Prometheus histograms expose three derived series per histogram: `_bucket`, `_sum`, `_count`. The translator strips a trailing suffix, looks up the base name, and reattaches the suffix:

```
sglang:e2e_request_latency_seconds_bucket{le="0.5"} 1
                ↓
vllm:e2e_request_latency_seconds_bucket{le="0.5"} 1
```

This avoids needing 33 entries in the rename map (11 histograms × 3 suffixes); only the base names are listed.

### 3. Synthesized series

Three series exist in vLLM's exposition format but have no direct SGLang equivalent. The translator emits them at the end of the output:

| vLLM series | Source | Logic |
|---|---|---|
| `vllm:kv_cache_usage_perc` | Derived | First pass collects `sglang:num_used_tokens` and `sglang:max_total_num_tokens` per label set. Then `usage = used / capacity * 100`, emitted with the original labels. Skipped entirely if either input series is missing. |
| `vllm:healthy_pods_total` | Synthesized | Always `1` with `endpoint="default"`. The fact that the middleware is responding implies the pod is healthy enough to scrape. |
| `vllm:num_requests_swapped` | Synthesized | Always `0`. SGLang has no swap; the metric exists in vLLM dashboards and alerting on a constant zero is fine. |

`kv_cache_usage_perc` is the only metric where the translator actually parses values rather than just renaming text; it walks the exposition twice, once to gather inputs and once to rewrite lines, then appends the derived series.

## Output format

The handler always returns `Content-Type: text/plain; version=0.0.4; charset=utf-8` (the Prometheus exposition format MIME type), regardless of what SGLang sent. This protects against backends that mis-set the content type.

## What's deliberately not translated

- **`sglang:spec_accept_rate`, `sglang:gen_throughput`, etc.** Speculative-decoding metrics and other SGLang-only series have no vLLM analog. They pass through with their `sglang:` prefix so they remain available to operators who want them.
- **vLLM-only metrics with no SGLang source** (other than the three synthesized above). They simply don't appear in the output.

## Health vs. metrics

`/health` and `/metrics` are independent code paths. A failing `/health` does not affect `/metrics` and vice versa. `MetricsHandler` falls back to a 503 only when SGLang's `/metrics` itself is unreachable, not based on health state. This keeps Prometheus scrapes informative even during transient backend issues.

## TRT-LLM metrics translator

TRT-LLM exposes Prometheus exposition at `/prometheus/metrics`, not `/metrics`. The `Backend.metrics_path` ClassVar carries this difference: SGLang inherits the default `/metrics`; `TRTLLMBackend` overrides to `/prometheus/metrics`. The `MetricsHandler` reads `backend.metrics_path` when constructing the upstream scrape URL.

Translation is text-rename, parallel to SGLang's. The canonical `TRTLLM_TO_VLLM` map lives in `vllm_shim.backend.trtllm.metrics`.

| TRT-LLM name | vLLM name |
|---|---|
| `trtllm_request_success_total` | `vllm:request_success_total` |
| `trtllm_e2e_request_latency_seconds` | `vllm:e2e_request_latency_seconds` |
| `trtllm_time_to_first_token_seconds` | `vllm:time_to_first_token_seconds` |
| `trtllm_request_queue_time_seconds` | `vllm:request_queue_time_seconds` |
| `trtllm_kv_cache_hit_rate` | `vllm:gpu_prefix_cache_hit_rate` |

Histogram suffix handling (`_bucket`, `_sum`, `_count`) is reused from the SGLang trick: strip suffix, look up base, reattach.

Pass-through (no clean vLLM equivalent today): `trtllm_kv_cache_reused_blocks_total`, `trtllm_kv_cache_missed_blocks_total`, `trtllm_kv_cache_utilization`. They keep their `trtllm_` prefix.

Synthesized series at the end of the output, mirroring SGLang:

| vLLM series | Source | Logic |
|---|---|---|
| `vllm:kv_cache_usage_perc` | Derived | `trtllm_kv_cache_utilization` (a 0-to-1 ratio) multiplied by 100. Skipped if the source series is absent. |
| `vllm:healthy_pods_total` | Synthesized | Always `1` with `endpoint="default"`. |
| `vllm:num_requests_swapped` | Synthesized | Always `0`. TRT-LLM has no swap. |

JSON `/metrics` (the per-iteration inflight-batching stats) is unused. Surfacing those would require a second HTTP fetch per scrape and a `metrics_aux_path` ABC extension, deferred to a follow-up if operators report missing series.
