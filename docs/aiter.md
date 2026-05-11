# AITER shape capture and config restore

AITER is AMD's optimized kernel library, used by SGLang on ROCm. It looks up tuned configs from CSV files under `/tmp/aiter_configs/` keyed by GEMM/MoE shape. When a shape has no tuned config it logs a one-line warning to stderr and falls back to a generic kernel, which is functionally correct but a measurable performance loss.

The shim closes this loop in two halves:

1. **Capture.** Tee the backend's stderr, parse `shape is M:... not found tuned config in <path>` lines, and persist them to `$HF_HOME/vllm-shim/aiter-shapes/...`. Captured shapes are the input the AITER tuner needs.
2. **Restore.** At startup, point AITER's `AITER_CONFIG_*` env vars at previously tuned configs under `$HF_HOME/vllm-shim/aiter-configs/<bucket>/`. AITER then reads directly from the persistent volume on first lookup (no symlinks, no writes into `/tmp`).

Both halves are silent no-ops when their prerequisites aren't met (no ROCm GPU, no resolvable HF cache), so the same image runs on CUDA hosts and dev boxes without surprise behavior.

The tuner itself (turning captured shapes into tuned configs) is not part of the shim today; that work happens offline, and the operator drops the result under the `aiter-configs/` tree before the next launch.

## Why under `$HF_HOME`

The captured shapes and tuned configs need to survive pod restarts. Production Stack already mounts a persistent volume at `$HF_HOME` for model snapshots, so reusing that mount means operators don't have to wire a second PV. The shim resolves `$HF_HOME` via `huggingface_hub.constants.HF_HOME` (which already honors the env var, `$XDG_CACHE_HOME`, and `~/.cache/huggingface` defaults) so the captured/tuned trees automatically follow wherever HF itself is reading and writing.

AITER's tuned-config CSV locations are env-overridable. Each kernel target has its own `AITER_CONFIG_*` env var (`AITER_CONFIG_GEMM_BF16` for `bf16_tuned_gemm.csv`, `AITER_CONFIG_GEMM_A8W8` for `a8w8_tuned_gemm.csv`, etc.; see `repos/aiter/aiter/jit/core.py` for the full list). Restore sets those env vars in the backend's environment so AITER reads the configs directly off the PV. No symlinks, no writes into `/tmp`, multiple pods on the same PV share the files read-only for free.

## Path layout

```
$HF_HOME/vllm-shim/
├── aiter-shapes/                          # captured (input to the tuner)
│   └── <bucket>/                          # e.g. gfx942-304cu
│       └── <model>/                       # e.g. moonshotai--Kimi-K2.6
│           └── <parallelism>/             # e.g. tp8-ep8
│               ├── bf16_gemm.csv
│               ├── fp8_blockscale_gemm.csv
│               └── ...
└── aiter-configs/                         # tuned (output from the tuner)
    └── <bucket>/                          # e.g. gfx942-304cu
        ├── bf16_gemm.csv
        ├── fp8_blockscale_gemm.csv
        └── ...
```

The partitioning differs between the two halves:

- **Captured shapes** are partitioned by `bucket × model × parallelism`. The same shape miss on a different model or different tensor parallel degree might want different kernel parameters, so the tuner sees them as separate inputs.
- **Tuned configs** are partitioned by `bucket` only. AITER's CSV is keyed by shape dimensions, not by model name, so a config tuned for one model is reusable by any other model that happens to hit the same shape.

`<bucket>` is `<gfx_target>-<compute_units>cu` (e.g. `gfx942-304cu` for MI300X). Same `gfx942` chip can ship with different CU counts and a kernel tuned for one CU count will mis-perform on another; the CU count must be in the key.

`<model>` is the sanitized model identifier: HF repo IDs become `org--name` (matching the HF cache directory convention sans the `models--` prefix), local paths become their basename. See `vllm_shim.aiter.path.sanitize_model`.

`<parallelism>` is `tp<n>[-ep<n>][-pp<n>]` (EP and PP are omitted when 1). Extracted from the post-translation argv via the backend's `ParallelismExtractor`; see "Parallelism extraction" below.

## Capture: how stderr becomes CSV rows

```
SGLang stderr
   │
   ▼
StreamTee (daemon thread)              vllm_shim.aiter.stream_tee
   │   reads line-by-line
   ├──► sink (sys.stderr.buffer)       # operator still sees raw output
   └──► callback(line)
           │
           ▼
        parse_line(line)               vllm_shim.aiter.log_parser
           │
           ▼  AiterShape | None
        store.add(shape)               vllm_shim.aiter.shape_store
           │
           ▼
        aiter-shapes/<bucket>/<model>/<parallelism>/<target>.csv
```

A few invariants to know about:

- **`StreamTee` is a daemon thread.** A stalled backend producer cannot keep the supervisor alive past its shutdown deadline; the tee disappears when the process exits.
- **Callback errors are swallowed.** Tee survival outranks any single shape-capture write. A broken store (disk full, read-only volume) must not silence the backend's stderr.
- **`ShapeStore` dedups twice.** An in-memory set per target prevents repeat writes during this process, and on first write to a target the existing CSV is loaded into that set so restarts converge on a deduped file. Concurrent writers on a shared persistent volume could in theory produce duplicate rows; we accept that and the operator can dedupe offline if it ever matters.
- **One CSV per AITER target.** Targets come from the basename of the `/tmp/aiter_configs/...csv` path the log line names. `bf16_gemm.csv`, `fp8_blockscale_gemm.csv`, etc. The directory part of that path is hardcoded by AITER and unrelated to where we write.

## Restore: how AITER gets pointed at our configs

```
$HF_HOME/vllm-shim/aiter-configs/<bucket>/bf16_tuned_gemm.csv
                   │
                   ▼  pre-launch, in vllm_shim.aiter.restore.restore_configs
       map basename -> AITER_CONFIG_GEMM_BF16
                   │
                   ▼
     backend_env["AITER_CONFIG_GEMM_BF16"] = "<that path>"
                   │
                   ▼
       AITER's jit/core.py reads the env at import time
```

The mapping from tuned-config basename to env var lives in `_TARGET_ENV` in `vllm_shim.aiter.restore`. It mirrors AITER's own `AITER_CONFIG_*` defaults defined in `repos/aiter/aiter/jit/core.py`. The known mapping today:

| Basename | Env var | AITER property |
|---|---|---|
| `bf16_tuned_gemm.csv` | `AITER_CONFIG_GEMM_BF16` | `AITER_CONFIG_GEMM_BF16_FILE` |
| `a4w4_blockscale_tuned_gemm.csv` | `AITER_CONFIG_GEMM_A4W4` | `AITER_CONFIG_GEMM_A4W4_FILE` |
| `a8w8_tuned_gemm.csv` | `AITER_CONFIG_GEMM_A8W8` | `AITER_CONFIG_GEMM_A8W8_FILE` |
| `a8w8_bpreshuffle_tuned_gemm.csv` | `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE` | `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE` |
| `a8w8_blockscale_tuned_gemm.csv` | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE` |
| `a8w8_blockscale_bpreshuffle_tuned_gemm.csv` | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE` | `..._BLOCKSCALE_BPRESHUFFLE_FILE` |
| `bf16_tuned_batched_gemm.csv` | `AITER_CONFIG_BF16_BATCHED_GEMM` | `AITER_CONFIG_BF16_BATCHED_GEMM_FILE` |
| `a8w8_tuned_batched_gemm.csv` | `AITER_CONFIG_A8W8_BATCHED_GEMM` | `AITER_CONFIG_A8W8_BATCHED_GEMM_FILE` |
| `tuned_fmoe.csv` | `AITER_CONFIG_FMOE` | `AITER_CONFIG_FMOE_FILE` |

`restore_configs` is read-only: it lists the bucket directory, looks up known basenames in the mapping, and returns the `{env_var: path}` dict. The entrypoint filters out any var the operator has already set, then merges the rest into `backend_env` before spawning. Unknown basenames are skipped (likely a future AITER target the shim doesn't recognise yet; the operator can set the env var manually if needed).

**Precedence**: operator-set `AITER_CONFIG_*` env vars (in the pod spec or container `ENV`) win over restore. Same principle as `translate_env_with_map`: if the operator wrote it down, they meant it. The launch-info dump only lists the overrides that actually took effect, so a missing entry in `aiter_restore.overrides` for a target whose file exists under `$HF_HOME` is the signal that the operator's env beat us to it.

## Prerequisites and decision flow

Both halves share the same two prerequisites:

| Check | Where | What it means |
|---|---|---|
| ROCm GPU present | `vllm_shim.cli.rocm_probe.probe` shells `rocminfo` | Without ROCm there's no AITER, so capture and restore would have no effect. |
| HF cache resolvable | `vllm_shim.aiter.capture.resolve_hf_home` | Without a writable cache root, captured data would be ephemeral and pointless. |

Either prerequisite missing yields a disabled plan with a stable `reason` string (`"no ROCm GPU detected"` or `"could not resolve HF cache directory"`) for the launch-info dump. The plans are pure decisions; the entrypoint reads the env once and feeds it to both halves so they stay in sync.

When the prerequisites pass, the entrypoint:

1. Calls `plan_restore` and `restore_configs` *synchronously* before spawning the backend, so the symlinks are in place by the time AITER does its first lookup.
2. Calls `plan_capture` and spawns the backend with `stderr=subprocess.PIPE`, then starts the `StreamTee` daemon. With capture disabled the backend gets the default inherited stderr and no tee.

The PIPE/inherit distinction matters: only spawn with PIPE when we are actually going to read the pipe, otherwise the kernel pipe buffer fills up and the backend blocks on writes.

## Parallelism extraction

Captured shapes are partitioned by tp/ep/pp degrees, but we can't trust that operators always use vLLM-style flag names. Some pods are configured with SGLang-native flags directly via the shim's passthrough chain. The only argv we can rely on is the *post-translation* argv (after `ArgTranslator` has rewritten it into the backend's vocabulary).

That's why parallelism extraction is a backend-owned component, parallel to `ArgTranslator`/`EnvTranslator`/etc. Each backend knows its own native flag spellings:

| Backend | TP flags | PP flags | EP flags |
|---|---|---|---|
| SGLang  | `--tp`, `--tp-size`, `--tensor-parallel-size` | `--pipeline-parallel-size`, `--pp-size` | `--ep`, `--ep-size`, `--expert-parallel-size` |
| TRT-LLM | `--tp_size` | `--pp_size` | `--ep_size`, `--moe_expert_parallel_size` |

The shared `last_int_for_flags` helper in `vllm_shim.backend._shared` does the actual scanning; the per-backend extractors are basically just the flag sets. Defaults to 1 (the "off" semantics for every backend) when no flag is found.

Dash/underscore parity is handled at the `ArgTranslator` layer: an operator writing `--tensor_parallel_size 8` for SGLang or `--tp-size 8` for TRT-LLM gets the same end result as the native spelling, so the extractor sees a canonical argv either way. See `docs/argument-translation.md`.

## Operator surface

The launch-info dump (`vllm-shim-info` and the stderr summary) surfaces both halves so an operator can confirm what happened without grep-spelunking SGLang's own logs:

```
vllm-shim 0.0.1 -> sglang listening on 0.0.0.0:8000
  model: moonshotai/Kimi-K2.6 -> /data/hub/models--moonshotai--Kimi-K2.6/snapshots/abc
  backend argv: sglang serve --model-path ... --tp 8
  aiter capture: enabled -> /data/hf/vllm-shim/aiter-shapes/gfx942-304cu/moonshotai--Kimi-K2.6/tp8
  aiter restore: 2 configs from /data/hf/vllm-shim/aiter-configs/gfx942-304cu (bf16_gemm.csv, fp8_blockscale_gemm.csv)
```

The restore line has three forms:

| Line | Meaning |
|---|---|
| `aiter restore: disabled (<reason>)` | Prerequisites failed; no work done. |
| `aiter restore: N configs from <source> (<names>)` | Newly symlinked N files into `/tmp/aiter_configs/`. |
| `aiter restore: enabled, nothing to restore from <source>` | Source dir is missing or empty (first-ever run, or pre-tuner). |

The JSON dump at `/tmp/vllm-shim-info.json` has the same data in `aiter_capture` and `aiter_restore` keys.

## Module layout

| Module | Role |
|---|---|
| `vllm_shim.aiter.log_parser` | `AiterShape` dataclass + `parse_line(str) -> AiterShape | None`. Pure. |
| `vllm_shim.aiter.shape_store` | `ShapeStore`: append a deduped CSV row per AITER shape. |
| `vllm_shim.aiter.stream_tee` | `StreamTee`: daemon thread that copies bytes to a sink and lines to a callback. |
| `vllm_shim.aiter.path` | `sanitize_model`, `shape_capture_root`. Pure. |
| `vllm_shim.aiter.capture` | `CapturePlan`, `plan_capture`, `resolve_hf_home`, `build_callback`. |
| `vllm_shim.aiter.restore` | `RestorePlan`, `plan_restore`, `restore_configs`. |
| `vllm_shim.cli.rocm_probe` | `parse_rocminfo`, `probe`, `bucket`. Shells out to `rocminfo`. |
| `vllm_shim.values.parallelism` | `Parallelism` value + `path_segment()`. |
| `vllm_shim.backend.base.parallelism` | `ParallelismExtractor` ABC. |
| `vllm_shim.backend.{sglang,trtllm}.parallelism` | Concrete extractors. |

The split into many small modules is deliberate: each piece is pure (or has a single well-defined side effect) and testable in isolation. The orchestration in `capture.py` and `restore.py` is the only place that knows about the cross-cutting decision.

## What this is not

- **Not the tuner.** Capturing shapes does not produce tuned configs. That's a separate offline step running AITER's own tuner against the captured CSVs. A future `vllm-shim-tune` subcommand may automate it; today it is an operator-driven step.
- **Not CUDA-relevant.** Capture and restore are no-ops on CUDA hosts (the rocm probe returns None). vLLM-on-CUDA does not use AITER. If a future vLLM-on-ROCm backend lands in the shim, it will reuse this same machinery without changes.
- **Not coupled to SGLang.** The capture logic reads stderr line patterns AITER itself emits; any backend that uses AITER and produces the same warning format will work. The pattern lives in `log_parser.py` and is intentionally narrow.
