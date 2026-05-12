# AITER shape capture and config restore

AITER is AMD's optimized kernel library, used by SGLang on ROCm. It looks up tuned configs from CSV files under `/tmp/aiter_configs/` keyed by GEMM/MoE shape. When a shape has no tuned config it logs a one-line warning to stderr and falls back to a generic kernel, which is functionally correct but a measurable performance loss.

The shim closes this loop in two halves:

1. **Capture.** Tee the backend's stderr, parse `shape is M:... not found tuned config in <path>` lines, and persist them to `$VLLM_SHIM_HOME/aiter/shapes/...`. Captured shapes are the input the AITER tuner needs.
2. **Restore.** At startup, point AITER's `AITER_CONFIG_*` env vars at previously tuned configs under `$VLLM_SHIM_HOME/aiter/configs/<bucket>/`. AITER then reads directly from the persistent volume on first lookup (no symlinks, no writes into `/tmp`).

Both halves are silent no-ops when their prerequisites aren't met (no ROCm GPU, no resolvable shim home), so the same image runs on CUDA hosts and dev boxes without surprise behavior.

The tuner step that turns captured shapes into tuned configs is wired up as a separate operator-driven console script, `vllm-shim-tune`. It reads from `aiter/shapes/<bucket>/.../<target>.csv`, shells out to AITER's per-target tuner, and writes to `aiter/configs/<bucket>/<target>.csv`. See "Running the tuner" below.

## Why a dedicated `$VLLM_SHIM_HOME`

The captured shapes and tuned configs need to survive pod restarts. Operators typically point `VLLM_SHIM_HOME` at a persistent volume (e.g. `VLLM_SHIM_HOME=/data/vllm-shim`); the shim falls back to `~/.vllm-shim` when the env var is unset, so dev hosts and forgetful operators both land somewhere coherent without crashing the shim. The resolver is `vllm_shim.aiter.capture.resolve_shim_home`: env var first (with `~` expanded), else `Path.home() / ".vllm-shim"`, else `None`.

The shim deliberately keeps its own tree rather than nesting under `$HF_HOME`. The HF cache is HF's domain; mixing AITER artefacts under it confuses cache tooling and ties our layout to a directory we don't own. A dedicated env var also lets the operator put model weights and tuned configs on different volumes if they want.

AITER's tuned-config CSV locations are env-overridable. Each kernel target has its own `AITER_CONFIG_*` env var (`AITER_CONFIG_GEMM_BF16` for `bf16_tuned_gemm.csv`, `AITER_CONFIG_GEMM_A8W8` for `a8w8_tuned_gemm.csv`, etc.; see `repos/aiter/aiter/jit/core.py` for the full list). Restore sets those env vars in the backend's environment so AITER reads the configs directly off the PV. No symlinks, no writes into `/tmp`, multiple pods on the same PV share the files read-only for free.

## Path layout

```
$VLLM_SHIM_HOME/aiter/
├── shapes/                                # captured (input to the tuner)
│   └── <bucket>/                          # e.g. gfx942-304cu
│       └── <model>/                       # e.g. moonshotai--Kimi-K2.6
│           └── <parallelism>/             # e.g. tp8-ep8
│               ├── bf16_tuned_gemm.csv
│               ├── a8w8_blockscale_tuned_gemm.csv
│               └── ...
└── configs/                               # tuned (output from the tuner)
    └── <bucket>/                          # e.g. gfx942-304cu
        ├── bf16_tuned_gemm.csv
        ├── a8w8_blockscale_tuned_gemm.csv
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
        aiter/shapes/<bucket>/<model>/<parallelism>/<target>.csv
```

A few invariants to know about:

- **`StreamTee` is a daemon thread.** A stalled backend producer cannot keep the supervisor alive past its shutdown deadline; the tee disappears when the process exits.
- **Callback errors are swallowed.** Tee survival outranks any single shape-capture write. A broken store (disk full, read-only volume) must not silence the backend's stderr.
- **`ShapeStore` dedups twice.** An in-memory set per target prevents repeat writes during this process, and on first write to a target the existing CSV is loaded into that set so restarts converge on a deduped file. Concurrent writers on a shared persistent volume could in theory produce duplicate rows; we accept that and the operator can dedupe offline if it ever matters.
- **One CSV per AITER target.** Targets come from the basename of the AITER config path the log line names. `bf16_tuned_gemm.csv`, `a8w8_blockscale_tuned_gemm.csv`, etc. The directory part of that path is whatever AITER was configured to read (its install dir by default, our `$VLLM_SHIM_HOME/aiter/configs/<bucket>/` when restore is active) and unrelated to where we write the capture file.

## Restore: how AITER gets pointed at our configs

```
$VLLM_SHIM_HOME/aiter/configs/<bucket>/bf16_tuned_gemm.csv
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

**Precedence**: operator-set `AITER_CONFIG_*` env vars (in the pod spec or container `ENV`) win over restore. Same principle as `translate_env_with_map`: if the operator wrote it down, they meant it. The launch-info dump only lists the overrides that actually took effect, so a missing entry in `aiter_restore.overrides` for a target whose file exists under `$VLLM_SHIM_HOME/aiter/configs/<bucket>` is the signal that the operator's env beat us to it.

## HIP online tuning anchor

When the operator sets `HIP_ONLINE_TUNING=1` (or `=true`), AITER's gradlib enters its hipBLASLt online-tuning path. On the first call for each new shape it runs a short measurement, picks the fastest hipBLASLt algorithm, and appends a row to `hip_online_tuning_res.csv`. Subsequent calls for the same shape read back the cached algorithm. The full process takes several minutes the first time; subsequent restarts are fast if the file is reused.

The catch: the CSV path is hardcoded *relative to the current working directory* in gradlib's C++ source (`repos/aiter/gradlib/csrc/hipbsolgemm.cu`, `get_algoIdx_hip_tuning_csv` / `append_hip_tuning_csv`). No env var redirects it. Default container layout puts CWD on an ephemeral layer, so without intervention every pod re-pays the full tune cost.

The shim's anchor: when `HIP_ONLINE_TUNING` is on, keep the canonical CSV at `$VLLM_SHIM_HOME/hip_online_tuning_res.csv` and symlink `<cwd>/hip_online_tuning_res.csv` onto it. AITER's relative-path open transparently follows the symlink and lands on the PV. See `vllm_shim.aiter.hip_online_tuning`.

The anchor is the right knob (not a copy-on-shutdown hook) because gradlib appends per-shape. A pod that exits abnormally between tune and shutdown would lose any rows accumulated since the last save; the symlink keeps every successful append on the PV in real time.

Edge cases the planner handles:

- **Symlink already correct**: idempotent no-op on restart; the existing file's accumulated rows survive.
- **Wrong-pointing symlink**: replaced (image upgrade moved the storage path, operator hand-linked somewhere else).
- **Regular file already at the CWD path**: refuse to clobber it. The launch info reports `non-symlink file at target`; the operator decides whether to move it onto the PV manually.
- **No shim home**: no anchor; launch info reports `no shim home` so the operator sees the data is going to be ephemeral.

The launch-info dump surfaces the state under `hip_online_tuning` (`enabled`, `storage`, `target`, `reason`). The stderr summary line prints only when the operator opted in, so the common (env-not-set) case stays silent. Format: `hip online tuning: <cwd-path> -> <pv-path>` for the anchored case, `hip online tuning: not anchored (<reason>)` for the failure cases.

## Prerequisites and decision flow

Both halves share the same two prerequisites:

| Check | Where | What it means |
|---|---|---|
| ROCm GPU present | `vllm_shim.cli.rocm_probe.probe` shells `rocminfo` | Without ROCm there's no AITER, so capture and restore would have no effect. |
| Shim home resolvable | `vllm_shim.aiter.capture.resolve_shim_home` | Without a writable root, captured data would be ephemeral and tuned configs have nowhere to live. |

Either prerequisite missing yields a disabled plan with a stable `reason` string (`"no ROCm GPU detected"` or `"could not resolve VLLM_SHIM_HOME"`) for the launch-info dump. The plans are pure decisions; the entrypoint reads the env once and feeds it to both halves so they stay in sync.

When the prerequisites pass, the entrypoint:

1. Calls `plan_restore` and `restore_configs` *synchronously* before spawning the backend, so the `AITER_CONFIG_*` env vars are in the backend's environment by the time AITER imports.
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
  aiter capture: enabled -> /data/vllm-shim/aiter/shapes/gfx942-304cu/moonshotai--Kimi-K2.6/tp8
  aiter restore: 2 configs from /data/vllm-shim/aiter/configs/gfx942-304cu (AITER_CONFIG_GEMM_BF16, AITER_CONFIG_GEMM_A8W8)
```

The restore line has three forms:

| Line | Meaning |
|---|---|
| `aiter restore: disabled (<reason>)` | Prerequisites failed; no work done. |
| `aiter restore: N configs from <source> (<env vars>)` | N `AITER_CONFIG_*` env vars injected into the backend env, pointing at our tuned CSVs. The full env-var-to-path mapping is in the JSON dump. |
| `aiter restore: enabled, nothing to restore from <source>` | Source dir is missing or empty (first-ever run, or pre-tuner). |

The JSON dump at `/tmp/vllm-shim-info.json` has the same data in `aiter_capture` and `aiter_restore` keys.

## Capture trade-offs and relationship to AITER's own tuning paths

AITER ships with two adjacent mechanisms it's worth being explicit about:

**`AITER_TUNE_GEMM=1`**: AITER's native shape capture. When set, `tuned_gemm.save_shapes` writes *every* GEMM call (hits and misses) to `<install>/aiter/configs/bf16_untuned_gemm.csv` using the canonical CSV schema. We don't use this for three reasons:

- It captures everything, not just the actionable misses. The tuner only needs the gap between observed and tuned, so the extra rows are waste.
- The write location is hardcoded inside the AITER install dir (not env-overridable), so we'd have no way to land the file under `$VLLM_SHIM_HOME` without monkey-patching AITER.
- It can't partition by model or parallelism; one process writes one global file. Our stderr-tee partitions naturally via the path layout.

Our stderr-tee is more constrained (only matches the BF16 GEMM log format today; other targets like a4w4 emit different formats and aren't yet parsed) but the trade is acceptable: GEMM is the dominant target by call count, and adding new log formats is local work in `log_parser.py`.

**`AITER_ONLINE_TUNE=1`**: lazy tuning at first miss, MoE-only. When a fused MoE forward pass misses, AITER acquires a multiprocess file lock, invokes `gemm_moe_tune.py` synchronously, and unblocks the request after tuning completes. Output lands in the same `AITER_CONFIG_FMOE` file our restore reads. We deliberately don't enable it by default because:

- It only covers MoE. GEMM-side misses still need our capture + offline tune flow, so it's not a substitute.
- The triggering request eats the tuning cost: seconds to minutes per shape, on the live serving GPU. Unacceptable p99 for an inference API.

An operator who explicitly wants laziness for MoE can set `AITER_ONLINE_TUNE=1` in their pod spec; restore's env-var injection uses a different set of variables and won't fight it. Our capture still records the MoE miss lines regardless.

**Background ("Flavor 2") tuning**: a tempting design where a sidecar tunes shapes during idle and the running pod picks up the new configs without a restart. We've chosen not to pursue this yet because of a concrete blocker: AITER caches the tuned-config dict in memory via `@functools.lru_cache(maxsize=1)` on `get_GEMM_*_config_`. The running pod won't re-read a freshly tuned CSV until the cache invalidates, which doesn't happen without a restart or an explicit signal we don't have. Background tuning could still write to the PV for *future* pods to pick up via restore, but that's effectively just an asynchronous offline tuner, not online tuning. If we ever want true zero-restart pickup, we need either an upstream AITER patch with a cache-invalidation hook or an external control-plane endpoint on the backend.

**Log level pin**: when capture is enabled, the entrypoint sets `AITER_LOG_LEVEL=INFO` in `backend_env` via `setdefault`. AITER emits the shape-not-found line at `logger.info`; an operator who raises log level globally would silence capture otherwise. The `setdefault` semantics preserve any explicit override (e.g. someone debugging who has good reason to quiet AITER).

## Running the tuner

`vllm-shim-tune` is a console script that turns captured shapes into tuned configs. Invoke it from a shell inside a pod that has ROCm + AITER available (the SGLang-ROCm shim image does):

```
vllm-shim-tune              # tune every target for the local GPU bucket
vllm-shim-tune --list       # show captured/tuned state per target, no work
vllm-shim-tune --target a8w8_tuned_gemm
vllm-shim-tune --dry-run    # print the AITER commands without running them
vllm-shim-tune --retune     # force re-tune of every shape (--all to AITER)
vllm-shim-tune --hot 100    # tune the 100 hottest shapes per target only
```

The default behaviour is incremental: AITER's tuners read the existing `configs/<bucket>/<target>.csv`, diff against the merged captured shapes, and only tune the new rows. `--retune` maps to AITER's own `--all` flag and re-tunes from scratch.

Key flags:

| Flag | Default | Purpose |
|---|---|---|
| `--bucket` | inferred via `rocminfo` | Hardware bucket to tune for (e.g. `gfx942-304cu`). Required when there's no local ROCm GPU (offline tuning on captured CSVs). |
| `--target` | all known targets | Tune only one tuned-config target. |
| `--shim-home` | `VLLM_SHIM_HOME`, else `~/.vllm-shim` | Override the persistent root. |
| `--aiter-root` | `/sgl-workspace/aiter` | Path to the AITER source tree (where the tuner scripts live). |
| `--python` | `/opt/venv/bin/python` | Python interpreter that has AITER installed. The shim itself runs under `/opt/shim`'s Python 3.12; AITER lives in `/opt/venv`'s 3.10. |
| `--hot N` | unset (tune all) | Tune only the N hottest shapes per target. See "Hot-shape filtering" below. |

The tuner consolidates inputs per (bucket, target): every captured `shapes/<bucket>/<model>/<parallelism>/<target>.csv` is unioned into one deduped untuned file before AITER is invoked. The intermediate file lands under `$VLLM_SHIM_HOME/aiter/untuned/<bucket>/<target>.csv` so the operator can inspect what was fed into AITER if a tune misbehaves.

### Hot-shape filtering

`--hot N` truncates the merged untuned CSV to the N shapes most likely to dominate runtime call count, before handing it to AITER's tuner. The summary line in the tuner's stderr output reports how many shapes were kept vs. dropped:

```
  bf16_tuned_gemm: 100 of 247 shapes (hot) -> /data/.../bf16_tuned_gemm.csv [ok]
```

**Heuristic, not measured frequency.** "Hot" is currently defined as smallest `(M, N, K)` first. In autoregressive LLM serving the decode phase runs at `M = active_batch` (small, typically 1-256) and invokes every layer's GEMMs once per token, while prefill shapes are large M but run once per request. Call count therefore skews orders of magnitude toward small-M shapes, so the M-ascending sort puts decode at the front of the queue and truncating preserves the long tail that actually moves serving latency.

When `--hot` is a poor fit:

- Long-context single-shot prefill workloads (large M, few decode steps) shouldn't use `--hot`, or should set N wide enough to cover the prefill shapes.
- Static-batch workloads where all captured shapes share the same M see arbitrary truncation; either skip `--hot` or set N to the total shape count.

**Composes with `--retune`.** Default incremental mode (no `--retune`) makes AITER diff the input against the existing tuned CSV and only tune new rows, so growing `--hot` across runs (e.g. 100 today, 500 next week, full set later) converges on full coverage without redoing earlier work.

**Future work.** The heuristic is a stand-in for measured frequency. Adding a count column to the capture path is a follow-up; the `--hot` knob will keep its current name and semantics so operator workflows don't have to change.

### Tuning at pod startup

Single-GPU deployments where running a separate tuning Job isn't possible can fold tuning into pod restarts via env vars:

```yaml
extraEnvs:
  - name: VLLM_SHIM_TUNE_AT_STARTUP_SECONDS
    value: "900"   # 15 min wall-clock budget; 0 / unset = off
  - name: VLLM_SHIM_TUNE_AT_STARTUP_HOT
    value: "100"   # tune only the 100 hottest shapes per target;
                   # 0 / unset = tune everything that was captured
```

When `VLLM_SHIM_TUNE_AT_STARTUP_SECONDS` is set to a positive integer, the entrypoint runs `vllm-shim-tune --shim-home <...> --bucket <...>` between the AITER restore step and the backend spawn, with that many seconds as a hard wall-clock budget. The GPU is uncontested during this window because the backend hasn't started yet, so tuner benchmark measurements are clean (concurrent serving traffic would poison them).

`VLLM_SHIM_TUNE_AT_STARTUP_HOT` appends `--hot N` to that command (see "Hot-shape filtering" above for what the heuristic does). Use it together with the budget when the captured-shape set is too large to tune in full within `progressDeadlineSeconds`: hot-filtering shrinks the input so the budget is far less likely to truncate mid-shape, and the remaining shapes accumulate on subsequent restarts (incremental mode skips already-tuned rows).

Failure modes are all swallowed: timeout, missing console script, AITER crash. The supervisor logs a single status line and proceeds to launch the backend. Partial tunes are safe because AITER writes tuned rows incrementally; a killed-mid-shape subprocess leaves a valid (partial) CSV that the next restart picks up.

Choosing a budget and hot count:

- Most incremental tunes (a handful of new shapes captured since the last restart) finish in seconds. The budget is a ceiling, not a floor.
- The first-ever tune of a fresh deployment with hundreds of captured shapes can take minutes to an hour. Set the budget below your k8s `progressDeadlineSeconds` to avoid CrashLoopBackOff while AITER works through the queue. Partial tunes accumulate across restarts.
- For latency-sensitive LLM serving workloads, `VLLM_SHIM_TUNE_AT_STARTUP_HOT=100` (or similar, depending on how much variety lives in your decode-phase shapes) typically captures most of the perf win at a fraction of the tune cost. Operators tracking exact coverage can step it up across deploys (100 → 500 → unset) without re-tuning earlier rows.
- Operators who don't want this behaviour leave both env vars unset; the entrypoint then behaves exactly like before.

### Target -> tuner script mapping

The mapping is hardcoded in `vllm_shim.aiter.tune._SPECS`; AITER doesn't expose it as metadata. Two CLI conventions exist among AITER's own tuners:

| Target | Tuner script | Input flag | Output flag | Extra args |
|---|---|---|---|---|
| `bf16_tuned_gemm` | `gradlib/gradlib/gemm_tuner.py` | `--input_file` | `--tuned_file` | |
| `a4w4_blockscale_tuned_gemm` | `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py` | `--untune_file` | `--tune_file` | `--libtype all` |
| `a8w8_tuned_gemm` | `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py` | `--untune_file` | `--tune_file` | `--libtype all` |
| `a8w8_bpreshuffle_tuned_gemm` | `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py` | `--untune_file` | `--tune_file` | `--libtype all` |
| `a8w8_blockscale_tuned_gemm` | `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py` | `--untune_file` | `--tune_file` | `--libtype all` |
| `a8w8_blockscale_bpreshuffle_tuned_gemm` | (same script as blockscale) | `--untune_file` | `--tune_file` | `--libtype all --preshuffle` |
| `bf16_tuned_batched_gemm` | `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py` | `--untune_file` | `--tune_file` | `--libtype all` |
| `a8w8_tuned_batched_gemm` | `csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py` | `--untune_file` | `--tune_file` | `--libtype all` |

`tuned_fmoe` is deliberately not in the table. Its tuner (`csrc/ck_moe/gemm_moe_tune.py`) uses a third CLI (`-i/-o/-o2/--last`) and its log lines don't match the capture-side parser today, so we'd capture nothing to feed it. Adding it is future work in `log_parser.py` plus a new spec entry.

When AITER adds a new tuned-config target, three places must move in lockstep: the capture-side regex/dataclass in `log_parser.py`, the restore-side env-var mapping in `restore._TARGET_ENV`, and the tuner spec in `tune._SPECS`. The test `test_known_targets_covers_all_restorable_targets` will fail loudly if `_SPECS` and `_TARGET_ENV` drift apart.

## Module layout

| Module | Role |
|---|---|
| `vllm_shim.aiter.log_parser` | `AiterShape` dataclass + `parse_line(str) -> AiterShape | None`. Pure. |
| `vllm_shim.aiter.shape_store` | `ShapeStore`: append a deduped CSV row per AITER shape. |
| `vllm_shim.aiter.stream_tee` | `StreamTee`: daemon thread that copies bytes to a sink and lines to a callback. |
| `vllm_shim.aiter.path` | `sanitize_model`, `shape_capture_root`. Pure. |
| `vllm_shim.aiter.capture` | `CapturePlan`, `plan_capture`, `resolve_shim_home`, `build_callback`. |
| `vllm_shim.aiter.restore` | `RestorePlan`, `plan_restore`, `restore_configs`. |
| `vllm_shim.aiter.tune` | `TunerSpec`, `_SPECS`, `tune_target`, `main` (the `vllm-shim-tune` console script). |
| `vllm_shim.cli.rocm_probe` | `parse_rocminfo`, `probe`, `bucket`. Shells out to `rocminfo`. |
| `vllm_shim.values.parallelism` | `Parallelism` value + `path_segment()`. |
| `vllm_shim.backend.base.parallelism` | `ParallelismExtractor` ABC. |
| `vllm_shim.backend.{sglang,trtllm}.parallelism` | Concrete extractors. |

The split into many small modules is deliberate: each piece is pure (or has a single well-defined side effect) and testable in isolation. The orchestration in `capture.py` and `restore.py` is the only place that knows about the cross-cutting decision.

## What this is not

- **Not online.** The `vllm-shim-tune` step runs offline (or on-demand from a pod shell) and writes results into `$VLLM_SHIM_HOME/aiter/configs/<bucket>/`. It is not invoked from the serve-time entrypoint; the running backend only picks up tuned configs at the next launch via the restore step.
- **Not CUDA-relevant.** Capture and restore are no-ops on CUDA hosts (the rocm probe returns None). vLLM-on-CUDA does not use AITER. If a future vLLM-on-ROCm backend lands in the shim, it will reuse this same machinery without changes.
- **Not coupled to SGLang.** The capture logic reads stderr line patterns AITER itself emits; any backend that uses AITER and produces the same warning format will work. The pattern lives in `log_parser.py` and is intentionally narrow.
