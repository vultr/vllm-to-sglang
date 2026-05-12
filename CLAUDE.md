# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`vllm-shim` is a drop-in replacement for the vLLM serving container. It accepts vLLM's `vllm serve <model> [flags]` CLI and delegates inference to SGLang or TensorRT-LLM, so existing vLLM Production Stack manifests, dashboards, and alerts continue to work unchanged. The shim intercepts the *server launch*, not library imports; there is no `vllm` Python API surface here.

## Common commands

The repo is a `uv` workspace with two member packages. All tooling is configured at the workspace root.

```bash
uv sync                          # one-time setup; resolves workspace, creates .venv/
uv run pytest                    # run all tests (unit + integration, both packages)
uv run pytest packages/vllm-shim/tests/unit/backend/sglang/test_args.py    # single file
uv run pytest -k test_filters_in_documented_order                          # single test
uv run mypy                      # strict type checking (workspace-wide, includes tests)
uv run ruff check .              # lint
uv run ruff format .             # format
```

Building images (run from repo root, the build context must be `.`):

```bash
docker build -f docker/sglang/Dockerfile.cuda  -t vllm-shim:sglang-cuda .
docker build -f docker/sglang/Dockerfile.rocm  -t vllm-shim:sglang-rocm .
docker build -f docker/trtllm/Dockerfile.cuda  -t vllm-shim:trtllm-cuda .
```

There is no end-to-end local run; the shim needs a real backend image to serve. To exercise just the middleware against a fake backend:

```bash
VLLM_SHIM_BACKEND_HOST=localhost VLLM_SHIM_BACKEND_PORT=9999 VLLM_SHIM_MIDDLEWARE_PORT=8080 uv run vllm-shim-middleware
```

## Architecture you must hold in your head

Three processes run inside one container, all children of `vllm_shim.cli.entrypoint.main`:

```
client :N -> haproxy -> :N+2 middleware (FastAPI) -> :N+1 SGLang/TRT-LLM
```

- `N` is the public `--port` (default 8000). The other two ports are derived by `+1` and `+2` and are loopback-only. The offsets live in `PortAllocation.from_listen` (`vllm_shim.values.port_allocation`).
- haproxy is the only public listener. It uses `nbsrv(sglang) gt 0` to serve a static 503 errorfile when the backend is down, so k8s liveness probes see a clean 200/503 boundary even during weight loading.
- The middleware (`vllm_shim.middleware`) handles `/health`, `/metrics`, and a catch-all proxy that runs the backend's `RequestFilter` chain over request bodies before forwarding.
- The supervisor (`vllm_shim.cli.supervisor.Supervisor`) sends SIGTERM to all children simultaneously, then `wait()`s on them in declared order against one shared 25 s deadline. The "send first, wait sequentially" structure is **load-bearing** (see the comment in `_terminate_all`); do not restructure it into "terminate one, wait one." See `docs/supervisor.md`.

### Backend abstraction

A `Backend` (`vllm_shim.backend.base.backend`) is a contract object holding six components plus class constants:

| Slot | ABC | Role |
|---|---|---|
| `args`        | `ArgTranslator`        | Pure function: `translate(passthrough) -> (backend_argv, dropped)`. |
| `env`         | `EnvTranslator`        | Pure function: `translate(os.environ) -> dict[str, str]`. Adds backend-side renames for selected `VLLM_*` env vars. |
| `launcher`    | `Launcher`             | Builds the subprocess argv. |
| `metrics`     | `MetricsTranslator`    | Rewrites Prometheus exposition into vLLM-named series. |
| `filters`     | `tuple[RequestFilter, ...]` | Body-rewriting filters that run in declared order. |
| `parallelism` | `ParallelismExtractor` | Reads tp/ep/pp out of the post-translation argv. Used by the AITER shape-capture path; see `docs/aiter.md`. |

Concrete backends live at `vllm_shim.backend.sglang` and `vllm_shim.backend.trtllm`. The registry (`vllm_shim.backend.registry`) is a single `dict` selected by `VLLM_SHIM_BACKEND` env (default `sglang`); to add a backend, edit that dict, do not introduce plugin discovery. The supervisor and middleware are separate processes, so each constructs its own `Backend` instance from the same env. See `docs/backends.md`.

The backend layer must not import from `vllm_shim.middleware` or `vllm_shim.cli`; it may depend on `vllm_shim.values`. This rule keeps the architecture diagram honest.

### Two-stage CLI translation

1. `ArgParser` (`vllm_shim.cli.parser`) extracts only what the supervisor needs (`model`, `host`, `port`, `revision`) and routes the rest into `passthrough` verbatim. Backend-agnostic.
2. Between stages, the entrypoint calls `resolve_model` (`vllm_shim.cli.model_resolver`) to turn HF repo IDs into local snapshot directories via `huggingface_hub.snapshot_download` (no-op on cache hit; honours `HF_HOME` and `HF_HUB_OFFLINE`). When this rewrites the path, the entrypoint also injects `--served-model-name <original>` so `/v1/models` keeps advertising what clients are calling with. Local-path inputs pass through unchanged and trigger no injection.
3. The selected backend's `ArgTranslator` rewrites `passthrough` via an `ARG_MAP` dict whose values are `(target_name | None, has_value: bool)`. Anything not in the map passes through unchanged. `=`-form and underscore variants are explicit map keys (no normalization layer). See `docs/argument-translation.md`.

### Launch-time info dump

Once translation is settled the entrypoint calls `vllm_shim.cli.info.collect`, writes the result as JSON to `/tmp/vllm-shim-info.json`, and prints a short summary to stderr. The dump captures the original argv, selected backend, original vs. resolved model path, revision, port allocation, the backend invocation, dropped args, env renames produced by the backend's `EnvTranslator`, the active shim config knobs, the HF cache state, and the AITER capture/restore plans. The `vllm-shim-info` console script (registered in `packages/vllm-shim/pyproject.toml`) just prints that file, so an operator can `kubectl exec` into the pod and run it instead of grepping logs. The set of shim-config and HF env keys surfaced lives in `_SHIM_CONFIG_KEYS` and `_HF_KEYS` in `vllm_shim.cli.info`; keep those in sync with the Configuration surface table below when adding knobs.

### AITER shape capture and config restore (ROCm only)

On ROCm backends (SGLang on AMD GPUs today), the entrypoint also does three AITER-specific things, gated on a ROCm GPU detection plus a resolvable HF cache:

1. **Restore.** Before launching the backend, point AITER's `AITER_CONFIG_*` env vars at any tuned CSVs found under `$VLLM_SHIM_HOME/aiter/configs/<bucket>/`. AITER reads the env at import time, so no symlinks or `/tmp` writes are needed. Operator-set `AITER_CONFIG_*` values always win. See `vllm_shim.aiter.restore`.
2. **Capture.** Spawn the backend with `stderr=PIPE` and start a `StreamTee` daemon (`vllm_shim.aiter.stream_tee`) that forwards every line to the real stderr while parsing `shape is M:... not found tuned config ...` lines and appending them, deduped, to `$VLLM_SHIM_HOME/aiter/shapes/<bucket>/<sanitized_model>/<parallelism>/<target>.csv`. The parallelism segment comes from the backend's `ParallelismExtractor` reading the *post-translation* argv (operators can pass either vLLM-style or backend-native flags, so we can't trust input-side flag names). See `vllm_shim.aiter.capture`.
3. **HIP online tuning anchor.** When the operator opts in via `HIP_ONLINE_TUNING=1` (or `=true`), AITER's gradlib reads/writes `./hip_online_tuning_res.csv` relative to the backend's CWD; the path is hardcoded in the C++ source (`repos/aiter/gradlib/csrc/hipbsolgemm.cu`) and no env var redirects it. The shim keeps the canonical file at `$VLLM_SHIM_HOME/hip_online_tuning_res.csv` and symlinks the CWD path onto it so accumulated tuning data survives pod restarts. See `vllm_shim.aiter.hip_online_tuning`.

On CUDA hosts and dev boxes all three no-op silently with stable reason strings in the launch-info dump. `$VLLM_SHIM_HOME` defaults to `~/.vllm-shim`; operators point it at a PV in production. The tuner step that turns captured shapes into tuned configs is the `vllm-shim-tune` console script (`vllm_shim.aiter.tune`); an operator runs it from a pod shell after capture has accumulated misses, and it writes into `$VLLM_SHIM_HOME/aiter/configs/<bucket>/`. See `docs/aiter.md` for path layout, prerequisites, the tuner CLI, and the operator surface.

Adjacent to AITER restore, the entrypoint also applies a small set of ROCm performance env-var defaults via `vllm_shim.cli.rocm_perf.rocm_perf_defaults(gpu, shim_home)` (generic + `gfx942`-gated additions, all `setdefault` so operator values win). MIOpen kernel cache lands under `$VLLM_SHIM_HOME/miopen` to share the PV survival story with AITER configs. The applied dict appears in the launch-info dump under `rocm_perf`. See `docs/rocm-perf.md`.

### The `vllm-entrypoints` stub package

`packages/vllm-entrypoints/src/vllm/` is a namespace stub whose `__main__.py` files redirect every `python -m vllm.X` invocation (e.g. `vllm.entrypoints.openai.api_server`) into `vllm_shim.cli.entrypoint.main`. Each leaf module is three lines: import `main`, `raise SystemExit(main())`. Do not add real vLLM API surface here; it exists purely to occupy the import namespace. See `docs/entrypoints.md`.

## Code conventions enforced by tooling or established by precedent

- **Absolute imports only.** Ruff's `TID` rule with `ban-relative-imports = "all"` rejects `from .foo import X`. Always write `from vllm_shim.cli.parser import ArgParser`.
- **Strict mypy.** `strict = true`, tests included. When you must escape, use a specific code (`# type: ignore[no-untyped-def]`).
- **Frozen dataclasses with `slots=True`** for value objects (see `vllm_shim.values.*`).
- **Module-level constants in `UPPER_CASE`** (`STRIP_PARAMS`, `ARG_MAP`, `_HOP_BY_HOP_HEADERS`).
- **Comments explain WHY, not WHAT.** The long block in `Supervisor._terminate_all` is the canonical example. Names should carry the WHAT.
- **No prose em dashes or en dashes** in source files or docs. Use commas, semicolons, colons, or periods. (This is a recurring style rule; grep before declaring docs done.)
- **Filter order in `Backend.__init__` is part of the contract.** `tests/unit/backend/sglang/test_backend.py::test_filters_in_documented_order` will catch a reorder; if you intentionally change order, update the test.
- **Python 3.12+** (`requires-python = ">=3.12"`, ruff `target-version = "py312"`).

## Configuration surface

The shim has no config file. Behavior is driven by CLI args (the `vllm serve` invocation) and a small set of env vars:

| Env var | Default | Effect |
|---|---|---|
| `VLLM_SHIM_BACKEND`        | `sglang`         | `sglang` or `trtllm`. |
| `VLLM_SHIM_HOME`           | `~/.vllm-shim`   | Root for the shim's persistent state (AITER capture + tuned configs). Point at a PV in production. |
| `VLLM_SHIM_LOG`            | `/tmp/vllm-shim.log` | Where 4xx/5xx error dumps are appended. Read once at module import. |
| `VLLM_SHIM_TUNE_AT_STARTUP_SECONDS` | unset (off) | When set to a positive integer, the entrypoint runs `vllm-shim-tune` between restore and backend spawn, with that many seconds as a hard wall-clock budget. Single-GPU deployments use this to fold tuning into pod restarts. See `docs/aiter.md`. |
| `SGLANG_TOOL_CALL_PARSER`  | `qwen3_coder`    | Forwarded to SGLang's `--tool-call-parser`. |
| `VLLM_SHIM_BACKEND_HOST` / `VLLM_SHIM_BACKEND_PORT` / `VLLM_SHIM_MIDDLEWARE_PORT` | derived | Set by the supervisor when spawning the middleware; not normally set by hand. |
| `TRTLLM_BACKEND`           | `pytorch`        | `pytorch`, `tensorrt`, or `_autodeploy`. |
| `TRTLLM_TOOL_PARSER`       | `qwen3_coder`    | Forwarded to `trtllm-serve --tool_parser`. |
| `TRTLLM_REASONING_PARSER`  | unset            | If set, appends `--reasoning_parser <value>`. |

Some constants are deliberately code-only (port offsets, shutdown grace, metrics cache TTL, httpx timeouts, haproxy timeouts, error-dump truncation, `STRIP_PARAMS`). Promoting one to env commits the project to a backwards-compatible name. See `docs/configuration.md`.

## Build matrix

The directory layout `docker/<backend>/Dockerfile.<platform>` *is* the matrix. Jenkins (`Jenkinsfile`) discovers combinations via `findFiles(glob: 'docker/*/Dockerfile.*')`, builds each in parallel via `docker buildx build` with registry-backed caching, and tags `${REGISTRY_URL}:${TAG}-${BACKEND}-${PLATFORM}`. Adding a backend or platform means dropping a Dockerfile at the right path; no pipeline edits. See `docs/build-and-deploy.md`.

## Where to read more

Topic docs under `docs/`: `aiter.md`, `architecture.md`, `argument-translation.md`, `backends.md`, `build-and-deploy.md`, `configuration.md`, `development.md`, `entrypoints.md`, `haproxy.md`, `metrics.md`, `middleware.md`, `rocm-perf.md`, `supervisor.md`. They are the authoritative reference for each layer; this file points at them.
