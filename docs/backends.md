# Backends

A "backend" in this codebase is everything that's specific to a particular inference engine: how its CLI is shaped, how its metrics are named, how its API quirks differ from vLLM's. Today the implemented backends are SGLang and TensorRT-LLM, with the abstraction structured to admit further engines (e.g., LMDeploy) as contained changes.

The contract is in `packages/vllm-shim/src/vllm_shim/backend/base/`. The dispatch is in `vllm_shim.backend.registry`. Concrete implementations are `vllm_shim.backend.sglang` and `vllm_shim.backend.trtllm`.

## The `Backend` ABC

```python
class Backend(ABC):
    name: ClassVar[str]
    health_path: ClassVar[str] = "/health"
    metrics_path: ClassVar[str] = "/metrics"

    args: ArgTranslator
    env: EnvTranslator
    metrics: MetricsTranslator
    launcher: Launcher
    filters: tuple[RequestFilter, ...]
    parallelism: ParallelismExtractor

    @abstractmethod
    def __init__(self) -> None: ...
```

A backend is a bag of six components plus two class-level constants. Subclasses set the six instance attributes in `__init__`. There's no logic on the base class; `Backend` is purely a contract object.

The six components are each their own ABC:

| Component | ABC | Role |
|---|---|---|
| `args` | `ArgTranslator` | `translate(vllm_args) -> (backend_args, dropped_args)`. Pure function. See `docs/argument-translation.md`. |
| `env` | `EnvTranslator` | `translate(parent_env) -> dict[str, str]`. Pure function over `os.environ` that adds backend-side renames for selected `VLLM_*` env vars. See `docs/configuration.md`. |
| `launcher` | `Launcher` | `build_command(model, address, extra_args) -> list[str]`. Builds the subprocess argv. |
| `metrics` | `MetricsTranslator` | `translate(prom_text) -> str`. Rewrites Prometheus exposition. See `docs/metrics.md`. |
| `filters` | `tuple[RequestFilter, ...]` | Body-rewriting filters that run in declared order. See `docs/middleware.md`. |
| `parallelism` | `ParallelismExtractor` | `extract(post_translation_argv) -> Parallelism`. Reads tp/ep/pp from the argv that's about to be handed to the launcher. Used by the AITER shape-capture path; see `docs/aiter.md`. |

All five are intentionally stateless or self-contained. The supervisor and middleware never reach into a backend's internals; they only call these methods.

`health_path` is a class-level constant for the upstream `/health` URL. SGLang exposes `/health`; another backend might use `/v1/health` or `/healthz` and would override this. haproxy's `httpchk` and the middleware's `HealthHandler` both read it.

`metrics_path` works the same way for the upstream `/metrics` URL: SGLang inherits the `/metrics` default; TRT-LLM overrides to `/prometheus/metrics`. The middleware's `MetricsHandler` reads this when scraping.

`name` is a class-level identifier (`"sglang"`) used as the registry key.

## The registry

```python
# packages/vllm-shim/src/vllm_shim/backend/registry.py
_BACKENDS: dict[str, type[Backend]] = {
    "sglang": SGLangBackend,
    "trtllm": TRTLLMBackend,
}

def select() -> Backend:
    name = os.environ.get("VLLM_SHIM_BACKEND", "sglang")
    cls = _BACKENDS.get(name)
    if cls is None:
        raise ValueError(...)
    return cls()
```

Two callers: `entrypoint.main` (the supervisor process) and `middleware.app.run` (the middleware process). Both pick the same backend because they read the same env var, but they don't share a Python instance; each process constructs its own.

The registry is deliberately tiny. There's no plugin discovery, no entry-point scanning, no decorator. To wire up a new backend, edit `_BACKENDS`. The cost of that edit is one line; the cost of plugin infrastructure is forever.

## How it's wired

`SGLangBackend.__init__` is the canonical example:

```python
class SGLangBackend(Backend):
    name: ClassVar[str] = "sglang"

    def __init__(self) -> None:
        self.args = SGLangArgTranslator()
        self.env = SGLangEnvTranslator()
        self.metrics = SGLangMetricsTranslator()
        self.launcher = SGLangLauncher()
        self.filters = (StripVLLMParams(), FixToolSchemas())
        self.parallelism = SGLangParallelismExtractor()
```

Filter order is documented in `tests/unit/backend/sglang/test_backend.py::test_filters_in_documented_order`. If you change it, change the test. Stripping happens before schema fixing because the stripper might delete keys the fixer would otherwise walk.

## Adding a new backend

The recipe, end to end:

### 1. Pick a directory

`packages/vllm-shim/src/vllm_shim/backend/<name>/` with the same shape as `sglang/`:

```
__init__.py
args.py        # XArgTranslator(ArgTranslator)
env.py         # XEnvTranslator(EnvTranslator)
launcher.py    # XLauncher(Launcher)
metrics.py     # XMetricsTranslator(MetricsTranslator)
backend.py     # XBackend(Backend)
filter/        # zero or more RequestFilter subclasses
```

The package can be flatter or deeper depending on complexity; `filter/` only exists if there are filters.

### 2. Implement the six components

- **`ArgTranslator.translate`**: takes the passthrough flags from `ArgParser`, returns `(backend_argv, dropped_argv)`. Pure function. Build a flag-rewrite map analogous to `ARG_MAP`, plus any custom logic.
- **`EnvTranslator.translate`**: takes a parent env mapping (typically `os.environ`), returns a child env dict. Pure function. Build an `ENV_MAP` of `VLLM_*` -> backend-side renames; the helper `translate_env_with_map` is usually all you need. See `docs/configuration.md` for the rationale.
- **`Launcher.build_command`**: takes `(model, ServiceAddress, extra_args)`, returns the full subprocess argv. Prefer the backend's installed console script (e.g. `sglang serve`, `trtllm-serve`) so the launcher does not need to share a Python interpreter with the backend.
- **`MetricsTranslator.translate`**: takes Prometheus exposition text, returns Prometheus exposition text. If the backend has no native `/metrics`, you can return synthesized vLLM-format output and skip the rename step.
- **`RequestFilter`s**: only needed if the backend rejects request shapes vLLM clients send. Each one sets `applies_to(method, path)` to gate when it runs.
- **`ParallelismExtractor.extract`**: takes the *post-translation* argv (after `ArgTranslator` has run), returns a `Parallelism` value. Pure function. Each backend knows its own native flag spellings for tp/ep/pp; the helper `last_int_for_flags` handles the common pattern.

### 3. Wire the backend class

```python
class XBackend(Backend):
    name: ClassVar[str] = "x"
    health_path: ClassVar[str] = "/health"  # override if different

    def __init__(self) -> None:
        self.args = XArgTranslator()
        self.env = XEnvTranslator()
        self.metrics = XMetricsTranslator()
        self.launcher = XLauncher()
        self.filters = (...)
        self.parallelism = XParallelismExtractor()
```

### 4. Register it

```python
# vllm_shim/backend/registry.py
from vllm_shim.backend.x.backend import XBackend

_BACKENDS = {
    "sglang": SGLangBackend,
    "x": XBackend,
}
```

Now `VLLM_SHIM_BACKEND=x` selects it.

### 5. Tests

Mirror the SGLang test layout under `tests/unit/backend/<name>/`. The base ABCs have a parametrized test (`tests/unit/backend/test_base.py`) that confirms they can't be instantiated directly; you don't need to touch that. The registry tests (`tests/unit/backend/test_registry.py`) should grow a "registers X" assertion if you want it covered.

### 6. Dockerfile

A new backend almost certainly means a new base image. The Dockerfile layout is `docker/<backend>/Dockerfile.<platform>` (e.g. `docker/x/Dockerfile.cuda`). The Jenkins matrix discovers them via `docker/*/Dockerfile.*` glob. See `docs/build-and-deploy.md`.

## What stays in `base/`

The base layer is intentionally thin: ABCs, no helpers, no shared code. Anything that would let two backends share an implementation should live in a separate utility module under the backend's own package or in `vllm_shim.values` if it's truly generic. This keeps backend coupling explicit; you can read all of `vllm_shim.backend.sglang` without grepping for `base.X` helpers.

## What lives one level up

`vllm_shim.values` (`ParsedArgs`, `PortAllocation`, `ServiceAddress`) is the dependency the backend layer is allowed to reach into. The backend layer must not depend on `vllm_shim.middleware` or `vllm_shim.cli`; that's the rule that keeps the architecture diagram in `docs/architecture.md` honest.
