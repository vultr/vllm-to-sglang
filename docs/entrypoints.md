# vllm-entrypoints

`packages/vllm-entrypoints/` is a tiny stub package whose only purpose is to occupy Python's import namespace where the real `vllm` package would be. It ships nothing of substance: just `__main__.py` files that redirect every common `python -m vllm.X` invocation into the shim.

## Why it exists

The vLLM production stack and many existing scripts launch the server two ways:

- `vllm serve <model> [flags]`: handled by the `vllm` console script (defined in `vllm-shim`'s `pyproject.toml`).
- `python -m vllm.entrypoints.openai.api_server <flags>`: handled by this package.

The console script alone covers the first form but not the second. If you `pip install vllm-shim` without this package, `python -m vllm.entrypoints.openai.api_server` would either fail with `No module named vllm` or, worse, find the real vLLM if it's installed alongside and start that instead.

## Layout

```
packages/vllm-entrypoints/src/vllm/
├── __init__.py
├── __main__.py
└── entrypoints/
    ├── __init__.py
    ├── cli/
    │   ├── __init__.py
    │   └── main.py
    └── openai/
        ├── __init__.py
        └── api_server.py
```

The `__init__.py` files are empty markers (the directory hierarchy is the API).

The `__main__.py` and the two leaf modules (`cli/main.py`, `openai/api_server.py`) are all the same three lines:

```python
from vllm_shim.cli.entrypoint import main

raise SystemExit(main())
```

That's the whole package. Each module redirects to the shim's main and propagates its exit code via `SystemExit`.

## What gets intercepted

Every form the vLLM ecosystem uses to launch the server:

| Invocation | Routes through |
|---|---|
| `vllm serve <model>` | The `vllm` console script (defined in `vllm-shim/pyproject.toml`). |
| `python -m vllm <args>` | `vllm/__main__.py` |
| `python -m vllm.entrypoints <args>` | (no `__main__` here; not commonly used) |
| `python -m vllm.entrypoints.cli <args>` | `vllm/entrypoints/cli/__init__.py` would be needed; instead `cli/main.py` exists for `python -m vllm.entrypoints.cli.main`. |
| `python -m vllm.entrypoints.cli.main <args>` | `cli/main.py` |
| `python -m vllm.entrypoints.openai.api_server <args>` | `openai/api_server.py` |

The argv passed by the caller becomes `sys.argv[1:]` inside `vllm_shim.cli.entrypoint.main`, where the regular `ArgParser` consumes it. The model and flags are extracted normally; the supervisor doesn't care which entry vector was used.

## Why it's a separate package

Two reasons it isn't merged into `vllm-shim`:

1. **Namespace ownership.** The `vllm/` directory must be a top-level package, since Python looks it up at the front of `sys.path`. Putting it inside `vllm-shim`'s `src/` would either collide with the `vllm_shim/` package or pollute `vllm-shim`'s wheel layout. A separate package with its own `[tool.hatch.build.targets.wheel] packages = ["src/vllm"]` is cleaner.
2. **Optional install.** A user who only ever invokes `vllm serve` doesn't strictly need the namespace stub. In practice the Dockerfiles always install both, but the split lets you depend on `vllm-shim` alone in environments where you control how the server is launched.

`vllm-entrypoints` declares `dependencies = ["vllm-shim"]` so installing it pulls the shim in transitively. The dev workspace wires both packages together via `[tool.uv.sources]` in the root `pyproject.toml`.

## What this is not

This is not a re-implementation or fork of the vLLM Python API. There's no `LLM` class, no `SamplingParams`, no model-loading code. Anything that imports from `vllm` expecting actual inference functionality will get an empty namespace and fail at attribute access. The stub only covers the launch surface (`python -m`).

If a downstream component needs to import vLLM as a library (rather than launch it as a server), this shim won't help, and that's by design. The shim's premise is intercepting the *server launch*, not impersonating the library.

## Tests

`packages/vllm-entrypoints/tests/test_smoke.py` confirms both packages are importable. It doesn't exercise the redirect because that would require a subprocess; the redirect is three lines and the import alone is enough to catch packaging regressions.

## Adding a new entry point

If a new `python -m vllm.X.Y` invocation appears in the wild, add the corresponding directory and a three-line `__main__.py` (or named module). Keep the contents identical to the existing stubs; divergence here is a footgun.
