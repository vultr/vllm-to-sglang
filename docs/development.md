# Development

The repo is a uv workspace with two member packages. All tooling (pytest, mypy, ruff) is configured at the workspace root and runs across both packages.

## Setup

```bash
uv sync
```

That's the whole setup. `uv sync` resolves the workspace, creates `.venv/`, and installs:

- `vllm-shim` (`packages/vllm-shim/`) in editable mode.
- `vllm-entrypoints` (`packages/vllm-entrypoints/`) in editable mode.
- The dev dependency group: pytest, pytest-asyncio, pytest-httpx, mypy, ruff.

The runtime dependencies (FastAPI, httpx, uvicorn) come in transitively from `vllm-shim`. SGLang itself is not installed locally; it's a runtime dependency satisfied by the Docker base image. Tests fake or mock the backend wherever they need it.

## Workspace layout

```
vllm-shim/
├── pyproject.toml               # workspace root: tool config, dev deps
├── uv.lock                      # lockfile for the whole workspace
├── packages/
│   ├── vllm-shim/               # the actual implementation
│   │   ├── pyproject.toml
│   │   ├── src/vllm_shim/
│   │   └── tests/
│   └── vllm-entrypoints/        # the namespace stub
│       ├── pyproject.toml
│       ├── src/vllm/
│       └── tests/
├── docker/                      # per-(backend, platform) Dockerfiles
├── docs/                        # this directory
└── Jenkinsfile                  # CI matrix
```

The workspace section in the root `pyproject.toml`:

```toml
[tool.uv.workspace]
members = ["packages/*"]

[tool.uv.sources]
vllm-shim       = { workspace = true }
vllm-entrypoints = { workspace = true }
```

`workspace = true` tells uv to resolve these as in-tree packages rather than fetching from PyPI, so `vllm-entrypoints`'s declared dependency on `vllm-shim` resolves to the local source.

## Tests

```bash
uv run pytest
```

Discovery is configured in the root `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["packages/vllm-shim/tests", "packages/vllm-entrypoints/tests"]
asyncio_mode = "auto"
```

`asyncio_mode = "auto"` means async test functions don't need explicit `@pytest.mark.asyncio` decorators (though some still have them for clarity).

Two test layers:

| Layer | Path | What it covers |
|---|---|---|
| Unit | `packages/vllm-shim/tests/unit/` | Individual modules: parser, translator, filters, value objects. |
| Integration | `packages/vllm-shim/tests/integration/` | The FastAPI app end-to-end via `ASGITransport`, the supervisor with real subprocesses. |

The integration tests don't need a running SGLang. `test_app.py` defines a `FakeBackend` with no-op `args`/`launcher` and minimal `metrics`/`filters`; `pytest-httpx` mocks the upstream HTTP calls. `test_supervisor.py` spawns short-lived `python -c "time.sleep(N)"` processes as stand-ins.

### Common patterns

- **`pytest-httpx`** for upstream mocking. Add response stubs with `httpx_mock.add_response(...)`. The integration tests are good templates.
- **`monkeypatch.setenv` / `delenv`** for env-driven behavior (registry tests, launcher tests).
- **`tmp_path`** fixture for tests that write files (haproxy config, error dumps).
- **No global state in tests.** The httpx client is process-global, but tests instantiate fresh `FastAPI` apps with `ASGITransport`, which gets its own client lifecycle via the lazy fallback in `vllm_shim.middleware.http_client.get_client`.

## Type checking

```bash
uv run mypy
```

Strict mode is on:

```toml
[tool.mypy]
python_version = "3.12"
strict = true
files = [
    "packages/vllm-shim/src",
    "packages/vllm-shim/tests",
    "packages/vllm-entrypoints/src",
    "packages/vllm-entrypoints/tests",
]
```

Tests are type-checked. A few `# type: ignore` escape hatches exist where pytest-style fixtures lose typing fidelity (e.g., `monkeypatch`); use them sparingly with the specific error code (`# type: ignore[no-untyped-def]`).

`vllm` itself is allowed to be untyped:

```toml
[[tool.mypy.overrides]]
module = ["vllm"]
ignore_missing_imports = true
```

This is for the shadowed namespace; the stubs don't import the real vLLM, so this is mostly defensive.

## Linting and formatting

```bash
uv run ruff check .
uv run ruff format .
```

Configured in the root `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF", "TID"]

[tool.ruff.lint.flake8-tidy-imports]
ban-relative-imports = "all"
```

Ruleset highlights:

- `I`: import sorting (replaces isort).
- `UP`: pyupgrade (modernize syntax for the target Python version).
- `B`: bugbear (real bug patterns, not style nits).
- `SIM`: simplify suggestions.
- `TID` with `ban-relative-imports = "all"`: every internal import must be absolute (`from vllm_shim.cli.parser import ArgParser`, never `from .parser import ArgParser`). Helps with refactor safety and module portability.

## Running the shim locally

The shim doesn't run end-to-end without a real inference backend, but you can exercise individual layers:

```bash
# Run just the middleware against a fake SGLang
SGLANG_HOST=localhost SGLANG_PORT=9999 MIDDLEWARE_PORT=8080 \
    uv run vllm-shim-middleware
```

For a full local run, use Docker (see `docs/build-and-deploy.md`).

## Code style notes

A few conventions worth knowing before writing changes:

- **Frozen dataclasses with `slots=True`** for value objects (see `vllm_shim.values.*`). Cheap, immutable, and equality/hash work for free.
- **Module-level constants in UPPER_CASE** (e.g., `STRIP_PARAMS`, `ARG_MAP`, `_HOP_BY_HOP_HEADERS`).
- **No comments explaining what code does.** Names should carry that. Comments are reserved for *why*: the long block in `Supervisor._terminate_all` is the canonical example.
- **Absolute imports always.** Enforced by ruff's TID rule.
- **No prose em dashes in source files or docs.** Use commas, semicolons, colons, or periods.

## Adding a new package

If a future backend needs its own substantial codebase, it can become its own workspace member:

1. `mkdir -p packages/<name>/src/<name>/`
2. Add a `pyproject.toml` matching the others' shape.
3. Reference it in the root `pyproject.toml`'s `[tool.uv.sources]`.
4. `uv sync`.

For most cases, adding a backend module under `packages/vllm-shim/src/vllm_shim/backend/<name>/` is enough; see `docs/backends.md`.
