# Argument translation

The shim accepts vLLM's CLI grammar (`vllm serve <model> [flags]`) and produces the selected backend's CLI grammar (`sglang serve --model-path <model> [flags]` for SGLang, `trtllm-serve <model> [flags]` for TRT-LLM). Two stages do this:

1. **`ArgParser`** (`packages/vllm-shim/src/vllm_shim/cli/parser.py`): extracts what the supervisor itself needs and packages the rest verbatim. Backend-agnostic.
2. **`<Backend>ArgTranslator`**: rewrites flags into the backend's vocabulary. `SGLangArgTranslator` (`vllm_shim.backend.sglang.args`) and `TRTLLMArgTranslator` (`vllm_shim.backend.trtllm.args`) are the two implementations.

Splitting them this way keeps the supervisor backend-agnostic: the parser knows nothing about either engine, and each translator is just a `Backend.args` slot that another backend could swap out.

## Stage 1: `ArgParser`

The parser walks `argv` and only consumes:

- `serve`: dropped (it's the vLLM subcommand name).
- `--host` / `--host=…`: captured for haproxy and SGLang. Default `0.0.0.0`.
- `--port` / `--port=…`: captured for the port allocation. Default `8000`.
- `--model` / `--model-name` (and `=`-form): captured as the model identifier.
- A bare positional that doesn't start with `-`, when no model has been seen yet: captured as the model.

Everything else is appended to `passthrough` in original order. The result is a `ParsedArgs` (`packages/vllm-shim/src/vllm_shim/values/parsed_args.py`):

```python
ParsedArgs(model=str, host=str, port=int, passthrough=tuple[str, ...])
```

If no model was found, the parser raises `ValueError("No model specified in vLLM args")`. That's the only failure mode at this stage; unknown flags are not errors here, they're just deferred to the translator.

The parser is deliberately lossy in only one direction: `--host` and `--port` are removed from the passthrough because the supervisor reconstructs them when it builds the SGLang command (with the +1 port offset, see `docs/architecture.md`). The model is also removed because it becomes the `--model-path` positional argument the launcher injects.

## Stage 2: `SGLangArgTranslator`

The translator consumes the `passthrough` tuple and produces `(backend_args, dropped_args)`. It's a pure function: no I/O, no env reads.

The core is `ARG_MAP`, a `dict[str, tuple[str | None, bool]]`:

```python
"--tensor-parallel-size":   ("--tp", True),                   # rename, takes value
"--enforce-eager":          ("--disable-cuda-graph", False),  # rename, no value
"--enable-prefix-caching":  (None, False),                    # silently drop
"--swap-space":             (None, True),                     # drop with its value
```

Three behaviors per entry:

| `(target, has_value)`     | Effect |
|---|---|
| `("--new-name", True)`    | Rewrite flag, keep value (`--old X` → `--new-name X`). |
| `("--new-name", False)`   | Rewrite flag, no value follows. |
| `(None, True)`            | Drop flag and its value. |
| `(None, False)`           | Drop flag only. |

Anything not in `ARG_MAP` passes through verbatim. That's deliberate: SGLang accepts many flag names identical to vLLM's (`--trust-remote-code`, `--dtype`, `--quantization`, …), and any genuinely unknown flag has a chance of being SGLang-only and worth forwarding.

### Equals-form handling

Both `--flag value` and `--flag=value` are accepted. The translator splits on `=` first, looks up the bare flag name, then routes through the same dispatch:

```
--max-model-len=4096   →   --context-length 4096
```

### Underscore variants

vLLM accepts both `--gpu-memory-utilization` and `--gpu_memory_utilization`. Both forms are explicit keys in `ARG_MAP` (similarly for `--max_model_len`, `--enforce_eager`, `--trust_remote_code`). There's no normalization step; if a vLLM flag has an underscore variant in the wild, it gets its own entry.

### Unknown-flag heuristic

When a flag isn't in `ARG_MAP`, the translator looks at the next token:

- If it's another flag (starts with `-`), the unknown flag is treated as a boolean and emitted alone.
- If it's a value-shaped token, both are emitted as a flag/value pair.

This means `--foo bar` and `--foo --bar` are both handled correctly without an explicit registration.

### Dropped vs. ignored

Two reasons an arg shows up in `dropped`:

1. **No SGLang equivalent.** `--swap-space`, `--block-size`, `--num-gpu-blocks-override`, etc. SGLang has no swap, no block-size knob (uses radix attention), and no manual block-count override.
2. **vLLM-default behavior already matches.** `--enable-prefix-caching` is on by default in SGLang's radix cache; `--enable-chunked-prefill` is the SGLang default; log flags (`--disable-log-requests`, `--disable-log-stats`) have no SGLang knob and the defaults are quiet enough.

The `dropped` list is currently informational; `entrypoint.main` discards it (the variable is named `_dropped`). Worth keeping in the API in case a future caller wants to log or surface it.

## Full mapping

For the canonical, current list see `ARG_MAP` itself (`packages/vllm-shim/src/vllm_shim/backend/sglang/args.py`). The README has a hand-curated table grouped by behavior; keep that in sync if you add entries.

## Adding a new translation

Most additions are one line. To rename a flag:

```python
"--vllm-name": ("--sglang-name", True),    # if it takes a value
"--vllm-name": ("--sglang-name", False),   # if it's boolean
```

To drop a flag silently:

```python
"--vllm-name": (None, True),   # has a value to swallow
"--vllm-name": (None, False),  # bare flag
```

Add a unit test in `packages/vllm-shim/tests/unit/backend/sglang/test_args.py` covering both `--flag value` and `--flag=value` forms. The existing tests are tight templates.

## Edge cases the parser handles

- `serve` may appear anywhere, not just as the first token (the parser skips it wherever it is).
- A bare positional after `--model …` is *not* re-captured as a model; the first one wins.
- `--port` with an `=`-form coexists with `--port` followed by a separate value; both are accepted.

## TRT-LLM ARG_MAP

The TRT-LLM backend uses a parallel `ARG_MAP` in `vllm_shim.backend.trtllm.args`. Same shape, different contents: TRT-LLM's CLI uses underscores (`--tp_size`, `--max_seq_len`, `--kv_cache_free_gpu_memory_fraction`) where vLLM uses dashes. Both dash and underscore variants of vLLM flags appear as explicit map keys (no normalization layer), matching the SGLang convention.

The canonical, current list lives in `vllm_shim.backend.trtllm.args.ARG_MAP`.

One semantic mismatch worth flagging: `--gpu-memory-utilization` maps to `--kv_cache_free_gpu_memory_fraction`, but the two values mean different things. vLLM's value is the fraction of total GPU memory the engine is allowed to use; TRT-LLM's is the fraction of free GPU memory reserved for KV cache after weights and activations are allocated. The numeric value is forwarded unchanged. Same pragmatic compromise SGLang carries.

YAML config injection is intentionally out of scope. Five vLLM flags (`--seed`, `--enable-prefix-caching`, `--no-enable-prefix-caching`, `--enforce-eager`, `--max-cpu-loras`) have clean YAML equivalents in TRT-LLM's `--config <yaml>` extra-llm-api-options surface. Honoring them via the shim would require an ABC change or breaking the translator's pure-function property; operators who need them pass `--config their.yaml` directly through the shim's pass-through.
