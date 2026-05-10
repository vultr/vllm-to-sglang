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

### Beyond ARG_MAP: the two pre-processors

A handful of vLLM inputs need handling that the (flag, value) shape of `ARG_MAP` can't express, so the SGLang translator runs two pre-processors before `translate_with_arg_map`:

1. **`strip_optimization_level`** (in `vllm_shim.backend._shared`, used by both backends). vLLM's `FlexibleArgumentParser` rewrites `-O3` / `-O=3` / `-Odecode` / `-O 3` into `--optimization-level <value>` before argparse runs. The shim's parser does no such rewrite, so the `-O…` shorthand arrives literally and has to be handled separately. Neither SGLang nor TRT-LLM has a CLI knob equivalent to vLLM's compilation level (SGLang's nearest knob is `--enable-torch-compile`), so every form is dropped. The `--optimization-level` long form is a regular `(None, True)` ARG_MAP entry.
2. **`_split_speculative_config`** (SGLang-only). vLLM's `--speculative-config` / `-sc` carries a JSON blob, and SGLang exposes the same configuration through several flat flags (`--speculative-algorithm`, `--speculative-num-steps`, …). The pre-processor extracts the JSON, parses it, and emits the SGLang flag list inline. Currently only `method=mtp` is recognised; the mapping is the SGLang-recommended MTP recipe (`EAGLE` algorithm, `num_steps = N`, `eagle_topk = 1`, `num_draft_tokens = N + 1`) from `repos/sglang/docs/basic_usage/deepseek_v32.md` and `repos/sglang/test/registered/8-gpu-models/test_deepseek_v3_mtp.py`. Unknown methods, malformed JSON, or missing `num_speculative_tokens` cause the flag to be dropped (and added to `dropped`); SGLang then runs without speculative decoding. TRT-LLM has no equivalent CLI surface, so the TRT-LLM translator just drops `--speculative-config` via ARG_MAP.

### Short aliases

vLLM accepts `-tp`, `-pp`, `-dp`, `-q`, `-n`, `-r`, `-asc`, `-sc`, `-ac`, `-cc`, `-ep`, `-dcp`, `-pcp`, and the `-dpa…-dpr` family for various data-parallel knobs. Neither SGLang nor TRT-LLM defines any of these. The two ARG_MAPs include explicit entries for all of them: those whose long-form has an equivalent are renamed to the long form (`-tp` → `--tp` for SGLang, `--tp_size` for TRT-LLM); the rest are dropped (with their value, where applicable).

## Full mapping

For the canonical, current list see `ARG_MAP` itself in each backend's `args.py` (`packages/vllm-shim/src/vllm_shim/backend/sglang/args.py` and `packages/vllm-shim/src/vllm_shim/backend/trtllm/args.py`).

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

Many vLLM flags (`--seed`, `--enable-prefix-caching`, `--no-enable-prefix-caching`, `--enforce-eager`, `--max-cpu-loras`, and others) map cleanly to entries in TRT-LLM's `--config <yaml>` extra-llm-api-options surface, but expanding YAML in the shim would require an ABC change or break the translator's pure-function property. Operators who need any of them pass `--config their.yaml` directly through the shim's pass-through; the YAML can carry whatever extra knobs `trtllm-serve`'s LLM API supports.

Coverage parity with the SGLang map: every vLLM flag (engine-level and frontend-level, in both dash and underscore form, plus the short aliases) has an explicit decision in `vllm_shim.backend.trtllm.args.ARG_MAP`. Source of truth for the trtllm-serve flag set is the click options in `repos/trtllm/tensorrt_llm/commands/serve.py`. The same `strip_optimization_level` pre-processor handles `-O…` shorthands; speculative decoding has no `trtllm-serve` CLI flag, so `--speculative-config` is dropped (operators wanting speculative decoding configure it via `--config their.yaml`).
