"""Helpers shared between concrete Backend implementations.

Lives outside backend/base/ per docs/backends.md ('What stays in base/: ABCs, no helpers').
"""

import re
from collections.abc import Sequence

# Internal regex constants for Prometheus exposition parsing.
_RE_METRIC_LINE = re.compile(r"^(#\s+(?:HELP|TYPE)\s+)?(\w[\w:]*)(.*)")
_RE_SAMPLE_LINE = re.compile(r"^(\w[\w:]*)(\{[^}]*\})?\s+(.+)$")


def translate_with_arg_map(
    vllm_args: Sequence[str],
    arg_map: dict[str, tuple[str | None, bool]],
) -> tuple[list[str], list[str]]:
    """Walk vllm_args, applying arg_map: rename, drop, or pass through each token.

    Returns (translated_args, dropped_args). Pure function, no I/O.

    arg_map values are (target_flag_or_None, has_value):
      ('--new-name', True)  -> rename, with value following
      ('--new-name', False) -> rename, no value follows
      (None, True)          -> drop flag and its value
      (None, False)         -> drop flag only
    Keys not in arg_map pass through verbatim. The unknown-flag heuristic looks at the
    next token: if it's another flag, the unknown flag is treated as boolean; otherwise
    flag and value pass through together.
    """
    out: list[str] = []
    dropped: list[str] = []
    i = 0
    args = list(vllm_args)
    while i < len(args):
        arg = args[i]

        # --flag=value form
        if arg.startswith("--") and "=" in arg:
            flag, val = arg.split("=", 1)
            mapping = arg_map.get(flag)
            if mapping is None:
                out.append(arg)
            else:
                target, has_val = mapping
                if target is None:
                    dropped.append(arg)
                elif has_val:
                    out.extend([target, val])
                else:
                    out.append(target)
            i += 1
            continue

        # --flag (with possible separate value)
        mapping = arg_map.get(arg)
        if mapping is None:
            # Pass through. If next token looks like a value, take it too.
            if (
                i + 1 < len(args)
                and not args[i + 1].startswith("-")
                and arg.startswith("--")
            ):
                out.extend([arg, args[i + 1]])
                i += 2
            else:
                out.append(arg)
                i += 1
            continue

        target, has_val = mapping
        if target is None:
            dropped.append(arg)
            if has_val and i + 1 < len(args):
                dropped.append(args[i + 1])
                i += 2
            else:
                i += 1
        elif has_val:
            if i + 1 < len(args):
                out.extend([target, args[i + 1]])
                i += 2
            else:
                i += 1
        else:
            out.append(target)
            i += 1
    return out, dropped


def translate_prom_line(line: str, name_map: dict[str, str]) -> list[str]:
    """Translate a single Prometheus exposition line using name_map.

    - Sample lines (`name{labels} value`): rename via name_map; if the bare name has
      a histogram suffix (_bucket / _sum / _count), strip-look up-reattach.
    - HELP/TYPE comment lines: rename the metric name in place.
    - Anything else (or names not in name_map): return the line unchanged.

    Returns a list (always length 1) so callers can use list.extend uniformly.
    """
    m = _RE_SAMPLE_LINE.match(line)
    if m:
        name, labels, value = m.group(1), m.group(2) or "", m.group(3)
        target = name_map.get(name)
        if target:
            return [f"{target}{labels} {value}"]
        for suffix in ("_bucket", "_sum", "_count"):
            if name.endswith(suffix):
                base = name_map.get(name[: -len(suffix)])
                if base:
                    return [f"{base}{suffix}{labels} {value}"]
        return [line]

    m = _RE_METRIC_LINE.match(line)
    if m:
        prefix, name, rest = m.group(1) or "", m.group(2), m.group(3)
        target = name_map.get(name)
        if target:
            return [f"{prefix}{target}{rest}"]
    return [line]


def strip_optimization_level(args: list[str]) -> tuple[list[str], list[str]]:
    """Strip vLLM's ``-O`` family (``-O3``, ``-O=3``, ``-Odecode``, ``-O 3``).

    vLLM's FlexibleArgumentParser pre-expands these into ``--optimization-level
    <value>`` before argparse sees them. We never go through that parser, so
    we receive the raw shorthand and have to handle it here. Neither SGLang
    nor TRT-LLM has a direct equivalent CLI knob, so every form gets dropped;
    the ``--optimization-level`` long form is dropped via each backend's
    ARG_MAP.
    """
    out: list[str] = []
    dropped: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "-O":
            dropped.append(arg)
            if i + 1 < len(args):
                dropped.append(args[i + 1])
                i += 2
            else:
                i += 1
            continue
        if arg.startswith("-O") and not arg.startswith("--"):
            dropped.append(arg)
            i += 1
            continue
        out.append(arg)
        i += 1
    return out, dropped


def vllm_synthesized_tail() -> list[str]:
    """Return the vLLM-shape synthesized series every backend emits at the end of /metrics.

    Two series, both vLLM-only (no backend equivalent):
    - vllm:healthy_pods_total: always 1; if the middleware is responding, the pod is healthy enough.
    - vllm:num_requests_swapped: always 0; no current backend has a swap notion.
    """
    return [
        "# HELP vllm:healthy_pods_total Number of healthy vLLM pods",
        "# TYPE vllm:healthy_pods_total gauge",
        'vllm:healthy_pods_total{endpoint="default"} 1',
        "# HELP vllm:num_requests_swapped Number of swapped requests",
        "# TYPE vllm:num_requests_swapped gauge",
        "vllm:num_requests_swapped 0",
    ]
