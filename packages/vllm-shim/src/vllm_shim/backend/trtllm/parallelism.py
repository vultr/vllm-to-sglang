"""TRT-LLM ParallelismExtractor: reads tp/ep/pp from a trtllm-serve argv.

Flag spellings come from ``repos/trtllm/tensorrt_llm/commands/serve.py``:

- TP: ``--tp_size``.
- PP: ``--pp_size``.
- EP: ``--moe_expert_parallel_size`` (canonical) / ``--ep_size`` (alias).
"""

from collections.abc import Sequence

from vllm_shim.backend._shared import last_int_for_flags
from vllm_shim.backend.base.parallelism import ParallelismExtractor
from vllm_shim.values.parallelism import Parallelism

_TP_FLAGS = frozenset({"--tp_size"})
_PP_FLAGS = frozenset({"--pp_size"})
_EP_FLAGS = frozenset({"--moe_expert_parallel_size", "--ep_size"})


class TRTLLMParallelismExtractor(ParallelismExtractor):
    """Scan post-translation trtllm-serve argv for parallel degree flags."""

    def extract(self, args: Sequence[str]) -> Parallelism:
        return Parallelism(
            tp=last_int_for_flags(args, _TP_FLAGS) or 1,
            ep=last_int_for_flags(args, _EP_FLAGS) or 1,
            pp=last_int_for_flags(args, _PP_FLAGS) or 1,
        )
