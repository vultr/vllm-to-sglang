"""vLLM ParallelismExtractor: reads tp/ep/pp from a vLLM-native argv.

Flag spellings come from ``repos/vllm/vllm/config/parallel.py``. vLLM's
argparse setup accepts both dashed and underscored long-flag forms plus
the ``-tp`` / ``-pp`` short aliases; the underlying argparse behaviour
treats those as equivalent.
"""

from collections.abc import Sequence

from vllm_shim.backend._shared import last_int_for_flags
from vllm_shim.backend.base.parallelism import ParallelismExtractor
from vllm_shim.values.parallelism import Parallelism

_TP_FLAGS = frozenset({"--tensor-parallel-size", "--tensor_parallel_size", "-tp"})
_PP_FLAGS = frozenset({"--pipeline-parallel-size", "--pipeline_parallel_size", "-pp"})
_EP_FLAGS = frozenset({"--expert-parallel-size", "--expert_parallel_size"})


class VLLMParallelismExtractor(ParallelismExtractor):
    """Scan post-translation vLLM argv for parallel degree flags."""

    def extract(self, args: Sequence[str]) -> Parallelism:
        return Parallelism(
            tp=last_int_for_flags(args, _TP_FLAGS) or 1,
            ep=last_int_for_flags(args, _EP_FLAGS) or 1,
            pp=last_int_for_flags(args, _PP_FLAGS) or 1,
        )
