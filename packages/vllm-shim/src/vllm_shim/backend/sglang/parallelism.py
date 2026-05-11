"""SGLang ParallelismExtractor: reads tp/ep/pp from a SGLang-native argv.

Flag spellings come straight from SGLang's argparse setup in
``repos/sglang/python/sglang/srt/server_args.py``:

- TP: ``--tp-size`` (canonical) / ``--tensor-parallel-size`` (alias).
  Our ARG_MAP also emits the bare ``--tp`` form, which argparse accepts
  as a prefix abbreviation of ``--tp-size``; we recognise it here too
  so post-translation argv reads back correctly.
- PP: ``--pipeline-parallel-size`` (canonical) / ``--pp-size`` (alias).
- EP: ``--expert-parallel-size`` (canonical) / ``--ep-size`` / ``--ep``.
"""

from collections.abc import Sequence

from vllm_shim.backend._shared import last_int_for_flags
from vllm_shim.backend.base.parallelism import ParallelismExtractor
from vllm_shim.values.parallelism import Parallelism

_TP_FLAGS = frozenset({"--tp", "--tp-size", "--tensor-parallel-size"})
_PP_FLAGS = frozenset({"--pipeline-parallel-size", "--pp-size"})
_EP_FLAGS = frozenset({"--expert-parallel-size", "--ep-size", "--ep"})


class SGLangParallelismExtractor(ParallelismExtractor):
    """Scan post-translation SGLang argv for parallel degree flags."""

    def extract(self, args: Sequence[str]) -> Parallelism:
        return Parallelism(
            tp=last_int_for_flags(args, _TP_FLAGS) or 1,
            ep=last_int_for_flags(args, _EP_FLAGS) or 1,
            pp=last_int_for_flags(args, _PP_FLAGS) or 1,
        )
