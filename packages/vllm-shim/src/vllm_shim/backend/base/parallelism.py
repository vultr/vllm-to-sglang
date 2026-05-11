"""ParallelismExtractor ABC: read tp/ep/pp out of an already-translated argv.

Every backend has its own flag spellings for parallel degrees (SGLang
uses ``--tp-size``/``--ep-size``/``--pipeline-parallel-size``, TRT-LLM
uses ``--tp_size``/``--ep_size``/``--pp_size``). Because the shim's
passthrough chain lets users pass either vLLM-style or backend-native
flags, the only argv we can reliably inspect for topology is the
*post-translation* argv that's about to be handed to the backend's
launcher. The extractor is therefore a backend-owned component.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from vllm_shim.values.parallelism import Parallelism


class ParallelismExtractor(ABC):
    """Reads tp/ep/pp values out of a backend-native argv."""

    @abstractmethod
    def extract(self, args: Sequence[str]) -> Parallelism:
        """Returns a Parallelism. Unknown flags default to 1."""
