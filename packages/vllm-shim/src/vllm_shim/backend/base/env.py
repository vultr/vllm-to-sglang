"""EnvTranslator ABC: rename selected vLLM env vars into the backend's namespace."""

from abc import ABC, abstractmethod
from collections.abc import Mapping


class EnvTranslator(ABC):
    """Translates vLLM-namespaced env vars into the backend's env namespace.

    Implementations are pure functions over a Mapping (typically ``os.environ``)
    that return a fresh ``dict`` ready to pass to ``subprocess.Popen(env=...)``.
    The vLLM-side names are left in place; concrete backends are expected to
    ignore env vars they don't understand. If the user has already set the
    backend-side name explicitly, the translator does NOT overwrite it.
    """

    @abstractmethod
    def translate(self, parent_env: Mapping[str, str]) -> dict[str, str]:
        """Returns a child env dict. Pure function, no I/O."""
