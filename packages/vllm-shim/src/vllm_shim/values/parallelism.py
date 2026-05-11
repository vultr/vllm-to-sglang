"""Parallelism: tensor / expert / pipeline degrees extracted from backend argv.

This value is part of the AITER shape-capture path layout. Captured shapes
are partitioned by (hardware bucket, model, parallelism) so a tuner can
later replay each combination against the matching topology. ``tp``/``ep``/
``pp`` mirror the three knobs both SGLang and TRT-LLM expose; defaults of
1 match each backend's "off" semantics (no parallelism = degree 1).
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Parallelism:
    """Tensor / expert / pipeline parallel degrees. 1 means "off"."""

    tp: int = 1
    ep: int = 1
    pp: int = 1

    def path_segment(self) -> str:
        """Filesystem-safe segment encoding the non-default knobs.

        TP is always included so the segment is never empty; EP and PP
        are appended only when greater than 1. Examples: ``tp1``,
        ``tp8``, ``tp8-ep8``, ``tp4-pp2``, ``tp4-ep4-pp2``.
        """
        parts = [f"tp{self.tp}"]
        if self.ep > 1:
            parts.append(f"ep{self.ep}")
        if self.pp > 1:
            parts.append(f"pp{self.pp}")
        return "-".join(parts)
