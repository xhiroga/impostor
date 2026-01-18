from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch


@dataclass(slots=True)
class MetricResult:
    model: str
    metric: str
    value: float
    frames_used: tuple[int, ...] | None = None
    detail: str | None = None


class EvaluationModel(Protocol):
    """Interface for pluggable evaluation models."""

    name: str

    def evaluate(
        self, frames: torch.Tensor, frame_indices: Sequence[int]
    ) -> list[MetricResult]:
        """Return metric results for the given frames."""

