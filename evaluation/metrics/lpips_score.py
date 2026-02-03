from __future__ import annotations

from typing import Sequence

import torch

from .base import EvaluationModel, MetricResult


class LpipsNotAvailable(RuntimeError):
    """Raised when lpips package is not installed."""


def _load_lpips():
    try:
        import lpips  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise LpipsNotAvailable(
            "lpips パッケージが必要です。`pip install lpips` を実行してください。"
        ) from exc
    return lpips.LPIPS(net="alex")


class LPIPSModel(EvaluationModel):
    """Learned perceptual image patch similarity (LPIPS)."""

    name = "lpips"

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lpips = _load_lpips().to(self.device)
        self.lpips.eval()

    def _prepare(self, frames: torch.Tensor) -> torch.Tensor:
        # LPIPS expects tensors in [-1, 1]
        return (frames * 2.0 - 1.0).to(self.device)

    def evaluate(
        self, frames: torch.Tensor, frame_indices: Sequence[int]
    ) -> list[MetricResult]:
        if len(frame_indices) < 2:
            raise ValueError("LPIPS には2枚以上のフレームが必要です。")

        prepared = self._prepare(frames)
        reference = prepared[frame_indices[0]].unsqueeze(0)
        results: list[MetricResult] = []
        distances = []

        with torch.no_grad():
            for idx in frame_indices[1:]:
                target = prepared[idx].unsqueeze(0)
                dist = self.lpips(reference, target).item()
                distances.append(dist)
                results.append(
                    MetricResult(
                        model=self.name,
                        metric="lpips",
                        value=float(dist),
                        frames_used=(frame_indices[0], idx),
                        detail=f"ref={frame_indices[0]}, cmp={idx}",
                    )
                )

        mean_dist = float(sum(distances) / len(distances))
        results.append(
            MetricResult(
                model=self.name,
                metric="lpips_mean",
                value=mean_dist,
                frames_used=tuple(frame_indices),
                detail="mean LPIPS distance vs reference frame 0",
            )
        )
        return results

