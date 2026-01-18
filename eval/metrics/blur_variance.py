from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from .base import EvaluationModel, MetricResult


class BlurVarianceModel(EvaluationModel):
    """Focus/blur estimator using variance of Laplacian."""

    name = "blur"

    def __init__(self) -> None:
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        self.registered_kernel = kernel.view(1, 1, 3, 3)
        self.rgb_to_gray = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(
            1, 3, 1, 1
        )

    def evaluate(
        self, frames: torch.Tensor, frame_indices: Sequence[int]
    ) -> list[MetricResult]:
        if not frame_indices:
            raise ValueError("At least one frame is required for blur analysis.")

        device = frames.device
        kernel = self.registered_kernel.to(device)
        rgb_weights = self.rgb_to_gray.to(device)

        gray = (frames * rgb_weights).sum(dim=1, keepdim=True)
        laplacian = F.conv2d(gray, kernel, padding=1)
        variances = laplacian.flatten(1).var(dim=1)

        results: list[MetricResult] = []
        for idx in frame_indices:
            results.append(
                MetricResult(
                    model=self.name,
                    metric="laplacian_variance",
                    value=float(variances[idx].item()),
                    frames_used=(idx,),
                )
            )

        mean_var = float(variances[frame_indices].mean().item())
        results.append(
            MetricResult(
                model=self.name,
                metric="laplacian_variance_mean",
                value=mean_var,
                frames_used=tuple(frame_indices),
                detail="higher means sharper on average",
            )
        )
        return results

