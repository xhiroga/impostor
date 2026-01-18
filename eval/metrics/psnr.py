from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from .base import EvaluationModel, MetricResult


class PSNRModel(EvaluationModel):
    name = "psnr"

    def evaluate(
        self, frames: torch.Tensor, frame_indices: Sequence[int]
    ) -> list[MetricResult]:
        if len(frame_indices) < 2:
            raise ValueError("PSNR evaluation needs at least two frames.")

        reference = frames[frame_indices[0]]
        results: list[MetricResult] = []
        psnr_values: list[float] = []

        for idx in frame_indices[1:]:
            target = frames[idx]
            mse = F.mse_loss(target, reference).item()
            psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8)).item()
            psnr_values.append(psnr)
            results.append(
                MetricResult(
                    model=self.name,
                    metric="psnr_pair",
                    value=psnr,
                    frames_used=(frame_indices[0], idx),
                    detail=f"ref={frame_indices[0]}, cmp={idx}",
                )
            )

        mean_psnr = sum(psnr_values) / len(psnr_values)
        results.append(
            MetricResult(
                model=self.name,
                metric="psnr_mean",
                value=mean_psnr,
                frames_used=tuple(frame_indices),
                detail="mean of all pairwise scores against reference frame 0",
            )
        )
        return results

