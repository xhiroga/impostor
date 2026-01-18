from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from .base import EvaluationModel, MetricResult


class CosineSimilarityModel(EvaluationModel):
    name = "cosine"

    def evaluate(
        self, frames: torch.Tensor, frame_indices: Sequence[int]
    ) -> list[MetricResult]:
        if len(frame_indices) < 2:
            raise ValueError("Cosine similarity needs at least two frames.")

        flat = frames.reshape(frames.shape[0], -1)
        reference = flat[frame_indices[0]].unsqueeze(0)
        comparisons = flat[frame_indices[1:]]

        scores = F.cosine_similarity(comparisons, reference.expand_as(comparisons), dim=1)

        results: list[MetricResult] = []
        for idx, score in zip(frame_indices[1:], scores):
            results.append(
                MetricResult(
                    model=self.name,
                    metric="cosine_pair",
                    value=float(score.item()),
                    frames_used=(frame_indices[0], idx),
                    detail=f"ref={frame_indices[0]}, cmp={idx}",
                )
            )

        mean_score = float(scores.mean().item())
        results.append(
            MetricResult(
                model=self.name,
                metric="cosine_mean",
                value=mean_score,
                frames_used=tuple(frame_indices),
                detail="mean of cosine similarity against reference frame 0",
            )
        )
        return results

