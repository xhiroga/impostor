from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import torch
import torchvision.transforms as T

from .base import EvaluationModel, MetricResult


class ClipNotAvailable(RuntimeError):
    """Raised when open_clip_torch is not installed."""


def _to_pil(frame: torch.Tensor):
    to_pil = T.ToPILImage()
    return to_pil(frame.clamp(0, 1))


@lru_cache(maxsize=1)
def _load_clip(model_name: str, pretrained: str):
    try:
        import open_clip
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ClipNotAvailable(
            "open_clip_torch is required for CLIP similarity. Install via "
            "`pip install open_clip_torch`."
        ) from exc

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


class ClipSimilarityModel(EvaluationModel):
    """CLIP image-embedding cosine similarity."""

    name = "clip"

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model, self.preprocess, _ = _load_clip(model_name, pretrained)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        images = [self.preprocess(_to_pil(frame)) for frame in frames]
        batch = torch.stack(images).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu()

    def evaluate(
        self, frames: torch.Tensor, frame_indices: Sequence[int]
    ) -> list[MetricResult]:
        if len(frame_indices) < 2:
            raise ValueError("CLIP similarityには2枚以上のフレームが必要です。")

        embeddings = self._encode_frames(frames)
        reference = embeddings[frame_indices[0]].unsqueeze(0)
        comparisons = embeddings[frame_indices[1:]]

        scores = torch.sum(reference * comparisons, dim=1)

        results: list[MetricResult] = []
        for idx, score in zip(frame_indices[1:], scores):
            results.append(
                MetricResult(
                    model=self.name,
                    metric="clip_cosine",
                    value=float(score.item()),
                    frames_used=(frame_indices[0], idx),
                    detail=f"ref={frame_indices[0]}, cmp={idx}",
                )
            )

        mean_score = float(scores.mean().item())
        results.append(
            MetricResult(
                model=self.name,
                metric="clip_cosine_mean",
                value=mean_score,
                frames_used=tuple(frame_indices),
                detail="mean of CLIP cosine similarity against reference frame 0",
            )
        )
        return results

