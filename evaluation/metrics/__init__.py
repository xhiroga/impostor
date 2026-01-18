from __future__ import annotations

from typing import Iterable

from .base import EvaluationModel
from .clip_similarity import ClipSimilarityModel
from .cosine_similarity import CosineSimilarityModel
from .lpips_score import LPIPSModel
from .psnr import PSNRModel
from .ssim import SSIMModel


_REGISTRY: dict[str, type[EvaluationModel]] = {
    ClipSimilarityModel.name: ClipSimilarityModel,
    CosineSimilarityModel.name: CosineSimilarityModel,
    LPIPSModel.name: LPIPSModel,
    PSNRModel.name: PSNRModel,
    SSIMModel.name: SSIMModel,
}


def available_models() -> list[str]:
    return sorted(_REGISTRY.keys())


def build_models(requested: Iterable[str] | None) -> list[EvaluationModel]:
    if not requested:
        return [cls() for cls in _REGISTRY.values()]

    missing = [name for name in requested if name not in _REGISTRY]
    if missing:
        raise ValueError(f"Unknown models requested: {', '.join(missing)}")
    return [(_REGISTRY[name])() for name in requested]
