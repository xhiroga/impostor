from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ModelPaths:
    dit: str
    vae: str
    text_encoder1: str
    text_encoder2: str
    image_encoder: str
    lora_weight: list[str] | None = None
    lora_multiplier: list[float] | None = None


@dataclass(slots=True)
class GenerationSettings:
    prompt: str
    video_sections: int
    fps: int
    infer_steps: int
    latent_window_size: int
    cache_dir: Path
    output_dir: Path
    bucket_resolution: int = 640
