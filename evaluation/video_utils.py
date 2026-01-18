from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch
import torchvision


class VideoLoadError(RuntimeError):
    """Raised when a video cannot be decoded."""


def load_video_frames(path: str | Path) -> tuple[torch.Tensor, float]:
    """Load video frames as a uint8 tensor of shape (T, H, W, C).

    Returns:
        frames: Tensor on CPU.
        fps: Frames per second reported by the video metadata.
    """
    video_path = Path(path)
    if not video_path.exists():
        raise VideoLoadError(f"Video not found: {video_path}")

    try:
        frames, _, info = torchvision.io.read_video(
            str(video_path), pts_unit="sec", output_format="THWC"
        )
    except Exception as exc:  # pragma: no cover - dependent on ffmpeg bindings
        raise VideoLoadError(f"Failed to load video: {video_path}") from exc

    fps: float = float(info.get("video_fps", 0.0)) or 0.0
    if fps <= 0:
        fps = 0.0  # keep zero to signal unknown fps, but continue
    return frames, fps


def resolve_frame_indices(
    total_frames: int,
    *,
    frame_indices: Sequence[int] | None,
    stride: int | None,
    max_frames: int,
) -> list[int]:
    """Select frame indices based on user preferences."""
    if total_frames == 0:
        return []

    if frame_indices:
        indices = [i for i in frame_indices if 0 <= i < total_frames]
        if not indices:
            raise ValueError("No valid frame indices after filtering.")
        return sorted(set(indices))

    if stride and stride > 0:
        return list(range(0, total_frames, stride))[:max_frames]

    step = max(total_frames // max_frames, 1)
    return list(range(0, total_frames, step))[:max_frames]


def extract_frames(
    frames: torch.Tensor,
    indices: Iterable[int],
) -> torch.Tensor:
    """Return selected frames normalized to float32 in (0, 1), shape (N, C, H, W)."""
    selected = frames[torch.tensor(list(indices), dtype=torch.long)]
    selected = selected.permute(0, 3, 1, 2)  # THWC -> TCHW
    return selected.float() / 255.0

