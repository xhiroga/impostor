import numpy as np

from .constants import MAX_PITCH, MIN_PITCH


def ease_in_out(t: np.ndarray) -> np.ndarray:
    """Smoothstep ライクな補間でカメラの上昇をなめらかにする。"""
    return t * t * (3 - 2 * t)


def generate_path(frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    """0..360° を一周しながらピッチを徐々に上げる経路を生成。"""
    t = np.linspace(0.0, 1.0, frame_count, endpoint=False)
    yaw = (t * 360.0) % 360.0
    pitch = MIN_PITCH + (MAX_PITCH - MIN_PITCH) * ease_in_out(t)
    return yaw, pitch


__all__ = ["generate_path", "ease_in_out"]
