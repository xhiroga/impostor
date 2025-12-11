import numpy as np

from .constants import MAX_PITCH, MIN_PITCH


def angular_distance(a: float, b: float) -> float:
    """角度の最短距離（度）。"""
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def pick_frame_index(
    yaw: float,
    pitch: float,
    available_frames: int,
    path_yaw: np.ndarray,
    path_pitch: np.ndarray,
) -> int:
    """スライダー値に最も近い経路上のフレームを返す。"""
    frame_len = len(path_yaw)
    available_frames = int(np.clip(available_frames, 2, frame_len))
    step = frame_len / available_frames
    candidate_indices = np.unique(np.round(np.arange(available_frames) * step).astype(int) % frame_len)

    best_idx = 0
    best_score = float("inf")
    for idx in candidate_indices:
        yaw_cost = angular_distance(yaw, path_yaw[idx]) / 180.0
        pitch_cost = (pitch - path_pitch[idx]) / (MAX_PITCH - MIN_PITCH)
        score = yaw_cost**2 + 1.6 * pitch_cost**2
        if score < best_score:
            best_idx = int(idx)
            best_score = score
    return best_idx


__all__ = ["pick_frame_index", "angular_distance"]
