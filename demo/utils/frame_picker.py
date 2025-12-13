import numpy as np

from .constants import MAX_PITCH, MIN_PITCH

# 動画は 16 フレーム単位でピッチが段階的に変化する（0 → 18 → 36 → 54 → 72 度）。
# スライダー入力とこの既知パターンを近似的に対応付けるための設定値。
PITCH_DEGREES = [0, 18, 36, 54, 72]
YAW_FRAMES_PER_RING = 16  # 0–360° を 16 分割（22.5°刻み）


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
    """スライダー値に最も近いフレームインデックスを返す。

    本アプリが扱う動画は「ピッチ 5 段階 × 16 枚のヨー」で構成される前提
    （0°, 18°, 36°, 54°, 72° の順に 16 フレームずつ）なので、まずはこの
    既知パターンに合わせて指数を計算し、当てはまらない場合のみ従来の
    経路スコアリングにフォールバックする。
    """

    frame_len = len(path_yaw)

    # --- 既知パターンに基づく近似マッピング ---
    full_rings = frame_len // YAW_FRAMES_PER_RING
    pitch_levels = min(full_rings, len(PITCH_DEGREES))
    if pitch_levels >= 1:
        # ヨーは 22.5 度刻み（16 分割）で丸める
        yaw_step = 360.0 / YAW_FRAMES_PER_RING
        yaw_idx = int(np.round((yaw % 360.0) / yaw_step)) % YAW_FRAMES_PER_RING

        # 利用可能なピッチ段から最も近いものを選択
        candidate_pitches = PITCH_DEGREES[:pitch_levels]
        pitch_idx = int(np.argmin([abs(pitch - p) for p in candidate_pitches]))

        discrete_idx = pitch_idx * YAW_FRAMES_PER_RING + yaw_idx
        if discrete_idx < frame_len:
            return discrete_idx

    # --- フォールバック: 旧来の経路スコアリング ---
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
