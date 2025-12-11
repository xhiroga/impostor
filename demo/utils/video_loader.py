import math
from typing import List, Tuple, Optional

import cv2
from PIL import Image, ImageDraw

from .constants import DEFAULT_HEIGHT, DEFAULT_WIDTH, FRAME_COUNT, VIDEO_PATH


def load_video_frames(
    target_frames: int = FRAME_COUNT,
    video_path=VIDEO_PATH,
    resize_to: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Image.Image], Tuple[int, int]]:
    """
    動画を等間隔サンプリングし、PIL.Image のリストを返す。
    読み込みに失敗した場合は例外を送出。
    """
    if not video_path.exists():
        raise FileNotFoundError(f"sample video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or target_frames
    step = max(1, math.floor(total_frames / target_frames))

    frames: List[Image.Image] = []
    target_w, target_h = resize_to if resize_to else (None, None)
    idx = 0
    while cap.isOpened():
        success, frame_bgr = cap.read()
        if not success:
            break
        if idx % step == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # キャプチャ解像度から初回に表示サイズを決定
            if target_w is None or target_h is None:
                target_h, target_w = frame_rgb.shape[:2]
            if resize_to:
                frame_rgb = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
            img = Image.fromarray(frame_rgb)
            frames.append(img)
        idx += 1
        if len(frames) >= target_frames:
            break

    cap.release()

    if not frames:
        raise RuntimeError("video read succeeded but produced 0 frames")

    assert target_w is not None and target_h is not None
    return frames, (target_w, target_h)


def fallback_frames(
    num_frames: int = 90,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> Tuple[List[Image.Image], Tuple[int, int]]:
    """テキストのみのプレースホルダフレームを生成。"""
    frames: List[Image.Image] = []
    for i in range(num_frames):
        img = Image.new("RGB", (width, height), color=(25, 35, 55))
        draw = ImageDraw.Draw(img)
        draw.text((30, 30), "sample video not found", fill=(230, 100, 100))
        draw.text((30, 70), f"frame {i+1}/{num_frames}", fill=(230, 230, 230))
        frames.append(img)
    return frames, (width, height)


__all__ = ["load_video_frames", "fallback_frames"]
