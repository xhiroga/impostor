from pathlib import Path

# 共通定数（描画サイズ・カメラ範囲など）
FRAME_COUNT = 360
# デフォルトの表示サイズ（動画から取得できない場合のフォールバック）
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 360
MIN_PITCH = 0
MAX_PITCH = 72
PITCH_STEP = 18

# プロジェクトルートとサンプル動画パス
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
VIDEO_PATH = DATA_DIR / "inference-sample" / "inference_sample_girl.mp4"

__all__ = [
    "FRAME_COUNT",
    "DEFAULT_WIDTH",
    "DEFAULT_HEIGHT",
    "MIN_PITCH",
    "MAX_PITCH",
    "PITCH_STEP",
    "VIDEO_PATH",
    "DATA_DIR",
    "ROOT_DIR",
]
