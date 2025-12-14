"""
Gradio デモ: サンプル動画閲覧 or LoRA付き FramePack 推論の両方を操作できる UI。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import gradio as gr
from PIL import Image, ImageDraw
import numpy as np

from utils.camera_path import generate_path
from utils.constants import FRAME_COUNT, MAX_PITCH, MIN_PITCH, VIDEO_PATH
from utils.frame_picker import pick_frame_index
from utils.video_loader import fallback_frames, load_video_frames
from utils.inference import run_framepack_inference, load_inference_params

# ディレクトリ関連の定数を一元管理
DEMO_DIR = Path(__file__).resolve().parent
DATA_ROOT = DEMO_DIR / "data"

# 生成結果の保存先（ユーザ指定のデフォルト）
DEMO_OUTPUT_DIR = DEMO_DIR / "output"
DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# デフォルトで読み込むデータセット／サンプル動画
DATASET_DIR = DATA_ROOT / "dataset"
DEFAULT_DATASET_VIDEO = DATASET_DIR / "dataset_sample_leopard.mp4"
DEFAULT_SAMPLE_VIDEO = DEFAULT_DATASET_VIDEO
# フレーム使用枚数（固定値にしたい場合ここを変更）
FRAME_BUDGET_DEFAULT = 120

# 推論パラメータを config から読み込む
INFER_CFG = load_inference_params()


# === ユーティリティ ===
def _scan_videos(dir_path: Path) -> list[str]:
    """指定ディレクトリ配下の mp4 を再帰で列挙。"""
    return sorted([str(p) for p in dir_path.glob("**/*.mp4")])


def scan_dataset_videos() -> list[str]:
    return _scan_videos(DATASET_DIR)


def scan_sample_videos() -> list[str]:
    if DEFAULT_SAMPLE_VIDEO.exists():
        return [str(DEFAULT_SAMPLE_VIDEO)]
    return []


def scan_output_videos() -> list[str]:
    return _scan_videos(DEMO_OUTPUT_DIR)


def load_frames(video_path: Path | None):
    """動画を読み込み、フレーム列とサイズ、経路を返す。失敗時はフォールバック。"""
    if video_path is None:
        return fallback_frames(), generate_path(FRAME_COUNT)
    try:
        frames, size = load_video_frames(video_path=video_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] video読み込みに失敗: {exc}. フォールバック画像で起動します。")
        frames, size = fallback_frames()
    path_yaw, path_pitch = generate_path(len(frames))
    return (frames, size), (path_yaw, path_pitch)


def overlay_info(frame: Image.Image, yaw: float, pitch: float) -> Image.Image:
    frame = frame.copy()
    draw = ImageDraw.Draw(frame)
    overlay = f"Yaw {yaw:5.1f}° / Pitch {pitch:5.1f}°"
    # 右上固定表示
    padding = 10
    # textbbox でサイズ取得
    bbox = draw.textbbox((0, 0), overlay)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    rect_w = text_w + 12
    rect_h = text_h + 8
    x1 = frame.width - padding
    x0 = x1 - rect_w
    y0 = padding
    y1 = y0 + rect_h
    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
    draw.text((x0 + 6, y0 + 4), overlay, fill=(240, 240, 240))
    return frame


# 初期サンプルロード
if DEFAULT_DATASET_VIDEO.exists():
    initial_video = DEFAULT_DATASET_VIDEO
elif DATASET_DIR.exists() and list(DATASET_DIR.glob("**/*.mp4")):
    initial_video = sorted(DATASET_DIR.glob("**/*.mp4"))[0]
elif DEFAULT_SAMPLE_VIDEO.exists():
    initial_video = DEFAULT_SAMPLE_VIDEO
else:
    initial_video = VIDEO_PATH if VIDEO_PATH.exists() else None

initial_frames, (PATH_YAW, PATH_PITCH) = load_frames(initial_video)
PRECOMPUTED_FRAMES, FRAME_SIZE = initial_frames
DISPLAY_WIDTH, DISPLAY_HEIGHT = FRAME_SIZE


def update_view(yaw: float, pitch: float, frame_budget: int, frames: List[Image.Image], path_yaw, path_pitch):
    """フレーム配列と経路を受け取り表示用イメージを返す。"""
    if not frames:
        frames = [fallback_frames()[0][0]]
        path_yaw, path_pitch = generate_path(len(frames))

    idx = pick_frame_index(yaw, pitch, frame_budget, np.array(path_yaw), np.array(path_pitch))
    frame = overlay_info(frames[idx], yaw, pitch)
    return frame


def switch_video(video_path: str | None):
    """選択された動画を読み込み、状態を更新。"""
    path = Path(video_path) if video_path else None
    (frames, size), (path_yaw, path_pitch) = load_frames(path)
    frame_budget_max = len(frames)
    frame_budget_val = min(FRAME_BUDGET_DEFAULT, frame_budget_max)
    first_frame = overlay_info(frames[0], 0, 10)
    return (
        frames,
        path_yaw,
        path_pitch,
        size,
        frame_budget_val,
        first_frame,
    )


def _resolve_lora_path() -> Path | None:
    """モジュール先頭で指定した LoRA パスを検証して Path を返す。"""
    lora_path_cfg = INFER_CFG.get("lora_weight")
    if isinstance(lora_path_cfg, (list, tuple)):
        lora_path_cfg = lora_path_cfg[0] if lora_path_cfg else None
    if not lora_path_cfg:
        return None
    lp = Path(lora_path_cfg)
    if not lp.exists():
        raise gr.Error(f"LoRA が見つかりません: {lp}")
    return lp


def run_inference(image):
    """推論ボタンクリック時の処理。"""
    if image is None:
        raise gr.Error("入力画像を指定してください")

    gr.Progress(track_tqdm=True)

    lora_path = _resolve_lora_path()
    # config 側で上書きされた保存先があれば使う
    output_dir = INFER_CFG.get("save_path") or DEMO_OUTPUT_DIR

    video_path = run_framepack_inference(
        prompt=INFER_CFG["prompt"],
        image=image,
        output_dir=Path(output_dir),
    )
    (frames, size), (path_yaw, path_pitch) = load_frames(video_path)
    frame_budget_max = len(frames)
    frame_budget_val = min(FRAME_BUDGET_DEFAULT, frame_budget_max)
    first_frame = overlay_info(frames[0], 0, 10)
    return (
        frames,
        path_yaw,
        path_pitch,
        size,
        frame_budget_val,
        first_frame,
    )


def build_demo() -> gr.Blocks:
    dataset_videos = scan_dataset_videos()
    sample_videos = scan_sample_videos()
    output_videos = scan_output_videos()
    default_dataset_value = (
        str(DEFAULT_DATASET_VIDEO)
        if DEFAULT_DATASET_VIDEO.exists()
        else (dataset_videos[0] if dataset_videos else None)
    )
    default_sample_value = (
        str(DEFAULT_SAMPLE_VIDEO)
        if DEFAULT_SAMPLE_VIDEO.exists()
        else (sample_videos[0] if sample_videos else None)
    )
    default_output_value = output_videos[0] if output_videos else None

    with gr.Blocks(title="Imposter Demo") as demo:
        gr.Markdown("### Imposter Demo")

        mode = gr.Radio(
            ["サンプルを動かす", "推論結果を動かす", "推論して動かす"],
            value="サンプルを動かす",
            label="モードを選択",
        )

        with gr.Row(visible=True) as dataset_row:
            dataset_dropdown = gr.Dropdown(
                dataset_videos,
                value=default_dataset_value,
                label="サンプル動画",
                scale=4,
            )
            dataset_refresh = gr.Button("リロード", scale=0, min_width=96)

        with gr.Row(visible=False) as sample_row:
            sample_dropdown = gr.Dropdown(
                sample_videos,
                value=default_sample_value,
                label="推論結果サンプル動画",
                scale=4,
            )
            # ボタン幅を小さく保つため scale=0 と min_width を指定
            sample_refresh = gr.Button("リロード", scale=0, min_width=96)

        with gr.Column(visible=False) as infer_col:
            image_input = gr.Image(type="pil", label="入力画像 (image2video)", sources=["upload", "clipboard"])
            output_dropdown = gr.Dropdown(
                output_videos,
                value=default_output_value,
                label="推論結果動画",
                scale=4,
            )
            output_refresh = gr.Button("リロード", scale=0, min_width=96)
            infer_button = gr.Button("推論スタート", variant="primary")
            spinner = gr.HTML(
                """<div style="text-align:center;">
                <style>
                .lds-ring {display:inline-block; position: relative; width: 64px; height: 64px;}
                .lds-ring div {box-sizing: border-box; display: block; position: absolute; width: 51px; height: 51px; margin: 6px; border: 6px solid #4fa3ff; border-radius: 50%; animation: lds-ring 1.2s cubic-bezier(0.5, 0, 0.5, 1) infinite; border-color: #4fa3ff transparent transparent transparent;}
                .lds-ring div:nth-child(1) {animation-delay: -0.45s;}
                .lds-ring div:nth-child(2) {animation-delay: -0.3s;}
                .lds-ring div:nth-child(3) {animation-delay: -0.15s;}
                @keyframes lds-ring {0% {transform: rotate(0deg);} 100% {transform: rotate(360deg);}}
                </style>
                <div class="lds-ring"><div></div><div></div><div></div><div></div></div>
                <p>推論中...</p>
                </div>""",
                visible=False,
            )

        image = gr.Image(
            value=PRECOMPUTED_FRAMES[0],
            label="擬似レンダリング結果",
            type="pil",
            height=DISPLAY_HEIGHT,
            width=DISPLAY_WIDTH,
        )

        with gr.Row():
            yaw_slider = gr.Slider(0, 360, value=0, step=1, label="水平 (Yaw) 0–360°", interactive=True)
            pitch_slider = gr.Slider(MIN_PITCH, MAX_PITCH, value=10, step=1, label=f"垂直 (Pitch) {MIN_PITCH}°〜{MAX_PITCH}°", interactive=True)

        # 状態保持
        frames_state = gr.State(PRECOMPUTED_FRAMES)
        path_yaw_state = gr.State(PATH_YAW)
        path_pitch_state = gr.State(PATH_PITCH)
        size_state = gr.State(FRAME_SIZE)
        frame_budget_state = gr.State(min(FRAME_BUDGET_DEFAULT, len(PRECOMPUTED_FRAMES)))

        # イベント: モード切替で表示を切り替える
        def toggle_mode(m):
            return (
                gr.update(visible=m == "サンプルを動かす"),
                gr.update(visible=m == "推論結果を動かす"),
                gr.update(visible=m == "推論して動かす"),
            )

        mode.change(toggle_mode, [mode], [dataset_row, sample_row, infer_col])

        # モード変更時に現在選択中の動画を読み直す
        def load_by_mode(m, d, s, o):
            if m == "サンプルを動かす":
                return switch_video(d)
            if m == "推論結果を動かす":
                return switch_video(s)
            return switch_video(o)

        mode.change(
            load_by_mode,
            [mode, dataset_dropdown, sample_dropdown, output_dropdown],
            [frames_state, path_yaw_state, path_pitch_state, size_state, frame_budget_state, image],
        )

        # サンプル／データセット再読み込み
        dataset_refresh.click(lambda: gr.update(choices=scan_dataset_videos()), None, [dataset_dropdown])
        sample_refresh.click(lambda: gr.update(choices=scan_sample_videos()), None, [sample_dropdown])
        output_refresh.click(lambda: gr.update(choices=scan_output_videos()), None, [output_dropdown])

        # ドロップダウン選択で動画差し替え
        dataset_dropdown.change(
            switch_video,
            [dataset_dropdown],
            [frames_state, path_yaw_state, path_pitch_state, size_state, frame_budget_state, image],
        )
        sample_dropdown.change(
            switch_video,
            [sample_dropdown],
            [frames_state, path_yaw_state, path_pitch_state, size_state, frame_budget_state, image],
        )
        output_dropdown.change(
            switch_video,
            [output_dropdown],
            [frames_state, path_yaw_state, path_pitch_state, size_state, frame_budget_state, image],
        )

        # 推論開始
        def show_spinner():
            return gr.update(visible=True)

        def hide_spinner():
            return gr.update(visible=False)

        infer_button.click(show_spinner, outputs=[spinner])
        infer_button.click(
            run_inference,
            [image_input],
            [frames_state, path_yaw_state, path_pitch_state, size_state, frame_budget_state, image],
        ).then(hide_spinner, outputs=[spinner])

        # Live updates for sliders
        yaw_slider.change(
            update_view,
            [yaw_slider, pitch_slider, frame_budget_state, frames_state, path_yaw_state, path_pitch_state],
            [image],
        )
        pitch_slider.change(
            update_view,
            [yaw_slider, pitch_slider, frame_budget_state, frames_state, path_yaw_state, path_pitch_state],
            [image],
        )

    return demo


def main():
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)


if __name__ == "__main__":
    main()
