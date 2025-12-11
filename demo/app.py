"""
Gradio デモ: サンプル動画閲覧 or LoRA付き FramePack 推論の両方を操作できる UI。
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Tuple

import gradio as gr
from PIL import Image, ImageDraw
import numpy as np

from utils.camera_path import generate_path
from utils.constants import FRAME_COUNT, MAX_PITCH, MIN_PITCH, VIDEO_PATH
from utils.frame_picker import pick_frame_index
from utils.video_loader import fallback_frames, load_video_frames
from utils.inference import run_framepack_inference

# 生成結果の保存先（ユーザ指定）
DEMO_OUTPUT_DIR = Path("/workspace/imposter/demo/output")
DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# === ユーティリティ ===
def scan_sample_videos() -> list[str]:
    """サンプル／過去生成動画をスキャンしてプルダウンに表示する。"""
    paths = set()
    for pattern in [
        "/workspace/imposter/data/sample/*.mp4",
        "/workspace/imposter/output/**/*.mp4",
        f"{DEMO_OUTPUT_DIR}/**/*.mp4",
    ]:
        paths.update(glob.glob(pattern, recursive=True))
    return sorted(paths)


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
initial_frames, (PATH_YAW, PATH_PITCH) = load_frames(VIDEO_PATH if VIDEO_PATH.exists() else None)
PRECOMPUTED_FRAMES, FRAME_SIZE = initial_frames
DISPLAY_WIDTH, DISPLAY_HEIGHT = FRAME_SIZE


def update_view(yaw: float, pitch: float, frame_budget: int, frames: List[Image.Image], path_yaw, path_pitch):
    """フレーム配列と経路を受け取り表示用イメージとテキストを返す。"""
    if not frames:
        frames = [fallback_frames()[0][0]]
        path_yaw, path_pitch = generate_path(len(frames))

    idx = pick_frame_index(yaw, pitch, frame_budget, np.array(path_yaw), np.array(path_pitch))
    frame = overlay_info(frames[idx], yaw, pitch)
    info = (
        f"Frame {idx+1:03d}/{len(frames)}  "
        f"(利用: {int(frame_budget)} 枚)  "
        f"yaw {path_yaw[idx]:.1f}°, pitch {path_pitch[idx]:.1f}°"
    )
    return frame, info


def switch_video(video_path: str | None):
    """選択された動画を読み込み、状態を更新。"""
    path = Path(video_path) if video_path else None
    (frames, size), (path_yaw, path_pitch) = load_frames(path)
    frame_budget_max = len(frames)
    first_frame = overlay_info(frames[0], 0, 10)
    slider_update = gr.update(
        minimum=2,
        maximum=frame_budget_max,
        value=min(120, frame_budget_max),
        step=max(1, frame_budget_max // 30),
    )
    return (
        frames,
        path_yaw,
        path_pitch,
        size,
        slider_update,
        first_frame,
        f"読み込み完了: {path}" if path else "読み込み失敗",
    )


def run_inference(prompt, lora_path, image, steps, seconds, fps, height, width, seed, solver, negative):
    """推論ボタンクリック時の処理。"""
    if image is None:
        raise gr.Error("入力画像を指定してください")

    gr.Progress(track_tqdm=True)

    lora_path = Path(lora_path) if lora_path else None
    if lora_path is not None and not lora_path.exists():
        raise gr.Error(f"LoRA が見つかりません: {lora_path}")

    video_path = run_framepack_inference(
        prompt=prompt,
        image=image,
        lora_path=lora_path,
        infer_steps=int(steps),
        video_seconds=float(seconds),
        fps=int(fps),
        video_size=(int(height), int(width)),
        seed=int(seed) if seed else None,
        sample_solver=solver,
        negative_prompt=negative or "",
        output_dir=DEMO_OUTPUT_DIR,
    )
    (frames, size), (path_yaw, path_pitch) = load_frames(video_path)
    frame_budget_max = len(frames)
    first_frame = overlay_info(frames[0], 0, 10)
    status = f"推論完了: {video_path}"
    slider_update = gr.update(
        minimum=2,
        maximum=frame_budget_max,
        value=min(120, frame_budget_max),
        step=max(1, frame_budget_max // 30),
    )
    return (
        str(video_path),
        frames,
        path_yaw,
        path_pitch,
        size,
        slider_update,
        first_frame,
        status,
    )


def build_demo() -> gr.Blocks:
    samples = scan_sample_videos()

    with gr.Blocks(title="Imposter FramePack Demo") as demo:
        gr.Markdown("### サンプル動画 / 推論切替デモ")

        mode = gr.Radio(
            ["サンプルを動かす", "LoRAで推論する"],
            value="サンプルを動かす",
            label="モードを選択",
        )

        with gr.Row(visible=True) as sample_row:
            sample_dropdown = gr.Dropdown(samples, value=samples[0] if samples else None, label="サンプル動画")
            sample_refresh = gr.Button("再スキャン")

        with gr.Column(visible=False) as infer_col:
            prompt = gr.Textbox(label="プロンプト", value="a girl walking in the snow, cinematic")
            image_input = gr.Image(type="pil", label="入力画像 (image2video)", sources=["upload", "clipboard"])
            lora_path = gr.Textbox(label="LoRA パス（空欄なら base モデルのみ）", value="", placeholder="/workspace/models/lora/foo.safetensors")
            with gr.Row():
                steps = gr.Slider(5, 40, value=25, step=1, label="ステップ数")
                seed = gr.Textbox(label="Seed (空でランダム)", value="")
            with gr.Row():
                seconds = gr.Slider(2.0, 8.0, value=5.0, step=0.5, label="動画秒数")
                fps = gr.Slider(12, 30, value=30, step=1, label="FPS")
            with gr.Row():
                height = gr.Slider(128, 512, value=256, step=8, label="高さ")
                width = gr.Slider(128, 512, value=256, step=8, label="幅")
            solver = gr.Dropdown(["unipc", "dpm++", "vanilla"], value="unipc", label="サンプラー")
            negative = gr.Textbox(label="ネガティブプロンプト", value="")
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

        frame_budget = gr.Slider(
            minimum=2,
            maximum=len(PRECOMPUTED_FRAMES),
            value=min(120, len(PRECOMPUTED_FRAMES)),
            step=max(1, len(PRECOMPUTED_FRAMES) // 30),
            label="利用フレーム数（少ないほどカクつき）",
            interactive=True,
        )

        info = gr.Textbox(label="ステータス", value="Ready", interactive=False)
        current_video = gr.Textbox(label="現在の動画パス", value=str(VIDEO_PATH if VIDEO_PATH.exists() else ""), interactive=False)

        # 状態保持
        frames_state = gr.State(PRECOMPUTED_FRAMES)
        path_yaw_state = gr.State(PATH_YAW)
        path_pitch_state = gr.State(PATH_PITCH)
        size_state = gr.State(FRAME_SIZE)

        # イベント: モード切替で表示を切り替える
        def toggle_mode(m):
            return (
                gr.update(visible=m == "サンプルを動かす"),
                gr.update(visible=m == "LoRAで推論する"),
            )

        mode.change(toggle_mode, [mode], [sample_row, infer_col])

        # サンプル再読み込み
        sample_refresh.click(lambda: gr.update(choices=scan_sample_videos()), None, [sample_dropdown])

        # サンプル選択で動画差し替え
        sample_dropdown.change(
            switch_video,
            [sample_dropdown],
            [frames_state, path_yaw_state, path_pitch_state, size_state, frame_budget, image, info],
        )

        # 推論開始
        def show_spinner():
            return gr.update(visible=True)

        def hide_spinner():
            return gr.update(visible=False)

        infer_button.click(show_spinner, outputs=[spinner])
        infer_button.click(
            run_inference,
            [prompt, lora_path, image_input, steps, seconds, fps, height, width, seed, solver, negative],
            [current_video, frames_state, path_yaw_state, path_pitch_state, size_state, frame_budget, image, info],
        ).then(hide_spinner, outputs=[spinner])

        # Live updates for sliders
        yaw_slider.input(
            update_view,
            [yaw_slider, pitch_slider, frame_budget, frames_state, path_yaw_state, path_pitch_state],
            [image, info],
        )
        pitch_slider.input(
            update_view,
            [yaw_slider, pitch_slider, frame_budget, frames_state, path_yaw_state, path_pitch_state],
            [image, info],
        )
        frame_budget.change(
            update_view,
            [yaw_slider, pitch_slider, frame_budget, frames_state, path_yaw_state, path_pitch_state],
            [image, info],
        )

    return demo


def main():
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)


if __name__ == "__main__":
    main()
