"""
Gradio デモ: スライダー操作で 360° オービット動画を「リアルタイム」操作。

`data/sample/sample-girl.mp4` のフレームをあらかじめ読み込み、
水平／垂直スライダーで最も近いカメラ角度のフレームを即座に表示します。
レンダリング風の演出として、利用フレーム数を減らすスライダーも用意。
"""

from __future__ import annotations

import gradio as gr
from PIL import Image, ImageDraw

from utils.camera_path import generate_path
from utils.constants import FRAME_COUNT, MAX_PITCH, MIN_PITCH
from utils.frame_picker import pick_frame_index
from utils.video_loader import fallback_frames, load_video_frames

# === Video / render path settings ===


try:
    PRECOMPUTED_FRAMES, FRAME_SIZE = load_video_frames()
except Exception as exc:  # noqa: BLE001
    print(f"[warn] video読み込みに失敗: {exc}. フォールバック画像で起動します。")
    PRECOMPUTED_FRAMES, FRAME_SIZE = fallback_frames()

PATH_YAW, PATH_PITCH = generate_path(len(PRECOMPUTED_FRAMES))
DISPLAY_WIDTH, DISPLAY_HEIGHT = FRAME_SIZE


def update_view(yaw: float, pitch: float, frame_budget: int):
    """Gradio callback: map slider values to a pseudo-rendered frame."""
    idx = pick_frame_index(yaw, pitch, frame_budget, PATH_YAW, PATH_PITCH)
    frame = PRECOMPUTED_FRAMES[idx].copy()
    draw = ImageDraw.Draw(frame)
    overlay = f"Yaw {yaw:5.1f}° / Pitch {pitch:5.1f}°"
    draw.rectangle([10, 10, 240, 44], fill=(0, 0, 0))
    draw.text((18, 16), overlay, fill=(240, 240, 240))
    info = (
        f"Frame {idx+1:03d}/{len(PRECOMPUTED_FRAMES)}  "
        f"(利用: {int(frame_budget)} 枚)  "
        f"yaw {PATH_YAW[idx]:.1f}°, pitch {PATH_PITCH[idx]:.1f}°"
    )
    return frame, info


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Orbiting Character Demo") as demo:
        gr.Markdown(
            """
            ### 疑似レンダリングデモ
            スライダーで視点を動かすと、あらかじめ用意した動画フレームの中から最も近いカメラ角度を選びます。
            """
        )

        image = gr.Image(
            value=PRECOMPUTED_FRAMES[0],
            label="擬似レンダリング結果",
            type="pil",
            height=DISPLAY_HEIGHT,
            width=DISPLAY_WIDTH,
        )

        with gr.Row():
            yaw_slider = gr.Slider(
                minimum=0,
                maximum=360,
                value=0,
                step=1,
                label="水平 (Yaw) 0–360°",
                interactive=True,
            )
            pitch_slider = gr.Slider(
                minimum=MIN_PITCH,
                maximum=MAX_PITCH,
                value=10,
                step=1,
                label=f"垂直 (Pitch) {MIN_PITCH}°〜{MAX_PITCH}°",
                interactive=True,
            )

        frame_budget = gr.Slider(
            minimum=2,
            maximum=len(PRECOMPUTED_FRAMES),
            value=min(120, len(PRECOMPUTED_FRAMES)),
            step=max(1, len(PRECOMPUTED_FRAMES) // 30),
            label="利用フレーム数（少ないほどカクつき）",
            interactive=True,
        )

        info = gr.Textbox(label="ステータス", value="Ready", interactive=False)

        # Live updates while dragging yaw/pitch, change for frame budget.
        yaw_slider.input(update_view, [yaw_slider, pitch_slider, frame_budget], [image, info])
        pitch_slider.input(update_view, [yaw_slider, pitch_slider, frame_budget], [image, info])
        frame_budget.change(update_view, [yaw_slider, pitch_slider, frame_budget], [image, info])

    return demo


def main():
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)


if __name__ == "__main__":
    main()
