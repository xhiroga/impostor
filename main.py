import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from inference import FramePackInference, InferenceError

load_dotenv()


logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = APP_DIR / "sample"
OUTPUT_DIR = APP_DIR / "output"
ASSET_DIR = APP_DIR / "assets"
TMP_DIR = APP_DIR / "tmp"
FRONTEND_BUILD_DIR = APP_DIR / "frontend" / "build"

INFER_ENGINE: FramePackInference | None = None
INFER_ENGINE_ERROR: str | None = None
try:
    INFER_ENGINE = FramePackInference.from_env()
except Exception as exc:
    INFER_ENGINE_ERROR = str(exc)
    logger.exception("Failed to initialize inference engine")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_SOURCES = (
    ("samples", SAMPLE_DIR),
    ("output", OUTPUT_DIR),
)


def _list_videos() -> list[dict[str, str]]:
    videos: list[dict[str, str]] = []
    for mount_name, directory in VIDEO_SOURCES:
        if not directory.exists():
            continue
        for video in sorted(directory.glob("*.mp4")):
            videos.append(
                {
                    "value": f"{mount_name}/{video.name}",
                    "label": f"{mount_name} · {video.stem}",
                }
            )
    if not videos:
        raise HTTPException(status_code=500, detail="動画ファイルが見つかりません")
    return videos


app = FastAPI()
for mount_path, directory in (
    ("/samples", SAMPLE_DIR),
    ("/output", OUTPUT_DIR),
    ("/assets", ASSET_DIR),
):
    if directory.exists():
        app.mount(
            mount_path, StaticFiles(directory=directory), name=mount_path.strip("/")
        )

@app.get("/api/videos")
async def list_videos():
    return _list_videos()


@app.get("/api/status")
async def app_status():
    return {
        "infer_ready": INFER_ENGINE is not None,
        "infer_error": INFER_ENGINE_ERROR,
    }


@app.post("/api/infer")
async def run_inference(
    image: Annotated[UploadFile, File(...)],
    steps: Annotated[int | None, Form()] = None,
    cfg: Annotated[float | None, Form()] = None,
    lora_multiplier: Annotated[float | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    total_frames: Annotated[int | None, Form()] = None,
    latent_window_size: Annotated[int | None, Form()] = None,
):
    if INFER_ENGINE is None:
        raise HTTPException(
            status_code=503,
            detail=INFER_ENGINE_ERROR or "推論エンジンが初期化されていません",
        )
    if not image.filename:
        raise HTTPException(status_code=400, detail="画像ファイルを指定してください")

    payload = await image.read()
    if not payload:
        raise HTTPException(
            status_code=400, detail="画像ファイルの読み込みに失敗しました"
        )

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.filename).suffix or ".png"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_path = TMP_DIR / f"upload_{timestamp}{suffix}"
    upload_path.write_bytes(payload)

    try:
        video_path: Path = await asyncio.to_thread(
            INFER_ENGINE.generate_to_path,
            payload,
            prompt=prompt,
            infer_steps=steps,
            guidance_scale=cfg,
            lora_multiplier=lora_multiplier,
            total_frames=total_frames,
            latent_window_size=latent_window_size,
        )
    except InferenceError as exc:
        logger.warning("FramePack inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during inference")
        raise HTTPException(
            status_code=500, detail=f"推論に失敗しました: {exc}"
        ) from exc

    if not video_path.is_relative_to(OUTPUT_DIR):
        raise HTTPException(status_code=500, detail="推論結果の保存先が不正です")
    relative_value = f"output/{video_path.relative_to(OUTPUT_DIR)}"
    label = f"output · {video_path.stem}"

    return {
        "message": f"推論完了: {video_path.name} を output に追加しました。",
        "video": {"value": relative_value, "label": label},
        "upload_filename": upload_path.name,
    }


if FRONTEND_BUILD_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=FRONTEND_BUILD_DIR, html=True),
        name="frontend",
    )
else:

    @app.get("/", response_class=HTMLResponse)
    async def frontend_placeholder():
        return HTMLResponse("<h1>Frontend build not found. Run npm run build in frontend/.</h1>")
