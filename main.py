from __future__ import annotations

from pathlib import Path
import asyncio
import json
import os
from datetime import datetime
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

from inference import FramePackInference, InferenceError

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = APP_DIR / "sample"
OUTPUT_DIR = APP_DIR / "output"
ASSET_DIR = APP_DIR / "assets"
TMP_DIR = APP_DIR / "tmp"

INFER_ENGINE: FramePackInference | None = None
INFER_ENGINE_ERROR: str | None = None
try:
    INFER_ENGINE = FramePackInference.from_env()
except Exception as exc:  # noqa: BLE001
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


def _render_select_options(selected: str | None = None) -> tuple[str, str]:
    entries = _list_videos()
    values = {entry["value"] for entry in entries}
    resolved = selected if selected in values else entries[0]["value"]
    options_html = "".join(
        f"<option value=\"{entry['value']}\" {'selected' if entry['value'] == resolved else ''}>{entry['label']}</option>"
        for entry in entries
    )
    return resolved, options_html


app = FastAPI()
for mount_path, directory in (("/samples", SAMPLE_DIR), ("/output", OUTPUT_DIR), ("/assets", ASSET_DIR)):
    if directory.exists():
        app.mount(mount_path, StaticFiles(directory=directory), name=mount_path.strip("/"))

templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    _, select_options = _render_select_options()
    return templates.TemplateResponse(
        "index.html.j2",
        {
            "request": request,
            "select_options": select_options,
            "infer_disabled": INFER_ENGINE is None,
            "infer_error": INFER_ENGINE_ERROR,
        },
    )

@app.get("/videos", response_class=HTMLResponse)
async def render_video_options(selected: str | None = Query(default=None)):
    _, options_html = _render_select_options(selected)
    return HTMLResponse(options_html)


@app.post("/infer", response_class=HTMLResponse)
async def run_inference(image: UploadFile = File(...)):
    if INFER_ENGINE is None:
        raise HTTPException(
            status_code=503,
            detail=INFER_ENGINE_ERROR or "推論エンジンが初期化されていません",
        )
    if not image.filename:
        raise HTTPException(status_code=400, detail="画像ファイルを指定してください")

    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="画像ファイルの読み込みに失敗しました")

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.filename).suffix or ".png"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_path = TMP_DIR / f"upload_{timestamp}{suffix}"
    upload_path.write_bytes(payload)

    try:
        video_path: Path = await asyncio.to_thread(INFER_ENGINE.generate_to_path, payload)
    except InferenceError as exc:
        logger.warning("FramePack inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during inference")
        raise HTTPException(status_code=500, detail=f"推論に失敗しました: {exc}") from exc

    if not video_path.is_relative_to(OUTPUT_DIR):
        raise HTTPException(status_code=500, detail="推論結果の保存先が不正です")
    relative_value = f"output/{video_path.relative_to(OUTPUT_DIR)}"

    status_html = (
        f"<p>推論完了: {video_path.name} を output に追加しました。</p>"
        f"<p style=\"font-size:0.9rem;color:#aaa;\">入力画像は {upload_path.name} として保存されています。</p>"
    )
    script = f"<script>window.__latestVideo = {json.dumps(relative_value)};</script>"
    return HTMLResponse(status_html + script)
