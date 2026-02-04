import asyncio
import base64
import json
import logging
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import GenerationSettings, ModelPaths

FramePackInference = Any  # type: ignore


class InferenceError(Exception):
    pass


load_dotenv()


logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = APP_DIR / "sample"
OUTPUT_DIR = APP_DIR / "output"
TMP_DIR = APP_DIR / "tmp"
FRONTEND_BUILD_DIR = APP_DIR / "frontend" / "build"

INFER_ENGINE: FramePackInference | None = None
INFER_ENGINE_ERROR: str | None = None

MODAL_APP_NAME = "impostor-app"
MODAL_FUNCTION_NAME = "generate_impostor"
MODAL_TIMEOUT_SECONDS = 600.0


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


def _execution_mode() -> str:
    value = os.getenv("IMPOSTOR_EXECUTION_MODE", "local").lower()
    return value


def _value(key: str, *env_names: str, required: bool = False) -> str:
    for env in env_names:
        raw = os.getenv(env, "").strip()
        if raw:
            return raw
    if required:
        fallbacks = " or ".join(env_names) or key
        raise InferenceError(f"{fallbacks} must be configured")
    return ""


def _slug(value: str) -> str:
    cleaned = value.strip().lower()
    return "".join("-" if ch == "_" else ch for ch in cleaned)


def _modal_workspace() -> str:
    return _value(
        "modal_workspace",
        "IMPOSTOR_MODAL_WORKSPACE",
        "MODAL_WORKSPACE",
        required=True,
    )


def _modal_web_url() -> str:
    override = _value(
        "modal_web_url",
        "IMPOSTOR_MODAL_WEB_URL",
        "IMPOSTOR_MODAL_ENDPOINT_URL",
    )
    if override:
        return override
    app_name = (
        os.getenv("IMPOSTOR_MODAL_APP_NAME", MODAL_APP_NAME).strip() or MODAL_APP_NAME
    )
    function_name = (
        os.getenv("IMPOSTOR_MODAL_FUNCTION_NAME", MODAL_FUNCTION_NAME).strip()
        or MODAL_FUNCTION_NAME
    )
    return f"https://{_modal_workspace()}--{_slug(app_name)}-{_slug(function_name)}.modal.run"


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise InferenceError(f"環境変数 {key} が未設定です")
    return value


def _resolve_path(value: str) -> str:
    path = Path(value)
    return str(path.expanduser().resolve())


def build_infer_engine_from_env() -> FramePackInference:
    global InferenceError
    from inference import (  # local-only import to avoid heavy deps in modal execution
        FramePackInference as _FramePackInference,
    )
    from inference import (
        InferenceError as _InferenceError,
    )

    InferenceError = _InferenceError
    model_paths = ModelPaths(
        dit=_resolve_path(_require_env("IMPOSTOR_DIT_PATH")),
        vae=_resolve_path(_require_env("IMPOSTOR_VAE_PATH")),
        text_encoder1=_resolve_path(_require_env("IMPOSTOR_TEXT_ENCODER1_PATH")),
        text_encoder2=_resolve_path(_require_env("IMPOSTOR_TEXT_ENCODER2_PATH")),
        image_encoder=_resolve_path(_require_env("IMPOSTOR_IMAGE_ENCODER_PATH")),
        lora_weight=[_resolve_path(weight)]
        if (weight := os.getenv("IMPOSTOR_LORA_WEIGHT"))
        else None,
        lora_multiplier=[float(os.getenv("IMPOSTOR_LORA_MULTIPLIER", "1.0"))],
    )
    settings = GenerationSettings(
        prompt=os.getenv(
            "IMPOSTOR_PROMPT",
            "360-degree orbit around the subject, camera rising in a spiral.",
        ),
        video_sections=int(os.getenv("IMPOSTOR_VIDEO_SECTIONS", "4")),
        fps=int(os.getenv("IMPOSTOR_FPS", "30")),
        infer_steps=int(os.getenv("IMPOSTOR_INFER_STEPS", "15")),
        latent_window_size=int(os.getenv("IMPOSTOR_LATENT_WINDOW", "5")),
        cache_dir=Path(_resolve_path(os.getenv("IMPOSTOR_CACHE_DIR", "cache"))),
        output_dir=Path(_resolve_path(os.getenv("IMPOSTOR_OUTPUT_DIR", "output"))),
        bucket_resolution=int(os.getenv("IMPOSTOR_BUCKET_RES", "640")),
    )
    return _FramePackInference(model_paths=model_paths, settings=settings)


def _get_infer_engine() -> FramePackInference | None:
    if _execution_mode() != "local":
        return None
    global INFER_ENGINE
    global INFER_ENGINE_ERROR
    if INFER_ENGINE is not None:
        return INFER_ENGINE
    if INFER_ENGINE_ERROR is not None:
        return None
    try:
        INFER_ENGINE = build_infer_engine_from_env()
    except Exception as exc:
        INFER_ENGINE_ERROR = str(exc)
        logger.exception("Failed to initialize inference engine")
        return None
    return INFER_ENGINE


def _run_modal_inference(
    image_bytes: bytes,
    *,
    prompt: str | None,
    steps: int | None,
    cfg: float | None,
    lora_multiplier: float | None,
    total_frames: int | None,
    latent_window_size: int | None,
) -> Path:
    logger.debug(
        f"prompt={prompt}, steps={steps}, cfg={cfg}, lora_multiplier={lora_multiplier}, "
        f"total_frames={total_frames}, latent_window_size={latent_window_size}"
    )
    payload = {
        "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
        "prompt": prompt,
        "steps": steps,
        "cfg": cfg,
        "lora_multiplier": lora_multiplier,
        "total_frames": total_frames,
        "latent_window_size": latent_window_size,
    }
    web_url = _modal_web_url()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Impostor/0.1",
    }
    modal_key = _value("modal_key", "IMPOSTOR_MODAL_KEY", "MODAL_PROXY_KEY")
    modal_secret = _value("modal_secret", "IMPOSTOR_MODAL_SECRET", "MODAL_PROXY_SECRET")
    if modal_key and modal_secret:
        headers["Modal-Key"] = modal_key
        headers["Modal-Secret"] = modal_secret

    request = urllib_request.Request(
        web_url,
        data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        ),
        headers=headers,
        method="POST",
    )
    try:
        timeout = float(
            os.getenv("IMPOSTOR_MODAL_TIMEOUT_SECONDS", str(MODAL_TIMEOUT_SECONDS))
        )
        with urllib_request.urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            response_body = response.read()
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        message = detail or exc.reason or f"HTTP {exc.code}"
        raise InferenceError(f"Modal endpoint rejected the request: {message}") from exc
    except urllib_error.URLError as exc:
        raise InferenceError(f"Modal endpoint is unreachable: {exc.reason}") from exc
    except (socket.timeout, TimeoutError) as exc:
        raise InferenceError("Modal endpoint timed out") from exc

    try:
        decoded = response_body.decode(charset)
        response = json.loads(decoded)
    except UnicodeDecodeError as exc:
        raise InferenceError("Modal response was not valid text") from exc
    except json.JSONDecodeError as exc:
        raise InferenceError("Modal response was not valid JSON") from exc
    logger.debug(f"{response=}")
    if "error" in response:
        raise InferenceError(response["error"])
    video_b64 = response.get("video_base64")
    filename = response.get("filename", "modal_output.mp4")
    if not video_b64:
        raise InferenceError("Modal から動画データを取得できませんでした")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    output_path.write_bytes(base64.b64decode(video_b64))
    return output_path


app = FastAPI()
for mount_path, directory in (
    ("/samples", SAMPLE_DIR),
    ("/output", OUTPUT_DIR),
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
    if _execution_mode() == "local":
        _get_infer_engine()
    return {
        "infer_ready": True
        if _execution_mode() == "modal"
        else INFER_ENGINE is not None,
        "infer_error": None if _execution_mode() == "modal" else INFER_ENGINE_ERROR,
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
    infer_engine = _get_infer_engine() if _execution_mode() == "local" else None
    if infer_engine is None and _execution_mode() == "local":
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
        if _execution_mode() == "modal":
            video_path: Path = await asyncio.to_thread(
                _run_modal_inference,
                payload,
                prompt=prompt,
                steps=steps,
                cfg=cfg,
                lora_multiplier=lora_multiplier,
                total_frames=total_frames,
                latent_window_size=latent_window_size,
            )
        else:
            video_path = await asyncio.to_thread(
                infer_engine.generate_to_path,
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
        return HTMLResponse(
            "<h1>Frontend build not found. Run npm run build in frontend/.</h1>"
        )
