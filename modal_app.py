import base64
from logging import getLogger
from pathlib import Path

import modal

from inference import FramePackInference, InferenceError
from main import build_infer_engine_from_env

logger = getLogger(__name__)

APP_NAME = "impostor-app"
FUNCTION_NAME = "generate_impostor"
MODEL_VOLUME_NAME = "impostor-models"
OUTPUT_VOLUME_NAME = "impostor-output"
CACHE_VOLUME_NAME = "impostor-cache"
MODEL_BASE = Path("/workspace/models")

MODEL_ENV = {
    "IMPOSTOR_DIT_PATH": str(
        MODEL_BASE / "diffusion_models/FramePackI2V_HY"
    ),
    "IMPOSTOR_VAE_PATH": str(MODEL_BASE / "vae/diffusion_pytorch_model.safetensors"),
    "IMPOSTOR_TEXT_ENCODER1_PATH": str(
        MODEL_BASE / "text_encoder/model-00001-of-00004.safetensors"
    ),
    "IMPOSTOR_TEXT_ENCODER2_PATH": str(
        MODEL_BASE / "text_encoder_2/model.safetensors"
    ),
    "IMPOSTOR_IMAGE_ENCODER_PATH": str(
        MODEL_BASE / "image_encoder/model.safetensors"
    ),
    "IMPOSTOR_LORA_WEIGHT": str(
        MODEL_BASE / "impostor-models/impostor-v4-step00005000.safetensors"
    ),
    "IMPOSTOR_CACHE_DIR": "/workspace/cache",
    "IMPOSTOR_OUTPUT_DIR": "/workspace/output",
}

modal_image = modal.Image.from_dockerfile(
    Path(__file__).resolve().parent / "Dockerfile.modal",
    context_dir=Path(__file__).resolve().parent,
).env(MODEL_ENV)

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

app = modal.App(
    APP_NAME,
    image=modal_image,
    volumes={
        "/workspace/models": model_volume,
        "/workspace/output": output_volume,
        "/workspace/cache": cache_volume,
    },
)

_INFER_ENGINE: FramePackInference | None = None


def _get_engine() -> FramePackInference:
    global _INFER_ENGINE
    if _INFER_ENGINE is None:
        _INFER_ENGINE = build_infer_engine_from_env()
    return _INFER_ENGINE


@app.function(
    gpu="L40S",
    timeout=60 * 60,
    name=FUNCTION_NAME,
)
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
def generate_impostor(payload: dict):
    logger.debug(f"{payload=}")
    image_b64 = payload.get("image_base64")
    if not image_b64:
        response = {"error": "image_base64 が必要です"}
        logger.debug(f"{response=}")
        return response

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception as exc:
        response = {"error": f"image_base64 のデコードに失敗しました: {exc}"}
        logger.debug(f"{response=}")
        return response

    try:
        engine = _get_engine()
        video_path = engine.generate_to_path(
            image_bytes,
            prompt=payload.get("prompt"),
            infer_steps=payload.get("steps"),
            guidance_scale=payload.get("cfg"),
            lora_multiplier=payload.get("lora_multiplier"),
            total_frames=payload.get("total_frames"),
            latent_window_size=payload.get("latent_window_size"),
        )
    except InferenceError as exc:
        response = {"error": str(exc)}
        logger.debug(f"{response=}")
        return response
    except Exception as exc:
        logger.exception("Unexpected inference error")
        response = {"error": f"推論に失敗しました: {exc}"}
        logger.debug(f"{response=}")
        return response

    video_bytes = video_path.read_bytes()
    video_b64 = base64.b64encode(video_bytes).decode("utf-8")

    response = {
        "path": str(video_path),
        "filename": video_path.name,
        "video_base64": video_b64,
    }
    logger.debug(f"{response=}")
    return response
