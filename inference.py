import io
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

import torch
import torchvision
from PIL import Image


class InferenceError(Exception):
    """Raised when FramePack inference fails."""


@dataclass(slots=True)
class ModelPaths:
    dit: str
    vae: str
    text_encoder1: str
    text_encoder2: str
    image_encoder: str
    lora_weight: list[str] | None = None
    lora_multiplier: list[float] | None = None


@dataclass(slots=True)
class GenerationSettings:
    prompt: str
    video_sections: int
    fps: int
    infer_steps: int
    latent_window_size: int
    cache_dir: Path
    output_dir: Path
    bucket_resolution: int = 640


BUCKET_OPTIONS: dict[int, list[tuple[int, int]]] = {
    640: [
        (416, 960),
        (448, 864),
        (480, 832),
        (512, 768),
        (544, 704),
        (576, 672),
        (608, 640),
        (640, 608),
        (672, 576),
        (704, 544),
        (768, 512),
        (832, 480),
        (864, 448),
        (960, 416),
    ]
}


def _nearest_bucket(height: int, width: int, resolution: int) -> tuple[int, int]:
    options = BUCKET_OPTIONS.get(resolution)
    if not options:
        raise InferenceError(
            f"Unsupported bucket resolution {resolution}. Supported: {sorted(BUCKET_OPTIONS.keys())}"
        )
    best = options[0]
    best_metric = float("inf")
    for h, w in options:
        metric = abs(height * w - width * h)
        if metric <= best_metric:
            best_metric = metric
            best = (h, w)
    return best


def resize_image_to_bucket(image_bytes: bytes, resolution: int) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
        bucket_h, bucket_w = _nearest_bucket(height, width, resolution)
        resized = image.resize((bucket_w, bucket_h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format=image.format or "PNG")
        return buf.getvalue()


def save_video_tensor(video: torch.Tensor, path: Path, fps: int) -> Path:
    video = video.unsqueeze(0)  # B=1
    frames = video.permute(2, 0, 1, 3, 4)  # T, B, C, H, W
    rendered: list[torch.Tensor] = []
    for frame in frames:
        grid = torchvision.utils.make_grid(frame, nrow=1)
        grid = grid.permute(1, 2, 0)
        grid = torch.clamp((grid + 1.0) / 2.0, 0, 1)
        rendered.append((grid * 255).byte().cpu())
    stack = torch.stack(rendered)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.io.write_video(str(path), stack, fps=fps)
    return path


class FramePackInference:
    def __init__(self, model_paths: ModelPaths, settings: GenerationSettings):
        self.model_paths = model_paths
        self.settings = settings
        self._shared_models: dict[str, Any] = {}

    @classmethod
    def from_env(cls) -> Self:
        def _require_env(key: str) -> str:
            value = os.getenv(key)
            if not value:
                raise InferenceError(f"環境変数 {key} が未設定です")
            return value

        def _resolve_path(value: str) -> str:
            path = Path(value)
            return str(path.expanduser().resolve())

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
        return cls(model_paths=model_paths, settings=settings)

    def generate_to_path(
        self,
        image_bytes: bytes,
        *,
        prompt: str | None = None,
        infer_steps: int | None = None,
        guidance_scale: float | None = None,
        lora_multiplier: float | None = None,
        total_frames: int | None = None,
        latent_window_size: int | None = None,
    ) -> Path:
        resized_bytes = resize_image_to_bucket(
            image_bytes, resolution=self.settings.bucket_resolution
        )
        video_tensor = self._run_framepack(
            resized_bytes,
            prompt=prompt,
            infer_steps=infer_steps,
            guidance_scale=guidance_scale,
            lora_multiplier=lora_multiplier,
            total_frames=total_frames,
            latent_window_size=latent_window_size,
        )
        filename = (
            f"infer_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.mp4"
        )
        output_path = self.settings.output_dir / filename
        save_video_tensor(video_tensor, output_path, fps=self.settings.fps)
        return output_path

    def _run_framepack(
        self,
        image_bytes: bytes,
        *,
        prompt: str | None = None,
        infer_steps: int | None = None,
        guidance_scale: float | None = None,
        lora_multiplier: float | None = None,
        total_frames: int | None = None,
        latent_window_size: int | None = None,
    ) -> torch.Tensor:
        effective_prompt = prompt or self.settings.prompt
        effective_infer_steps = infer_steps or self.settings.infer_steps
        effective_latent_window_size = (
            latent_window_size or self.settings.latent_window_size
        )
        effective_video_sections = self.settings.video_sections

        if total_frames is not None and total_frames > 1:
            sections = max(
                int(round((total_frames - 1) / (effective_latent_window_size * 4))), 1
            )
            effective_video_sections = sections

        model_paths = asdict(self.model_paths)
        if lora_multiplier is not None:
            model_paths["lora_multiplier"] = [lora_multiplier]

        video = generate_video(
            model_paths=model_paths,
            prompt=effective_prompt,
            image_bytes=image_bytes,
            video_sections=effective_video_sections,
            fps=self.settings.fps,
            infer_steps=effective_infer_steps,
            latent_window_size=effective_latent_window_size,
            cache_dir=str(self.settings.cache_dir),
            shared_models=self._shared_models,
            guidance_scale=guidance_scale,
            lora_multiplier=lora_multiplier,
        )
        if video is None:
            raise InferenceError("FramePack pipeline returned no video")
        return video


def construct_args(
    prompt: str,
    image_path: str,
    width: int,
    height: int,
    video_sections: int,
    fps: int,
    infer_steps: int,
    latent_window_size: int,
    dit_path: str,
    vae_path: str,
    text_encoder1_path: str,
    text_encoder2_path: str,
    image_encoder_path: str,
    lora_weight: list[str] | None = None,
    lora_multiplier: list[float] | None = None,
) -> Any:
    import argparse

    args = argparse.Namespace()
    args.sample_solver = "unipc"
    args.compile = False
    args.dit = dit_path
    args.vae = vae_path
    args.text_encoder1 = text_encoder1_path
    args.text_encoder2 = text_encoder2_path
    args.image_encoder = image_encoder_path
    args.f1 = False
    args.lora_weight = lora_weight
    args.lora_multiplier = lora_multiplier or [1.0]
    args.include_patterns = None
    args.exclude_patterns = None
    args.save_merged_model = None
    args.prompt = prompt
    args.negative_prompt = None
    args.custom_system_prompt = None
    args.video_size = (height, width)
    args.video_seconds = 5.0
    args.video_sections = video_sections
    args.one_frame_inference = None
    args.control_image_path = None
    args.control_image_mask_path = None
    args.fps = fps
    args.infer_steps = infer_steps
    args.save_path = "/app/output"
    args.seed = 42
    args.latent_window_size = latent_window_size
    args.embedded_cfg_scale = 10.0
    args.guidance_scale = 1.0
    args.guidance_rescale = 0.0
    args.image_path = image_path
    args.end_image_path = None
    args.latent_paddings = None
    args.latent_path = None
    args.bulk_decode = False
    args.flow_shift = None
    args.fp8 = False
    args.fp8_scaled = True
    args.one_frame_auto_resize = False
    args.rope_scaling_factor = 0.5
    args.rope_scaling_timestep_threshold = None
    args.fp8_llm = False
    args.device = "cuda"
    args.attn_mode = "sdpa"
    args.vae_chunk_size = 32
    args.vae_spatial_tile_sample_min_size = 128
    args.vae_tiling = False
    args.disable_numpy_memmap = False
    args.blocks_to_swap = 0
    args.use_pinned_memory_for_block_swap = False
    args.output_type = "video"
    args.no_metadata = False
    args.lycoris = False
    args.magcache_mag_ratios = None
    args.magcache_retention_ratio = 0.2
    args.magcache_threshold = 0.24
    args.magcache_k = 6
    args.magcache_calibration = False
    args.from_file = None
    args.interactive = False
    return args


def generate_video(
    model_paths: dict[str, Any],
    prompt: str,
    image_bytes: bytes,
    video_sections: int,
    fps: int,
    infer_steps: int,
    latent_window_size: int,
    cache_dir: str,
    shared_models: dict[str, Any],
    guidance_scale: float | None = None,
    lora_multiplier: float | None = None,
) -> torch.Tensor:
    from musubi_tuner.fpack_generate_video import (
        decode_latent,
        generate,
        get_generation_settings,
        load_dit_model,
        load_shared_models,
    )
    from musubi_tuner.frame_pack.framepack_utils import load_vae

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache_root, suffix=".png", delete=False
        ) as tmp:
            tmp.write(image_bytes)
            image_path = tmp.name
            tmp_path = Path(image_path)

        with Image.open(image_path) as img:
            width, height = img.size

        args = construct_args(
            prompt=prompt,
            image_path=image_path,
            width=width,
            height=height,
            video_sections=video_sections,
            fps=fps,
            infer_steps=infer_steps,
            latent_window_size=latent_window_size,
            dit_path=model_paths["dit"],
            vae_path=model_paths["vae"],
            text_encoder1_path=model_paths["text_encoder1"],
            text_encoder2_path=model_paths["text_encoder2"],
            image_encoder_path=model_paths["image_encoder"],
            lora_weight=model_paths.get("lora_weight"),
            lora_multiplier=model_paths.get("lora_multiplier"),
        )

        gen_settings = get_generation_settings(args)
        device = gen_settings.device

        if not shared_models:
            shared_models.update(load_shared_models(args))
            shared_models["conds_cache"] = {}
            shared_models["model"] = load_dit_model(args, device)
            shared_models["lora_multiplier"] = (
                lora_multiplier
                if lora_multiplier is not None
                else (model_paths.get("lora_multiplier") or [1.0])[0]
            )
            shared_models["vae"] = load_vae(
                args.vae,
                args.vae_chunk_size,
                args.vae_spatial_tile_sample_min_size,
                args.vae_tiling,
                device,
            )
        elif (
            lora_multiplier is not None
            and shared_models.get("lora_multiplier") != lora_multiplier
        ):
            shared_models["model"] = load_dit_model(args, device)
            shared_models["lora_multiplier"] = lora_multiplier

        if guidance_scale is not None:
            args.guidance_scale = guidance_scale

        vae, latent = generate(args, gen_settings, shared_models=shared_models)
        if vae is None:
            raise InferenceError("VAE was None after generation")

        total_latent_sections = (args.video_seconds * 30) / (
            args.latent_window_size * 4
        )
        total_latent_sections = int(max(round(total_latent_sections), 1))
        video = decode_latent(
            args.latent_window_size,
            total_latent_sections,
            args.bulk_decode,
            vae,
            latent,
            device,
            args.one_frame_inference is not None,
        )
        return video
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)
