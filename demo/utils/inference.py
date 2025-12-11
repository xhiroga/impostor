from __future__ import annotations

import sys
import time
import tempfile
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
import tomllib
from PIL import Image

from musubi_tuner import fpack_generate_video as fpack
from musubi_tuner.utils.device_utils import clean_memory_on_device

# プロジェクトルートとデフォルト設定ファイル
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "v1" / "config.toml"
# Gradio からの呼び出し時は、このディレクトリ配下に生成結果を置く
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "gradio"


@dataclass
class FramePackConfig:
    """FramePack 推論で使う主要パスと既定値。"""

    dit: Path
    vae: Path
    text_encoder1: Path
    text_encoder2: Path
    image_encoder: Path
    vae_chunk_size: Optional[int] = None
    vae_spatial_tile_sample_min_size: Optional[int] = None
    fp8: bool = False
    fp8_scaled: bool = False
    fp8_llm: bool = False
    blocks_to_swap: int = 0
    attn_mode: str = "torch"  # torch / sdpa / xformers / flash / sageattn


def _load_framepack_config(config_path: Path = DEFAULT_CONFIG_PATH) -> FramePackConfig:
    """toml からベースモデルのパス類を読み込む。"""

    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    return FramePackConfig(
        dit=Path(cfg["dit"]),
        vae=Path(cfg["vae"]),
        text_encoder1=Path(cfg["text_encoder1"]),
        text_encoder2=Path(cfg["text_encoder2"]),
        image_encoder=Path(cfg["image_encoder"]),
        vae_chunk_size=cfg.get("vae_chunk_size"),
        vae_spatial_tile_sample_min_size=cfg.get("vae_spatial_tile_sample_min_size"),
        fp8=cfg.get("fp8_base", False),
        fp8_scaled=cfg.get("fp8_scaled", False),
        fp8_llm=cfg.get("fp8_llm", False),
        blocks_to_swap=cfg.get("blocks_to_swap", 0),
        attn_mode="sdpa" if cfg.get("sdpa") else "torch",
    )


def _as_list(value: str | Path | Sequence[str | Path]) -> list[str]:
    """文字列/Path またはリストを必ず文字列リストに変換。"""

    if isinstance(value, (str, Path)):
        return [str(value)]
    return [str(v) for v in value]


def _parse_args(cli_args: Iterable[str]):
    """musubi_tuner 側の argparse をそのまま使って Namespace を生成。"""

    argv_backup = sys.argv
    sys.argv = ["fpack_infer"] + list(cli_args)
    try:
        args = fpack.parse_args()
    finally:
        sys.argv = argv_backup
    return args


def _prepare_cli_args(
    prompt: str,
    lora_paths: list[str] | None,
    save_dir: Path,
    negative_prompt: str,
    cfg: FramePackConfig,
    video_size: Tuple[int, int],
    video_seconds: float,
    fps: int,
    infer_steps: int,
    lora_multipliers: list[float],
    seed: Optional[int],
    device: Optional[str],
    sample_solver: str,
) -> list[str]:
    """CLI 互換の引数リストを組み立てる。"""

    h, w = video_size
    cli_args: list[str] = [
        "--dit",
        str(cfg.dit),
        "--vae",
        str(cfg.vae),
        "--text_encoder1",
        str(cfg.text_encoder1),
        "--text_encoder2",
        str(cfg.text_encoder2),
        "--image_encoder",
        str(cfg.image_encoder),
        "--prompt",
        prompt,
        "--save_path",
        str(save_dir),
        "--video_size",
        str(h),
        str(w),
        "--video_seconds",
        str(video_seconds),
        "--fps",
        str(fps),
        "--infer_steps",
        str(infer_steps),
        "--sample_solver",
        sample_solver,
    ]
    if lora_paths:
        cli_args.append("--lora_weight")
        cli_args.extend(lora_paths)
        cli_args.extend(["--lora_multiplier", *[str(m) for m in lora_multipliers]])

    if negative_prompt:
        cli_args.extend(["--negative_prompt", negative_prompt])
    if seed is not None:
        cli_args.extend(["--seed", str(seed)])
    if device is not None:
        cli_args.extend(["--device", device])

    # config 由来のオプション
    if cfg.vae_chunk_size is not None:
        cli_args.extend(["--vae_chunk_size", str(cfg.vae_chunk_size)])
    if cfg.vae_spatial_tile_sample_min_size is not None:
        cli_args.extend(
            [
                "--vae_spatial_tile_sample_min_size",
                str(cfg.vae_spatial_tile_sample_min_size),
                "--vae_tiling",
            ]
        )
    if cfg.fp8:
        cli_args.append("--fp8")
    if cfg.fp8_scaled:
        cli_args.append("--fp8_scaled")
    if cfg.fp8_llm:
        cli_args.append("--fp8_llm")
    if cfg.blocks_to_swap:
        cli_args.extend(["--blocks_to_swap", str(cfg.blocks_to_swap)])
    if cfg.attn_mode:
        cli_args.extend(["--attn_mode", cfg.attn_mode])

    return cli_args


def run_framepack_inference(
    prompt: str,
    image: Image.Image,
    *,
    lora_path: str | Path | None = None,
    negative_prompt: str = "",
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    video_size: Tuple[int, int] = (256, 256),
    video_seconds: float = 5.0,
    fps: int = 30,
    infer_steps: int = 25,
    lora_multiplier: float | Sequence[float] = 1.0,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    sample_solver: str = "unipc",
) -> Path:
    """
    FramePack で image2video を行うユーティリティ。

    Gradio コールバックから直接呼ぶことを想定。
    戻り値は生成された mp4 ファイルのパス。
    入力画像は一時ファイルにのみ保存し、生成後は削除する。
    """

    config_path = Path(config_path)
    output_dir = Path(output_dir)

    cfg = _load_framepack_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_list = None
    multiplier_list: list[float] = []
    if lora_path is not None:
        lora_list = _as_list(lora_path)
        for lp in lora_list:
            if not Path(lp).exists():
                raise FileNotFoundError(f"LoRA weight not found: {lp}")
        multiplier_list = (
            lora_multiplier
            if isinstance(lora_multiplier, Sequence) and not isinstance(lora_multiplier, (str, bytes))
            else [lora_multiplier]  # type: ignore[arg-type]
        )
        if len(multiplier_list) == 1 and len(lora_list) > 1:
            multiplier_list = list(multiplier_list) * len(lora_list)

    # 保存先ディレクトリを一意にする（タイムスタンプを subdir に付与）
    time_suffix = time.strftime("%Y%m%d_%H%M%S")
    tag = Path(lora_list[0]).stem if lora_list else "nolora"
    save_dir = output_dir / f"{tag}_{time_suffix}"

    # 入力画像を一時ファイルに保存（推論後削除）
    with tempfile.NamedTemporaryFile(prefix="framepack_i2v_", suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        temp_image_path = tmp.name

    cli_args = _prepare_cli_args(
        prompt=prompt,
        lora_paths=lora_list,
        save_dir=save_dir,
        negative_prompt=negative_prompt,
        cfg=cfg,
        video_size=video_size,
        video_seconds=video_seconds,
        fps=fps,
        infer_steps=infer_steps,
        lora_multipliers=[float(m) for m in multiplier_list] if lora_list else [],
        seed=seed,
        device=device,
        sample_solver=sample_solver,
    )
    cli_args.extend(["--image_path", temp_image_path])

    try:
        args = _parse_args(cli_args)

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        args.device = torch.device(resolved_device)

        gen_settings = fpack.get_generation_settings(args)
        vae, latent = fpack.generate(args, gen_settings)

        if latent is None or vae is None:
            raise RuntimeError("generation returned no latent/vae; check arguments.")

        total_latent_sections = int(max(round(video_seconds * 30 / (args.latent_window_size * 4)), 1))
        video_tensor = fpack.decode_latent(
            args.latent_window_size,
            total_latent_sections,
            args.bulk_decode,
            vae,
            latent,
            gen_settings.device,
            args.one_frame_inference is not None,
        )
        video_path = fpack.save_video(video_tensor, args)
    finally:
        if "temp_image_path" in locals() and os.path.exists(temp_image_path):
            os.remove(temp_image_path)

    clean_memory_on_device(gen_settings.device)
    return Path(video_path)


__all__ = ["run_framepack_inference"]
