from __future__ import annotations

import sys
import time
import tempfile
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Iterable, Optional, Tuple, Any, Dict

import torch
import tomllib
from PIL import Image

from musubi_tuner import fpack_generate_video as fpack
from musubi_tuner.utils.device_utils import clean_memory_on_device

# プロジェクトルートとデフォルト設定ファイル
ROOT_DIR = Path(__file__).resolve().parents[2]
# デモ用推論設定をここから読む
DEFAULT_CONFIG_PATH = ROOT_DIR / "demo" / "configs" / "inference.toml"
# Gradio からの呼び出し時は、このディレクトリ配下に生成結果を置く（デフォルト）
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "gradio"


def load_inference_params(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """推論時の可変パラメータを config のトップレベルから読み取る。"""

    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    def _fetch(key: str, default=None):
        return cfg.get(key, default)

    # LoRA はトップレベルの lora_weight のみを見る
    lora_weight = _fetch("lora_weight", None)

    # save_path: 空/未指定なら None（呼び出し側でデフォルトにフォールバック）
    save_path_raw = _fetch("save_path", None)
    save_path = Path(save_path_raw) if save_path_raw else None

    # video_size が配列で与えられる場合を優先
    video_size_val = _fetch("video_size", None)
    if isinstance(video_size_val, (list, tuple)) and len(video_size_val) == 2:
        height_val, width_val = int(video_size_val[0]), int(video_size_val[1])
    else:
        height_val = int(_fetch("height", 256))
        width_val = int(_fetch("width", 256))

    return {
        "prompt": _fetch("prompt", "a girl walking in the snow, cinematic"),
        "negative_prompt": _fetch("negative_prompt", ""),
        "video_seconds": float(_fetch("video_seconds", 5.0)),
        "fps": int(_fetch("fps", 30)),
        "height": height_val,
        "width": width_val,
        "infer_steps": int(_fetch("infer_steps", 25)),
        "seed": int(_fetch("seed")) if _fetch("seed") not in (None, "", "null") else None,
        "sample_solver": _fetch("sample_solver", "unipc"),
        "lora_weight": lora_weight,
        "lora_multiplier": float(_fetch("lora_multiplier", 1.0)),
        "save_path": save_path,
    }


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


def _config_to_cli_args(cfg: Dict[str, Any], image_path: Path, default_output_dir: Path) -> tuple[list[str], Path]:
    """
    TOML の内容をそのまま musubi_tuner に渡す CLI 引数列へ変換する。
    - キー名は変えずに `--key value` 形式で渡す
    - bool は \"true\"/\"false\" に変換
    - list / tuple は `--key v1 v2 ...` として渡す
    - save_path が未指定なら default_output_dir/ {tag}_{timestamp} を設定する
    """

    cfg = dict(cfg)  # shallow copy
    def _first_lora_path() -> Optional[Path]:
        lw = cfg.get("lora_weight")
        if lw is None:
            return None
        if isinstance(lw, (list, tuple)):
            return Path(lw[0])
        return Path(lw)

    # save_path を決定
    base_save = cfg.get("save_path")
    if not base_save:
        time_suffix = time.strftime("%Y%m%d_%H%M%S")
        tag = _first_lora_path().stem if _first_lora_path() else "nolora"
        base_save = str(default_output_dir / f"{tag}_{time_suffix}")
        cfg["save_path"] = base_save

    save_path = Path(base_save)
    save_path.mkdir(parents=True, exist_ok=True)

    args: list[str] = []

    flag_keys = {
        "fp8",
        "fp8_scaled",
        "fp8_llm",
        "vae_tiling",
        "bulk_decode",
        "use_pinned_memory_for_block_swap",
        "compile_fullgraph",
        "compile",
        "f1",
    }

    def push(k: str, v: Any):
        if v is None:
            return

        # special handling
        if k == "sdpa":
            if v:
                args.extend(["--attn_mode", "sdpa"])
            return

        if k in flag_keys:
            if v:
                args.append(f"--{k}")
            return

        if isinstance(v, bool):
            # booleans that are not flag_keys: pass as true/false string
            args.extend([f"--{k}", str(v).lower()])
        elif isinstance(v, (list, tuple)):
            args.append(f"--{k}")
            args.extend(str(x) for x in v)
        else:
            args.extend([f"--{k}", str(v)])

    # トップレベルのみをそのまま CLI へ
    for key, val in cfg.items():
        if key == "inference":
            continue  # 明示的に無視
        push(key, val)

    # 画像パスは常に末尾に追加
    push("image_path", str(image_path))

    return args, save_path


def run_framepack_inference(
    prompt: str,
    image: Image.Image,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: Optional[str] = None,
) -> Path:
    """
    FramePack で image2video を行うユーティリティ。

    Gradio コールバックから直接呼ぶことを想定。
    戻り値は生成された mp4 ファイルのパス。
    入力画像は一時ファイルにのみ保存し、生成後は削除する。
    """

    config_path = Path(config_path)
    output_dir = Path(output_dir)

    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    # LoRA の存在チェック（設定にある場合のみ）
    lora_val = cfg.get("inference", {}).get("lora_weight") or cfg.get("lora_weight")
    if lora_val:
        for lp in _as_list(lora_val):
            if not Path(lp).exists():
                raise FileNotFoundError(f"LoRA weight not found: {lp}")

    # 入力画像を一時ファイルに保存（推論後削除）
    with tempfile.NamedTemporaryFile(prefix="framepack_i2v_", suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        temp_image_path = tmp.name

    cli_args, _ = _config_to_cli_args(cfg, Path(temp_image_path), output_dir)

    # device は明示指定があれば上書き
    if device is not None:
        cli_args.extend(["--device", device])

    try:
        args = _parse_args(cli_args)

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        args.device = torch.device(resolved_device)

        gen_settings = fpack.get_generation_settings(args)
        vae, latent = fpack.generate(args, gen_settings)

        if latent is None or vae is None:
            raise RuntimeError("generation returned no latent/vae; check arguments.")

        video_seconds = float(getattr(args, "video_seconds", 5.0))
        fps = float(getattr(args, "fps", 30))
        total_latent_sections = int(max(round(video_seconds * fps / (args.latent_window_size * 4)), 1))
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


__all__ = ["run_framepack_inference", "load_inference_params"]
