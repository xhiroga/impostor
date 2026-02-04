import os
from logging import getLogger
from pathlib import Path, PurePosixPath

import modal
from dotenv import load_dotenv
from grpclib import GRPCError
from grpclib.const import Status
from modal.volume import FileEntryType

logger = getLogger(__name__)

VOLUME_NAME = "impostor-models"
REMOTE_BASE = PurePosixPath("/")
ENV_PATHS: tuple[tuple[str, PurePosixPath], ...] = (
    ("IMPOSTOR_DIT_PATH", REMOTE_BASE / "diffusion_models/FramePackI2V_HY"),
    ("IMPOSTOR_VAE_PATH", REMOTE_BASE / "vae/diffusion_pytorch_model.safetensors"),
    (
        "IMPOSTOR_TEXT_ENCODER1_PATH",
        REMOTE_BASE / "text_encoder/model-00001-of-00004.safetensors",
    ),
    (
        "IMPOSTOR_TEXT_ENCODER2_PATH",
        REMOTE_BASE / "text_encoder_2/model.safetensors",
    ),
    ("IMPOSTOR_IMAGE_ENCODER_PATH", REMOTE_BASE / "image_encoder/model.safetensors"),
    (
        "IMPOSTOR_LORA_WEIGHT",
        REMOTE_BASE / "impostor-models/impostor-v4-step00005000.safetensors",
    ),
)


def _targets() -> list[tuple[Path, PurePosixPath]]:
    logger.debug("start _targets")
    targets: list[tuple[Path, PurePosixPath]] = []
    for env_key, remote in ENV_PATHS:
        raw = os.getenv(env_key)
        if not raw:
            raise RuntimeError(f"環境変数 {env_key} が未設定です")
        local = Path(raw).expanduser()
        if env_key == "IMPOSTOR_DIT_PATH" and local.is_dir():
            for filename in (
                "diffusion_pytorch_model-00001-of-00003.safetensors",
                "diffusion_pytorch_model-00002-of-00003.safetensors",
                "diffusion_pytorch_model-00003-of-00003.safetensors",
            ):
                targets.append((local / filename, remote / filename))
            continue
        if env_key == "IMPOSTOR_TEXT_ENCODER1_PATH":
            shard_dir = local if local.is_dir() else local.parent
            for filename in (
                "model-00001-of-00004.safetensors",
                "model-00002-of-00004.safetensors",
                "model-00003-of-00004.safetensors",
                "model-00004-of-00004.safetensors",
            ):
                targets.append((shard_dir / filename, remote.parent / filename))
            continue
        targets.append((local, remote))

    logger.debug(f"{len(targets)=}")
    return targets


def _require_locals(targets: list[tuple[Path, PurePosixPath]]) -> None:
    logger.debug(f"{targets=}")
    missing = [local for local, _ in targets if not local.is_file()]
    if missing:
        raise FileNotFoundError(
            "以下のモデルファイルを配置してください:\n" + "\n".join(str(p) for p in missing)
        )
    logger.debug("missing files not found")


def _volume() -> modal.Volume:
    logger.debug("start _volume")
    modal.Volume.objects.create(VOLUME_NAME, allow_existing=True)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    logger.debug(f"{volume=}")
    return volume


def _needs_upload(
    volume: modal.Volume, local: Path, remote: PurePosixPath
) -> bool:
    logger.debug(f"{local=}, {remote=}")
    try:
        entry = volume.listdir(remote.as_posix())[0]
    except (FileNotFoundError, IndexError):
        logger.debug("remote not found")
        return True
    except GRPCError as exc:
        if exc.status == Status.NOT_FOUND:
            logger.debug("remote not found (grpc)")
            return True
        logger.debug("grpc error")
        return True
    if entry.type != FileEntryType.FILE:
        raise RuntimeError(f"{remote} はファイルではありません")
    needs = entry.size != local.stat().st_size
    logger.debug(f"{needs=}")
    return needs


def _upload(volume: modal.Volume, targets: list[tuple[Path, PurePosixPath]]) -> None:
    logger.debug(f"{targets=}")
    force_upload = os.getenv("IMPOSTOR_MODAL_FORCE_UPLOAD", "false").lower() == "true"
    pending = (
        targets
        if force_upload
        else [
            (local, remote)
            for local, remote in targets
            if _needs_upload(volume, local, remote)
        ]
    )
    if not pending:
        print("already uploaded")
        return
    with modal.enable_output():
        with volume.batch_upload(force=True) as batch:
            for local, remote in pending:
                print(f"upload {local} -> {remote}")
                batch.put_file(local, remote.as_posix())
    logger.debug("upload completed")


def main() -> None:
    logger.debug("start main")
    load_dotenv()
    targets = _targets()
    _require_locals(targets)
    _upload(_volume(), targets)
    logger.debug("end main")


if __name__ == "__main__":
    main()
