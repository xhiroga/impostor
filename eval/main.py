from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from eval.video_utils import extract_frames, load_video_frames, resolve_frame_indices
    from eval.metrics import available_models, build_models
else:  # pragma: no cover - import path shim
    from .video_utils import extract_frames, load_video_frames, resolve_frame_indices
    from .metrics import available_models, build_models


def _parse_indices(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    cleaned = raw.replace(" ", "")
    indices: list[int] = []
    for part in cleaned.split(","):
        if not part:
            continue
        try:
            indices.append(int(part))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid frame index: {part}")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video evaluator: sample frames and run pluggable metrics."
    )
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Models to run. Available: {', '.join(available_models())}",
    )
    parser.add_argument(
        "--frame-indices",
        type=_parse_indices,
        default=None,
        help="Comma-separated frame indices to sample (0-based). Overrides stride/max-frames.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Take every Nth frame. Ignored when frame-indices provided.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=16,
        help="Maximum number of frames to sample when indices not provided.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("eval/result"),
        help="Directory to store evaluation outputs.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit.",
    )
    return parser.parse_args()


def format_stdout(results: Sequence[dict[str, str | float]]) -> str:
    lines = []
    header = f"{'model':<10} {'metric':<22} {'value':>10} {'frames':<12} detail"
    lines.append(header)
    lines.append("-" * len(header))
    for row in results:
        frames = row.get("frames", "")
        detail = row.get("detail", "") or ""
        lines.append(
            f"{row['model']:<10} {row['metric']:<22} {row['value']:>10.4f} {frames:<12} {detail}"
        )
    return "\n".join(lines)


def save_results(
    rows: Sequence[dict[str, str | float]], *, directory: Path
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"evaluation_{timestamp}.csv"

    fieldnames = ["model", "metric", "value", "frames", "detail"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def main() -> None:
    args = parse_args()

    if args.list_models:
        print("Available models:", ", ".join(available_models()))
        return

    frames_raw, fps = load_video_frames(args.video)
    selected_indices = resolve_frame_indices(
        total_frames=frames_raw.shape[0],
        frame_indices=args.frame_indices,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    if not selected_indices:
        raise SystemExit("No frames were selected from the video.")

    frames = extract_frames(frames_raw, selected_indices)
    relative_indices = list(range(frames.shape[0]))
    models = build_models(args.models)

    results: list[dict[str, str | float]] = []

    def remap_frames(frames_used: tuple[int, ...] | None) -> str:
        if not frames_used:
            return ""
        valid = [i for i in frames_used if i < len(selected_indices)]
        return ",".join(str(selected_indices[i]) for i in valid)
    for model in models:
        for metric in model.evaluate(frames, relative_indices):
            results.append(
                {
                    "model": metric.model,
                    "metric": metric.metric,
                    "value": metric.value,
                    "frames": remap_frames(metric.frames_used),
                    "detail": metric.detail or "",
                }
            )

    print(f"Video: {args.video}")
    if fps:
        print(f"FPS: {fps:.2f}")
    print(f"Frames sampled: {selected_indices}")
    print(format_stdout(results))

    output_path = save_results(results, directory=args.result_dir)
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
