# Eval CLI

Video evaluation helper that samples frames from a video and runs pluggable metrics.

## Quick start

```bash
# Run as a module (recommended)
python -m evaluation.main --video path/to/video.mp4

# Choose models and specific frames
python -m evaluation.main --video sample/video.mp4 --models psnr cosine --frame-indices 0,15,30
```

Results print to stdout and are saved as CSV under `evaluation/result/`.

## Options

- `--models`: subset from `psnr`, `cosine`, `clip`, `lpips`, `ssim` (default: all).
- `--frame-indices`: comma-separated indices to sample; overrides stride/max-frames.
- `--stride`: take every Nth frame when indices are not given.
- `--max-frames`: cap the number of sampled frames (default: 16).
- `--result-dir`: output directory (default: `evaluation/result`).
- Frame numbers in the output correspond to the original video indices; the detail column refers to the sampled order (0-based).

Add new metrics by creating a class in `evaluation/metrics/` that implements `EvaluationModel` and registering it in `evaluation/metrics/__init__.py`.

### Optional dependencies
- `clip` 指標: `pip install open_clip_torch`
- `lpips` 指標: `pip install lpips`
