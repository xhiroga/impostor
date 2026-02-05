# Impostor

We propose a method for 360° rendering from a single image, leveraging the consistency of video generation models.
By combining generated videos with a WebUI, we provide an experience where users can manipulate objects like 3D models at a low cost.

## Inference Demo

```sh
make demo
```

## Train

1. Start a Pod with a Volume attached on RunPod.
2. Connect via Remote SSH.
3. Execute commands.

```sh
cd /workspace
mkdir -p models
mkdir -p tmp
git clone https://huggingface.co/datasets/sawara-dev/impostor-data
git clone https://github.com/xhiroga/impostor
cd impostor
uv pip install hatchling
uv sync --extra gpu
zellij  # or tmux
make models
make cache
make train
```

## Deploy

```sh
uv run --extra gpu --env-file $(git rev-parse --show-toplevel)/.env setup_modal.py
uv run --extra gpu --env-file $(git rev-parse --show-toplevel)/.env modal deploy -m modal_app
```
