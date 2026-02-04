# Impostor

## Inference Demo

```sh
make demo
```

## Train

1. RunPodでVolumeをアタッチしたPodを起動する
2. Remote SSHで接続
3. コマンドを実行

```sh
cd /workspace
mkdir -p models
mkdir -p tmp
git clone https://huggingface.co/datasets/sawara-dev/impostor-data
git clone https://github.com/xhiroga/impostor
cd impostor
uv pip install hatchling
uv sync
zellij  # or tmux
make models
make cache
make train
```

## Deploy

```sh
uv run --env-file $(git rev-parse --show-toplevel)/.env setup_modal.py
uv run --env-file $(git rev-parse --show-toplevel)/.env modal deploy -m modal_app.app
```
