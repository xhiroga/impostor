# Imposter

## Setup

1. RunPodでVolumeをアタッチしたPodを起動する
2. Remote SSHで接続
3. コマンドを実行

```sh
cd /workspace
mkdir -p models
mkdir -p tmp
git clone https://huggingface.co/datasets/sawara-dev/imposter-data
git clone https://huggingface.co/datasets/xhiroga/data
git clone https://github.com/xhiroga/imposter
cd imposter
uv venv
uv pip install hatchling
uv sync
zellij  # or tmux
make models
make cache
make train
```
