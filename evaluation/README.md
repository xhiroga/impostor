# Evaluation CLI

動画からフレームをサンプリングし、複数の指標を一括で計算するツールです。`python -m evaluation.main` または Makefile の `make eval` で実行できます。

## クイックスタート

```bash
# 推奨: モジュール実行
uv run --extra eval evaluation/main.py --video path/to/video.mp4

# モデルを指定して実行（例: PSNR と Cosine）
uv run --extra eval evaluation/main.py --video sample/sample-rion.mp4 --models psnr cosine --frame-indices 0,15,30
```

または、
```make
make eval
```

出力は標準出力に表形式で表示され、`evaluation/result/` 以下に CSV (`evaluation_YYYYMMDD_HHMMSS.csv`) で保存されます。

## 主要オプション

- `--models`: 実行するモデル名を1つ以上指定。省略時は全モデルを実行。利用可能: `psnr`, `cosine`, `clip`, `lpips`, `ssim`
- `--frame-indices`: 0始まりのフレーム番号をカンマ区切りで指定。未指定なら `--stride` / `--max-frames` に従い自動サンプリング。
- `--stride`: フレームを N 枚ごとに間引いて取得。
- `--max-frames`: 指定がない場合にサンプリングする最大フレーム数（デフォルト 16）。
- `--result-dir`: 結果 CSV の出力先（デフォルト `evaluation/result`）。
- `--list-models`: 利用可能なモデル名を一覧表示して終了。

## 依存関係（オプション）

- `clip` 指標: `open_clip_torch`
- `lpips` 指標: `lpips`

評価時だけ追加したい場合は `pyproject.toml` の `optional-dependencies`（例: `evaluation` extra）を使い、`uv run --extra evaluation -m evaluation.main ...` のように実行してください。

## モデル追加の手順

1. `evaluation/metrics/` に `EvaluationModel` プロトコルを満たすクラスを作成。
2. `evaluation/metrics/__init__.py` の `_REGISTRY` に登録。
3. 依存ライブラリがあれば `optional-dependencies` に追記し、README のモデル一覧も更新。


## 注意

- `torchvision` の動画デコードは v0.24 で非推奨予定です。将来的には TorchCodec などへの移行を検討してください。
- CLIP や LPIPS の初回実行時はモデル重みをダウンロードするため、ネットワーク環境が必要です。
