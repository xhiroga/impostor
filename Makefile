include .env	# TMP, MODEL_PATH

CONFIG_FILE ?= /workspace/impostor/configs/v2/config.toml
MODEL_PATH ?= /workspace/models
# クラウドの場合はマウントしているボリューム配下のパスにすること
TMP ?= /workspace/tmp

LORA_DIR ?= impostor-v2-step00000800-state

PNPM ?= pnpm

.ONESHELL:
.PHONY: train cache demo frontend-build

# Models
models = \
	$(MODEL_PATH)/diffusion_models/FramePackI2V_HY/diffusion_pytorch_model-00001-of-00003.safetensors \
	$(MODEL_PATH)/diffusion_models/FramePackI2V_HY/diffusion_pytorch_model-00002-of-00003.safetensors \
	$(MODEL_PATH)/diffusion_models/FramePackI2V_HY/diffusion_pytorch_model-00003-of-00003.safetensors \
	$(MODEL_PATH)/diffusion_models/FramePack_F1_I2V_HY_20250503/diffusion_pytorch_model-00001-of-00003.safetensors \
	$(MODEL_PATH)/diffusion_models/FramePack_F1_I2V_HY_20250503/diffusion_pytorch_model-00002-of-00003.safetensors \
	$(MODEL_PATH)/diffusion_models/FramePack_F1_I2V_HY_20250503/diffusion_pytorch_model-00003-of-00003.safetensors \
	$(MODEL_PATH)/vae/diffusion_pytorch_model.safetensors \
	$(MODEL_PATH)/text_encoder/model-00001-of-00004.safetensors \
	$(MODEL_PATH)/text_encoder/model-00002-of-00004.safetensors \
	$(MODEL_PATH)/text_encoder/model-00003-of-00004.safetensors \
	$(MODEL_PATH)/text_encoder/model-00004-of-00004.safetensors \
	$(MODEL_PATH)/text_encoder_2/model.safetensors \
	$(MODEL_PATH)/image_encoder/model.safetensors \
	$(MODEL_PATH)/loras/$(LORA_DIR)/model.safetensors

$(MODEL_PATH)/diffusion_models/FramePackI2V_HY/diffusion_pytorch_model-00001-of-00003.safetensors: REPO=lllyasviel/FramePackI2V_HY
$(MODEL_PATH)/diffusion_models/FramePackI2V_HY/diffusion_pytorch_model-00001-of-00003.safetensors: FILE=diffusion_pytorch_model-00001-of-00003.safetensors
$(MODEL_PATH)/diffusion_models/FramePackI2V_HY/diffusion_pytorch_model-00002-of-00003.safetensors: REPO=lllyasviel/FramePackI2V_HY
$(MODEL_PATH)/diffusion_models/FramePackI2V_HY/diffusion_pytorch_model-00002-of-00003.safetensors: FILE=diffusion_pytorch_model-00002-of-00003.safetensors
$(MODEL_PATH)/diffusion_models/FramePackI2V_HY/diffusion_pytorch_model-00003-of-00003.safetensors: REPO=lllyasviel/FramePackI2V_HY
$(MODEL_PATH)/diffusion_models/FramePackI2V_HY/diffusion_pytorch_model-00003-of-00003.safetensors: FILE=diffusion_pytorch_model-00003-of-00003.safetensors

$(MODEL_PATH)/diffusion_models/FramePack_F1_I2V_HY_20250503/diffusion_pytorch_model-00001-of-00003.safetensors: REPO=lllyasviel/FramePack_F1_I2V_HY_20250503
$(MODEL_PATH)/diffusion_models/FramePack_F1_I2V_HY_20250503/diffusion_pytorch_model-00001-of-00003.safetensors: FILE=diffusion_pytorch_model-00001-of-00003.safetensors
$(MODEL_PATH)/diffusion_models/FramePack_F1_I2V_HY_20250503/diffusion_pytorch_model-00002-of-00003.safetensors: REPO=lllyasviel/FramePack_F1_I2V_HY_20250503
$(MODEL_PATH)/diffusion_models/FramePack_F1_I2V_HY_20250503/diffusion_pytorch_model-00002-of-00003.safetensors: FILE=diffusion_pytorch_model-00002-of-00003.safetensors
$(MODEL_PATH)/diffusion_models/FramePack_F1_I2V_HY_20250503/diffusion_pytorch_model-00003-of-00003.safetensors: REPO=lllyasviel/FramePack_F1_I2V_HY_20250503
$(MODEL_PATH)/diffusion_models/FramePack_F1_I2V_HY_20250503/diffusion_pytorch_model-00003-of-00003.safetensors: FILE=diffusion_pytorch_model-00003-of-00003.safetensors

$(MODEL_PATH)/vae/diffusion_pytorch_model.safetensors: REPO=hunyuanvideo-community/HunyuanVideo
$(MODEL_PATH)/vae/diffusion_pytorch_model.safetensors: FILE=vae/diffusion_pytorch_model.safetensors

$(MODEL_PATH)/text_encoder/model-00001-of-00004.safetensors: REPO=hunyuanvideo-community/HunyuanVideo
$(MODEL_PATH)/text_encoder/model-00001-of-00004.safetensors: FILE=text_encoder/model-00001-of-00004.safetensors
$(MODEL_PATH)/text_encoder/model-00002-of-00004.safetensors: REPO=hunyuanvideo-community/HunyuanVideo
$(MODEL_PATH)/text_encoder/model-00002-of-00004.safetensors: FILE=text_encoder/model-00002-of-00004.safetensors
$(MODEL_PATH)/text_encoder/model-00003-of-00004.safetensors: REPO=hunyuanvideo-community/HunyuanVideo
$(MODEL_PATH)/text_encoder/model-00003-of-00004.safetensors: FILE=text_encoder/model-00003-of-00004.safetensors
$(MODEL_PATH)/text_encoder/model-00004-of-00004.safetensors: REPO=hunyuanvideo-community/HunyuanVideo
$(MODEL_PATH)/text_encoder/model-00004-of-00004.safetensors: FILE=text_encoder/model-00004-of-00004.safetensors

$(MODEL_PATH)/text_encoder_2/model.safetensors: REPO=hunyuanvideo-community/HunyuanVideo
$(MODEL_PATH)/text_encoder_2/model.safetensors: FILE=text_encoder_2/model.safetensors

$(MODEL_PATH)/image_encoder/model.safetensors: REPO=lllyasviel/flux_redux_bfl
$(MODEL_PATH)/image_encoder/model.safetensors: FILE=image_encoder/model.safetensors

$(MODEL_PATH)/loras/$(LORA_DIR)/model.safetensors: REPO=sawara-dev/impostor-models
$(MODEL_PATH)/loras/$(LORA_DIR)/model.safetensors: FILE=$(LORA_DIR)/model.safetensors

train: .venv
	IMAGE_ENCODER=$$(uv run python -c 'from tomllib import load; d=load(open("$(CONFIG_FILE)", "rb")); print(d["image_encoder"])')

	uv run wandb login $(WANDB_API_KEY)
	uv  run \
		accelerate launch \
			--num_processes 1 \
			--dynamo_backend=no \
			--mixed_precision bf16 \
			-m musubi_tuner.fpack_train_network \
				--image_encoder $$IMAGE_ENCODER \
				--config_file $(CONFIG_FILE) \
				--huggingface_token $(HUGGINGFACE_TOKEN)
	sleep 10m ; runpodctl stop pod $(RUNPOD_POD_ID) &

cache: .venv $(models)
	DATASET_CONFIG=$$(uv run python -c 'from tomllib import load; d=load(open("$(CONFIG_FILE)", "rb")); print(d["dataset_config"])')
	VAE=$$(uv run python -c 'from tomllib import load; d=load(open("$(CONFIG_FILE)", "rb")); print(d["vae"])')
	IMAGE_ENCODER=$$(uv run python -c 'from tomllib import load; d=load(open("$(CONFIG_FILE)", "rb")); print(d["image_encoder"])')
	TEXT_ENCODER1=$$(uv run python -c 'from tomllib import load; d=load(open("$(CONFIG_FILE)", "rb")); print(d["text_encoder1"])')
	TEXT_ENCODER2=$$(uv run python -c 'from tomllib import load; d=load(open("$(CONFIG_FILE)", "rb")); print(d["text_encoder2"])')

	uv run -m musubi_tuner.fpack_cache_latents \
		--dataset_config $$DATASET_CONFIG \
		--vae $$VAE \
		--image_encoder $$IMAGE_ENCODER \
		--vae_chunk_size 32 --vae_tiling

	uv run -m musubi_tuner.fpack_cache_text_encoder_outputs \
		--dataset_config $$DATASET_CONFIG \
		--text_encoder1 $$TEXT_ENCODER1 \
		--text_encoder2 $$TEXT_ENCODER2 \
		--batch_size 16

# VRAM48GiBあるなら、--vae_chunk_size と --vae_spatial_tile_sample_min_size の設定は不要。
# というか24GiBでも不要かも。--vae_chunk_size なしでも、Memory-Usageは13000MiB程度。
# https://deepwiki.com/search/vaetiling-vaespatialtilesample_d53d814c-27e9-405f-a43f-111b316047a3

.venv:
	uv sync

models: $(models)
$(models):
	if uvx --from "huggingface_hub[cli]" hf auth whoami | grep -q 'Not logged in'; then uvx --from "huggingface_hub[cli]" hf auth login --token=$(HUGGINGFACE_TOKEN); fi
	uvx --from "huggingface_hub[cli]" hf download $(REPO) $(FILE) --local-dir $(TMP)/$(REPO)
	mkdir -p $(dir $@)
	mv $(TMP)/$(REPO)/$(FILE) $@

app:
	@PYTHONPATH=demo uv run python -m app

frontend-build:
	cd frontend && $(PNPM) install && $(PNPM) run build

demo: frontend-build
	uv run fastapi dev main.py
