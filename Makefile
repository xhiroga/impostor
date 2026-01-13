include .env	# TMP, MODEL_PATH

CONFIG_FILE ?= /workspace/impostor/configs/wan/v3/config.toml
MODEL_PATH ?= /workspace/models
# クラウドの場合はマウントしているボリューム配下のパスにすること
TMP ?= /workspace/tmp

LORA_DIR ?= impostor-v2-step00000800-state

.ONESHELL:
.PHONY: train cache models wan_train wan_cache wan_models run-comfyui frontend-build demo

# Models (Total < 200GB)
models = \
	$(MODEL_PATH)/diffusion_models/FramePackI2V_HY/diffusion_pytorch_model-00001-of-00003.safetensors \
	$(MODEL_PATH)/diffusion_models/FramePackI2V_HY/diffusion_pytorch_model-00002-of-00003.safetensors \
	$(MODEL_PATH)/diffusion_models/FramePackI2V_HY/diffusion_pytorch_model-00003-of-00003.safetensors \
	$(MODEL_PATH)/diffusion_models/FramePack_F1_I2V_HY_20250503/diffusion_pytorch_model-00001-of-00003.safetensors \
	$(MODEL_PATH)/diffusion_models/FramePack_F1_I2V_HY_20250503/diffusion_pytorch_model-00002-of-00003.safetensors \
	$(MODEL_PATH)/diffusion_models/FramePack_F1_I2V_HY_20250503/diffusion_pytorch_model-00003-of-00003.safetensors \
	$(MODEL_PATH)/vae/diffusion_pytorch_model.safetensors \
	$(MODEL_PATH)/text_encoders/model-00001-of-00004.safetensors \
	$(MODEL_PATH)/text_encoders/model-00002-of-00004.safetensors \
	$(MODEL_PATH)/text_encoders/model-00003-of-00004.safetensors \
	$(MODEL_PATH)/text_encoders/model-00004-of-00004.safetensors \
	$(MODEL_PATH)/text_encoder_2/model.safetensors \
	$(MODEL_PATH)/image_encoder/model.safetensors \
	$(MODEL_PATH)/loras/$(LORA_DIR)/model.safetensors \
	$(MODEL_PATH)/text_encoders/clip_l.safetensors \
	$(MODEL_PATH)/text_encoders/llava_llama3_fp8_scaled.safetensors \
	$(MODEL_PATH)/clip_vision/sigclip_vision_patch14_384.safetensors \

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

$(MODEL_PATH)/text_encoders/model-00001-of-00004.safetensors: REPO=hunyuanvideo-community/HunyuanVideo
$(MODEL_PATH)/text_encoders/model-00001-of-00004.safetensors: FILE=text_encoder/model-00001-of-00004.safetensors
$(MODEL_PATH)/text_encoders/model-00002-of-00004.safetensors: REPO=hunyuanvideo-community/HunyuanVideo
$(MODEL_PATH)/text_encoders/model-00002-of-00004.safetensors: FILE=text_encoder/model-00002-of-00004.safetensors
$(MODEL_PATH)/text_encoders/model-00003-of-00004.safetensors: REPO=hunyuanvideo-community/HunyuanVideo
$(MODEL_PATH)/text_encoders/model-00003-of-00004.safetensors: FILE=text_encoder/model-00003-of-00004.safetensors
$(MODEL_PATH)/text_encoders/model-00004-of-00004.safetensors: REPO=hunyuanvideo-community/HunyuanVideo
$(MODEL_PATH)/text_encoders/model-00004-of-00004.safetensors: FILE=text_encoder/model-00004-of-00004.safetensors

$(MODEL_PATH)/text_encoder_2/model.safetensors: REPO=hunyuanvideo-community/HunyuanVideo
$(MODEL_PATH)/text_encoder_2/model.safetensors: FILE=text_encoder_2/model.safetensors

$(MODEL_PATH)/image_encoder/model.safetensors: REPO=lllyasviel/flux_redux_bfl
$(MODEL_PATH)/image_encoder/model.safetensors: FILE=image_encoder/model.safetensors

$(MODEL_PATH)/loras/$(LORA_DIR)/model.safetensors: REPO=sawara-dev/impostor-models
$(MODEL_PATH)/loras/$(LORA_DIR)/model.safetensors: FILE=$(LORA_DIR)/model.safetensors

## for ComfyUI
$(MODEL_PATH)/text_encoders/clip_l.safetensors: REPO=Comfy-Org/HunyuanVideo_repackaged
$(MODEL_PATH)/text_encoders/clip_l.safetensors: FILE=split_files/text_encoders/clip_l.safetensors
$(MODEL_PATH)/text_encoders/llava_llama3_fp8_scaled.safetensors: REPO=Comfy-Org/HunyuanVideo_repackaged
$(MODEL_PATH)/text_encoders/llava_llama3_fp8_scaled.safetensors: FILE=split_files/text_encoders/llava_llama3_fp8_scaled.safetensors
$(MODEL_PATH)/clip_vision/sigclip_vision_patch14_384.safetensors: REPO=Comfy-Org/sigclip_vision_384
$(MODEL_PATH)/clip_vision/sigclip_vision_patch14_384.safetensors: FILE=sigclip_vision_patch14_384.safetensors

# WAN2.1 / WAN2.2 Models (Total < 200GB)
wan_models = \
	$(MODEL_PATH)/text_encoders/models_t5_umt5-xxl-enc-bf16.pth \
	$(MODEL_PATH)/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
	$(MODEL_PATH)/clip_vision/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
	$(MODEL_PATH)/clip_vision/clip_vision_h.safetensors \
	$(MODEL_PATH)/vae/wan_2.1_vae.safetensors \
	$(MODEL_PATH)/diffusion_models/wan2.1_i2v_720p_14B_bf16.safetensors \
	$(MODEL_PATH)/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors \
	$(MODEL_PATH)/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors \
	$(MODEL_PATH)/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors \
	$(MODEL_PATH)/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors \
	$(MODEL_PATH)/diffusion_models/wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors \
	$(MODEL_PATH)/diffusion_models/wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors \
	$(MODEL_PATH)/diffusion_models/wan2.2_fun_camera_high_noise_14B_fp8_scaled.safetensors \
	$(MODEL_PATH)/diffusion_models/wan2.2_fun_camera_low_noise_14B_fp8_scaled.safetensors \
	$(MODEL_PATH)/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors \
	$(MODEL_PATH)/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors \
	$(MODEL_PATH)/diffusion_models/diffusion_pytorch_model.safetensors \
	$(MODEL_PATH)/diffusion_models/Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors

# text_encoders
$(MODEL_PATH)/text_encoders/models_t5_umt5-xxl-enc-bf16.pth: REPO=Wan-AI/Wan2.1-I2V-14B-720P
$(MODEL_PATH)/text_encoders/models_t5_umt5-xxl-enc-bf16.pth: FILE=models_t5_umt5-xxl-enc-bf16.pth
## ComfyUI
$(MODEL_PATH)/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors: REPO=Comfy-Org/Wan_2.1_ComfyUI_repackaged
$(MODEL_PATH)/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors: FILE=split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors

# clip_vision
$(MODEL_PATH)/clip_vision/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth: REPO=Wan-AI/Wan2.1-I2V-14B-720P
$(MODEL_PATH)/clip_vision/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth: FILE=models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
## ComfyUI
$(MODEL_PATH)/clip_vision/clip_vision_h.safetensors: REPO=Comfy-Org/Wan_2.1_ComfyUI_repackaged
$(MODEL_PATH)/clip_vision/clip_vision_h.safetensors: FILE=split_files/clip_vision/clip_vision_h.safetensors

# wan_2.1_vae is OK for WAN2.2 14B
$(MODEL_PATH)/vae/wan_2.1_vae.safetensors: REPO=Comfy-Org/Wan_2.1_ComfyUI_repackaged
$(MODEL_PATH)/vae/wan_2.1_vae.safetensors: FILE=split_files/vae/wan_2.1_vae.safetensors

# musubi-tuner only supports bf16, fp16 and fp8_e4m3fn
$(MODEL_PATH)/diffusion_models/wan2.1_i2v_720p_14B_bf16.safetensors: REPO=Comfy-Org/Wan_2.1_ComfyUI_repackaged
$(MODEL_PATH)/diffusion_models/wan2.1_i2v_720p_14B_bf16.safetensors: FILE=split_files/diffusion_models/wan2.1_i2v_720p_14B_bf16.safetensors
## Wan2.2 i2v
$(MODEL_PATH)/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors: REPO=Comfy-Org/Wan_2.2_ComfyUI_repackaged
$(MODEL_PATH)/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors: FILE=split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors
$(MODEL_PATH)/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors: REPO=Comfy-Org/Wan_2.2_ComfyUI_repackaged
$(MODEL_PATH)/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors: FILE=split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors
$(MODEL_PATH)/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors: REPO=Comfy-Org/Wan_2.2_ComfyUI_repackaged
$(MODEL_PATH)/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors: FILE=split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
$(MODEL_PATH)/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors: REPO=Comfy-Org/Wan_2.2_ComfyUI_repackaged
$(MODEL_PATH)/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors: FILE=split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
## Wan2.2 Fun Control
$(MODEL_PATH)/diffusion_models/wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors: REPO=Comfy-Org/Wan_2.2_ComfyUI_Repackaged
$(MODEL_PATH)/diffusion_models/wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors: FILE=split_files/diffusion_models/wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors
$(MODEL_PATH)/diffusion_models/wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors: REPO=Comfy-Org/Wan_2.2_ComfyUI_Repackaged
$(MODEL_PATH)/diffusion_models/wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors: FILE=split_files/diffusion_models/wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors

## Wan2.2 Camera Control
$(MODEL_PATH)/diffusion_models/wan2.2_fun_camera_high_noise_14B_fp8_scaled.safetensors: REPO=Comfy-Org/Wan_2.2_ComfyUI_Repackaged
$(MODEL_PATH)/diffusion_models/wan2.2_fun_camera_high_noise_14B_fp8_scaled.safetensors: FILE=split_files/diffusion_models/wan2.2_fun_camera_high_noise_14B_fp8_scaled.safetensors
$(MODEL_PATH)/diffusion_models/wan2.2_fun_camera_low_noise_14B_fp8_scaled.safetensors: REPO=Comfy-Org/Wan_2.2_ComfyUI_Repackaged
$(MODEL_PATH)/diffusion_models/wan2.2_fun_camera_low_noise_14B_fp8_scaled.safetensors: FILE=split_files/diffusion_models/wan2.2_fun_camera_low_noise_14B_fp8_scaled.safetensors
## Wan2.2 LoRA
$(MODEL_PATH)/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors: REPO=Comfy-Org/Wan_2.2_ComfyUI_Repackaged
$(MODEL_PATH)/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors: FILE=split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors
$(MODEL_PATH)/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors: REPO=Comfy-Org/Wan_2.2_ComfyUI_Repackaged
$(MODEL_PATH)/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors: FILE=split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors

## Wan2.1 Fun Control for ComfyUI
$(MODEL_PATH)/diffusion_models/diffusion_pytorch_model.safetensors: REPO=alibaba-pai/Wan2.1-Fun-1.3B-Control
$(MODEL_PATH)/diffusion_models/diffusion_pytorch_model.safetensors: FILE=diffusion_pytorch_model.safetensors
## Wan 2.1 ATI for ComfyUI
$(MODEL_PATH)/diffusion_models/Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors: REPO=Kijai/WanVideo_comfy
$(MODEL_PATH)/diffusion_models/Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors: FILE=Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors

train:
	IMAGE_ENCODER=$$(uv run python -c 'from tomllib import load; d=load(open("$(CONFIG_FILE)", "rb")); print(d["image_encoder"])')

	uv run wandb login $(WANDB_API_KEY)
	uv run \
		accelerate launch \
			--num_processes 1 \
			--dynamo_backend=no \
			--mixed_precision bf16 \
			-m musubi_tuner.fpack_train_network \
				--image_encoder $$IMAGE_ENCODER \
				--config_file $(CONFIG_FILE) \
				--huggingface_token $(HUGGINGFACE_TOKEN)
	sleep 10m ; runpodctl stop pod $(RUNPOD_POD_ID) &

cache: $(models)
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

wan_train:
	uv run wandb login $(WANDB_API_KEY)
	uv run \
		accelerate launch \
			--num_processes 1 \
			--dynamo_backend=no \
			--mixed_precision bf16 \
			-m musubi_tuner.wan_train_network \
				--config_file $(CONFIG_FILE) \
				--huggingface_token $(HUGGINGFACE_TOKEN)
	sleep 10m ; runpodctl stop pod $(RUNPOD_POD_ID) &

wan_cache: $(wan_models)
	DATASET_CONFIG=$$(uv run python -c 'from tomllib import load; d=load(open("$(CONFIG_FILE)", "rb")); print(d["dataset_config"])')
	VAE=$$(uv run python -c 'from tomllib import load; d=load(open("$(CONFIG_FILE)", "rb")); print(d["vae"])')
	T5=$$(uv run python -c 'from tomllib import load; d=load(open("$(CONFIG_FILE)", "rb")); print(d["t5"])')
	CLIP=$$(uv run python -c 'from tomllib import load; d=load(open("$(CONFIG_FILE)", "rb")); print(d["clip"])')

	uv run -m musubi_tuner.wan_cache_latents \
		--dataset_config $$DATASET_CONFIG \
		--vae $$VAE \
		$${CLIP:+--clip $$CLIP} \
		--i2v

	uv run -m musubi_tuner.wan_cache_text_encoder_outputs \
		--dataset_config $$DATASET_CONFIG \
		--t5 $$T5 \
		--batch_size 16

diffsynth_train:
	uv run accelerate launch -m examples.wanvideo.model_training.train \
		--dataset_base_path /workspace/impostor-data \
		--dataset_metadata_path /workspace/impostor/configs/diffsynth/v1/metadata_camera_control.csv \
		--data_file_keys "video,control_video,reference_image" \
		--height 480 \
		--width 832 \
		--dataset_repeat 100 \
		--model_id_with_origin_paths "PAI/Wan2.2-Fun-A14B-Control-Camera:low_noise_model/diffusion_pytorch_model*.safetensors,PAI/Wan2.2-Fun-A14B-Control-Camera:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.2-Fun-A14B-Control-Camera:Wan2.1_VAE.pth" \
		--learning_rate 1e-4 \
		--num_epochs 5 \
		--remove_prefix_in_ckpt "pipe.dit." \
		--output_path "./models/train/Wan2.2-Fun-A14B-Control-Camera_low_noise_lora" \
		--lora_base_model "dit" \
		--lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
		--lora_rank 32 \
		--extra_inputs "input_image,camera_control_direction,camera_control_speed" \
		--max_timestep_boundary 1 \
		--min_timestep_boundary 0.358
# boundary corresponds to timesteps [0, 900]

	uv run accelerate launch -m examples.wanvideo.model_training.train \
		--dataset_base_path /workspace/impostor-data \
		--dataset_metadata_path /workspace/impostor/configs/diffsynth/v1/metadata_camera_control.csv \
		--data_file_keys "video,control_video,reference_image" \
		--height 480 \
		--width 832 \
		--dataset_repeat 100 \
		--model_id_with_origin_paths "PAI/Wan2.2-Fun-A14B-Control-Camera:high_noise_model/diffusion_pytorch_model*.safetensors,PAI/Wan2.2-Fun-A14B-Control-Camera:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.2-Fun-A14B-Control-Camera:Wan2.1_VAE.pth" \
		--learning_rate 1e-4 \
		--num_epochs 5 \
		--remove_prefix_in_ckpt "pipe.dit." \
		--output_path "./models/train/Wan2.2-Fun-A14B-Control-Camera_high_noise_lora" \
		--lora_base_model "dit" \
		--lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
		--lora_rank 32 \
		--extra_inputs "input_image,camera_control_direction,camera_control_speed" \
		--max_timestep_boundary 0.358 \
		--min_timestep_boundary 0
# boundary corresponds to timesteps [900, 1000]

models: $(models)
$(models):
	if uvx --from "huggingface_hub[cli]" hf auth whoami | grep -q 'Not logged in'; then uvx --from "huggingface_hub[cli]" hf auth login --token=$(HUGGINGFACE_TOKEN); fi
	uvx --from "huggingface_hub[cli]" hf download $(REPO) $(FILE) --local-dir $(TMP)/$(REPO)
	mkdir -p $(dir $@)
	mv $(TMP)/$(REPO)/$(FILE) $@

wan_models: $(wan_models)
$(wan_models):
	if uvx --from "huggingface_hub[cli]" hf auth whoami | grep -q 'Not logged in'; then uvx --from "huggingface_hub[cli]" hf auth login --token=$(HUGGINGFACE_TOKEN); fi
	uvx --from "huggingface_hub[cli]" hf download $(REPO) $(FILE) --local-dir $(TMP)/$(REPO)
	mkdir -p $(dir $@)
	mv $(TMP)/$(REPO)/$(FILE) $@

comfy/ComfyUI:
	git -C $@ pull || git clone https://github.com/comfyanonymous/ComfyUI $@
	cd $@ && python -m venv venv && venv/bin/pip install -r requirements.txt && venv/bin/pip install -r manager_requirements.txt

run-comfyui:
	comfy/ComfyUI/venv/bin/python comfy/ComfyUI/main.py --enable-manager --extra-model-paths-config $(COMFY_EXTRA_MODEL_PATHS_CONFIG)

frontend-build:
	cd frontend && pnpm install && pnpm run build

demo: frontend-build
	uv run fastapi dev main.py
