#!/bin/bash

# Log the parsed options
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "PRETRAIN_LEARNING_RATE: $PRETRAIN_LEARNING_RATE"
echo "FINETUNE_LEARNING_RATE: $FINETUNE_LEARNING_RATE"
echo "PRETRAIN_RANK_BATCH: $PRETRAIN_RANK_BATCH"
echo "FINETUNE_RANK_BATCH: $FINETUNE_RANK_BATCH"
echo "ACC_STEPS: $ACCU_STEPS"
echo "DATA_PATH: $DATA_PATH"
echo "MM_PROJECTOR_CHECKPOINT: $MM_PROJECTOR_CHECKPOINT"

# Step 1: Record the current time
time1=$(date +%s)

# Install necessary Python dependencies
pip install imgaug
pip install openpyxl
pip install --upgrade pip  # Enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

# Update PATH for local binaries
export PATH="/home/user/.local/bin:$PATH"

# Set WandB to offline mode
wandb offline

# Set checkpoint directories
CKPT=llava-qwen-uhd-stage15
OUTPUT_DIR=/model_weight/$EXPERIMENT_NAME/$CKPT
LLM_CKPT_DIR=/model_weight/Qwen2-7B-Instruct
CLIP_CKPT_DIR=/model_weight/huggingface/hub/models--openai--clip-vit-large-patch14-336

mkdir -p $OUTPUT_DIR
echo "Output directory: $OUTPUT_DIR"

# Log environment variables
env

# Prepare fine-tuning data
echo "Current working directory: $PWD"
cp /data/dataset/llava-recap-558/images.tar $PWD
tar -xvf ./images.tar >/dev/null 2>&1

training_weights="mm_vision_tower,mm_mlp_adapter,mm_language_model"
# Run the fine-tuning script with torchrun and distributed training
ACCELERATE_CPU_AFFINITY=1 torchrun $DISTRIBUTED_ARGS llava/train/train_mem.py \
    --deepspeed $PWD/scripts/zero2_old.json \
    --model_name_or_path ${LLM_CKPT_DIR} \
    --version qwen_1_5 \
    --model_mode 'uhd_v1' \
    --data_path ${DATA_PATH} \
    --image_folder ./ \
    --vision_tower ${CLIP_CKPT_DIR} \
    --pretrain_mm_mlp_adapter /model_weight/llava-qwen2-uhd-projector/mm_projector.bin \
    --mm_projector_type adapt_spatial_resampler_v1 \
    --mm_tunable_parts ${training_weights} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${FINETUNE_RANK_BATCH} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${ACCU_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 2 \
    --learning_rate ${FINETUNE_LEARNING_RATE} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --run_name ${EXPERIMENT_NAME} \
    --attn_implementation flash_attention_2 \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --single False

# Archive WandB logs
cp -r ./wandb /data/log/$EXPERIMENT_NAME

# Log end of the script
echo "Job finished successfully."