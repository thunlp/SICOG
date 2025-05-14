#!/bin/bash

echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "PRETRAIN_LEARNING_RATE: $PRETRAIN_LEARNING_RATE"
echo "FINETUNE_LEARNING_RATE: $FINETUNE_LEARNING_RATE"
echo "PRETRAIN_BATCH: $PRETRAIN_BATCH"
echo "FINETUNE_BATCH: $FINETUNE_BATCH"
echo "ACC_STEPS: $ACCU_STEPS"
echo "DATA_PATH: $DATA_PATH"
echo "PREV_STAGE_CHECKPOINT: $PREV_STAGE_CHECKPOINT"

# Step 1: Record current time as time1
time1=$(date +%s)

#install
pip install imgaug
pip install openpyxl

pip install --upgrade pip  # enable PEP 660 support
pip install -e .

pip install -e ".[train]"
pip install flash-attn --no-build-isolation
sudo apt-get install zip unzip

export PATH="/home/user/.local/bin:$PATH"
wandb offline

CKPT=llava-qwen-uhd-stage0
OUTPUT_DIR=/model_weight/$EXPERIMENT_NAME/$CKPT
LLM_CKPT_DIR=/model_weight/Qwen2-7B-Instruct
CLIP_CKPT_DIR=/model_weight/huggingface/hub/models--openai--clip-vit-large-patch14-336
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo $OUTPUT_DIR

mkdir -p $OUTPUT_DIR

# Prepare fine-tuning data

cp -r /data/dataset/image.zip $PWD
unzip image.zip >/dev/null 2>&1


training_weights="mm_vision_tower,mm_mlp_adapter,mm_language_model"
ACCELERATE_CPU_AFFINITY=1 torchrun $DISTRIBUTED_ARGS llava/train/train_mem.py \
    --deepspeed $PWD/scripts/zero2_old.json \
    --model_name_or_path ${PREV_STAGE_CHECKPOINT} \
    --version qwen_1_5 \
    --model_mode 'uhd_v1' \
    --data_path ${DATA_PATH} \
    --image_folder ./ \
    --vision_tower ${CLIP_CKPT_DIR} \
    --mm_projector_type adapt_spatial_resampler_v1 \
    --mm_tunable_parts ${training_weights} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${FINETUNE_BATCH} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${ACCU_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
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

#archive
cp -r ./wandb /data/log/$EXPERIMENT_NAME

# Log end of the script
echo "Job finished successfully."