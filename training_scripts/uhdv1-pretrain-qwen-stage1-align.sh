#!/bin/bash

echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "PRETRAIN_LEARNING_RATE: $PRETRAIN_LEARNING_RATE"
echo "FINETUNE_LEARNING_RATE: $FINETUNE_LEARNING_RATE"
echo "PRETRAIN_BATCH: $PRETRAIN_BATCH"
echo "FINETUNE_RANK_BATCH: $FINETUNE_RANK_BATCH"
echo "DATA_PATH: $DATA_PATH"
echo "ACC_STEPS: $ACCU_STEPS"

#install
pip install imgaug
pip install openpyxl

pip install --upgrade pip  # enable PEP 660 support
pip install -e .

pip install -e ".[train]"
pip install flash-attn --no-build-isolation

export PATH="/home/user/.local/bin:$PATH"
wandb offline

CKPT=llava-qwen-uhd-stage1
OUTPUT_DIR=/model_weight/$EXPERIMENT_NAME/checkpoints_new/$CKPT
mkdir -p $OUTPUT_DIR
LLM_CKPT_DIR=/model_weight/Qwen2-7B-Instruct
CLIP_CKPT_DIR=/model_weight/huggingface/hub/models--openai--clip-vit-large-patch14-336

echo $OUTPUT_DIR

#pretrain data
echo $PWD
mkdir -p $PWD/playground/data
cp -r $DATA_PATH $PWD/playground/data
tar -xvf $PWD/playground/data/LLaVA-Pretrain/images.tar -C $PWD/playground/data/LLaVA-Pretrain/  >/dev/null 2>&1

env

training_weights="mm_mlp_adapter"
ACCELERATE_CPU_AFFINITY=1 torchrun $DISTRIBUTED_ARGS llava/train/train_mem.py \
    --deepspeed $PWD/scripts/zero2_old.json \
    --model_name_or_path ${LLM_CKPT_DIR} \
    --version qwen_1_5 \
    --model_mode 'uhd_v1' \
    --data_path $PWD/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder $PWD/playground/data/LLaVA-Pretrain/images \
    --vision_tower ${CLIP_CKPT_DIR} \
    --mm_projector_type adapt_spatial_resampler_v1 \
    --mm_tunable_parts ${training_weights} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${PRETRAIN_BATCH} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${ACCU_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate ${PRETRAIN_LEARNING_RATE} \
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
    --run_name ${EXPERIMENT_NAME} \
    --attn_implementation flash_attention_2 \
    --single True \
    --dataloader_drop_last True

#archive
cp -r ./wandb /data/log/$EXPERIMENT_NAME
