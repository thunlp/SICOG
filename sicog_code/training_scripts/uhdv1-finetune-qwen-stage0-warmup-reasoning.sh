#!/bin/bash

while getopts 'n:j:s:p:f:x:y:z:d:' OPTION; do
  case "$OPTION" in
    n)
      JOB_NAME=$OPTARG ;;
    j)
      JBU_CKPT=$OPTARG ;;
    s)
      SCALE=$OPTARG ;;
    p)
      PRETRAIN_LR=$OPTARG ;;
    f)
      FINETUNE_LR=$OPTARG ;;
    x)
      PRBATCH=$OPTARG ;;
    y)
      FTRBATCH=$OPTARG ;;
    z)
      ACCU_STEPS=$OPTARG ;;
    d)
      DATA_PATH=$OPTARG ;;
    ?)
      echo "Invalid option: -$OPTARG"
      exit 1
      ;;
  esac
done

echo "JOB_NAME: $JOB_NAME"
echo "JBU_CKPT: $JBU_CKPT"
echo "SCALE: $SCALE"
echo "PRETRAIN_LR: $PRETRAIN_LR"
echo "FINETUNE_LR: $FINETUNE_LR"
echo "PRBATCH: $PRBATCH"
echo "FTRBATCH: $FTRBATCH"
echo "ACC_STEPS: $ACCU_STEPS"
echo "DATA_PATH: $DATA_PATH"

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

export PATH="/home/jeeves/.local/bin:$PATH"
wandb offline

CKPT=llava-qwen-uhd-stage0-reasoning
OUTPUT_DIR=/data/checkpoints/xyzhang/$JOB_NAME/$CKPT
LLM_CKPT_DIR=/mnt/data/user/tc_agi/zyp/llavafeatup/model_checkpoints/Qwen2-7B-Instruct
CLIP_CKPT_DIR=/mnt/data/user/tc_agi/zyp/featup/models/huggingface/hub/models--openai--clip-vit-large-patch14-336
PREV_STAGE_CHECKPOINT="/mnt/data/user/tc_agi/xyzhang/personal_model/llava-qwen2-uhd-144-7b"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
env

# Prepare fine-tuning data
echo "Current working directory: $PWD"

cp -r /data/checkpoints/xyzhang/image.zip $PWD
unzip image.zip >/dev/null 2>&1


ACCELERATE_CPU_AFFINITY=1 torchrun $DISTRIBUTED_ARGS llava/train/train_mem.py \
    --deepspeed $PWD/scripts/zero2_old.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version qwen_1_5 \
    --model_mode='uhd_v1' \
    --data_path $DATA_PATH \
    --image_folder ./ \
    --vision_tower $CLIP_CKPT_DIR \
    --mm_projector_type adapt_spatial_resampler_v1 \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $FTRBATCH \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $ACCU_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 2 \
    --learning_rate $FINETUNE_LR \
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
    --run_name $JOB_NAME \
    --attn_implementation flash_attention_2 \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --single False

#archive
cp -r ./wandb /data/checkpoints/xyzhang/$JOB_NAME

# Log end of the script
echo "Job finished successfully."