#!/bin/bash
#parse options
while getopts 'n:j:p:f:x:y:z:c:' OPTION; do
  case "$OPTION" in
    n)
      JOB_NAME=$OPTARG ;;
    j)
      JBU_CKPT=$OPTARG ;;
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
    c)
      PRECKPT=$OPTARG ;;
    ?)
      echo "Invalid option: -$OPTARG"
      exit 1
      ;;
  esac
done

echo "JOB_NAME: $JOB_NAME"
echo "JBU_CKPT: $JBU_CKPT"
echo "PRETRAIN_LR: $PRETRAIN_LR"
echo "FINETUNE_LR: $FINETUNE_LR"
echo "PRBATCH: $PRBATCH"
echo "FTRBATCH: $FTRBATCH"
echo "PRECKPT: $PRECKPT"
echo "ACCU_STEPS: $ACCU_STEPS"

wandb offline

CKPT=llava-qwen-uhd-stage2
OUTPUT_DIR=/data/checkpoints/$JOB_NAME/checkpoints_new/$CKPT
mkdir -p $OUTPUT_DIR
LLM_CKPT_DIR=$PRECKPT
CLIP_CKPT_DIR=./pretrained_models/clip-vit-large-patch14-336

echo "Output directory: $OUTPUT_DIR"

GPUS_PER_NODE=${GPUS_PER_NODE:-8}

WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-12345}

DISTRIBUTED_ARGS="
  --nproc_per_node $GPUS_PER_NODE \
  --nnodes $WORLD_SIZE \
  --node_rank $RANK \
  --master_addr $MASTER_ENDPOINT \
  --master_port $MASTER_PORT "

echo $DISTRIBUTED_ARGS

ACCELERATE_CPU_AFFINITY=1 torchrun $DISTRIBUTED_ARGS llava/train/train_mem.py \
    --deepspeed $PWD/scripts/zero2_old.json \
    --model_name_or_path $LLM_CKPT_DIR \
    --version llava_llama_3 \
    --model_mode='uhd_v1' \
    --data_path $PWD/playground/data/LLaVA-SFT/llava_new_replace_text-new.json \
    --image_folder $PWD/playground/data/LLaVA-SFT/llava_new \
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
    --save_steps 4000 \
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
