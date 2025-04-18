#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# Configuration parameters
N=6
MAX_TOKENS=1024
PROMPT=default
ENCODER_MODEL_NAME_OR_PATH=/mnt/data/user/tc_agi/zyp/models/nv/nvidia/NV-Embed-v2
SCORE=minimum_bayes_risk_score_sentence_embedding

# RAW_DATASET=/mnt/data/user/tc_agi/xyzhang-blackbox/recap_data/mix-70k-short-long/recap-mix-cot-default/recap_cot2_short_1_117983.json
# RAW_DATASET=/mnt/data/user/tc_agi/xyzhang-blackbox/low_reso/boostrapped_data/recaption/mix_cot_default/recap_cot2_short_1_117984.json
# RAW_DATASET=/mnt/data/user/tc_agi/xyzhang-blackbox/low-reso-recap-mix-cot-default-temp07-topp-095-0212/recap_cot2_short_1_117984.json
# RAW_DATASET=/mnt/data/user/tc_agi/xyzhang-blackbox/recap-440k/recap_cot1_short_2_440105.json
# RAW_DATASET=/mnt/data/user/tc_agi/xyzhang-blackbox/high_reso_iter2_mix_step_cot_70k_train_caption/recap_cot2_short_1_117983.json
# RAW_DATASET=/mnt/data/user/tc_agi/xyzhang-blackbox/low-reso-dpo-recap/recap_cot2_short_1_117984.json
# RAW_DATASET=/mnt/data/user/tc_agi/xyzhang-blackbox/recap_allava/recap_cot2_short_1_148590.json
# RAW_DATASET=/mnt/data/user/tc_agi/xyzhang-blackbox/iter2_data/allava_recap/combined_recap_cot2_short_1_148590.json
RAW_DATASET=/mnt/data/user/tc_agi/xyzhang-blackbox/uhd_llama_recap_data/combined_recap_cot2_short_1_117983.json
MODEL_NAME_OR_PATH=llava_qwen_uhd_allava_recap
# Directory path for results
# DIR_PATH=/data/xyzhang/caption_results/recap118k_low_reso0214/mix_cot2_direct1/${PROMPT}_${TEMPERATURE}_${N}/$MODEL_NAME_OR_PATH
DIR_PATH=/data/xyzhang/caption_results/llava_llama_uhd_recap118k/mix_cot2_direct1/${PROMPT}_${TEMPERATURE}_${N}/$MODEL_NAME_OR_PATH

# Ensure the output directories exist
mkdir -p $DIR_PATH
mkdir -p $DIR_PATH/$SCORE
mkdir -p $DIR_PATH/predictions
mkdir -p $DIR_PATH/$SCORE/predictions

# Update dataset_output_path to include a filename
DATASET_OUTPUT_PATH=$DIR_PATH/predictions/final_dataset.json
FILTERED_OUTPUT_PATH=$DIR_PATH/predictions/filtered_dataset.json
SCORE_DATASET_OUTPUT_PATH=$DIR_PATH/$SCORE/predictions/scored_dataset.json

# Run the Python script
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi-gpu --mixed_precision=fp16 --num_processes 8 caption_quality_eval.py \
    --sample_dataset $RAW_DATASET \
    --dataset_output_path $DATASET_OUTPUT_PATH \
    --filtered_output_path $FILTERED_OUTPUT_PATH \
    --score_dataset_output_path $SCORE_DATASET_OUTPUT_PATH \
    --output_path $DIR_PATH/$SCORE/results.json \
    --encoder_model_name_or_path $ENCODER_MODEL_NAME_OR_PATH