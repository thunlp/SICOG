#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# Configuration parameters
N=6
MAX_TOKENS=1024
PROMPT=default
ENCODER_MODEL_NAME_OR_PATH=/mnt/data/user/tc_agi/zyp/models/nv/nvidia/NV-Embed-v2
SCORE=minimum_bayes_risk_score_sentence_embedding

# RAW_DATASET=/mnt/data/user/tc_agi/xyzhang-blackbox/uhd-sl-reason_data/eval_cot_2_direct_1_split1_164785.json
# RAW_DATASET=/mnt/data/user/tc_agi/xyzhang-blackbox/low_reso/boostrapped_data/reason/mix-recap-70k-reason-70k-low-reso-stage0-cot-temp07-topp095/eval_cot_2_direct_1_split1_164832.json
# RAW_DATASET=/mnt/data/user/tc_agi/xyzhang-blackbox/iter2_data/reason/combined_iter2_iter1_cot_4_direct_2_len_164785.json
# RAW_DATASET=/mnt/data/user/tc_agi/xyzhang-blackbox/iter3_data/combined_iter123_cot_6_direct_3_len_164785.json
RAW_DATASET=/mnt/data/user/tc_agi/xyzhang-blackbox/uhd_llama_reason_data/eval_cot_2_direct_1_164785.json
MODEL_NAME_OR_PATH=llava_llama_uhd_reason
# Directory path for results
DIR_PATH=/data/xyzhang/caption_results/uhd_llama_reason_165k/mix_cot2_direct1/${PROMPT}_${TEMPERATURE}_${N}/$MODEL_NAME_OR_PATH

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
CUDA_VISIBLE_DEVICES=0,1,6,7 accelerate launch --multi-gpu --mixed_precision=fp16 --num_processes 4 reasoning_quality_eval.py \
    --sample_dataset $RAW_DATASET \
    --dataset_output_path $DATASET_OUTPUT_PATH \
    --filtered_output_path $FILTERED_OUTPUT_PATH \
    --score_dataset_output_path $SCORE_DATASET_OUTPUT_PATH \
    --output_path $DIR_PATH/$SCORE/results.json \
    --encoder_model_name_or_path $ENCODER_MODEL_NAME_OR_PATH