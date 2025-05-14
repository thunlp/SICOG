#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Configuration parameters
N=6
MAX_TOKENS=1024
PROMPT=default
ENCODER_MODEL_NAME_OR_PATH=/model_weight/NV-Embed-v2
SCORE=minimum_bayes_risk_score_sentence_embedding

echo "RAW_DATASET: $RAW_DATASET"
echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH"

DIR_PATH=/data/datasets/caption_results/caption/${PROMPT}_${TEMPERATURE}_${N}/$MODEL_NAME_OR_PATH

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