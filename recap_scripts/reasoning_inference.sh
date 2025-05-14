#!/bin/bash

# Print configuration for debugging
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "MODEL_PATH: $MODEL_PATH"
echo "DATA_PATH: $DATA_PATH"
echo "SPLIT: $SPLIT"
echo "TEMPERATURE: ${TEMPERATURE:-1.0}"
echo "BEAMS: ${BEAMS:-1}"
echo "TOP_P: ${TOP_P:-0.9}"
echo "PROMPT: ${PROMPT:-default}"

# Install dependencies
echo "Installing dependencies..."
pip install -q imgaug openpyxl
pip install -q --upgrade pip
pip install -q -e .[train]
pip install -q accelerate flash-attn --no-build-isolation
sudo apt-get install zip unzip

# Set up environment variables
export PATH="/home/user/.local/bin:$PATH"
export PYTHONPATH=$PWD:$PYTHONPATH

# Create output directory
OUTPUT_DIR="/data/datasets/$EXPERIMENT_NAME"
mkdir -p "$OUTPUT_DIR"

# Set up GPU list
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
CHUNKS=${#GPULIST[@]}
echo "Number of GPUs (chunks): $CHUNKS"

# Create dataset directory
DATASET_DIR="./datasets/uhdv1-reasoning-164"
mkdir -p "$DATASET_DIR"

# Extract images
echo "Extracting images..."
cp -r /data/datasets/image.zip $PWD # your image.zip path
unzip image.zip >/dev/null 2>&1

# Run inference on each GPU chunk
for IDX in $(seq 0 $((CHUNKS - 1))); do
  RESULT_FILE="$DATASET_DIR/sl_reasoning_164k_${SPLIT}_${IDX}.json"

  # Clear the result file for the current chunk
  > "$RESULT_FILE"

  echo "Processing chunk $IDX on GPU ${GPULIST[$IDX]}..."
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} accelerate launch --mixed_precision "bf16" \
    --dynamo_backend no \
    --num_processes 80 \
    --num_machines 1 \
    --main_process_port 24000 \
    llava/inference/recap_558k_reasoning.py \
    --model-path ${MODEL_PATH} \
    --image-folder ./ \
    --annotation-file "${DATA_PATH}_${SPLIT}.json" \
    --result-file ${RESULT_FILE} \
    --max-new-tokens 1200 \
    --num-chunks ${CHUNKS} \
    --chunk-idx ${IDX} \
    --temperature ${TEMPERATURE:-1.0} \
    --num-beams ${BEAMS:-1} \
    --conv-mode 'qwen_1_5' \
    --default-prompt ${PROMPT:-default} \
    --top_p ${TOP_P:-0.9} &
done

# Wait for all processes to finish
wait

# Merge results into a single JSON file
MERGED_OUTPUT_FILE="$DATASET_DIR/sl_reasoning_split${SPLIT}_t${TEMPERATURE}_b${BEAMS}_p${TOP_P}_${PROMPT}.json"
echo "Merging results into $MERGED_OUTPUT_FILE..."

python3 - <<EOF
import json
import os

merged_data = []
for idx in range(${CHUNKS}):
    file_path = f"./datasets/uhdv1-reasoning-164/sl_reasoning_164k_${SPLIT}_{idx}.json"
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                merged_data.append(data)
        os.remove(file_path)  # Clean up individual chunk files

with open("${MERGED_OUTPUT_FILE}", 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=2)
EOF

# Copy the merged output file to the final output directory
cp -r "$MERGED_OUTPUT_FILE" "$OUTPUT_DIR"

echo "Inference completed successfully. Results saved to $MERGED_OUTPUT_FILE."
