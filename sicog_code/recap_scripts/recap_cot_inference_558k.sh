#!/bin/bash


# bash recap_cot_inference.sh -m /mnt/data/user/tc_agi/xyzhang-blackbox/self-caption-capstage0-35k-0119/checkpoints/llava-uhd-caption-train1000 -n recap -f /mnt/data/user/tc_agi/pd/datasets/llava-recap-558/llava_recap_558k_118k -s 4 -t 1 -b 1 -p 0.9 -q "default"

# Export HDFS credentials
# Parse command-line options
while getopts 'n:m:f:s:t:b:p:q:' OPTION; do
  case "$OPTION" in
    n)
      JOB_NAME=$OPTARG ;;
    m)
      MODEL_PATH=$OPTARG ;;
    f)
      ANNOTATION_FILE=$OPTARG ;;
    s)
      SPLIT=$OPTARG ;;
    t)
      TEMPERATURE=$OPTARG ;;
    b)
      BEAMS=$OPTARG ;;
    p)
      TOP_P=$OPTARG ;;
    q)
      PROMPT=$OPTARG ;;
    ?)
      echo "Invalid option: -$OPTARG"
      exit 1
      ;;
  esac
done

# Ensure required parameters are provided
if [ -z "$JOB_NAME" ] || [ -z "$MODEL_PATH" ] || [ -z "$ANNOTATION_FILE" ] || [ -z "$SPLIT" ]; then
  echo "Error: Missing required parameters."
  echo "Usage: $0 -n JOB_NAME -m MODEL_PATH -f ANNOTATION_FILE -s SPLIT [-t TEMPERATURE] [-b BEAMS] [-p TOP_P] [-q PROMPT]"
  exit 1
fi

# Print configuration for debugging
echo "JOB_NAME: $JOB_NAME"
echo "MODEL_PATH: $MODEL_PATH"
echo "ANNOTATION_FILE: $ANNOTATION_FILE"
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

# Set up environment variables
export PATH="/home/jeeves/.local/bin:$PATH"
export PYTHONPATH=$PWD:$PYTHONPATH

# Create output directory
OUTPUT_DIR="/data/checkpoints/xyzhang/$JOB_NAME"
mkdir -p "$OUTPUT_DIR"

# Set up GPU list
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
CHUNKS=${#GPULIST[@]}
echo "Number of GPUs (chunks): $CHUNKS"

# Create dataset directory
DATASET_DIR="./datasets/uhdv1-recap-118"
mkdir -p "$DATASET_DIR"

# Extract images
echo "Extracting images..."
cp /mnt/data/user/tc_agi/zyp/datasets/llava-recap-558/images.tar ./
tar -xf images.tar >/dev/null 2>&1 || { echo "Error: Failed to extract images.tar"; exit 1; }
rm -f images.tar  # Clean up tar file after extraction

# Run inference on each GPU chunk
for IDX in $(seq 0 $((CHUNKS - 1))); do
  RESULT_FILE="$DATASET_DIR/recap_558k_total.json"

  # Clear the result file for the current chunk
  > "$RESULT_FILE"

  echo "Processing chunk $IDX on GPU ${GPULIST[$IDX]}..."
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} accelerate launch --mixed_precision "bf16" \
    --dynamo_backend no \
    --num_processes 64 \
    --num_machines 1 \
    --main_process_port 24000 \
    llava/inference/recap_558k_new.py \
    --model-path "$MODEL_PATH" \
    --image-folder ./ \
    --annotation-file "/mnt/data/user/tc_agi/pd/datasets/llava-recap-558/recap_558k.json" \
    --result-file "$RESULT_FILE" \
    --max-new-tokens 1200 \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature "${TEMPERATURE:-1.0}" \
    --num-beams "${BEAMS:-1}" \
    --conv-mode 'qwen_1_5' \
    --default-prompt "${PROMPT:-default}" \
    --top_p "${TOP_P:-0.9}" &
done

# Wait for all processes to finish
wait

# Merge results into a single JSON file
MERGED_OUTPUT_FILE="$DATASET_DIR/recap_cot_558k_118k_split${SPLIT}_t${TEMPERATURE}_b${BEAMS}_p${TOP_P}_${PROMPT}.json"
echo "Merging results into $MERGED_OUTPUT_FILE..."

python3 - <<EOF
import json
import os

merged_data = []
for idx in range(${CHUNKS}):
    file_path = f"./datasets/uhdv1-recap-118/recap_558k_118k_${SPLIT}_{idx}.json"
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
