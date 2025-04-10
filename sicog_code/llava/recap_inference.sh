# !/bin/bash
# export PYTHONPATH="/home/jeeves/xyzhang/mm_self_learning/llava-uhd2:$PYTHONPATH"
# # Create a directory for experiment results
# EXP_RESULTS_DIR="./exp_results"
# mkdir -p "$EXP_RESULTS_DIR"
# echo "Created directory: $EXP_RESULTS_DIR"

# # Define variables
# GPU_ID=0
# MODEL_NAME=/mnt/data/user/tc_agi/xyzhang/download_models/Lin-Chen/open-llava-next-llama3-8b
# OUTPUT_DIR="./logs"
# WORKER_PORT=24000
# WORKING_ADDRESS="http://localhost:${WORKER_PORT}/v1"

# # # Ensure the log directory exists
# # mkdir -p "$OUTPUT_DIR"

# # # Start the VLLM API server
# # echo "Starting VLLM API server..."
# # CUDA_VISIBLE_DEVICES=$GPU_ID python3 -u -m vllm.entrypoints.openai.api_server \
# #     --model "$MODEL_NAME" \
# #     --port "$WORKER_PORT" \
# #     --gpu-memory-utilization 0.98 \
# #     --tensor-parallel-size 2 \
# #     > "$OUTPUT_DIR/vllm_log_llava-uhd-qwen2-instruct_$GPU_ID.log" 2>&1 &

# # # Start serving the Qwen2-VL-7B-Instruct model
# # echo "Serving Qwen2-VL-7B-Instruct model..."
# CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve $MODEL_NAME \
#     --served-model-name "Llava-next-llama3-8b" \
#     --port 54188 \
#     --host 0.0.0.0 \
#     --trust-remote-code \
#     --gpu-memory-utilization 0.98 \
#     --tensor-parallel-size 2
# export PYTHONPATH=$PYTHONPATH:/home/jeeves/xyzhang/mm_self_learning/llava-uhd2/llava
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 recap_558k_multi_qwen.py \
#     --num-processes 1 \
#     --model-path "/mnt/data/user/tc_agi/zyp-blackbox/uhdv1-onevision-clip-2e-5-qwen-2e-5-uhddata-pretrain2e-4-m4096/checkpoints_new/llava-uhd-144-7b" \
#     --annotation-file /home/jeeves/xyzhang/ALLaVA/download/allava_vflan/ALLaVA-Caption-VFLAN-4V.json \
#     --image-folder /home/jeeves/xyzhang/ALLaVA/download/allava_vflan/ \
#     --result-file ./test.json \
#     --temperature 0.9 \
#     --top_p 0.9 

#!/bin/bash

# Ensure CUDA_VISIBLE_DEVICES is set with a fallback to "0"
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Parse GPU list into an array
IFS=',' read -ra GPULIST <<< "$gpu_list"

# Calculate the number of chunks based on the GPU list
CHUNKS=${#GPULIST[@]}

# Ensure the checkpoint (CKPT) argument is provided
if [ -z "$1" ]; then
    echo "Error: Checkpoint argument (CKPT) is required."
    echo "Usage: $0 <checkpoint-name>"
    exit 1
fi

CKPT=$1
SPLIT="llava_mme"

# Print configuration for verification
echo "Checkpoint: $CKPT"
echo "Split: $SPLIT"
echo "GPU List: ${GPULIST[@]}"
echo "Number of Chunks: $CHUNKS"

# Output directory for logs
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"  # Ensure the log directory exists

# Loop through GPUs and process chunks in parallel
for IDX in $(seq 0 $((CHUNKS - 1))); do
    # Define result file for the current chunk
    RESULT_FILE="/data/xyzhang/caption_results/tuned_${CKPT}_${SPLIT}_${IDX}.json"

    # Print current chunk processing status
    echo "Processing chunk $IDX on GPU ${GPULIST[$IDX]}"
    echo "Result file: $RESULT_FILE"

    # Launch the inference script
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} accelerate launch --mixed_precision "bf16" \
        --dynamo_backend no \
        --num_processes 64 \
        --num_machines 1 \
        --main_process_port 24000 \
        inference/recap_558k_new.py \
        --model-path "/mnt/data/user/tc_agi/xyzhang/personal_model/llava-qwen2-uhd-144-7b" \
        --annotation-file "/data/xyzhang/ALLaVA/download/allava_vflan/ALLaVA-Caption-VFLAN-4V_split_test500_valid.json" \
        --image-folder "/data/xyzhang/ALLaVA/download/allava_vflan" \
        --result-file "$RESULT_FILE" \
        --temperature 0.9 \
        --top_p 0.9 \
        --chunk-idx "$IDX" \
        --num-chunks "$CHUNKS" \
        >"${LOG_DIR}/chunk_${IDX}.out" 2>"${LOG_DIR}/chunk_${IDX}.err" &  # Save logs for each chunk
done

# Wait for all background processes to finish
wait

echo "All chunks processed successfully."
# ## single gpu inference
# CUDA_VISIBLE_DEVICES=0 accelerate launch --dynamo_backend "no" --num_processes 4 --num_machines 1 --main_process_port 24000 inference/recap_558k_new.py \
#         --model-path "/mnt/data/user/tc_agi/xyzhang/personal_model/llava-qwen2-uhd-144-7b" \
#         --annotation-file /data/xyzhang/ALLaVA/download/allava_vflan/ALLaVA-Caption-VFLAN-4V_split_test500_valid.json \
#         --image-folder /data/xyzhang/ALLaVA/download/allava_vflan \
#         --result-file /data/xyzhang/caption_results/tuned1000.json \
#         --temperature 0.9 \
#         --top_p 0.9




