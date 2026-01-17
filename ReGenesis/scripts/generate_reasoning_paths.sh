#!/bin/bash
# Generate reasoning paths for ReGenesis
#
# Usage:
#   ./scripts/generate_reasoning_paths.sh <model_name> [dataset] [start_idx] [end_idx]
#
# Examples:
#   ./scripts/generate_reasoning_paths.sh meta-llama/Llama-3.1-8B-Instruct
#   ./scripts/generate_reasoning_paths.sh meta-llama/Llama-3.1-8B-Instruct gsm8k 0 1000
#   ./scripts/generate_reasoning_paths.sh Qwen/Qwen2.5-7B-Instruct math

set -e

# Default values
MODEL_NAME=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
DATASET=${2:-"gsm8k"}
START_IDX=${3:-0}
END_IDX=${4:-}
OUTPUT_DIR=${5:-"data/generated"}
NUM_SAMPLES=${6:-20}

echo "=========================================="
echo "ReGenesis Reasoning Path Generation"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Start Index: $START_IDX"
echo "End Index: ${END_IDX:-'all'}"
echo "Output Dir: $OUTPUT_DIR"
echo "Num Samples: $NUM_SAMPLES"
echo "=========================================="

# Build command
CMD="python -m src.reasoning.multi_model_reasoning_gen \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR \
    --start_idx $START_IDX \
    --num_samples $NUM_SAMPLES"

# Add end_idx if specified
if [ -n "$END_IDX" ]; then
    CMD="$CMD --end_idx $END_IDX"
fi

echo "Running: $CMD"
$CMD

echo "=========================================="
echo "Generation completed!"
echo "=========================================="
