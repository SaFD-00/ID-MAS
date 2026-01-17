#!/bin/bash
# Run full ReGenesis pipeline for a model
#
# Usage:
#   ./scripts/run_full_pipeline.sh <model_name> [datasets]
#
# Examples:
#   ./scripts/run_full_pipeline.sh meta-llama/Llama-3.1-8B-Instruct
#   ./scripts/run_full_pipeline.sh Qwen/Qwen2.5-7B-Instruct "gsm8k math"

set -e

# Default values
MODEL_NAME=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
DATASETS=${2:-"gsm8k math reclor arc_c"}
OUTPUT_DIR=${3:-"data"}
NUM_SAMPLES=${4:-20}

echo "=========================================="
echo "ReGenesis Full Pipeline"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Datasets: $DATASETS"
echo "Output Dir: $OUTPUT_DIR"
echo "Num Samples: $NUM_SAMPLES"
echo "=========================================="

python -m src.pipeline.training_pipeline \
    --model_name "$MODEL_NAME" \
    --datasets $DATASETS \
    --output_dir "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES

echo "=========================================="
echo "Pipeline completed!"
echo "=========================================="
