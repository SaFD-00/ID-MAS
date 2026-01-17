#!/bin/bash
# Filter reasoning paths for ReGenesis
#
# Usage:
#   ./scripts/filter_data.sh <model_name> [dataset] [answer_type]
#
# Examples:
#   ./scripts/filter_data.sh meta-llama/Llama-3.1-8B-Instruct gsm8k numeric
#   ./scripts/filter_data.sh meta-llama/Llama-3.1-8B-Instruct reclor option

set -e

# Default values
MODEL_NAME=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
DATASET=${2:-"gsm8k"}
ANSWER_TYPE=${3:-"numeric"}
MAX_PATHS=${4:-5}

# Extract model short name
MODEL_SHORT=$(echo $MODEL_NAME | rev | cut -d'/' -f1 | rev)

INPUT_DIR="data/generated/${MODEL_SHORT}"
OUTPUT_DIR="data/filtered/${MODEL_SHORT}"

echo "=========================================="
echo "ReGenesis Data Filtering"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Model Short: $MODEL_SHORT"
echo "Dataset: $DATASET"
echo "Answer Type: $ANSWER_TYPE"
echo "Max Paths: $MAX_PATHS"
echo "Input Dir: $INPUT_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="

python -m src.pipeline.filtering \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --dataset "$DATASET" \
    --answer_type "$ANSWER_TYPE" \
    --max_paths "$MAX_PATHS"

echo "=========================================="
echo "Filtering completed!"
echo "=========================================="
