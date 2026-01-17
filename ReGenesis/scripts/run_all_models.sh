#!/bin/bash
# Run ReGenesis pipeline for all 8 target models
#
# Usage:
#   ./scripts/run_all_models.sh [stage] [datasets]
#
# Stages:
#   all        - Run full pipeline for all models
#   generate   - Only data generation
#   filter     - Only filtering
#   train      - Only training
#
# Examples:
#   ./scripts/run_all_models.sh all
#   ./scripts/run_all_models.sh generate "gsm8k math"
#   ./scripts/run_all_models.sh train

set -e

STAGE=${1:-"all"}
DATASETS=${2:-"gsm8k math reclor arc_c"}
OUTPUT_DIR="data"

# All 8 target models
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.1-70B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
)

# Models that require LoRA (70B+)
LORA_MODELS=(
    "meta-llama/Llama-3.1-70B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
)

echo "=========================================="
echo "ReGenesis Multi-Model Pipeline"
echo "=========================================="
echo "Stage: $STAGE"
echo "Datasets: $DATASETS"
echo "Models: ${#MODELS[@]}"
echo "=========================================="

# Function to check if model needs LoRA
needs_lora() {
    local model=$1
    for lora_model in "${LORA_MODELS[@]}"; do
        if [ "$model" = "$lora_model" ]; then
            return 0
        fi
    done
    return 1
}

# Process each model
for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo $MODEL | rev | cut -d'/' -f1 | rev)

    echo ""
    echo "=========================================="
    echo "Processing: $MODEL"
    echo "=========================================="

    case $STAGE in
        "generate")
            for DATASET in $DATASETS; do
                echo "Generating for $DATASET..."
                ./scripts/generate_reasoning_paths.sh "$MODEL" "$DATASET"
            done
            ;;

        "filter")
            for DATASET in $DATASETS; do
                ANSWER_TYPE="numeric"
                if [ "$DATASET" = "reclor" ] || [ "$DATASET" = "arc_c" ]; then
                    ANSWER_TYPE="option"
                fi
                echo "Filtering $DATASET with answer_type=$ANSWER_TYPE..."
                ./scripts/filter_data.sh "$MODEL" "$DATASET" "$ANSWER_TYPE"
            done
            ;;

        "train")
            USE_LORA="false"
            NUM_GPUS=1

            if needs_lora "$MODEL"; then
                USE_LORA="true"
                NUM_GPUS=4
            fi

            # Check for merged training data
            TRAIN_DATA="$OUTPUT_DIR/filtered/$MODEL_SHORT/merged_train.json"
            if [ ! -f "$TRAIN_DATA" ]; then
                echo "Warning: Merged training data not found at $TRAIN_DATA"
                echo "Looking for individual filtered files..."
                TRAIN_DATA=$(ls $OUTPUT_DIR/filtered/$MODEL_SHORT/*_filtered.json 2>/dev/null | head -1)
            fi

            if [ -n "$TRAIN_DATA" ]; then
                echo "Training with data: $TRAIN_DATA"
                ./scripts/train_model.sh "$MODEL" "$TRAIN_DATA" "" "$USE_LORA" "$NUM_GPUS"
            else
                echo "Error: No training data found for $MODEL"
            fi
            ;;

        "all")
            # Run full pipeline
            EXTRA_ARGS=""
            if needs_lora "$MODEL"; then
                EXTRA_ARGS="--use_lora --load_in_4bit"
            fi

            python -m src.pipeline.training_pipeline \
                --model_name "$MODEL" \
                --datasets $DATASETS \
                --output_dir "$OUTPUT_DIR" \
                $EXTRA_ARGS
            ;;

        *)
            echo "Unknown stage: $STAGE"
            echo "Available stages: all, generate, filter, train"
            exit 1
            ;;
    esac
done

echo ""
echo "=========================================="
echo "All models processed!"
echo "=========================================="
