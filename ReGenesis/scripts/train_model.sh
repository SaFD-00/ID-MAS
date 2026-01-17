#!/bin/bash
# Train model with ReGenesis
#
# Usage:
#   ./scripts/train_model.sh <model_name> <train_data> [output_dir] [use_lora] [num_gpus]
#
# Examples:
#   ./scripts/train_model.sh meta-llama/Llama-3.1-8B-Instruct data/filtered/train.json
#   ./scripts/train_model.sh meta-llama/Llama-3.1-70B-Instruct data/filtered/train.json checkpoints/llama-70b true 4

set -e

# Default values
MODEL_NAME=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
TRAIN_DATA=${2:-"data/filtered/train.json"}
OUTPUT_DIR=${3:-}
USE_LORA=${4:-"false"}
NUM_GPUS=${5:-1}

# Extract model short name for default output dir
MODEL_SHORT=$(echo $MODEL_NAME | rev | cut -d'/' -f1 | rev)
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/${MODEL_SHORT}"}

echo "=========================================="
echo "ReGenesis Model Training"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Train Data: $TRAIN_DATA"
echo "Output Dir: $OUTPUT_DIR"
echo "Use LoRA: $USE_LORA"
echo "Num GPUs: $NUM_GPUS"
echo "=========================================="

# Build base command
BASE_CMD="python -m src.finetune_code.multi_model_finetune \
    --model_name_or_path $MODEL_NAME \
    --data_path $TRAIN_DATA \
    --output_dir $OUTPUT_DIR \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4"

# Add LoRA flags
if [ "$USE_LORA" = "true" ]; then
    BASE_CMD="$BASE_CMD --use_lora --load_in_4bit"
fi

# Run with torchrun for multi-GPU
if [ "$NUM_GPUS" -gt 1 ]; then
    CMD="torchrun --nproc_per_node=$NUM_GPUS $BASE_CMD"
else
    CMD="$BASE_CMD"
fi

echo "Running: $CMD"
eval $CMD

echo "=========================================="
echo "Training completed!"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
