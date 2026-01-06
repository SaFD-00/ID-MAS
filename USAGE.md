# 서버 코드
cd project/LLaMA-Factory
conda activate llama-factory

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0 API_PORT=2000 llamafactory-cli api examples/inference_custom/qwen3_30b_a3b_instruct_fp8.yaml


# 학습 코드
cd project/ID-MAS
conda activate ID-MAS

CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain math --train-dataset gsm8k \
    --student-model Qwen/Qwen3-4B-Instruct-2507 \
    --teacher-model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8

CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain math --train-dataset math \
    --student-model Qwen/Qwen3-4B-Instruct-2507 \
    --teacher-model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
