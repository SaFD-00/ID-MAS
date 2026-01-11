# 학습 코드
cd project/ID-MAS
conda activate ID-MAS

CUDA_VISIBLE_DEVICES=0 python main.py --mode train --domain math --train-dataset math \
    --student-model meta-llama/Llama-3.1-8B-Instruct \
    --teacher-model meta-llama/Llama-3.1-8B-Instruct

CUDA_VISIBLE_DEVICES=0 python main.py --mode train --domain math --train-dataset math \
    --student-model meta-llama/Llama-3.2-3B-Instruct \
    --teacher-model meta-llama/Llama-3.2-3B-Instruct

CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain math --train-dataset math \
    --student-model Qwen/Qwen2.5-3B-Instruct \
    --teacher-model Qwen/Qwen2.5-3B-Instruct

CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain math --train-dataset math \
    --student-model Qwen/Qwen2.5-7B-Instruct \
    --teacher-model Qwen/Qwen2.5-7B-Instruct

CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain math --train-dataset math \
    --student-model Qwen/Qwen2.5-14B-Instruct \
    --teacher-model Qwen/Qwen2.5-14B-Instruct

CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain math --train-dataset math \
    --student-model Qwen/Qwen3-4B-Instruct-2507 \
    --teacher-model Qwen/Qwen3-4B-Instruct-2507