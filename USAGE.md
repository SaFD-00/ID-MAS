# 서버 코드
cd project/LLaMA-Factory
conda activate llama-factory

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=1 API_PORT=2000 llamafactory-cli api examples/inference_custom/llama3_1_8b_instruct.yaml
DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0 API_PORT=2000 llamafactory-cli api examples/inference_custom/qwen2_5_7b_instruct.yaml
DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0 API_PORT=2000 llamafactory-cli api examples/inference_custom/qwen3_4b_instruct.yaml

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
    --student-model Qwen/Qwen2.5-7B-Instruct \
    --teacher-model Qwen/Qwen2.5-7B-Instruct

CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain math --train-dataset math \
    --student-model Qwen/Qwen3-4B-Instruct-2507 \
    --teacher-model Qwen/Qwen3-4B-Instruct-2507