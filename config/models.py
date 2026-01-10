"""
Teacher and student model configurations.
"""
from config.api import OPENAI_API_KEY

# =============================================================================
# Teacher Model Configuration
# =============================================================================

# 지원하는 Teacher 모델 목록
AVAILABLE_TEACHER_MODELS = [
    # OpenAI
    "gpt-5-2025-08-07",
    # LLaMA-Factory API (OpenAI-compatible)
    "openai/gpt-oss-20b",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

# 기본 Teacher 모델
DEFAULT_TEACHER_MODEL = "gpt-5-2025-08-07"


def create_teacher_config(model_name: str = None) -> dict:
    """
    Teacher model config 생성

    Args:
        model_name: Teacher 모델 이름 (None이면 기본 모델 사용)

    Returns:
        Teacher model 설정 딕셔너리
    """
    if model_name is None:
        model_name = DEFAULT_TEACHER_MODEL

    # OpenAI 모델 (gpt-로 시작)
    if model_name.startswith("gpt-"):
        return {
            "model": model_name,
            "base_url": None,  # OpenAI 기본 endpoint
            "api_key": OPENAI_API_KEY,
            "reasoning": {"effort": "medium"},
            "text": {"verbosity": "medium"},
            "max_tokens": 8192
        }
    # LLaMA-Factory API 모델 (OpenAI-compatible endpoint)
    else:
        import os
        base_url = os.getenv("LLAMA_FACTORY_BASE_URL", "http://localhost:2000/v1")
        return {
            "model": model_name,
            "base_url": base_url,
            "api_key": "0",
            "max_tokens": 8192
        }


# =============================================================================
# Student Model Configuration
# =============================================================================

# 지원하는 학생 모델 목록
AVAILABLE_STUDENT_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

# 기본 학생 모델
DEFAULT_STUDENT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# 학생 모델 공통 설정
STUDENT_MODEL_BASE_CONFIG = {
    "device": "cuda",
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "do_sample": True
}


def get_student_model_config(model_name: str = None) -> dict:
    """
    학생 모델 설정 생성

    Args:
        model_name: 모델 이름 (None이면 기본 모델 사용)

    Returns:
        모델 설정 딕셔너리
    """
    if model_name is None:
        model_name = DEFAULT_STUDENT_MODEL

    if model_name not in AVAILABLE_STUDENT_MODELS:
        raise ValueError(
            f"지원하지 않는 모델입니다: {model_name}\n"
            f"지원 모델: {AVAILABLE_STUDENT_MODELS}"
        )

    config = STUDENT_MODEL_BASE_CONFIG.copy()
    config["model_name"] = model_name

    return config


def get_model_short_name(model_name: str = None) -> str:
    """
    모델의 짧은 이름 반환 (폴더명용)

    Args:
        model_name: 전체 모델 이름 (예: "Qwen/Qwen3-4B-Instruct-2507")

    Returns:
        짧은 모델 이름 (예: "Qwen3-4B-Instruct-2507")
    """
    if model_name is None:
        model_name = DEFAULT_STUDENT_MODEL

    # "Qwen/Qwen3-4B-Instruct-2507" → "Qwen3-4B-Instruct-2507"
    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name
