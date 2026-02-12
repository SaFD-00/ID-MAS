"""Teacher 및 Student 모델 설정 모듈.

이 모듈은 ID-MAS 시스템에서 사용하는 Teacher(교사) 및 Student(학생)
모델의 설정을 관리합니다. OpenAI API 모델과 로컬 HuggingFace 모델을
모두 지원합니다.

주요 상수:
    AVAILABLE_TEACHER_MODELS: 지원하는 교사 모델 목록
    DEFAULT_TEACHER_MODEL: 기본 교사 모델
    AVAILABLE_STUDENT_MODELS: 지원하는 학생 모델 목록
    DEFAULT_STUDENT_MODEL: 기본 학생 모델
    STUDENT_MODEL_BASE_CONFIG: 학생 모델 공통 설정

주요 함수:
    create_teacher_config: 교사 모델 설정 생성
    get_student_model_config: 학생 모델 설정 생성
    get_model_short_name: 모델명의 짧은 버전 반환
"""
from config.api import OPENAI_API_KEY

# =============================================================================
# 교사 모델 설정
# =============================================================================

# 지원하는 교사 모델 목록
AVAILABLE_TEACHER_MODELS = [
    # OpenAI API
    "gpt-5.2",
    # 로컬 HuggingFace 모델 (ModelCache를 통해 직접 로드)
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
]

# 기본 교사 모델
DEFAULT_TEACHER_MODEL = "gpt-5.2"


def create_teacher_config(model_name: str = None, gpu_id: int = None) -> dict:
    """교사 모델 설정 딕셔너리를 생성합니다.

    모델명에 따라 OpenAI API 설정 또는 로컬 HuggingFace 모델 설정을 반환합니다.
    gpt-*로 시작하는 모델은 OpenAI API로, 그 외는 로컬 모델로 처리합니다.

    Args:
        model_name: 교사 모델명. None이면 DEFAULT_TEACHER_MODEL 사용
        gpu_id: GPU 인덱스. None이면 CUDA_VISIBLE_DEVICES 기반 자동 할당.
            API 모델(gpt-*)인 경우 무시됩니다.

    Returns:
        모델 설정 딕셔너리. OpenAI 모델의 경우:
        - model: 모델명
        - base_url: API endpoint (OpenAI는 None)
        - api_key: API 키
        - reasoning: 추론 설정
        - max_tokens: 최대 토큰 수

        로컬 모델의 경우:
        - model: 모델명
        - device: 실행 장치 (cuda)
        - max_new_tokens: 최대 생성 토큰 수
        - temperature: 샘플링 온도
        - do_sample: 샘플링 활성화 여부
        - gpu_id: GPU 인덱스 (None이면 자동)
    """
    if model_name is None:
        model_name = DEFAULT_TEACHER_MODEL

    # OpenAI API 모델 판단
    is_openai_model = model_name.startswith("gpt-")

    # OpenAI API 모델 설정
    if is_openai_model:
        return {
            "model": model_name,
            "base_url": None,  # OpenAI 기본 endpoint
            "api_key": OPENAI_API_KEY,
            "reasoning": {"effort": "medium"},
            "text": {"verbosity": "medium"},
            "max_tokens": 8192
        }

    # 로컬 모델 설정 (vLLM)
    config = {
        "model": model_name,
        "device": "cuda",
        "max_new_tokens": 8192,
        "temperature": 0.7,
        "do_sample": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.90,
    }

    if gpu_id is not None:
        config["gpu_id"] = gpu_id

    return config


# =============================================================================
# 학생 모델 설정
# =============================================================================

# 지원하는 학생 모델 목록
AVAILABLE_STUDENT_MODELS = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
]

# 기본 학생 모델
DEFAULT_STUDENT_MODEL = "Qwen/Qwen3-1.7B"

# 학생 모델 공통 설정
STUDENT_MODEL_BASE_CONFIG = {
    "device": "cuda",
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "do_sample": True,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.90,
}


def get_student_model_config(model_name: str = None, gpu_id: int = None) -> dict:
    """학생 모델 설정 딕셔너리를 생성합니다.

    기본 설정에 모델명을 추가하여 반환합니다.

    Args:
        model_name: 학생 모델명. None이면 DEFAULT_STUDENT_MODEL 사용
        gpu_id: GPU 인덱스. None이면 CUDA_VISIBLE_DEVICES 기반 자동 할당.

    Returns:
        모델 설정 딕셔너리:
        - model_name: 모델명
        - device: 실행 장치
        - max_new_tokens: 최대 생성 토큰 수
        - temperature: 샘플링 온도
        - do_sample: 샘플링 활성화 여부
        - gpu_id: GPU 인덱스 (None이면 자동)

    Raises:
        ValueError: 지원하지 않는 모델명인 경우
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

    if gpu_id is not None:
        config["gpu_id"] = gpu_id

    return config


def get_model_short_name(model_name: str = None) -> str:
    """모델명의 짧은 버전을 반환합니다.

    폴더명이나 파일명에 사용하기 적합한 형태로 변환합니다.
    "Qwen/Qwen3-1.7B" → "Qwen3-1.7B"

    Args:
        model_name: 전체 모델명. None이면 DEFAULT_STUDENT_MODEL 사용

    Returns:
        짧은 모델명 (슬래시 이후 부분)
    """
    if model_name is None:
        model_name = DEFAULT_STUDENT_MODEL

    # "Qwen/Qwen3-1.7B" → "Qwen3-1.7B"
    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name
