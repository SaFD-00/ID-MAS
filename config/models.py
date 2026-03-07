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
    normalize_gpu_ids: GPU ID를 정규화된 tuple로 변환
    create_teacher_config: 교사 모델 설정 생성
    get_student_model_config: 학생 모델 설정 생성
    get_model_short_name: 모델명의 짧은 버전 반환
"""
from typing import Optional, Tuple, Union
from config.api import OPENAI_API_KEY


def normalize_gpu_ids(
    gpu_ids: Union[None, int, Tuple[int, ...], list]
) -> Optional[Tuple[int, ...]]:
    """GPU ID를 정규화된 tuple로 변환합니다.

    다양한 입력 형태를 hashable한 tuple로 통일합니다.

    Args:
        gpu_ids: GPU 인덱스. None, int, tuple, list 모두 허용.
            - None → None (자동 할당)
            - int → (int,) (단일 GPU)
            - tuple/list → tuple (다중 GPU)

    Returns:
        정규화된 GPU ID tuple. None이면 자동 할당.
    """
    if gpu_ids is None:
        return None
    if isinstance(gpu_ids, int):
        return (gpu_ids,)
    return tuple(gpu_ids)

# =============================================================================
# 교사 모델 설정
# =============================================================================

# 지원하는 교사 모델 목록
AVAILABLE_TEACHER_MODELS = [
    # OpenAI API
    "gpt-5.2",
    # 로컬 HuggingFace 모델 (ModelCache를 통해 직접 로드)
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
]

# 기본 교사 모델
DEFAULT_TEACHER_MODEL = "Qwen/Qwen3-4B"


def create_teacher_config(
    model_name: str = None,
    gpu_ids: Union[None, int, Tuple[int, ...], list] = None
) -> dict:
    """교사 모델 설정 딕셔너리를 생성합니다.

    모델명에 따라 OpenAI API 설정 또는 로컬 HuggingFace 모델 설정을 반환합니다.
    gpt-*로 시작하는 모델은 OpenAI API로, 그 외는 로컬 모델로 처리합니다.

    Args:
        model_name: 교사 모델명. None이면 DEFAULT_TEACHER_MODEL 사용
        gpu_ids: GPU 인덱스(들). None, int, tuple, list 허용.
            None이면 CUDA_VISIBLE_DEVICES 기반 자동 할당.
            API 모델(gpt-*)인 경우 무시됩니다.
            다중 GPU 지정 시 tensor_parallel_size가 자동 결정됩니다.

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
        - gpu_ids: 정규화된 GPU ID tuple (None이면 자동)
        - tensor_parallel_size: GPU 수에 연동
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

    # GPU ID 정규화
    normalized = normalize_gpu_ids(gpu_ids)

    # 로컬 모델 설정 (vLLM)
    config = {
        "model": model_name,
        "device": "cuda",
        "max_new_tokens": 8192,
        "temperature": 0.7,
        "do_sample": True,
        "tensor_parallel_size": len(normalized) if normalized else 1,
        "gpu_memory_utilization": 0.90,
    }

    if normalized is not None:
        config["gpu_ids"] = normalized

    return config


# =============================================================================
# 학생 모델 설정
# =============================================================================

# 지원하는 학생 모델 목록
AVAILABLE_STUDENT_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
]

# 기본 학생 모델
DEFAULT_STUDENT_MODEL = "Qwen/Qwen3-4B"

# 학생 모델 공통 설정
STUDENT_MODEL_BASE_CONFIG = {
    "device": "cuda",
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "do_sample": True,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.90,
}


def get_student_model_config(
    model_name: str = None,
    gpu_ids: Union[None, int, Tuple[int, ...], list] = None
) -> dict:
    """학생 모델 설정 딕셔너리를 생성합니다.

    기본 설정에 모델명을 추가하여 반환합니다.

    Args:
        model_name: 학생 모델명. None이면 DEFAULT_STUDENT_MODEL 사용
        gpu_ids: GPU 인덱스(들). None, int, tuple, list 허용.
            None이면 CUDA_VISIBLE_DEVICES 기반 자동 할당.
            다중 GPU 지정 시 tensor_parallel_size가 자동 결정됩니다.

    Returns:
        모델 설정 딕셔너리:
        - model_name: 모델명
        - device: 실행 장치
        - max_new_tokens: 최대 생성 토큰 수
        - temperature: 샘플링 온도
        - do_sample: 샘플링 활성화 여부
        - gpu_ids: 정규화된 GPU ID tuple (None이면 자동)
        - tensor_parallel_size: GPU 수에 연동
    """
    if model_name is None:
        model_name = DEFAULT_STUDENT_MODEL

    # GPU ID 정규화
    normalized = normalize_gpu_ids(gpu_ids)

    config = STUDENT_MODEL_BASE_CONFIG.copy()
    config["model_name"] = model_name

    # 다중 GPU 시 tensor_parallel_size 자동 결정
    if normalized is not None:
        config["gpu_ids"] = normalized
        config["tensor_parallel_size"] = len(normalized)

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
