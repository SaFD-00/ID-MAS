"""ID-MAS 통합 설정 모듈 (레거시 호환성).

이 모듈은 하위 호환성을 위해 유지되며, 주요 설정은 각 서브모듈로 분리되었습니다:
- api.py: API 자격증명
- models.py: Teacher/Student 모델 설정
- domains.py: 도메인 및 데이터셋 설정
- paths.py: 디렉토리 경로 헬퍼
- sft.py: SFT 모델명 매핑

새 코드에서는 config/__init__.py 또는 개별 서브모듈을 사용하세요.

Note:
    이 파일의 일부 함수와 상수는 서브모듈에서 정의된 것과 중복됩니다.
    점진적으로 서브모듈로 마이그레이션 예정입니다.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

# .env 파일 로드 (기존 환경 변수를 덮어씀)
load_dotenv(PROJECT_ROOT / ".env", override=True)

# API 키
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# =============================================================================
# 교사 모델 설정
# =============================================================================

# 지원하는 교사 모델 목록
AVAILABLE_TEACHER_MODELS = [
    # OpenAI
    "gpt-5-2025-08-07",
    # 로컬 HuggingFace 모델
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
]

# 기본 교사 모델
DEFAULT_TEACHER_MODEL = "gpt-5-2025-08-07"


def create_teacher_config(model_name: str = None, use_api: bool = None) -> dict:
    """교사 모델 설정 딕셔너리를 생성합니다.

    Args:
        model_name: 교사 모델명. None이면 기본 모델 사용
        use_api: API 모드 강제 지정 (레거시, 현재 미사용)

    Returns:
        모델 설정 딕셔너리
    """
    if model_name is None:
        model_name = DEFAULT_TEACHER_MODEL

    # OpenAI API 모델 판단
    is_openai_model = (
        model_name.startswith("gpt-") or
        model_name.startswith("o1") or
        model_name.startswith("o3")
    )

    # OpenAI API 모델 설정
    if is_openai_model:
        return {
            "model": model_name,
            "base_url": None,
            "api_key": OPENAI_API_KEY,
            "reasoning": {"effort": "medium"},
            "text": {"verbosity": "medium"},
            "max_tokens": 8192
        }

    # 로컬 HuggingFace 모델 설정
    return {
        "model": model_name,
        "device": "cuda",
        "max_new_tokens": 8192,
        "temperature": 0.7,
        "do_sample": True
    }


# 기본 교사 모델 설정 (하위 호환성)
DESIGN_MODEL_CONFIG = create_teacher_config(DEFAULT_TEACHER_MODEL)

# 지원하는 학생 모델 목록
AVAILABLE_STUDENT_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
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

# 데이터 경로
DATA_DIR = PROJECT_ROOT / "data"

# 기본 디렉토리 생성
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_student_model_config(model_name: str = None) -> dict:
    """학생 모델 설정 딕셔너리를 생성합니다.

    Args:
        model_name: 학생 모델명. None이면 기본 모델 사용

    Returns:
        모델 설정 딕셔너리

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

    return config


def get_model_short_name(model_name: str = None) -> str:
    """모델명의 짧은 버전을 반환합니다.

    Args:
        model_name: 전체 모델명. None이면 기본 모델 사용

    Returns:
        짧은 모델명 (슬래시 이후 부분)
    """
    if model_name is None:
        model_name = DEFAULT_STUDENT_MODEL

    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name


# =============================================================================
# SFT 모델 설정 (HuggingFace Hub)
# =============================================================================

# 기본 모델명 → HF Hub 레포지토리용 짧은 이름 매핑
MODEL_NAME_TO_SHORT = {
    "meta-llama/Llama-3.1-8B-Instruct": "llama3.1-8b",
    "meta-llama/Llama-3.1-70B-Instruct": "llama3.1-70b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama3.2-3b",
    "meta-llama/Llama-3.3-70B-Instruct": "llama3.3-70b",
    "Qwen/Qwen2.5-3B-Instruct": "qwen2.5-3b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen2.5-7b",
    "Qwen/Qwen2.5-14B-Instruct": "qwen2.5-14b",
    "Qwen/Qwen2.5-72B-Instruct": "qwen2.5-72b",
    "Qwen/Qwen3-4B-Instruct-2507": "qwen3-4b",
}


def get_sft_model_name(base_model_name: str, domain: str) -> str:
    """SFT 모델의 HuggingFace Hub 이름을 생성합니다.

    Args:
        base_model_name: 기본 모델명
        domain: 도메인명

    Returns:
        SFT 모델 HF Hub 이름

    Raises:
        ValueError: 지원하지 않는 모델이거나 알 수 없는 도메인인 경우
    """
    if base_model_name not in MODEL_NAME_TO_SHORT:
        raise ValueError(
            f"모델 '{base_model_name}'은(는) SFT 평가를 지원하지 않습니다.\n"
            f"지원 모델: {list(MODEL_NAME_TO_SHORT.keys())}"
        )

    available_domains = list(DOMAIN_CONFIG.keys())
    if domain not in available_domains:
        raise ValueError(f"도메인은 {available_domains} 중 하나여야 합니다. 입력: {domain}")

    short_name = MODEL_NAME_TO_SHORT[base_model_name]
    return f"SaFD-00/{short_name}-{domain}"


def get_sft_idmas_model_name(base_model_name: str, domain: str) -> str:
    """SFT_ID-MAS 모델의 HuggingFace Hub 이름을 생성합니다.

    Args:
        base_model_name: 기본 모델명
        domain: 도메인명

    Returns:
        SFT_ID-MAS 모델 HF Hub 이름

    Raises:
        ValueError: 지원하지 않는 모델이거나 알 수 없는 도메인인 경우
    """
    if base_model_name not in MODEL_NAME_TO_SHORT:
        raise ValueError(
            f"모델 '{base_model_name}'은(는) SFT_ID-MAS 평가를 지원하지 않습니다.\n"
            f"지원 모델: {list(MODEL_NAME_TO_SHORT.keys())}"
        )

    available_domains = list(DOMAIN_CONFIG.keys())
    if domain not in available_domains:
        raise ValueError(f"도메인은 {available_domains} 중 하나여야 합니다. 입력: {domain}")

    short_name = MODEL_NAME_TO_SHORT[base_model_name]
    return f"SaFD-00/{short_name}-{domain}_id-mas"


# =============================================================================
# 도메인 기반 설정 (config/domains.py로 이동됨)
# =============================================================================

# 도메인 설정 (config/domains.py에서 import)
from config.domains import DOMAIN_CONFIG


# 기본 학생 모델 설정 (하위 호환성)
STUDENT_MODEL_CONFIG = get_student_model_config(DEFAULT_STUDENT_MODEL)
