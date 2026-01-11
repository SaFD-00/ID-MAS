"""
설정 파일
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

# .env 파일 로드 (기존 환경 변수를 덮어쓰기)
load_dotenv(PROJECT_ROOT / ".env", override=True)

# API 키
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# =============================================================================
# Teacher Model Configuration
# =============================================================================

# 지원하는 Teacher 모델 목록
AVAILABLE_TEACHER_MODELS = [
    # OpenAI
    "gpt-5-2025-08-07",
    # LLaMA-Factory API (OpenAI-compatible)
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

# 기본 Teacher 모델
DEFAULT_TEACHER_MODEL = "gpt-5-2025-08-07"


def create_teacher_config(model_name: str = None, use_api: bool = None) -> dict:
    """
    Teacher model config 생성

    Args:
        model_name: Teacher 모델 이름 (None이면 기본 모델 사용)
        use_api: API 모드 강제 지정 (None이면 모델명으로 자동 판단)
            - True: LLaMA-Factory API 사용
            - False: 로컬 HuggingFace 모델 로드
            - None: gpt-*, o1-*, o3-*는 OpenAI API, 그 외는 로컬 로드

    Returns:
        Teacher model 설정 딕셔너리
    """
    if model_name is None:
        model_name = DEFAULT_TEACHER_MODEL

    # API 모델 판단 (OpenAI 모델)
    is_openai_model = (
        model_name.startswith("gpt-") or
        model_name.startswith("o1") or
        model_name.startswith("o3")
    )

    # OpenAI 모델
    if is_openai_model:
        return {
            "model": model_name,
            "base_url": None,  # OpenAI 기본 endpoint
            "api_key": OPENAI_API_KEY,
            "reasoning": {"effort": "medium"},
            "text": {"verbosity": "medium"},
            "max_tokens": 8192
        }

    # 로컬 HuggingFace 모델 (기본값)
    return {
        "model": model_name,
        "device": "cuda",
        "max_new_tokens": 8192,
        "temperature": 0.7,
        "do_sample": True
    }


# 기본 Teacher 모델 설정 (하위 호환성 - DESIGN_MODEL_CONFIG)
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

# 데이터 경로 (함수에서 사용하기 전에 정의)
DATA_DIR = PROJECT_ROOT / "data"

# 기본 디렉토리 생성 (나머지는 도메인별/모델별로 생성됨)
DATA_DIR.mkdir(parents=True, exist_ok=True)


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


# =============================================================================
# SFT Model Configuration (HuggingFace Hub)
# =============================================================================

# Model name mapping for SFT fine-tuned models on HuggingFace Hub
# Maps base model names to short names used in SaFD-00/{model}-{domain} repos
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
    """
    Get SFT fine-tuned model name from HuggingFace Hub.

    Args:
        base_model_name: Base model name (e.g., "Qwen/Qwen2.5-3B-Instruct")
        domain: Domain name (e.g., "math")

    Returns:
        SFT model HF Hub name (e.g., "SaFD-00/qwen2.5-3b-math")

    Raises:
        ValueError: If model is not supported for SFT or domain is invalid

    Example:
        >>> get_sft_model_name("Qwen/Qwen2.5-3B-Instruct", "math")
        'SaFD-00/qwen2.5-3b-math'
    """
    if base_model_name not in MODEL_NAME_TO_SHORT:
        raise ValueError(
            f"Model '{base_model_name}' not supported for SFT evaluation.\n"
            f"Supported models: {list(MODEL_NAME_TO_SHORT.keys())}"
        )

    available_domains = list(DOMAIN_CONFIG.keys())
    if domain not in available_domains:
        raise ValueError(f"Domain must be one of {available_domains}, got: {domain}")

    short_name = MODEL_NAME_TO_SHORT[base_model_name]
    return f"SaFD-00/{short_name}-{domain}"


def get_sft_idmas_model_name(base_model_name: str, domain: str) -> str:
    """
    Get SFT_ID-MAS fine-tuned model name from HuggingFace Hub.

    Args:
        base_model_name: Base model name (e.g., "Qwen/Qwen2.5-3B-Instruct")
        domain: Domain name (e.g., "math")

    Returns:
        SFT_ID-MAS model HF Hub name (e.g., "SaFD-00/qwen2.5-3b-math_id-mas")

    Raises:
        ValueError: If model is not supported or domain is invalid

    Example:
        >>> get_sft_idmas_model_name("Qwen/Qwen2.5-3B-Instruct", "math")
        'SaFD-00/qwen2.5-3b-math_id-mas'
    """
    if base_model_name not in MODEL_NAME_TO_SHORT:
        raise ValueError(
            f"Model '{base_model_name}' not supported for SFT_ID-MAS evaluation.\n"
            f"Supported models: {list(MODEL_NAME_TO_SHORT.keys())}"
        )

    available_domains = list(DOMAIN_CONFIG.keys())
    if domain not in available_domains:
        raise ValueError(f"Domain must be one of {available_domains}, got: {domain}")

    short_name = MODEL_NAME_TO_SHORT[base_model_name]
    return f"SaFD-00/{short_name}-{domain}_id-mas"


def get_model_data_dirs(model_name: str = None) -> dict:
    """
    모델별 데이터 디렉토리 경로 반환 (기존 방식 - 하위 호환성)

    Args:
        model_name: 모델 이름 (None이면 기본 모델 사용)

    Returns:
        경로 딕셔너리 (learning_logs_dir, eval_results_dir, knowledge_base_dir)
    """
    short_name = get_model_short_name(model_name)
    model_dir = DATA_DIR / short_name

    dirs = {
        "model_dir": model_dir,
        "learning_logs_dir": model_dir / "learning_logs",
        "eval_results_dir": model_dir / "eval_results",
        "knowledge_base_dir": model_dir / "knowledge_base"
    }

    # 디렉토리 생성
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def get_dataset_model_dirs(dataset: str, model_name: str = None) -> dict:
    """
    데이터셋별, 모델별 데이터 디렉토리 경로 반환 (기존 방식 - deprecated)

    Structure: data/{dataset}/{model}/

    Args:
        dataset: 데이터셋 이름 (예: "gsm8k", "mmlu", "arc")
        model_name: 모델 이름 (None이면 기본 모델 사용)

    Returns:
        경로 딕셔너리
    """
    short_name = get_model_short_name(model_name)
    dataset_dir = DATA_DIR / dataset.lower() / short_name

    dirs = {
        "dataset_dir": dataset_dir,
        "learning_logs_dir": dataset_dir / "learning_logs",
        "eval_results_dir": dataset_dir / "eval_results",
        "knowledge_base_dir": dataset_dir / "knowledge_base"
    }

    # 디렉토리 생성
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


# =============================================================================
# Domain-based Configuration (New)
# =============================================================================

# Terminal Goals for each training dataset
TERMINAL_GOALS = {
    "gsm8k": "Generate coherent, step-by-step mathematical reasoning in natural language that leads to a correct numerical answer for grade-school level math problems.",
    "math": "Solve advanced mathematical problems by selecting appropriate mathematical concepts and constructing logically valid, multi-step reasoning that leads to a correct solution."
}

# Dataset to domain mapping
DATASET_TO_DOMAIN = {
    "gsm8k": "math",
    "math": "math"
}

# Available training datasets per domain
TRAINING_DATASETS = {
    "math": ["gsm8k", "math"]
}

# Domain configurations
DOMAIN_CONFIG = {
    "math": {
        "data_dir": DATA_DIR / "math",
        "training_datasets": ["gsm8k", "math"],
        "eval_datasets": ["gsm8k", "math", "svamp", "asdiv", "mawps"],
        "default_eval": "gsm8k"
    }
}


def get_domain_data_dirs(domain: str, model_name: str = None, train_dataset: str = None, mode: str = "train", teacher_model_name: str = None) -> dict:
    """
    도메인별, 모델별 데이터 디렉토리 경로 반환 (새 구조)

    New Structure:
        Train: data/{domain}/train/{teacher_model}/{student_model}/
        Eval:  data/{domain}/eval/{student_model}/

    Args:
        domain: 도메인 이름 (예: "math")
        model_name: Student 모델 이름 (None이면 기본 모델 사용)
        train_dataset: 학습 데이터셋 이름 (파일명 생성에 사용, 폴더 구조에는 미사용)
        mode: "train" 또는 "eval"
        teacher_model_name: Teacher 모델 이름 (train 모드에서 사용, None이면 기본 모델 사용)

    Returns:
        경로 딕셔너리:
        - model_dir: 모델별 출력 디렉토리
        - raw_data_dir: 원본 데이터 디렉토리
        - design_dir: 설계 결과 디렉토리 (train 모드만)
    """
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_CONFIG.keys())}")

    model_short = get_model_short_name(model_name)

    if mode == "train":
        # Teacher 모델 이름으로 상위 디렉토리 생성
        teacher_short = get_model_short_name(teacher_model_name) if teacher_model_name else get_model_short_name(DEFAULT_TEACHER_MODEL)
        model_dir = DATA_DIR / domain / "train" / teacher_short / model_short
        dirs = {
            "model_dir": model_dir,
            "raw_data_dir": DATA_DIR / domain / "train" / "data",
            "design_dir": DATA_DIR / domain / "train" / teacher_short / "instructional-design",
        }
    else:  # eval
        model_dir = DATA_DIR / domain / "eval" / model_short
        dirs = {
            "model_dir": model_dir,
            "raw_data_dir": DATA_DIR / domain / "eval" / "data",
        }

    # 디렉토리 생성
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def get_available_domains() -> list:
    """Get list of available domains."""
    return list(DOMAIN_CONFIG.keys())


def get_eval_datasets_for_domain(domain: str) -> list:
    """Get available evaluation datasets for a domain."""
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_CONFIG.keys())}")
    return DOMAIN_CONFIG[domain]["eval_datasets"]


def get_training_datasets_for_domain(domain: str) -> list:
    """Get available training datasets for a domain."""
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_CONFIG.keys())}")
    return DOMAIN_CONFIG[domain]["training_datasets"]


def get_terminal_goal(dataset: str) -> str:
    """Get Terminal Goal for a training dataset."""
    if dataset not in TERMINAL_GOALS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(TERMINAL_GOALS.keys())}")
    return TERMINAL_GOALS[dataset]


# 기본 학생 모델 설정 (하위 호환성)
STUDENT_MODEL_CONFIG = get_student_model_config(DEFAULT_STUDENT_MODEL)

# 학습 루프 설정
LEARNING_LOOP_CONFIG = {
    "max_iterations": 5,  # 최대 반복 횟수
    "convergence_threshold": 0.9  # 수렴 기준 (루브릭 점수)
}
