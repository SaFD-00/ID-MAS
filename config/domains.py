"""
Domain and dataset configuration.
"""
import json
from pathlib import Path
from typing import Optional
from config.api import PROJECT_ROOT

# =============================================================================
# Domain-based Configuration
# =============================================================================

# Dataset to domain mapping
DATASET_TO_DOMAIN = {
    "gsm8k": "math",
    "math": "math",
    "reclor": "logical",
    "arc_c": "commonsense",
}

# Available training datasets per domain
TRAINING_DATASETS = {
    "math": ["gsm8k", "math"],
    "logical": ["reclor"],
    "commonsense": ["arc_c"],
}

# Data directory path
DATA_DIR = PROJECT_ROOT / "data"

# Domain configurations
DOMAIN_CONFIG = {
    "math": {
        "data_dir": DATA_DIR / "math",
        "training_datasets": ["gsm8k", "math"],
        "eval_datasets": ["gsm8k", "math", "svamp", "asdiv", "mawps"],
        "default_eval": "gsm8k"
    },
    "logical": {
        "data_dir": DATA_DIR / "logical",
        "training_datasets": ["reclor"],
        "eval_datasets": [
            "reclor", "anli_r2", "anli_r3",
            "bbh_boolean_expressions", "bbh_formal_fallacies",
            "bbh_logical_deduction_three_objects", "bbh_logical_deduction_five_objects",
            "bbh_logical_deduction_seven_objects",
            "bbh_tracking_shuffled_objects_three_objects",
            "bbh_tracking_shuffled_objects_five_objects",
            "bbh_tracking_shuffled_objects_seven_objects",
            "bbh_web_of_lies"
        ],
        "default_eval": "reclor"
    },
    "commonsense": {
        "data_dir": DATA_DIR / "commonsense",
        "training_datasets": ["arc_c"],
        "eval_datasets": ["arc_c", "strategyqa", "openbookqa"],
        "default_eval": "arc_c"
    }
}


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


def get_domain_data_dirs(domain: str, model_name: str = None, train_dataset: str = None, mode: str = "train", teacher_model_name: str = None) -> dict:
    """
    도메인별, 모델별 데이터 디렉토리 경로 반환 (새 구조)

    New Structure:
        Train: data/{domain}/train/{teacher_model}/{student_model}/
        Eval:  data/{domain}/eval/{student_model}/

    Args:
        domain: 도메인 이름 (예: "math", "logical", "commonsense")
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
    from config.models import get_model_short_name, DEFAULT_TEACHER_MODEL

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


def get_terminal_goal(
    dataset: str,
    teacher_model: Optional[str] = None
) -> Optional[str]:
    """
    Get Terminal Goal for a training dataset from design JSON.

    Design JSON 파일에서만 로드하며, 없으면 None 반환 (fallback 없음).

    Args:
        dataset: 데이터셋 이름 (e.g., "gsm8k", "math")
        teacher_model: Teacher 모델 이름 (필수 - design JSON 경로 결정에 사용)

    Returns:
        Terminal Goal 문자열 또는 None (생성된 것이 없는 경우)
    """
    if not teacher_model:
        return None

    domain = DATASET_TO_DOMAIN.get(dataset)
    if not domain:
        return None

    return _load_terminal_goal_from_design(domain, dataset, teacher_model)


def _load_terminal_goal_from_design(
    domain: str,
    dataset: str,
    teacher_model: str
) -> Optional[str]:
    """
    Design JSON 파일에서 terminal_goal 로드

    Args:
        domain: 도메인 이름
        dataset: 데이터셋 이름
        teacher_model: Teacher 모델 이름

    Returns:
        Terminal Goal 또는 None (파일 없거나 필드 없음)
    """
    from config.config import get_model_short_name

    try:
        teacher_short = get_model_short_name(teacher_model)
        design_dir = DATA_DIR / domain / "train" / teacher_short / "instructional-design"
        design_path = design_dir / f"{domain}_{dataset}_design.json"

        if design_path.exists():
            with open(design_path, 'r', encoding='utf-8') as f:
                design_data = json.load(f)

            terminal_goal = design_data.get("terminal_goal")
            if terminal_goal:
                return terminal_goal

    except Exception:
        # 로드 실패 시 무시하고 fallback 사용
        pass

    return None
