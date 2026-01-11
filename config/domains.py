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

# Fallback Terminal Goals for each training dataset (design JSON 파일이 없을 때 사용)
TERMINAL_GOALS = {
    # Math domain
    "gsm8k": "Generate coherent, step-by-step mathematical reasoning in natural language that leads to a correct numerical answer for grade-school level math problems.",
    "math": "Solve advanced mathematical problems by selecting appropriate mathematical concepts and constructing logically valid, multi-step reasoning that leads to a correct solution.",

    # Logical domain
    "reclor": "Analyze logical reasoning problems by comprehending complex passages, identifying logical relationships, and selecting the most appropriate conclusion based on formal reasoning principles.",

    # Commonsense domain
    "arc_c": "Apply commonsense scientific knowledge to solve elementary science problems by understanding fundamental concepts and selecting the correct answer from multiple choices.",
}

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


def get_terminal_goal(
    dataset: str,
    teacher_model: Optional[str] = None,
    use_cache: bool = True
) -> str:
    """
    Get Terminal Goal for a training dataset.

    우선순위:
    1. Design JSON 파일에서 로드 (동적 생성된 terminal_goal)
    2. Fallback: 하드코딩된 TERMINAL_GOALS

    Args:
        dataset: 데이터셋 이름 (e.g., "gsm8k", "math")
        teacher_model: Teacher 모델 이름 (None이면 design JSON 검색 안 함)
        use_cache: 캐시 사용 여부 (기본 True)

    Returns:
        Terminal Goal 문자열
    """
    # Design JSON에서 로드 시도 (teacher_model이 지정된 경우)
    if teacher_model:
        domain = DATASET_TO_DOMAIN.get(dataset)
        if domain:
            design_goal = _load_terminal_goal_from_design(domain, dataset, teacher_model)
            if design_goal:
                return design_goal

    # Fallback: 하드코딩된 값
    if dataset not in TERMINAL_GOALS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(TERMINAL_GOALS.keys())}")
    return TERMINAL_GOALS[dataset]


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
