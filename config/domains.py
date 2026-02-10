"""도메인 및 데이터셋 설정 모듈.

이 모듈은 ID-MAS 시스템의 도메인(math, logical, commonsense) 및
데이터셋 관련 설정을 관리합니다.

주요 상수:
    DATASET_TO_DOMAIN: 데이터셋 → 도메인 매핑
    TRAINING_DATASETS: 도메인별 학습 데이터셋 목록
    DOMAIN_CONFIG: 도메인별 상세 설정
    DATA_DIR: 데이터 디렉토리 경로

주요 함수:
    get_available_domains: 사용 가능한 도메인 목록 반환
    get_eval_datasets_for_domain: 도메인별 평가 데이터셋 반환
    get_training_datasets_for_domain: 도메인별 학습 데이터셋 반환
    get_domain_data_dirs: 도메인별 디렉토리 경로 반환
    get_instructional_goal: 설계 JSON에서 Instructional Goal 로드
"""
import json
from pathlib import Path
from typing import Optional
from config.api import PROJECT_ROOT

# =============================================================================
# 도메인 기반 설정
# =============================================================================

# 데이터셋 → 도메인 매핑
DATASET_TO_DOMAIN = {
    "gsm8k": "math",
    "math": "math",
    "reclor": "logical",
    "arc_c": "commonsense",
}

# 도메인별 학습 데이터셋
TRAINING_DATASETS = {
    "math": ["gsm8k", "math"],
    "logical": ["reclor"],
    "commonsense": ["arc_c"],
}

# 데이터 디렉토리 경로
DATA_DIR = PROJECT_ROOT / "data"

# 출력 디렉토리 경로 (학습 결과물)
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 도메인별 상세 설정
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
            "bbh"  # 통합 BBH (모든 하위 태스크가 단일 파일, subtask 정보는 metadata에)
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
    """사용 가능한 도메인 목록을 반환합니다.

    Returns:
        도메인명 리스트 (예: ["math", "logical", "commonsense"])
    """
    return list(DOMAIN_CONFIG.keys())


def get_eval_datasets_for_domain(domain: str) -> list:
    """도메인의 평가 데이터셋 목록을 반환합니다.

    Args:
        domain: 도메인명 (예: "math")

    Returns:
        평가 데이터셋 리스트 (예: ["gsm8k", "math", "svamp", ...])

    Raises:
        ValueError: 알 수 없는 도메인인 경우
    """
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"알 수 없는 도메인: {domain}. 가능한 도메인: {list(DOMAIN_CONFIG.keys())}")
    return DOMAIN_CONFIG[domain]["eval_datasets"]


def get_training_datasets_for_domain(domain: str) -> list:
    """도메인의 학습 데이터셋 목록을 반환합니다.

    Args:
        domain: 도메인명 (예: "math")

    Returns:
        학습 데이터셋 리스트 (예: ["gsm8k", "math"])

    Raises:
        ValueError: 알 수 없는 도메인인 경우
    """
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"알 수 없는 도메인: {domain}. 가능한 도메인: {list(DOMAIN_CONFIG.keys())}")
    return DOMAIN_CONFIG[domain]["training_datasets"]


def get_domain_data_dirs(
    domain: str,
    model_name: str = None,
    train_dataset: str = None,
    mode: str = "train",
    teacher_model_name: str = None
) -> dict:
    """도메인별, 모델별 데이터 디렉토리 경로를 반환합니다.

    디렉토리 구조:
    - Train 모드: outputs/{domain}/train/{teacher_short}/{student_short}/
    - Eval 모드: outputs/{domain}/eval/{student_short}/

    Args:
        domain: 도메인명 (예: "math", "logical", "commonsense")
        model_name: Student 모델명. None이면 기본 모델 사용
        train_dataset: 학습 데이터셋명 (파일명 생성에만 사용)
        mode: "train" 또는 "eval"
        teacher_model_name: Teacher 모델명 (train 모드에서만 사용)

    Returns:
        경로 딕셔너리:
        - model_dir: 모델별 출력 디렉토리
        - raw_data_dir: 원본 데이터 디렉토리
        - design_dir: 설계 결과 디렉토리 (train 모드만)
        - enhanced_data_dir: Enhanced training data 디렉토리 (train 모드만)

    Raises:
        ValueError: 알 수 없는 도메인인 경우
    """
    from config.models import get_model_short_name, DEFAULT_TEACHER_MODEL

    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"알 수 없는 도메인: {domain}. 가능한 도메인: {list(DOMAIN_CONFIG.keys())}")

    model_short = get_model_short_name(model_name)

    if mode == "train":
        # Teacher 모델명으로 상위 디렉토리 결정
        teacher_short = get_model_short_name(teacher_model_name) if teacher_model_name else get_model_short_name(DEFAULT_TEACHER_MODEL)
        model_dir = OUTPUT_DIR / domain / "train" / teacher_short / model_short
        dirs = {
            "model_dir": model_dir,
            "raw_data_dir": DATA_DIR / domain / "train" / "data",
            "design_dir": OUTPUT_DIR / domain / "train" / teacher_short / "instructional-design",
            "enhanced_data_dir": OUTPUT_DIR / domain / "train" / model_short / "data",
        }
    else:  # eval 모드
        model_dir = OUTPUT_DIR / domain / "eval" / model_short
        dirs = {
            "model_dir": model_dir,
            "raw_data_dir": DATA_DIR / domain / "eval" / "data",
        }

    # 디렉토리 생성
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def get_instructional_goal(
    dataset: str,
    teacher_model: Optional[str] = None
) -> Optional[str]:
    """설계 JSON 파일에서 Instructional Goal을 로드합니다.

    설계 JSON 파일이 없으면 None을 반환합니다 (fallback 없음).

    Args:
        dataset: 데이터셋명 (예: "gsm8k", "math")
        teacher_model: Teacher 모델명 (설계 JSON 경로 결정에 필수)

    Returns:
        Instructional Goal 문자열 또는 None (파일이 없거나 미생성)
    """
    if not teacher_model:
        return None

    domain = DATASET_TO_DOMAIN.get(dataset)
    if not domain:
        return None

    return _load_instructional_goal_from_design(domain, dataset, teacher_model)


def _load_instructional_goal_from_design(
    domain: str,
    dataset: str,
    teacher_model: str
) -> Optional[str]:
    """설계 JSON 파일에서 instructional_goal 필드를 로드합니다.

    내부 헬퍼 함수로, get_instructional_goal에서 호출됩니다.

    Args:
        domain: 도메인명
        dataset: 데이터셋명
        teacher_model: Teacher 모델명

    Returns:
        Instructional Goal 문자열 또는 None
    """
    from config.config import get_model_short_name

    try:
        teacher_short = get_model_short_name(teacher_model)
        design_dir = OUTPUT_DIR / domain / "train" / teacher_short / "instructional-design"
        design_path = design_dir / f"{domain}_{dataset}_design.json"

        if design_path.exists():
            with open(design_path, 'r', encoding='utf-8') as f:
                design_data = json.load(f)

            instructional_goal = design_data.get("instructional_goal")
            if instructional_goal:
                return instructional_goal

    except Exception:
        # 로드 실패 시 무시
        pass

    return None
