"""
Directory structure and path helpers.
"""
from pathlib import Path
from config.domains import DATA_DIR, DOMAIN_CONFIG
from config.models import get_model_short_name

# 기본 디렉토리 생성
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_design_output_dir(domain: str, teacher_model_name: str = None) -> Path:
    """
    도메인별 설계 출력 디렉토리 경로 반환

    Args:
        domain: 도메인 이름 (예: "math")
        teacher_model_name: Teacher 모델 이름 (None이면 기본 모델 사용)

    Returns:
        설계 결과 저장 디렉토리 (Path 객체)
        New structure: data/{domain}/train/{teacher_model}/instructional-design/
    """
    from config.models import DEFAULT_TEACHER_MODEL

    if not domain:
        raise ValueError("Domain cannot be empty for design outputs")

    teacher_short = get_model_short_name(teacher_model_name) if teacher_model_name else get_model_short_name(DEFAULT_TEACHER_MODEL)
    design_dir = DATA_DIR / domain.lower() / "train" / teacher_short / "instructional-design"
    design_dir.mkdir(parents=True, exist_ok=True)
    return design_dir


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
        - domain_dir: 도메인 디렉토리
        - model_dir: 모델별 출력 디렉토리
        - dataset_dir: 데이터셋 디렉토리
        - sft_data_dir: SFT 데이터 디렉토리 (train 모드만)
        - learning_loop_dir: 학습 루프 디렉토리 (train 모드만)
    """
    from config.models import DEFAULT_TEACHER_MODEL

    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_CONFIG.keys())}")

    model_short = get_model_short_name(model_name)
    domain_dir = DATA_DIR / domain

    if mode == "train":
        # Teacher 모델 이름으로 상위 디렉토리 생성
        teacher_short = get_model_short_name(teacher_model_name) if teacher_model_name else get_model_short_name(DEFAULT_TEACHER_MODEL)
        model_dir = domain_dir / "train" / teacher_short / model_short
        dataset_key = f"{domain}_{train_dataset}" if train_dataset else domain

        dirs = {
            "domain_dir": domain_dir,
            "model_dir": model_dir,
            "dataset_dir": domain_dir / "train" / "data",
            "sft_data_dir": model_dir / f"{dataset_key}_sft_data.json",
            "learning_loop_dir": model_dir / "learning_loops"
        }
    else:  # eval
        model_dir = domain_dir / "eval" / model_short

        dirs = {
            "domain_dir": domain_dir,
            "model_dir": model_dir,
            "dataset_dir": domain_dir / "eval" / "data"
        }

    # 디렉토리 생성 (파일 경로는 제외)
    for key, dir_path in dirs.items():
        if key != "sft_data_dir" and isinstance(dir_path, Path):
            dir_path.mkdir(parents=True, exist_ok=True)

    return dirs
