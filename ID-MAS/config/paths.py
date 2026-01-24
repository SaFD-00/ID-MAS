"""디렉토리 구조 및 경로 헬퍼 모듈.

이 모듈은 ID-MAS 시스템에서 사용하는 디렉토리 경로를 생성하고 관리합니다.
도메인별, 모델별 디렉토리 구조를 자동으로 생성합니다.

주요 함수:
    get_design_output_dir: 설계 결과 저장 디렉토리
    get_domain_data_dirs: 도메인별 데이터 디렉토리
"""
from pathlib import Path
from config.domains import DATA_DIR, DOMAIN_CONFIG
from config.models import get_model_short_name

# 기본 데이터 디렉토리 생성
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_design_output_dir(domain: str, teacher_model_name: str = None) -> Path:
    """설계 결과 저장 디렉토리 경로를 반환합니다.

    디렉토리 구조: data/{domain}/train/{teacher_model}/instructional-design/

    Args:
        domain: 도메인명 (예: "math", "logical")
        teacher_model_name: Teacher 모델명. None이면 기본 모델 사용

    Returns:
        설계 결과 저장 디렉토리 Path 객체

    Raises:
        ValueError: domain이 비어있는 경우
    """
    from config.models import DEFAULT_TEACHER_MODEL

    if not domain:
        raise ValueError("설계 결과 저장에는 domain이 필수입니다")

    teacher_short = get_model_short_name(teacher_model_name) if teacher_model_name else get_model_short_name(DEFAULT_TEACHER_MODEL)
    design_dir = DATA_DIR / domain.lower() / "train" / teacher_short / "instructional-design"
    design_dir.mkdir(parents=True, exist_ok=True)
    return design_dir


def get_domain_data_dirs(
    domain: str,
    model_name: str = None,
    train_dataset: str = None,
    mode: str = "train",
    teacher_model_name: str = None
) -> dict:
    """도메인별, 모델별 데이터 디렉토리 경로를 반환합니다.

    신규 디렉토리 구조를 사용합니다:
    - Train 모드: data/{domain}/train/{teacher_model}/{student_model}/
    - Eval 모드: data/{domain}/eval/{student_model}/

    Args:
        domain: 도메인명 (예: "math", "logical", "commonsense")
        model_name: Student 모델명. None이면 기본 모델 사용
        train_dataset: 학습 데이터셋명 (파일명 생성에만 사용)
        mode: "train" 또는 "eval"
        teacher_model_name: Teacher 모델명 (train 모드에서만 사용)

    Returns:
        경로 딕셔너리:
        - domain_dir: 도메인 루트 디렉토리
        - model_dir: 모델별 출력 디렉토리
        - dataset_dir: 데이터셋 디렉토리
        - sft_data_dir: SFT 데이터 파일 경로 (train 모드만)
        - learning_loop_dir: 학습 루프 디렉토리 (train 모드만)

    Raises:
        ValueError: 알 수 없는 도메인인 경우
    """
    from config.models import DEFAULT_TEACHER_MODEL

    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"알 수 없는 도메인: {domain}. 가능한 도메인: {list(DOMAIN_CONFIG.keys())}")

    model_short = get_model_short_name(model_name)
    domain_dir = DATA_DIR / domain

    if mode == "train":
        # Teacher 모델명으로 상위 디렉토리 결정
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
    else:  # eval 모드
        model_dir = domain_dir / "eval" / model_short

        dirs = {
            "domain_dir": domain_dir,
            "model_dir": model_dir,
            "dataset_dir": domain_dir / "eval" / "data"
        }

    # 디렉토리 자동 생성 (파일 경로 제외)
    for key, dir_path in dirs.items():
        if key != "sft_data_dir" and isinstance(dir_path, Path):
            dir_path.mkdir(parents=True, exist_ok=True)

    return dirs
