"""디렉토리 구조 및 경로 헬퍼 모듈.

이 모듈은 ID-MAS 시스템에서 사용하는 디렉토리 경로를 생성하고 관리합니다.
도메인별, 모델별 디렉토리 구조를 자동으로 생성합니다.

주요 함수:
    get_design_output_dir: 설계 결과 저장 디렉토리

Note:
    get_domain_data_dirs 함수는 config/domains.py에서 제공됩니다.
    해당 함수는 OUTPUT_DIR (outputs/)을 사용하여 학습 결과를 저장합니다.
"""
from pathlib import Path
from config.domains import DATA_DIR, OUTPUT_DIR
from config.models import get_model_short_name

# 기본 데이터 디렉토리 생성
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_design_output_dir(domain: str, teacher_model_name: str = None) -> Path:
    """설계 결과 저장 디렉토리 경로를 반환합니다.

    디렉토리 구조: outputs/{domain}/train/{teacher_model}/instructional-design/

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
    design_dir = OUTPUT_DIR / domain.lower() / "train" / teacher_short / "instructional-design"
    design_dir.mkdir(parents=True, exist_ok=True)
    return design_dir
