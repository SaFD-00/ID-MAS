"""ID-MAS 설정 패키지.

이 패키지는 ID-MAS 시스템의 모든 설정을 통합 인터페이스로 제공합니다.
설정은 논리적 서브모듈로 구성되어 있으며, 이 __init__.py를 통해
단일 진입점으로 접근할 수 있습니다.

서브모듈 구성:
    api: API 자격증명 및 환경 변수
    models: Teacher/Student 모델 설정
    sft: SFT 파인튜닝 모델 매핑
    domains: 도메인 및 데이터셋 설정
    paths: 디렉토리 경로 헬퍼
사용 예시:
    >>> from config import create_teacher_config, get_model_short_name
    >>> teacher_config = create_teacher_config("gpt-5.2")
    >>> short_name = get_model_short_name("Qwen/Qwen3-1.7B")
"""

# API 자격증명
from config.api import (
    PROJECT_ROOT,
    OPENAI_API_KEY,
    HF_TOKEN,
)

# 모델 설정
from config.models import (
    AVAILABLE_TEACHER_MODELS,
    DEFAULT_TEACHER_MODEL,
    create_teacher_config,
    AVAILABLE_STUDENT_MODELS,
    DEFAULT_STUDENT_MODEL,
    STUDENT_MODEL_BASE_CONFIG,
    get_student_model_config,
    get_model_short_name,
    normalize_gpu_ids,
)

# SFT 모델 매핑
from config.sft import (
    MODEL_NAME_TO_SHORT,
    get_sft_model_name,
    get_sft_idmas_model_name,
)

# 도메인 및 데이터셋 설정
from config.domains import (
    DATASET_TO_DOMAIN,
    TRAINING_DATASETS,
    DOMAIN_CONFIG,
    DATA_DIR,
    get_available_domains,
    get_eval_datasets_for_domain,
    get_training_datasets_for_domain,
    get_instructional_goal,
    get_domain_data_dirs,
)

# 경로 헬퍼
from config.paths import get_design_output_dir

# 하위 호환성을 위한 레거시 별칭
DESIGN_MODEL_CONFIG = create_teacher_config(DEFAULT_TEACHER_MODEL)

__all__ = [
    # API
    'PROJECT_ROOT', 'OPENAI_API_KEY', 'HF_TOKEN',

    # 모델
    'AVAILABLE_TEACHER_MODELS', 'DEFAULT_TEACHER_MODEL',
    'create_teacher_config', 'AVAILABLE_STUDENT_MODELS', 'DEFAULT_STUDENT_MODEL',
    'STUDENT_MODEL_BASE_CONFIG', 'get_student_model_config', 'get_model_short_name',
    'normalize_gpu_ids',

    # SFT
    'MODEL_NAME_TO_SHORT', 'get_sft_model_name', 'get_sft_idmas_model_name',

    # 도메인
    'DATASET_TO_DOMAIN', 'TRAINING_DATASETS', 'DOMAIN_CONFIG',
    'DATA_DIR', 'get_available_domains', 'get_eval_datasets_for_domain',
    'get_training_datasets_for_domain', 'get_instructional_goal',

    # 경로
    'get_design_output_dir', 'get_domain_data_dirs',

    # 레거시
    'DESIGN_MODEL_CONFIG',
]
