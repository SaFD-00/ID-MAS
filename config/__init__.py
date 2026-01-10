"""
Configuration module for ID-MAS.

Provides a unified interface while organizing config into logical submodules.
"""

# API credentials
from config.api import (
    PROJECT_ROOT,
    OPENAI_API_KEY,
    HF_TOKEN,
)

# Model configurations
from config.models import (
    AVAILABLE_TEACHER_MODELS,
    DEFAULT_TEACHER_MODEL,
    create_teacher_config,
    AVAILABLE_STUDENT_MODELS,
    DEFAULT_STUDENT_MODEL,
    STUDENT_MODEL_BASE_CONFIG,
    get_student_model_config,
    get_model_short_name,
)

# SFT model mappings
from config.sft import (
    MODEL_NAME_TO_SHORT,
    get_sft_model_name,
    get_sft_idmas_model_name,
)

# Domain and dataset config
from config.domains import (
    TERMINAL_GOALS,
    DATASET_TO_DOMAIN,
    TRAINING_DATASETS,
    DOMAIN_CONFIG,
    DATA_DIR,
    get_available_domains,
    get_eval_datasets_for_domain,
    get_training_datasets_for_domain,
    get_terminal_goal,
)

# Path helpers
from config.paths import (
    get_design_output_dir,
    get_model_data_dirs,
    get_dataset_model_dirs,
    get_domain_data_dirs,
)

# Legacy aliases for backward compatibility
DESIGN_MODEL_CONFIG = create_teacher_config(DEFAULT_TEACHER_MODEL)
STUDENT_MODEL_CONFIG = get_student_model_config(DEFAULT_STUDENT_MODEL)

LEARNING_LOOP_CONFIG = {
    "max_iterations": 5,  # 최대 반복 횟수
    "convergence_threshold": 0.9  # 수렴 기준 (루브릭 점수)
}

__all__ = [
    # API
    'PROJECT_ROOT', 'OPENAI_API_KEY', 'HF_TOKEN',

    # Models
    'AVAILABLE_TEACHER_MODELS', 'DEFAULT_TEACHER_MODEL',
    'create_teacher_config', 'AVAILABLE_STUDENT_MODELS', 'DEFAULT_STUDENT_MODEL',
    'STUDENT_MODEL_BASE_CONFIG', 'get_student_model_config', 'get_model_short_name',

    # SFT
    'MODEL_NAME_TO_SHORT', 'get_sft_model_name', 'get_sft_idmas_model_name',

    # Domains
    'TERMINAL_GOALS', 'DATASET_TO_DOMAIN', 'TRAINING_DATASETS', 'DOMAIN_CONFIG',
    'DATA_DIR', 'get_available_domains', 'get_eval_datasets_for_domain',
    'get_training_datasets_for_domain', 'get_terminal_goal',

    # Paths
    'get_design_output_dir', 'get_model_data_dirs',
    'get_dataset_model_dirs', 'get_domain_data_dirs',

    # Legacy
    'DESIGN_MODEL_CONFIG', 'STUDENT_MODEL_CONFIG', 'LEARNING_LOOP_CONFIG',
]
