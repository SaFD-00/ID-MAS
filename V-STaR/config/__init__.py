"""V-STaR Configuration Module"""

from .models import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    get_model_short_name,
    is_valid_model,
)
from .domains import (
    DOMAIN_CONFIG,
    get_train_datasets,
    get_eval_datasets,
    get_data_path,
)
from .training import (
    TrainingConfig,
    DPOConfig,
    LoRAConfig,
    DEFAULT_TRAINING_CONFIG,
)
from .paths import (
    PROJECT_ROOT,
    DATA_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    get_checkpoint_path,
    get_output_path,
)

__all__ = [
    # models
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL",
    "get_model_short_name",
    "is_valid_model",
    # domains
    "DOMAIN_CONFIG",
    "get_train_datasets",
    "get_eval_datasets",
    "get_data_path",
    # training
    "TrainingConfig",
    "DPOConfig",
    "LoRAConfig",
    "DEFAULT_TRAINING_CONFIG",
    # paths
    "PROJECT_ROOT",
    "DATA_DIR",
    "CHECKPOINT_DIR",
    "OUTPUT_DIR",
    "get_checkpoint_path",
    "get_output_path",
]
