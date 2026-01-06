"""
Domain and dataset configuration.
"""
from config.api import PROJECT_ROOT

# =============================================================================
# Domain-based Configuration
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

# Data directory path
DATA_DIR = PROJECT_ROOT / "data"

# Domain configurations
DOMAIN_CONFIG = {
    "math": {
        "data_dir": DATA_DIR / "math",
        "training_datasets": ["gsm8k", "math"],
        "eval_datasets": ["gsm8k", "math", "svamp", "asdiv", "mawps", "mmlu"],
        "default_eval": "gsm8k"
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


def get_terminal_goal(dataset: str) -> str:
    """Get Terminal Goal for a training dataset."""
    if dataset not in TERMINAL_GOALS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(TERMINAL_GOALS.keys())}")
    return TERMINAL_GOALS[dataset]
