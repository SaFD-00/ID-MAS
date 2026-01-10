"""
Domain and dataset configuration.
"""
from config.api import PROJECT_ROOT

# =============================================================================
# Domain-based Configuration
# =============================================================================

# Terminal Goals for each training dataset
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
        "eval_datasets": ["gsm8k", "math", "svamp", "asdiv", "mawps", "mmlu"],
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


def get_terminal_goal(dataset: str) -> str:
    """Get Terminal Goal for a training dataset."""
    if dataset not in TERMINAL_GOALS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(TERMINAL_GOALS.keys())}")
    return TERMINAL_GOALS[dataset]
