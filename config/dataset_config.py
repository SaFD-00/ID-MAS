"""
Dataset Configuration

Contains configuration for each supported dataset including
HuggingFace paths, split names, and answer types.
"""
from typing import Dict, Any, List

# Dataset-specific configurations
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gsm8k": {
        "hf_name": "openai/gsm8k",
        "hf_config": "main",
        "train_split": "train",
        "test_split": "test",
        "answer_type": "numeric",
        "domain": "math_logic",
        "description": "Grade School Math 8K - Elementary math word problems",
    },
    "math": {
        "hf_name": "hendrycks/competition_math",
        "hf_config": None,
        "train_split": "train",
        "test_split": "test",
        "answer_type": "latex",
        "domain": "math_logic",
        "default_levels": [1, 2, 3],
        "description": "Competition Mathematics - High school competition problems",
    },
    "svamp": {
        "hf_name": "ChilleD/SVAMP",
        "hf_config": None,
        "train_split": None,  # SVAMP is evaluation only
        "test_split": "test",
        "answer_type": "numeric",
        "domain": "math_logic",
        "description": "Simple Variations on Arithmetic Math Problems - Robustness evaluation",
    },
    "bbh": {
        "hf_name": "lukaemon/bbh",
        "hf_config": "required",  # Subtask required
        "train_split": None,
        "test_split": "test",
        "answer_type": "mixed",  # Varies by subtask
        "domain": "math_logic",
        "description": "Big Bench Hard - 23 challenging reasoning tasks",
    },
    # SciBench configuration (currently not used - knowledge domain removed)
    "scibench": {
        "hf_name": "xw27/scibench",
        "hf_config": None,
        "train_split": "train",
        "test_split": "test",
        "answer_type": "numeric",
        "domain": "science_knowledge",
        "description": "College-level science problems (Physics, Chemistry)",
    },
    # ARC configuration (currently not used - knowledge domain removed)
    "arc": {
        "hf_name": "allenai/ai2_arc",
        "hf_config": "ARC-Challenge",  # Default to Challenge
        "train_split": "train",
        "test_split": "test",
        "answer_type": "mcq",
        "domain": "science_knowledge",
        "description": "AI2 Reasoning Challenge - Elementary science MCQ",
    },
    "mmlu": {
        "hf_name": "cais/mmlu",
        "hf_config": "required",  # Subject required
        "train_split": "validation",  # MMLU uses validation for training
        "test_split": "test",
        "answer_type": "mcq",
        "domain": "science_knowledge",
        "description": "Massive Multitask Language Understanding - 57 subjects",
    },
}


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """
    Get configuration for a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If dataset is unknown
    """
    config = DATASET_CONFIGS.get(dataset_name.lower())
    if config is None:
        available = list(DATASET_CONFIGS.keys())
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {available}"
        )
    return config


def get_datasets_by_domain(domain: str) -> List[str]:
    """
    Get datasets for a specific domain.

    Args:
        domain: "math_logic" or "science_knowledge"

    Returns:
        List of dataset names
    """
    return [
        name for name, config in DATASET_CONFIGS.items()
        if config["domain"] == domain
    ]


def get_training_datasets(domain: str) -> List[str]:
    """
    Get datasets suitable for training in a domain.

    Args:
        domain: "math_logic" or "science_knowledge"

    Returns:
        List of dataset names that have a train split
    """
    return [
        name for name, config in DATASET_CONFIGS.items()
        if config["domain"] == domain and config["train_split"] is not None
    ]


def get_evaluation_datasets(domain: str) -> List[str]:
    """
    Get datasets suitable for evaluation in a domain.

    Args:
        domain: "math_logic" or "science_knowledge"

    Returns:
        List of dataset names that have a test split
    """
    return [
        name for name, config in DATASET_CONFIGS.items()
        if config["domain"] == domain and config["test_split"] is not None
    ]


# BBH subtasks for reference
BBH_SUBTASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

# MMLU STEM subjects for reference
MMLU_STEM_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
]
