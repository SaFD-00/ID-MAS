"""Domain and dataset configuration for V-STaR"""

from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum


class AnswerType(Enum):
    """Answer types for different datasets"""
    MCQ = "mcq"          # Multiple choice (A/B/C/D)
    NUMERIC = "numeric"  # Numeric answer (GSM8K, SVAMP)
    LATEX = "latex"      # LaTeX expression (MATH)
    TEXT = "text"        # Free text (BBH)
    BOOLEAN = "boolean"  # Yes/No (StrategyQA)


# Domain configuration
DOMAIN_CONFIG: Dict[str, Dict] = {
    "math": {
        "training_datasets": ["gsm8k", "math"],
        "eval_datasets": ["gsm8k", "math", "svamp", "asdiv", "mawps"],
        "description": "Mathematical reasoning",
    },
    "logical": {
        "training_datasets": ["reclor"],
        "eval_datasets": ["reclor", "anli_r2", "anli_r3", "bbh"],
        "description": "Logical reasoning",
    },
    "commonsense": {
        "training_datasets": ["arc_c"],
        "eval_datasets": ["arc_c", "strategyqa", "openbookqa"],
        "description": "Commonsense reasoning",
    },
}

# Dataset-specific configuration
DATASET_CONFIG: Dict[str, Dict] = {
    # Math domain
    "gsm8k": {
        "domain": "math",
        "answer_type": AnswerType.NUMERIC,
        "train_file": "gsm8k_train.json",
        "test_file": "gsm8k_test.json",
        "description": "Grade school math word problems",
    },
    "math": {
        "domain": "math",
        "answer_type": AnswerType.LATEX,
        "train_file": "math_train.json",
        "test_file": "math_test.json",
        "description": "Competition mathematics",
    },
    "svamp": {
        "domain": "math",
        "answer_type": AnswerType.NUMERIC,
        "train_file": None,
        "test_file": "svamp_test.json",
        "description": "Simple math word problems",
    },
    "asdiv": {
        "domain": "math",
        "answer_type": AnswerType.NUMERIC,
        "train_file": None,
        "test_file": "asdiv_test.json",
        "description": "Arithmetic division problems",
    },
    "mawps": {
        "domain": "math",
        "answer_type": AnswerType.NUMERIC,
        "train_file": None,
        "test_file": "mawps_test.json",
        "description": "Math word problem benchmark",
    },
    # Logical domain
    "reclor": {
        "domain": "logical",
        "answer_type": AnswerType.MCQ,
        "train_file": "reclor_train.json",
        "test_file": "reclor_test.json",
        "description": "Reading comprehension + logical reasoning",
    },
    "anli_r2": {
        "domain": "logical",
        "answer_type": AnswerType.MCQ,
        "train_file": None,
        "test_file": "anli_r2_test.json",
        "description": "Adversarial NLI Round 2",
    },
    "anli_r3": {
        "domain": "logical",
        "answer_type": AnswerType.MCQ,
        "train_file": None,
        "test_file": "anli_r3_test.json",
        "description": "Adversarial NLI Round 3",
    },
    "bbh": {
        "domain": "logical",
        "answer_type": AnswerType.TEXT,
        "train_file": None,
        "test_file": "bbh_test.json",
        "description": "Big Bench Hard",
    },
    # Commonsense domain
    "arc_c": {
        "domain": "commonsense",
        "answer_type": AnswerType.MCQ,
        "train_file": "arc_c_train.json",
        "test_file": "arc_c_test.json",
        "description": "AI2 Reasoning Challenge",
    },
    "strategyqa": {
        "domain": "commonsense",
        "answer_type": AnswerType.BOOLEAN,
        "train_file": None,
        "test_file": "strategyqa_test.json",
        "description": "Strategy question answering",
    },
    "openbookqa": {
        "domain": "commonsense",
        "answer_type": AnswerType.MCQ,
        "train_file": None,
        "test_file": "openbookqa_test.json",
        "description": "Open book question answering",
    },
}


def get_train_datasets(domain: str) -> List[str]:
    """Get training datasets for a domain"""
    if domain == "all":
        datasets = []
        for d in DOMAIN_CONFIG.values():
            datasets.extend(d["training_datasets"])
        return datasets
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain: {domain}")
    return DOMAIN_CONFIG[domain]["training_datasets"]


def get_eval_datasets(domain: str) -> List[str]:
    """Get evaluation datasets for a domain"""
    if domain == "all":
        datasets = []
        for d in DOMAIN_CONFIG.values():
            datasets.extend(d["eval_datasets"])
        return datasets
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain: {domain}")
    return DOMAIN_CONFIG[domain]["eval_datasets"]


def get_dataset_config(dataset: str) -> Dict:
    """Get configuration for a specific dataset"""
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset}")
    return DATASET_CONFIG[dataset]


def get_answer_type(dataset: str) -> AnswerType:
    """Get answer type for a dataset"""
    return get_dataset_config(dataset)["answer_type"]


def get_data_path(
    data_dir: Path,
    dataset: str,
    split: str = "train"
) -> Optional[Path]:
    """
    Get data file path for a dataset

    Args:
        data_dir: Base data directory
        dataset: Dataset name
        split: "train" or "test"

    Returns:
        Path to data file or None if not available
    """
    config = get_dataset_config(dataset)
    domain = config["domain"]

    if split == "train":
        filename = config["train_file"]
        subdir = "train"
    else:
        filename = config["test_file"]
        subdir = "eval"

    if filename is None:
        return None

    return data_dir / domain / subdir / "data" / filename


def get_all_domains() -> List[str]:
    """Get list of all domains"""
    return list(DOMAIN_CONFIG.keys())


def get_domain_for_dataset(dataset: str) -> str:
    """Get domain for a dataset"""
    return get_dataset_config(dataset)["domain"]
