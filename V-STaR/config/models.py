"""Model configuration for V-STaR"""

from typing import List, Optional

# Supported models for V-STaR
AVAILABLE_MODELS: List[str] = [
    # Llama models
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    # Qwen models
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
]

DEFAULT_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"

# Model short names for file paths
MODEL_SHORT_NAMES = {
    "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b",
    "meta-llama/Llama-3.1-70B-Instruct": "llama-3.1-70b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama-3.2-3b",
    "meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b",
    "Qwen/Qwen2.5-3B-Instruct": "qwen2.5-3b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen2.5-7b",
    "Qwen/Qwen2.5-14B-Instruct": "qwen2.5-14b",
    "Qwen/Qwen2.5-72B-Instruct": "qwen2.5-72b",
}


def get_model_short_name(model_name: str) -> str:
    """Get short name for model (used in file paths)"""
    if model_name in MODEL_SHORT_NAMES:
        return MODEL_SHORT_NAMES[model_name]
    # Fallback: extract from model name
    return model_name.split("/")[-1].lower().replace("-instruct", "")


def is_valid_model(model_name: str) -> bool:
    """Check if model is supported"""
    return model_name in AVAILABLE_MODELS


def get_model_family(model_name: str) -> str:
    """Get model family (llama or qwen)"""
    if "llama" in model_name.lower():
        return "llama"
    elif "qwen" in model_name.lower():
        return "qwen"
    return "unknown"


def get_model_size(model_name: str) -> Optional[str]:
    """Extract model size from name"""
    import re
    match = re.search(r"(\d+)[Bb]", model_name)
    if match:
        return f"{match.group(1)}B"
    return None
