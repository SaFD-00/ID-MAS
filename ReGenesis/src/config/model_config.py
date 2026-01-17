"""Model configurations for ReGenesis multi-model training.

Supports Llama 3.x and Qwen2.5 model families.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    template: str
    system_message: str
    stop_tokens: list
    tensor_parallel: int
    training_strategy: str  # "full" or "lora"
    max_model_len: int = 4096
    dtype: str = "bfloat16"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "template": self.template,
            "system_message": self.system_message,
            "stop_tokens": self.stop_tokens,
            "tensor_parallel": self.tensor_parallel,
            "training_strategy": self.training_strategy,
            "max_model_len": self.max_model_len,
            "dtype": self.dtype,
        }


# Model configurations for all supported models
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    # Llama 3.1 Models
    "meta-llama/Llama-3.1-8B-Instruct": ModelConfig(
        name="meta-llama/Llama-3.1-8B-Instruct",
        template="llama-3",
        system_message="You are a helpful, respectful and honest assistant.",
        stop_tokens=["<|eot_id|>", "<|end_of_text|>"],
        tensor_parallel=1,
        training_strategy="full",
        max_model_len=4096,
    ),
    "meta-llama/Llama-3.1-70B-Instruct": ModelConfig(
        name="meta-llama/Llama-3.1-70B-Instruct",
        template="llama-3",
        system_message="You are a helpful, respectful and honest assistant.",
        stop_tokens=["<|eot_id|>", "<|end_of_text|>"],
        tensor_parallel=4,
        training_strategy="lora",
        max_model_len=4096,
    ),

    # Llama 3.2 Models
    "meta-llama/Llama-3.2-3B-Instruct": ModelConfig(
        name="meta-llama/Llama-3.2-3B-Instruct",
        template="llama-3",
        system_message="You are a helpful, respectful and honest assistant.",
        stop_tokens=["<|eot_id|>", "<|end_of_text|>"],
        tensor_parallel=1,
        training_strategy="full",
        max_model_len=4096,
    ),

    # Llama 3.3 Models
    "meta-llama/Llama-3.3-70B-Instruct": ModelConfig(
        name="meta-llama/Llama-3.3-70B-Instruct",
        template="llama-3",
        system_message="You are a helpful, respectful and honest assistant.",
        stop_tokens=["<|eot_id|>", "<|end_of_text|>"],
        tensor_parallel=4,
        training_strategy="lora",
        max_model_len=4096,
    ),

    # Qwen2.5 Models
    "Qwen/Qwen2.5-3B-Instruct": ModelConfig(
        name="Qwen/Qwen2.5-3B-Instruct",
        template="qwen",
        system_message="You are a helpful assistant.",
        stop_tokens=["<|im_end|>", "<|endoftext|>"],
        tensor_parallel=1,
        training_strategy="full",
        max_model_len=4096,
    ),
    "Qwen/Qwen2.5-7B-Instruct": ModelConfig(
        name="Qwen/Qwen2.5-7B-Instruct",
        template="qwen",
        system_message="You are a helpful assistant.",
        stop_tokens=["<|im_end|>", "<|endoftext|>"],
        tensor_parallel=1,
        training_strategy="full",
        max_model_len=4096,
    ),
    "Qwen/Qwen2.5-14B-Instruct": ModelConfig(
        name="Qwen/Qwen2.5-14B-Instruct",
        template="qwen",
        system_message="You are a helpful assistant.",
        stop_tokens=["<|im_end|>", "<|endoftext|>"],
        tensor_parallel=2,
        training_strategy="full",
        max_model_len=4096,
    ),
    "Qwen/Qwen2.5-72B-Instruct": ModelConfig(
        name="Qwen/Qwen2.5-72B-Instruct",
        template="qwen",
        system_message="You are a helpful assistant.",
        stop_tokens=["<|im_end|>", "<|endoftext|>"],
        tensor_parallel=4,
        training_strategy="lora",
        max_model_len=4096,
    ),
}

# Model name aliases for convenience
MODEL_ALIASES: Dict[str, str] = {
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration by name or alias.

    Args:
        model_name: Model name or alias (e.g., "llama-3.1-8b" or "meta-llama/Llama-3.1-8B-Instruct")

    Returns:
        ModelConfig object for the specified model

    Raises:
        ValueError: If model is not found
    """
    # Check if it's an alias
    if model_name in MODEL_ALIASES:
        model_name = MODEL_ALIASES[model_name]

    if model_name not in MODEL_CONFIGS:
        available = list(MODEL_CONFIGS.keys()) + list(MODEL_ALIASES.keys())
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {available}"
        )

    return MODEL_CONFIGS[model_name]


def list_available_models() -> list:
    """List all available model names and aliases."""
    return list(MODEL_CONFIGS.keys()) + list(MODEL_ALIASES.keys())
