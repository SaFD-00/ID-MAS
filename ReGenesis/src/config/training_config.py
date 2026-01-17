"""Training configurations for ReGenesis.

Based on hyperparameters from the ReGenesis paper (ICLR 2025).
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 1e-6
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # effective batch size = 16
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    weight_decay: float = 0.1
    model_max_length: int = 4096
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    dataloader_num_workers: int = 4
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_ratio": self.warmup_ratio,
            "lr_scheduler_type": self.lr_scheduler_type,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "max_grad_norm": self.max_grad_norm,
            "weight_decay": self.weight_decay,
            "model_max_length": self.model_max_length,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "dataloader_num_workers": self.dataloader_num_workers,
            "seed": self.seed,
        }


@dataclass
class LoRAConfig:
    """LoRA configuration for large models (70B+)."""
    r: int = 64
    lora_alpha: int = 16
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    use_gradient_checkpointing: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "task_type": self.task_type,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
        }


@dataclass
class GenerationConfig:
    """Configuration for reasoning path generation."""
    temperature: float = 0.85
    top_p: float = 0.9
    max_tokens: int = 2048
    num_samples_per_task: int = 25

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "num_samples_per_task": self.num_samples_per_task,
        }


# Default configurations
DEFAULT_TRAINING_CONFIG = TrainingConfig()
LORA_CONFIG = LoRAConfig()
DEFAULT_GENERATION_CONFIG = GenerationConfig()

# Training configurations for different model sizes
TRAINING_CONFIGS_BY_SIZE: Dict[str, TrainingConfig] = {
    "small": TrainingConfig(  # 3B models
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
    ),
    "medium": TrainingConfig(  # 7B-8B models
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
    ),
    "large": TrainingConfig(  # 14B models
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
    ),
    "xlarge": TrainingConfig(  # 70B+ models (with LoRA)
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        learning_rate=2e-5,  # Higher LR for LoRA
    ),
}


def get_training_config(
    model_size: str = "medium",
    use_lora: bool = False
) -> Dict[str, Any]:
    """Get training configuration for a specific model size.

    Args:
        model_size: One of "small", "medium", "large", "xlarge"
        use_lora: Whether to include LoRA configuration

    Returns:
        Dictionary containing training configuration
    """
    if model_size not in TRAINING_CONFIGS_BY_SIZE:
        raise ValueError(
            f"Unknown model size: {model_size}. "
            f"Available: {list(TRAINING_CONFIGS_BY_SIZE.keys())}"
        )

    config = TRAINING_CONFIGS_BY_SIZE[model_size].to_dict()

    if use_lora:
        config["lora"] = LORA_CONFIG.to_dict()

    return config


def get_size_category(model_name: str) -> str:
    """Determine model size category from model name.

    Args:
        model_name: Full model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")

    Returns:
        Size category: "small", "medium", "large", or "xlarge"
    """
    model_lower = model_name.lower()

    if "3b" in model_lower:
        return "small"
    elif "7b" in model_lower or "8b" in model_lower:
        return "medium"
    elif "14b" in model_lower:
        return "large"
    elif "70b" in model_lower or "72b" in model_lower:
        return "xlarge"
    else:
        return "medium"  # Default
