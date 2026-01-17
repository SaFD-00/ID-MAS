"""Training configuration for V-STaR"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration"""
    r: int = 8                          # LoRA rank
    lora_alpha: int = 32                # LoRA alpha
    lora_dropout: float = 0.05          # LoRA dropout
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"                  # Bias type
    task_type: str = "CAUSAL_LM"        # Task type


@dataclass
class DPOConfig:
    """DPO (Direct Preference Optimization) configuration"""
    beta: float = 0.1                   # DPO beta parameter
    learning_rate: float = 5e-7         # Learning rate for DPO
    num_train_epochs: int = 1           # Number of epochs
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    max_length: int = 2048              # Max sequence length
    max_prompt_length: int = 1024       # Max prompt length
    loss_type: str = "sigmoid"          # DPO loss type


@dataclass
class SFTConfig:
    """SFT (Supervised Fine-Tuning) configuration"""
    learning_rate: float = 2e-5
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500


@dataclass
class TrainingConfig:
    """Main V-STaR training configuration"""
    # V-STaR specific (Algorithm 1 parameters)
    num_iterations: int = 3             # Number of V-STaR iterations (T)
    samples_per_query: int = 16         # Number of samples per query (k)
    best_of_k: int = 64                 # Best-of-k for evaluation
    max_pairs_per_question: Optional[int] = None  # Max preference pairs per question (None = unlimited)

    # Model configuration
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    use_lora: bool = True

    # Generation configuration
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 1024

    # Inference configuration
    num_candidates: int = 128           # Number of candidates for evaluation

    # Device configuration
    device: str = "cuda"
    fp16: bool = True
    bf16: bool = False

    # Sub-configurations
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)

    # Seed
    seed: int = 42

    @property
    def k(self) -> int:
        """Alias for samples_per_query (논문 Algorithm 1의 k)"""
        return self.samples_per_query

    @k.setter
    def k(self, value: int):
        """Set samples_per_query via k alias"""
        self.samples_per_query = value

    def __post_init__(self):
        """Validate configuration"""
        assert self.num_iterations >= 1, "num_iterations must be >= 1"
        assert self.samples_per_query >= 1, "samples_per_query must be >= 1"
        assert self.best_of_k >= 1, "best_of_k must be >= 1"
        assert 0.0 < self.temperature <= 2.0, "temperature must be in (0, 2]"


# Default configuration
DEFAULT_TRAINING_CONFIG = TrainingConfig()


def get_generation_config(config: TrainingConfig) -> dict:
    """Get generation configuration for model.generate()"""
    return {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_new_tokens": config.max_new_tokens,
        "do_sample": True,
        "pad_token_id": None,  # Set by model
    }


def get_lora_config(config: LoRAConfig) -> dict:
    """Get LoRA configuration for PEFT"""
    return {
        "r": config.r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": config.target_modules,
        "bias": config.bias,
        "task_type": config.task_type,
    }
