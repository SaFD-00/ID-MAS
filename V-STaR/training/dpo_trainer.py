"""DPO Trainer for V-STaR Verifier"""

from typing import Optional, Dict, Any, List, Union
from pathlib import Path

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from datasets import Dataset as HFDataset

from config.training import TrainingConfig
from data.preference_dataset import PreferencePair, PreferenceDataset


class DPOTrainerWrapper:
    """
    DPO Trainer Wrapper for V-STaR Verifier

    Uses trl DPOTrainer for efficient preference learning
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: Optional[PreTrainedModel],
        tokenizer: PreTrainedTokenizer,
        config: Optional[TrainingConfig] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize DPO Trainer

        Args:
            model: Model to train (policy model)
            ref_model: Reference model (G_SFT)
            tokenizer: Tokenizer
            config: Training configuration
            output_dir: Output directory for checkpoints
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config or TrainingConfig()
        self.output_dir = output_dir or "./checkpoints/dpo"

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def prepare_dataset(
        self,
        preference_pairs: Union[List[PreferencePair], PreferenceDataset, List[Dict]],
    ) -> HFDataset:
        """
        Prepare dataset for DPO training

        Args:
            preference_pairs: Preference pairs or PreferenceDataset

        Returns:
            HuggingFace Dataset
        """
        if isinstance(preference_pairs, PreferenceDataset):
            pairs = preference_pairs.pairs
        elif isinstance(preference_pairs, list) and len(preference_pairs) > 0:
            if isinstance(preference_pairs[0], PreferencePair):
                pairs = preference_pairs
            else:
                # Dict format
                pairs = [PreferencePair.from_dict(d) for d in preference_pairs]
        else:
            pairs = []

        # Convert to trl DPO format
        data = {
            "prompt": [p.prompt for p in pairs],
            "chosen": [p.chosen for p in pairs],
            "rejected": [p.rejected for p in pairs],
        }

        return HFDataset.from_dict(data)

    def train(
        self,
        train_data: Union[List[PreferencePair], PreferenceDataset, List[Dict]],
        eval_data: Optional[Union[List[PreferencePair], PreferenceDataset, List[Dict]]] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        beta: Optional[float] = None,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
        gradient_accumulation_steps: int = 4,
        warmup_ratio: float = 0.1,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the verifier using DPO

        Args:
            train_data: Training preference pairs
            eval_data: Optional evaluation preference pairs
            num_epochs: Number of training epochs
            batch_size: Per-device batch size
            learning_rate: Learning rate
            beta: DPO beta parameter
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            logging_steps: Log every N steps
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_ratio: Warmup ratio
            resume_from_checkpoint: Resume from checkpoint path

        Returns:
            Training metrics
        """
        num_epochs = num_epochs or self.config.dpo.num_epochs
        batch_size = batch_size or self.config.dpo.batch_size
        learning_rate = learning_rate or self.config.dpo.learning_rate
        beta = beta or self.config.dpo.beta

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_data)
        eval_dataset = self.prepare_dataset(eval_data) if eval_data else None

        # Configure DPO
        dpo_config = DPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            beta=beta,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            logging_steps=logging_steps,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            evaluation_strategy="steps" if eval_dataset else "no",
            max_length=self.config.dpo.max_length,
            max_prompt_length=self.config.dpo.max_prompt_length,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            optim="adamw_torch",
            report_to="none",
            remove_unused_columns=False,
        )

        # Create DPO trainer
        trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        # Train
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final model
        trainer.save_model()

        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save model and tokenizer"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        print(f"DPO model saved to {path}")


def train_dpo(
    model: PreTrainedModel,
    ref_model: Optional[PreTrainedModel],
    tokenizer: PreTrainedTokenizer,
    train_data: Union[List[PreferencePair], PreferenceDataset, List[Dict]],
    eval_data: Optional[Union[List[PreferencePair], PreferenceDataset, List[Dict]]] = None,
    config: Optional[TrainingConfig] = None,
    output_dir: str = "./checkpoints/dpo",
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function for DPO training

    Args:
        model: Model to train
        ref_model: Reference model
        tokenizer: Tokenizer
        train_data: Training preference pairs
        eval_data: Optional evaluation preference pairs
        config: Training configuration
        output_dir: Output directory
        **kwargs: Additional training arguments

    Returns:
        Training metrics
    """
    trainer = DPOTrainerWrapper(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=config,
        output_dir=output_dir,
    )

    return trainer.train(
        train_data=train_data,
        eval_data=eval_data,
        **kwargs,
    )


def create_dpo_trainer(
    model_name: str,
    ref_model: Optional[PreTrainedModel] = None,
    config: Optional[TrainingConfig] = None,
    use_lora: bool = True,
    device: str = "cuda",
    output_dir: str = "./checkpoints/dpo",
) -> DPOTrainerWrapper:
    """
    Factory function to create DPO trainer

    Args:
        model_name: HuggingFace model name
        ref_model: Reference model (if None, will load separately)
        config: Training configuration
        use_lora: Whether to use LoRA
        device: Device to use
        output_dir: Output directory

    Returns:
        DPOTrainerWrapper instance
    """
    from models.model_cache import get_model_cache

    config = config or TrainingConfig()
    cache = get_model_cache()

    model, tokenizer = cache.get_or_load(model_name, device)

    # Load reference model if not provided
    if ref_model is None:
        ref_model, _ = cache.get_or_load(model_name, device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

    if use_lora:
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=config.lora.target_modules,
            bias=config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    return DPOTrainerWrapper(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=config,
        output_dir=output_dir,
    )
