"""SFT Trainer for V-STaR Generator"""

from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl import SFTTrainer as TRLSFTTrainer, SFTConfig
from peft import LoraConfig, TaskType, get_peft_model

from config.training import TrainingConfig


class SFTDataset(Dataset):
    """Dataset for SFT training"""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Combine prompt and solution
        prompt = item.get("prompt", item.get("question", ""))
        solution = item.get("solution", item.get("completion", ""))

        full_text = f"{prompt}\n\n{solution}"

        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": encodings["input_ids"].squeeze(0),
        }


class SFTTrainer:
    """
    SFT Trainer for V-STaR Generator

    Uses trl SFTTrainer for efficient supervised fine-tuning
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[TrainingConfig] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize SFT Trainer

        Args:
            model: Model to train
            tokenizer: Tokenizer
            config: Training configuration
            output_dir: Output directory for checkpoints
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TrainingConfig()
        self.output_dir = output_dir or "./checkpoints/sft"

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def train(
        self,
        train_data: List[Dict[str, Any]],
        eval_data: Optional[List[Dict[str, Any]]] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
        gradient_accumulation_steps: int = 4,
        warmup_ratio: float = 0.1,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the model using SFT

        Args:
            train_data: Training data (list of dicts with prompt/solution)
            eval_data: Optional evaluation data
            num_epochs: Number of training epochs
            batch_size: Per-device batch size
            learning_rate: Learning rate
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            logging_steps: Log every N steps
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_ratio: Warmup ratio
            resume_from_checkpoint: Resume from checkpoint path

        Returns:
            Training metrics
        """
        num_epochs = num_epochs or self.config.sft.num_epochs
        batch_size = batch_size or self.config.sft.batch_size
        learning_rate = learning_rate or self.config.sft.learning_rate

        # Format data for trl SFTTrainer
        def formatting_func(examples):
            texts = []
            for prompt, solution in zip(examples["prompt"], examples["solution"]):
                text = f"{prompt}\n\n{solution}"
                texts.append(text)
            return texts

        # Convert to HuggingFace Dataset format
        from datasets import Dataset as HFDataset

        train_dict = {
            "prompt": [d.get("prompt", d.get("question", "")) for d in train_data],
            "solution": [d.get("solution", d.get("completion", "")) for d in train_data],
        }
        train_dataset = HFDataset.from_dict(train_dict)

        eval_dataset = None
        if eval_data:
            eval_dict = {
                "prompt": [d.get("prompt", d.get("question", "")) for d in eval_data],
                "solution": [d.get("solution", d.get("completion", "")) for d in eval_data],
            }
            eval_dataset = HFDataset.from_dict(eval_dict)

        # Configure SFT
        sft_config = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            logging_steps=logging_steps,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            evaluation_strategy="steps" if eval_dataset else "no",
            max_seq_length=self.config.sft.max_seq_length,
            packing=False,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            optim="adamw_torch",
            report_to="none",
        )

        # Create trainer
        trainer = TRLSFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            formatting_func=formatting_func,
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

        print(f"SFT model saved to {path}")


def train_sft(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_data: List[Dict[str, Any]],
    eval_data: Optional[List[Dict[str, Any]]] = None,
    config: Optional[TrainingConfig] = None,
    output_dir: str = "./checkpoints/sft",
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function for SFT training

    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_data: Training data
        eval_data: Optional evaluation data
        config: Training configuration
        output_dir: Output directory
        **kwargs: Additional training arguments

    Returns:
        Training metrics
    """
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        output_dir=output_dir,
    )

    return trainer.train(
        train_data=train_data,
        eval_data=eval_data,
        **kwargs,
    )


def create_sft_trainer(
    model_name: str,
    config: Optional[TrainingConfig] = None,
    use_lora: bool = True,
    device: str = "cuda",
    output_dir: str = "./checkpoints/sft",
) -> SFTTrainer:
    """
    Factory function to create SFT trainer

    Args:
        model_name: HuggingFace model name
        config: Training configuration
        use_lora: Whether to use LoRA
        device: Device to use
        output_dir: Output directory

    Returns:
        SFTTrainer instance
    """
    from models.model_cache import get_model_cache

    config = config or TrainingConfig()
    cache = get_model_cache()

    model, tokenizer = cache.get_or_load(model_name, device)

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

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        output_dir=output_dir,
    )
