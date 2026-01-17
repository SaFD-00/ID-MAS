"""Multi-model fine-tuning for ReGenesis with LoRA support.

Refactored from finetune_code.py with:
- Multi-model support (Llama 3.x, Qwen2.5)
- LoRA integration for large models (70B+)
- Configuration-based training
"""

import sys
import os
import copy
import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Any

import torch
import transformers
import datasets
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)

# LoRA imports
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

from src.config import get_model_config, get_training_config, LORA_CONFIG
from src.config.training_config import get_size_category
from src.reasoning.template_utils import get_instruction_prefix

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Model configuration arguments."""
    model_name_or_path: str = field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for training"}
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to load model in 4-bit quantization"}
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to load model in 8-bit quantization"}
    )


@dataclass
class DataArguments:
    """Data configuration arguments."""
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data JSON file"}
    )
    dataset_type: str = field(
        default="math",
        metadata={"help": "Type of dataset: math, logical, commonsense"}
    )


@dataclass
class CustomTrainingArguments(transformers.TrainingArguments):
    """Custom training arguments."""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length"}
    )
    # LoRA specific arguments
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})


def load_json_data(file_path: str) -> List[Dict]:
    """Load training data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding for new special tokens."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning with multi-model support."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        dataset_type: str = "math",
        model_max_length: int = 4096,
    ):
        super(SupervisedDataset, self).__init__()
        logging.info(f"Loading data from: {data_path}")

        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.model_max_length = model_max_length
        self.instruction_prefix = get_instruction_prefix(dataset_type)

        list_data_dict = load_json_data(data_path)
        logging.info(f"Loaded {len(list_data_dict)} examples")

        self.input_ids = []
        self.labels = []
        self.attention_masks = []

        self._preprocess_data(list_data_dict)

    def _preprocess_data(self, list_data_dict: List[Dict]):
        """Preprocess and tokenize data."""
        for example in list_data_dict:
            instruction = example["instruction"]
            answer = example["answer"]

            # Format messages for chat template
            messages = [
                {
                    "role": "user",
                    "content": self.instruction_prefix + instruction
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]

            # Tokenize with chat template
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Tokenize for input
            source_messages = [messages[0]]
            source_text = self.tokenizer.apply_chat_template(
                source_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Tokenize
            full_tokenized = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.model_max_length,
                padding=False,
                return_tensors=None,
            )

            source_tokenized = self.tokenizer(
                source_text,
                truncation=True,
                max_length=self.model_max_length,
                padding=False,
                return_tensors=None,
            )

            input_ids = torch.tensor(full_tokenized["input_ids"])
            labels = input_ids.clone()

            # Mask source tokens in labels
            source_len = len(source_tokenized["input_ids"])
            labels[:source_len] = IGNORE_INDEX

            self.input_ids.append(input_ids)
            self.labels.append(labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
        )


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels")
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def get_quantization_config(
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> Optional[BitsAndBytesConfig]:
    """Get quantization configuration."""
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    return None


def get_lora_config(
    r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> LoraConfig:
    """Get LoRA configuration."""
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def trainer_save_model_safe(trainer: Trainer):
    """Safe model saving with FSDP support."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


class TrainingPipeline:
    """Training pipeline for multi-model fine-tuning."""

    def __init__(
        self,
        model_name: str,
        data_path: str,
        output_dir: str,
        dataset_type: str = "math",
        use_lora: bool = None,
        load_in_4bit: bool = False,
        **training_kwargs,
    ):
        self.model_name = model_name
        self.data_path = data_path
        self.output_dir = output_dir
        self.dataset_type = dataset_type
        self.load_in_4bit = load_in_4bit

        # Get model config
        self.model_config = get_model_config(model_name)

        # Determine if LoRA should be used
        if use_lora is None:
            self.use_lora = self.model_config.training_strategy == "lora"
        else:
            self.use_lora = use_lora

        # Get training config
        size_category = get_size_category(model_name)
        self.training_config = get_training_config(size_category, self.use_lora)

        # Override with kwargs
        self.training_config.update(training_kwargs)

    def load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        logger.info(f"Use LoRA: {self.use_lora}, 4-bit: {self.load_in_4bit}")

        # Get quantization config
        quantization_config = None
        if self.use_lora and self.load_in_4bit:
            quantization_config = get_quantization_config(load_in_4bit=True)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto" if self.use_lora else None,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=self.training_config.get("model_max_length", 4096),
            padding_side="right",
            trust_remote_code=True,
        )

        # Add pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

        # Apply LoRA if needed
        if self.use_lora:
            if self.load_in_4bit:
                model = prepare_model_for_kbit_training(model)

            lora_config = get_lora_config(
                r=self.training_config.get("lora", {}).get("r", 64),
                lora_alpha=self.training_config.get("lora", {}).get("lora_alpha", 16),
                lora_dropout=self.training_config.get("lora", {}).get("lora_dropout", 0.05),
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        return model, tokenizer

    def prepare_dataset(self, tokenizer):
        """Prepare training dataset."""
        return SupervisedDataset(
            data_path=self.data_path,
            tokenizer=tokenizer,
            dataset_type=self.dataset_type,
            model_max_length=self.training_config.get("model_max_length", 4096),
        )

    def train(self):
        """Run training."""
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()

        # Prepare dataset
        train_dataset = self.prepare_dataset(tokenizer)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.training_config.get("learning_rate", 1e-6),
            num_train_epochs=self.training_config.get("num_train_epochs", 3),
            per_device_train_batch_size=self.training_config.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=self.training_config.get("gradient_accumulation_steps", 8),
            warmup_ratio=self.training_config.get("warmup_ratio", 0.03),
            lr_scheduler_type=self.training_config.get("lr_scheduler_type", "cosine"),
            bf16=self.training_config.get("bf16", True),
            gradient_checkpointing=self.training_config.get("gradient_checkpointing", True),
            logging_steps=self.training_config.get("logging_steps", 10),
            save_steps=self.training_config.get("save_steps", 500),
            save_total_limit=self.training_config.get("save_total_limit", 3),
            dataloader_num_workers=self.training_config.get("dataloader_num_workers", 4),
            seed=self.training_config.get("seed", 42),
            report_to="tensorboard",
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save
        logger.info("Saving model...")
        model.config.use_cache = True
        trainer.save_state()

        if self.use_lora:
            # Save LoRA adapter
            model.save_pretrained(self.output_dir)
        else:
            trainer_save_model_safe(trainer)

        logger.info(f"Model saved to: {self.output_dir}")
        return self.output_dir


def train():
    """Main training entry point."""
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    # Run training
    pipeline = TrainingPipeline(
        model_name=model_args.model_name_or_path,
        data_path=data_args.data_path,
        output_dir=training_args.output_dir,
        dataset_type=data_args.dataset_type,
        use_lora=model_args.use_lora,
        load_in_4bit=model_args.load_in_4bit,
        learning_rate=training_args.learning_rate,
        num_train_epochs=int(training_args.num_train_epochs),
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        model_max_length=training_args.model_max_length,
    )

    pipeline.train()


if __name__ == "__main__":
    train()
