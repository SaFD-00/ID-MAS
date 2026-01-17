"""V-STaR Generator Model"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
)

from config.training import TrainingConfig, get_lora_config, get_generation_config
from config.paths import get_checkpoint_path
from .model_cache import get_model_cache


class VSTaRGenerator:
    """
    V-STaR Generator Model

    Responsible for:
    1. Generating candidate solutions for problems
    2. SFT training on correct solutions
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[TrainingConfig] = None,
        device: str = "cuda",
        use_lora: bool = True,
    ):
        """
        Initialize Generator

        Args:
            model_name: HuggingFace model name
            config: Training configuration
            device: Device to use
            use_lora: Whether to use LoRA
        """
        self.model_name = model_name
        self.config = config or TrainingConfig()
        self.device = device
        self.use_lora = use_lora

        # Load model and tokenizer
        cache = get_model_cache()
        self.model, self.tokenizer = cache.get_or_load(model_name, device)

        # Apply LoRA if requested
        if use_lora and not isinstance(self.model, PeftModel):
            self._apply_lora()

        # Set generation config
        self.generation_config = GenerationConfig(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def _apply_lora(self) -> None:
        """Apply LoRA to the model"""
        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            target_modules=self.config.lora.target_modules,
            bias=self.config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        print(f"LoRA applied. Trainable params: {self.model.print_trainable_parameters()}")

    def generate(
        self,
        prompts: Union[str, List[str]],
        k: int = 1,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        return_full: bool = False,
    ) -> Union[List[str], List[List[str]]]:
        """
        Generate solutions for given prompts

        Args:
            prompts: Single prompt or list of prompts
            k: Number of solutions to generate per prompt
            temperature: Override temperature (optional)
            max_new_tokens: Override max tokens (optional)
            return_full: If True, return full output including prompt

        Returns:
            If k=1: List of solutions (one per prompt)
            If k>1: List of lists of solutions
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        # Prepare generation config
        gen_config = GenerationConfig(
            temperature=temperature or self.config.temperature,
            top_p=self.config.top_p,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=k,
        )

        results = []
        self.model.eval()

        with torch.no_grad():
            for prompt in prompts:
                solutions = self._generate_single(prompt, k, gen_config, return_full)
                if k == 1:
                    results.append(solutions[0])
                else:
                    results.append(solutions)

        return results

    def _generate_single(
        self,
        prompt: str,
        k: int,
        gen_config: GenerationConfig,
        return_full: bool,
    ) -> List[str]:
        """Generate k solutions for a single prompt"""

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.sft.max_seq_length - gen_config.max_new_tokens,
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # Generate
        outputs = self.model.generate(
            **inputs,
            generation_config=gen_config,
        )

        # Decode
        solutions = []
        prompt_length = inputs["input_ids"].shape[1]

        for output in outputs:
            if return_full:
                text = self.tokenizer.decode(output, skip_special_tokens=True)
            else:
                text = self.tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
            solutions.append(text.strip())

        return solutions

    def generate_batch(
        self,
        prompts: List[str],
        k: int = 1,
        batch_size: int = 4,
        show_progress: bool = True,
    ) -> List[List[str]]:
        """
        Generate solutions in batches

        Args:
            prompts: List of prompts
            k: Number of solutions per prompt
            batch_size: Batch size for generation
            show_progress: Show progress bar

        Returns:
            List of lists of solutions
        """
        from tqdm import tqdm

        all_solutions = []
        iterator = range(0, len(prompts), batch_size)

        if show_progress:
            iterator = tqdm(iterator, desc="Generating solutions")

        for i in iterator:
            batch_prompts = prompts[i:i + batch_size]
            batch_solutions = []

            for prompt in batch_prompts:
                solutions = self.generate(prompt, k=k)
                if k == 1:
                    batch_solutions.append([solutions])
                else:
                    batch_solutions.append(solutions)

            all_solutions.extend(batch_solutions)

        return all_solutions

    def save(self, path: Union[str, Path]) -> None:
        """Save model to path"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if isinstance(self.model, PeftModel):
            # Save LoRA adapter
            self.model.save_pretrained(path)
        else:
            # Save full model
            self.model.save_pretrained(path)

        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        base_model_name: Optional[str] = None,
        config: Optional[TrainingConfig] = None,
        device: str = "cuda",
    ) -> "VSTaRGenerator":
        """
        Load model from path

        Args:
            path: Path to saved model
            base_model_name: Base model name (required for LoRA)
            config: Training configuration
            device: Device to use

        Returns:
            Loaded VSTaRGenerator
        """
        path = Path(path)
        config = config or TrainingConfig()

        # Check if this is a LoRA adapter
        is_lora = (path / "adapter_config.json").exists()

        if is_lora:
            if base_model_name is None:
                raise ValueError("base_model_name required for loading LoRA adapter")

            # Load base model
            generator = cls(
                model_name=base_model_name,
                config=config,
                device=device,
                use_lora=False,
            )

            # Load LoRA adapter
            generator.model = PeftModel.from_pretrained(
                generator.model,
                path,
            )
        else:
            # Load full model
            cache = get_model_cache()
            model, tokenizer = cache.load_from_checkpoint(path, device)

            generator = cls.__new__(cls)
            generator.model_name = str(path)
            generator.config = config
            generator.device = device
            generator.use_lora = False
            generator.model = model
            generator.tokenizer = tokenizer

        return generator

    def get_model_for_training(self) -> PreTrainedModel:
        """Get model for training"""
        return self.model

    def set_train_mode(self) -> None:
        """Set model to training mode"""
        self.model.train()

    def set_eval_mode(self) -> None:
        """Set model to evaluation mode"""
        self.model.eval()


def create_generator(
    model_name: str,
    config: Optional[TrainingConfig] = None,
    use_lora: bool = True,
    device: str = "cuda",
) -> VSTaRGenerator:
    """
    Factory function to create a generator

    Args:
        model_name: Model name
        config: Training configuration
        use_lora: Whether to use LoRA
        device: Device

    Returns:
        VSTaRGenerator instance
    """
    return VSTaRGenerator(
        model_name=model_name,
        config=config,
        device=device,
        use_lora=use_lora,
    )
