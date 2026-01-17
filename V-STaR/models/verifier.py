"""V-STaR Verifier Model"""

from typing import List, Optional, Dict, Any, Tuple, Union
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
)

from config.training import TrainingConfig
from .model_cache import get_model_cache


class VSTaRVerifier:
    """
    V-STaR Verifier Model

    Responsible for:
    1. Scoring candidate solutions
    2. Ranking solutions by likelihood
    3. DPO training with preference pairs
    """

    def __init__(
        self,
        model_name: str,
        ref_model: Optional[PreTrainedModel] = None,
        config: Optional[TrainingConfig] = None,
        device: str = "cuda",
        use_lora: bool = True,
    ):
        """
        Initialize Verifier

        Args:
            model_name: HuggingFace model name
            ref_model: Reference model (G_SFT) for DPO
            config: Training configuration
            device: Device to use
            use_lora: Whether to use LoRA
        """
        self.model_name = model_name
        self.config = config or TrainingConfig()
        self.device = device
        self.use_lora = use_lora
        self.ref_model = ref_model
        self.beta = self.config.dpo.beta

        # Load model and tokenizer
        cache = get_model_cache()
        self.model, self.tokenizer = cache.get_or_load(model_name, device)

        # Apply LoRA if requested
        if use_lora and not isinstance(self.model, PeftModel):
            self._apply_lora()

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

    def score(
        self,
        problem: str,
        solution: str,
    ) -> float:
        """
        Score a single solution

        Uses log likelihood: log V(y|x)

        Args:
            problem: Problem text
            solution: Solution text

        Returns:
            Score (log likelihood)
        """
        self.model.eval()

        # Combine problem and solution
        full_text = f"{problem}\n\nSolution:\n{solution}"

        # Tokenize
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.dpo.max_length,
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Calculate log likelihood of the solution part
            # Shift logits and labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()

            # Calculate cross entropy (negative log likelihood)
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Sum log likelihood (negative of loss)
            log_likelihood = -loss.sum().item()

        return log_likelihood

    def score_batch(
        self,
        problems: List[str],
        solutions: List[str],
        batch_size: int = 8,
    ) -> List[float]:
        """
        Score multiple solutions

        Args:
            problems: List of problems
            solutions: List of solutions
            batch_size: Batch size

        Returns:
            List of scores
        """
        assert len(problems) == len(solutions)

        scores = []
        for i in range(0, len(problems), batch_size):
            batch_problems = problems[i:i + batch_size]
            batch_solutions = solutions[i:i + batch_size]

            for problem, solution in zip(batch_problems, batch_solutions):
                score = self.score(problem, solution)
                scores.append(score)

        return scores

    def rank(
        self,
        problem: str,
        solutions: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Rank solutions by score

        Args:
            problem: Problem text
            solutions: List of candidate solutions

        Returns:
            List of (solution, score) tuples, sorted by score descending
        """
        scores = []
        for solution in solutions:
            score = self.score(problem, solution)
            scores.append((solution, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def select_best(
        self,
        problem: str,
        solutions: List[str],
    ) -> Tuple[str, float]:
        """
        Select the best solution

        Args:
            problem: Problem text
            solutions: List of candidate solutions

        Returns:
            Tuple of (best_solution, score)
        """
        ranked = self.rank(problem, solutions)
        return ranked[0]

    def get_logps(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> torch.Tensor:
        """
        Get log probabilities for prompt-response pairs

        Used for DPO training

        Args:
            prompts: List of prompts
            responses: List of responses

        Returns:
            Tensor of log probabilities
        """
        self.model.eval()
        logps = []

        with torch.no_grad():
            for prompt, response in zip(prompts, responses):
                full_text = f"{prompt}{response}"

                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.dpo.max_length,
                )
                inputs = {key: val.to(self.device) for key, val in inputs.items()}

                # Get prompt length for masking
                prompt_inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                )
                prompt_len = prompt_inputs["input_ids"].shape[1]

                outputs = self.model(**inputs)
                logits = outputs.logits

                # Calculate log probs only for response tokens
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()

                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = torch.gather(
                    log_probs,
                    dim=-1,
                    index=shift_labels.unsqueeze(-1)
                ).squeeze(-1)

                # Mask prompt tokens
                response_log_probs = token_log_probs[0, prompt_len - 1:]
                total_log_prob = response_log_probs.sum()
                logps.append(total_log_prob)

        return torch.stack(logps)

    def set_reference_model(self, ref_model: PreTrainedModel) -> None:
        """Set reference model for DPO"""
        self.ref_model = ref_model

    def save(self, path: Union[str, Path]) -> None:
        """Save model to path"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(path)
        else:
            self.model.save_pretrained(path)

        self.tokenizer.save_pretrained(path)
        print(f"Verifier saved to {path}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        base_model_name: Optional[str] = None,
        ref_model: Optional[PreTrainedModel] = None,
        config: Optional[TrainingConfig] = None,
        device: str = "cuda",
    ) -> "VSTaRVerifier":
        """Load verifier from path"""
        path = Path(path)
        config = config or TrainingConfig()

        is_lora = (path / "adapter_config.json").exists()

        if is_lora:
            if base_model_name is None:
                raise ValueError("base_model_name required for loading LoRA adapter")

            verifier = cls(
                model_name=base_model_name,
                ref_model=ref_model,
                config=config,
                device=device,
                use_lora=False,
            )

            verifier.model = PeftModel.from_pretrained(
                verifier.model,
                path,
            )
        else:
            cache = get_model_cache()
            model, tokenizer = cache.load_from_checkpoint(path, device)

            verifier = cls.__new__(cls)
            verifier.model_name = str(path)
            verifier.config = config
            verifier.device = device
            verifier.use_lora = False
            verifier.model = model
            verifier.tokenizer = tokenizer
            verifier.ref_model = ref_model
            verifier.beta = config.dpo.beta

        return verifier

    def get_model_for_training(self) -> PreTrainedModel:
        """Get model for training"""
        return self.model

    def set_train_mode(self) -> None:
        """Set model to training mode"""
        self.model.train()

    def set_eval_mode(self) -> None:
        """Set model to evaluation mode"""
        self.model.eval()


def create_verifier(
    model_name: str,
    ref_model: Optional[PreTrainedModel] = None,
    config: Optional[TrainingConfig] = None,
    use_lora: bool = True,
    device: str = "cuda",
) -> VSTaRVerifier:
    """
    Factory function to create a verifier

    Args:
        model_name: Model name
        ref_model: Reference model for DPO
        config: Training configuration
        use_lora: Whether to use LoRA
        device: Device

    Returns:
        VSTaRVerifier instance
    """
    return VSTaRVerifier(
        model_name=model_name,
        ref_model=ref_model,
        config=config,
        device=device,
        use_lora=use_lora,
    )
