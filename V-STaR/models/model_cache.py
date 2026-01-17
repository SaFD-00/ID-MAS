"""Model cache for memory-efficient model loading"""

import os
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# Try to get HuggingFace token
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


class ModelCache:
    """
    Singleton model cache for memory-efficient model management

    Stores loaded models and tokenizers to avoid reloading.
    Supports sharing models between Generator and Verifier.
    """

    _instance: Optional["ModelCache"] = None
    _models: Dict[str, PreTrainedModel] = {}
    _tokenizers: Dict[str, PreTrainedTokenizer] = {}

    def __new__(cls) -> "ModelCache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
            cls._instance._tokenizers = {}
        return cls._instance

    def get_or_load(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Get model and tokenizer from cache or load them

        Args:
            model_name: HuggingFace model name
            device: Device to load model on
            dtype: Model dtype (default: auto)
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization

        Returns:
            Tuple of (model, tokenizer)
        """
        cache_key = self._get_cache_key(model_name, device, dtype, load_in_8bit, load_in_4bit)

        if cache_key not in self._models:
            print(f"Loading model: {model_name}")
            model, tokenizer = self._load_model(
                model_name, device, dtype, load_in_8bit, load_in_4bit
            )
            self._models[cache_key] = model
            self._tokenizers[cache_key] = tokenizer

        return self._models[cache_key], self._tokenizers[cache_key]

    def _get_cache_key(
        self,
        model_name: str,
        device: str,
        dtype: Optional[torch.dtype],
        load_in_8bit: bool,
        load_in_4bit: bool,
    ) -> str:
        """Generate cache key for model"""
        dtype_str = str(dtype) if dtype else "auto"
        return f"{model_name}_{device}_{dtype_str}_{load_in_8bit}_{load_in_4bit}"

    def _load_model(
        self,
        model_name: str,
        device: str,
        dtype: Optional[torch.dtype],
        load_in_8bit: bool,
        load_in_4bit: bool,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer from HuggingFace"""

        # Determine dtype
        if dtype is None:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=HF_TOKEN,
            trust_remote_code=True,
        )

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load model
        model_kwargs = {
            "token": HF_TOKEN,
            "trust_remote_code": True,
        }

        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = dtype
            if device == "cuda":
                model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Move to device if not using device_map
        if "device_map" not in model_kwargs and device != "cpu":
            model = model.to(device)

        return model, tokenizer

    def is_loaded(self, model_name: str, device: str = "cuda") -> bool:
        """Check if model is already loaded"""
        # Check all possible cache keys for this model
        for key in self._models.keys():
            if key.startswith(f"{model_name}_{device}"):
                return True
        return False

    def get_tokenizer(self, model_name: str) -> Optional[PreTrainedTokenizer]:
        """Get tokenizer for a model (if loaded)"""
        for key, tokenizer in self._tokenizers.items():
            if key.startswith(model_name):
                return tokenizer
        return None

    def clear(self, model_name: Optional[str] = None) -> None:
        """
        Clear cache

        Args:
            model_name: If provided, only clear this model. Otherwise, clear all.
        """
        if model_name:
            keys_to_remove = [k for k in self._models.keys() if model_name in k]
            for key in keys_to_remove:
                del self._models[key]
                if key in self._tokenizers:
                    del self._tokenizers[key]
        else:
            self._models.clear()
            self._tokenizers.clear()

        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        info = {
            "num_models": len(self._models),
            "model_names": list(self._models.keys()),
        }

        if torch.cuda.is_available():
            info["cuda_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            info["cuda_reserved_gb"] = torch.cuda.memory_reserved() / 1e9

        return info

    def load_from_checkpoint(
        self,
        checkpoint_path: Path,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model from local checkpoint"""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Determine dtype
        if dtype is None:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        if device == "cuda":
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)

        return model, tokenizer


# Global instance
_model_cache = ModelCache()


def get_model_cache() -> ModelCache:
    """Get global model cache instance"""
    return _model_cache
