"""글로벌 모델 캐시 매니저 모듈.

이 모듈은 Teacher와 Student 모델 간 HuggingFace 모델을 공유하기 위한
캐시 매니저를 제공합니다. 동일한 모델을 여러 번 로드하지 않고
메모리를 절약할 수 있습니다.

주요 클래스:
    ModelCache: 글로벌 모델 캐시 클래스 (싱글톤 패턴)

사용 예시:
    >>> from models.model_cache import ModelCache
    >>> # 모델 로드 (첫 호출 시 로드, 이후 캐시에서 반환)
    >>> cached = ModelCache.get_or_load("Qwen/Qwen2.5-3B-Instruct", "cuda")
    >>> model, tokenizer = cached["model"], cached["tokenizer"]

Note:
    캐시는 (model_name, device) 튜플을 키로 사용합니다.
    같은 모델을 다른 디바이스에서 사용하면 별도로 로드됩니다.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Tuple, Optional
from config.config import HF_TOKEN


class ModelCache:
    """글로벌 모델 캐시 클래스.

    Teacher/Student 간 동일한 HuggingFace 모델을 공유하여 메모리를 절약합니다.
    클래스 메서드만 사용하는 싱글톤 패턴으로 구현되어 있습니다.

    캐시 키: (model_name, device) 튜플
    캐시 값: {"model": model, "tokenizer": tokenizer} 딕셔너리

    Example:
        >>> # 첫 번째 로드 (실제 로드 발생)
        >>> cached1 = ModelCache.get_or_load("Qwen/Qwen2.5-3B-Instruct", "cuda")
        >>> # 두 번째 호출 (캐시에서 반환)
        >>> cached2 = ModelCache.get_or_load("Qwen/Qwen2.5-3B-Instruct", "cuda")
        >>> cached1["model"] is cached2["model"]  # True
    """
    _cache: Dict[Tuple[str, str], Dict[str, object]] = {}

    @classmethod
    def get_or_load(
        cls,
        model_name: str,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None
    ) -> Dict[str, object]:
        """캐시에서 모델을 반환하거나, 없으면 로드합니다.

        Args:
            model_name: HuggingFace 모델명 (예: "Qwen/Qwen2.5-3B-Instruct")
            device: 실행 디바이스 ("cuda" 또는 "cpu")
            dtype: 모델 데이터 타입. None이면 자동 결정
                   (cuda=float16, cpu=float32)

        Returns:
            {"model": model, "tokenizer": tokenizer} 형태의 딕셔너리
        """
        cache_key = (model_name, device)

        if cache_key in cls._cache:
            print(f"[ModelCache] Using cached model: {model_name} on {device}")
            return cls._cache[cache_key]

        print(f"[ModelCache] Loading model: {model_name}...")

        # dtype 결정
        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32

        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=HF_TOKEN
        )

        # 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            token=HF_TOKEN
        )

        # pad_token이 없는 경우 설정 (Llama 등)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # CPU인 경우 명시적으로 이동
        if device == "cpu":
            model = model.to(device)

        # 캐시에 저장
        cls._cache[cache_key] = {
            "model": model,
            "tokenizer": tokenizer
        }

        print(f"[ModelCache] Model loaded and cached: {model_name} on {device}")

        return cls._cache[cache_key]

    @classmethod
    def is_loaded(cls, model_name: str, device: str = "cuda") -> bool:
        """모델이 캐시에 로드되어 있는지 확인합니다.

        Args:
            model_name: 확인할 모델명
            device: 디바이스 ("cuda" 또는 "cpu")

        Returns:
            캐시에 로드되어 있으면 True, 아니면 False
        """
        return (model_name, device) in cls._cache

    @classmethod
    def get_loaded_models(cls) -> list:
        """현재 캐시에 로드된 모델 목록을 반환합니다.

        Returns:
            [(model_name, device), ...] 형태의 튜플 리스트
        """
        return list(cls._cache.keys())

    @classmethod
    def clear(cls, model_name: Optional[str] = None, device: Optional[str] = None):
        """캐시를 초기화합니다.

        특정 모델만 삭제하거나 전체 캐시를 삭제할 수 있습니다.

        Args:
            model_name: 삭제할 모델명. None이면 모든 모델 대상.
            device: 삭제할 디바이스. None이면 모든 디바이스 대상.

        Note:
            model_name과 device가 모두 None이면 전체 캐시를 삭제합니다.
        """
        if model_name is None and device is None:
            # 전체 캐시 삭제
            cls._cache.clear()
            print("[ModelCache] All cached models cleared")
        else:
            # 특정 모델 삭제
            keys_to_delete = []
            for key in cls._cache:
                if model_name is not None and key[0] != model_name:
                    continue
                if device is not None and key[1] != device:
                    continue
                keys_to_delete.append(key)

            for key in keys_to_delete:
                del cls._cache[key]
                print(f"[ModelCache] Cleared: {key[0]} on {key[1]}")

    @classmethod
    def memory_usage(cls) -> Dict[str, str]:
        """캐시된 모델의 메모리 사용량을 반환합니다.

        float16 기준으로 대략적인 메모리 사용량을 계산합니다.

        Returns:
            {model_name@device: memory_info} 형태의 딕셔너리
            예: {"Qwen/Qwen2.5-3B-Instruct@cuda": "3.00B params (~6000MB)"}
        """
        usage = {}
        for (model_name, device), data in cls._cache.items():
            model = data["model"]
            param_count = sum(p.numel() for p in model.parameters())
            memory_mb = param_count * 2 / (1024 * 1024)  # float16 기준
            usage[f"{model_name}@{device}"] = f"{param_count/1e9:.2f}B params (~{memory_mb:.0f}MB)"
        return usage
