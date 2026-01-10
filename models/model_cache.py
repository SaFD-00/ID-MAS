"""
공유 모델 캐시 매니저
Teacher와 Student 모델 간 HuggingFace 모델 공유
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Tuple, Optional
from config.config import HF_TOKEN


class ModelCache:
    """
    글로벌 모델 캐시 (Teacher/Student 공유)

    동일한 모델명과 디바이스를 사용하는 경우 한 번만 로드하여 메모리 절약
    """
    _cache: Dict[Tuple[str, str], Dict[str, object]] = {}

    @classmethod
    def get_or_load(
        cls,
        model_name: str,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None
    ) -> Dict[str, object]:
        """
        캐시에서 모델 반환, 없으면 로드

        Args:
            model_name: HuggingFace 모델 이름 (예: "Qwen/Qwen2.5-3B-Instruct")
            device: 디바이스 ("cuda" 또는 "cpu")
            dtype: 모델 데이터 타입 (None이면 자동 결정)

        Returns:
            {"model": model, "tokenizer": tokenizer}
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
        """
        모델이 로드되어 있는지 확인

        Args:
            model_name: 모델 이름
            device: 디바이스

        Returns:
            로드 여부
        """
        return (model_name, device) in cls._cache

    @classmethod
    def get_loaded_models(cls) -> list:
        """
        현재 로드된 모델 목록 반환

        Returns:
            [(model_name, device), ...] 형태의 리스트
        """
        return list(cls._cache.keys())

    @classmethod
    def clear(cls, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        캐시 초기화

        Args:
            model_name: 특정 모델만 삭제 (None이면 전체)
            device: 특정 디바이스만 삭제 (None이면 전체)
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
        """
        캐시된 모델의 메모리 사용량 반환 (참고용)

        Returns:
            {model_name: memory_info} 형태의 딕셔너리
        """
        usage = {}
        for (model_name, device), data in cls._cache.items():
            model = data["model"]
            param_count = sum(p.numel() for p in model.parameters())
            memory_mb = param_count * 2 / (1024 * 1024)  # float16 기준
            usage[f"{model_name}@{device}"] = f"{param_count/1e9:.2f}B params (~{memory_mb:.0f}MB)"
        return usage
