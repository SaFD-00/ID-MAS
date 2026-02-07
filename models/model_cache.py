"""글로벌 모델 캐시 매니저 모듈.

이 모듈은 Teacher와 Student 모델 간 vLLM 모델을 공유하기 위한
캐시 매니저를 제공합니다. 동일한 모델을 여러 번 로드하지 않고
메모리를 절약할 수 있습니다.

주요 클래스:
    ModelCache: 글로벌 모델 캐시 클래스 (싱글톤 패턴)

사용 예시:
    >>> from models.model_cache import ModelCache
    >>> cached = ModelCache.get_or_load("Qwen/Qwen3-1.7B", "cuda")
    >>> llm = cached["llm"]

Note:
    캐시는 (model_name, device) 튜플을 키로 사용합니다.
"""
from vllm import LLM
from typing import Dict, Tuple, Optional


class ModelCache:
    """글로벌 모델 캐시 클래스.

    Teacher/Student 간 동일한 vLLM 모델을 공유하여 메모리를 절약합니다.
    클래스 메서드만 사용하는 싱글톤 패턴으로 구현되어 있습니다.

    캐시 키: (model_name, device) 튜플
    캐시 값: {"llm": llm} 딕셔너리

    Example:
        >>> cached1 = ModelCache.get_or_load("Qwen/Qwen3-1.7B", "cuda")
        >>> cached2 = ModelCache.get_or_load("Qwen/Qwen3-1.7B", "cuda")
        >>> cached1["llm"] is cached2["llm"]  # True
    """
    _cache: Dict[Tuple[str, str], Dict[str, object]] = {}

    @classmethod
    def get_or_load(
        cls,
        model_name: str,
        device: str = "cuda",
        dtype: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
    ) -> Dict[str, object]:
        """캐시에서 모델을 반환하거나, 없으면 로드합니다.

        Args:
            model_name: 모델명 (예: "Qwen/Qwen3-1.7B")
            device: 실행 디바이스 ("cuda" 또는 "cpu")
            dtype: 모델 데이터 타입. None이면 "auto" (vLLM 자동 결정)
            tensor_parallel_size: 텐서 병렬 처리 GPU 수
            gpu_memory_utilization: GPU 메모리 활용률 (0.0~1.0)
            max_model_len: 최대 시퀀스 길이. None이면 모델 기본값 사용.

        Returns:
            {"llm": llm} 형태의 딕셔너리
        """
        cache_key = (model_name, device)

        if cache_key in cls._cache:
            print(f"[ModelCache] Using cached model: {model_name} on {device}")
            return cls._cache[cache_key]

        print(f"[ModelCache] Loading model with vLLM: {model_name}...")

        llm_kwargs = {
            "model": model_name,
            "dtype": dtype or "auto",
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
        }

        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        llm = LLM(**llm_kwargs)

        cls._cache[cache_key] = {"llm": llm}

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

        Args:
            model_name: 삭제할 모델명. None이면 모든 모델 대상.
            device: 삭제할 디바이스. None이면 모든 디바이스 대상.
        """
        if model_name is None and device is None:
            cls._cache.clear()
            print("[ModelCache] All cached models cleared")
        else:
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

        Returns:
            {model_name@device: "loaded"} 형태의 딕셔너리
        """
        usage = {}
        for (model_name, device), data in cls._cache.items():
            usage[f"{model_name}@{device}"] = "loaded"
        return usage
