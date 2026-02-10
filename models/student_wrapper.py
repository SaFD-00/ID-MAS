"""Student 모델 래퍼 모듈.

이 모듈은 ID-MAS 시스템의 Student 모델을 위한 래퍼를 제공합니다.
Qwen 등 다양한 HuggingFace 로컬 모델을 지원합니다.

SFT(Supervised Fine-Tuning) 모델과 SFT_ID-MAS 모델도 지원하여
파인튜닝된 모델의 평가가 가능합니다.

주요 클래스:
    StudentModelWrapper: Student 모델 래퍼

사용 예시:
    >>> from models.student_wrapper import StudentModelWrapper
    >>> # 기본 모델
    >>> student = StudentModelWrapper("Qwen/Qwen3-1.7B")
    >>> # SFT 파인튜닝 모델
    >>> student = StudentModelWrapper("Qwen/Qwen3-1.7B", use_sft_model=True, sft_domain="math")
"""
from typing import Optional, List, Dict
from config.config import get_student_model_config, DEFAULT_STUDENT_MODEL
from models.base_wrapper import BaseModelWrapper
from models.local_model_mixin import LocalModelMixin
from models.model_cache import ModelCache


class StudentModelWrapper(BaseModelWrapper, LocalModelMixin):
    """Student 모델 래퍼 클래스.

    vLLM을 사용하여 응답을 생성합니다.
    Teacher 모델의 스캐폴딩을 받아 학습하는 학생 역할을 합니다.

    지원 기능:
        - 기본 모델 (Qwen 등)
        - SFT 파인튜닝 모델 (HuggingFace Hub)
        - SFT_ID-MAS 파인튜닝 모델 (HuggingFace Hub)
        - ModelCache를 통한 Teacher와 모델 공유

    Attributes:
        model_name: 실제 사용 중인 모델명 (SFT 모델 포함)
        base_model_name: 기본 모델명 (SFT 모델의 경우 원본 모델명)
        config: 모델 설정 딕셔너리
        llm: vLLM LLM 인스턴스
    """

    def __init__(
        self,
        model_name: str = None,
        use_sft_model: bool = False,
        use_sft_idmas_model: bool = False,
        sft_domain: str = None
    ):
        """StudentModelWrapper를 초기화합니다.

        Args:
            model_name: 사용할 모델명. None이면 기본 모델(Qwen3-1.7B) 사용.
            use_sft_model: SFT 파인튜닝 모델 사용 여부
            use_sft_idmas_model: SFT_ID-MAS 파인튜닝 모델 사용 여부
            sft_domain: SFT/SFT_ID-MAS 모델의 도메인 (예: "math", "logical")

        Raises:
            ValueError: SFT 모델 사용 시 sft_domain이 지정되지 않은 경우
        """
        if model_name is None:
            model_name = DEFAULT_STUDENT_MODEL

        # SFT 모델 이름 resolution
        from config.config import get_sft_model_name, get_sft_idmas_model_name

        actual_model_name = model_name
        self.base_model_name = model_name
        if use_sft_idmas_model:
            if sft_domain is None:
                raise ValueError("sft_domain is required when use_sft_idmas_model=True")
            actual_model_name = get_sft_idmas_model_name(model_name, sft_domain)
            print(f"Loading SFT_ID-MAS fine-tuned model: {actual_model_name} (base: {model_name})")
        elif use_sft_model:
            if sft_domain is None:
                raise ValueError("sft_domain is required when use_sft_model=True")
            actual_model_name = get_sft_model_name(model_name, sft_domain)
            print(f"Loading SFT fine-tuned model: {actual_model_name} (base: {model_name})")

        # Get config using base model name (SFT models use same config as base)
        self.config = get_student_model_config(self.base_model_name)
        self.model_name = actual_model_name  # Use actual model name for loading
        self.device = self.config["device"]

        # LocalModelMixin에서 사용하는 속성들
        self.max_new_tokens = self.config["max_new_tokens"]
        self.temperature = self.config["temperature"]
        self.do_sample = self.config["do_sample"]

        # 공유 ModelCache를 사용하여 vLLM 모델 로드 (Teacher와 동일 모델일 경우 공유됨)
        cached = ModelCache.get_or_load(
            actual_model_name,
            self.device,
            tensor_parallel_size=self.config.get("tensor_parallel_size", 1),
            gpu_memory_utilization=self.config.get("gpu_memory_utilization", 0.90),
        )
        self.llm = cached["llm"]

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """텍스트를 생성합니다.

        vLLM을 사용하여 응답을 생성합니다.

        Args:
            prompt: 사용자 프롬프트 (문제 또는 질문)
            system_message: 시스템 메시지 (모델 행동 지침)
            chat_history: 대화 히스토리 (Ms-Mt 스캐폴딩 루프용)
            response_format: 응답 형식 (로컬 모델에서는 무시됨, 인터페이스 통일용)

        Returns:
            모델이 생성한 텍스트
        """
        return self._generate_with_local_model(prompt, system_message, chat_history)

