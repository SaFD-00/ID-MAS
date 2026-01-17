"""로컬 HuggingFace 모델 생성 로직 믹스인 모듈.

이 모듈은 StudentModelWrapper와 TeacherModelWrapper에서 공통으로 사용하는
로컬 모델 텍스트 생성 로직을 믹스인 클래스로 제공합니다.

주요 클래스:
    LocalModelMixin: 로컬 모델 생성 로직 믹스인

Note:
    이 믹스인을 사용하는 클래스는 다음 속성을 가져야 합니다:
    - model: HuggingFace 모델 객체
    - tokenizer: HuggingFace 토크나이저
    - device: 실행 디바이스 ("cuda" 또는 "cpu")
    - max_new_tokens: 최대 생성 토큰 수
    - temperature: 샘플링 온도
    - do_sample: 샘플링 사용 여부
"""
import torch
from typing import Optional, List, Dict


class LocalModelMixin:
    """로컬 HuggingFace 모델 텍스트 생성 믹스인 클래스.

    Teacher/Student 모델 래퍼에서 공통으로 사용하는 로컬 모델 생성 로직을
    제공합니다. 다중 상속을 통해 사용합니다.

    필수 속성 (서브클래스에서 설정):
        model: HuggingFace AutoModelForCausalLM 객체
        tokenizer: HuggingFace AutoTokenizer 객체
        device: 실행 디바이스 ("cuda" 또는 "cpu")
        max_new_tokens: 최대 생성 토큰 수
        temperature: 샘플링 온도 (0.0~2.0)
        do_sample: 샘플링 사용 여부 (False면 greedy decoding)

    Example:
        >>> class MyWrapper(BaseModelWrapper, LocalModelMixin):
        ...     def __init__(self):
        ...         self.model = ...
        ...         self.tokenizer = ...
        ...         self.device = "cuda"
        ...         self.max_new_tokens = 2048
        ...         self.temperature = 0.7
        ...         self.do_sample = True
    """

    # 서브클래스에서 설정해야 하는 속성 타입 힌트
    model: object
    tokenizer: object
    device: str
    max_new_tokens: int
    temperature: float
    do_sample: bool

    def _generate_with_local_model(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_length: int = 4096
    ) -> str:
        """로컬 HuggingFace 모델로 텍스트를 생성합니다.

        chat template을 적용하여 메시지를 포맷팅하고,
        모델을 통해 텍스트를 생성합니다.

        Args:
            prompt: 사용자 프롬프트 (현재 턴의 입력)
            system_message: 시스템 메시지 (모델 행동 지침)
            chat_history: 대화 히스토리 (멀티턴 대화용, role/content 딕셔너리 리스트)
            max_length: 입력 토큰 최대 길이 (truncation 기준)

        Returns:
            모델이 생성한 텍스트 (strip 처리됨)
        """
        # 메시지 구성
        messages = []

        if system_message:
            messages.append({
                "role": "system",
                "content": system_message
            })

        # 대화 히스토리 추가
        if chat_history:
            messages.extend(chat_history)

        # 현재 프롬프트 추가
        messages.append({
            "role": "user",
            "content": prompt
        })

        # 토크나이징 (chat template 적용)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 디코딩
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()
