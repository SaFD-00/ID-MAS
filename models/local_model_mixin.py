"""로컬 vLLM 모델 생성 로직 믹스인 모듈.

이 모듈은 StudentModelWrapper와 TeacherModelWrapper에서 공통으로 사용하는
로컬 모델 텍스트 생성 로직을 믹스인 클래스로 제공합니다.

주요 클래스:
    LocalModelMixin: 로컬 모델 생성 로직 믹스인

Note:
    이 믹스인을 사용하는 클래스는 다음 속성을 가져야 합니다:
    - llm: vLLM LLM 인스턴스
    - max_new_tokens: 최대 생성 토큰 수
    - temperature: 샘플링 온도
    - do_sample: 샘플링 사용 여부
"""
from vllm import SamplingParams
from typing import Optional, List, Dict


class LocalModelMixin:
    """로컬 vLLM 모델 텍스트 생성 믹스인 클래스.

    Teacher/Student 모델 래퍼에서 공통으로 사용하는 로컬 모델 생성 로직을
    제공합니다. 다중 상속을 통해 사용합니다.

    필수 속성 (서브클래스에서 설정):
        llm: vLLM LLM 인스턴스
        max_new_tokens: 최대 생성 토큰 수
        temperature: 샘플링 온도 (0.0~2.0)
        do_sample: 샘플링 사용 여부 (False면 temperature=0.0)

    Example:
        >>> class MyWrapper(BaseModelWrapper, LocalModelMixin):
        ...     def __init__(self):
        ...         self.llm = ...  # vLLM LLM 인스턴스
        ...         self.max_new_tokens = 2048
        ...         self.temperature = 0.7
        ...         self.do_sample = True
    """

    llm: object
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
        """vLLM으로 텍스트를 생성합니다.

        chat API를 사용하여 메시지를 포맷팅하고,
        모델을 통해 텍스트를 생성합니다.

        Args:
            prompt: 사용자 프롬프트 (현재 턴의 입력)
            system_message: 시스템 메시지 (모델 행동 지침)
            chat_history: 대화 히스토리 (멀티턴 대화용, role/content 딕셔너리 리스트)
            max_length: 하위 호환성을 위해 유지 (vLLM에서는 미사용)

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

        if chat_history:
            messages.extend(chat_history)

        messages.append({
            "role": "user",
            "content": prompt
        })

        # SamplingParams 구성
        temperature = self.temperature if self.do_sample else 0.0
        sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=temperature,
        )

        # vLLM chat API (chat template 자동 적용)
        outputs = self.llm.chat(
            messages=[messages],
            sampling_params=sampling_params,
        )

        return outputs[0].outputs[0].text.strip()
