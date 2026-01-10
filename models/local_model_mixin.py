"""
로컬 HuggingFace 모델 생성 로직 공유 믹스인

StudentModelWrapper와 TeacherModelWrapper에서 공통으로 사용하는
로컬 모델 텍스트 생성 로직을 추출.
"""
import torch
from typing import Optional, List, Dict


class LocalModelMixin:
    """
    로컬 HuggingFace 모델 생성 로직 믹스인

    사용하는 클래스는 다음 속성을 가져야 함:
    - self.model: HuggingFace 모델
    - self.tokenizer: HuggingFace 토크나이저
    - self.device: 디바이스 ("cuda" 또는 "cpu")
    - self.max_new_tokens: 최대 생성 토큰 수
    - self.temperature: 온도
    - self.do_sample: 샘플링 여부
    """

    # 서브클래스에서 설정해야 하는 속성들
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
        """
        로컬 모델로 텍스트 생성 (공통 로직)

        Args:
            prompt: 사용자 프롬프트
            system_message: 시스템 메시지
            chat_history: 대화 히스토리 (multi-turn용)
            max_length: 최대 입력 길이

        Returns:
            생성된 텍스트
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
