"""
Qwen 모델 래퍼
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Dict
from config.config import STUDENT_MODEL_CONFIG
from models.base_wrapper import BaseModelWrapper


class QwenWrapper(BaseModelWrapper):
    """Qwen/Qwen2.5-3B-Instruct 모델을 위한 래퍼 클래스"""

    def __init__(self):
        self.config = STUDENT_MODEL_CONFIG
        self.model_name = self.config["model_name"]
        self.device = self.config["device"]

        # 모델과 토크나이저 로드
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        print(f"Model loaded on {self.device}")

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Qwen 모델로 텍스트 생성

        Args:
            prompt: 사용자 프롬프트
            system_message: 시스템 메시지
            chat_history: 대화 히스토리 (Ms-Mt 루프용)

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

        # 토크나이징
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config["max_new_tokens"],
                temperature=self.config["temperature"],
                do_sample=self.config["do_sample"],
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 디코딩
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()
