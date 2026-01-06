"""
범용 학생 모델 래퍼
Qwen, Llama 등 다양한 HuggingFace 모델 지원
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Dict, Tuple
from config.config import get_student_model_config, DEFAULT_STUDENT_MODEL, HF_TOKEN
from models.base_wrapper import BaseModelWrapper


class StudentModelWrapper(BaseModelWrapper):
    """범용 학생 모델 래퍼 클래스 (Qwen, Llama 등 지원)"""
    _model_cache: Dict[Tuple[str, str], Dict[str, object]] = {}

    def __init__(
        self,
        model_name: str = None,
        use_sft_model: bool = False,
        use_sft_idmas_model: bool = False,
        sft_domain: str = None
    ):
        """
        Args:
            model_name: 사용할 모델 이름 (None이면 기본 모델)
            use_sft_model: True면 SFT fine-tuned 모델 사용
            use_sft_idmas_model: True면 SFT_ID-MAS fine-tuned 모델 사용
            sft_domain: SFT/SFT_ID-MAS 모델의 도메인 (예: "math")
        """
        if model_name is None:
            model_name = DEFAULT_STUDENT_MODEL

        # SFT 모델 이름 resolution
        from config.config import get_sft_model_name, get_sft_idmas_model_name

        actual_model_name = model_name
        self.base_model_name = model_name
        self.is_sft = use_sft_model
        self.is_sft_idmas = use_sft_idmas_model

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

        # Cache key uses actual model name (SFT or base)
        cache_key = (actual_model_name, self.device)
        cached = self._model_cache.get(cache_key)
        if cached:
            self.tokenizer = cached["tokenizer"]
            self.model = cached["model"]
            print(f"Using cached {actual_model_name} on {self.device}")
            return

        # 모델과 토크나이저 로드
        print(f"Loading {actual_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            actual_model_name,
            token=HF_TOKEN
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            actual_model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            token=HF_TOKEN
        )

        # pad_token이 없는 경우 설정 (Llama 모델 등)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self._model_cache[cache_key] = {
            "tokenizer": self.tokenizer,
            "model": self.model
        }

        print(f"Model loaded on {self.device}")

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        텍스트 생성

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
