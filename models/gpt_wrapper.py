"""
GPT/LLM 모델 래퍼 (OpenAI API 및 vLLM 호환)
"""
from openai import OpenAI
from typing import Dict, Any, Optional
import json
import os
from config.config import OPENAI_API_KEY, DESIGN_MODEL_CONFIG


class GPTWrapper:
    """GPT-5 또는 vLLM 호환 모델을 위한 래퍼 클래스"""

    def __init__(self, config: dict = None):
        """
        GPTWrapper 초기화

        Args:
            config: Teacher model 설정 딕셔너리 (None이면 기본 DESIGN_MODEL_CONFIG 사용)
        """
        self.model_config = config if config is not None else DESIGN_MODEL_CONFIG

        # Custom API endpoint 지원 (LLaMA-Factory vLLM 등)
        base_url = self.model_config.get("base_url")
        api_key = self.model_config.get("api_key") or OPENAI_API_KEY

        # Large models (30B+) require longer timeout (5 minutes)
        timeout = self.model_config.get("timeout", 300.0)

        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
            self.is_custom_endpoint = True
        else:
            self.client = OpenAI(api_key=api_key, timeout=timeout)
            self.is_custom_endpoint = False

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        GPT-5 로 텍스트 생성

        Args:
            prompt: 사용자 프롬프트
            system_message: 시스템 메시지
            response_format: 응답 형식 (예: {"type": "json_object"})

        Returns:
            생성된 텍스트
        """
        messages = []

        if system_message:
            messages.append({
                "role": "system",
                "content": system_message
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        # API 호출 파라미터 구성
        # LLaMA-Factory API 서버는 model="model"을 사용 (실제 모델명이 아님)
        if self.is_custom_endpoint:
            model_name = "model"
        else:
            model_name = self.model_config["model"]

        params = {
            "model": model_name,
            "messages": messages,
        }

        # max_tokens 파라미터 (vLLM은 max_tokens, OpenAI GPT-5는 max_completion_tokens)
        max_tokens = self.model_config.get("max_tokens", 6000)
        if self.is_custom_endpoint:
            params["max_tokens"] = max_tokens
        else:
            params["max_completion_tokens"] = max_tokens

        # GPT-5 전용 파라미터 (custom endpoint에서는 스킵)
        if not self.is_custom_endpoint:
            # reasoning.effort -> reasoning_effort
            if "reasoning" in self.model_config and "effort" in self.model_config["reasoning"]:
                params["reasoning_effort"] = self.model_config["reasoning"]["effort"]

            # text.verbosity -> verbosity
            if "text" in self.model_config and "verbosity" in self.model_config["text"]:
                params["verbosity"] = self.model_config["text"]["verbosity"]

        # JSON 형식 응답 요청 시
        if response_format:
            params["response_format"] = response_format

        try:
            response = self.client.chat.completions.create(**params)
            if os.getenv("IDMAS_DEBUG_OPENAI") == "1":
                print("=== OpenAI raw response ===")
                print(json.dumps(response.to_dict(), indent=2, ensure_ascii=True))
            # print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"GPT API 호출 오류: {str(e)}")

    def generate_json(
        self,
        prompt: str,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        JSON 형식으로 응답 생성

        Args:
            prompt: 사용자 프롬프트
            system_message: 시스템 메시지

        Returns:
            JSON 파싱된 딕셔너리
        """
        response_text = self.generate(
            prompt=prompt,
            system_message=system_message,
            response_format={"type": "json_object"}
        )

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            raise Exception(f"JSON 파싱 오류: {str(e)}\n응답: {response_text}")
