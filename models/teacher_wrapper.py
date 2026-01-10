"""
Teacher 모델 래퍼 - API와 로컬 모델 모두 지원

- OpenAI API 모델 (gpt-*, o1-*, o3-*): 직접 API 호출
- 로컬 모델 (Qwen, Llama 등): ModelCache를 통해 로드 및 추론
"""
import re
import json
import time
import os
import torch
from openai import OpenAI
from typing import Dict, Any, Optional, List
from models.base_wrapper import BaseModelWrapper
from models.local_model_mixin import LocalModelMixin
from models.model_cache import ModelCache
from config.config import DESIGN_MODEL_CONFIG, OPENAI_API_KEY


def _fix_control_characters(text: str) -> str:
    """
    JSON 문자열 내의 제어 문자(0x00-0x1F)를 유니코드 이스케이프로 변환

    LLM이 JSON 응답 생성 시 문자열 내에 실제 개행/탭 등을 포함할 때
    'Invalid control character' 오류를 방지
    """
    result = []
    in_string = False
    escape_next = False

    for char in text:
        if escape_next:
            result.append(char)
            escape_next = False
            continue

        if char == '\\' and in_string:
            result.append(char)
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            result.append(char)
            continue

        # 문자열 내부의 제어 문자만 이스케이프
        if in_string and ord(char) < 32:
            # 일반적인 제어 문자는 유니코드 이스케이프로 변환
            if char == '\n':
                result.append('\\n')
            elif char == '\r':
                result.append('\\r')
            elif char == '\t':
                result.append('\\t')
            else:
                result.append(f'\\u{ord(char):04x}')
        else:
            result.append(char)

    return ''.join(result)


def _fix_json_escapes(text: str) -> str:
    """
    JSON에서 유효하지 않은 백슬래시 이스케이프를 수정

    LaTeX 문법(\\(, \\), \\frac 등)이 JSON 문자열에 포함될 때 발생하는
    'Invalid \\escape' 오류를 방지

    유효한 JSON 이스케이프: \\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX
    """
    # 먼저 제어 문자를 이스케이프
    text = _fix_control_characters(text)

    result = []
    i = 0
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text):
            next_char = text[i + 1]
            # 유효한 JSON 이스케이프 시퀀스
            if next_char in '"\\/bfnrt':
                result.append(text[i:i+2])
                i += 2
            # 이미 이스케이프된 백슬래시 (\\)
            elif next_char == '\\':
                result.append('\\\\')
                i += 2
            # 유니코드 이스케이프 \uXXXX
            elif next_char == 'u' and i + 5 < len(text) and all(c in '0123456789abcdefABCDEF' for c in text[i+2:i+6]):
                result.append(text[i:i+6])
                i += 6
            # 유효하지 않은 이스케이프 - 백슬래시 추가
            else:
                result.append('\\\\' + next_char)
                i += 2
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


def _is_api_model(model_name: str) -> bool:
    """API 모델인지 확인 (OpenAI 모델)"""
    if not model_name:
        return True  # 기본 모델은 API 모델
    return (
        model_name.startswith("gpt-") or
        model_name.startswith("o1") or
        model_name.startswith("o3")
    )


class TeacherModelWrapper(BaseModelWrapper, LocalModelMixin):
    """
    Teacher 모델 래퍼 클래스

    API 모델과 로컬 HuggingFace 모델 모두 지원:
    - API 모델 (gpt-*, o1-*, o3-*): OpenAI API 직접 호출
    - 로컬 모델: ModelCache를 통해 로드 및 직접 추론
    """

    def __init__(self, config: dict = None):
        """
        TeacherModelWrapper 초기화

        Args:
            config: Teacher model 설정 딕셔너리 (None이면 기본 설정 사용)
                - model: 모델 이름
                - base_url: API endpoint (로컬 모델은 무시)
                - device: 디바이스 (기본: "cuda")
                - max_new_tokens: 최대 생성 토큰 (기본: 2048)
                - temperature: 온도 (기본: 0.7)
        """
        self.config = config if config is not None else DESIGN_MODEL_CONFIG
        self.model_name = self.config.get("model", "")
        self.device = self.config.get("device", "cuda")

        # API 모델인지 확인
        self._use_api = _is_api_model(self.model_name)

        if self._use_api:
            # API 모델: OpenAI 클라이언트 초기화
            self._init_api_client()
            self.model = None
            self.tokenizer = None
            print(f"[TeacherModelWrapper] Using API model: {self.model_name}")
        else:
            # 로컬 모델: ModelCache를 통해 로드
            self._api_client = None
            self._is_custom_endpoint = False
            cached = ModelCache.get_or_load(self.model_name, self.device)
            self.model = cached["model"]
            self.tokenizer = cached["tokenizer"]
            print(f"[TeacherModelWrapper] Using local model: {self.model_name}")

        # 로컬 모델용 생성 설정 (LocalModelMixin에서 사용)
        self.max_new_tokens = self.config.get("max_new_tokens", 2048)
        self.temperature = self.config.get("temperature", 0.7)
        self.do_sample = self.config.get("do_sample", True)

    def _init_api_client(self):
        """OpenAI API 클라이언트 초기화"""
        base_url = self.config.get("base_url")
        api_key = self.config.get("api_key") or OPENAI_API_KEY
        timeout = self.config.get("timeout", 300.0)

        if base_url:
            self._api_client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
            self._is_custom_endpoint = True
        else:
            self._api_client = OpenAI(api_key=api_key, timeout=timeout)
            self._is_custom_endpoint = False

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        텍스트 생성

        Args:
            prompt: 사용자 프롬프트
            system_message: 시스템 메시지
            chat_history: 대화 히스토리 (로컬 모델만 지원)
            response_format: 응답 형식 (API 모델만 지원)

        Returns:
            생성된 텍스트
        """
        if self._use_api:
            return self._generate_api(prompt, system_message, response_format)
        else:
            return self._generate_with_local_model(prompt, system_message, chat_history)

    def _generate_api(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        API를 통한 텍스트 생성 (LLMWrapper 로직 통합)

        Args:
            prompt: 사용자 프롬프트
            system_message: 시스템 메시지
            response_format: 응답 형식

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
        if self._is_custom_endpoint:
            model_name = "model"
        else:
            model_name = self.config["model"]

        params = {
            "model": model_name,
            "messages": messages,
        }

        # max_tokens 파라미터
        max_tokens = self.config.get("max_tokens", 6000)
        if self._is_custom_endpoint:
            params["max_tokens"] = max_tokens
        else:
            params["max_completion_tokens"] = max_tokens

        # OpenAI 전용 파라미터
        if not self._is_custom_endpoint:
            if "reasoning" in self.config and "effort" in self.config["reasoning"]:
                params["reasoning_effort"] = self.config["reasoning"]["effort"]

            if "text" in self.config and "verbosity" in self.config["text"]:
                params["verbosity"] = self.config["text"]["verbosity"]

        # JSON 형식 응답 요청 시
        if response_format:
            params["response_format"] = response_format

        # 재시도 로직 (5xx 서버 오류 시)
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self._api_client.chat.completions.create(**params)
                if os.getenv("IDMAS_DEBUG_API") == "1":
                    print("=== LLM API raw response ===")
                    print(json.dumps(response.to_dict(), indent=2, ensure_ascii=True))
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                error_str = str(e)
                if any(code in error_str for code in ["500", "502", "503", "Internal Server Error"]):
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"  서버 오류 발생, {wait_time}초 후 재시도 ({attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                break

        endpoint_info = self.config.get("base_url", "OpenAI API")
        raise Exception(f"LLM API 호출 오류 ({endpoint_info}): {str(last_error)}")

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
        if self._use_api:
            return self._generate_json_api(prompt, system_message)
        else:
            return self._generate_json_local(prompt, system_message)

    def _generate_json_api(
        self,
        prompt: str,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """API를 통한 JSON 응답 생성"""
        response_text = self._generate_api(
            prompt=prompt,
            system_message=system_message,
            response_format={"type": "json_object"}
        )

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            try:
                fixed_text = _fix_json_escapes(response_text)
                return json.loads(fixed_text)
            except json.JSONDecodeError as e:
                raise Exception(f"JSON 파싱 오류: {str(e)}\n응답: {response_text}")

    def _generate_json_local(
        self,
        prompt: str,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        로컬 모델로 JSON 응답 생성

        Args:
            prompt: 사용자 프롬프트
            system_message: 시스템 메시지

        Returns:
            JSON 파싱된 딕셔너리
        """
        json_prompt = prompt
        if "json" not in prompt.lower():
            json_prompt = prompt + "\n\nRespond in valid JSON format only."

        response_text = self._generate_with_local_model(json_prompt, system_message)

        return self._extract_json(response_text)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        텍스트에서 JSON 추출

        여러 패턴을 시도하여 JSON 추출:
        1. 전체 텍스트가 JSON인 경우
        2. ```json ... ``` 블록
        3. { ... } 패턴
        4. LaTeX 이스케이프 수정 후 재시도

        Args:
            text: 응답 텍스트

        Returns:
            JSON 딕셔너리
        """
        # 1. 전체 텍스트가 JSON인 경우
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # 2. ```json ... ``` 블록 추출
        json_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(json_block_pattern, text)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        # 3. { ... } 패턴 추출 (가장 바깥쪽 중괄호)
        brace_pattern = r'\{[\s\S]*\}'
        matches = re.findall(brace_pattern, text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                try:
                    fixed_text = _fix_json_escapes(match)
                    return json.loads(fixed_text)
                except json.JSONDecodeError:
                    continue

        # 4. 마지막 시도: 전체 텍스트에서 이스케이프 수정
        try:
            fixed_text = _fix_json_escapes(text)
            return json.loads(fixed_text)
        except json.JSONDecodeError as e:
            raise Exception(f"JSON 파싱 오류: {str(e)}\n응답: {text[:500]}...")

    @property
    def is_api_model(self) -> bool:
        """API 모델 사용 여부"""
        return self._use_api

    @property
    def is_local_model(self) -> bool:
        """로컬 모델 사용 여부"""
        return not self._use_api
