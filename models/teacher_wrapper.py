"""Teacher 모델 래퍼 모듈.

이 모듈은 ID-MAS 시스템의 Teacher 모델을 위한 통합 래퍼를 제공합니다.
OpenAI API 모델과 로컬 HuggingFace 모델을 동일한 인터페이스로 사용할 수 있습니다.

지원 모델:
    - OpenAI API (gpt-*): OpenAI API를 통한 직접 호출
    - 로컬 모델 (Qwen 등): ModelCache를 통한 로드 및 추론

주요 클래스:
    TeacherModelWrapper: Teacher 모델 통합 래퍼

주요 함수:
    _fix_control_characters: JSON 문자열 내 제어 문자 이스케이프
    _fix_json_escapes: 유효하지 않은 JSON 이스케이프 수정
    _find_matching_brace: 중괄호 매칭 위치 탐색
    _strip_non_json_content: JSON 외부 텍스트 제거
    _is_api_model: API 모델 여부 확인

사용 예시:
    >>> from models.teacher_wrapper import TeacherModelWrapper
    >>> teacher = TeacherModelWrapper({"model": "gpt-4o"})
    >>> response = teacher.generate("Hello, world!")
"""
import re
import json
import time
import os
from openai import OpenAI
from typing import Dict, Any, Optional, List
from models.base_wrapper import BaseModelWrapper
from models.local_model_mixin import LocalModelMixin
from models.model_cache import ModelCache
from config import DESIGN_MODEL_CONFIG
from config.api import OPENAI_API_KEY


def _fix_control_characters(text: str) -> str:
    """JSON 문자열 내 제어 문자를 유니코드 이스케이프로 변환합니다.

    LLM이 JSON 응답 생성 시 문자열 내에 실제 개행/탭 등의 제어 문자(0x00-0x1F)를
    포함할 때 발생하는 'Invalid control character' 오류를 방지합니다.

    Args:
        text: 처리할 JSON 텍스트

    Returns:
        제어 문자가 이스케이프 처리된 텍스트
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
    """유효하지 않은 JSON 백슬래시 이스케이프를 수정합니다.

    LaTeX 문법(\\(, \\), \\frac 등)이 JSON 문자열에 포함될 때 발생하는
    'Invalid \\escape' 오류를 방지합니다.

    유효한 JSON 이스케이프 시퀀스:
        - \\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t
        - \\uXXXX (유니코드)

    Args:
        text: 처리할 JSON 텍스트

    Returns:
        이스케이프가 수정된 텍스트
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


def _find_matching_brace(text: str, start_pos: int) -> int:
    """여는 중괄호에 대응하는 닫는 중괄호 위치를 찾습니다.

    중첩된 중괄호와 문자열 내부의 중괄호를 올바르게 처리합니다.

    Args:
        text: 검색 대상 텍스트
        start_pos: 여는 중괄호 '{' 의 인덱스

    Returns:
        대응하는 닫는 중괄호 '}' 의 인덱스.
        찾지 못하면 -1 반환.
    """
    if start_pos >= len(text) or text[start_pos] != '{':
        return -1

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start_pos, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if not in_string:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return i

    return -1


def _strip_non_json_content(text: str) -> str:
    """JSON 외부의 텍스트를 제거합니다.

    첫 번째 '{' 이전과 대응하는 '}' 이후의 모든 텍스트를 제거하여
    순수 JSON 부분만 추출합니다.

    Args:
        text: JSON이 포함된 원본 텍스트

    Returns:
        JSON 부분만 추출된 텍스트.
        '{' 를 찾지 못하면 원본 텍스트 반환.
    """
    # 첫 번째 { 찾기
    first_brace = text.find('{')
    if first_brace == -1:
        return text

    # 마지막 } 찾기 (중첩 고려)
    last_brace = _find_matching_brace(text, first_brace)
    if last_brace == -1:
        return text

    return text[first_brace:last_brace+1]


def _is_api_model(model_name: str) -> bool:
    """OpenAI API 모델 여부를 확인합니다.

    Args:
        model_name: 확인할 모델명

    Returns:
        gpt-* 패턴이면 True, 아니면 False.
        model_name이 비어있으면 True (기본 API 모델 가정).
    """
    if not model_name:
        return True  # 기본 모델은 API 모델
    return model_name.startswith("gpt-")


class TeacherModelWrapper(BaseModelWrapper, LocalModelMixin):
    """Teacher 모델 통합 래퍼 클래스.

    API 모델(OpenAI)과 로컬 vLLM 모델을 동일한 인터페이스로 사용합니다.
    교수설계(Instructional Design) 및 스캐폴딩(Scaffolding) 생성에 사용됩니다.

    지원 모델:
        - API 모델 (gpt-*): OpenAI API 직접 호출
        - 로컬 모델 (Qwen 등): vLLM ModelCache를 통한 공유 로드

    Attributes:
        config: 모델 설정 딕셔너리
        model_name: 사용 중인 모델명
        device: 실행 디바이스 ("cuda" 또는 "cpu")
        llm: vLLM LLM 인스턴스 (API 모델이면 None)

    Example:
        >>> # OpenAI API 모델 사용
        >>> teacher = TeacherModelWrapper({"model": "gpt-5.2"})
        >>> response = teacher.generate("Solve: 2+2=?")

        >>> # 로컬 모델 사용
        >>> teacher = TeacherModelWrapper({"model": "Qwen/Qwen3-8B"})
        >>> response = teacher.generate_json("Return JSON: {answer: ...}")
    """

    def __init__(self, config: dict = None):
        """TeacherModelWrapper를 초기화합니다.

        Args:
            config: Teacher 모델 설정 딕셔너리. None이면 기본 설정 사용.
                필드:
                - model (str): 모델명 (예: "gpt-5.2", "Qwen/Qwen3-8B")
                - base_url (str): API 엔드포인트 URL (로컬 모델은 무시)
                - api_key (str): API 키 (None이면 환경변수 사용)
                - device (str): 디바이스 (기본: "cuda")
                - max_new_tokens (int): 최대 생성 토큰 (기본: 8192)
                - temperature (float): 샘플링 온도 (기본: 0.7)
                - do_sample (bool): 샘플링 사용 여부 (기본: True)
        """
        self.config = config if config is not None else DESIGN_MODEL_CONFIG
        self.model_name = self.config.get("model", "")
        self.device = self.config.get("device", "cuda")

        # API 모델인지 확인
        self._use_api = _is_api_model(self.model_name)

        if self._use_api:
            # API 모델: OpenAI 클라이언트 초기화
            self._init_api_client()
            self.llm = None
            print(f"[TeacherModelWrapper] Using API model: {self.model_name}")
        else:
            # 로컬 모델: vLLM ModelCache를 통해 로드
            self._api_client = None
            self._is_custom_endpoint = False
            self._gpu_id = self.config.get("gpu_id")
            cached = ModelCache.get_or_load(
                self.model_name,
                self.device,
                tensor_parallel_size=self.config.get("tensor_parallel_size", 1),
                gpu_memory_utilization=self.config.get("gpu_memory_utilization", 0.90),
                gpu_id=self._gpu_id,
            )
            self.llm = cached["llm"]
            print(f"[TeacherModelWrapper] Using local model (vLLM): {self.model_name} (gpu_id={self._gpu_id})")

        # 로컬 모델용 생성 설정 (LocalModelMixin에서 사용)
        # JSON 생성을 위해 기본값을 4096으로 증가 (긴 응답 지원)
        self.max_new_tokens = self.config.get("max_new_tokens", 8192)
        self.temperature = self.config.get("temperature", 0.7)
        self.do_sample = self.config.get("do_sample", True)

    def _init_api_client(self):
        """OpenAI API 클라이언트를 초기화합니다.

        config의 base_url과 api_key를 기반으로 OpenAI 클라이언트를 생성합니다.
        커스텀 엔드포인트 사용 여부는 _is_custom_endpoint 플래그로 구분합니다.
        """
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
        """텍스트를 생성합니다.

        API 모델과 로컬 모델을 자동으로 구분하여 적절한 생성 방식을 선택합니다.

        Args:
            prompt: 사용자 프롬프트 (생성 요청 내용)
            system_message: 시스템 메시지 (모델 행동 지침)
            chat_history: 대화 히스토리 (로컬 모델 멀티턴 대화용)
            response_format: 응답 형식 (API 모델 전용, 예: {"type": "json_object"})

        Returns:
            모델이 생성한 텍스트
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
        """OpenAI API를 통해 텍스트를 생성합니다.

        재시도 로직(5xx 서버 오류 시)과 OpenAI 전용 파라미터를 지원합니다.

        Args:
            prompt: 사용자 프롬프트
            system_message: 시스템 메시지
            response_format: 응답 형식 (예: {"type": "json_object"})

        Returns:
            API가 생성한 텍스트

        Raises:
            Exception: API 호출 실패 시 (재시도 횟수 초과 포함)
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
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """JSON 형식으로 응답을 생성합니다.

        API 모델은 response_format을 사용하고, 로컬 모델은 프롬프트에
        JSON 지시를 추가하여 응답을 유도합니다.

        Args:
            prompt: 사용자 프롬프트 (JSON 응답 요청 포함)
            system_message: 시스템 메시지
            max_tokens: 최대 생성 토큰. None이면 기본값(8192) 사용.

        Returns:
            파싱된 JSON 딕셔너리

        Raises:
            Exception: JSON 파싱 실패 시
        """
        # max_tokens 동적 설정
        if max_tokens is not None:
            original_max_tokens = self.max_new_tokens
            self.max_new_tokens = max_tokens

        try:
            if self._use_api:
                result = self._generate_json_api(prompt, system_message)
            else:
                result = self._generate_json_local(prompt, system_message)
            return result
        finally:
            # 원래 값으로 복원
            if max_tokens is not None:
                self.max_new_tokens = original_max_tokens

    def _generate_json_api(
        self,
        prompt: str,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """OpenAI API를 통해 JSON 응답을 생성합니다.

        response_format={"type": "json_object"}를 사용하여 JSON 응답을 보장합니다.

        Args:
            prompt: 사용자 프롬프트
            system_message: 시스템 메시지

        Returns:
            파싱된 JSON 딕셔너리

        Raises:
            Exception: JSON 파싱 실패 시
        """
        response_text = self._generate_api(
            prompt=prompt,
            system_message=system_message,
            response_format={"type": "json_object"}
        )

        try:
            result = json.loads(response_text)
            result['_raw_response'] = response_text
            return result
        except json.JSONDecodeError:
            try:
                fixed_text = _fix_json_escapes(response_text)
                result = json.loads(fixed_text)
                result['_raw_response'] = response_text  # 수정 전 원본 저장
                return result
            except json.JSONDecodeError as e:
                raise Exception(f"JSON 파싱 오류: {str(e)}\n응답: {response_text}")

    def _generate_json_local(
        self,
        prompt: str,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """로컬 vLLM 모델로 JSON 응답을 생성합니다.

        프롬프트에 JSON 응답 지시가 없으면 자동으로 추가합니다.

        Args:
            prompt: 사용자 프롬프트
            system_message: 시스템 메시지

        Returns:
            파싱된 JSON 딕셔너리

        Raises:
            Exception: JSON 추출/파싱 실패 시
        """
        json_prompt = prompt
        if "json" not in prompt.lower():
            json_prompt = prompt + "\n\nRespond in valid JSON format only."

        response_text = self._generate_with_local_model(json_prompt, system_message)

        result = self._extract_json(response_text)
        result['_raw_response'] = response_text  # 파싱 전 원본 저장
        return result

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """텍스트에서 JSON을 추출합니다.

        여러 패턴을 순차적으로 시도하여 JSON을 추출합니다:
            1. JSON 외부 텍스트 제거 (첫 '{' 이전, 마지막 '}' 이후)
            2. 전체 텍스트가 유효한 JSON인 경우
            3. ```json ... ``` 코드 블록 내 JSON
            4. { ... } 패턴 매칭
            5. LaTeX 이스케이프 수정 후 재시도

        Args:
            text: JSON이 포함된 응답 텍스트

        Returns:
            추출 및 파싱된 JSON 딕셔너리

        Raises:
            Exception: 모든 추출 시도 실패 시 (상세 오류 정보 포함)
        """
        # 0. JSON 외부 텍스트 제거
        text = _strip_non_json_content(text)

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
            # 상세 오류 정보 생성
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "error_position": f"line {e.lineno}, column {e.colno}" if hasattr(e, 'lineno') else "unknown",
                "response_length": len(text),
                "response": text,
            }

            # 상세 오류 메시지 생성
            error_msg = f"JSON Parsing Error: {error_details['error_message']}"
            if error_details['error_position'] != "unknown":
                error_msg += f" at {error_details['error_position']}"
            error_msg += f"\nResponse Length: {error_details['response_length']} characters"
            error_msg += f"\nResponse:\n{error_details['response']}"

            raise Exception(error_msg)

