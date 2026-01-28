"""모델 래퍼 추상 기본 클래스 모듈.

이 모듈은 모든 모델 래퍼(Teacher, Student)의 공통 인터페이스를 정의합니다.
텍스트 생성 및 JSON 응답 생성을 위한 표준 메서드를 제공합니다.

주요 클래스:
    BaseModelWrapper: 추상 기본 클래스 (ABC)

Note:
    이 클래스를 직접 인스턴스화할 수 없습니다.
    TeacherModelWrapper 또는 StudentModelWrapper를 사용하세요.
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class BaseModelWrapper(ABC):
    """모델 래퍼의 추상 기본 클래스.

    Teacher/Student 모델 래퍼가 구현해야 하는 공통 인터페이스를 정의합니다.
    서브클래스는 반드시 generate() 메서드를 구현해야 합니다.

    인터페이스:
        generate: 텍스트 생성 (추상 메서드)
        generate_json: JSON 형식 응답 생성 (선택적 구현)
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """텍스트를 생성합니다.

        서브클래스에서 반드시 구현해야 하는 추상 메서드입니다.

        Args:
            prompt: 사용자 프롬프트 (생성 요청 내용)
            system_message: 시스템 메시지 (모델 행동 지침)
            chat_history: 대화 히스토리 (멀티턴 대화용)
            response_format: 응답 형식 지정 (예: {"type": "json_object"}, API 모델 전용)

        Returns:
            모델이 생성한 텍스트
        """
        pass

    def generate_json(
        self,
        prompt: str,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """JSON 형식으로 응답을 생성합니다.

        기본 구현은 NotImplementedError를 발생시킵니다.
        서브클래스에서 필요 시 오버라이드하세요.

        Args:
            prompt: 사용자 프롬프트
            system_message: 시스템 메시지

        Returns:
            파싱된 JSON 딕셔너리

        Raises:
            NotImplementedError: 서브클래스에서 구현하지 않은 경우
        """
        raise NotImplementedError("서브클래스에서 generate_json을 구현해야 합니다")
