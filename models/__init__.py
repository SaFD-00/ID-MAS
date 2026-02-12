"""LLM 모델 래퍼 패키지.

이 패키지는 ID-MAS 시스템에서 사용하는 LLM 모델 래퍼들을 제공합니다.
Teacher/Student 모델 아키텍처를 지원하며, API 및 로컬 HuggingFace 모델을
통합된 인터페이스로 사용할 수 있습니다.

주요 클래스:
    BaseModelWrapper: 모든 래퍼의 추상 기본 클래스
    TeacherModelWrapper: Teacher 모델 래퍼 (OpenAI API + 로컬 모델 지원)
    StudentModelWrapper: Student 모델 래퍼 (로컬 HuggingFace 모델 전용)
    ModelCache: 글로벌 모델 캐시 (Teacher/Student 간 메모리 공유)

사용 예시:
    >>> from models import TeacherModelWrapper, StudentModelWrapper
    >>> teacher = TeacherModelWrapper(config)
    >>> student = StudentModelWrapper("Qwen/Qwen3-1.7B")
"""

from models.base_wrapper import BaseModelWrapper
from models.teacher_wrapper import TeacherModelWrapper
from models.student_wrapper import StudentModelWrapper
from models.model_cache import ModelCache
from models.remote_model import RemoteLLMProxy

__all__ = [
    "BaseModelWrapper",
    "TeacherModelWrapper",
    "StudentModelWrapper",
    "ModelCache",
    "RemoteLLMProxy",
]
