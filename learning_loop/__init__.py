"""ID-MAS 학습 루프 모듈.

이 모듈은 LangGraph 기반의 Iterative Scaffolding Pipeline을 제공합니다.
학생-교사 상호작용을 통한 반복적 학습 프로세스를 구현합니다.

주요 클래스:
    StudentModel: 학생 모델 - 문제 응답 생성
    TeacherModel: 교사 모델 - 평가 및 스캐폴딩 제공

사용 예시:
    >>> from learning_loop import StudentModel, TeacherModel
    >>> student = StudentModel(model_name="Qwen/Qwen2.5-3B-Instruct")
    >>> teacher = TeacherModel(config)
"""

from learning_loop.student_model import StudentModel
from learning_loop.teacher_model import TeacherModel

__all__ = [
    "StudentModel",
    "TeacherModel",
]
