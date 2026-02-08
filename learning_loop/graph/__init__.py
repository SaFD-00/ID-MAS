"""LangGraph 기반 ID-MAS Iterative Scaffolding Pipeline 패키지.

이 패키지는 LangGraph를 활용한 반복적 스캐폴딩 파이프라인을 제공합니다.
Dick & Carey 교수설계 모델을 기반으로 구현되었습니다.

파이프라인 구성:
    - Task Analysis + 초기 응답 생성
    - Performance Objectives 기반 평가 (피드백 질문 포함)
    - Case A/B/C SFT 데이터 생성

주요 클래스:
    IDMASState: 파이프라인 상태 스키마
    QuestionResult: 개별 문제 처리 결과
    IDMASGraphRunner: 파이프라인 실행기

주요 함수:
    create_idmas_graph: StateGraph 생성
"""
from learning_loop.graph.state import IDMASState, QuestionResult
from learning_loop.graph.graph import create_idmas_graph, IDMASGraphRunner

__all__ = [
    "IDMASState",
    "QuestionResult",
    "create_idmas_graph",
    "IDMASGraphRunner",
]
