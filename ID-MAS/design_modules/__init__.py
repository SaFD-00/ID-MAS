"""ID-MAS 교수설계(Instructional Design) 모듈 패키지.

이 패키지는 Dick & Carey 모델 기반의 교수설계 에이전트들을 제공합니다.
학습 목표 생성부터 루브릭 개발까지의 설계 단계를 지원합니다.

설계 단계 (Design Phase):
    Step 0: Instructional Goal Generation - 학습 목표 생성
    Step 1: Instructional Analysis - 교수 분석
    Step 2: Performance Objectives - 수행목표 진술
    Step 3: Rubric Development - 루브릭 개발

주요 클래스:
    InstructionalGoalGenerator: 샘플 데이터 기반 학습 목표 생성
    InstructionalAnalysis: 학습 목표 분석 및 하위 스킬 도출
    PerformanceObjectives: 수행목표(PO) 진술 생성
    RubricDevelopment: Essay형 평가용 루브릭 생성

사용 예시:
    >>> from design_modules import InstructionalGoalGenerator
    >>> generator = InstructionalGoalGenerator(teacher_config)
    >>> goal = generator.generate(samples, domain="math", dataset="gsm8k")
"""
from design_modules.instructional_goal import InstructionalGoalGenerator
from design_modules.analysis import InstructionalAnalysis
from design_modules.objectives import PerformanceObjectives
from design_modules.rubric import RubricDevelopment

__all__ = [
    "InstructionalGoalGenerator",
    "InstructionalAnalysis",
    "PerformanceObjectives",
    "RubricDevelopment",
]
