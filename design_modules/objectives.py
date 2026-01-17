"""교수설계 Step 2: 수행목표 진술(Performance Objectives) 모듈.

이 모듈은 교수 분석 결과로부터 수행목표(PO)를 생성합니다.
Dick & Carey 모델의 수행목표 진술 단계를 구현합니다.

주요 클래스:
    PerformanceObjectives: 수행목표 진술 에이전트

수행목표 구성 요소 (ABCD 모델):
    - Target: 목표 대상 (Terminal Goal, Subskill 등)
    - Behavior: 관찰 가능한 행동
    - Condition: 행동 수행 조건
    - Criterion: 성공 기준

사용 예시:
    >>> from design_modules.objectives import PerformanceObjectives
    >>> obj_gen = PerformanceObjectives(teacher_config)
    >>> result = obj_gen.generate_objectives(analysis_result)
"""
from models.teacher_wrapper import TeacherModelWrapper
from prompts.design_prompts import PERFORMANCE_OBJECTIVES_PROMPT
from typing import Dict, Any, List
import json


class PerformanceObjectives:
    """수행목표 진술 에이전트 클래스.

    교수 분석 결과를 기반으로 ABCD 모델의 수행목표를 생성합니다.
    Teacher 모델을 사용하여 JSON 형식으로 수행목표를 도출합니다.

    Attributes:
        llm: Teacher 모델 래퍼
    """

    def __init__(self, teacher_config: dict = None):
        """PerformanceObjectives를 초기화합니다.

        Args:
            teacher_config: Teacher 모델 설정. None이면 기본 설정 사용.
        """
        self.llm = TeacherModelWrapper(teacher_config)

    def generate_objectives(
        self,
        instructional_analysis: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """교수 분석 결과로부터 수행목표를 생성합니다.

        ABCD 모델 기반의 수행목표를 JSON 형식으로 생성합니다.
        실패 시 최대 max_retries 횟수만큼 재시도합니다.

        Args:
            instructional_analysis: 교수 분석 결과 텍스트
            max_retries: 최대 재시도 횟수 (기본: 3)

        Returns:
            수행목표 딕셔너리:
                - performance_objectives (list): 수행목표 리스트
                    각 항목: {target, Behavior, Condition, Criterion}

        Raises:
            RuntimeError: max_retries 횟수 초과 시
        """
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                prompt = PERFORMANCE_OBJECTIVES_PROMPT.format(
                    instructional_analysis=instructional_analysis
                )

                # LLM으로 JSON 형식으로 생성
                result = self.llm.generate_json(prompt)

                if not result:
                    raise ValueError("Empty response from LLM")

                # 결과 검증
                if not self.validate_objectives(result):
                    raise ValueError("Invalid objectives format")

                return result

            except Exception as e:
                last_error = e
                print(f"  [Attempt {attempt}/{max_retries}] Performance Objectives generation failed: {e}")

                if attempt < max_retries:
                    print(f"  Retrying...")
                else:
                    print(f"  [FATAL] All {max_retries} attempts failed.")

        raise RuntimeError(
            f"Performance Objectives generation failed after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def validate_objectives(self, objectives: Dict[str, Any]) -> bool:
        """생성된 수행목표의 유효성을 검증합니다.

        수행목표가 ABCD 모델의 필수 필드를 모두 포함하는지 확인합니다.

        Args:
            objectives: 검증할 수행목표 딕셔너리

        Returns:
            모든 필수 필드가 존재하면 True, 아니면 False
        """
        if "performance_objectives" not in objectives:
            return False

        for obj in objectives["performance_objectives"]:
            required_fields = ["target", "Behavior", "Condition", "Criterion"]
            if not all(field in obj for field in required_fields):
                return False

        return True


if __name__ == "__main__":
    # 테스트
    obj_generator = PerformanceObjectives()

    # 샘플 교수 분석 결과
    sample_analysis = """
Terminal Goal: classify Iris dataset using linear regression (Apply – Procedural Knowledge)
├── [1] prepare dataset for modeling (Apply – Procedural Knowledge)
│   ├── [1-1] load Iris dataset (Remember – Factual Knowledge)
│   └── [1-2] preprocess data for regression (Apply – Procedural Knowledge)
├── [2] construct linear regression model (Apply – Procedural Knowledge)
│   ├── [2-1] select predictor and target variables (Understand – Conceptual Knowledge)
│   └── [2-2] fit linear regression algorithm (Apply – Procedural Knowledge)
└── [3] evaluate model performance (Analyze – Procedural Knowledge)
    └── [3-1] compute and interpret evaluation metrics (Analyze – Conceptual Knowledge)
"""

    result = obj_generator.generate_objectives(sample_analysis)

    print("=== Performance Objectives ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n=== Validation ===")
    print(f"Valid: {obj_generator.validate_objectives(result)}")
