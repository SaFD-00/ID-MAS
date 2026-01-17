"""교수설계 Step 3: 루브릭 개발(Rubric Development) 모듈.

이 모듈은 Essay형 Test Item에 대한 루브릭을 개발합니다.
Dick & Carey 모델의 평가도구 개발 단계를 구현합니다.

주요 클래스:
    RubricDevelopment: 루브릭 개발 에이전트

지원하는 출력 타입:
    - explanatory_text: 설명적 텍스트
    - analytical_essay: 분석적 에세이
    - evaluative_essay: 평가적 에세이
    - argumentative_essay: 논증적 에세이
    - comparative_analysis: 비교 분석
    - design_proposal: 설계 제안서

루브릭 구조:
    - criteria: 평가 기준 리스트
        - name: 기준명
        - levels: 수준별 설명 (1-4점)

사용 예시:
    >>> from design_modules.rubric import RubricDevelopment
    >>> rubric_dev = RubricDevelopment(teacher_config)
    >>> result = rubric_dev.generate_rubric(task_desc, "explanatory_text", objectives)
"""
from models.teacher_wrapper import TeacherModelWrapper
from prompts.rubric_templates import (
    RUBRIC_CRITERION_TEMPLATES,
    RUBRIC_GENERATION_PROMPT
)
from typing import Dict, Any
import json


class RubricDevelopment:
    """루브릭 개발 에이전트 클래스.

    수행목표와 과제 설명을 기반으로 4점 척도 루브릭을 생성합니다.
    Teacher 모델을 사용하여 JSON 형식으로 루브릭을 도출합니다.

    Attributes:
        llm: Teacher 모델 래퍼
        templates: 루브릭 기준 템플릿
    """

    def __init__(self, teacher_config: dict = None):
        """RubricDevelopment를 초기화합니다.

        Args:
            teacher_config: Teacher 모델 설정. None이면 기본 설정 사용.
        """
        self.llm = TeacherModelWrapper(teacher_config)
        self.templates = RUBRIC_CRITERION_TEMPLATES

    def generate_rubric(
        self,
        task_description: str,
        output_type: str,
        performance_objectives: Dict[str, Any],
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Essay형 Test Item에 대한 루브릭을 생성합니다.

        과제 설명과 수행목표를 기반으로 4점 척도 루브릭을 생성합니다.
        실패 시 최대 max_retries 횟수만큼 재시도합니다.

        Args:
            task_description: 평가 과제 설명
            output_type: 기대 출력 유형. 다음 중 하나:
                - explanatory_text: 설명적 텍스트
                - analytical_essay: 분석적 에세이
                - evaluative_essay: 평가적 에세이
                - argumentative_essay: 논증적 에세이
                - comparative_analysis: 비교 분석
                - design_proposal: 설계 제안서
            performance_objectives: 수행목표 딕셔너리
            max_retries: 최대 재시도 횟수 (기본: 3)

        Returns:
            루브릭 딕셔너리:
                - rubric (dict): 루브릭 정보
                    - criteria (list): 평가 기준 리스트
                        - name: 기준명
                        - levels: {"1": "...", "2": "...", "3": "...", "4": "..."}

        Raises:
            ValueError: 지원하지 않는 output_type인 경우
            RuntimeError: max_retries 횟수 초과 시
        """
        # 출력 타입 검증
        valid_types = [
            "explanatory_text", "analytical_essay", "evaluative_essay",
            "argumentative_essay", "comparative_analysis", "design_proposal"
        ]

        if output_type not in valid_types:
            raise ValueError(
                f"Invalid output_type: {output_type}. "
                f"Must be one of {valid_types}"
            )

        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                prompt = RUBRIC_GENERATION_PROMPT.format(
                    task_description=task_description,
                    performance_objectives=json.dumps(
                        performance_objectives,
                        ensure_ascii=False
                    ),
                    rubric_criterion_templates=json.dumps(
                        self.templates,
                        ensure_ascii=False
                    )
                )

                # 시스템 메시지에 output_type 포함
                system_message = f"Expected output type: {output_type}"

                result = self.llm.generate_json(prompt, system_message)

                if not result:
                    raise ValueError("Empty response from LLM")

                # 결과 검증
                if not self.validate_rubric(result):
                    raise ValueError("Invalid rubric format")

                return result

            except Exception as e:
                last_error = e
                print(f"  [Attempt {attempt}/{max_retries}] Rubric Development failed: {e}")

                if attempt < max_retries:
                    print(f"  Retrying...")
                else:
                    print(f"  [FATAL] All {max_retries} attempts failed.")

        raise RuntimeError(
            f"Rubric Development failed after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def validate_rubric(self, rubric: Dict[str, Any]) -> bool:
        """생성된 루브릭의 유효성을 검증합니다.

        루브릭이 필수 구조(rubric > criteria > levels 1-4)를 갖추었는지 확인합니다.

        Args:
            rubric: 검증할 루브릭 딕셔너리

        Returns:
            모든 필수 구조가 존재하면 True, 아니면 False
        """
        if "rubric" not in rubric:
            return False

        if "criteria" not in rubric["rubric"]:
            return False

        for criterion in rubric["rubric"]["criteria"]:
            if "name" not in criterion or "levels" not in criterion:
                return False

            levels = criterion["levels"]
            required_levels = ["1", "2", "3", "4"]
            if not all(level in levels for level in required_levels):
                return False

        return True


if __name__ == "__main__":
    # 테스트
    rubric_dev = RubricDevelopment()

    # 샘플 데이터
    task_desc = "Explain the concept of linear regression and how it applies to the Iris dataset classification problem."
    output_type = "explanatory_text"

    sample_objectives = {
        "performance_objectives": [
            {
                "target": "Terminal Goal",
                "Behavior": "Apply procedural knowledge to classify the Iris dataset using linear regression",
                "Condition": "Given a complete Iris dataset and a Python coding environment",
                "Criterion": "The final classification output must achieve correct execution without errors"
            }
        ]
    }

    result = rubric_dev.generate_rubric(
        task_description=task_desc,
        output_type=output_type,
        performance_objectives=sample_objectives
    )

    print("=== Generated Rubric ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n=== Validation ===")
    print(f"Valid: {rubric_dev.validate_rubric(result)}")
