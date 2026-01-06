"""
5-1단계: 루브릭 개발 (Essay형 Test Item용)
"""
from models.gpt_wrapper import GPTWrapper
from prompts.rubric_templates import (
    RUBRIC_CRITERION_TEMPLATES,
    RUBRIC_GENERATION_PROMPT
)
from typing import Dict, Any
import json


class RubricDevelopment:
    """루브릭 개발 모듈"""

    def __init__(self, teacher_config: dict = None):
        """
        Args:
            teacher_config: Teacher model 설정 (None이면 기본 설정 사용)
        """
        self.gpt = GPTWrapper(teacher_config)
        self.templates = RUBRIC_CRITERION_TEMPLATES

    def generate_rubric(
        self,
        task_description: str,
        output_type: str,
        performance_objectives: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Essay형 Test Item에 대한 루브릭 생성

        Args:
            task_description: 평가 과제 설명
            output_type: 기대 출력 유형
                (explanatory_text, analytical_essay, evaluative_essay,
                 argumentative_essay, comparative_analysis, design_proposal)
            performance_objectives: 수행목표 딕셔너리

        Returns:
            루브릭 딕셔너리 (JSON)
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

        result = self.gpt.generate_json(prompt, system_message)

        return result

    def validate_rubric(self, rubric: Dict[str, Any]) -> bool:
        """
        생성된 루브릭의 유효성 검증

        Args:
            rubric: 루브릭 딕셔너리

        Returns:
            유효성 여부
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
