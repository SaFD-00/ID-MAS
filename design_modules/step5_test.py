"""
5단계: Test Item 개발
"""
from models.gpt_wrapper import GPTWrapper
from prompts.design_prompts import TEST_ITEM_DEVELOPMENT_PROMPT
from typing import Dict, Any, List, Optional
import json


class TestItemDevelopment:
    """Test Item 개발 모듈"""

    def __init__(self, teacher_config: dict = None):
        """
        Args:
            teacher_config: Teacher model 설정 (None이면 기본 설정 사용)
        """
        self.gpt = GPTWrapper(teacher_config)

    def generate_test_items(
        self,
        performance_objectives: Dict[str, Any],
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        수행목표로부터 Test Item 생성

        Args:
            performance_objectives: 수행목표 딕셔너리
            ground_truth: Ground Truth 정보 (선택)

        Returns:
            Test Item 딕셔너리 (JSON)
        """
        # 각 수행목표에 대해 Test Item 생성
        all_items = []

        for obj in performance_objectives.get("performance_objectives", []):
            prompt = TEST_ITEM_DEVELOPMENT_PROMPT.format(
                performance_objective=json.dumps(obj, ensure_ascii=False),
                ground_truth=ground_truth or "N/A"
            )

            result = self.gpt.generate_json(prompt)
            all_items.extend(result.get("assessment_items", []))

        return {"assessment_items": all_items}

    def is_essay_type(self, test_item: Dict[str, Any]) -> bool:
        """
        Test Item이 Essay 타입인지 확인

        Args:
            test_item: Test Item 딕셔너리

        Returns:
            Essay 타입 여부
        """
        return test_item.get("type") == "Essay"

    def get_essay_items(self, test_items: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Essay 타입 Test Item 필터링

        Args:
            test_items: 전체 Test Item 딕셔너리

        Returns:
            Essay 타입 Test Item 리스트
        """
        return [
            item for item in test_items.get("assessment_items", [])
            if self.is_essay_type(item)
        ]


if __name__ == "__main__":
    # 테스트
    test_dev = TestItemDevelopment()

    # 샘플 수행목표
    sample_objectives = {
        "performance_objectives": [
            {
                "target": "Terminal Goal",
                "Behavior": "Apply procedural knowledge to classify the Iris dataset using linear regression",
                "Condition": "Given a complete Iris dataset and a Python coding environment",
                "Criterion": "The final classification output must achieve correct execution without errors and produce predictions for all 150 instances"
            },
            {
                "target": "Subtask 1-1",
                "Behavior": "Recall factual knowledge to load the Iris dataset",
                "Condition": "Using standard Python data-loading libraries",
                "Criterion": "Dataset must be loaded successfully on the first attempt with 100% of data fields correctly imported"
            }
        ]
    }

    result = test_dev.generate_test_items(sample_objectives)

    print("=== Test Items ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n=== Essay Items ===")
    essay_items = test_dev.get_essay_items(result)
    print(json.dumps(essay_items, indent=2, ensure_ascii=False))
