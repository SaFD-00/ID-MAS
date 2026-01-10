"""
2단계: 교수 분석 (Instructional Analysis)
"""
from models.teacher_wrapper import TeacherModelWrapper
from prompts.design_prompts import INSTRUCTIONAL_ANALYSIS_PROMPT
from typing import Dict, Any
import json


class InstructionalAnalysis:
    """교수 분석 모듈"""

    def __init__(self, teacher_config: dict = None):
        """
        Args:
            teacher_config: Teacher model 설정 (None이면 기본 설정 사용)
        """
        self.llm = TeacherModelWrapper(teacher_config)

    def analyze(self, learning_objective: str) -> Dict[str, Any]:
        """
        학습 목표를 분석하여 Terminal Goal, Subskills, Subtasks 생성

        Args:
            learning_objective: 학습 목표

        Returns:
            분석 결과 딕셔너리
        """
        prompt = INSTRUCTIONAL_ANALYSIS_PROMPT.format(
            learning_objective=learning_objective
        )

        # LLM으로 분석 수행
        result_text = self.llm.generate(prompt)

        # 결과 파싱
        parsed_result = self._parse_analysis_result(result_text)

        return {
            "learning_objective": learning_objective,
            "raw_output": result_text,
            "parsed": parsed_result
        }

    def _parse_analysis_result(self, result_text: str) -> Dict[str, Any]:
        """
        분석 결과 텍스트를 구조화된 데이터로 파싱

        Args:
            result_text: LLM 출력 텍스트

        Returns:
            구조화된 분석 결과
        """
        lines = result_text.strip().split('\n')

        parsed = {
            "terminal_goal": "",
            "subskills": [],
            "prerequisite_knowledge": []
        }

        current_section = None

        for line in lines:
            line = line.strip()

            if "Terminal Goal:" in line:
                parsed["terminal_goal"] = line.replace("Terminal Goal:", "").strip()
                current_section = "subskills"

            elif "### Prerequisite Knowledge" in line:
                current_section = "prerequisite"

            elif current_section == "subskills" and line and not line.startswith("#"):
                # Subskill/Subtask 파싱
                if line.startswith("├──") or line.startswith("└──") or line.startswith("│"):
                    parsed["subskills"].append(line)

            elif current_section == "prerequisite" and line and "/" in line:
                parsed["prerequisite_knowledge"].append(line)

        return parsed


if __name__ == "__main__":
    # 테스트
    analyzer = InstructionalAnalysis()
    result = analyzer.analyze(
        "classify the given Iris Species dataset using linear regression."
    )

    print("=== Instructional Analysis Result ===")
    print(result["raw_output"])
    print("\n=== Parsed Result ===")
    print(json.dumps(result["parsed"], indent=2, ensure_ascii=False))
