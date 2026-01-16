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

    def analyze(self, learning_objective: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        학습 목표를 분석하여 Instructional Goal, Subskills, Subtasks 생성 (최대 3번 재시도)

        Args:
            learning_objective: 학습 목표
            max_retries: 최대 재시도 횟수 (기본 3)

        Returns:
            분석 결과 딕셔너리

        Raises:
            RuntimeError: max_retries 초과 시
        """
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                prompt = INSTRUCTIONAL_ANALYSIS_PROMPT.format(
                    learning_objective=learning_objective
                )

                # LLM으로 분석 수행
                result_text = self.llm.generate(prompt)

                if not result_text or not result_text.strip():
                    raise ValueError("Empty response from LLM")

                # 결과 파싱
                parsed_result = self._parse_analysis_result(result_text)

                return {
                    "learning_objective": learning_objective,
                    "raw_output": result_text,
                    "parsed": parsed_result
                }

            except Exception as e:
                last_error = e
                print(f"  [Attempt {attempt}/{max_retries}] Instructional Analysis failed: {e}")

                if attempt < max_retries:
                    print(f"  Retrying...")
                else:
                    print(f"  [FATAL] All {max_retries} attempts failed.")

        raise RuntimeError(
            f"Instructional Analysis failed after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

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
            "instructional_goal": "",
            "subskills": []
        }

        for line in lines:
            line = line.strip()

            if "Instructional Goal:" in line:
                parsed["instructional_goal"] = line.replace("Instructional Goal:", "").strip()

            elif line and not line.startswith("#"):
                # Subskill/Subtask 파싱
                if line.startswith("├──") or line.startswith("└──") or line.startswith("│"):
                    parsed["subskills"].append(line)

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
