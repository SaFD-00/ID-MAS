"""교수설계 Step 1: 교수 분석(Instructional Analysis) 모듈.

이 모듈은 학습 목표를 분석하여 Instructional Goal, Subskills, Subtasks를
도출합니다. Dick & Carey 모델의 교수 분석 단계를 구현합니다.

주요 클래스:
    InstructionalAnalysis: 교수 분석 에이전트

교수 분석 출력:
    - Instructional Goal: 최종 학습 목표
    - Subskills: 하위 기술 (계층적 구조)
    - Subtasks: 세부 과제

사용 예시:
    >>> from design_modules.analysis import InstructionalAnalysis
    >>> analyzer = InstructionalAnalysis(teacher_config)
    >>> result = analyzer.analyze("학습 목표 텍스트")
"""
from models.teacher_wrapper import TeacherModelWrapper
from prompts.design_prompts import INSTRUCTIONAL_ANALYSIS_SYSTEM_PROMPT, INSTRUCTIONAL_ANALYSIS_USER_PROMPT
from typing import Dict, Any
import json


class InstructionalAnalysis:
    """교수 분석 에이전트 클래스.

    학습 목표를 분석하여 하위 기술과 세부 과제를 계층적 구조로 도출합니다.
    Teacher 모델을 사용하여 분석을 수행합니다.

    Attributes:
        llm: Teacher 모델 래퍼
    """

    def __init__(self, teacher_config: dict = None):
        """InstructionalAnalysis를 초기화합니다.

        Args:
            teacher_config: Teacher 모델 설정. None이면 기본 설정 사용.
        """
        self.llm = TeacherModelWrapper(teacher_config)

    def analyze(self, learning_objective: str, max_retries: int = 3) -> Dict[str, Any]:
        """학습 목표를 분석합니다.

        학습 목표를 분석하여 Instructional Goal, Subskills, Subtasks를 생성합니다.
        실패 시 최대 max_retries 횟수만큼 재시도합니다.

        Args:
            learning_objective: 분석할 학습 목표 텍스트
            max_retries: 최대 재시도 횟수 (기본: 3)

        Returns:
            분석 결과 딕셔너리:
                - learning_objective (str): 입력된 학습 목표
                - raw_output (str): LLM 원본 출력
                - parsed (dict): 파싱된 구조화 결과
                    - instructional_goal: 최종 목표
                    - subskills: 하위 기술 리스트

        Raises:
            RuntimeError: max_retries 횟수 초과 시
        """
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                system_message = INSTRUCTIONAL_ANALYSIS_SYSTEM_PROMPT
                user_prompt = INSTRUCTIONAL_ANALYSIS_USER_PROMPT.format(
                    learning_objective=learning_objective
                )

                # LLM으로 분석 수행
                result_text = self.llm.generate(user_prompt, system_message=system_message)

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
        """분석 결과 텍스트를 구조화된 데이터로 파싱합니다.

        LLM 출력 텍스트에서 Instructional Goal과 Subskills를 추출합니다.
        트리 구조(├──, └──, │)를 파싱하여 계층적 구조를 보존합니다.

        Args:
            result_text: LLM 출력 텍스트

        Returns:
            구조화된 분석 결과:
                - instructional_goal: 최종 학습 목표
                - subskills: 하위 기술 리스트
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
