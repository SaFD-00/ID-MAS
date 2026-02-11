"""교사 모델 (Teacher Model, Mt) 모듈.

Iterative Scaffolding Pipeline에서 교사 역할을 담당합니다.
학생 응답을 평가하고 스캐폴딩을 제공하여 학습을 지원합니다.

파이프라인 담당 단계:
    - Step 2 (Evaluation): Performance Objectives 기반 평가
    - Step 3 (Scaffolding): HOT/LOT 스캐폴딩 아티팩트 + 서술형 피드백 생성
    - Step 5 (Reconstruction): 성공 응답 재구성 또는 최종 솔루션 생성

전체 파이프라인 흐름:
    Step 1: 초기 응답 (Student)
    Step 2: PO 평가 (Teacher) - 이 모듈
    Step 3: 스캐폴딩 (Teacher) - 이 모듈
    Step 4: 재응답 (Student)
    Step 5: 재구성 (Teacher) - 이 모듈
    Step 6: SFT 데이터 생성

주요 클래스:
    TeacherModel: 교사 모델 에이전트

사용 예시:
    >>> from learning_loop.teacher_model import TeacherModel
    >>> teacher = TeacherModel(config)
    >>> result = teacher.evaluate_with_performance_objectives(...)
"""
from models.teacher_wrapper import TeacherModelWrapper
from prompts.learning_prompts import (
    TEACHER_INTERVENTION_PROMPT,
    SCAFFOLDING_ARTIFACT_PROMPT,
    TEACHER_FINAL_SOLUTION_PROMPT,
)
from typing import Dict, Any, List
import json


class TeacherModel:
    """교사 모델 에이전트 클래스.

    Iterative Scaffolding Pipeline에서 학생 응답을 평가하고
    적절한 스캐폴딩을 제공합니다.

    주요 기능:
        - Performance Objectives 기반 평가 (평가 전용)
        - HOT/LOT 구분 스캐폴딩 아티팩트 + 서술형 피드백 생성
        - 성공 응답 재구성 (Case B, 평문)
        - 최종 솔루션 생성 (Case C, 평문)

    Attributes:
        llm: Teacher 모델 래퍼
    """

    def __init__(self, config: dict = None):
        """TeacherModel을 초기화합니다.

        Args:
            config: Teacher 모델 설정. None이면 기본 설정 사용.
        """
        self.llm = TeacherModelWrapper(config)

    def evaluate_with_performance_objectives(
        self,
        student_response: str,
        performance_objectives: List[Dict],
        problem_text: str,
        ground_truth: str,
        iteration_number: int = 1,
        previous_response: str = None,
    ) -> Dict[str, Any]:
        """학생 응답을 Performance Objectives 기준으로 평가합니다.

        평가 전용 — 피드백은 SCAFFOLDING_ARTIFACT_PROMPT에서 생성합니다.

        Args:
            student_response: 학생의 응답
            performance_objectives: Performance Objectives 리스트
            problem_text: 문제 텍스트
            ground_truth: 정답 (교사 참고용, 학생에게 비공개)
            iteration_number: 현재 반복 횟수. 기본값: 1
            previous_response: 이전 iteration의 학생 응답. 기본값: None

        Returns:
            평가 결과 딕셔너리:
                - performance_evaluation (list): 각 PO별 평가 결과
                    - objective_content: PO 내용
                    - is_satisfied: 충족 여부
                    - feedback: 충족 시 강점 / 미충족 시 사유 (Student response 참조)
                - all_satisfied (bool): 전체 PO 충족 여부
                - _failure_metadata: 실패 메타데이터 (fallback 발생 시)
        """
        prompt = TEACHER_INTERVENTION_PROMPT.format(
            problem_text=problem_text,
            student_response=student_response,
            performance_objectives=json.dumps(performance_objectives, ensure_ascii=False, indent=2),
            ground_truth=ground_truth,
            iteration_number=iteration_number,
            prev_iteration_number=max(1, iteration_number - 1),
            previous_response=previous_response if previous_response else "(This is the first attempt. No previous response.)",
        )

        # 최대 3회 재시도
        max_retries = 3
        errors = []  # 모든 에러를 배열로 수집

        for attempt in range(1, max_retries + 1):
            try:
                result = self.llm.generate_json(prompt)
                # _raw_response는 generate_json에서 자동 포함됨 (Task 1)
                result['_input_prompt'] = prompt

                # Ensure required fields exist
                if 'performance_evaluation' not in result:
                    result['performance_evaluation'] = []

                # performance_evaluation에서 all_satisfied 계산
                pe = result.get('performance_evaluation', [])
                result['all_satisfied'] = all(
                    po.get('is_satisfied', False) for po in pe
                ) if pe else False

                # 성공 시 failure metadata 추가 (step2 = PO Evaluation)
                result['_failure_metadata'] = {
                    "step2_performance_objectives_evaluation": {
                        "is_fallback": False,
                        "attempts_needed": attempt
                    }
                }
                return result
            except Exception as e:
                errors.append(str(e))
                if attempt < max_retries:
                    print(f"  Warning: PO evaluation attempt {attempt}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"  Warning: All {max_retries} PO evaluation attempts failed. Last error: {e}")

        # 모든 재시도 실패 시 fallback 반환 (step2 = PO Evaluation)
        return {
            "performance_evaluation": [],
            "all_satisfied": False,
            "_input_prompt": prompt,
            "_failure_metadata": {
                "step2_performance_objectives_evaluation": {
                    "is_fallback": True,
                    "failure_reason": "json_parse_error",
                    "last_error": errors if errors else ["Unknown error"],
                    "max_retries_exceeded": max_retries
                }
            }
        }

    # =========================================================================
    # Scaffolding Artifact Methods
    # =========================================================================

    def generate_scaffolding_artifact(
        self,
        problem_text: str,
        student_response: str,
        po_evaluation: Dict[str, Any],
        iteration_number: int,
        task_analysis: str,
        max_iterations: int = 5,
        previous_iteration_summaries: List[Dict] = None,
        instructional_goal: str = "",
    ) -> Dict[str, Any]:
        """Scaffolding Artifact와 서술형 피드백을 생성합니다.

        미충족 PO에 대해 HOT(High-Order Thinking)/LOT(Low-Order Thinking)를
        구분하여 교수적 스캐폴딩을 설계합니다. 추가로 학생에게 전달할
        통합 서술형 feedback과 iteration summary를 생성합니다.

        Args:
            problem_text: 문제 텍스트
            student_response: 학생의 응답
            po_evaluation: PO 평가 결과
            iteration_number: 현재 반복 횟수
            task_analysis: 과제 분석 결과
            max_iterations: 최대 반복 횟수. 기본값: 5
            previous_iteration_summaries: 이전 iteration의 요약 리스트. 기본값: None
            instructional_goal: 학습 목표. 기본값: ""

        Returns:
            스캐폴딩 결과 딕셔너리:
                - scaffolding_artifacts (list): 스캐폴딩 아티팩트 목록
                - feedback (str): 학생에게 전달할 통합 서술형 피드백
                - iteration_summary: 이 iteration의 요약 (response + scaffolding)
                - _raw_text (str): 전체 구조화된 마크다운 텍스트
                - _failure_metadata: 실패 메타데이터 (fallback 발생 시)
        """
        # Extract failed POs
        failed_objectives = [
            po for po in po_evaluation.get("performance_evaluation", [])
            if not po.get("is_satisfied", True)
        ]

        previous_summaries_str = self._format_iteration_summaries(previous_iteration_summaries)

        prompt = SCAFFOLDING_ARTIFACT_PROMPT.format(
            problem_text=problem_text,
            student_response=student_response[:2000],
            po_evaluation=json.dumps(po_evaluation, ensure_ascii=False, indent=2),
            task_analysis=task_analysis[:1500],
            previous_iteration_summaries=previous_summaries_str,
            instructional_goal=instructional_goal,
        )

        # Retry logic
        max_retries = 3
        errors = []  # 모든 에러를 배열로 수집

        for attempt in range(1, max_retries + 1):
            try:
                raw_text = self.llm.generate(prompt)
                result = self._parse_scaffolding_text(raw_text)
                result['_input_prompt'] = prompt

                # Ensure required fields exist
                if 'scaffolding_artifacts' not in result:
                    result['scaffolding_artifacts'] = []
                if 'feedback' not in result:
                    result['feedback'] = "Review your previous response and try to address the unsatisfied objectives."
                if 'iteration_summary' not in result:
                    result['iteration_summary'] = "Review your previous response and try again with more careful reasoning."

                # Add success metadata (step3 = Scaffolding)
                result['_failure_metadata'] = {
                    "step3_scaffolding_artifact_generation": {
                        "is_fallback": False,
                        "attempts_needed": attempt
                    }
                }
                return result

            except Exception as e:
                errors.append(str(e))
                if attempt < max_retries:
                    print(f"  Warning: Scaffolding artifact attempt {attempt}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"  Warning: All {max_retries} scaffolding artifact attempts failed. Last error: {e}")

        # Fallback: create basic scaffolding from failed objectives
        fallback_artifacts = []
        for failed_po in failed_objectives[:3]:  # Limit to top 3
            fallback_artifacts.append({
                "target_objective": failed_po.get("objective_content", "Unknown objective"),
                "skill_type": "LOT",
                "cognitive_level": "Understand",
                "failure_analysis": failed_po.get("feedback", "Unable to analyze"),
                "scaffolding_content": {
                    "strategy_suggestion": None,
                    "partial_example": None,
                    "feedback": "What key concept might you be missing?",
                    "missed_concept": "Review the problem requirements carefully",
                    "brief_explanation": "Focus on the specific criteria mentioned in the problem",
                    "key_attention_points": "Pay attention to what the question is actually asking"
                }
            })

        return {
            "scaffolding_artifacts": fallback_artifacts,
            "feedback": "Your previous response did not fully meet the performance objectives. Please review the problem requirements and try to address each objective more carefully.",
            "iteration_summary": "Student's response did not meet the performance objectives. Basic scaffolding was provided to guide improvement.",
            "_input_prompt": prompt,
            "_failure_metadata": {
                "step3_scaffolding_artifact_generation": {
                    "is_fallback": True,
                    "failure_reason": "scaffolding_artifact_generation_failed",
                    "last_error": errors if errors else ["Unknown error"],
                    "max_retries_exceeded": max_retries
                }
            }
        }

    def _parse_scaffolding_text(self, text: str) -> Dict[str, Any]:
        """구조화된 마크다운 스캐폴딩 텍스트를 딕셔너리로 파싱합니다."""
        import re

        scaffolding_artifacts = []

        # 1. Scaffolding 섹션 추출 (HOT/LOT)
        section_pattern = r'\[Scaffolding for Task \[(\d+)\] \((High Order Skill|Low Order Skill)\)\][:\s]*'
        parts = re.split(section_pattern, text)
        # parts[0] = 헤더 (Instructional Goal, Analysis)
        # 이후 triplets: [index, skill_type, content]

        i = 1
        while i + 2 < len(parts):
            idx = parts[i]
            skill_label = parts[i + 1]
            content = parts[i + 2]

            # 다음 섹션 경계에서 자르기
            content = re.split(
                r'\[Scaffolding for Task|\[Feedback\]|\[Iteration Summary\]',
                content
            )[0]

            skill_type = "HOT" if "High" in skill_label else "LOT"
            artifact = self._parse_single_scaffold(content, skill_type)
            scaffolding_artifacts.append(artifact)
            i += 3

        # 2. Feedback 추출
        feedback = ""
        feedback_match = re.search(
            r'\[Feedback\]\s*\n(.*?)(?=\[Iteration Summary\]|$)',
            text, re.DOTALL
        )
        if feedback_match:
            feedback = feedback_match.group(1).strip()

        # 3. Iteration Summary 추출
        iteration_summary = ""
        summary_match = re.search(
            r'\[Iteration Summary\]\s*\n(.*?)$',
            text, re.DOTALL
        )
        if summary_match:
            iteration_summary = summary_match.group(1).strip()

        return {
            "scaffolding_artifacts": scaffolding_artifacts,
            "feedback": feedback,
            "iteration_summary": iteration_summary,
            "_raw_text": text,
        }

    def _parse_single_scaffold(self, content: str, skill_type: str) -> Dict[str, Any]:
        """개별 스캐폴딩 섹션을 파싱합니다."""
        import re

        def extract(pattern, text, default=""):
            m = re.search(pattern, text, re.DOTALL)
            return m.group(1).strip() if m else default

        target = extract(r'Target Objective:\s*(.*?)(?=\n- |\Z)', content)
        cognitive = extract(r'Cognitive Level:\s*(.*?)(?=\n- |\Z)', content)
        failure = extract(r'Failure Analysis:\s*(.*?)(?=\n- |\Z)', content)

        if skill_type == "HOT":
            strategy = extract(r'Suggested Strategy:\s*\n(.*?)(?=\n- Key Attention|\Z)', content)
            key_points = extract(r'Key Attention Points:\s*(.*?)(?=\n\[|\Z)', content)
            return {
                "target_objective": target,
                "skill_type": "HOT",
                "cognitive_level": cognitive,
                "failure_analysis": failure,
                "scaffolding_content": {
                    "strategy_suggestion": strategy or None,
                    "partial_example": None,
                    "feedback": None,
                    "missed_concept": None,
                    "brief_explanation": None,
                    "key_attention_points": key_points or None,
                }
            }
        else:  # LOT
            missed = extract(r'Missed Concept/Information:\s*(.*?)(?=\n- |\Z)', content)
            explanation = extract(r'Brief Explanation:\s*(.*?)(?=\n\[|\Z)', content)
            return {
                "target_objective": target,
                "skill_type": "LOT",
                "cognitive_level": cognitive,
                "failure_analysis": failure,
                "scaffolding_content": {
                    "strategy_suggestion": None,
                    "partial_example": None,
                    "feedback": None,
                    "missed_concept": missed or None,
                    "brief_explanation": explanation or None,
                    "key_attention_points": None,
                }
            }

    def generate_final_solution(
        self,
        problem_text: str,
        ground_truth: str,
        task_analysis: str,
        student_weaknesses: List[str],
        max_iterations: int = 5,
        last_iteration_summary: str = "",
    ) -> Dict[str, Any]:
        """Case C: 최종 정답 풀이를 생성합니다.

        학생이 max_iterations 시도 후에도 실패한 경우,
        마지막 iteration의 요약을 참고하여 교육적 정답 풀이를 생성합니다.

        출력 형식: 평문 텍스트 (JSON 아님)

        Args:
            problem_text: 문제 텍스트
            ground_truth: 정답
            task_analysis: 과제 분석 결과
            student_weaknesses: 학생의 약점 목록
            max_iterations: 최대 반복 횟수. 기본값: 5
            last_iteration_summary: 마지막 iteration의 요약. 기본값: ""

        Returns:
            최종 솔루션 딕셔너리:
                - solution_explanation: 완전한 풀이 설명 (평문)
                - _failure_metadata: 실패 메타데이터 (fallback 발생 시)
        """
        weaknesses_str = "\n".join(f"- {w}" for w in student_weaknesses) if student_weaknesses else "- Unable to identify specific weaknesses"

        prompt = TEACHER_FINAL_SOLUTION_PROMPT.format(
            problem_text=problem_text,
            ground_truth=ground_truth,
            task_analysis=task_analysis[:1500],
            last_iteration_summary=last_iteration_summary if last_iteration_summary else "(No iteration summary available)",
            student_weaknesses=weaknesses_str,
            max_iterations=max_iterations,
        )

        # Retry logic
        max_retries = 3
        errors = []  # 모든 에러를 배열로 수집

        for attempt in range(1, max_retries + 1):
            try:
                # 평문 출력 (generate_json → generate)
                plain_text_response = self.llm.generate(prompt, max_tokens=4096)

                result = {
                    "solution_explanation": plain_text_response,
                    "_input_prompt": prompt,
                    "_raw_output": plain_text_response,
                }

                # Add success metadata (step5 = Reconstruction, Case C)
                result['_failure_metadata'] = {
                    "step5_case_c_final_solution": {
                        "is_fallback": False,
                        "attempts_needed": attempt,
                        "case": "C"
                    }
                }
                return result

            except Exception as e:
                errors.append(str(e))
                if attempt < max_retries:
                    print(f"  Warning: Final solution attempt {attempt}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"  Warning: All {max_retries} final solution attempts failed. Last error: {e}")

        # Fallback: create basic solution (step5 = Reconstruction, Case C)
        return {
            "solution_explanation": f"""[Understanding the Problem]
Let me solve this problem step by step.

[Step-by-Step Solution]
Following the correct approach:

The answer is \\boxed{{{ground_truth}}}""",
            "_input_prompt": prompt,
            "_failure_metadata": {
                "step5_case_c_final_solution": {
                    "is_fallback": True,
                    "failure_reason": "final_solution_generation_failed",
                    "last_error": errors if errors else ["Unknown error"],
                    "max_retries_exceeded": max_retries,
                    "case": "C"
                }
            }
        }

    def _format_iteration_summaries(self, summaries: List[Dict]) -> str:
        """이전 iteration summaries를 프롬프트용 문자열로 포맷팅합니다.

        Args:
            summaries: iteration summary 리스트. 각 항목은
                {"iteration": int, "summary": str} 형식.

        Returns:
            포맷된 iteration summaries 문자열
        """
        if not summaries:
            return "(No previous iteration summaries. This is the first attempt.)"

        formatted = []
        for entry in summaries:
            iteration = entry.get("iteration", "?")
            summary = entry.get("summary", "")
            if summary:
                formatted.append(f"[Iteration {iteration} Summary]\n{summary}")

        return "\n\n".join(formatted) if formatted else "(No previous iteration summaries)"

    def _format_scaffolding_history(self, scaffolding_history: List[Dict]) -> str:
        """스캐폴딩 히스토리를 프롬프트용 문자열로 변환합니다.

        Args:
            scaffolding_history: 스캐폴딩 히스토리 리스트

        Returns:
            프롬프트에 포함할 수 있는 포맷된 문자열
        """
        if not scaffolding_history:
            return "(No scaffolding history)"

        formatted = []
        for entry in scaffolding_history:
            iteration = entry.get("iteration", "?")
            summary = entry.get("summary", "")
            artifacts = entry.get("artifacts", [])

            formatted.append(f"\n[Iteration {iteration}]")
            if summary:
                formatted.append(f"Summary: {summary[:500]}")

            for i, artifact in enumerate(artifacts[:2], 1):  # Limit to 2 artifacts per iteration
                skill_type = artifact.get("skill_type", "Unknown")
                target = artifact.get("target_objective", "")[:100]
                formatted.append(f"  Artifact {i} ({skill_type}): {target}")

        return "\n".join(formatted)

    def _format_feedback_history(self, feedback_list: List[Dict]) -> str:
        """전체 iteration의 feedback을 포맷팅합니다.

        Args:
            feedback_list: feedback 히스토리 리스트. 각 항목은
                {"iteration": int, "feedback": str} 형식.

        Returns:
            포맷된 feedback 히스토리 문자열
        """
        if not feedback_list:
            return "(No feedback history)"

        formatted = []
        for entry in feedback_list:
            iteration = entry.get("iteration", "?")
            feedback = entry.get("feedback", "")
            if feedback:
                formatted.append(f"[Iteration {iteration} Feedback]\n{feedback}")

        return "\n\n".join(formatted) if formatted else "(No feedback history)"

    def extract_student_weaknesses(
        self,
        conversation_history: List[Dict]
    ) -> List[str]:
        """대화 히스토리에서 학생의 약점을 추출합니다.

        PO 평가와 스캐폴딩 아티팩트에서 학생의 약점을 수집합니다.

        Args:
            conversation_history: 대화 기록

        Returns:
            학생 약점 목록 (최대 5개)
        """
        weaknesses = []
        seen = set()

        for entry in conversation_history:
            if entry.get("role") == "teacher":
                # From PO evaluation
                evaluation = entry.get("evaluation", {})
                for po in evaluation.get("performance_evaluation", []):
                    if not po.get("is_satisfied", True):
                        reason = po.get("feedback", "")
                        if reason and reason not in seen:
                            weaknesses.append(reason[:200])
                            seen.add(reason)

                # From scaffolding artifact
                artifact = entry.get("scaffolding_artifact", {})
                for art in artifact.get("scaffolding_artifacts", []):
                    failure = art.get("failure_analysis", "")
                    if failure and failure not in seen:
                        weaknesses.append(failure[:200])
                        seen.add(failure)

        return weaknesses[:5]  # Limit to top 5


if __name__ == "__main__":
    # 테스트
    teacher = TeacherModel()

    # 샘플 데이터
    problem = "Calculate 15 + 27"
    student_resp = "15 + 27 = 41"  # Wrong answer
    ground_truth = "42"
    task_analysis = "Instructional Goal: Perform basic arithmetic correctly"

    performance_objectives = [
        {
            "target": "Addition",
            "performance_objective": "Correctly add two numbers, given two integers, producing the exact correct sum."
        }
    ]

    print("=== Performance Objective Evaluation (Iterative Scaffolding) ===")
    evaluation = teacher.evaluate_with_performance_objectives(
        student_response=student_resp,
        performance_objectives=performance_objectives,
        problem_text=problem,
        ground_truth=ground_truth
    )
    print(json.dumps(evaluation, indent=2, ensure_ascii=False))
