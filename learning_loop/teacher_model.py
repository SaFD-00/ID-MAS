"""
교사 모델 (Teacher Model, Mt)

Iterative Scaffolding Pipeline Support:
- Scaffolding: evaluate_with_performance_objectives, generate_initial_hint, generate_progressive_hint
- Case B: summarize_and_reconstruct (5회 실패 후 대화 분석 및 재구성)
"""
from models.teacher_wrapper import TeacherModelWrapper
from prompts.learning_prompts import (
    INITIAL_HINT_PROMPT,
    PROGRESSIVE_HINT_PROMPT,
    SUMMARY_RECONSTRUCTION_PROMPT,
    TEACHER_INTERVENTION_PROMPT,
    CONVERSATION_SUMMARIZATION_PROMPT,
    SUCCESSFUL_SCAFFOLDING_RECONSTRUCTION_PROMPT,
)
from typing import Dict, Any, Optional, List, Tuple
import json


class TeacherModel:
    """교사 모델 - Iterative Scaffolding을 위한 학생 응답 평가 및 가이드"""

    def __init__(self, config: dict = None):
        """
        TeacherModel 초기화

        Args:
            config: Teacher model 설정 딕셔너리 (None이면 기본 설정 사용)
        """
        self.llm = TeacherModelWrapper(config)

    def evaluate_with_performance_objectives(
        self,
        student_response: str,
        performance_objectives: List[Dict],
        problem_text: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        Iterative Scaffolding: ReAct-style PO 평가 및 Socratic 질문 생성

        학생 응답을 Performance Objectives 기준으로 평가하고,
        미충족 목표에 대해 Socratic 질문을 생성합니다.

        Args:
            student_response: 학생의 응답
            performance_objectives: Performance Objectives 리스트
            problem_text: 문제 텍스트
            ground_truth: 정답 (교사 참고용, 공개 금지)

        Returns:
            {
                "performance_evaluation": [
                    {
                        "objective_content": str,
                        "is_satisfied": bool,
                        "reason_for_unmet_objective": str or None,
                        "socratic_question": str or None
                    }
                ],
                "overall_assessment": {
                    "objectives_met": str,
                    "all_satisfied": bool,
                    "primary_weakness": str or None,
                    "recommended_focus": str or None
                }
            }
        """
        prompt = TEACHER_INTERVENTION_PROMPT.format(
            problem_text=problem_text,
            student_response=student_response,
            performance_objectives=json.dumps(performance_objectives, ensure_ascii=False, indent=2),
            ground_truth=ground_truth
        )

        # 최대 3회 재시도
        max_retries = 3
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                result = self.llm.generate_json(prompt)
                # Ensure required fields exist
                if 'performance_evaluation' not in result:
                    result['performance_evaluation'] = []
                if 'overall_assessment' not in result:
                    result['overall_assessment'] = {
                        "objectives_met": "0 of 0",
                        "all_satisfied": False,
                        "primary_weakness": "Unable to evaluate",
                        "recommended_focus": "Review the problem"
                    }
                # 성공 시 failure metadata 추가
                result['_failure_metadata'] = {
                    "evaluation": {
                        "is_fallback": False,
                        "attempts_needed": attempt,
                        "stage": "performance_objectives_evaluation"
                    }
                }
                return result
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"  Warning: PO evaluation attempt {attempt}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"  Warning: All {max_retries} PO evaluation attempts failed. Last error: {e}")

        # 모든 재시도 실패 시 fallback 반환
        return {
            "performance_evaluation": [],
            "overall_assessment": {
                "objectives_met": "0 of 0",
                "all_satisfied": False,
                "primary_weakness": "Evaluation failed",
                "recommended_focus": "Try again"
            },
            "_failure_metadata": {
                "evaluation": {
                    "is_fallback": True,
                    "failure_reason": "json_parse_error",
                    "last_error": str(last_error) if last_error else "Unknown error",
                    "max_retries_exceeded": max_retries,
                    "stage": "performance_objectives_evaluation"
                }
            }
        }

    # =========================================================================
    # Iterative Scaffolding Methods
    # =========================================================================

    def generate_initial_hint(
        self,
        problem_text: str,
        task_analysis: str,
        ground_truth: str
    ) -> Tuple[str, Optional[Dict]]:
        """
        Iterative Scaffolding: 첫 번째 힌트 생성

        학생이 문제를 풀기 전에 방향을 제시하는 힌트.
        답을 알려주지 않으면서 올바른 접근법을 안내.

        Args:
            problem_text: 문제 텍스트
            task_analysis: 과제 분석 결과
            ground_truth: 정답 (교사 참고용)

        Returns:
            Tuple[hint_text, failure_metadata]
        """
        prompt = INITIAL_HINT_PROMPT.format(
            problem_text=problem_text,
            task_analysis=task_analysis[:2000],
            ground_truth=ground_truth
        )

        try:
            response = self.llm.generate(prompt)
            return response, None
        except Exception as e:
            print(f"  Warning: Failed to generate initial hint: {e}")
            fallback_hint = "Let's start by carefully reading the problem and identifying what information is given and what we need to find."
            failure_metadata = {
                "is_fallback": True,
                "failure_reason": "generation_failed",
                "last_error": str(e),
                "stage": "initial_hint"
            }
            return fallback_hint, failure_metadata

    def generate_progressive_hint(
        self,
        problem_text: str,
        task_analysis: str,
        conversation_history: List[Dict],
        last_response: str,
        iteration_number: int,
        ground_truth: str,
        max_iterations: int = 5
    ) -> Tuple[str, Optional[Dict]]:
        """
        Iterative Scaffolding: 점진적 힌트 생성

        이전 시도를 분석하고 더 구체적인 힌트 제공.
        iteration이 높아질수록 더 상세한 가이드 제공.

        Args:
            problem_text: 문제 텍스트
            task_analysis: 과제 분석 결과
            conversation_history: 이전 대화 기록
            last_response: 학생의 마지막 응답
            iteration_number: 현재 반복 횟수 (2-5)
            ground_truth: 정답 (교사 참고용)
            max_iterations: 최대 반복 횟수

        Returns:
            Tuple[hint_text, failure_metadata]
        """
        formatted_history = self._format_conversation_history(conversation_history)

        prompt = PROGRESSIVE_HINT_PROMPT.format(
            problem_text=problem_text,
            task_analysis=task_analysis[:1500],
            conversation_history=formatted_history,
            last_response=last_response[:1500],
            iteration_number=iteration_number,
            max_iterations=max_iterations,
            ground_truth=ground_truth
        )

        try:
            response = self.llm.generate(prompt)
            return response, None
        except Exception as e:
            print(f"  Warning: Failed to generate progressive hint: {e}")
            fallback_hint = f"Look at your previous answer. There seems to be an error. Let me help you: Review step by step and check if the calculation is correct."
            failure_metadata = {
                "is_fallback": True,
                "failure_reason": "generation_failed",
                "last_error": str(e),
                "iteration": iteration_number,
                "stage": "progressive_hint"
            }
            return fallback_hint, failure_metadata

    def reconstruct_successful_scaffolding(
        self,
        problem_text: str,
        ground_truth: str,
        task_analysis: str,
        conversation_history: List[Dict],
        final_response: str,
        iterations_needed: int
    ) -> Dict[str, Any]:
        """
        Case B: Iterative Scaffolding 성공 후 SFT 데이터용 응답 재구성

        학생이 2~5회차에 성공한 경우:
        1. scaffolding 과정에서 얻은 핵심 학습 포인트 추출
        2. 최종 성공 응답을 기반으로 깔끔하게 재구성
        3. teacher guidance의 핵심을 자연스럽게 통합

        Args:
            problem_text: 문제 텍스트
            ground_truth: 정답
            task_analysis: 과제 분석 결과
            conversation_history: 전체 대화 기록
            final_response: 최종 성공한 학생 응답
            iterations_needed: 성공까지 걸린 iteration 수

        Returns:
            {
                "reconstructed_response": str,  # SFT 데이터로 사용할 재구성된 응답
                "key_learning_points": List[str],  # scaffolding에서 얻은 핵심 학습 포인트
                "improvement_summary": str  # 학생이 어떻게 개선했는지 요약
            }
        """
        # AI 기반 대화 히스토리 축약
        conversation_summary, summarization_failure = self.summarize_conversation_with_ai(
            problem_text=problem_text,
            ground_truth=ground_truth,
            conversation_history=conversation_history
        )

        prompt = SUCCESSFUL_SCAFFOLDING_RECONSTRUCTION_PROMPT.format(
            problem_text=problem_text,
            ground_truth=ground_truth,
            task_analysis=task_analysis[:1500],
            iterations_needed=iterations_needed,
            conversation_summary=conversation_summary,
            final_response=final_response
        )

        # 최대 3회 재시도
        max_retries = 3
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                # max_tokens를 4096으로 제한 (컨텍스트 초과 방지)
                result = self.llm.generate_json(prompt, max_tokens=4096)
                # Ensure required fields exist
                if 'reconstructed_response' not in result:
                    result['reconstructed_response'] = final_response
                if 'key_learning_points' not in result:
                    result['key_learning_points'] = ["Successfully solved through iterative refinement"]
                if 'improvement_summary' not in result:
                    result['improvement_summary'] = f"Improved through {iterations_needed} iterations of scaffolding"
                # 성공 시 메타데이터를 Dict of dicts로 저장
                result['_failure_metadata'] = {
                    "reconstruction": {
                        "is_fallback": False,
                        "attempts_needed": attempt,
                        "stage": "case_b_reconstruction"
                    }
                }
                # Summarization failure가 있으면 추가
                if summarization_failure:
                    result['_failure_metadata']['summarization'] = summarization_failure
                return result
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"  Warning: Attempt {attempt}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"  Warning: All {max_retries} attempts failed. Last error: {e}")

        # 모든 재시도 실패 시 fallback 반환
        fallback_result = {
            "reconstructed_response": final_response,
            "key_learning_points": ["Successfully solved through iterative scaffolding"],
            "improvement_summary": f"Student succeeded after {iterations_needed} iterations with teacher guidance",
            # Fallback 메타데이터를 Dict of dicts로 저장
            "_failure_metadata": {
                "reconstruction": {
                    "is_fallback": True,
                    "failure_reason": "case_b_reconstruction_failed",
                    "last_error": str(last_error) if last_error else "Unknown error",
                    "max_retries_exceeded": max_retries,
                    "stage": "case_b_reconstruction"
                }
            }
        }
        # Summarization failure가 있으면 추가
        if summarization_failure:
            fallback_result['_failure_metadata']['summarization'] = summarization_failure
        return fallback_result

    def summarize_and_reconstruct(
        self,
        problem_text: str,
        ground_truth: str,
        task_analysis: str,
        conversation_history: List[Dict]
    ) -> Dict[str, Any]:
        """
        Iterative Scaffolding: 5회 실패 후 요약 및 정답 재구성 (Case C)

        학생이 5번 시도해도 정답을 못 맞춘 경우:
        1. 대화를 요약하여 학생의 약점 분석
        2. 정답 솔루션을 재구성 (학생 약점 보완 포함)
        3. 학습 포인트 도출

        Args:
            problem_text: 문제 텍스트
            ground_truth: 정답
            task_analysis: 과제 분석 결과
            conversation_history: 전체 대화 기록

        Returns:
            {
                "summary": str,
                "student_weaknesses": List[str],
                "reconstructed_response": str,
                "learning_points": List[str]
            }
        """
        # AI 기반 대화 히스토리 축약 (JSON 파싱 오류 방지)
        # Teacher 모델이 중요한 학습 포인트를 파악하여 축약
        summarized_history, summarization_failure = self.summarize_conversation_with_ai(
            problem_text=problem_text,
            ground_truth=ground_truth,
            conversation_history=conversation_history
        )

        prompt = SUMMARY_RECONSTRUCTION_PROMPT.format(
            problem_text=problem_text,
            ground_truth=ground_truth,
            task_analysis=task_analysis[:1500],
            conversation_history=summarized_history
        )

        # 최대 3회 재시도
        max_retries = 3
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                # max_tokens를 4096으로 제한 (컨텍스트 초과 방지)
                result = self.llm.generate_json(prompt, max_tokens=4096)
                # Ensure required fields exist
                if 'summary' not in result:
                    result['summary'] = "Student struggled with this problem after multiple attempts."
                if 'student_weaknesses' not in result:
                    result['student_weaknesses'] = ["Could not identify the correct approach"]
                if 'reconstructed_response' not in result:
                    result['reconstructed_response'] = f"Let me solve this correctly.\n\nAnswer: {ground_truth}"
                if 'learning_points' not in result:
                    result['learning_points'] = ["Review the fundamental concepts"]
                # 성공 시 메타데이터를 Dict of dicts로 저장
                result['_failure_metadata'] = {
                    "reconstruction": {
                        "is_fallback": False,
                        "attempts_needed": attempt,
                        "stage": "case_c_reconstruction"
                    }
                }
                # Summarization failure가 있으면 추가
                if summarization_failure:
                    result['_failure_metadata']['summarization'] = summarization_failure
                return result
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"  Warning: Attempt {attempt}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"  Warning: All {max_retries} attempts failed. Last error: {e}")

        # 모든 재시도 실패 시 fallback 반환
        fallback_result = {
            "summary": "Student needed multiple attempts but could not solve the problem.",
            "student_weaknesses": ["Fundamental understanding of the problem"],
            "reconstructed_response": f"""[Understanding the problem]
Let me break down this problem carefully.

[Step-by-step solution]
Following the correct approach:

Answer: {ground_truth}""",
            "learning_points": ["Review the problem-solving approach", "Practice similar problems"],
            # Fallback 메타데이터를 Dict of dicts로 저장
            "_failure_metadata": {
                "reconstruction": {
                    "is_fallback": True,
                    "failure_reason": "reconstruction_failed",
                    "last_error": str(last_error) if last_error else "Unknown error",
                    "max_retries_exceeded": max_retries,
                    "stage": "case_c_reconstruction"
                }
            }
        }
        # Summarization failure가 있으면 추가
        if summarization_failure:
            fallback_result['_failure_metadata']['summarization'] = summarization_failure
        return fallback_result

    def _format_conversation_history(self, history: List[Dict]) -> str:
        """
        대화 기록을 프롬프트에 포함할 수 있는 형식으로 변환

        Args:
            history: 대화 기록 리스트

        Returns:
            포맷된 문자열
        """
        if not history:
            return "(No previous conversation)"

        formatted = []
        for entry in history:
            if entry.get("role") == "teacher":
                hint = entry.get("hint", "")
                iteration = entry.get("iteration", "?")
                formatted.append(f"[Teacher Hint {iteration}]\n{hint}")
            elif entry.get("role") == "student":
                response = entry.get("response", "")
                iteration = entry.get("iteration", "?")
                formatted.append(f"[Student Response {iteration}]\n{response}")

        return "\n\n".join(formatted)

    def summarize_conversation_with_ai(
        self,
        problem_text: str,
        ground_truth: str,
        conversation_history: List[Dict]
    ) -> Tuple[str, Optional[Dict]]:
        """
        AI 기반 대화 히스토리 축약

        Teacher 모델이 대화 히스토리를 분석하여 중요한 학습 포인트를 파악하고 축약.
        Rule-based 축약보다 맥락을 더 잘 보존함.

        Args:
            problem_text: 문제 텍스트
            ground_truth: 정답
            conversation_history: 전체 대화 기록

        Returns:
            Tuple[summary_text, failure_metadata]
        """
        # 전체 히스토리를 포맷팅
        formatted_history = self._format_conversation_history(conversation_history)

        prompt = CONVERSATION_SUMMARIZATION_PROMPT.format(
            problem_text=problem_text,
            ground_truth=ground_truth,
            conversation_history=formatted_history
        )

        try:
            # 일반 텍스트로 생성 (JSON 아님)
            summary = self.llm.generate(prompt)
            return summary, None
        except Exception as e:
            print(f"  Warning: AI summarization failed: {e}")
            # Fallback: 기본 포맷 사용 (truncated)
            truncated = self._fallback_truncate_history(conversation_history)
            failure_metadata = {
                "is_fallback": True,
                "failure_reason": "summarization_failed",
                "last_error": str(e),
                "stage": "conversation_summarization"
            }
            return truncated, failure_metadata

    def _fallback_truncate_history(self, history: List[Dict], max_total_length: int = 1500) -> str:
        """
        Fallback: AI 축약 실패 시 단순 truncation

        Args:
            history: 대화 기록
            max_total_length: 최대 총 길이

        Returns:
            truncated 히스토리
        """
        formatted = self._format_conversation_history(history)
        if len(formatted) > max_total_length:
            return formatted[:max_total_length] + "\n\n[... truncated ...]"
        return formatted


if __name__ == "__main__":
    # 테스트
    teacher = TeacherModel()

    # 샘플 데이터
    problem = "Calculate 15 + 27"
    student_resp = "15 + 27 = 41"  # Wrong answer
    ground_truth = "42"
    task_analysis = "Terminal Goal: Perform basic arithmetic correctly"

    performance_objectives = [
        {
            "target": "Addition",
            "Behavior": "Correctly add two numbers",
            "Condition": "Given two integers",
            "Criterion": "Produce the correct sum"
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

    print("\n=== Initial Hint ===")
    hint = teacher.generate_initial_hint(
        problem_text=problem,
        task_analysis=task_analysis,
        ground_truth=ground_truth
    )
    print(hint)
