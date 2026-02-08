"""교사 모델 (Teacher Model, Mt) 모듈.

Iterative Scaffolding Pipeline에서 교사 역할을 담당합니다.
학생 응답을 평가하고 스캐폴딩을 제공하여 학습을 지원합니다.

파이프라인 담당 단계:
    - Step 2 (Evaluation): Performance Objectives 기반 평가
    - Step 3 (Scaffolding): HOT/LOT 스캐폴딩 아티팩트 생성
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
    INITIAL_HINT_PROMPT,
    PROGRESSIVE_HINT_PROMPT,
    SUMMARY_RECONSTRUCTION_PROMPT,
    TEACHER_INTERVENTION_PROMPT,
    CONVERSATION_SUMMARIZATION_PROMPT,
    SUCCESSFUL_SCAFFOLDING_RECONSTRUCTION_PROMPT,
    # New Scaffolding Artifact prompts
    SCAFFOLDING_ARTIFACT_PROMPT,
    TEACHER_FINAL_SOLUTION_PROMPT,
)
from typing import Dict, Any, Optional, List, Tuple
import json


class TeacherModel:
    """교사 모델 에이전트 클래스.

    Iterative Scaffolding Pipeline에서 학생 응답을 평가하고
    적절한 스캐폴딩을 제공합니다.

    주요 기능:
        - Performance Objectives 기반 평가 (ReAct-style)
        - HOT/LOT 구분 스캐폴딩 아티팩트 생성
        - 성공 응답 재구성 (Case B)
        - 최종 솔루션 생성 (Case C)

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
        ground_truth: str
    ) -> Dict[str, Any]:
        """학생 응답을 Performance Objectives 기준으로 평가합니다.

        ReAct-style 평가를 수행하고 미충족 목표에 대해
        피드백 질문을 생성합니다.

        Args:
            student_response: 학생의 응답
            performance_objectives: Performance Objectives 리스트
            problem_text: 문제 텍스트
            ground_truth: 정답 (교사 참고용, 학생에게 비공개)

        Returns:
            평가 결과 딕셔너리:
                - performance_evaluation (list): 각 PO별 평가 결과
                    - objective_content: PO 내용
                    - is_satisfied: 충족 여부
                    - reason_for_unmet_objective: 미충족 사유
                    - feedback_question: 피드백 질문
                - overall_assessment (dict): 전체 평가 요약
                    - objectives_met: 충족 비율 (예: "3 of 5")
                    - all_satisfied: 전체 충족 여부
                    - primary_weakness: 주요 약점
                    - recommended_focus: 권장 집중 영역
                - _failure_metadata: 실패 메타데이터 (fallback 발생 시)
        """
        prompt = TEACHER_INTERVENTION_PROMPT.format(
            problem_text=problem_text,
            student_response=student_response,
            performance_objectives=json.dumps(performance_objectives, ensure_ascii=False, indent=2),
            ground_truth=ground_truth
        )

        # 최대 3회 재시도
        max_retries = 3
        errors = []  # 모든 에러를 배열로 수집

        for attempt in range(1, max_retries + 1):
            try:
                result = self.llm.generate_json(prompt)

                # overall_assessment 타입 검증 - 잘못되면 예외 발생시켜 재시도
                if 'overall_assessment' in result and not isinstance(result['overall_assessment'], dict):
                    raise ValueError(f"overall_assessment must be dict, got {type(result['overall_assessment']).__name__}")

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
            "overall_assessment": {
                "objectives_met": "0 of 0",
                "all_satisfied": False,
                "primary_weakness": "Evaluation failed",
                "recommended_focus": "Try again"
            },
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
    # Iterative Scaffolding Methods
    # =========================================================================

    def generate_initial_hint(
        self,
        problem_text: str,
        task_analysis: str,
        ground_truth: str
    ) -> Tuple[str, Optional[Dict]]:
        """첫 번째 힌트를 생성합니다.

        학생이 문제를 풀기 전에 방향을 제시합니다.
        정답을 직접 알려주지 않으면서 올바른 접근법을 안내합니다.

        Args:
            problem_text: 문제 텍스트
            task_analysis: 과제 분석 결과
            ground_truth: 정답 (교사 참고용)

        Returns:
            튜플 (hint_text, failure_metadata):
                - hint_text: 생성된 힌트 텍스트
                - failure_metadata: 실패 시 메타데이터, 성공 시 None
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
        """점진적 힌트를 생성합니다.

        이전 시도를 분석하고 더 구체적인 힌트를 제공합니다.
        iteration이 증가할수록 더 상세한 가이드를 제공합니다.

        Args:
            problem_text: 문제 텍스트
            task_analysis: 과제 분석 결과
            conversation_history: 이전 대화 기록
            last_response: 학생의 마지막 응답
            iteration_number: 현재 반복 횟수 (2-5)
            ground_truth: 정답 (교사 참고용)
            max_iterations: 최대 반복 횟수. 기본값: 5

        Returns:
            튜플 (hint_text, failure_metadata):
                - hint_text: 생성된 힌트 텍스트
                - failure_metadata: 실패 시 메타데이터, 성공 시 None
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
        """Case B: 스캐폴딩 성공 후 SFT 데이터용 응답을 재구성합니다.

        학생이 2~5회차에 성공한 경우 사용됩니다:
            1. 스캐폴딩 과정의 핵심 학습 포인트 추출
            2. 최종 성공 응답을 기반으로 정리된 응답 재구성
            3. 교사 가이드의 핵심을 자연스럽게 통합

        Args:
            problem_text: 문제 텍스트
            ground_truth: 정답
            task_analysis: 과제 분석 결과
            conversation_history: 전체 대화 기록
            final_response: 최종 성공한 학생 응답
            iterations_needed: 성공까지 걸린 반복 횟수

        Returns:
            재구성 결과 딕셔너리:
                - reconstructed_response: SFT 데이터로 사용할 재구성된 응답
                - key_learning_points: 스캐폴딩에서 얻은 핵심 학습 포인트
                - improvement_summary: 학생 개선 과정 요약
                - _failure_metadata: 실패 메타데이터 (fallback 발생 시)
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
        errors = []  # 모든 에러를 배열로 수집

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
                # 성공 시 메타데이터 저장 (step5 = Reconstruction, Case B)
                result['_failure_metadata'] = {
                    "step5_case_b_reconstruction": {
                        "is_fallback": False,
                        "attempts_needed": attempt,
                        "case": "B"
                    }
                }
                # Summarization failure가 있으면 추가
                if summarization_failure:
                    result['_failure_metadata']['step5_summarization'] = summarization_failure
                return result
            except Exception as e:
                errors.append(str(e))
                if attempt < max_retries:
                    print(f"  Warning: Attempt {attempt}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"  Warning: All {max_retries} attempts failed. Last error: {e}")

        # 모든 재시도 실패 시 fallback 반환 (step5 = Reconstruction, Case B)
        fallback_result = {
            "reconstructed_response": final_response,
            "key_learning_points": ["Successfully solved through iterative scaffolding"],
            "improvement_summary": f"Student succeeded after {iterations_needed} iterations with teacher guidance",
            "_failure_metadata": {
                "step5_case_b_reconstruction": {
                    "is_fallback": True,
                    "failure_reason": "case_b_reconstruction_failed",
                    "last_error": errors if errors else ["Unknown error"],
                    "max_retries_exceeded": max_retries,
                    "case": "B"
                }
            }
        }
        # Summarization failure가 있으면 추가
        if summarization_failure:
            fallback_result['_failure_metadata']['step5_summarization'] = summarization_failure
        return fallback_result

    def summarize_and_reconstruct(
        self,
        problem_text: str,
        ground_truth: str,
        task_analysis: str,
        conversation_history: List[Dict]
    ) -> Dict[str, Any]:
        """Case C: 최대 시도 후 요약 및 정답을 재구성합니다.

        학생이 max_iterations 시도 후에도 정답을 맞추지 못한 경우:
            1. 대화를 요약하여 학생의 약점 분석
            2. 정답 솔루션을 재구성 (학생 약점 보완 포함)
            3. 학습 포인트 도출

        Args:
            problem_text: 문제 텍스트
            ground_truth: 정답
            task_analysis: 과제 분석 결과
            conversation_history: 전체 대화 기록

        Returns:
            재구성 결과 딕셔너리:
                - summary: 대화 요약
                - student_weaknesses: 학생 약점 목록
                - reconstructed_response: 재구성된 정답 응답
                - learning_points: 학습 포인트 목록
                - _failure_metadata: 실패 메타데이터 (fallback 발생 시)
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
        errors = []  # 모든 에러를 배열로 수집

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
                # 성공 시 메타데이터 저장 (step5 = summarize_and_reconstruct, Case C)
                result['_failure_metadata'] = {
                    "step5_summarize_and_reconstruct": {
                        "is_fallback": False,
                        "attempts_needed": attempt,
                        "case": "C"
                    }
                }
                # Summarization failure가 있으면 추가
                if summarization_failure:
                    result['_failure_metadata']['step5_summarization'] = summarization_failure
                return result
            except Exception as e:
                errors.append(str(e))
                if attempt < max_retries:
                    print(f"  Warning: Attempt {attempt}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"  Warning: All {max_retries} attempts failed. Last error: {e}")

        # 모든 재시도 실패 시 fallback 반환 (step5 = summarize_and_reconstruct, Case C)
        fallback_result = {
            "summary": "Student needed multiple attempts but could not solve the problem.",
            "student_weaknesses": ["Fundamental understanding of the problem"],
            "reconstructed_response": f"""[Understanding the problem]
Let me break down this problem carefully.

[Step-by-step solution]
Following the correct approach:

Answer: {ground_truth}""",
            "learning_points": ["Review the problem-solving approach", "Practice similar problems"],
            "_failure_metadata": {
                "step5_summarize_and_reconstruct": {
                    "is_fallback": True,
                    "failure_reason": "reconstruction_failed",
                    "last_error": errors if errors else ["Unknown error"],
                    "max_retries_exceeded": max_retries,
                    "case": "C"
                }
            }
        }
        # Summarization failure가 있으면 추가
        if summarization_failure:
            fallback_result['_failure_metadata']['step5_summarization'] = summarization_failure
        return fallback_result

    def _format_conversation_history(self, history: List[Dict]) -> str:
        """대화 기록을 프롬프트용 문자열로 변환합니다.

        Args:
            history: 대화 기록 리스트

        Returns:
            프롬프트에 포함할 수 있는 포맷된 문자열
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
        """AI 기반으로 대화 히스토리를 축약합니다.

        Teacher 모델이 대화 히스토리를 분석하여 중요한 학습 포인트를
        파악하고 축약합니다. Rule-based 축약보다 맥락을 더 잘 보존합니다.

        Args:
            problem_text: 문제 텍스트
            ground_truth: 정답
            conversation_history: 전체 대화 기록

        Returns:
            튜플 (summary_text, failure_metadata):
                - summary_text: 축약된 대화 요약
                - failure_metadata: 실패 시 메타데이터, 성공 시 None
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
        """AI 축약 실패 시 단순 truncation을 수행합니다.

        Args:
            history: 대화 기록
            max_total_length: 최대 총 길이. 기본값: 1500

        Returns:
            잘린 히스토리 문자열
        """
        formatted = self._format_conversation_history(history)
        if len(formatted) > max_total_length:
            return formatted[:max_total_length] + "\n\n[... truncated ...]"
        return formatted

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
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """Scaffolding Artifact를 생성합니다.

        미충족 PO에 대해 HOT(High-Order Thinking)/LOT(Low-Order Thinking)를
        구분하여 교수적 스캐폴딩을 설계합니다. 이 스캐폴딩은 학생이
        다음 시도에서 참조하는 DB로 사용됩니다.

        Args:
            problem_text: 문제 텍스트
            student_response: 학생의 응답
            po_evaluation: PO 평가 결과
            iteration_number: 현재 반복 횟수
            task_analysis: 과제 분석 결과
            max_iterations: 최대 반복 횟수. 기본값: 5

        Returns:
            스캐폴딩 결과 딕셔너리:
                - scaffolding_artifacts (list): 스캐폴딩 아티팩트 목록
                    - target_objective: 대상 목표
                    - skill_type: "HOT" 또는 "LOT"
                    - cognitive_level: 인지 수준
                    - failure_analysis: 실패 분석
                    - scaffolding_content: 스캐폴딩 내용
                - scaffolding_summary: 3-5문장 요약
                - _failure_metadata: 실패 메타데이터 (fallback 발생 시)
        """
        # Extract failed POs
        failed_objectives = [
            po for po in po_evaluation.get("performance_evaluation", [])
            if not po.get("is_satisfied", True)
        ]

        prompt = SCAFFOLDING_ARTIFACT_PROMPT.format(
            problem_text=problem_text,
            student_response=student_response[:2000],
            po_evaluation=json.dumps(po_evaluation, ensure_ascii=False, indent=2),
            failed_objectives=json.dumps(failed_objectives, ensure_ascii=False, indent=2),
            task_analysis=task_analysis[:1500],
            iteration_number=iteration_number,
            max_iterations=max_iterations
        )

        # Retry logic
        max_retries = 3
        errors = []  # 모든 에러를 배열로 수집

        for attempt in range(1, max_retries + 1):
            try:
                result = self.llm.generate_json(prompt, max_tokens=4096)

                # Ensure required fields exist
                if 'scaffolding_artifacts' not in result:
                    result['scaffolding_artifacts'] = []
                if 'scaffolding_summary' not in result:
                    result['scaffolding_summary'] = "Review your previous response and try again with more careful reasoning."

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
                "failure_analysis": failed_po.get("reason_for_unmet_objective", "Unable to analyze"),
                "scaffolding_content": {
                    "strategy_suggestion": None,
                    "partial_example": None,
                    "feedback_question": failed_po.get("feedback_question", "What key concept might you be missing?"),
                    "missed_concept": "Review the problem requirements carefully",
                    "brief_explanation": "Focus on the specific criteria mentioned in the problem",
                    "key_attention_points": "Pay attention to what the question is actually asking"
                }
            })

        return {
            "scaffolding_artifacts": fallback_artifacts,
            "scaffolding_summary": "Your previous response did not fully meet the performance objectives. Please review the problem requirements and try to address each objective more carefully.",
            "_failure_metadata": {
                "step3_scaffolding_artifact_generation": {
                    "is_fallback": True,
                    "failure_reason": "scaffolding_artifact_generation_failed",
                    "last_error": errors if errors else ["Unknown error"],
                    "max_retries_exceeded": max_retries
                }
            }
        }

    def generate_final_solution(
        self,
        problem_text: str,
        ground_truth: str,
        task_analysis: str,
        scaffolding_history: List[Dict],
        student_weaknesses: List[str],
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """Case C: 최종 정답 풀이를 생성합니다.

        학생이 max_iterations 시도 후에도 실패한 경우,
        학생의 약점을 반영한 교육적 정답 풀이를 생성합니다.

        Args:
            problem_text: 문제 텍스트
            ground_truth: 정답
            task_analysis: 과제 분석 결과
            scaffolding_history: 누적된 스캐폴딩 히스토리
            student_weaknesses: 학생의 약점 목록
            max_iterations: 최대 반복 횟수. 기본값: 5

        Returns:
            최종 솔루션 딕셔너리:
                - solution_explanation: 완전한 풀이 설명
                - addressed_weaknesses: 다룬 약점 목록
                - key_learning_points: 핵심 학습 포인트
                - final_answer: 최종 정답
                - _failure_metadata: 실패 메타데이터 (fallback 발생 시)
        """
        # Format scaffolding history
        history_str = self._format_scaffolding_history(scaffolding_history)
        weaknesses_str = "\n".join(f"- {w}" for w in student_weaknesses) if student_weaknesses else "- Unable to identify specific weaknesses"

        prompt = TEACHER_FINAL_SOLUTION_PROMPT.format(
            problem_text=problem_text,
            ground_truth=ground_truth,
            task_analysis=task_analysis[:1500],
            scaffolding_history=history_str,
            student_weaknesses=weaknesses_str,
            iterations_count=len(scaffolding_history),
            max_iterations=max_iterations
        )

        # Retry logic
        max_retries = 3
        errors = []  # 모든 에러를 배열로 수집

        for attempt in range(1, max_retries + 1):
            try:
                result = self.llm.generate_json(prompt, max_tokens=4096)

                # Ensure required fields exist
                if 'solution_explanation' not in result:
                    result['solution_explanation'] = f"The correct approach leads to: {ground_truth}"
                if 'addressed_weaknesses' not in result:
                    result['addressed_weaknesses'] = student_weaknesses[:3] if student_weaknesses else []
                if 'key_learning_points' not in result:
                    result['key_learning_points'] = ["Review the problem-solving approach"]
                if 'final_answer' not in result:
                    result['final_answer'] = ground_truth

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

[Final Answer]
Answer: {ground_truth}""",
            "addressed_weaknesses": student_weaknesses[:3] if student_weaknesses else ["Fundamental problem-solving approach"],
            "key_learning_points": [
                "Review the problem requirements carefully",
                "Apply systematic reasoning",
                "Verify each step before proceeding"
            ],
            "final_answer": ground_truth,
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
                        reason = po.get("reason_for_unmet_objective", "")
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
