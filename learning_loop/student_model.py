"""
학생 모델 (Student Model, Ms)

3-Phase Pipeline Support:
- Phase 1: generate_initial_response_with_scaffolding
- Phase 2: generate_fixed_response_with_coaching
"""
from models.student_wrapper import StudentModelWrapper
from prompts.learning_prompts import (
    SCAFFOLDING_SYSTEM_PROMPT,
    COACHING_RESPONSE_PROMPT,
    ITERATIVE_SCAFFOLDING_SYSTEM_PROMPT,
    STUDENT_WITH_HINT_PROMPT,
    STUDENT_SOCRATIC_RESPONSE_PROMPT,
)
from typing import Dict, Any, Optional, List
import json


class StudentModel:
    """학생 모델 - 문제 응답 생성"""

    def __init__(
        self,
        model_name: str = None,
        use_sft_model: bool = False,
        use_sft_idmas_model: bool = False,
        sft_domain: str = None
    ):
        """
        Args:
            model_name: 사용할 모델 이름 (None이면 기본 모델)
            use_sft_model: True면 SFT fine-tuned 모델 사용
            use_sft_idmas_model: True면 SFT_ID-MAS fine-tuned 모델 사용
            sft_domain: SFT/SFT_ID-MAS 모델의 도메인 (예: "math")
        """
        self.model = StudentModelWrapper(
            model_name=model_name,
            use_sft_model=use_sft_model,
            use_sft_idmas_model=use_sft_idmas_model,
            sft_domain=sft_domain
        )
        self.model_name = self.model.model_name

    def generate_initial_response(
        self,
        problem_text: str,
        system_message: Optional[str] = None
    ) -> str:
        """
        기본 초기 응답 생성 (평가용)

        Args:
            problem_text: 문제 설명
            system_message: 시스템 메시지 (선택)

        Returns:
            생성된 응답
        """
        if system_message is None:
            system_message = "You are a student learning to solve problems. Answer the following question step by step."

        response = self.model.generate(
            prompt=problem_text,
            system_message=system_message
        )

        return response

    def generate_initial_response_with_scaffolding(
        self,
        problem_text: str,
        task_analysis: str,
        terminal_goal: str = ""
    ) -> str:
        """
        Phase 1: Scaffolding과 함께 초기 응답 생성

        과제분석 결과를 System Prompt에 포함하여 문제 풀이 지원
        Terminal Goal 달성을 목표로 응답 생성

        Args:
            problem_text: 문제 설명
            task_analysis: 과제분석 결과 (Instructional Analysis)
            terminal_goal: 학습 목표 (Terminal Goal)

        Returns:
            생성된 응답
        """
        system_message = SCAFFOLDING_SYSTEM_PROMPT.format(
            terminal_goal=terminal_goal if terminal_goal else "solve the problem correctly",
            task_analysis=task_analysis
        )

        response = self.model.generate(
            prompt=problem_text,
            system_message=system_message
        )

        return response

    def generate_fixed_response_with_coaching(
        self,
        problem_text: str,
        coaching_db: Dict[str, Any],
        task_analysis: str,
        learning_objective: str
    ) -> str:
        """
        Phase 2: Coaching DB를 참고하여 수정 응답 생성

        Args:
            problem_text: 문제 설명
            coaching_db: Coaching Database (HOT/LOT 스캐폴딩 포함)
            task_analysis: 과제분석 결과
            learning_objective: 학습 목표 (Terminal Goal)

        Returns:
            생성된 응답
        """
        # Format coaching DB as JSON string for the new prompt format
        coaching_db_str = json.dumps(coaching_db, ensure_ascii=False, indent=2)

        prompt = COACHING_RESPONSE_PROMPT.format(
            learning_objective=learning_objective,
            task_analysis=task_analysis[:2000],  # Truncate if too long
            coaching_db=coaching_db_str,
            problem_text=problem_text
        )

        response = self.model.generate(
            prompt=prompt,
            system_message="You are a student using coaching guidance to solve a problem correctly. You MUST explicitly cite which information from the Coaching DB you are using."
        )

        return response

    def respond_to_socratic_questions(
        self,
        problem_text: str,
        teacher_evaluation: Dict[str, Any],
        previous_response: str,
        task_analysis: str
    ) -> str:
        """
        Phase 1 Iterative: 교사의 Socratic 질문에 응답하여 개선된 풀이 생성

        Args:
            problem_text: 문제 텍스트
            teacher_evaluation: 교사의 PO 평가 및 Socratic 질문
            previous_response: 이전 응답
            task_analysis: 과제분석 결과

        Returns:
            개선된 응답
        """
        # Format teacher evaluation for the prompt
        teacher_eval_str = json.dumps(teacher_evaluation, ensure_ascii=False, indent=2)

        prompt = STUDENT_SOCRATIC_RESPONSE_PROMPT.format(
            problem_text=problem_text,
            previous_response=previous_response[:1500],  # Truncate if too long
            teacher_evaluation=teacher_eval_str,
            task_analysis=task_analysis[:1500]
        )

        response = self.model.generate(
            prompt=prompt,
            system_message="You are a student learning from teacher feedback. Carefully address the Socratic questions and improve your solution."
        )

        return response

    # =========================================================================
    # Phase 1: Iterative Scaffolding Methods (NEW)
    # =========================================================================

    def generate_response_with_hint(
        self,
        problem_text: str,
        teacher_hint: str,
        conversation_history: Optional[List[Dict]] = None,
        task_analysis: Optional[str] = None
    ) -> str:
        """
        Phase 1 Iterative: 교사 힌트를 참고하여 응답 생성

        Args:
            problem_text: 문제 텍스트
            teacher_hint: 교사가 제공한 현재 힌트
            conversation_history: 이전 대화 기록 (선택)
            task_analysis: 과제 분석 결과 (선택)

        Returns:
            생성된 응답
        """
        # Build system message
        if task_analysis:
            system_message = ITERATIVE_SCAFFOLDING_SYSTEM_PROMPT.format(
                task_analysis=task_analysis[:1500]
            )
        else:
            system_message = "You are a student learning to solve problems with teacher guidance. Show your thinking step by step and provide your final answer clearly."

        # Build the prompt
        prompt = STUDENT_WITH_HINT_PROMPT.format(
            problem_text=problem_text,
            teacher_hint=teacher_hint
        )

        # Include conversation context if available
        if conversation_history:
            context_str = self._format_previous_attempts(conversation_history)
            if context_str:
                prompt = f"[Previous Attempts]\n{context_str}\n\n{prompt}"

        response = self.model.generate(
            prompt=prompt,
            system_message=system_message
        )

        return response

    def _format_previous_attempts(self, history: List[Dict]) -> str:
        """
        이전 시도를 포맷하여 컨텍스트로 사용

        Args:
            history: 대화 기록

        Returns:
            포맷된 문자열
        """
        if not history:
            return ""

        formatted = []
        for entry in history:
            if entry.get("role") == "student":
                response = entry.get("response", "")
                iteration = entry.get("iteration", "?")
                # Truncate long responses
                if len(response) > 500:
                    response = response[:500] + "..."
                formatted.append(f"[My Attempt {iteration}]\n{response}")

        return "\n\n".join(formatted)


if __name__ == "__main__":
    # 테스트
    student = StudentModel()

    # 초기 응답 테스트
    problem = "Explain how linear regression works and why it might be suitable for the Iris dataset."

    print("=== Initial Response ===")
    initial_response = student.generate_initial_response(problem)
    print(initial_response)

    # Scaffolding 테스트
    task_analysis = """
    Terminal Goal: Explain machine learning concepts clearly
    ├── [1] Understand the algorithm
    │   ├── [1-1] Define the objective function
    │   └── [1-2] Explain optimization process
    └── [2] Apply to specific dataset
        └── [2-1] Analyze dataset characteristics
    """

    print("\n=== Response with Scaffolding ===")
    scaffolded_response = student.generate_initial_response_with_scaffolding(
        problem_text=problem,
        task_analysis=task_analysis
    )
    print(scaffolded_response)
