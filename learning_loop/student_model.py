"""
학생 모델 (Student Model, Ms)

Iterative Scaffolding Pipeline Support:
- Step 1 (Initial Response): generate_initial_response_with_scaffolding
- Step 4 (Re-response): respond_with_scaffolding_artifact

Pipeline Steps:
- Step 1: Initial Response (Student) - this module
- Step 2: PO Evaluation (Teacher)
- Step 3: Scaffolding Artifact (Teacher)
- Step 4: Re-response (Student) - this module
- Step 5: Reconstruction (Teacher)
- Step 6: SFT Generation
"""
from models.student_wrapper import StudentModelWrapper
from prompts.learning_prompts import (
    SCAFFOLDING_SYSTEM_PROMPT,
    ITERATIVE_SCAFFOLDING_SYSTEM_PROMPT,
    STUDENT_WITH_HINT_PROMPT,
    STUDENT_SOCRATIC_RESPONSE_PROMPT,
    # New Scaffolding Artifact prompt
    STUDENT_WITH_ARTIFACT_PROMPT,
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
        instructional_goal: str = ""
    ) -> str:
        """
        Scaffolding과 함께 초기 응답 생성

        과제분석 결과를 System Prompt에 포함하여 문제 풀이 지원
        Terminal Goal 달성을 목표로 응답 생성

        Args:
            problem_text: 문제 설명
            task_analysis: 과제분석 결과 (Instructional Analysis)
            instructional_goal: 학습 목표 (Terminal Goal)

        Returns:
            생성된 응답
        """
        system_message = SCAFFOLDING_SYSTEM_PROMPT.format(
            instructional_goal=instructional_goal if instructional_goal else "solve the problem correctly",
            task_analysis=task_analysis
        )

        response = self.model.generate(
            prompt=problem_text,
            system_message=system_message
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
        Iterative Scaffolding: 교사의 Socratic 질문에 응답하여 개선된 풀이 생성

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
    # Iterative Scaffolding Methods
    # =========================================================================

    def generate_response_with_hint(
        self,
        problem_text: str,
        teacher_hint: str,
        conversation_history: Optional[List[Dict]] = None,
        task_analysis: Optional[str] = None
    ) -> str:
        """
        Iterative Scaffolding: 교사 힌트를 참고하여 응답 생성

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

    # =========================================================================
    # Scaffolding Artifact Methods (NEW - Replaces Socratic Questioning)
    # =========================================================================

    def respond_with_scaffolding_artifact(
        self,
        problem_text: str,
        previous_response: str,
        scaffolding_artifact: Dict[str, Any],
        task_analysis: str
    ) -> str:
        """
        Scaffolding Artifact를 참조하여 개선된 응답 생성

        Teacher가 생성한 Scaffolding DB(artifact)를 참조하여
        개선된 풀이를 생성합니다. 학생은 DB에서 어떤 정보를
        참조했는지 명시적으로 언급해야 합니다.

        Args:
            problem_text: 문제 텍스트
            previous_response: 이전 응답
            scaffolding_artifact: Scaffolding DB (artifacts + summary)
            task_analysis: 과제 분석 결과

        Returns:
            DB 참조 정보를 명시한 개선된 응답
        """
        # Format scaffolding artifacts for the prompt
        artifacts_str = self._format_scaffolding_artifacts(
            scaffolding_artifact.get("scaffolding_artifacts", [])
        )
        scaffolding_summary = scaffolding_artifact.get("scaffolding_summary", "")

        prompt = STUDENT_WITH_ARTIFACT_PROMPT.format(
            problem_text=problem_text,
            previous_response=previous_response[:1500],
            scaffolding_summary=scaffolding_summary,
            scaffolding_artifacts=artifacts_str,
            task_analysis=task_analysis[:1500]
        )

        response = self.model.generate(
            prompt=prompt,
            system_message="You are a student learning from scaffolding guidance. Apply the provided scaffolding to improve your solution and explicitly acknowledge what you learned from the Scaffolding DB."
        )

        return response

    def _format_scaffolding_artifacts(self, artifacts: List[Dict]) -> str:
        """
        Scaffolding artifacts를 읽기 쉬운 형식으로 포맷

        Args:
            artifacts: Scaffolding artifact 리스트

        Returns:
            포맷된 문자열
        """
        if not artifacts:
            return "(No detailed artifacts available)"

        formatted = []
        for i, artifact in enumerate(artifacts, 1):
            skill_type = artifact.get("skill_type", "Unknown")
            target = artifact.get("target_objective", "Unknown objective")
            cognitive_level = artifact.get("cognitive_level", "")
            failure_analysis = artifact.get("failure_analysis", "")
            content = artifact.get("scaffolding_content", {})

            formatted.append(f"\n=== Artifact {i} ({skill_type} - {cognitive_level}) ===")
            formatted.append(f"Target Objective: {target}")

            if failure_analysis:
                formatted.append(f"Your Issue: {failure_analysis}")

            if skill_type == "HOT":
                # High-Order Thinking scaffolding
                if content.get("strategy_suggestion"):
                    formatted.append(f"Suggested Strategy: {content['strategy_suggestion']}")
                if content.get("partial_example"):
                    formatted.append(f"Partial Example: {content['partial_example']}")
                if content.get("socratic_question"):
                    formatted.append(f"Guiding Question: {content['socratic_question']}")
            else:
                # Low-Order Thinking scaffolding
                if content.get("missed_concept"):
                    formatted.append(f"Missed Concept: {content['missed_concept']}")
                if content.get("brief_explanation"):
                    formatted.append(f"Explanation: {content['brief_explanation']}")

            if content.get("key_attention_points"):
                formatted.append(f"Key Points to Remember: {content['key_attention_points']}")

        return "\n".join(formatted)

    def extract_db_references(self, response: str) -> List[str]:
        """
        학생 응답에서 DB 참조 정보 추출

        Args:
            response: 학생의 응답

        Returns:
            참조된 DB 정보 목록
        """
        references = []

        # Look for the "Information Retrieved from Scaffolding DB" section
        if "Information Retrieved from Scaffolding DB:" in response:
            # Extract the section
            parts = response.split("Information Retrieved from Scaffolding DB:")
            if len(parts) > 1:
                ref_section = parts[1].split("Improved Reasoning:")[0] if "Improved Reasoning:" in parts[1] else parts[1]
                # Extract bullet points
                for line in ref_section.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("-") or line.startswith("•"):
                        references.append(line[1:].strip())

        return references


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
