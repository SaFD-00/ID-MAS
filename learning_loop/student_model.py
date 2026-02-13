"""학생 모델 (Student Model, Ms) 모듈.

Iterative Scaffolding Pipeline에서 학생 역할을 담당합니다.
문제에 대한 응답을 생성하고 교사 피드백을 반영하여 개선합니다.

파이프라인 담당 단계:
    - Step 1 (Initial Response): 초기 응답 생성
    - Step 4 (Re-response): 교사 피드백 기반 재응답

전체 파이프라인 흐름:
    Step 1: 초기 응답 (Student) - 이 모듈
    Step 2: PO 평가 (Teacher)
    Step 3: 스캐폴딩 (Teacher)
    Step 4: 재응답 (Student) - 이 모듈
    Step 5: 재구성 (Teacher)
    Step 6: SFT 데이터 생성

주요 클래스:
    StudentModel: 학생 모델 에이전트

사용 예시:
    >>> from learning_loop.student_model import StudentModel
    >>> student = StudentModel(model_name="Qwen/Qwen3-1.7B")
    >>> response = student.generate_initial_response_with_scaffolding(...)
"""
from models.student_wrapper import StudentModelWrapper
from prompts.learning_prompts import (
    SCAFFOLDING_SYSTEM_PROMPT,
    STUDENT_FEEDBACK_RESPONSE_PROMPT,
    STUDENT_SELF_REFINEMENT_PROMPT,
)
from typing import Dict, Any, Optional, List


class StudentModel:
    """학생 모델 에이전트 클래스.

    Iterative Scaffolding Pipeline에서 문제에 대한 응답을 생성합니다.
    SCAFFOLDING_SYSTEM_PROMPT를 중심으로 일관된 system message를 사용합니다.

    주요 기능:
        - 초기 응답 생성 (Task Analysis 기반)
        - 교사 피드백 기반 재응답 생성
        - Self-Refinement 응답 생성 (Case A/B, 긍정 피드백 기반)

    Attributes:
        model: Student 모델 래퍼
        model_name: 사용 중인 모델 이름
    """

    def __init__(
        self,
        model_name: str = None,
        use_sft_model: bool = False,
        use_sft_idmas_model: bool = False,
        sft_domain: str = None,
        gpu_ids=None
    ):
        """StudentModel을 초기화합니다.

        Args:
            model_name: 사용할 모델 이름. None이면 기본 모델 사용.
            use_sft_model: True면 SFT fine-tuned 모델 사용
            use_sft_idmas_model: True면 SFT_ID-MAS fine-tuned 모델 사용
            sft_domain: SFT/SFT_ID-MAS 모델의 도메인 (예: "math")
            gpu_ids: GPU 인덱스 tuple (예: (0,), (0,1,2)).
                None이면 CUDA_VISIBLE_DEVICES 기반 자동 할당.
        """
        self.model = StudentModelWrapper(
            model_name=model_name,
            use_sft_model=use_sft_model,
            use_sft_idmas_model=use_sft_idmas_model,
            sft_domain=sft_domain,
            gpu_ids=gpu_ids,
        )
        self.model_name = self.model.model_name

    def generate_initial_response(
        self,
        problem_text: str,
        system_message: Optional[str] = None
    ) -> str:
        """기본 초기 응답을 생성합니다.

        스캐폴딩 없이 단순 응답을 생성합니다. 평가용으로 사용됩니다.

        Args:
            problem_text: 문제 텍스트
            system_message: 시스템 메시지. None이면 기본 메시지 사용.

        Returns:
            생성된 응답 텍스트
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
        instructional_goal: str = "",
        dataset_prompt: str = "",
    ) -> str:
        """스캐폴딩과 함께 초기 응답을 생성합니다.

        과제분석 결과를 System Prompt에 포함하여 문제 풀이를 지원합니다.
        Instructional Goal 달성을 목표로 응답을 생성합니다.

        Args:
            problem_text: 문제 텍스트 (순수 input)
            task_analysis: 과제 분석 결과 (Instructional Analysis)
            instructional_goal: 학습 목표 (Instructional Goal). 기본값: ""
            dataset_prompt: 데이터셋별 instruction. 기본값: ""

        Returns:
            생성된 응답 텍스트
        """
        scaffolding_prompt = SCAFFOLDING_SYSTEM_PROMPT.format(
            instructional_goal=instructional_goal if instructional_goal else "solve the problem correctly",
            task_analysis=task_analysis
        )
        if dataset_prompt:
            system_message = f"{dataset_prompt}\n\n{scaffolding_prompt}"
        else:
            system_message = scaffolding_prompt

        response = self.model.generate(
            prompt=problem_text,
            system_message=system_message
        )

        return response

    def respond_to_feedback(
        self,
        problem_text: str,
        scaffolding_text: str,
        task_analysis: str,
        instructional_goal: str = "",
        dataset_prompt: str = ""
    ) -> str:
        """교사의 scaffolding artifact를 참조하여 개선된 풀이를 생성합니다.

        dataset_prompt + SCAFFOLDING_SYSTEM_PROMPT + scaffolding artifact를
        system message에 통합하고 problem_text(순수 input)만 user message로 전달합니다.

        Args:
            problem_text: 문제 텍스트 (순수 input)
            scaffolding_text: 교사의 전체 scaffolding artifact 텍스트
            task_analysis: 과제 분석 결과
            instructional_goal: 학습 목표. 기본값: ""
            dataset_prompt: 데이터셋별 instruction. 기본값: ""

        Returns:
            개선된 응답 텍스트
        """
        feedback_prompt = STUDENT_FEEDBACK_RESPONSE_PROMPT.format(
            scaffolding_system_prompt=SCAFFOLDING_SYSTEM_PROMPT.format(
                instructional_goal=instructional_goal if instructional_goal else "solve the problem correctly",
                task_analysis=task_analysis
            ),
            scaffolding_artifact=scaffolding_text
        )
        if dataset_prompt:
            system_message = f"{dataset_prompt}\n\n{feedback_prompt}"
        else:
            system_message = feedback_prompt

        response = self.model.generate(
            prompt=problem_text,
            system_message=system_message
        )

        return response

    def self_refine_response(
        self,
        problem_text: str,
        positive_feedback: str,
        task_analysis: str,
        instructional_goal: str = "",
    ) -> str:
        """교사의 긍정 피드백을 반영하여 응답을 Self-Refine합니다.

        모든 PO를 충족한 후, 교사의 피드백을 반영하여
        추론 과정의 질을 높이되 최종 답은 유지합니다.

        Args:
            problem_text: 문제 텍스트
            positive_feedback: 교사의 긍정 피드백 텍스트
            task_analysis: 과제 분석 결과
            instructional_goal: 학습 목표. 기본값: ""

        Returns:
            Self-Refined 응답 텍스트
        """
        scaffolding_system_prompt = SCAFFOLDING_SYSTEM_PROMPT.format(
            instructional_goal=instructional_goal if instructional_goal else "solve the problem correctly",
            task_analysis=task_analysis,
        )

        system_message = STUDENT_SELF_REFINEMENT_PROMPT.format(
            scaffolding_system_prompt=scaffolding_system_prompt,
            positive_feedback=positive_feedback,
        )

        response = self.model.generate(
            prompt=problem_text,
            system_message=system_message,
        )

        return response

    def get_self_refinement_system_message(
        self,
        positive_feedback: str,
        task_analysis: str,
        instructional_goal: str = "",
    ) -> str:
        """Self-Refinement의 system message를 반환합니다 (로깅용)."""
        scaffolding_system_prompt = SCAFFOLDING_SYSTEM_PROMPT.format(
            instructional_goal=instructional_goal if instructional_goal else "solve the problem correctly",
            task_analysis=task_analysis,
        )
        return STUDENT_SELF_REFINEMENT_PROMPT.format(
            scaffolding_system_prompt=scaffolding_system_prompt,
            positive_feedback=positive_feedback,
        )

    def get_initial_system_message(
        self,
        instructional_goal: str = "",
        task_analysis: str = "",
        dataset_prompt: str = "",
    ) -> str:
        """Step 1의 system message를 반환합니다 (로깅용)."""
        scaffolding_prompt = SCAFFOLDING_SYSTEM_PROMPT.format(
            instructional_goal=instructional_goal if instructional_goal else "solve the problem correctly",
            task_analysis=task_analysis
        )
        if dataset_prompt:
            return f"{dataset_prompt}\n\n{scaffolding_prompt}"
        return scaffolding_prompt

    def get_feedback_system_message(
        self,
        scaffolding_text: str,
        task_analysis: str,
        instructional_goal: str = "",
        dataset_prompt: str = "",
    ) -> str:
        """Step 4의 system message를 반환합니다 (로깅용)."""
        feedback_prompt = STUDENT_FEEDBACK_RESPONSE_PROMPT.format(
            scaffolding_system_prompt=SCAFFOLDING_SYSTEM_PROMPT.format(
                instructional_goal=instructional_goal if instructional_goal else "solve the problem correctly",
                task_analysis=task_analysis
            ),
            scaffolding_artifact=scaffolding_text
        )
        if dataset_prompt:
            return f"{dataset_prompt}\n\n{feedback_prompt}"
        return feedback_prompt

    def extract_db_references(self, response: str) -> List[str]:
        """학생 응답에서 Artifact 참조 정보를 추출합니다.

        "Information Retrieved from Scaffolding Artifact:" 섹션에서
        참조된 정보 목록을 추출합니다.

        Args:
            response: 학생의 응답 텍스트

        Returns:
            참조된 Artifact 정보 목록
        """
        references = []

        # Look for the "Information Retrieved from Scaffolding Artifact" section
        if "Information Retrieved from Scaffolding Artifact:" in response:
            # Extract the section
            parts = response.split("Information Retrieved from Scaffolding Artifact:")
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
    Instructional Goal: Explain machine learning concepts clearly
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
