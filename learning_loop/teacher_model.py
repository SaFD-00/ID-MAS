"""
교사 모델 (Teacher Model, Mt)

3-Phase Pipeline Support:
- Phase 2: score_by_performance_objectives, analyze_weak_objectives, generate_coaching_db
- Phase 3: generate_modeling_response
"""
from models.gpt_wrapper import GPTWrapper
from prompts.learning_prompts import (
    PERFORMANCE_SCORING_PROMPT,
    WEAK_OBJECTIVE_ANALYSIS_PROMPT,
    COACHING_DB_GENERATION_PROMPT,
    MODELING_PROMPT,
    INITIAL_HINT_PROMPT,
    PROGRESSIVE_HINT_PROMPT,
    SUMMARY_RECONSTRUCTION_PROMPT,
)
from typing import Dict, Any, Optional, List
import json


class TeacherModel:
    """교사 모델 - 학생 응답 평가, Coaching, Modeling"""

    def __init__(self, config: dict = None):
        """
        TeacherModel 초기화

        Args:
            config: Teacher model 설정 딕셔너리 (None이면 기본 설정 사용)
        """
        self.gpt = GPTWrapper(config)

    def score_by_performance_objectives(
        self,
        student_response: str,
        performance_objectives: List[Dict],
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        Phase 2: Performance Objective 기준으로 학생 응답 채점

        Args:
            student_response: 학생의 응답
            performance_objectives: Performance Objectives 리스트
            ground_truth: 정답

        Returns:
            {
                "overall_correct": bool,
                "objective_scores": [
                    {
                        "objective_target": str,
                        "score": float (0.0-1.0),
                        "demonstrated_behavior": str,
                        "weaknesses": List[str]
                    }
                ],
                "weak_objectives": List[str]
            }
        """
        prompt = PERFORMANCE_SCORING_PROMPT.format(
            student_response=student_response,
            performance_objectives=json.dumps(performance_objectives, ensure_ascii=False, indent=2),
            ground_truth=ground_truth
        )

        try:
            result = self.gpt.generate_json(prompt)
            # Ensure required fields exist
            if 'overall_correct' not in result:
                result['overall_correct'] = False
            if 'objective_scores' not in result:
                result['objective_scores'] = []
            if 'weak_objectives' not in result:
                result['weak_objectives'] = []
            return result
        except Exception as e:
            print(f"  Warning: Failed to parse scoring result: {e}")
            return {
                "overall_correct": False,
                "objective_scores": [],
                "weak_objectives": []
            }

    def analyze_weak_objectives(
        self,
        weak_objectives: List[Dict],
        student_responses: List[str],
        task_analysis: str
    ) -> Dict[str, Any]:
        """
        Phase 2: 40% 이상 오류를 보인 Performance Objective 분석

        Args:
            weak_objectives: 취약 PO 리스트 (error_rate 포함)
            student_responses: 오류를 보인 학생 응답들
            task_analysis: 과제분석 결과

        Returns:
            {
                "weak_performance_areas": [...],
                "recommended_strategies": [...],
                "examples_needed": [...]
            }
        """
        # Limit responses to avoid context overflow
        sample_responses = student_responses[:5]

        prompt = WEAK_OBJECTIVE_ANALYSIS_PROMPT.format(
            weak_objectives=json.dumps(weak_objectives, ensure_ascii=False, indent=2),
            student_responses=json.dumps(sample_responses, ensure_ascii=False, indent=2),
            task_analysis=task_analysis[:3000]  # Truncate if too long
        )

        try:
            result = self.gpt.generate_json(prompt)
            if 'weak_performance_areas' not in result:
                result['weak_performance_areas'] = []
            if 'recommended_strategies' not in result:
                result['recommended_strategies'] = []
            if 'examples_needed' not in result:
                result['examples_needed'] = []
            return result
        except Exception as e:
            print(f"  Warning: Failed to analyze weak objectives: {e}")
            return {
                "weak_performance_areas": [],
                "recommended_strategies": ["Review the problem carefully"],
                "examples_needed": []
            }

    def generate_coaching_db(
        self,
        learning_objective: str,
        task_analysis: str,
        weak_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Phase 2: Coaching Database 생성

        Args:
            learning_objective: 학습 목표 (Terminal Goal)
            task_analysis: 과제분석 결과
            weak_analysis: 취약 영역 분석 결과

        Returns:
            {
                "learning_objective": str,
                "task_analysis_summary": str,
                "performance_areas": [...],
                "general_tips": [...]
            }
        """
        prompt = COACHING_DB_GENERATION_PROMPT.format(
            learning_objective=learning_objective,
            task_analysis=task_analysis[:3000],
            weak_analysis=json.dumps(weak_analysis, ensure_ascii=False, indent=2)
        )

        try:
            result = self.gpt.generate_json(prompt)
            if 'learning_objective' not in result:
                result['learning_objective'] = learning_objective
            if 'task_analysis_summary' not in result:
                result['task_analysis_summary'] = task_analysis[:500]
            if 'performance_areas' not in result:
                result['performance_areas'] = []
            if 'general_tips' not in result:
                result['general_tips'] = []
            return result
        except Exception as e:
            print(f"  Warning: Failed to generate coaching DB: {e}")
            return {
                "learning_objective": learning_objective,
                "task_analysis_summary": task_analysis[:500],
                "performance_areas": [],
                "general_tips": ["Review the problem step by step"]
            }

    def generate_modeling_response(
        self,
        problem_text: str,
        ground_truth: str,
        task_analysis: str
    ) -> str:
        """
        Phase 3: Modeling - 교사가 추론 명료화(articulate reasoning) 제공

        올바른 풀이과정의 정석을 학생에게 보여줌

        Args:
            problem_text: 문제
            ground_truth: 정답
            task_analysis: 과제분석 결과

        Returns:
            Teacher의 모범 풀이 응답
        """
        prompt = MODELING_PROMPT.format(
            problem_text=problem_text,
            ground_truth=ground_truth,
            task_analysis=task_analysis[:2000]
        )

        try:
            response = self.gpt.generate(prompt)
            return response
        except Exception as e:
            print(f"  Warning: Failed to generate modeling response: {e}")
            # Fallback: return a basic structured response
            return f"""Problem-solving strategy and flow:
- Strategy selection: Apply systematic problem-solving approach
- Step-by-step reasoning:
  * Step 1: Understand the problem
  * Step 2: Apply appropriate method
- Key insights: Follow the task analysis structure

Answer: {ground_truth}
"""

    # =========================================================================
    # Phase 1: Iterative Scaffolding Methods (NEW)
    # =========================================================================

    def generate_initial_hint(
        self,
        problem_text: str,
        task_analysis: str,
        ground_truth: str
    ) -> str:
        """
        Phase 1 Iterative: 첫 번째 힌트 생성

        학생이 문제를 풀기 전에 방향을 제시하는 힌트.
        답을 알려주지 않으면서 올바른 접근법을 안내.

        Args:
            problem_text: 문제 텍스트
            task_analysis: 과제 분석 결과
            ground_truth: 정답 (교사 참고용)

        Returns:
            첫 번째 힌트 텍스트
        """
        prompt = INITIAL_HINT_PROMPT.format(
            problem_text=problem_text,
            task_analysis=task_analysis[:2000],
            ground_truth=ground_truth
        )

        try:
            response = self.gpt.generate(prompt)
            return response
        except Exception as e:
            print(f"  Warning: Failed to generate initial hint: {e}")
            return "Let's start by carefully reading the problem and identifying what information is given and what we need to find."

    def generate_progressive_hint(
        self,
        problem_text: str,
        task_analysis: str,
        conversation_history: List[Dict],
        last_response: str,
        iteration_number: int,
        ground_truth: str,
        max_iterations: int = 5
    ) -> str:
        """
        Phase 1 Iterative: 점진적 힌트 생성

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
            점진적 힌트 텍스트
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
            response = self.gpt.generate(prompt)
            return response
        except Exception as e:
            print(f"  Warning: Failed to generate progressive hint: {e}")
            return f"Look at your previous answer. There seems to be an error. Let me help you: Review step by step and check if the calculation is correct."

    def summarize_and_reconstruct(
        self,
        problem_text: str,
        ground_truth: str,
        task_analysis: str,
        conversation_history: List[Dict]
    ) -> Dict[str, Any]:
        """
        Phase 1 Iterative: 5회 실패 후 요약 및 정답 재구성

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
        formatted_history = self._format_conversation_history(conversation_history)

        prompt = SUMMARY_RECONSTRUCTION_PROMPT.format(
            problem_text=problem_text,
            ground_truth=ground_truth,
            task_analysis=task_analysis[:1500],
            conversation_history=formatted_history
        )

        try:
            result = self.gpt.generate_json(prompt)
            # Ensure required fields exist
            if 'summary' not in result:
                result['summary'] = "Student struggled with this problem after multiple attempts."
            if 'student_weaknesses' not in result:
                result['student_weaknesses'] = ["Could not identify the correct approach"]
            if 'reconstructed_response' not in result:
                result['reconstructed_response'] = f"Let me solve this correctly.\n\nAnswer: {ground_truth}"
            if 'learning_points' not in result:
                result['learning_points'] = ["Review the fundamental concepts"]
            return result
        except Exception as e:
            print(f"  Warning: Failed to summarize and reconstruct: {e}")
            return {
                "summary": "Student needed multiple attempts but could not solve the problem.",
                "student_weaknesses": ["Fundamental understanding of the problem"],
                "reconstructed_response": f"""[Understanding the problem]
Let me break down this problem carefully.

[Step-by-step solution]
Following the correct approach:

Answer: {ground_truth}""",
                "learning_points": ["Review the problem-solving approach", "Practice similar problems"]
            }

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


if __name__ == "__main__":
    # 테스트
    teacher = TeacherModel()

    # 샘플 데이터
    problem = "Calculate 15 + 27"
    student_resp = "15 + 27 = 41"  # Wrong answer
    ground_truth = "42"

    performance_objectives = [
        {
            "target": "Addition",
            "Behavior": "Correctly add two numbers",
            "Condition": "Given two integers",
            "Criterion": "Produce the correct sum"
        }
    ]

    print("=== Performance Objective Scoring ===")
    scores = teacher.score_by_performance_objectives(
        student_response=student_resp,
        performance_objectives=performance_objectives,
        ground_truth=ground_truth
    )
    print(json.dumps(scores, indent=2, ensure_ascii=False))

    print("\n=== Modeling Response ===")
    modeling = teacher.generate_modeling_response(
        problem_text=problem,
        ground_truth=ground_truth,
        task_analysis="Terminal Goal: Perform basic arithmetic correctly"
    )
    print(modeling)
