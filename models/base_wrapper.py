"""
Base class for model wrappers.

Provides common functionality shared across GPT, Student, and other model wrappers.
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict


class BaseModelWrapper(ABC):
    """
    Abstract base class for all model wrappers.

    Subclasses must implement the generate() method.
    Provides shared generate_with_reflection() for all student models.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate text from model.

        Must be implemented by subclasses.

        Args:
            prompt: User prompt
            system_message: System message
            chat_history: Conversation history

        Returns:
            Generated text
        """
        pass

    def generate_with_reflection(
        self,
        prompt: str,
        teacher_feedback: Dict,
        reflection_result: Dict,
        system_message: Optional[str] = None
    ) -> str:
        """
        성찰 결과를 반영하여 텍스트 생성 (ReAct 스타일)

        Common implementation shared across all student models.

        Args:
            prompt: 원본 문제
            teacher_feedback: 교사 피드백
            reflection_result: 성찰 결과
            system_message: 시스템 메시지

        Returns:
            성찰 기반 생성 텍스트
        """
        # 성찰 내용을 프롬프트에 통합
        reflection_prompt = f"""
Original Problem:
{prompt}

Your Previous Reflection:
- Recognized Strengths: {reflection_result.get('recognized_strengths', [])}
- Recognized Weaknesses: {reflection_result.get('recognized_weaknesses', [])}
- Planned Reasoning Strategy: {reflection_result.get('planned_reasoning_strategy', [])}

Teacher's Suggested Next Actions:
{teacher_feedback.get('next_iteration_reasoning_actions', [])}

Now, generate a new response following your planned reasoning strategy and addressing the unsatisfied criteria.
"""

        return self.generate(reflection_prompt, system_message)
