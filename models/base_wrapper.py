"""
Base class for model wrappers.

Provides common functionality shared across Teacher, Student, and other model wrappers.
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class BaseModelWrapper(ABC):
    """
    Abstract base class for all model wrappers.

    모든 모델 래퍼(Teacher, Student)의 공통 인터페이스 정의.
    Subclasses must implement the generate() method.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate text from model.

        Must be implemented by subclasses.

        Args:
            prompt: User prompt
            system_message: System message
            chat_history: Conversation history (for multi-turn)
            response_format: Response format (e.g., {"type": "json_object"}, API only)

        Returns:
            Generated text
        """
        pass

    def generate_json(
        self,
        prompt: str,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate JSON response.

        Default implementation raises NotImplementedError.
        Subclasses should override this method.

        Args:
            prompt: User prompt
            system_message: System message

        Returns:
            Parsed JSON dictionary
        """
        raise NotImplementedError("Subclass must implement generate_json")
