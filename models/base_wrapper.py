"""
Base class for model wrappers.

Provides common functionality shared across LLM, Student, and other model wrappers.
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict


class BaseModelWrapper(ABC):
    """
    Abstract base class for all model wrappers.

    Subclasses must implement the generate() method.
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
