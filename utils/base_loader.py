"""
Base Dataset Loader - Abstract interface for all dataset loaders
Supports multiple answer types: MCQ, Numeric, LaTeX, Text, Boolean
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional


class AnswerType(Enum):
    """Supported answer types across different datasets"""
    MCQ = "mcq"           # Multiple choice: A/B/C/D
    NUMERIC = "numeric"   # Integer or decimal (GSM8K, SVAMP, SciBench)
    LATEX = "latex"       # LaTeX expression (MATH)
    TEXT = "text"         # Free-form text (BBH)
    BOOLEAN = "boolean"   # Yes/No, True/False (BBH)


@dataclass
class QuestionData:
    """
    Unified question data structure for all datasets.

    This dataclass provides a consistent interface for questions
    regardless of the source dataset.
    """
    dataset: str                           # Dataset name (e.g., "gsm8k", "mmlu")
    question_id: str                       # Unique identifier within dataset
    question: str                          # Question text
    answer_type: AnswerType               # Type of expected answer
    ground_truth: Any                      # Raw ground truth value
    ground_truth_formatted: str            # Human-readable format for teacher model
    choices: Optional[List[str]] = None    # Only for MCQ type
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)  # Dataset-specific metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "dataset": self.dataset,
            "question_id": self.question_id,
            "question": self.question,
            "answer_type": self.answer_type.value,
            "ground_truth": self.ground_truth,
            "ground_truth_formatted": self.ground_truth_formatted,
            "choices": self.choices,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuestionData":
        """Create from dictionary"""
        return cls(
            dataset=data["dataset"],
            question_id=data["question_id"],
            question=data["question"],
            answer_type=AnswerType(data["answer_type"]),
            ground_truth=data["ground_truth"],
            ground_truth_formatted=data["ground_truth_formatted"],
            choices=data.get("choices"),
            metadata=data.get("metadata", {})
        )


class BaseDatasetLoader(ABC):
    """
    Abstract base class for all dataset loaders.

    Each dataset loader must implement:
    - dataset_name: HuggingFace dataset identifier
    - answer_type: Primary answer type for this dataset
    - domain_category: 'math_logic' or 'science_knowledge'
    - load_data(): Load data from HuggingFace
    - format_question_as_prompt(): Convert question to LLM prompt
    - format_ground_truth(): Format ground truth for teacher evaluation
    """

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """HuggingFace dataset identifier (e.g., 'openai/gsm8k')"""
        pass

    @property
    @abstractmethod
    def answer_type(self) -> AnswerType:
        """Primary answer type for this dataset"""
        pass

    @property
    @abstractmethod
    def domain_category(self) -> str:
        """Domain category: 'math_logic' or 'science_knowledge'"""
        pass

    @abstractmethod
    def load_data(
        self,
        split: str = "train",
        subset: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[QuestionData]:
        """
        Load dataset from HuggingFace.

        Args:
            split: Data split ('train', 'validation', 'test')
            subset: Dataset subset (e.g., MMLU subject, BBH subtask)
            limit: Maximum number of questions to load

        Returns:
            List of QuestionData objects
        """
        pass

    @abstractmethod
    def format_question_as_prompt(self, question: QuestionData) -> str:
        """
        Format question for LLM input.

        Args:
            question: QuestionData object

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def format_ground_truth(self, question: QuestionData) -> str:
        """
        Format ground truth for teacher model evaluation.

        Args:
            question: QuestionData object

        Returns:
            Human-readable ground truth string
        """
        pass

    def get_learning_objective(self, subset: Optional[str] = None) -> str:
        """
        Generate learning objective for this dataset.

        Args:
            subset: Optional subset name for more specific objective

        Returns:
            Learning objective string
        """
        return f"Solve {self.dataset_name} problems with high accuracy"

    def get_available_subsets(self) -> Optional[List[str]]:
        """
        Return list of available subsets for this dataset.

        Returns:
            List of subset names, or None if not applicable
        """
        return None
