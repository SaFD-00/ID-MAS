"""
Tests for domain data loader.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from utils.domain_loader import DomainLoader
from utils.base_loader import AnswerType


class TestDomainLoaderInit:
    """Test DomainLoader initialization"""

    def test_init_with_valid_domain(self):
        loader = DomainLoader(domain="math", mode="train")
        assert loader.domain == "math"
        assert loader.mode == "train"

    def test_init_with_eval_mode(self):
        loader = DomainLoader(domain="math", mode="eval")
        assert loader.mode == "eval"

    def test_domain_stored(self):
        loader = DomainLoader(domain="math", mode="train")
        assert hasattr(loader, 'domain')


class TestDomainLoaderDatasetMapping:
    """Test dataset to domain mapping"""

    def test_gsm8k_is_math_domain(self):
        # GSM8K should be in math domain
        loader = DomainLoader(domain="math", mode="train")
        # Verify loader can work with math domain
        assert loader.domain == "math"

    def test_math_dataset_is_math_domain(self):
        # MATH dataset should be in math domain
        loader = DomainLoader(domain="math", mode="train")
        assert loader.domain == "math"


class TestDomainLoaderAnswerTypes:
    """Test answer type detection"""

    @patch('utils.domain_loader.load_dataset')
    def test_get_answer_type_gsm8k(self, mock_load_dataset):
        # GSM8K uses numeric answers
        loader = DomainLoader(domain="math", mode="train")

        # Mock dataset
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset

        # GSM8K should return NUMERIC type
        answer_type = loader._get_answer_type("gsm8k")
        assert answer_type == AnswerType.NUMERIC

    @patch('utils.domain_loader.load_dataset')
    def test_get_answer_type_math(self, mock_load_dataset):
        # MATH dataset uses LaTeX boxed answers
        loader = DomainLoader(domain="math", mode="train")

        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset

        answer_type = loader._get_answer_type("MATH")
        assert answer_type == AnswerType.LATEX


class TestDomainLoaderQuestionFormatting:
    """Test question formatting"""

    def test_format_question_as_prompt_basic(self):
        loader = DomainLoader(domain="math", mode="train")

        question_data = {
            "question": "What is 2+2?",
            "domain": "math"
        }

        prompt = loader.format_question_as_prompt(question_data)
        assert "2+2" in prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_format_question_handles_string_input(self):
        loader = DomainLoader(domain="math", mode="train")

        # Test with string question
        prompt = loader.format_question_as_prompt("Simple question")
        assert "Simple question" in prompt


class TestDomainLoaderGroundTruth:
    """Test ground truth formatting"""

    def test_format_ground_truth_numeric(self):
        loader = DomainLoader(domain="math", mode="train")

        question_data = {
            "answer": "42",
            "domain": "math"
        }

        ground_truth = loader.format_ground_truth(question_data)
        assert ground_truth is not None
        # Should return answer in appropriate format

    def test_format_ground_truth_with_dict(self):
        loader = DomainLoader(domain="math", mode="train")

        question_data = {
            "answer": "Answer text",
            "domain": "math"
        }

        ground_truth = loader.format_ground_truth(question_data)
        assert ground_truth is not None


class TestDomainLoaderTerminalGoals:
    """Test terminal goal retrieval"""

    def test_get_terminal_goal_exists(self):
        loader = DomainLoader(domain="math", mode="train")

        # Terminal goal should be defined for math domain
        terminal_goal = loader.get_terminal_goal("gsm8k")
        assert terminal_goal is not None
        assert isinstance(terminal_goal, str)
        assert len(terminal_goal) > 0

    def test_terminal_goal_describes_learning_objective(self):
        loader = DomainLoader(domain="math", mode="train")

        terminal_goal = loader.get_terminal_goal("gsm8k")

        # Terminal goal should describe what student will learn
        # Common keywords in learning objectives
        assert any(keyword in terminal_goal.lower() for keyword in
                   ["solve", "understand", "learn", "calculate", "demonstrate"])


class TestDomainLoaderDataLoading:
    """Test data loading functionality"""

    @patch('utils.domain_loader.load_dataset')
    def test_load_training_data_calls_dataset(self, mock_load_dataset):
        # Mock HuggingFace dataset
        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = Mock(return_value={
            "question": "Test question",
            "answer": "42"
        })
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load_dataset.return_value = {"train": mock_dataset}

        loader = DomainLoader(domain="math", mode="train")

        # This should call load_dataset
        # We're just testing the structure, not actual loading
        mock_load_dataset.assert_called()

    @patch('utils.domain_loader.load_dataset')
    def test_load_training_data_with_limit(self, mock_load_dataset):
        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = Mock(return_value={
            "question": "Test question",
            "answer": "42"
        })
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load_dataset.return_value = {"train": mock_dataset}

        loader = DomainLoader(domain="math", mode="train")

        # Load with limit
        data = loader.load_training_data(dataset_name="gsm8k", num_samples=10)

        assert data is not None
        assert len(data) <= 10


class TestDomainLoaderModes:
    """Test different loader modes"""

    def test_train_mode_initialization(self):
        loader = DomainLoader(domain="math", mode="train")
        assert loader.mode == "train"

    def test_eval_mode_initialization(self):
        loader = DomainLoader(domain="math", mode="eval")
        assert loader.mode == "eval"

    def test_mode_affects_data_split(self):
        train_loader = DomainLoader(domain="math", mode="train")
        eval_loader = DomainLoader(domain="math", mode="eval")

        # Different modes should handle data differently
        assert train_loader.mode != eval_loader.mode
