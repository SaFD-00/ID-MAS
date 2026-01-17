"""Tests for SFTDataLoader (T2 verification)"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from data.loader import (
    SFTSample,
    SFTDataLoader,
    DataLoader,
    QuestionData,
    get_sft_data_loader,
)
from config.domains import AnswerType


class TestSFTSample:
    """Test SFTSample dataclass"""

    def test_sample_creation(self):
        """Test SFTSample creation"""
        sample = SFTSample(
            question_id="q1",
            question="What is 2+2?",
            response="The answer is 4.",
            is_correct=True,
        )
        assert sample.question_id == "q1"
        assert sample.is_correct == True

    def test_sample_to_dict(self):
        """Test SFTSample serialization"""
        sample = SFTSample(
            question_id="q1",
            question="test",
            response="answer",
        )
        d = sample.to_dict()
        assert d["question_id"] == "q1"
        assert d["question"] == "test"
        assert d["response"] == "answer"
        assert d["is_correct"] == True  # Default

    def test_sample_from_dict(self):
        """Test SFTSample deserialization"""
        data = {
            "question_id": "q2",
            "question": "test2",
            "response": "answer2",
            "is_correct": False,
            "metadata": {"source": "test"},
        }
        sample = SFTSample.from_dict(data)
        assert sample.question_id == "q2"
        assert sample.is_correct == False
        assert sample.metadata["source"] == "test"


class TestSFTDataLoader:
    """Test SFTDataLoader for D_SFT loading"""

    def test_loader_initialization(self):
        """Test loader initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = SFTDataLoader(data_dir=Path(tmpdir))
            assert loader.data_dir == Path(tmpdir)

    def test_load_sft_data_standard_format(self):
        """AC6: Test loading D_SFT in standard format (question/response)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            data_dir = Path(tmpdir)
            dataset_dir = data_dir / "gsm8k" / "train"
            dataset_dir.mkdir(parents=True)

            sft_data = [
                {
                    "question_id": "q1",
                    "question": "What is 2+2?",
                    "response": "Let me calculate. 2+2=4. The answer is 4.",
                },
                {
                    "question_id": "q2",
                    "question": "What is 3*3?",
                    "response": "3 times 3 equals 9. The answer is 9.",
                },
            ]

            sft_file = dataset_dir / "sft_data.json"
            with open(sft_file, "w") as f:
                json.dump(sft_data, f)

            # Load
            loader = SFTDataLoader(data_dir=data_dir)
            samples = loader.load_sft_data("gsm8k", split="train")

            assert len(samples) == 2
            assert samples[0].question_id == "q1"
            assert samples[0].is_correct == True

    def test_load_sft_data_vstar_format(self):
        """AC6: Test loading D_SFT in V-STaR format (input/output)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            dataset_dir = data_dir / "gsm8k" / "train"
            dataset_dir.mkdir(parents=True)

            vstar_data = [
                {
                    "input": "Calculate 5+5",
                    "output": "The answer is \\boxed{10}",
                    "metadata": {"id": "test_1"},
                },
            ]

            sft_file = dataset_dir / "gsm8k.json"
            with open(sft_file, "w") as f:
                json.dump(vstar_data, f)

            loader = SFTDataLoader(data_dir=data_dir)
            samples = loader.load_sft_data("gsm8k", split="train")

            assert len(samples) == 1
            assert samples[0].question == "Calculate 5+5"
            assert samples[0].response == "The answer is \\boxed{10}"
            assert samples[0].question_id == "test_1"

    def test_load_sft_data_file_not_found(self):
        """Test error handling when file not found"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = SFTDataLoader(data_dir=Path(tmpdir))

            with pytest.raises(FileNotFoundError):
                loader.load_sft_data("nonexistent_dataset")

    def test_find_sft_file_search_order(self):
        """Test SFT file search order"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create multiple potential files
            for pattern in [
                "gsm8k/train/sft_data.json",
                "gsm8k/train/gsm8k.json",
                "math/train/gsm8k.json",
            ]:
                file_path = data_dir / pattern
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "w") as f:
                    json.dump([{"input": f"from {pattern}", "output": "test"}], f)

            loader = SFTDataLoader(data_dir=data_dir)

            # Should find sft_data.json first
            found = loader._find_sft_file("gsm8k", "train")
            assert found is not None
            assert "sft_data.json" in str(found)

    def test_convert_questions_to_sft(self):
        """Test converting QuestionData to SFTSample"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = SFTDataLoader(data_dir=Path(tmpdir))

            questions = [
                QuestionData(
                    dataset="test",
                    question_id="q1",
                    question="Test question",
                    answer_type=AnswerType.NUMERIC,
                    ground_truth=42,
                    ground_truth_formatted="42",
                    metadata={"output": "The answer is 42"},
                ),
            ]

            samples = loader.convert_questions_to_sft(questions)

            assert len(samples) == 1
            assert samples[0].question_id == "q1"
            assert samples[0].response == "The answer is 42"

    def test_save_sft_data(self):
        """Test saving SFT data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = SFTDataLoader(data_dir=Path(tmpdir))

            samples = [
                SFTSample(
                    question_id="q1",
                    question="test",
                    response="answer",
                ),
            ]

            save_path = Path(tmpdir) / "saved_sft.json"
            loader.save_sft_data(samples, save_path)

            assert save_path.exists()

            with open(save_path, "r") as f:
                loaded = json.load(f)

            assert len(loaded) == 1
            assert loaded[0]["question_id"] == "q1"


class TestGetSFTDataLoader:
    """Test factory function"""

    def test_factory_creates_loader(self):
        """get_sft_data_loader should return SFTDataLoader"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = get_sft_data_loader(data_dir=tmpdir)
            assert isinstance(loader, SFTDataLoader)

    def test_factory_with_none_uses_default(self):
        """get_sft_data_loader with None should use default data_dir"""
        loader = get_sft_data_loader(data_dir=None)
        assert isinstance(loader, SFTDataLoader)
