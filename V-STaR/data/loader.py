"""Data loader for V-STaR"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator
from enum import Enum

from config.domains import (
    AnswerType,
    get_dataset_config,
    get_data_path,
    get_answer_type,
)
from config.paths import get_data_dir


@dataclass
class QuestionData:
    """Data structure for a single question"""
    dataset: str
    question_id: str
    question: str
    answer_type: AnswerType
    ground_truth: Any
    ground_truth_formatted: str
    choices: Optional[List[str]] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "dataset": self.dataset,
            "question_id": self.question_id,
            "question": self.question,
            "answer_type": self.answer_type.value,
            "ground_truth": self.ground_truth,
            "ground_truth_formatted": self.ground_truth_formatted,
            "choices": self.choices,
            "context": self.context,
            "metadata": self.metadata,
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
            context=data.get("context"),
            metadata=data.get("metadata", {}),
        )


class DataLoader:
    """
    Data loader for V-STaR datasets

    Supports loading data from JSON files in the V-STaR/data directory.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize data loader

        Args:
            data_dir: Base data directory. If None, auto-detect.
        """
        self.data_dir = Path(data_dir) if data_dir else get_data_dir()

    def load_dataset(
        self,
        dataset: str,
        split: str = "train",
        limit: Optional[int] = None
    ) -> List[QuestionData]:
        """
        Load a dataset

        Args:
            dataset: Dataset name (e.g., "gsm8k", "reclor")
            split: "train" or "test"
            limit: Optional limit on number of samples

        Returns:
            List of QuestionData objects
        """
        data_path = get_data_path(self.data_dir, dataset, split)

        if data_path is None:
            raise ValueError(f"No {split} data available for dataset: {dataset}")

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load JSON data
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Convert to QuestionData objects
        questions = []
        answer_type = get_answer_type(dataset)

        for idx, item in enumerate(raw_data):
            if limit and idx >= limit:
                break

            question_data = self._parse_item(item, dataset, answer_type, idx)
            questions.append(question_data)

        return questions

    def _parse_item(
        self,
        item: Dict[str, Any],
        dataset: str,
        answer_type: AnswerType,
        idx: int
    ) -> QuestionData:
        """Parse a single data item into QuestionData"""

        # Extract question ID
        question_id = item.get("metadata", {}).get("id", f"{dataset}_{idx}")
        if isinstance(question_id, dict):
            question_id = str(question_id.get("id", idx))

        # Extract question text
        question = item.get("input", item.get("question", ""))

        # Extract ground truth from output
        output = item.get("output", "")
        ground_truth = self._extract_ground_truth(output, answer_type)
        ground_truth_formatted = self._format_ground_truth(ground_truth, answer_type)

        # Extract choices for MCQ
        choices = self._extract_choices(item, dataset)

        # Extract context (for ReClor, etc.)
        context = self._extract_context(item, dataset)

        # Metadata
        metadata = item.get("metadata", {})
        if "instruction" in item:
            metadata["instruction"] = item["instruction"]

        return QuestionData(
            dataset=dataset,
            question_id=str(question_id),
            question=question,
            answer_type=answer_type,
            ground_truth=ground_truth,
            ground_truth_formatted=ground_truth_formatted,
            choices=choices,
            context=context,
            metadata=metadata,
        )

    def _extract_ground_truth(
        self,
        output: str,
        answer_type: AnswerType
    ) -> Any:
        """Extract ground truth from output string"""
        import re

        # Try to extract from boxed format
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', output)
        if boxed_match:
            answer = boxed_match.group(1).strip()

            if answer_type == AnswerType.NUMERIC:
                # Parse numeric answer
                try:
                    # Remove commas and parse
                    clean = answer.replace(",", "").replace(" ", "")
                    if "." in clean:
                        return float(clean)
                    return int(clean)
                except ValueError:
                    return answer

            elif answer_type == AnswerType.MCQ:
                # Return letter (A, B, C, D)
                return answer.upper()

            elif answer_type == AnswerType.BOOLEAN:
                return answer.lower() in ["yes", "true", "1"]

            return answer

        # Fallback: return original output
        return output.strip()

    def _format_ground_truth(
        self,
        ground_truth: Any,
        answer_type: AnswerType
    ) -> str:
        """Format ground truth for comparison"""
        if answer_type == AnswerType.NUMERIC:
            if isinstance(ground_truth, float):
                # Remove trailing zeros
                if ground_truth == int(ground_truth):
                    return str(int(ground_truth))
                return str(ground_truth)
            return str(ground_truth)

        elif answer_type == AnswerType.MCQ:
            return str(ground_truth).upper()

        elif answer_type == AnswerType.BOOLEAN:
            return "Yes" if ground_truth else "No"

        return str(ground_truth)

    def _extract_choices(
        self,
        item: Dict[str, Any],
        dataset: str
    ) -> Optional[List[str]]:
        """Extract choices for MCQ datasets"""
        answer_type = get_answer_type(dataset)

        if answer_type not in [AnswerType.MCQ, AnswerType.BOOLEAN]:
            return None

        # Try to extract from input field
        input_text = item.get("input", "")

        # Parse choices from input (format: "A. choice\nB. choice\n...")
        import re
        choices = []
        pattern = r'([A-D])\.\s*([^\n]+)'
        matches = re.findall(pattern, input_text)

        if matches:
            choices = [m[1].strip() for m in matches]

        return choices if choices else None

    def _extract_context(
        self,
        item: Dict[str, Any],
        dataset: str
    ) -> Optional[str]:
        """Extract context for datasets like ReClor"""
        if dataset not in ["reclor"]:
            return None

        input_text = item.get("input", "")

        # For ReClor, context is before "Question:"
        if "Question:" in input_text:
            context = input_text.split("Question:")[0]
            # Remove "Context:" prefix if present
            if context.startswith("Context:"):
                context = context[8:]
            return context.strip()

        return None

    def load_multiple_datasets(
        self,
        datasets: List[str],
        split: str = "train",
        limit_per_dataset: Optional[int] = None
    ) -> List[QuestionData]:
        """
        Load multiple datasets

        Args:
            datasets: List of dataset names
            split: "train" or "test"
            limit_per_dataset: Optional limit per dataset

        Returns:
            Combined list of QuestionData
        """
        all_data = []
        for dataset in datasets:
            try:
                data = self.load_dataset(dataset, split, limit_per_dataset)
                all_data.extend(data)
            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Could not load {dataset}: {e}")

        return all_data

    def iter_dataset(
        self,
        dataset: str,
        split: str = "train",
        batch_size: int = 1
    ) -> Iterator[List[QuestionData]]:
        """
        Iterate over dataset in batches

        Args:
            dataset: Dataset name
            split: "train" or "test"
            batch_size: Batch size

        Yields:
            Batches of QuestionData
        """
        data = self.load_dataset(dataset, split)

        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def get_dataset_info(self, dataset: str) -> Dict[str, Any]:
        """Get information about a dataset"""
        config = get_dataset_config(dataset)

        info = {
            "name": dataset,
            "domain": config["domain"],
            "answer_type": config["answer_type"].value,
            "description": config.get("description", ""),
            "has_train": config.get("train_file") is not None,
            "has_test": config.get("test_file") is not None,
        }

        # Try to get counts
        for split in ["train", "test"]:
            try:
                data = self.load_dataset(dataset, split)
                info[f"{split}_count"] = len(data)
            except (FileNotFoundError, ValueError):
                info[f"{split}_count"] = 0

        return info


def load_json_data(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSON data from file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_data(data: List[Dict[str, Any]], file_path: Path) -> None:
    """Save data to JSON file"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@dataclass
class SFTSample:
    """SFT 학습용 샘플 (D_SFT의 각 요소)"""
    question_id: str
    question: str
    response: str
    is_correct: bool = True  # SFT 데이터는 기본적으로 correct
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "response": self.response,
            "is_correct": self.is_correct,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SFTSample":
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            response=data["response"],
            is_correct=data.get("is_correct", True),
            metadata=data.get("metadata", {}),
        )


class SFTDataLoader:
    """
    D_SFT 데이터 로더 (Algorithm 1의 D_SFT)

    V-STaR 논문에서 D_SFT는 초기 SFT 학습 데이터로,
    D_GEN 초기화에 사용됩니다.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize SFT data loader

        Args:
            data_dir: Base data directory. If None, auto-detect.
        """
        self.data_dir = Path(data_dir) if data_dir else get_data_dir()

    def load_sft_data(
        self,
        dataset: str,
        split: str = "train",
        sft_file: Optional[str] = None,
    ) -> List[SFTSample]:
        """
        D_SFT 데이터 로드

        Args:
            dataset: Dataset name (e.g., "gsm8k")
            split: "train" or "test"
            sft_file: Optional specific SFT file name

        Returns:
            List of SFTSample (D_SFT)
        """
        # SFT 파일 경로 탐색
        sft_path = self._find_sft_file(dataset, split, sft_file)

        if sft_path is None:
            raise FileNotFoundError(
                f"SFT data not found for {dataset}/{split}. "
                f"Searched in: {self.data_dir}"
            )

        return self._load_sft_file(sft_path, dataset)

    def _find_sft_file(
        self,
        dataset: str,
        split: str,
        sft_file: Optional[str] = None,
    ) -> Optional[Path]:
        """SFT 파일 경로 찾기"""
        # 1. 지정된 파일이 있으면 사용
        if sft_file:
            path = self.data_dir / sft_file
            if path.exists():
                return path

        # 2. 표준 경로 탐색
        search_patterns = [
            # V-STaR 표준 형식
            f"{dataset}/{split}/sft_data.json",
            f"{dataset}/{split}/{dataset}_sft.json",
            # 도메인별 형식
            f"math/{split}/{dataset}_sft.json",
            f"logical/{split}/{dataset}_sft.json",
            f"commonsense/{split}/{dataset}_sft.json",
            # 기존 학습 데이터 (정답 포함된 것)
            f"{dataset}/{split}/{dataset}.json",
            f"math/{split}/{dataset}.json",
            f"logical/{split}/{dataset}.json",
            f"commonsense/{split}/{dataset}.json",
        ]

        for pattern in search_patterns:
            path = self.data_dir / pattern
            if path.exists():
                return path

        return None

    def _load_sft_file(
        self,
        path: Path,
        dataset: str,
    ) -> List[SFTSample]:
        """SFT 파일 로드 및 파싱"""
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        samples = []
        for idx, item in enumerate(raw_data):
            sample = self._parse_sft_item(item, dataset, idx)
            if sample:
                samples.append(sample)

        return samples

    def _parse_sft_item(
        self,
        item: Dict[str, Any],
        dataset: str,
        idx: int,
    ) -> Optional[SFTSample]:
        """SFT 아이템 파싱"""
        # 표준 SFT 형식 (question, response)
        if "question" in item and "response" in item:
            return SFTSample(
                question_id=item.get("question_id", f"{dataset}_{idx}"),
                question=item["question"],
                response=item["response"],
                is_correct=item.get("is_correct", True),
                metadata=item.get("metadata", {}),
            )

        # V-STaR 학습 데이터 형식 (input, output)
        if "input" in item and "output" in item:
            question_id = item.get("metadata", {}).get("id", f"{dataset}_{idx}")
            if isinstance(question_id, dict):
                question_id = str(question_id.get("id", idx))

            return SFTSample(
                question_id=str(question_id),
                question=item["input"],
                response=item["output"],
                is_correct=True,  # 원본 데이터는 correct
                metadata=item.get("metadata", {}),
            )

        return None

    def convert_questions_to_sft(
        self,
        questions: List[QuestionData],
        prompt_fn: Optional[callable] = None,
    ) -> List[SFTSample]:
        """
        QuestionData를 SFTSample로 변환 (D_SFT 생성)

        Args:
            questions: List of QuestionData
            prompt_fn: Optional prompt function

        Returns:
            List of SFTSample
        """
        samples = []
        for q in questions:
            question_text = prompt_fn(q) if prompt_fn else q.question

            # ground_truth를 response로 사용
            # 실제로는 CoT 포함된 정답이 필요하지만, 여기서는 기본값 사용
            response = q.metadata.get("output", q.ground_truth_formatted)

            samples.append(SFTSample(
                question_id=q.question_id,
                question=question_text,
                response=response,
                is_correct=True,
                metadata={"dataset": q.dataset, "answer_type": q.answer_type.value},
            ))

        return samples

    def save_sft_data(
        self,
        samples: List[SFTSample],
        path: Path,
    ) -> None:
        """SFT 데이터 저장"""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [s.to_dict() for s in samples]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def get_sft_data_loader(data_dir: Optional[str] = None) -> SFTDataLoader:
    """Factory function for SFTDataLoader"""
    return SFTDataLoader(data_dir=Path(data_dir) if data_dir else None)
