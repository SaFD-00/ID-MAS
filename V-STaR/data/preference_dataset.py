"""Preference Dataset for V-STaR DPO Training"""

from typing import List, Dict, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, field
from itertools import product
import json
import random
from pathlib import Path

from torch.utils.data import Dataset


@dataclass
class PreferencePair:
    """A single preference pair for DPO training"""
    question_id: str
    prompt: str
    chosen: str      # Correct solution (y+)
    rejected: str    # Incorrect solution (y-)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferencePair":
        return cls(
            question_id=data["question_id"],
            prompt=data["prompt"],
            chosen=data["chosen"],
            rejected=data["rejected"],
            metadata=data.get("metadata", {}),
        )


def create_preference_pairs(
    correct_solutions: List[Dict[str, Any]],
    incorrect_solutions: List[Dict[str, Any]],
    max_pairs_per_question: Optional[int] = None,
    shuffle: bool = True,
) -> List[PreferencePair]:
    """
    Create preference pairs from correct and incorrect solutions

    Uses Cartesian product: all combinations of (correct, incorrect)

    Args:
        correct_solutions: List of correct solution dicts
        incorrect_solutions: List of incorrect solution dicts
        max_pairs_per_question: Max pairs per question (None for all)
        shuffle: Whether to shuffle the pairs

    Returns:
        List of PreferencePair objects
    """
    # Group solutions by question_id
    correct_by_qid: Dict[str, List[Dict]] = {}
    incorrect_by_qid: Dict[str, List[Dict]] = {}

    for sol in correct_solutions:
        qid = sol["question_id"]
        if qid not in correct_by_qid:
            correct_by_qid[qid] = []
        correct_by_qid[qid].append(sol)

    for sol in incorrect_solutions:
        qid = sol["question_id"]
        if qid not in incorrect_by_qid:
            incorrect_by_qid[qid] = []
        incorrect_by_qid[qid].append(sol)

    # Create pairs using Cartesian product
    all_pairs = []

    for qid in correct_by_qid:
        if qid not in incorrect_by_qid:
            continue

        correct_sols = correct_by_qid[qid]
        incorrect_sols = incorrect_by_qid[qid]

        # Generate all combinations
        question_pairs = []
        for correct, incorrect in product(correct_sols, incorrect_sols):
            pair = PreferencePair(
                question_id=qid,
                prompt=correct.get("prompt", correct.get("question", "")),
                chosen=correct["solution"],
                rejected=incorrect["solution"],
                metadata={
                    "chosen_answer": correct.get("extracted_answer"),
                    "rejected_answer": incorrect.get("extracted_answer"),
                    "ground_truth": correct.get("ground_truth"),
                },
            )
            question_pairs.append(pair)

        # Limit pairs per question if specified
        if max_pairs_per_question and len(question_pairs) > max_pairs_per_question:
            if shuffle:
                random.shuffle(question_pairs)
            question_pairs = question_pairs[:max_pairs_per_question]

        all_pairs.extend(question_pairs)

    if shuffle:
        random.shuffle(all_pairs)

    return all_pairs


def create_preference_pairs_from_samples(
    all_solutions: List[Dict[str, Any]],
    max_pairs_per_question: Optional[int] = None,
    shuffle: bool = True,
) -> List[PreferencePair]:
    """
    Create preference pairs from a list of all solutions (labeled)

    Args:
        all_solutions: List of solution dicts with 'is_correct' field
        max_pairs_per_question: Max pairs per question
        shuffle: Whether to shuffle

    Returns:
        List of PreferencePair objects
    """
    correct = [s for s in all_solutions if s.get("is_correct", False)]
    incorrect = [s for s in all_solutions if not s.get("is_correct", False)]

    return create_preference_pairs(
        correct_solutions=correct,
        incorrect_solutions=incorrect,
        max_pairs_per_question=max_pairs_per_question,
        shuffle=shuffle,
    )


class PreferenceDataset(Dataset):
    """
    PyTorch Dataset for DPO training

    Compatible with trl DPOTrainer format
    """

    def __init__(
        self,
        pairs: List[PreferencePair],
        tokenizer=None,
        max_length: int = 2048,
    ):
        """
        Initialize dataset

        Args:
            pairs: List of preference pairs
            tokenizer: Optional tokenizer (for pre-tokenization)
            max_length: Maximum sequence length
        """
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item in trl DPOTrainer format

        Returns dict with:
            - prompt: The prompt/question
            - chosen: The preferred (correct) response
            - rejected: The dispreferred (incorrect) response
        """
        pair = self.pairs[idx]

        item = {
            "prompt": pair.prompt,
            "chosen": pair.chosen,
            "rejected": pair.rejected,
        }

        return item

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert to list of dicts (for HuggingFace datasets)"""
        return [self[i] for i in range(len(self))]

    @classmethod
    def from_solutions(
        cls,
        all_solutions: List[Dict[str, Any]],
        tokenizer=None,
        max_length: int = 2048,
        max_pairs_per_question: Optional[int] = None,
        shuffle: bool = True,
    ) -> "PreferenceDataset":
        """
        Create dataset from list of labeled solutions

        Args:
            all_solutions: List of solutions with 'is_correct' labels
            tokenizer: Optional tokenizer
            max_length: Max sequence length
            max_pairs_per_question: Max pairs per question
            shuffle: Whether to shuffle

        Returns:
            PreferenceDataset instance
        """
        pairs = create_preference_pairs_from_samples(
            all_solutions=all_solutions,
            max_pairs_per_question=max_pairs_per_question,
            shuffle=shuffle,
        )

        return cls(
            pairs=pairs,
            tokenizer=tokenizer,
            max_length=max_length,
        )

    def save(self, path: str) -> None:
        """Save dataset to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [pair.to_dict() for pair in self.pairs]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(self.pairs)} preference pairs to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        tokenizer=None,
        max_length: int = 2048,
    ) -> "PreferenceDataset":
        """Load dataset from JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pairs = [PreferencePair.from_dict(d) for d in data]

        return cls(
            pairs=pairs,
            tokenizer=tokenizer,
            max_length=max_length,
        )


class DGenDataset(Dataset):
    """
    D_GEN Dataset for SFT training

    Contains only correct solutions
    """

    def __init__(
        self,
        solutions: List[Dict[str, Any]],
        tokenizer=None,
        max_length: int = 2048,
    ):
        """
        Initialize D_GEN dataset

        Args:
            solutions: List of correct solution dicts
            tokenizer: Optional tokenizer
            max_length: Max sequence length
        """
        self.solutions = solutions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.solutions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item for SFT training

        Returns dict with:
            - prompt: The prompt/question
            - completion: The correct solution
        """
        sol = self.solutions[idx]

        return {
            "prompt": sol.get("prompt", sol.get("question", "")),
            "completion": sol["solution"],
            "text": f"{sol.get('prompt', sol.get('question', ''))}\n\n{sol['solution']}",
        }

    @classmethod
    def from_sampling_results(
        cls,
        correct_solutions: List[Dict[str, Any]],
        tokenizer=None,
        max_length: int = 2048,
    ) -> "DGenDataset":
        """Create from correct solutions only"""
        return cls(
            solutions=correct_solutions,
            tokenizer=tokenizer,
            max_length=max_length,
        )

    def save(self, path: str) -> None:
        """Save dataset to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.solutions, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(self.solutions)} correct solutions to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        tokenizer=None,
        max_length: int = 2048,
    ) -> "DGenDataset":
        """Load from JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            solutions = json.load(f)

        return cls(
            solutions=solutions,
            tokenizer=tokenizer,
            max_length=max_length,
        )


def get_statistics(
    correct_solutions: List[Dict],
    incorrect_solutions: List[Dict],
    preference_pairs: List[PreferencePair],
) -> Dict[str, Any]:
    """Get statistics about the preference dataset"""

    # Count unique questions
    correct_qids = set(s["question_id"] for s in correct_solutions)
    incorrect_qids = set(s["question_id"] for s in incorrect_solutions)
    pair_qids = set(p.question_id for p in preference_pairs)

    return {
        "num_correct_solutions": len(correct_solutions),
        "num_incorrect_solutions": len(incorrect_solutions),
        "num_preference_pairs": len(preference_pairs),
        "num_questions_with_correct": len(correct_qids),
        "num_questions_with_incorrect": len(incorrect_qids),
        "num_questions_with_pairs": len(pair_qids),
        "avg_correct_per_question": len(correct_solutions) / len(correct_qids) if correct_qids else 0,
        "avg_incorrect_per_question": len(incorrect_solutions) / len(incorrect_qids) if incorrect_qids else 0,
        "avg_pairs_per_question": len(preference_pairs) / len(pair_qids) if pair_qids else 0,
    }
