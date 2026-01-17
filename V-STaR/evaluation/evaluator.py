"""V-STaR Evaluator"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

from tqdm import tqdm

from config.training import TrainingConfig
from config.domains import AnswerType
from data.loader import QuestionData
from models.generator import VSTaRGenerator
from models.verifier import VSTaRVerifier
from .answer_checker import AnswerChecker, grade_answer
from .metrics import (
    pass_at_k,
    best_of_k,
    best_of_k_from_scores,
    majority_voting,
    weighted_majority_voting,
    MetricsCalculator,
)


@dataclass
class EvaluationResult:
    """Result of evaluating a single question"""
    question_id: str
    solutions: List[str]
    extracted_answers: List[str]
    correctness: List[int]  # 0 or 1
    scores: List[float]
    ground_truth: str
    selected_answer: Optional[str] = None
    is_selected_correct: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "num_solutions": len(self.solutions),
            "num_correct": sum(self.correctness),
            "scores": self.scores,
            "correctness": self.correctness,
            "selected_answer": self.selected_answer,
            "is_selected_correct": self.is_selected_correct,
            "ground_truth": self.ground_truth,
        }


class VSTaREvaluator:
    """
    V-STaR Evaluator

    Evaluates generator and verifier performance using:
    - Pass@k: Probability of generating at least one correct solution
    - Best-of-k: Accuracy when selecting top solution by verifier score
    - Majority Voting: Self-consistency baseline
    """

    def __init__(
        self,
        generator: VSTaRGenerator,
        verifier: VSTaRVerifier,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize evaluator

        Args:
            generator: Generator model
            verifier: Verifier model
            config: Training configuration
        """
        self.generator = generator
        self.verifier = verifier
        self.config = config or TrainingConfig()

    def evaluate_question(
        self,
        question: QuestionData,
        prompt: str,
        num_samples: int = 64,
        temperature: float = 0.7,
    ) -> EvaluationResult:
        """
        Evaluate a single question

        Args:
            question: Question data
            prompt: Formatted prompt
            num_samples: Number of solutions to sample
            temperature: Sampling temperature

        Returns:
            EvaluationResult
        """
        # Generate solutions
        solutions = self.generator.generate(
            prompt,
            k=num_samples,
            temperature=temperature,
        )

        if not isinstance(solutions, list):
            solutions = [solutions]

        # Extract answers and grade
        extracted_answers = []
        correctness = []

        for sol in solutions:
            answer = AnswerChecker.extract_answer(sol, question.answer_type)
            extracted_answers.append(answer)

            is_correct = False
            if answer is not None:
                is_correct = grade_answer(
                    answer,
                    question.ground_truth,
                    question.answer_type
                )
            correctness.append(1 if is_correct else 0)

        # Score with verifier
        scores = []
        for sol in solutions:
            score = self.verifier.score(question.question, sol)
            scores.append(score)

        # Select best by verifier score
        best_idx = scores.index(max(scores))
        selected_answer = extracted_answers[best_idx]
        is_selected_correct = bool(correctness[best_idx])

        return EvaluationResult(
            question_id=question.question_id,
            solutions=solutions,
            extracted_answers=extracted_answers,
            correctness=correctness,
            scores=scores,
            ground_truth=question.ground_truth_formatted,
            selected_answer=selected_answer,
            is_selected_correct=is_selected_correct,
        )

    def evaluate(
        self,
        questions: List[QuestionData],
        prompt_fn: Callable[[QuestionData], str],
        num_samples: int = 64,
        k_values: Optional[List[int]] = None,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate on a dataset

        Args:
            questions: List of questions
            prompt_fn: Function to create prompts
            num_samples: Solutions per question
            k_values: K values for metrics
            show_progress: Show progress bar

        Returns:
            Dictionary of metrics
        """
        k_values = k_values or [1, 4, 8, 16, 32, 64]

        results = []
        metrics_calc = MetricsCalculator(k_values=k_values)

        iterator = questions
        if show_progress:
            iterator = tqdm(questions, desc="Evaluating")

        for question in iterator:
            prompt = prompt_fn(question)
            result = self.evaluate_question(
                question=question,
                prompt=prompt,
                num_samples=num_samples,
            )
            results.append(result)

            # Add to metrics calculator
            metrics_calc.add(
                solutions=result.solutions,
                scores=result.scores,
                correctness=result.correctness,
            )

        # Compute metrics
        metrics = metrics_calc.compute()

        # Add additional metrics
        metrics["accuracy_best_of_n"] = sum(
            r.is_selected_correct for r in results
        ) / len(results) if results else 0

        # Coverage (questions with at least one correct)
        metrics["coverage"] = sum(
            1 for r in results if any(r.correctness)
        ) / len(results) if results else 0

        # Average correct per question
        metrics["avg_correct_per_question"] = sum(
            sum(r.correctness) for r in results
        ) / len(results) if results else 0

        return {
            "metrics": metrics,
            "results": [r.to_dict() for r in results],
            "num_questions": len(questions),
            "num_samples_per_question": num_samples,
        }


def evaluate_generator_only(
    generator: VSTaRGenerator,
    questions: List[QuestionData],
    prompt_fn: Callable[[QuestionData], str],
    num_samples: int = 16,
    k_values: Optional[List[int]] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate generator without verifier (Pass@k only)

    Args:
        generator: Generator model
        questions: Questions to evaluate
        prompt_fn: Prompt formatting function
        num_samples: Solutions per question
        k_values: K values for Pass@k
        show_progress: Show progress

    Returns:
        Metrics dictionary
    """
    k_values = k_values or [1, 4, 8, 16]

    all_correctness = []

    iterator = questions
    if show_progress:
        iterator = tqdm(questions, desc="Evaluating generator")

    for question in iterator:
        prompt = prompt_fn(question)

        # Generate solutions
        solutions = generator.generate(prompt, k=num_samples)

        if not isinstance(solutions, list):
            solutions = [solutions]

        # Grade each solution
        correctness = []
        for sol in solutions:
            answer = AnswerChecker.extract_answer(sol, question.answer_type)
            is_correct = False
            if answer is not None:
                is_correct = grade_answer(
                    answer,
                    question.ground_truth,
                    question.answer_type
                )
            correctness.append(1 if is_correct else 0)

        all_correctness.append(correctness)

    # Calculate Pass@k for each k
    metrics = {}
    num_questions = len(all_correctness)

    for k in k_values:
        if k > num_samples:
            continue

        total = 0.0
        for corr in all_correctness:
            total += pass_at_k(sum(corr), len(corr), k)

        metrics[f"pass@{k}"] = total / num_questions if num_questions > 0 else 0.0

    # Coverage
    metrics["coverage"] = sum(
        1 for corr in all_correctness if any(corr)
    ) / num_questions if num_questions > 0 else 0.0

    # Accuracy
    total_correct = sum(sum(corr) for corr in all_correctness)
    total_samples = sum(len(corr) for corr in all_correctness)
    metrics["accuracy"] = total_correct / total_samples if total_samples > 0 else 0.0

    return {
        "metrics": metrics,
        "num_questions": num_questions,
        "num_samples_per_question": num_samples,
    }


def evaluate_with_self_consistency(
    generator: VSTaRGenerator,
    questions: List[QuestionData],
    prompt_fn: Callable[[QuestionData], str],
    num_samples: int = 16,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate using majority voting (self-consistency)

    Args:
        generator: Generator model
        questions: Questions
        prompt_fn: Prompt function
        num_samples: Solutions per question
        show_progress: Show progress

    Returns:
        Metrics dictionary
    """
    correct = 0

    iterator = questions
    if show_progress:
        iterator = tqdm(questions, desc="Self-consistency evaluation")

    for question in iterator:
        prompt = prompt_fn(question)

        # Generate solutions
        solutions = generator.generate(prompt, k=num_samples)

        if not isinstance(solutions, list):
            solutions = [solutions]

        # Extract answers
        answers = []
        for sol in solutions:
            answer = AnswerChecker.extract_answer(sol, question.answer_type)
            if answer is not None:
                answers.append(answer)

        if not answers:
            continue

        # Majority vote
        majority_answer, _ = majority_voting(answers)

        # Check if correct
        is_correct = grade_answer(
            majority_answer,
            question.ground_truth,
            question.answer_type
        )

        if is_correct:
            correct += 1

    accuracy = correct / len(questions) if questions else 0.0

    return {
        "accuracy": accuracy,
        "num_correct": correct,
        "num_questions": len(questions),
    }


def save_evaluation_results(
    results: Dict[str, Any],
    path: str,
) -> None:
    """Save evaluation results to file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Evaluation results saved to {path}")


def load_evaluation_results(path: str) -> Dict[str, Any]:
    """Load evaluation results from file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
