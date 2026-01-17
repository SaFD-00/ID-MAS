"""Evaluation metrics for V-STaR"""

from typing import List, Tuple, Optional
from math import comb
from collections import Counter


def pass_at_k(
    num_correct: int,
    num_samples: int,
    k: int
) -> float:
    """
    Calculate Pass@k metric

    Pass@k estimates the probability that at least one of k samples is correct.

    Formula: 1 - C(n-c, k) / C(n, k)

    Args:
        num_correct: Number of correct samples (c)
        num_samples: Total number of samples (n)
        k: Number of samples to consider

    Returns:
        Pass@k probability
    """
    if num_samples < k:
        return 1.0 if num_correct > 0 else 0.0

    if num_samples - num_correct < k:
        return 1.0

    return 1.0 - comb(num_samples - num_correct, k) / comb(num_samples, k)


def best_of_k(
    correctness: List[int],
    k: int
) -> float:
    """
    Calculate Best-of-k metric (V-STaR Eq. 3)

    Estimates the probability that the top-ranked solution (by verifier)
    out of k samples is correct.

    Formula: Best-of-k = (1 / C(N,k)) * sum_{i=0}^{N-k} C(N-i-1, k-1) * alpha_i

    Args:
        correctness: Binary correctness values sorted by verifier score (descending)
                    [1, 0, 1, 1, 0, ...] where 1=correct, 0=incorrect
        k: Number of candidates to consider

    Returns:
        Best-of-k accuracy
    """
    N = len(correctness)

    if N == 0:
        return 0.0

    if k > N:
        k = N

    if k == 0:
        return 0.0

    numerator = 0.0
    for i in range(N - k + 1):
        # C(N-i-1, k-1) * alpha_i
        numerator += comb(N - i - 1, k - 1) * correctness[i]

    denominator = comb(N, k)

    if denominator == 0:
        return 0.0

    return numerator / denominator


def best_of_k_from_scores(
    solutions: List[str],
    scores: List[float],
    correctness: List[int],
    k: int
) -> float:
    """
    Calculate Best-of-k given solutions, scores, and correctness

    Args:
        solutions: List of solution strings
        scores: List of verifier scores
        correctness: List of correctness labels (0 or 1)
        k: Number of candidates

    Returns:
        Best-of-k accuracy
    """
    # Sort by scores descending
    sorted_data = sorted(
        zip(scores, correctness),
        key=lambda x: x[0],
        reverse=True
    )
    sorted_correctness = [c for _, c in sorted_data]

    return best_of_k(sorted_correctness, k)


def majority_voting(
    answers: List[str],
    correctness_map: Optional[dict] = None
) -> Tuple[str, float]:
    """
    Majority voting (Self-Consistency)

    Args:
        answers: List of extracted answers
        correctness_map: Optional mapping from answer to correctness

    Returns:
        Tuple of (majority_answer, accuracy)
        accuracy is 1.0 if majority is correct, 0.0 otherwise
    """
    if not answers:
        return "", 0.0

    # Count answers
    counter = Counter(answers)
    majority_answer, count = counter.most_common(1)[0]

    # Calculate accuracy if correctness_map provided
    if correctness_map is not None:
        is_correct = correctness_map.get(majority_answer, False)
        return majority_answer, 1.0 if is_correct else 0.0

    return majority_answer, count / len(answers)


def weighted_majority_voting(
    answers: List[str],
    scores: List[float],
    correctness_map: Optional[dict] = None
) -> Tuple[str, float]:
    """
    Weighted majority voting using verifier scores

    Args:
        answers: List of extracted answers
        scores: List of verifier scores
        correctness_map: Optional mapping from answer to correctness

    Returns:
        Tuple of (best_answer, accuracy)
    """
    if not answers:
        return "", 0.0

    # Aggregate scores by answer
    answer_scores = {}
    for answer, score in zip(answers, scores):
        if answer not in answer_scores:
            answer_scores[answer] = 0.0
        answer_scores[answer] += score

    # Find best answer
    best_answer = max(answer_scores.keys(), key=lambda a: answer_scores[a])

    if correctness_map is not None:
        is_correct = correctness_map.get(best_answer, False)
        return best_answer, 1.0 if is_correct else 0.0

    return best_answer, answer_scores[best_answer]


def calculate_accuracy(
    predictions: List[str],
    ground_truths: List[str],
    normalize: bool = True
) -> float:
    """
    Calculate accuracy

    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        normalize: Whether to normalize strings before comparison

    Returns:
        Accuracy
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have same length")

    if not predictions:
        return 0.0

    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        if normalize:
            pred = str(pred).strip().lower()
            gt = str(gt).strip().lower()
        if pred == gt:
            correct += 1

    return correct / len(predictions)


def calculate_metrics(
    solutions: List[List[str]],
    scores: List[List[float]],
    correctness: List[List[int]],
    k_values: List[int] = [1, 4, 8, 16, 32, 64]
) -> dict:
    """
    Calculate multiple metrics

    Args:
        solutions: List of solution lists (per problem)
        scores: List of score lists (per problem)
        correctness: List of correctness lists (per problem)
        k_values: K values to evaluate

    Returns:
        Dictionary of metrics
    """
    metrics = {}
    num_problems = len(solutions)

    for k in k_values:
        # Pass@k
        pass_k_sum = 0.0
        for corr in correctness:
            num_correct = sum(corr)
            num_samples = len(corr)
            pass_k_sum += pass_at_k(num_correct, num_samples, k)
        metrics[f"pass@{k}"] = pass_k_sum / num_problems if num_problems > 0 else 0.0

        # Best-of-k
        best_k_sum = 0.0
        for sc, corr in zip(scores, correctness):
            # Sort by scores descending
            sorted_data = sorted(zip(sc, corr), key=lambda x: x[0], reverse=True)
            sorted_corr = [c for _, c in sorted_data]
            best_k_sum += best_of_k(sorted_corr, k)
        metrics[f"best_of_{k}"] = best_k_sum / num_problems if num_problems > 0 else 0.0

    return metrics


class MetricsCalculator:
    """Calculator for V-STaR metrics"""

    def __init__(self, k_values: List[int] = None):
        """
        Initialize calculator

        Args:
            k_values: K values to evaluate
        """
        self.k_values = k_values or [1, 4, 8, 16, 32, 64]
        self.reset()

    def reset(self):
        """Reset accumulated data"""
        self.all_solutions = []
        self.all_scores = []
        self.all_correctness = []

    def add(
        self,
        solutions: List[str],
        scores: List[float],
        correctness: List[int]
    ):
        """Add data for one problem"""
        self.all_solutions.append(solutions)
        self.all_scores.append(scores)
        self.all_correctness.append(correctness)

    def compute(self) -> dict:
        """Compute all metrics"""
        return calculate_metrics(
            self.all_solutions,
            self.all_scores,
            self.all_correctness,
            self.k_values
        )

    def get_pass_at_k(self, k: int) -> float:
        """Get Pass@k for current data"""
        total = 0.0
        for corr in self.all_correctness:
            num_correct = sum(corr)
            num_samples = len(corr)
            total += pass_at_k(num_correct, num_samples, k)
        return total / len(self.all_correctness) if self.all_correctness else 0.0

    def get_best_of_k(self, k: int) -> float:
        """Get Best-of-k for current data"""
        total = 0.0
        for sc, corr in zip(self.all_scores, self.all_correctness):
            sorted_data = sorted(zip(sc, corr), key=lambda x: x[0], reverse=True)
            sorted_corr = [c for _, c in sorted_data]
            total += best_of_k(sorted_corr, k)
        return total / len(self.all_scores) if self.all_scores else 0.0
