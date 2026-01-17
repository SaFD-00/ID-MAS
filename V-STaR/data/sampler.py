"""Solution sampler for V-STaR"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

from .loader import QuestionData
from evaluation.answer_checker import AnswerChecker, grade_answer


@dataclass
class GeneratedSolution:
    """Data structure for a generated solution"""
    question_id: str
    solution: str
    extracted_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "solution": self.solution,
            "extracted_answer": self.extracted_answer,
            "is_correct": self.is_correct,
            "score": self.score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedSolution":
        return cls(
            question_id=data["question_id"],
            solution=data["solution"],
            extracted_answer=data.get("extracted_answer"),
            is_correct=data.get("is_correct"),
            score=data.get("score"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SamplingResult:
    """Result of sampling solutions for a question"""
    question: QuestionData
    solutions: List[GeneratedSolution]
    num_correct: int = 0
    num_incorrect: int = 0

    def get_correct_solutions(self) -> List[GeneratedSolution]:
        return [s for s in self.solutions if s.is_correct]

    def get_incorrect_solutions(self) -> List[GeneratedSolution]:
        return [s for s in self.solutions if not s.is_correct]


class SolutionSampler:
    """
    Solution sampler for V-STaR

    Generates k solutions per question and labels them for correctness.
    """

    def __init__(
        self,
        generator,  # VSTaRGenerator
        k: int = 16,
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
    ):
        """
        Initialize sampler

        Args:
            generator: VSTaRGenerator instance
            k: Number of solutions to sample per question
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
        """
        self.generator = generator
        self.k = k
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def sample(
        self,
        question: QuestionData,
        prompt: str,
    ) -> SamplingResult:
        """
        Sample k solutions for a single question

        Args:
            question: Question data
            prompt: Formatted prompt

        Returns:
            SamplingResult with generated solutions
        """
        # Generate k solutions
        raw_solutions = self.generator.generate(
            prompt,
            k=self.k,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )

        if not isinstance(raw_solutions, list):
            raw_solutions = [raw_solutions]

        # Process each solution
        solutions = []
        num_correct = 0
        num_incorrect = 0

        for sol_text in raw_solutions:
            # Extract answer
            extracted = AnswerChecker.extract_answer(sol_text, question.answer_type)

            # Grade answer
            is_correct = False
            if extracted is not None:
                is_correct = grade_answer(
                    extracted,
                    question.ground_truth,
                    question.answer_type
                )

            if is_correct:
                num_correct += 1
            else:
                num_incorrect += 1

            solution = GeneratedSolution(
                question_id=question.question_id,
                solution=sol_text,
                extracted_answer=extracted,
                is_correct=is_correct,
            )
            solutions.append(solution)

        return SamplingResult(
            question=question,
            solutions=solutions,
            num_correct=num_correct,
            num_incorrect=num_incorrect,
        )

    def sample_batch(
        self,
        questions: List[QuestionData],
        prompts: List[str],
        show_progress: bool = True,
    ) -> List[SamplingResult]:
        """
        Sample solutions for multiple questions

        Args:
            questions: List of questions
            prompts: List of prompts (one per question)
            show_progress: Show progress bar

        Returns:
            List of SamplingResult
        """
        results = []
        iterator = zip(questions, prompts)

        if show_progress:
            iterator = tqdm(
                list(iterator),
                desc=f"Sampling {self.k} solutions per question"
            )

        for question, prompt in iterator:
            result = self.sample(question, prompt)
            results.append(result)

        return results

    def sample_and_label(
        self,
        questions: List[QuestionData],
        prompt_fn,  # Function to create prompt from question
        show_progress: bool = True,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Sample and label solutions, returning D_GEN and D_VER data

        Args:
            questions: List of questions
            prompt_fn: Function that takes QuestionData and returns prompt string
            show_progress: Show progress bar

        Returns:
            Tuple of (correct_data, all_data) for D_GEN and D_VER
        """
        correct_data = []  # For D_GEN (correct solutions only)
        all_data = []      # For D_VER (all solutions)

        iterator = questions
        if show_progress:
            iterator = tqdm(questions, desc="Sampling and labeling")

        for question in iterator:
            prompt = prompt_fn(question)
            result = self.sample(question, prompt)

            for solution in result.solutions:
                data_item = {
                    "question_id": question.question_id,
                    "question": question.question,
                    "prompt": prompt,
                    "solution": solution.solution,
                    "extracted_answer": solution.extracted_answer,
                    "is_correct": solution.is_correct,
                    "ground_truth": question.ground_truth_formatted,
                    "answer_type": question.answer_type.value,
                }

                all_data.append(data_item)

                if solution.is_correct:
                    correct_data.append(data_item)

        return correct_data, all_data

    def get_statistics(
        self,
        results: List[SamplingResult]
    ) -> Dict[str, Any]:
        """Get statistics from sampling results"""
        total_questions = len(results)
        total_solutions = sum(len(r.solutions) for r in results)
        total_correct = sum(r.num_correct for r in results)
        total_incorrect = sum(r.num_incorrect for r in results)

        # Questions with at least one correct solution
        questions_with_correct = sum(1 for r in results if r.num_correct > 0)

        return {
            "total_questions": total_questions,
            "total_solutions": total_solutions,
            "total_correct": total_correct,
            "total_incorrect": total_incorrect,
            "accuracy_per_solution": total_correct / total_solutions if total_solutions > 0 else 0,
            "questions_with_correct": questions_with_correct,
            "coverage": questions_with_correct / total_questions if total_questions > 0 else 0,
        }
