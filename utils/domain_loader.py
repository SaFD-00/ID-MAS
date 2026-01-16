"""
Domain-based Dataset Loader for ID-MAS.
Provides unified interface for loading domain data from local JSON files.

Each training dataset has its own Instructional Goal for separate learning.

Usage:
    loader = DomainLoader("math")
    train_data = loader.load_training_data(dataset="gsm8k", limit=100)
    eval_data = loader.load_eval_data("svamp", limit=50)
    instructional_goal = loader.get_learning_objective("gsm8k")
"""
import json
import random
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

from utils.base_loader import BaseDatasetLoader, QuestionData, AnswerType
from utils.answer_extractor import extract_boxed_answer


# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class DomainLoader(BaseDatasetLoader):
    """
    Domain-based data loader for local JSON files.

    Currently supports:
    - math: GSM8K, MATH (training) + SVAMP, ASDiv, MAWPS (evaluation)

    New domains can be added by extending DOMAIN_CONFIG.
    Each training dataset has its own Instructional Goal and is trained separately.
    Evaluation data is loaded from a single file per dataset.
    """

    # Instructional Goals for each training dataset
    INSTRUCTIONAL_GOALS = {
        # Math domain
        "gsm8k": "Generate coherent, step-by-step mathematical reasoning in natural language that leads to a correct numerical answer for grade-school level math problems.",
        "math": "Solve advanced mathematical problems by selecting appropriate mathematical concepts and constructing logically valid, multi-step reasoning that leads to a correct solution.",

        # Logical domain
        "reclor": "Analyze logical reasoning problems by comprehending complex passages, identifying logical relationships, and selecting the most appropriate conclusion based on formal reasoning principles.",

        # Commonsense domain
        "arc_c": "Apply commonsense scientific knowledge to solve elementary science problems by understanding fundamental concepts and selecting the correct answer from multiple choices.",

        # BBH (Big Bench Hard)
        "bbh": "Evaluate and solve various logical reasoning tasks including boolean expressions, formal fallacies, logical deduction, and object tracking.",
    }

    DOMAIN_CONFIG = {
        "math": {
            "training_datasets": {
                "gsm8k": {"filename": "gsm8k_train.json", "answer_type": AnswerType.NUMERIC},
                "math": {"filename": "math_train.json", "answer_type": AnswerType.LATEX},
            },
            "eval_datasets": {
                "gsm8k": {"filename": "gsm8k_test.json", "answer_type": AnswerType.NUMERIC},
                "math": {"filename": "math_test.json", "answer_type": AnswerType.LATEX},
                "svamp": {"filename": "svamp_test.json", "answer_type": AnswerType.NUMERIC},
                "asdiv": {"filename": "asdiv_test.json", "answer_type": AnswerType.NUMERIC},
                "mawps": {"filename": "mawps_test.json", "answer_type": AnswerType.LATEX},  # 분수 포함
            },
            "default_answer_type": AnswerType.NUMERIC,
            "domain_category": "math_logic",
            "data_dir": "data/math"
        },
        "logical": {
            "training_datasets": {
                "reclor": {"filename": "reclor_train.json", "answer_type": AnswerType.MCQ},
            },
            "eval_datasets": {
                "reclor": {"filename": "reclor_test.json", "answer_type": AnswerType.MCQ},
                "anli_r2": {"filename": "anli_r2_test.json", "answer_type": AnswerType.MCQ},
                "anli_r3": {"filename": "anli_r3_test.json", "answer_type": AnswerType.MCQ},
                # BBH unified (all subtasks combined, subtask info in metadata.subtask)
                "bbh": {"filename": "bbh_test.json", "answer_type": AnswerType.TEXT},
            },
            "default_answer_type": AnswerType.MCQ,
            "domain_category": "logical_reasoning",
            "data_dir": "data/logical"
        },
        "commonsense": {
            "training_datasets": {
                "arc_c": {"filename": "arc_c_train.json", "answer_type": AnswerType.MCQ},
            },
            "eval_datasets": {
                "arc_c": {"filename": "arc_c_test.json", "answer_type": AnswerType.MCQ},
                "strategyqa": {"filename": "strategyqa_test.json", "answer_type": AnswerType.BOOLEAN},
                "openbookqa": {"filename": "openbookqa_test.json", "answer_type": AnswerType.MCQ},
            },
            "default_answer_type": AnswerType.MCQ,
            "domain_category": "commonsense_reasoning",
            "data_dir": "data/commonsense"
        }
    }

    def __init__(self, domain: str):
        """
        Initialize domain loader.

        Args:
            domain: Domain name (e.g., "math")
        """
        domain = domain.lower()
        if domain not in self.DOMAIN_CONFIG:
            raise ValueError(
                f"Unknown domain: {domain}. "
                f"Available domains: {list(self.DOMAIN_CONFIG.keys())}"
            )

        self.domain = domain
        self.config = self.DOMAIN_CONFIG[domain]
        self.data_dir = PROJECT_ROOT / self.config["data_dir"]
        self._current_eval_dataset: Optional[str] = None

    @property
    def dataset_name(self) -> str:
        """Return domain name as dataset identifier."""
        return f"domain_{self.domain}"

    @property
    def answer_type(self) -> AnswerType:
        """Return default answer type for this domain."""
        return self.config["default_answer_type"]

    @property
    def domain_category(self) -> str:
        """Return domain category."""
        return self.config["domain_category"]

    def load_data(
        self,
        split: str = "train",
        subset: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[QuestionData]:
        """
        Load data based on split.

        Args:
            split: "train" for training data, "test" for evaluation
            subset: For train split, specify training dataset; for test, specify eval dataset
            limit: Maximum number of questions

        Returns:
            List of QuestionData objects
        """
        if split == "train":
            if not subset:
                raise ValueError("Training dataset must be specified via 'subset' parameter")
            return self.load_training_data(dataset=subset, limit=limit)
        else:
            eval_dataset = subset or self._get_default_eval()
            return self.load_eval_data(eval_dataset, limit=limit)

    def load_training_data(
        self,
        dataset: str,
        limit: Optional[int] = None,
        shuffle: bool = False
    ) -> List[QuestionData]:
        """
        Load a specific training dataset for this domain.

        Each training dataset (GSM8K, MATH, SciBench, ARC) is loaded separately
        because each has its own Instructional Goal.

        Args:
            dataset: Training dataset name (e.g., "gsm8k", "math", "scibench", "arc")
            limit: Maximum number of questions to load
            shuffle: Whether to shuffle the data (default: False)

        Returns:
            List of QuestionData objects
        """
        dataset = dataset.lower()
        training_datasets = self.config["training_datasets"]

        if dataset not in training_datasets:
            raise ValueError(
                f"Unknown training dataset '{dataset}' for {self.domain} domain. "
                f"Available: {list(training_datasets.keys())}"
            )

        ds_config = training_datasets[dataset]
        filename = self._get_filename(ds_config)
        # New structure: data/{domain}/train/data/{dataset}_train.json
        file_path = self.data_dir / "train" / "data" / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Training file not found: {file_path}")

        questions = self._load_json_file(file_path, dataset, "train")
        print(f"  Loaded {len(questions)} questions from {filename}")

        # Shuffle data
        if shuffle and questions:
            random.shuffle(questions)
            print(f"  Shuffled {len(questions)} training questions")

        # Apply limit after shuffling
        if limit and len(questions) > limit:
            questions = questions[:limit]

        print(f"[{self.domain.upper()}/{dataset.upper()}] Loaded {len(questions)} training questions")
        return questions

    def load_enhanced_training_data(
        self,
        dataset: str,
        model_suffix: str,
        limit: Optional[int] = None,
        shuffle: bool = False
    ) -> List[QuestionData]:
        """
        Load enhanced training data with Instructional Goal and Task Analysis.

        Enhanced data files have the naming pattern:
            {dataset}_train_ID-MAS_{model_suffix}.json

        These files contain instruction fields that include:
        - Original instruction
        - Instructional Goal
        - Task Analysis (subskills and subtasks)

        Args:
            dataset: Training dataset name (e.g., "gsm8k", "math", "reclor", "arc_c")
            model_suffix: Model suffix used when generating enhanced data
                (e.g., "Qwen2.5-72B-Instruct", "gpt-5-2025-08-07")
            limit: Maximum number of questions to load
            shuffle: Whether to shuffle the data (default: False)

        Returns:
            List of QuestionData objects

        Raises:
            FileNotFoundError: If enhanced data file doesn't exist
            ValueError: If dataset is not a valid training dataset for this domain

        Example:
            loader = DomainLoader("math")
            data = loader.load_enhanced_training_data(
                dataset="gsm8k",
                model_suffix="Qwen2.5-72B-Instruct",
                limit=100
            )
        """
        dataset = dataset.lower()
        training_datasets = self.config["training_datasets"]

        if dataset not in training_datasets:
            raise ValueError(
                f"Unknown training dataset '{dataset}' for {self.domain} domain. "
                f"Available: {list(training_datasets.keys())}"
            )

        # Enhanced data filename pattern
        filename = f"{dataset}_train_ID-MAS_{model_suffix}.json"
        file_path = self.data_dir / "train" / "data" / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Enhanced training file not found: {file_path}\n"
                f"Run 'python scripts/enhance_data.py --domain {self.domain} --dataset {dataset}' first."
            )

        questions = self._load_json_file(file_path, dataset, "train")
        print(f"  Loaded {len(questions)} enhanced questions from {filename}")

        # Shuffle data
        if shuffle and questions:
            random.shuffle(questions)
            print(f"  Shuffled {len(questions)} enhanced training questions")

        # Apply limit after shuffling
        if limit and len(questions) > limit:
            questions = questions[:limit]

        print(f"[{self.domain.upper()}/{dataset.upper()}] Loaded {len(questions)} enhanced training questions")
        return questions

    def get_available_enhanced_data(self, dataset: str = None) -> List[Dict[str, str]]:
        """
        List available enhanced data files for this domain.

        Args:
            dataset: Optional filter by dataset name

        Returns:
            List of dicts with 'dataset', 'model_suffix', 'path' keys
        """
        data_dir = self.data_dir / "train" / "data"
        if not data_dir.exists():
            return []

        results = []
        pattern = "*_train_ID-MAS_*.json"

        for file_path in data_dir.glob(pattern):
            # Parse filename: {dataset}_train_ID-MAS_{model_suffix}.json
            name = file_path.stem  # Remove .json
            parts = name.split("_train_ID-MAS_")
            if len(parts) == 2:
                ds_name, model_suffix = parts
                if dataset is None or ds_name.lower() == dataset.lower():
                    results.append({
                        "dataset": ds_name,
                        "model_suffix": model_suffix,
                        "path": str(file_path)
                    })

        return results

    def load_eval_data(
        self,
        dataset: str,
        limit: Optional[int] = None
    ) -> List[QuestionData]:
        """
        Load specific evaluation dataset.

        Args:
            dataset: Evaluation dataset name (e.g., "gsm8k", "svamp", "arc")
            limit: Maximum number of questions

        Returns:
            List of QuestionData objects
        """
        dataset = dataset.lower()
        if dataset not in self.config["eval_datasets"]:
            raise ValueError(
                f"Unknown eval dataset '{dataset}' for {self.domain} domain. "
                f"Available: {list(self.config['eval_datasets'].keys())}"
            )

        self._current_eval_dataset = dataset
        ds_config = self.config["eval_datasets"][dataset]
        filename = self._get_filename(ds_config)
        # New structure: data/{domain}/eval/data/{dataset}_test.json
        file_path = self.data_dir / "eval" / "data" / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Evaluation file not found: {file_path}")

        questions = self._load_json_file(file_path, dataset, "test")

        if limit and len(questions) > limit:
            questions = questions[:limit]

        print(f"[{self.domain.upper()}] Loaded {len(questions)} eval questions from {dataset}")
        return questions

    def _get_filename(self, ds_config) -> str:
        """
        Extract filename from dataset config.

        Args:
            ds_config: Either a string (filename) or dict with 'filename' key

        Returns:
            Filename string
        """
        if isinstance(ds_config, str):
            return ds_config
        return ds_config.get("filename", "")

    def _get_dataset_answer_type(self, dataset_name: str) -> AnswerType:
        """
        Get answer type for a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            AnswerType for the dataset
        """
        dataset_name = dataset_name.lower()

        # Check training_datasets
        if dataset_name in self.config.get("training_datasets", {}):
            ds_config = self.config["training_datasets"][dataset_name]
            if isinstance(ds_config, dict) and "answer_type" in ds_config:
                return ds_config["answer_type"]

        # Check eval_datasets
        if dataset_name in self.config.get("eval_datasets", {}):
            ds_config = self.config["eval_datasets"][dataset_name]
            if isinstance(ds_config, dict) and "answer_type" in ds_config:
                return ds_config["answer_type"]

        # Fallback: use default_answer_type
        return self.config.get("default_answer_type", AnswerType.TEXT)

    def _load_json_file(
        self,
        file_path: Path,
        dataset_name: str,
        split: str
    ) -> List[QuestionData]:
        """
        Load and parse a JSON file into QuestionData objects.

        Args:
            file_path: Path to JSON file
            dataset_name: Name of the dataset
            split: Data split (train/test)

        Returns:
            List of QuestionData objects
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            items = json.load(f)

        questions = []
        for i, item in enumerate(items):
            question_data = self._parse_item(
                item=item,
                dataset_name=dataset_name,
                question_id=f"{dataset_name}_{split}_{i}",
            )
            questions.append(question_data)

        return questions

    def _parse_item(
        self,
        item: Dict[str, Any],
        dataset_name: str,
        question_id: str
    ) -> QuestionData:
        """
        Parse a JSON item into QuestionData.

        Expected JSON format:
        {
            "instruction": "system prompt",
            "input": "question text",
            "output": "reasoning...\n\n#### answer"
        }

        Args:
            item: JSON item dictionary
            dataset_name: Name of the dataset
            question_id: Unique question identifier

        Returns:
            QuestionData object
        """
        question_text = item.get("input", "")
        output = item.get("output", "")
        instruction = item.get("instruction", "")

        # Extract answer from output using #### pattern
        answer, answer_type = self._extract_answer(output, dataset_name)

        # Detect MCQ choices from question text
        choices = self._extract_choices(question_text)

        return QuestionData(
            dataset=dataset_name,
            question_id=question_id,
            question=question_text,
            answer_type=answer_type,
            ground_truth=answer,
            ground_truth_formatted=f"The correct answer is {answer}",
            choices=choices,
            metadata={
                "instruction": instruction,
                "full_output": output
            }
        )

    def _extract_answer(self, output: str, dataset_name: str) -> tuple:
        """
        Extract answer and determine type from output field.

        Priority:
        1. \\boxed{...} pattern (handles nested braces)
        2. #### pattern (GSM8K style)
        3. Fallback: last line

        Args:
            output: Output text containing answer
            dataset_name: Name of dataset (for type lookup)

        Returns:
            Tuple of (answer, AnswerType)
        """
        # 1. Try \boxed{...} pattern first (handles nested braces)
        boxed_answer = extract_boxed_answer(output)
        if boxed_answer:
            answer = boxed_answer
        # 2. Try #### pattern (GSM8K style)
        elif match := re.search(r'####\s*(.+?)$', output.strip(), re.MULTILINE):
            answer = match.group(1).strip()
        # 3. Fallback: use last line
        else:
            answer = output.strip().split('\n')[-1].strip()

        # Get answer type from dataset config (not inferred)
        answer_type = self._get_dataset_answer_type(dataset_name)

        return answer, answer_type

    def _infer_answer_type(self, answer: str, dataset_name: str) -> AnswerType:
        """
        Infer answer type from answer content and dataset name.

        Args:
            answer: Extracted answer string
            dataset_name: Name of dataset

        Returns:
            AnswerType enum value
        """
        # MCQ: single letter A-E or 1-5
        if answer.upper() in ['A', 'B', 'C', 'D', 'E', '1', '2', '3', '4', '5']:
            return AnswerType.MCQ

        # MATH dataset uses LaTeX
        if dataset_name == "math":
            return AnswerType.LATEX

        # Try numeric
        try:
            # Handle various numeric formats
            cleaned = answer.replace(',', '').replace(' ', '')
            # Remove units if present
            numeric_match = re.match(r'^-?[\d.]+', cleaned)
            if numeric_match:
                float(numeric_match.group())
                return AnswerType.NUMERIC
        except (ValueError, AttributeError):
            pass

        # Default based on domain
        return self.config["default_answer_type"]

    def _extract_choices(self, question_text: str) -> Optional[List[str]]:
        """
        Extract MCQ choices from question text if present.

        Args:
            question_text: Question text potentially containing choices

        Returns:
            List of choices or None if not MCQ
        """
        # Pattern: A. choice text or A) choice text
        pattern = r'^([A-E])[.)]\s*(.+)$'
        lines = question_text.strip().split('\n')

        choices = []
        for line in lines:
            match = re.match(pattern, line.strip())
            if match:
                choices.append(match.group(2).strip())

        return choices if len(choices) >= 2 else None

    def _get_default_eval(self) -> str:
        """Get default evaluation dataset for domain."""
        # Use first eval dataset as default
        eval_datasets = list(self.config["eval_datasets"].keys())
        return eval_datasets[0] if eval_datasets else None

    def get_available_eval_datasets(self) -> List[str]:
        """Return list of available evaluation datasets for this domain."""
        return list(self.config["eval_datasets"].keys())

    def get_available_training_datasets(self) -> List[str]:
        """Return list of training datasets for this domain."""
        return list(self.config["training_datasets"].keys())

    def format_question_as_prompt(self, question: QuestionData) -> str:
        """
        Format question for LLM input using the instruction from JSON.

        Args:
            question: QuestionData object

        Returns:
            Formatted prompt string
        """
        instruction = question.metadata.get("instruction", "")
        if instruction:
            return f"{instruction}\n\n{question.question}"
        return question.question

    def format_ground_truth(self, question: QuestionData) -> str:
        """
        Format ground truth for teacher model evaluation.

        Args:
            question: QuestionData object

        Returns:
            Human-readable ground truth string
        """
        return question.ground_truth_formatted

    def get_learning_objective(self, dataset: str) -> str:
        """
        Get Instructional Goal (learning objective) for a specific training dataset.

        Each training dataset has its own Instructional Goal:
        - GSM8K: Grade-school math step-by-step reasoning
        - MATH: Advanced mathematical problem solving
        - SciBench: Scientific problem solving with mathematical formulations
        - ARC: Abstract transformation rule inference

        Args:
            dataset: Training dataset name (gsm8k, math, scibench, arc)

        Returns:
            Instructional Goal string for the dataset
        """
        dataset = dataset.lower()
        if dataset not in self.INSTRUCTIONAL_GOALS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Available: {list(self.INSTRUCTIONAL_GOALS.keys())}"
            )
        return self.INSTRUCTIONAL_GOALS[dataset]

    def get_available_subsets(self) -> Optional[List[str]]:
        """Return list of available evaluation datasets as subsets."""
        return self.get_available_eval_datasets()


# Convenience functions for external use
def get_domain_loader(domain: str) -> DomainLoader:
    """Get a domain loader instance."""
    return DomainLoader(domain)


def get_available_domains() -> List[str]:
    """Get list of available domains."""
    return list(DomainLoader.DOMAIN_CONFIG.keys())


def get_eval_datasets_for_domain(domain: str) -> List[str]:
    """Get available evaluation datasets for a domain."""
    return DomainLoader(domain).get_available_eval_datasets()


if __name__ == "__main__":
    # Test the loader
    print("=" * 60)
    print("Testing DomainLoader")
    print("=" * 60)

    # Test math domain
    print("\n[Math Domain]")
    math_loader = DomainLoader("math")
    print(f"  Available eval datasets: {math_loader.get_available_eval_datasets()}")
    print(f"  Training datasets: {math_loader.get_available_training_datasets()}")

    # Test Instructional Goals
    for dataset in math_loader.get_available_training_datasets():
        print(f"\n  Instructional Goal for {dataset.upper()}:")
        print(f"    {math_loader.get_learning_objective(dataset)[:80]}...")

    # Load a few training samples from GSM8K
    print("\n  Loading GSM8K training data (limit=3):")
    try:
        train_data = math_loader.load_training_data(dataset="gsm8k", limit=3)
        print(f"  Sample training questions ({len(train_data)}):")
        for q in train_data:
            print(f"    [{q.dataset}] {q.question[:50]}... -> {q.ground_truth}")
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")

    # Test evaluation data loading
    print("\n[Evaluation Data Test]")
    print("  Testing cross-dataset evaluation with SVAMP:")
    try:
        svamp_data = loader.load_eval_data("svamp", limit=3)
        print(f"  Loaded {len(svamp_data)} SVAMP questions")
        for q in svamp_data:
            print(f"    [{q.dataset}] {q.question[:50]}... -> {q.ground_truth}")
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")
