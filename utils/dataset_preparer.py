#!/usr/bin/env python3
"""
Dataset Preparation Script
Downloads and processes datasets from HuggingFace for math and knowledge domains.

Output format:
[
  {
    "instruction": "system prompt (different for train vs test)",
    "input": "question text",
    "output": "{reasoning}\n\n\\boxed{answer}" or "\\boxed{answer}"
  },
  ...
]

System Prompts:
- Train: "You are a helpful math assistant."
- Test: "You are a helpful math assistant. Solve the problem step by step and provide your final answer within \\boxed{}."
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"

# System prompts - same for both train and test
TRAIN_SYSTEM_PROMPT = "You are a helpful math assistant.\nSolve the problem step by step and provide your final answer within \\boxed{}."

TEST_SYSTEM_PROMPT = "You are a helpful math assistant.\nSolve the problem step by step and provide your final answer within \\boxed{}."

# MMLU Math subjects
MMLU_MATH_SUBJECTS = [
    "abstract_algebra",
    "college_mathematics",
    "elementary_mathematics",
    "high_school_mathematics",
    "high_school_statistics",
]

# MMLU Science subjects (similar to SciBench/ARC training domains)
MMLU_SCIENCE_SUBJECTS = [
    "college_biology",
    "college_chemistry",
    "college_physics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_physics",
    "anatomy",
    "astronomy",
    "conceptual_physics",
]


# =============================================================================
# Helper Functions
# =============================================================================

def format_output(reasoning: Optional[str], answer: str) -> str:
    """
    Format output with reasoning and answer in \\boxed{} format.

    Args:
        reasoning: Step-by-step reasoning (can be None)
        answer: Final answer

    Returns:
        Formatted output string with \\boxed{answer} format
    """
    # Format answer in \boxed{} - escape backslash for proper string formatting
    boxed_answer = f"\\boxed{{{answer}}}"

    if reasoning and reasoning.strip():
        return f"{reasoning.strip()}\n\n{boxed_answer}"
    return boxed_answer


def format_mcq_input(question: str, choices: List[str]) -> str:
    """
    Format multiple choice question with choices.

    Args:
        question: Question text
        choices: List of answer choices

    Returns:
        Formatted input string
    """
    formatted = question + "\n\n"
    for i, choice in enumerate(choices):
        formatted += f"{chr(65 + i)}. {choice}\n"
    return formatted.strip()


def save_json(data: List[Dict], output_path: Path):
    """Save data to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(data)} records to {output_path.name}")


def extract_boxed_answer(solution: str) -> str:
    """
    Extract answer from LaTeX \\boxed{} command in MATH dataset.

    Args:
        solution: Solution text with boxed answer

    Returns:
        Extracted answer or empty string
    """
    # Try to find \boxed{...} pattern
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, solution)
    if matches:
        return matches[-1]  # Return the last boxed answer

    # Fallback: try simpler pattern
    simple_pattern = r'\\boxed\{(.+?)\}'
    simple_matches = re.findall(simple_pattern, solution)
    if simple_matches:
        return simple_matches[-1]

    return ""


# =============================================================================
# Dataset Processors
# =============================================================================

def process_gsm8k(output_dir: Path):
    """
    Process GSM8K dataset.

    GSM8K format:
    - question: math word problem
    - answer: "reasoning steps #### final_answer"
    """
    print("\n[GSM8K] Processing...")
    dataset_id = "openai/gsm8k"

    for split in ["train", "test"]:
        print(f"  Loading {split} split...")
        data = load_dataset(dataset_id, "main", split=split)

        records = []
        for item in data:
            question = item["question"]
            answer_text = item["answer"]

            # GSM8K answer format: "reasoning\n#### number"
            if "####" in answer_text:
                parts = answer_text.split("####")
                reasoning = parts[0].strip()
                final_answer = parts[1].strip()
            else:
                reasoning = answer_text
                final_answer = answer_text

            # Use split-aware prompt
            instruction = TRAIN_SYSTEM_PROMPT if split == "train" else TEST_SYSTEM_PROMPT

            records.append({
                "instruction": instruction,
                "input": question,
                "output": format_output(reasoning, final_answer)
            })

        save_json(records, output_dir / f"gsm8k_{split}.json")


def process_math(output_dir: Path):
    """
    Process MATH (competition_math) dataset.
    Uses EleutherAI/hendrycks_math as alternative since original is DMCA restricted.

    MATH format:
    - problem: problem text
    - solution: step-by-step solution with \\boxed{answer}
    - level: difficulty level
    - type: subject category
    """
    print("\n[MATH] Processing...")
    # Use EleutherAI version since hendrycks/competition_math has DMCA restriction
    dataset_id = "EleutherAI/hendrycks_math"

    # MATH dataset has 7 subject configs
    math_configs = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus"
    ]

    for split in ["train", "test"]:
        print(f"  Loading {split} split...")
        records = []
        level_counts = {}

        for config in math_configs:
            print(f"    Loading {config}...")
            try:
                data = load_dataset(dataset_id, config, split=split)

                for item in data:
                    problem = item["problem"]
                    solution = item["solution"]
                    level = item.get("level", "")

                    # Extract answer from \boxed{}
                    boxed_answer = extract_boxed_answer(solution)
                    if not boxed_answer:
                        boxed_answer = "See solution"

                    # Track level distribution
                    level_counts[level] = level_counts.get(level, 0) + 1

                    # Use split-aware prompt
                    instruction = TRAIN_SYSTEM_PROMPT if split == "train" else TEST_SYSTEM_PROMPT

                    records.append({
                        "instruction": instruction,
                        "input": problem,
                        "output": format_output(solution, boxed_answer)
                    })
            except Exception as e:
                print(f"      Error loading {config}: {e}")

        print(f"  Level distribution: {level_counts}")
        save_json(records, output_dir / f"math_{split}.json")


def process_mmlu(output_dir: Path, subjects: List[str], output_filename: str, is_math_domain: bool = False):
    """
    Process MMLU dataset for specific subjects.

    MMLU format:
    - question: question text
    - choices: list of 4 choices
    - answer: index (0-3)
    """
    print(f"\n[MMLU] Processing {output_filename}...")
    dataset_id = "cais/mmlu"

    # Always test data - use TEST_SYSTEM_PROMPT
    system_prompt = TEST_SYSTEM_PROMPT

    all_records = []

    for subject in subjects:
        print(f"  Loading subject: {subject}...")
        try:
            data = load_dataset(dataset_id, subject, split="test")

            for item in data:
                question = item["question"]
                choices = item["choices"]
                answer_idx = item["answer"]
                answer_letter = chr(65 + answer_idx)  # 0 -> A, 1 -> B, etc.

                all_records.append({
                    "instruction": system_prompt,
                    "input": format_mcq_input(question, choices),
                    "output": format_output(None, answer_letter)
                })

            print(f"    Loaded {len(data)} questions")
        except Exception as e:
            print(f"    Error loading {subject}: {e}")

    save_json(all_records, output_dir / output_filename)


def process_svamp(output_dir: Path):
    """
    Process SVAMP dataset.

    SVAMP format:
    - Body: problem context
    - Question: the question
    - Answer: numerical answer
    - Equation: mathematical equation
    """
    print("\n[SVAMP] Processing...")
    dataset_id = "ChilleD/SVAMP"

    print("  Loading test split...")
    data = load_dataset(dataset_id, split="test")

    records = []
    for item in data:
        body = item.get("Body", "")
        question = item.get("Question", "")
        answer = str(item.get("Answer", ""))

        full_question = f"{body} {question}".strip()

        records.append({
            "instruction": TEST_SYSTEM_PROMPT,
            "input": full_question,
            "output": format_output(None, answer)
        })

    save_json(records, output_dir / "svamp_test.json")


def process_asdiv(output_dir: Path):
    """
    Process ASDiv dataset.
    Uses EleutherAI/asdiv which has only validation split.

    ASDiv format:
    - body: problem context
    - question: the question
    - answer: numerical answer
    - formula: mathematical formula
    """
    print("\n[ASDiv] Processing...")
    dataset_id = "EleutherAI/asdiv"

    print("  Loading validation split (used as test)...")
    try:
        data = load_dataset(dataset_id, split="validation")
    except Exception as e:
        print(f"  Could not load ASDiv dataset: {e}")
        return

    records = []
    for item in data:
        body = item.get("body", "")
        question = item.get("question", "")
        answer = str(item.get("answer", ""))

        # Combine body and question
        full_question = f"{body} {question}".strip()

        # Clean answer - extract just the number
        # Answer format might be "9 (apples)" -> extract "9"
        answer_clean = answer.split()[0] if answer else answer

        if full_question:
            records.append({
                "instruction": TEST_SYSTEM_PROMPT,
                "input": full_question,
                "output": format_output(None, answer_clean)
            })

    if records:
        save_json(records, output_dir / "asdiv_test.json")
    else:
        print("  No records processed for ASDiv")


def process_mawps(output_dir: Path):
    """
    Process MAWPS dataset.
    Uses MU-NLPC/Calc-mawps which has filtered train/validation/test splits.

    MAWPS format:
    - question: math word problem
    - result: numerical answer
    - result_float: answer as float
    """
    print("\n[MAWPS] Processing...")
    dataset_id = "MU-NLPC/Calc-mawps"

    print("  Loading test split...")
    try:
        data = load_dataset(dataset_id, split="test")
    except Exception as e:
        print(f"  Could not load MAWPS dataset: {e}")
        return

    records = []
    for item in data:
        question = item.get("question", "")
        # Use result field for the answer
        answer = item.get("result", item.get("result_float", ""))

        if question:
            records.append({
                "instruction": TEST_SYSTEM_PROMPT,
                "input": question,
                "output": format_output(None, str(answer))
            })

    if records:
        save_json(records, output_dir / "mawps_test.json")
    else:
        print("  No records processed for MAWPS")


def process_scibench(output_dir: Path):
    """
    Process SciBench dataset with 80:20 split.

    SciBench format:
    - problem_text: problem description
    - solution: solution text
    - answer_number / answer_latex: answer
    - unit: unit of measurement
    """
    print("\n[SciBench] Processing...")
    dataset_id = "xw27/scibench"

    print("  Loading dataset...")
    try:
        data = load_dataset(dataset_id, split="train")
    except Exception as e:
        print(f"  Could not load SciBench dataset: {e}")
        return

    all_records = []
    for item in data:
        problem = item.get("problem_text", "")
        solution = item.get("solution", "")
        answer = item.get("answer_number", item.get("answer_latex", ""))
        unit = item.get("unit", "")

        # Include unit in answer if available
        if unit:
            answer_with_unit = f"{answer} {unit}"
        else:
            answer_with_unit = str(answer)

        if problem:
            all_records.append({
                "instruction": TEST_SYSTEM_PROMPT,  # Temporary, will update train split
                "input": problem,
                "output": format_output(solution if solution else None, answer_with_unit)
            })

    # 80:20 split
    if all_records:
        train_records, test_records = train_test_split(
            all_records,
            test_size=0.2,
            random_state=42
        )

        # Update train records to use train prompt
        for record in train_records:
            record["instruction"] = TRAIN_SYSTEM_PROMPT

        # test_records already have TEST_SYSTEM_PROMPT

        print(f"  Split: {len(train_records)} train, {len(test_records)} test")
        save_json(train_records, output_dir / "scibench_train.json")
        save_json(test_records, output_dir / "scibench_test.json")


def process_arc(output_dir: Path):
    """
    Process ARC dataset (ARC-Challenge and ARC-Easy combined).

    ARC format:
    - question: question text
    - choices: {"text": [...], "label": [...]}
    - answerKey: correct answer label
    """
    print("\n[ARC] Processing...")
    dataset_id = "allenai/ai2_arc"

    for split in ["train", "test"]:
        all_records = []

        for config in ["ARC-Challenge", "ARC-Easy"]:
            print(f"  Loading {config} {split} split...")
            try:
                data = load_dataset(dataset_id, config, split=split)

                for item in data:
                    question = item["question"]
                    choices_data = item["choices"]
                    answer_key = item["answerKey"]

                    # choices_data is {"text": [...], "label": [...]}
                    choices_text = choices_data["text"]
                    choices_labels = choices_data["label"]

                    # Format choices with labels
                    formatted_choices = []
                    for label, text in zip(choices_labels, choices_text):
                        formatted_choices.append(text)

                    # Build input with question and choices
                    input_text = question + "\n\n"
                    for label, text in zip(choices_labels, choices_text):
                        input_text += f"{label}. {text}\n"

                    # Use split-aware prompt
                    instruction = TRAIN_SYSTEM_PROMPT if split == "train" else TEST_SYSTEM_PROMPT

                    all_records.append({
                        "instruction": instruction,
                        "input": input_text.strip(),
                        "output": format_output(None, answer_key)
                    })

                print(f"    Loaded {len(data)} questions")
            except Exception as e:
                print(f"    Error loading {config}: {e}")

        if all_records:
            save_json(all_records, output_dir / f"arc_{split}.json")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 60)
    print("Dataset Preparation Script")
    print("=" * 60)

    math_dir = DATA_DIR / "math"
    knowledge_dir = DATA_DIR / "knowledge"

    math_dir.mkdir(parents=True, exist_ok=True)
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directories:")
    print(f"  Math: {math_dir}")
    print(f"  Knowledge: {knowledge_dir}")

    # ==========================================
    # Process Math Domain
    # ==========================================
    print("\n" + "=" * 60)
    print("MATH DOMAIN")
    print("=" * 60)

    # 1. GSM8K (train + test)
    process_gsm8k(math_dir)

    # 2. MATH (train + test)
    process_math(math_dir)

    # 3. MMLU Math subjects (test only)
    process_mmlu(math_dir, MMLU_MATH_SUBJECTS, "mmlu_test.json", is_math_domain=True)

    # 4. SVAMP (test only)
    process_svamp(math_dir)

    # 5. ASDiv (test only)
    process_asdiv(math_dir)

    # 6. MAWPS (test only)
    process_mawps(math_dir)

    # ==========================================
    # Process Knowledge Domain
    # ==========================================
    print("\n" + "=" * 60)
    print("KNOWLEDGE DOMAIN")
    print("=" * 60)

    # 1. SciBench (80:20 split)
    process_scibench(knowledge_dir)

    # 2. ARC (train + test)
    process_arc(knowledge_dir)

    # 3. MMLU Science subjects (test only)
    process_mmlu(knowledge_dir, MMLU_SCIENCE_SUBJECTS, "mmlu_test.json")

    print("\n" + "=" * 60)
    print("Dataset preparation completed!")
    print("=" * 60)

    # Print summary
    print("\n[Summary]")
    for domain_dir in [math_dir, knowledge_dir]:
        print(f"\n{domain_dir.name}/")
        for json_file in sorted(domain_dir.glob("*.json")):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"  {json_file.name}: {len(data)} records")


if __name__ == "__main__":
    main()
