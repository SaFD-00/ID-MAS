#!/usr/bin/env python3
"""
Dataset Preparation Script
Downloads and processes datasets from HuggingFace for math and knowledge domains.

Output format:
[
  {
    "instruction": "system prompt (same for train and test)",
    "input": "question text",
    "output": "{reasoning}\n\n\\boxed{answer}" or "\\boxed{answer}"
  },
  ...
]

System Prompts:
- Prompts are dataset-specific and reused for both train and test splits.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
RANDOM_SEED = 42

# Dataset-specific system prompts
DATASET_PROMPTS = {
    # GSM8K - Grade school math, numeric answers
    "gsm8k": """You are a helpful math assistant.
Solve this grade-school math problem step by step. Show your calculations clearly using <<calculation=result>> format for each step.
Your final answer MUST be a single number within \\boxed{}.
Example: \\boxed{42}""",

    # MATH - Advanced math, LaTeX answers
    "math": """You are a helpful math assistant.
Solve this mathematical problem step by step. Show your reasoning clearly and use proper mathematical notation.
Your final answer MUST be within \\boxed{}. Use LaTeX notation for fractions (\\frac{a}{b}), exponents, and other mathematical expressions.
Example: \\boxed{\\frac{1}{8}} or \\boxed{2\\sqrt{3}}""",

    # SVAMP - Simple math word problems, numeric answers
    "svamp": """You are a helpful math assistant.
Solve this math word problem step by step. Identify the key information, set up the calculation, and solve.
Your final answer MUST be a single number within \\boxed{}.
Example: \\boxed{27}""",

    # ASDiv - Arithmetic word problems, numeric answers
    "asdiv": """You are a helpful math assistant.
Solve this arithmetic word problem step by step. Extract the relevant numbers, determine the operation(s) needed, and calculate.
Your final answer MUST be a single number within \\boxed{}.
Example: \\boxed{15}""",

    # MAWPS - Math word problems, may include fractions
    "mawps": """You are a helpful math assistant.
Solve this math word problem step by step. Show your work clearly.
Your final answer MUST be within \\boxed{}. If the answer is a fraction, write it as a/b or use \\frac{a}{b}.
Example: \\boxed{49} or \\boxed{56/9}""",

    # Logical domain
    "reclor": """You are a logical reasoning assistant. Read the passage and question, then select the correct option (A, B, C, or D). Your final answer MUST be a single letter within \\boxed{}.
Example: \\boxed{A}""",

    "anli": """You are a natural language inference assistant. Determine the relationship between the premise and hypothesis. Choose from: A. entailment, B. neutral, C. contradiction. Your final answer MUST be a single letter within \\boxed{}.
Example: \\boxed{A}""",

    # Commonsense domain
    "arc_c": """You are a helpful commonsense science assistant. Solve the problem and select the correct option (A, B, C, or D). Your final answer MUST be a single letter within \\boxed{}.
Example: \\boxed{A}""",

    "strategyqa": """You are a helpful commonsense reasoning assistant. Answer the question with Yes or No based on reliable commonsense knowledge. Your final answer MUST be \\boxed{Yes} or \\boxed{No}.
Example: \\boxed{Yes}""",

    "openbookqa": """You are a helpful science question-answering assistant. Use the given options and choose the best answer (A, B, C, or D). Your final answer MUST be a single letter within \\boxed{}.
Example: \\boxed{A}""",
}

# BBH subtask-specific prompts
BBH_PROMPTS = {
    "boolean_expressions": """You are a helpful reasoning assistant. Evaluate the boolean expression and answer with True or False. Your final answer MUST be \\boxed{True} or \\boxed{False}.
Example: \\boxed{True}""",

    "formal_fallacies": """You are a helpful reasoning assistant. Determine if the argument is valid or invalid. Your final answer MUST be \\boxed{valid} or \\boxed{invalid}.
Example: \\boxed{valid}""",

    "logical_deduction_three_objects": """You are a helpful reasoning assistant. Solve the logical deduction problem and select the correct option (A, B, or C). Your final answer MUST be a single letter within \\boxed{}.
Example: \\boxed{A}""",

    "logical_deduction_five_objects": """You are a helpful reasoning assistant. Solve the logical deduction problem and select the correct option. Your final answer MUST be a single letter within \\boxed{}.
Example: \\boxed{A}""",

    "logical_deduction_seven_objects": """You are a helpful reasoning assistant. Solve the logical deduction problem and select the correct option. Your final answer MUST be a single letter within \\boxed{}.
Example: \\boxed{A}""",

    "tracking_shuffled_objects_three_objects": """You are a helpful reasoning assistant. Track the positions of the shuffled objects and select the correct option. Your final answer MUST be a single letter within \\boxed{}.
Example: \\boxed{A}""",

    "tracking_shuffled_objects_five_objects": """You are a helpful reasoning assistant. Track the positions of the shuffled objects and select the correct option. Your final answer MUST be a single letter within \\boxed{}.
Example: \\boxed{A}""",

    "tracking_shuffled_objects_seven_objects": """You are a helpful reasoning assistant. Track the positions of the shuffled objects and select the correct option. Your final answer MUST be a single letter within \\boxed{}.
Example: \\boxed{A}""",

    "web_of_lies": """You are a helpful reasoning assistant. Determine the truth value based on the web of lies. Your final answer MUST be \\boxed{Yes} or \\boxed{No}.
Example: \\boxed{Yes}""",

    "default_mcq": """You are a helpful reasoning assistant. Solve the problem and select the correct option. Your final answer MUST be a single letter within \\boxed{}.
Example: \\boxed{A}""",
}

# BBH logical reasoning subtasks
BBH_LOGICAL_SUBTASKS = [
    "boolean_expressions",
    "formal_fallacies",
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "web_of_lies",
]


# =============================================================================
# Helper Functions
# =============================================================================

def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed for reproducible dataset processing."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import datasets as hf_datasets
        if hasattr(hf_datasets, "set_seed"):
            hf_datasets.set_seed(seed)
    except Exception:
        pass

def format_output(answer: str, reasoning: Optional[str] = None, include_reasoning: bool = False) -> str:
    """
    Format output with "The answer is \\boxed{answer}" format.

    Args:
        answer: Final answer
        reasoning: Step-by-step reasoning (optional)
        include_reasoning: If True, include reasoning before answer

    Returns:
        Formatted output string with "The answer is \\boxed{answer}" format
    """
    # Format answer in \boxed{} - escape backslash for proper string formatting
    boxed_answer = f"\\boxed{{{answer}}}"
    final_answer = f"The answer is {boxed_answer}"

    if include_reasoning and reasoning and reasoning.strip():
        return f"{reasoning.strip()}\n\n{final_answer}"
    return final_answer


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

def process_gsm8k(train_dir: Path, eval_dir: Path):
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

        records_short = []  # Without reasoning
        records_full = []   # With reasoning

        for item in data:
            question = item["question"]
            answer_text = item["answer"]

            # GSM8K answer format: "reasoning\n#### number"
            if "####" in answer_text:
                parts = answer_text.split("####")
                reasoning = parts[0].strip()
                final_answer = parts[1].strip()
            else:
                reasoning = ""
                final_answer = answer_text

            # Short version: "The answer is \boxed{answer}"
            records_short.append({
                "instruction": DATASET_PROMPTS["gsm8k"],
                "input": question,
                "output": format_output(final_answer)
            })

            # Full version: "{reasoning}\n\nThe answer is \boxed{answer}"
            records_full.append({
                "instruction": DATASET_PROMPTS["gsm8k"],
                "input": question,
                "output": format_output(final_answer, reasoning=reasoning, include_reasoning=True)
            })

        output_base = train_dir if split == "train" else eval_dir
        save_json(records_short, output_base / f"gsm8k_{split}.json")
        save_json(records_full, output_base / f"gsm8k_reasoning_{split}.json")


def process_math(train_dir: Path, eval_dir: Path):
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
        records_short = []  # Without reasoning
        records_full = []   # With reasoning
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

                    # Short version: "The answer is \boxed{answer}"
                    records_short.append({
                        "instruction": DATASET_PROMPTS["math"],
                        "input": problem,
                        "output": format_output(boxed_answer)
                    })

                    # Full version: "{solution}\n\nThe answer is \boxed{answer}"
                    records_full.append({
                        "instruction": DATASET_PROMPTS["math"],
                        "input": problem,
                        "output": format_output(boxed_answer, reasoning=solution, include_reasoning=True)
                    })
            except Exception as e:
                print(f"      Error loading {config}: {e}")

        print(f"  Level distribution: {level_counts}")
        output_base = train_dir if split == "train" else eval_dir
        save_json(records_short, output_base / f"math_{split}.json")
        save_json(records_full, output_base / f"math_reasoning_{split}.json")


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
            "instruction": DATASET_PROMPTS["svamp"],
            "input": full_question,
            "output": format_output(answer)
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
                "instruction": DATASET_PROMPTS["asdiv"],
                "input": full_question,
                "output": format_output(answer_clean)
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
                "instruction": DATASET_PROMPTS["mawps"],
                "input": question,
                "output": format_output(str(answer))
            })

    if records:
        save_json(records, output_dir / "mawps_test.json")
    else:
        print("  No records processed for MAWPS")


def process_reclor(train_dir: Path, eval_dir: Path):
    """
    Process ReClor dataset from local JSON files.

    Local files location: .claude/references/data/reclor_data/
    - train.json
    - val.json
    - test.json

    Local JSON format:
    - context: passage text
    - question: question text
    - answers: list of 4 choices
    - label: answer index (0-3)
    - id_string: unique identifier
    """
    print("\n[ReClor - Local] Processing...")

    # 로컬 데이터 경로
    local_data_dir = Path(__file__).parent.parent / ".claude" / "references" / "data" / "reclor_data"

    # 파일 매핑: split -> (파일명, 출력 디렉토리)
    split_mapping = {
        "train": ("train.json", train_dir),
        "val": ("val.json", eval_dir),
        "test": ("test.json", eval_dir)
    }

    for split_name, (filename, output_dir) in split_mapping.items():
        json_path = local_data_dir / filename

        if not json_path.exists():
            print(f"  Warning: {json_path} not found, skipping {split_name}")
            continue

        print(f"  Loading {split_name} from {filename}...")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        records = []
        for item in data:
            context = item.get("context", "")
            question = item.get("question", "")
            answers = item.get("answers", [])
            label = item.get("label", 0)

            # 사용자 지정 형식으로 input 구성
            input_text = f"Context:\n{context}\nQuestion: {question}\nOptions:\n"
            for i, answer in enumerate(answers):
                input_text += f"{chr(65 + i)}. {answer}\n"

            # 정답 레터 (0-3 -> A-D)
            answer_letter = chr(65 + label)

            records.append({
                "instruction": DATASET_PROMPTS["reclor"],
                "input": input_text.strip(),
                "output": format_output(answer_letter)
            })

        # 파일명 결정
        if split_name == "train":
            output_file = "reclor_train.json"
        elif split_name == "val":
            output_file = "reclor_val.json"
        else:
            output_file = "reclor_test.json"

        save_json(records, output_dir / output_file)
        print(f"  Saved {len(records)} records to {output_file}")


def process_arc_c(train_dir: Path, eval_dir: Path):
    """
    Process ARC-Challenge dataset.

    ARC format:
    - question: question text
    - choices: {"text": [...], "label": [...]}
    - answerKey: answer label (e.g., "A", "1")
    """
    print("\n[ARC-Challenge] Processing...")
    dataset_id = "allenai/ai2_arc"
    config = "ARC-Challenge"

    for split in ["train", "test"]:
        print(f"  Loading {split} split...")
        data = load_dataset(dataset_id, config, split=split)

        records = []
        for item in data:
            question = item["question"]
            choices = item["choices"]
            answer_key = item["answerKey"]

            # Extract choice texts
            choice_texts = choices["text"]

            # Format MCQ
            question_with_choices = format_mcq_input(question, choice_texts)

            # Convert answerKey to A-D (1->A, 2->B or A->A)
            if answer_key.isdigit():
                answer_letter = chr(65 + int(answer_key) - 1)
            else:
                answer_letter = answer_key

            records.append({
                "instruction": DATASET_PROMPTS["arc_c"],
                "input": question_with_choices,
                "output": format_output(answer_letter)
            })

        output_base = train_dir if split == "train" else eval_dir
        save_json(records, output_base / f"arc_c_{split}.json")


def process_strategyqa(eval_dir: Path):
    """
    Process StrategyQA dataset (test only).

    StrategyQA format:
    - question: yes/no question
    - answer: boolean (true/false)
    """
    print("\n[StrategyQA] Processing...")
    dataset_id = "ChilleD/StrategyQA"

    print("  Loading test split...")
    data = load_dataset(dataset_id, split="test")

    records = []
    for item in data:
        question = item["question"]
        answer_bool = item["answer"]

        # Convert boolean to Yes/No
        answer_text = "Yes" if answer_bool else "No"

        records.append({
            "instruction": DATASET_PROMPTS["strategyqa"],
            "input": question,
            "output": format_output(answer_text)
        })

    save_json(records, eval_dir / "strategyqa_test.json")


def process_openbookqa(eval_dir: Path):
    """
    Process OpenBookQA dataset.

    OpenBookQA format:
    - question_stem: question text
    - choices: {"text": [...], "label": [...]}
    - answerKey: answer label (e.g., "A")
    """
    print("\n[OpenBookQA] Processing...")
    dataset_id = "allenai/openbookqa"
    config = "main"

    print("  Loading test split...")
    data = load_dataset(dataset_id, config, split="test")

    records = []
    for item in data:
        question = item["question_stem"]
        choices = item["choices"]
        answer_key = item["answerKey"]

        # Extract choice texts
        choice_texts = choices["text"]

        # Format MCQ
        question_with_choices = format_mcq_input(question, choice_texts)

        records.append({
            "instruction": DATASET_PROMPTS["openbookqa"],
            "input": question_with_choices,
            "output": format_output(answer_key)
        })

    save_json(records, eval_dir / "openbookqa_test.json")


def process_anli(eval_dir: Path, round_name: str):
    """
    Process ANLI dataset for a specific round.

    ANLI format:
    - premise: premise text
    - hypothesis: hypothesis text
    - label: 0 (entailment), 1 (neutral), 2 (contradiction)

    Args:
        eval_dir: Output directory
        round_name: "r2" or "r3"
    """
    print(f"\n[ANLI-{round_name.upper()}] Processing...")
    dataset_id = "facebook/anli"

    split_name = f"test_{round_name}"
    print(f"  Loading {split_name} split...")
    data = load_dataset(dataset_id, split=split_name)

    records = []
    label_map = {0: "A", 1: "B", 2: "C"}

    for item in data:
        premise = item["premise"]
        hypothesis = item["hypothesis"]
        label = item["label"]

        # Input format: Premise + Hypothesis + choices
        input_text = f"""Premise: {premise}
Hypothesis: {hypothesis}

A. entailment
B. neutral
C. contradiction"""

        answer_letter = label_map[label]

        records.append({
            "instruction": DATASET_PROMPTS["anli"],
            "input": input_text,
            "output": format_output(answer_letter)
        })

    save_json(records, eval_dir / f"anli_{round_name}_test.json")


def process_bbh(eval_dir: Path, subtasks: List[str]):
    """
    Process BBH dataset for specific subtasks.

    BBH format varies by subtask:
    - input: question text
    - target: answer text

    Args:
        eval_dir: Output directory
        subtasks: List of subtask names
    """
    print("\n[BBH] Processing...")
    dataset_id = "lukaemon/bbh"

    all_records = []  # 모든 subtask의 레코드를 저장할 리스트

    for subtask in subtasks:
        print(f"  Loading subtask: {subtask}...")
        try:
            data = load_dataset(dataset_id, subtask, split="test")

            # Select prompt based on subtask
            prompt = BBH_PROMPTS.get(subtask, BBH_PROMPTS["default_mcq"])

            for item in data:
                input_text = item["input"]
                target = item["target"]

                all_records.append({
                    "instruction": prompt,
                    "input": input_text,
                    "output": format_output(target)
                })

        except Exception as e:
            print(f"    Error loading {subtask}: {e}")

    # 단일 파일로 저장
    save_json(all_records, eval_dir / "bbh_test.json")
    print(f"  Saved {len(all_records)} total records to bbh_test.json")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    set_random_seed()
    print("=" * 60)
    print("Dataset Preparation Script")
    print("=" * 60)

    math_dir = DATA_DIR / "math"
    train_dir = math_dir / "train" / "data"
    eval_dir = math_dir / "eval" / "data"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory (train): {train_dir}")
    print(f"Output directory (eval): {eval_dir}")

    # ==========================================
    # Process Math Domain
    # ==========================================
    print("\n" + "=" * 60)
    print("MATH DOMAIN")
    print("=" * 60)

    # 1. GSM8K (train + test)
    process_gsm8k(train_dir, eval_dir)

    # 2. MATH (train + test)
    process_math(train_dir, eval_dir)

    # 3. SVAMP (test only)
    process_svamp(eval_dir)

    # 4. ASDiv (test only)
    process_asdiv(eval_dir)

    # 5. MAWPS (test only)
    process_mawps(eval_dir)

    # ==========================================
    # Process Logical Domain
    # ==========================================
    print("\n" + "=" * 60)
    print("LOGICAL DOMAIN")
    print("=" * 60)

    logical_dir = DATA_DIR / "logical"
    logical_train_dir = logical_dir / "train" / "data"
    logical_eval_dir = logical_dir / "eval" / "data"
    logical_train_dir.mkdir(parents=True, exist_ok=True)
    logical_eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. ReClor (train + test)
    process_reclor(logical_train_dir, logical_eval_dir)

    # 2. ANLI R2, R3 (test only)
    process_anli(logical_eval_dir, "r2")
    process_anli(logical_eval_dir, "r3")

    # 3. BBH logical subtasks (test only)
    process_bbh(logical_eval_dir, BBH_LOGICAL_SUBTASKS)

    # ==========================================
    # Process Commonsense Domain
    # ==========================================
    print("\n" + "=" * 60)
    print("COMMONSENSE DOMAIN")
    print("=" * 60)

    commonsense_dir = DATA_DIR / "commonsense"
    commonsense_train_dir = commonsense_dir / "train" / "data"
    commonsense_eval_dir = commonsense_dir / "eval" / "data"
    commonsense_train_dir.mkdir(parents=True, exist_ok=True)
    commonsense_eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. ARC-Challenge (train + test)
    process_arc_c(commonsense_train_dir, commonsense_eval_dir)

    # 2. StrategyQA (test only)
    process_strategyqa(commonsense_eval_dir)

    # 3. OpenBookQA (test only)
    process_openbookqa(commonsense_eval_dir)

    # ==========================================
    # Print Summary
    # ==========================================
    print("\n" + "=" * 60)
    print("Dataset preparation completed!")
    print("=" * 60)

    print("\n[Summary]")
    for domain_dir in [math_dir, logical_dir, commonsense_dir]:
        if not domain_dir.exists():
            continue
        print(f"\n{domain_dir.name}/")
        for section_name, section_dir in [("train/data", domain_dir / "train" / "data"),
                                           ("eval/data", domain_dir / "eval" / "data")]:
            if not section_dir.exists():
                continue
            print(f"  {section_name}/")
            for json_file in sorted(section_dir.glob("*.json")):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"    {json_file.name}: {len(data)} records")


if __name__ == "__main__":
    main()
