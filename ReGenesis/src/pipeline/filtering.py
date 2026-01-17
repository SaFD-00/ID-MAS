"""Data filtering module for ReGenesis.

Filters reasoning paths based on ground truth exact match.
Based on process_reason.py with improvements for multi-domain support.
"""

import json
import os
import re
import random
import argparse
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} format.

    Args:
        text: Text containing boxed answer

    Returns:
        Extracted answer or None if not found
    """
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return text[left_brace_idx + 1: right_brace_idx].strip()


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison.

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer
    """
    if answer is None:
        return ""

    # Handle leading decimal point
    if answer.startswith('.'):
        answer = '0' + answer

    # Remove formatting
    answer = answer.replace('$', '')
    answer = answer.replace(',', '')
    answer = answer.replace('%', '')

    # Handle decimal trailing zeros
    answer = answer.replace('.00', '')
    answer = answer.replace('.0', '')

    return answer.strip().lower()


def extract_numeric_answers(text: str) -> List[str]:
    """Extract all numeric values from text.

    Args:
        text: Text to extract numbers from

    Returns:
        List of normalized numeric strings
    """
    text = normalize_answer(text)
    text = text.replace('-', ' - ')  # Handle negative numbers

    try:
        numbers = re.findall(r'-?\d+\.?\d*', text)
        normalized = []
        for num in numbers:
            normalized.append(num)
            # Add version without trailing decimals
            clean = num.replace('.00', '').replace('.0', '')
            if clean != num:
                normalized.append(clean)
        return normalized
    except Exception:
        return []


def check_results_exact_match(
    solutions: List[str],
    truth: str,
    answer_type: str = "numeric",
) -> List[bool]:
    """Check solutions against ground truth using exact match.

    Args:
        solutions: List of solution texts
        truth: Ground truth answer
        answer_type: Type of answer ("numeric", "option", "text")

    Returns:
        List of boolean correctness values
    """
    truth_normalized = normalize_answer(truth)
    correctness = []

    for solution in solutions:
        is_correct = False
        boxed_answer = extract_boxed_answer(solution)

        if answer_type == "numeric":
            # Extract numbers and compare
            solution_answers = extract_numeric_answers(solution)
            boxed_answers = extract_numeric_answers(boxed_answer) if boxed_answer else []

            # Check if truth matches any extracted number
            for ans in boxed_answers + solution_answers[:2]:  # Prioritize boxed and first/last
                if normalize_answer(ans) == truth_normalized:
                    is_correct = True
                    break

        elif answer_type == "option":
            # For multiple choice (A, B, C, D)
            if boxed_answer:
                boxed_lower = boxed_answer.lower().strip()
                truth_lower = truth_normalized.lower().strip()

                # Direct match
                if boxed_lower == truth_lower:
                    is_correct = True
                # Check if option letter is in boxed answer
                elif truth_lower in boxed_lower:
                    is_correct = True

        else:  # text
            if boxed_answer and normalize_answer(boxed_answer) == truth_normalized:
                is_correct = True

        correctness.append(is_correct)

    return correctness


def filter_reasoning_paths(
    reasoning_data: List[Dict],
    answer_type: str = "numeric",
    max_paths_per_task: int = 5,
    max_token_length: int = 800,
) -> List[Dict]:
    """Filter reasoning paths based on correctness and quality.

    Args:
        reasoning_data: List of reasoning path entries
        answer_type: Type of answer for matching
        max_paths_per_task: Maximum paths to keep per task
        max_token_length: Maximum token length for paths

    Returns:
        Filtered list of correct reasoning paths
    """
    # Group by instruction
    instruction_groups = defaultdict(list)
    for entry in reasoning_data:
        instruction_groups[entry["instruction"]].append(entry)

    filtered_data = []

    for instruction, entries in tqdm(instruction_groups.items(), desc="Filtering"):
        if not entries:
            continue

        truth = entries[0]["answer"]
        solutions = [e["reasoning"] for e in entries]

        # Check correctness
        correctness = check_results_exact_match(solutions, truth, answer_type)

        # Filter correct entries
        correct_entries = []
        for entry, is_correct in zip(entries, correctness):
            if is_correct and "boxed" in entry["reasoning"].lower():
                correct_entries.append(entry)

        # Limit paths per task
        if len(correct_entries) > max_paths_per_task:
            correct_entries = random.sample(correct_entries, max_paths_per_task)

        filtered_data.extend(correct_entries)

    return filtered_data


def format_for_training(
    filtered_data: List[Dict],
    dataset_type: str = "math",
) -> List[Dict]:
    """Format filtered data for fine-tuning.

    Args:
        filtered_data: Filtered reasoning paths
        dataset_type: Type of dataset for instruction formatting

    Returns:
        List of training-ready entries
    """
    training_data = []

    for entry in filtered_data:
        # Clean up reasoning text
        reasoning = entry["reasoning"].strip()

        # Format reasoning with step markers
        lines = reasoning.split('\n\n')
        lines = [re.sub(r'\s+', ' ', line.strip()) for line in lines if line.strip()]

        # Find step-by-step section
        step_idx = None
        for i, line in enumerate(lines):
            if '1.' in line and '2.' in ''.join(lines[i:]):
                step_idx = i
                break

        if step_idx is not None:
            formatted_reasoning = "Let's think step by step.\n\n" + '\n\n'.join(lines[step_idx:])
        else:
            formatted_reasoning = reasoning

        training_entry = {
            "instruction": entry["instruction"],
            "answer": formatted_reasoning,
        }
        training_data.append(training_entry)

    return training_data


def process_and_filter_data(
    input_dir: str,
    output_dir: str,
    dataset_name: str,
    answer_type: str = "numeric",
    max_paths_per_task: int = 5,
) -> str:
    """Process and filter reasoning data for training.

    Args:
        input_dir: Directory containing generated reasoning paths
        output_dir: Output directory for filtered data
        dataset_name: Name of dataset
        answer_type: Type of answer for matching
        max_paths_per_task: Maximum paths per task

    Returns:
        Path to output file
    """
    # Find all jsonl files for dataset
    jsonl_files = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.jsonl') and dataset_name in filename:
            jsonl_files.append(os.path.join(input_dir, filename))

    if not jsonl_files:
        raise ValueError(f"No files found for dataset: {dataset_name}")

    # Load all data
    all_data = []
    for filepath in jsonl_files:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    all_data.append(entry)
                except json.JSONDecodeError:
                    continue

    print(f"Loaded {len(all_data)} entries from {len(jsonl_files)} files")

    # Filter data
    filtered_data = filter_reasoning_paths(
        all_data,
        answer_type=answer_type,
        max_paths_per_task=max_paths_per_task,
    )

    print(f"Filtered to {len(filtered_data)} correct entries")

    # Format for training
    training_data = format_for_training(filtered_data, dataset_name)

    # Shuffle
    random.shuffle(training_data)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}_filtered.json")

    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)

    print(f"Saved {len(training_data)} entries to: {output_path}")
    return output_path


def merge_filtered_datasets(
    filtered_dir: str,
    output_path: str,
    datasets: List[str] = None,
) -> str:
    """Merge multiple filtered datasets into one training file.

    Args:
        filtered_dir: Directory containing filtered json files
        output_path: Output path for merged file
        datasets: List of dataset names to merge (None for all)

    Returns:
        Path to merged file
    """
    merged_data = []

    for filename in os.listdir(filtered_dir):
        if not filename.endswith('_filtered.json'):
            continue

        if datasets:
            dataset_name = filename.replace('_filtered.json', '')
            if dataset_name not in datasets:
                continue

        filepath = os.path.join(filtered_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
            merged_data.extend(data)

    # Shuffle merged data
    random.shuffle(merged_data)

    # Save
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print(f"Merged {len(merged_data)} entries to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Filter reasoning paths")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory with generated reasoning paths",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for filtered data",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name to filter",
    )
    parser.add_argument(
        "--answer_type",
        type=str,
        default="numeric",
        choices=["numeric", "option", "text"],
        help="Answer type for matching",
    )
    parser.add_argument(
        "--max_paths",
        type=int,
        default=5,
        help="Maximum paths per task",
    )

    args = parser.parse_args()

    process_and_filter_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        answer_type=args.answer_type,
        max_paths_per_task=args.max_paths,
    )


if __name__ == "__main__":
    main()
