#!/usr/bin/env python3
"""
Script to update instruction field in all train JSON files.
Updates from old prompt to new prompt with step-by-step instruction.
"""
import json
from pathlib import Path

OLD_PROMPT = "You are a helpful math assistant."
NEW_PROMPT = "You are a helpful math assistant.\nSolve the problem step by step and provide your final answer within \\boxed{}."

# Files to update
TRAIN_FILES = [
    "data/math/gsm8k_train.json",
    "data/math/math_train.json",
    "data/knowledge/scibench_train.json",
    "data/knowledge/arc_train.json",
]

def update_file(file_path: Path):
    """Update instruction field in a JSON file."""
    print(f"\nProcessing: {file_path}")

    if not file_path.exists():
        print(f"  File not found: {file_path}")
        return

    # Load data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Count updates
    updated_count = 0
    for item in data:
        if item.get("instruction") == OLD_PROMPT:
            item["instruction"] = NEW_PROMPT
            updated_count += 1

    # Save updated data
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  Updated {updated_count}/{len(data)} records")

def main():
    """Main entry point."""
    print("=" * 60)
    print("Updating train data instruction prompts")
    print("=" * 60)

    project_root = Path(__file__).parent

    for file_path_str in TRAIN_FILES:
        file_path = project_root / file_path_str
        update_file(file_path)

    print("\n" + "=" * 60)
    print("Update completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
