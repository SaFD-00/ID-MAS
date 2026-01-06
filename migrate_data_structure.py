#!/usr/bin/env python3
"""
Data Directory Migration Script

Migrates from old structure to new structure based on data_example/
Old: data/{domain}/{Model}/{dataset}/{eval_results|learning_logs|...}/
New: data/{domain}/train/{Model}/ and data/{domain}/eval/{Model}/
"""

import sys
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
BACKUP_DIR = PROJECT_ROOT / f"data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Training datasets per domain (from data_example)
TRAINING_DATASETS = {
    "math": ["gsm8k", "math"]
}

# Evaluation datasets per domain (from data_example)
EVAL_DATASETS = {
    "math": ["gsm8k", "math", "svamp", "asdiv", "mawps", "mmlu"]
}


def create_backup():
    """Create a backup of the current data directory."""
    print(f"\n{'='*80}")
    print("STEP 1: Creating backup...")
    print(f"{'='*80}")

    if DATA_DIR.exists():
        print(f"Backing up {DATA_DIR} to {BACKUP_DIR}...")
        shutil.copytree(DATA_DIR, BACKUP_DIR)
        print(f"✓ Backup created: {BACKUP_DIR}")
    else:
        print(f"✗ Data directory not found: {DATA_DIR}")


def get_models_in_domain(domain_dir: Path) -> List[str]:
    """Get list of model directories in a domain."""
    models = []
    if not domain_dir.exists():
        return models

    for item in domain_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Skip new structure directories and legacy folders
            if item.name not in ["train", "eval", "checkpoints", "design_outputs"]:
                models.append(item.name)

    return models


def migrate_raw_data_files(domain: str):
    """Migrate raw data files to train/data/ and eval/data/."""
    print(f"\n  Migrating raw data files for {domain}...")

    domain_dir = DATA_DIR / domain
    if not domain_dir.exists():
        print(f"    ✗ Domain directory not found: {domain_dir}")
        return

    # Training data
    train_data_dir = domain_dir / "train" / "data"
    train_data_dir.mkdir(parents=True, exist_ok=True)

    for dataset in TRAINING_DATASETS.get(domain, []):
        old_file = domain_dir / f"{dataset}_train.json"
        new_file = train_data_dir / f"{dataset}_train.json"

        if old_file.exists() and not new_file.exists():
            shutil.move(str(old_file), str(new_file))
            print(f"    ✓ Moved {old_file.name} → {new_file.relative_to(DATA_DIR)}")

    # Evaluation data
    eval_data_dir = domain_dir / "eval" / "data"
    eval_data_dir.mkdir(parents=True, exist_ok=True)

    for dataset in EVAL_DATASETS.get(domain, []):
        old_file = domain_dir / f"{dataset}_test.json"
        new_file = eval_data_dir / f"{dataset}_test.json"

        if old_file.exists() and not new_file.exists():
            shutil.move(str(old_file), str(new_file))
            print(f"    ✓ Moved {old_file.name} → {new_file.relative_to(DATA_DIR)}")


def migrate_design_outputs(domain: str):
    """Migrate design outputs to train/instructional-design/."""
    print(f"\n  Migrating design outputs for {domain}...")

    domain_dir = DATA_DIR / domain
    old_design_dir = domain_dir / "design_outputs"
    new_design_dir = domain_dir / "train" / "instructional-design"

    new_design_dir.mkdir(parents=True, exist_ok=True)

    if old_design_dir.exists():
        for file in old_design_dir.glob("*.json"):
            new_file = new_design_dir / file.name
            if not new_file.exists():
                shutil.move(str(file), str(new_file))
                print(f"    ✓ Moved {file.name} → {new_file.relative_to(DATA_DIR)}")

        # Remove empty directory
        if not any(old_design_dir.iterdir()):
            old_design_dir.rmdir()
            print(f"    ✓ Removed empty directory: design_outputs/")


def migrate_checkpoints(domain: str, models: List[str]):
    """Migrate checkpoints to train/{Model}/."""
    print(f"\n  Migrating checkpoints for {domain}...")

    domain_dir = DATA_DIR / domain
    checkpoint_dir = domain_dir / "checkpoints"

    if not checkpoint_dir.exists():
        print(f"    ✗ No checkpoints directory found")
        return

    # Group checkpoints by pattern: checkpoint_{dataset}_{timestamp}.json
    for checkpoint_file in checkpoint_dir.glob("checkpoint_*.json"):
        # Parse filename to determine which model it belongs to
        # For now, we can't determine the model, so we'll skip this
        # Users may need to manually organize these or we could keep them in a legacy folder
        print(f"    ! Checkpoint file needs manual migration: {checkpoint_file.name}")

    # Note: Checkpoints need to be manually organized or we need more metadata
    print(f"    ! Checkpoint migration requires manual intervention or metadata")


def migrate_model_data(domain: str, model: str):
    """Migrate a single model's data from old to new structure."""
    print(f"\n  Migrating model: {model}")

    domain_dir = DATA_DIR / domain
    old_model_dir = domain_dir / model

    if not old_model_dir.exists():
        return

    # Process each dataset folder under the model
    for dataset_dir in old_model_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name

        # Determine if this is training or evaluation data
        is_training = dataset in TRAINING_DATASETS.get(domain, [])
        is_eval = dataset in EVAL_DATASETS.get(domain, [])

        # Migrate evaluation results
        eval_results_dir = dataset_dir / "eval_results"
        if eval_results_dir.exists():
            new_eval_dir = domain_dir / "eval" / model
            new_eval_dir.mkdir(parents=True, exist_ok=True)

            for eval_file in eval_results_dir.glob("*.json"):
                new_file = new_eval_dir / eval_file.name
                if not new_file.exists():
                    shutil.move(str(eval_file), str(new_file))
                    print(f"    ✓ Eval: {eval_file.name} → {new_file.relative_to(DATA_DIR)}")

        # Migrate SFT data (training output)
        sft_data_dir = dataset_dir / "sft_data"
        if sft_data_dir.exists() and is_training:
            new_train_dir = domain_dir / "train" / model
            new_train_dir.mkdir(parents=True, exist_ok=True)

            for sft_file in sft_data_dir.glob("*.json"):
                # Rename according to new convention: {dataset}_train_id-mas_{Model}.json
                new_filename = f"{dataset}_train_id-mas_{model}.json"
                new_file = new_train_dir / new_filename

                if not new_file.exists():
                    shutil.move(str(sft_file), str(new_file))
                    print(f"    ✓ SFT: {sft_file.name} → {new_file.relative_to(DATA_DIR)}")

        # Archive or remove learning_logs and knowledge_base (not in new structure)
        for subdir in ["learning_logs", "knowledge_base"]:
            subdir_path = dataset_dir / subdir
            if subdir_path.exists() and any(subdir_path.iterdir()):
                # Move to archive instead of deleting
                archive_dir = domain_dir / "archive" / model / dataset / subdir
                archive_dir.mkdir(parents=True, exist_ok=True)

                for file in subdir_path.glob("*"):
                    archive_file = archive_dir / file.name
                    if not archive_file.exists():
                        shutil.move(str(file), str(archive_file))

                print(f"    ✓ Archived: {dataset}/{subdir}/ → archive/{model}/{dataset}/{subdir}/")

    # Remove empty old model directory structure
    cleanup_empty_dirs(old_model_dir)


def cleanup_empty_dirs(directory: Path):
    """Recursively remove empty directories."""
    if not directory.exists():
        return

    for item in directory.iterdir():
        if item.is_dir():
            cleanup_empty_dirs(item)

    # Remove if empty
    try:
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()
            print(f"    ✓ Removed empty directory: {directory.relative_to(DATA_DIR)}")
    except OSError:
        pass


def migrate_domain(domain: str):
    """Migrate all data for a specific domain."""
    print(f"\n{'='*80}")
    print(f"Migrating domain: {domain.upper()}")
    print(f"{'='*80}")

    domain_dir = DATA_DIR / domain
    if not domain_dir.exists():
        print(f"  ✗ Domain directory not found: {domain_dir}")
        return

    # Get list of models in this domain
    models = get_models_in_domain(domain_dir)
    print(f"  Found {len(models)} models: {', '.join(models)}")

    # Step 1: Migrate raw data files
    migrate_raw_data_files(domain)

    # Step 2: Migrate design outputs
    migrate_design_outputs(domain)

    # Step 3: Migrate checkpoints
    migrate_checkpoints(domain, models)

    # Step 4: Migrate each model's data
    for model in models:
        migrate_model_data(domain, model)

    print(f"\n✓ Completed migration for domain: {domain}")


def verify_migration():
    """Verify the migration by checking new structure."""
    print(f"\n{'='*80}")
    print("VERIFICATION: Checking new structure...")
    print(f"{'='*80}")

    for domain in ["math"]:
        domain_dir = DATA_DIR / domain
        if not domain_dir.exists():
            continue

        print(f"\n{domain.upper()}:")

        # Check train structure
        train_dir = domain_dir / "train"
        if train_dir.exists():
            print(f"  train/")

            data_dir = train_dir / "data"
            if data_dir.exists():
                files = list(data_dir.glob("*.json"))
                print(f"    data/ ({len(files)} files)")
                for f in files:
                    print(f"      - {f.name}")

            design_dir = train_dir / "instructional-design"
            if design_dir.exists():
                files = list(design_dir.glob("*.json"))
                print(f"    instructional-design/ ({len(files)} files)")
                for f in files:
                    print(f"      - {f.name}")

            # Model directories
            for model_dir in sorted(train_dir.iterdir()):
                if model_dir.is_dir() and model_dir.name not in ["data", "instructional-design"]:
                    files = list(model_dir.glob("*.json"))
                    print(f"    {model_dir.name}/ ({len(files)} files)")
                    for f in sorted(files):
                        print(f"      - {f.name}")

        # Check eval structure
        eval_dir = domain_dir / "eval"
        if eval_dir.exists():
            print(f"  eval/")

            data_dir = eval_dir / "data"
            if data_dir.exists():
                files = list(data_dir.glob("*.json"))
                print(f"    data/ ({len(files)} files)")
                for f in files:
                    print(f"      - {f.name}")

            # Model directories
            for model_dir in sorted(eval_dir.iterdir()):
                if model_dir.is_dir() and model_dir.name != "data":
                    files = list(model_dir.glob("*.json"))
                    print(f"    {model_dir.name}/ ({len(files)} files)")


def main():
    """Main migration function."""
    print(f"\n{'#'*80}")
    print("DATA DIRECTORY MIGRATION SCRIPT")
    print(f"{'#'*80}")
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Backup Directory: {BACKUP_DIR}")

    # Check for --yes flag for automatic execution
    auto_yes = "--yes" in sys.argv or "-y" in sys.argv

    if not auto_yes:
        # Confirm before proceeding
        print(f"\n{'!'*80}")
        print("WARNING: This script will restructure your data directory.")
        print("A backup will be created before any changes are made.")
        print(f"{'!'*80}")

        response = input("\nProceed with migration? (yes/no): ").strip().lower()
        if response not in ["yes", "y"]:
            print("\n✗ Migration cancelled.")
            return
    else:
        print(f"\n{'!'*80}")
        print("WARNING: This script will restructure your data directory.")
        print("A backup will be created before any changes are made.")
        print("Running in auto-yes mode (--yes flag detected)")
        print(f"{'!'*80}")

    # Step 1: Create backup
    create_backup()

    # Step 2: Migrate math domain
    migrate_domain("math")

    # Step 3: Verify migration
    verify_migration()

    print(f"\n{'#'*80}")
    print("MIGRATION COMPLETE")
    print(f"{'#'*80}")
    print(f"\nBackup location: {BACKUP_DIR}")
    print("If everything looks correct, you can delete the backup directory.")
    print("\nNOTE: Checkpoints may require manual organization.")
    print("Archived files (learning_logs, knowledge_base) are in data/{domain}/archive/")


if __name__ == "__main__":
    main()
