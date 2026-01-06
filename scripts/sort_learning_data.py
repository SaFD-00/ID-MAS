#!/usr/bin/env python3
"""
Learning Logs 파일명 마이그레이션 스크립트

기존 형식 (q{순번}): math_gsm8k_q1_loop.json, math_gsm8k_q2_loop.json, ...
새 형식 (q{train_idx}): math_gsm8k_q398_loop.json, math_gsm8k_q1267_loop.json, ...

KB 파일 내부의 question_id를 기준으로 파일명을 training index로 변환합니다.

사용법:
    # Dry-run (변경 내용 확인만)
    python scripts/sort_learning_data.py --dry-run

    # 실제 마이그레이션 실행 (백업 포함)
    python scripts/sort_learning_data.py

    # 특정 데이터셋만 마이그레이션
    python scripts/sort_learning_data.py --dataset math
    python scripts/sort_learning_data.py --dataset gsm8k
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from datetime import datetime


# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def extract_training_index(question_id: str) -> int:
    """
    question_id에서 training index 추출

    예: "math_train_1267" → 1267
        "gsm8k_train_2610" → 2610
    """
    match = re.search(r'_(\d+)$', question_id)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot extract index from question_id: {question_id}")


def extract_q_number(filename: str) -> int:
    """
    파일명에서 q번호 추출

    예: "math_math_q1_loop.json" → 1
        "math_gsm8k_q100_loop.json" → 100
    """
    match = re.search(r'_q(\d+)_loop\.json$', filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot extract q number from filename: {filename}")


def load_kb(kb_path: Path) -> dict:
    """Knowledge Base 파일 로드"""
    with open(kb_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_kb(kb_data: dict, kb_path: Path):
    """Knowledge Base 파일 저장"""
    with open(kb_path, 'w', encoding='utf-8') as f:
        json.dump(kb_data, f, ensure_ascii=False, indent=2)


def create_backup(source_dir: Path, backup_name: str = None):
    """백업 디렉토리 생성"""
    if backup_name is None:
        backup_name = f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    backup_dir = source_dir.parent / backup_name
    if backup_dir.exists():
        print(f"⚠️  Backup directory already exists: {backup_dir}")
        return backup_dir

    shutil.copytree(source_dir, backup_dir)
    print(f"📦 Backup created: {backup_dir}")
    return backup_dir


def migrate_dataset(
    domain: str,
    dataset: str,
    model_name: str = "Qwen3-4B-Instruct-2507",
    dry_run: bool = False,
    skip_backup: bool = False
):
    """
    특정 데이터셋의 learning logs를 새 형식으로 마이그레이션

    기존: q{순번} (q1, q2, q3, ...)
    새: q{train_idx} (q398, q1267, q5000, ...)

    Args:
        domain: 도메인 (예: "math")
        dataset: 데이터셋 (예: "math", "gsm8k")
        model_name: 모델 이름
        dry_run: True면 실제 변경 없이 출력만
        skip_backup: True면 백업 생성 안함
    """
    print(f"\n{'='*60}")
    print(f"Migrating {domain}/{dataset} ({model_name})")
    print(f"{'='*60}")

    # 경로 설정
    base_dir = DATA_DIR / domain / model_name / dataset
    logs_dir = base_dir / "learning_logs"
    kb_dir = base_dir / "knowledge_base"
    kb_path = kb_dir / f"{domain}_{dataset}_knowledge_base.json"

    # 파일 존재 확인
    if not logs_dir.exists():
        print(f"❌ Learning logs directory not found: {logs_dir}")
        return False

    if not kb_path.exists():
        print(f"❌ Knowledge base file not found: {kb_path}")
        return False

    # KB 로드
    print(f"\n📂 Loading KB from: {kb_path}")
    kb_data = load_kb(kb_path)
    entries = kb_data.get("learning_entries", [])

    if not entries:
        print("❌ No learning entries found in KB")
        return False

    print(f"   Found {len(entries)} entries")

    # 현재 로그 파일 목록
    log_files = list(logs_dir.glob(f"{domain}_{dataset}_q*_loop.json"))
    print(f"   Found {len(log_files)} log files")

    # 파일명에서 q번호 추출하여 entry와 매핑
    # 현재 KB entries는 q1, q2, ... 순서대로 저장되어 있음
    migrations = []
    already_migrated = 0

    identifier = f"{domain}_{dataset}"

    for i, entry in enumerate(entries):
        current_q = i + 1  # 1-indexed (기존 형식)
        question_id = entry.get("question_id", "")

        try:
            train_idx = extract_training_index(question_id)
        except ValueError as e:
            print(f"⚠️  Warning: {e}")
            continue

        old_file = logs_dir / f"{identifier}_q{current_q}_loop.json"
        new_file = logs_dir / f"{identifier}_q{train_idx}_loop.json"

        # 이미 새 형식으로 마이그레이션된 경우
        if new_file.exists() and not old_file.exists():
            already_migrated += 1
            if dry_run:
                print(f"⏭️  q{train_idx} already exists (train_idx={train_idx})")
            continue

        # 기존 파일이 존재하고 변환이 필요한 경우
        if old_file.exists():
            # current_q와 train_idx가 같으면 이미 올바른 형식
            if current_q == train_idx:
                already_migrated += 1
                if dry_run:
                    print(f"⏭️  q{current_q} = q{train_idx} (no change needed)")
                continue

            migrations.append({
                "old_q": current_q,
                "train_idx": train_idx,
                "question_id": question_id,
                "old_file": old_file,
                "new_file": new_file,
                "entry": entry
            })
            if dry_run:
                print(f"🔄 q{current_q} → q{train_idx} ({question_id})")

    print(f"\n📊 Summary:")
    print(f"   Total entries: {len(entries)}")
    print(f"   Already migrated/correct: {already_migrated}")
    print(f"   Need to migrate: {len(migrations)}")

    if not migrations:
        print("\n✅ All files are already in correct format!")
        # KB 정렬만 수행
        if not dry_run:
            print("\n📝 Sorting KB entries by question_id...")
            sorted_entries = sorted(
                entries,
                key=lambda x: extract_training_index(x.get("question_id", ""))
            )
            kb_data["learning_entries"] = sorted_entries
            kb_data["last_updated"] = datetime.now().isoformat()
            save_kb(kb_data, kb_path)
            print(f"   KB saved to: {kb_path}")
        return True

    if dry_run:
        print("\n🔍 Dry-run mode - no changes made")
        return True

    # 백업 생성
    if not skip_backup:
        print("\n📦 Creating backup...")
        create_backup(logs_dir, f"_backup_before_migrate_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # 파일명 변경 (충돌 방지를 위해 2단계로)
    print("\n🔄 Renaming files (Phase 1: to temp names)...")

    # Phase 1: 모든 파일을 임시 이름으로
    for m in migrations:
        temp_file = logs_dir / f"{identifier}_temp_{m['old_q']}_loop.json"
        if m["old_file"].exists():
            m["old_file"].rename(temp_file)
            m["temp_file"] = temp_file

    # Phase 2: 임시 이름을 새 이름으로
    print("🔄 Renaming files (Phase 2: to final names)...")

    for m in migrations:
        temp_file = m.get("temp_file")
        if temp_file and temp_file.exists():
            m["new_file"].parent.mkdir(parents=True, exist_ok=True)
            temp_file.rename(m["new_file"])
            print(f"   q{m['old_q']} → q{m['train_idx']}")

    # KB entries 정렬 (question_id 기준)
    print("\n📝 Sorting KB entries by question_id...")
    sorted_entries = sorted(
        entries,
        key=lambda x: extract_training_index(x.get("question_id", ""))
    )
    kb_data["learning_entries"] = sorted_entries
    kb_data["last_updated"] = datetime.now().isoformat()

    # KB 저장
    save_kb(kb_data, kb_path)
    print(f"   KB saved to: {kb_path}")

    print(f"\n✅ Migrated {len(migrations)} files, skipped {already_migrated} files")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate learning logs from q{순번} to q{train_idx} format"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["math", "gsm8k", "all"],
        default="all",
        help="Dataset to migrate (default: all)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen3-4B-Instruct-2507",
        help="Model name (default: Qwen3-4B-Instruct-2507)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Skip backup creation"
    )

    args = parser.parse_args()

    print("="*60)
    print("Learning Data Migration: q{순번} → q{train_idx}")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Dry-run: {args.dry_run}")

    datasets = ["math", "gsm8k"] if args.dataset == "all" else [args.dataset]

    success = True
    for dataset in datasets:
        result = migrate_dataset(
            domain="math",
            dataset=dataset,
            model_name=args.model,
            dry_run=args.dry_run,
            skip_backup=args.skip_backup
        )
        success = success and result

    print("\n" + "="*60)
    if success:
        print("✅ All done!")
    else:
        print("❌ Some errors occurred")
    print("="*60)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
