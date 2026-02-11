"""기존 로그 파일의 skip statistics를 재계산하는 마이그레이션 스크립트.

버그: is_skipped=True인 결과의 skip_details에서 step별 카운트가 누락됨.
수정: skip_details를 기반으로 step별 skip 통계를 재계산.

Usage:
    python scripts/migrate_skip_statistics.py                    # dry-run (변경 없이 확인만)
    python scripts/migrate_skip_statistics.py --apply            # 실제 적용
"""

import argparse
import json
import shutil
from pathlib import Path


def recalculate_skip_statistics(logs: dict) -> dict:
    """로그의 scaffolding_results에서 skip 통계를 재계산합니다."""
    step1_skip = 0
    step2_skip = 0
    step3_skip = 0
    step4_skip = 0
    step5_skip = 0
    step5_case_b_skip = 0
    step5_case_c_skip = 0
    step5_summarization_skip = 0
    skipped_count = 0

    for result in logs.get("scaffolding_results", []):
        skip_details = result.get("skip_details", {})

        if result.get("is_skipped", False):
            skipped_count += 1
            # is_skipped=True 결과: skip_details 형식
            if skip_details.get("step2_performance_objectives_evaluation", {}).get("is_fallback"):
                step2_skip += 1
            if skip_details.get("step3_scaffolding_artifact_generation", {}).get("is_fallback"):
                step3_skip += 1
            if skip_details.get("step5_case_c_final_solution", {}).get("is_fallback"):
                step5_skip += 1
                step5_case_c_skip += 1
        else:
            # 정상 결과: skip_details에서 non-skip fallback 체크
            if skip_details.get("step2_performance_objectives_evaluation", {}).get("is_fallback"):
                step2_skip += 1
            if skip_details.get("step3_scaffolding_artifact_generation", {}).get("is_fallback"):
                step3_skip += 1
            if skip_details.get("step5_case_c_final_solution", {}).get("is_fallback"):
                step5_skip += 1
                step5_case_c_skip += 1

    processed = logs.get("statistics", {}).get("scaffolding_processed", 0)

    return {
        "step1_initial_response": {
            "count": step1_skip,
            "rate": step1_skip / processed if processed > 0 else 0,
        },
        "step2_evaluation": {
            "count": step2_skip,
            "rate": step2_skip / processed if processed > 0 else 0,
        },
        "step3_scaffolding": {
            "count": step3_skip,
            "rate": step3_skip / processed if processed > 0 else 0,
        },
        "step4_reresponse": {
            "count": step4_skip,
            "rate": step4_skip / processed if processed > 0 else 0,
        },
        "step5_reconstruction": {
            "count": step5_skip,
            "rate": step5_skip / processed if processed > 0 else 0,
            "case_b": step5_case_b_skip,
            "case_c": step5_case_c_skip,
            "summarization": step5_summarization_skip,
        },
        "analysis": {
            "count": skipped_count,
            "rate": skipped_count / processed if processed > 0 else 0,
        },
    }


def migrate_log_file(log_path: Path, apply: bool = False) -> bool:
    """단일 로그 파일의 skip 통계를 마이그레이션합니다."""
    print(f"\n{'='*60}")
    print(f"File: {log_path}")

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  -> ERROR: JSON 파싱 실패 - {e}")
        return False

    if "statistics" not in logs or "skip" not in logs.get("statistics", {}):
        print("  -> SKIP: statistics.skip 없음")
        return False

    old_skip = logs["statistics"]["skip"]
    new_skip = recalculate_skip_statistics(logs)

    # 변경사항 비교
    changed = False
    for key in new_skip:
        old_val = old_skip.get(key, {})
        new_val = new_skip[key]
        if old_val != new_val:
            changed = True
            print(f"  {key}:")
            print(f"    before: {old_val}")
            print(f"    after:  {new_val}")

    if not changed:
        print("  -> 변경 없음")
        return False

    if apply:
        backup_path = log_path.with_suffix(".json.bak")
        shutil.copy2(log_path, backup_path)
        print(f"  -> Backup: {backup_path}")

        logs["statistics"]["skip"] = new_skip
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        print(f"  -> APPLIED")
    else:
        print(f"  -> DRY-RUN (--apply 로 실행하면 적용)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate skip statistics in log files")
    parser.add_argument("--apply", action="store_true", help="실제로 파일을 수정합니다")
    args = parser.parse_args()

    outputs_dir = Path(__file__).parent.parent / "outputs"
    log_files = list(outputs_dir.rglob("*_logs.json"))

    if not log_files:
        print("로그 파일을 찾을 수 없습니다.")
        return

    print(f"Found {len(log_files)} log file(s)")
    if not args.apply:
        print("DRY-RUN 모드 (변경사항만 표시, --apply 로 적용)")

    migrated = 0
    for log_path in log_files:
        if migrate_log_file(log_path, apply=args.apply):
            migrated += 1

    print(f"\n{'='*60}")
    print(f"Total: {migrated}/{len(log_files)} file(s) {'migrated' if args.apply else 'need migration'}")


if __name__ == "__main__":
    main()
