"""
Re-parse and Update Evaluation Results

This script re-parses student_response fields in eval_results JSON files
using the updated extraction logic from answer_extractor.py, then updates:
- predicted_answer
- is_correct
- correct_count
- accuracy

Usage:
    # Full re-parse with backup
    python utils/reparse_eval_results.py

    # Dry-run mode (no modifications)
    python utils/reparse_eval_results.py --dry-run

    # Test on specific file
    python utils/reparse_eval_results.py --file path/to/file.json

    # Skip backup
    python utils/reparse_eval_results.py --no-backup
"""

import argparse
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Import answer extraction logic
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.base_loader import AnswerType
from utils.answer_extractor import get_extractor


@dataclass
class ReparseResult:
    """Results from re-parsing a single file"""
    filepath: Path
    total_questions: int
    no_change: int
    answer_changed: int
    correctness_changed: int
    extraction_failed: int
    before_accuracy: float
    after_accuracy: float
    accuracy_delta: float

    def to_dict(self) -> Dict:
        return {
            'filepath': str(self.filepath),
            'total_questions': self.total_questions,
            'changes': {
                'no_change': self.no_change,
                'answer_changed': self.answer_changed,
                'correctness_changed': self.correctness_changed,
                'extraction_failed': self.extraction_failed
            },
            'accuracy': {
                'before': self.before_accuracy,
                'after': self.after_accuracy,
                'delta': self.accuracy_delta
            }
        }


class EvalResultsReparser:
    """Re-parses evaluation results using updated extraction logic"""

    def __init__(self, data_dir: Path, backup_dir: Optional[Path] = None,
                 dry_run: bool = False, skip_backup: bool = False):
        """
        Initialize the reparser.

        Args:
            data_dir: Root data directory (e.g., /path/to/ID-MAS/data)
            backup_dir: Directory for backups (default: data_dir/backups/eval_results/{timestamp})
            dry_run: If True, don't modify files
            skip_backup: If True, skip creating backups
        """
        self.data_dir = Path(data_dir)
        self.dry_run = dry_run
        self.skip_backup = skip_backup

        if backup_dir:
            self.backup_dir = Path(backup_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.backup_dir = self.data_dir / "backups" / "eval_results" / timestamp

    def discover_eval_files(self, specific_file: Optional[Path] = None) -> List[Path]:
        """
        Discover all eval_results JSON files.

        Args:
            specific_file: If provided, only process this file

        Returns:
            List of paths to eval_results files
        """
        if specific_file:
            # Convert to absolute path
            specific_file = specific_file.resolve()
            if not specific_file.exists():
                raise FileNotFoundError(f"File not found: {specific_file}")
            return [specific_file]

        # Pattern: data/{domain}/{model}/{dataset}/eval_results/*_eval_results-*.json
        eval_files = []
        for domain_dir in self.data_dir.iterdir():
            if not domain_dir.is_dir() or domain_dir.name == 'backups':
                continue

            for model_dir in domain_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                for dataset_dir in model_dir.iterdir():
                    if not dataset_dir.is_dir():
                        continue

                    eval_results_dir = dataset_dir / "eval_results"
                    if eval_results_dir.exists():
                        for json_file in eval_results_dir.glob("*_eval_results-*.json"):
                            eval_files.append(json_file)

        return sorted(eval_files)

    def create_backup(self, files: List[Path]) -> Dict:
        """
        Create backups of files before modification.

        Args:
            files: List of files to backup

        Returns:
            Manifest dictionary with file info and checksums
        """
        if self.skip_backup:
            print("Skipping backup (--no-backup flag)")
            return {}

        print(f"\nCreating backup in: {self.backup_dir}")
        manifest = {
            'timestamp': datetime.now().isoformat(),
            'total_files': len(files),
            'files': []
        }

        for filepath in files:
            # Calculate relative path from data_dir
            rel_path = filepath.relative_to(self.data_dir)
            backup_path = self.backup_dir / rel_path

            # Create parent directories
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            shutil.copy2(filepath, backup_path)

            # Calculate checksum
            with open(filepath, 'rb') as f:
                checksum = hashlib.md5(f.read()).hexdigest()

            manifest['files'].append({
                'original': str(filepath),
                'backup': str(backup_path),
                'checksum': checksum,
                'size': filepath.stat().st_size
            })

        # Save manifest
        manifest_path = self.backup_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"Backed up {len(files)} files")
        print(f"Manifest saved to: {manifest_path}")

        return manifest

    def get_answer_type_for_file(self, filepath: Path) -> AnswerType:
        """
        Determine answer type from file path.

        Path pattern: data/{domain}/{model}/{dataset}/eval_results/{dataset}_eval_results-{method}.json

        Args:
            filepath: Path to eval_results file

        Returns:
            AnswerType enum value
        """
        # Extract domain and dataset from path
        parts = filepath.parts

        # Find domain (knowledge or math)
        domain_idx = -1
        for i, part in enumerate(parts):
            if part in ['knowledge', 'math']:
                domain_idx = i
                break

        if domain_idx == -1:
            # Fallback: try to read from JSON
            with open(filepath) as f:
                data = json.load(f)
                domain = data.get('domain', 'knowledge')
                dataset = data.get('eval_dataset', '')
        else:
            domain = parts[domain_idx]
            # Dataset is 2 positions after domain: {domain}/{model}/{dataset}
            dataset = parts[domain_idx + 2] if len(parts) > domain_idx + 2 else ''

        # Map domain/dataset to answer type
        if domain == 'knowledge':
            # All knowledge datasets use MCQ
            return AnswerType.MCQ
        elif domain == 'math':
            # math dataset uses LATEX, others use NUMERIC
            if dataset == 'math':
                return AnswerType.LATEX
            else:
                return AnswerType.NUMERIC
        else:
            # Default fallback
            return AnswerType.TEXT

    def reparse_file(self, filepath: Path) -> ReparseResult:
        """
        Re-parse a single eval_results file.

        Args:
            filepath: Path to eval_results JSON file

        Returns:
            ReparseResult with statistics
        """
        # Load file
        with open(filepath) as f:
            data = json.load(f)

        # Get answer type and extractor
        answer_type = self.get_answer_type_for_file(filepath)
        extractor = get_extractor(answer_type)

        # Track changes
        stats = {
            'no_change': 0,
            'answer_changed': 0,
            'correctness_changed': 0,
            'extraction_failed': 0
        }

        # Store old accuracy
        old_accuracy = data.get('accuracy', 0.0)
        old_correct_count = data.get('correct_count', 0)

        new_correct_count = 0
        question_results = data.get('question_results', [])

        # Process each question
        for i, result in enumerate(question_results):
            student_response = result.get('student_response', '')
            ground_truth = result.get('ground_truth', '')

            old_predicted = result.get('predicted_answer', '')
            old_is_correct = result.get('is_correct', False)

            # Extract new answer
            try:
                new_predicted = extractor.extract(student_response)
                if new_predicted is None:
                    new_predicted = ""
                    stats['extraction_failed'] += 1
            except Exception as e:
                print(f"  Warning: Extraction failed for question {i}: {e}")
                new_predicted = ""
                stats['extraction_failed'] += 1

            # Compare with ground truth
            try:
                new_is_correct = extractor.compare(new_predicted, ground_truth)
            except Exception as e:
                print(f"  Warning: Comparison failed for question {i}: {e}")
                new_is_correct = False

            # Track changes
            if new_predicted != old_predicted:
                stats['answer_changed'] += 1

            if new_is_correct != old_is_correct:
                stats['correctness_changed'] += 1

            if new_predicted == old_predicted and new_is_correct == old_is_correct:
                stats['no_change'] += 1

            # Update result
            result['predicted_answer'] = new_predicted
            result['is_correct'] = new_is_correct

            if new_is_correct:
                new_correct_count += 1

        # Recalculate metrics
        total_questions = len(question_results)
        new_accuracy = new_correct_count / total_questions if total_questions > 0 else 0.0

        data['correct_count'] = new_correct_count
        data['accuracy'] = new_accuracy
        data['evaluated_questions'] = total_questions

        # Save updated file (if not dry-run)
        if not self.dry_run:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        # Return results
        return ReparseResult(
            filepath=filepath,
            total_questions=total_questions,
            no_change=stats['no_change'],
            answer_changed=stats['answer_changed'],
            correctness_changed=stats['correctness_changed'],
            extraction_failed=stats['extraction_failed'],
            before_accuracy=old_accuracy,
            after_accuracy=new_accuracy,
            accuracy_delta=new_accuracy - old_accuracy
        )

    def run(self, specific_file: Optional[Path] = None) -> List[ReparseResult]:
        """
        Run the re-parsing pipeline.

        Args:
            specific_file: If provided, only process this file

        Returns:
            List of ReparseResult objects
        """
        print("="*80)
        print("Eval Results Re-parser")
        print("="*80)

        if self.dry_run:
            print("DRY RUN MODE: No files will be modified")

        # Discover files
        print("\nDiscovering eval_results files...")
        files = self.discover_eval_files(specific_file)
        print(f"Found {len(files)} file(s)")

        if len(files) == 0:
            print("No files to process")
            return []

        # Create backup
        if not self.dry_run:
            self.create_backup(files)

        # Process files
        print("\nRe-parsing files...")
        results = []

        for i, filepath in enumerate(files, 1):
            rel_path = filepath.relative_to(self.data_dir)
            print(f"\n[{i}/{len(files)}] Processing: {rel_path}")

            try:
                result = self.reparse_file(filepath)
                results.append(result)

                # Print summary
                print(f"  Total questions: {result.total_questions}")
                print(f"  Changes: {result.answer_changed} answer(s), "
                      f"{result.correctness_changed} correctness")
                print(f"  Accuracy: {result.before_accuracy:.4f} → {result.after_accuracy:.4f} "
                      f"(Δ{result.accuracy_delta:+.4f})")

                if result.extraction_failed > 0:
                    print(f"  ⚠ Extraction failed for {result.extraction_failed} question(s)")

            except Exception as e:
                print(f"  ERROR: Failed to process file: {e}")
                import traceback
                traceback.print_exc()

        return results

    def generate_report(self, results: List[ReparseResult]) -> str:
        """
        Generate summary report.

        Args:
            results: List of ReparseResult objects

        Returns:
            Formatted report string
        """
        if not results:
            return "No files processed"

        total_questions = sum(r.total_questions for r in results)
        total_answer_changed = sum(r.answer_changed for r in results)
        total_correctness_changed = sum(r.correctness_changed for r in results)
        total_extraction_failed = sum(r.extraction_failed for r in results)

        avg_accuracy_before = sum(r.before_accuracy for r in results) / len(results)
        avg_accuracy_after = sum(r.after_accuracy for r in results) / len(results)
        avg_delta = avg_accuracy_after - avg_accuracy_before

        # Find biggest improvements and regressions
        improvements = sorted(results, key=lambda r: r.accuracy_delta, reverse=True)[:5]
        regressions = sorted(results, key=lambda r: r.accuracy_delta)[:5]

        report = []
        report.append("="*80)
        report.append("SUMMARY REPORT")
        report.append("="*80)
        report.append(f"Files processed: {len(results)}")
        report.append(f"Total questions: {total_questions}")
        report.append(f"")
        report.append(f"Changes:")
        report.append(f"  Answers changed: {total_answer_changed}")
        report.append(f"  Correctness changed: {total_correctness_changed}")
        report.append(f"  Extraction failures: {total_extraction_failed}")
        report.append(f"")
        report.append(f"Accuracy:")
        report.append(f"  Average before: {avg_accuracy_before:.4f}")
        report.append(f"  Average after:  {avg_accuracy_after:.4f}")
        report.append(f"  Average delta:  {avg_delta:+.4f}")
        report.append(f"")

        if improvements and improvements[0].accuracy_delta > 0:
            report.append(f"Top improvements:")
            for r in improvements[:5]:
                if r.accuracy_delta > 0:
                    rel_path = r.filepath.relative_to(self.data_dir)
                    report.append(f"  {rel_path}")
                    report.append(f"    {r.before_accuracy:.4f} → {r.after_accuracy:.4f} "
                                f"({r.accuracy_delta:+.4f})")
            report.append(f"")

        if regressions and regressions[0].accuracy_delta < 0:
            report.append(f"Top regressions:")
            for r in regressions[:5]:
                if r.accuracy_delta < 0:
                    rel_path = r.filepath.relative_to(self.data_dir)
                    report.append(f"  {rel_path}")
                    report.append(f"    {r.before_accuracy:.4f} → {r.after_accuracy:.4f} "
                                f"({r.accuracy_delta:+.4f})")
            report.append(f"")

        report.append("="*80)

        return "\n".join(report)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Re-parse evaluation results using updated extraction logic"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'data',
        help='Root data directory (default: ../data)'
    )
    parser.add_argument(
        '--backup-dir',
        type=Path,
        help='Backup directory (default: data/backups/eval_results/{timestamp})'
    )
    parser.add_argument(
        '--file',
        type=Path,
        help='Process only this specific file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without modifying files'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backups'
    )
    parser.add_argument(
        '--save-report',
        type=Path,
        help='Save detailed report to file'
    )

    args = parser.parse_args()

    # Initialize reparser
    reparser = EvalResultsReparser(
        data_dir=args.data_dir,
        backup_dir=args.backup_dir,
        dry_run=args.dry_run,
        skip_backup=args.no_backup
    )

    # Run
    results = reparser.run(specific_file=args.file)

    # Generate and print report
    report = reparser.generate_report(results)
    print("\n" + report)

    # Save detailed results
    if args.save_report:
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'dry_run': args.dry_run,
            'files_processed': len(results),
            'results': [r.to_dict() for r in results]
        }

        with open(args.save_report, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\nDetailed report saved to: {args.save_report}")


if __name__ == '__main__':
    main()
