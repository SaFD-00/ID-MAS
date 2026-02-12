import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.models import create_teacher_config, get_model_short_name
from config.domains import DATASET_TO_DOMAIN, TRAINING_DATASETS
from design_modules.instructional_goal import InstructionalGoalGenerator
from design_modules.analysis import InstructionalAnalysis


class DataEnhancer:
    """데이터 Enhancement를 위한 클래스.

    학습 데이터에 Instructional Goal과 Task Analysis를 추가하여
    SFT 학습의 효과를 향상시킵니다.

    Attributes:
        teacher_config: Teacher 모델 설정
        goal_generator: Instructional Goal 생성기
        analysis_generator: Task Analysis 생성기
        model_suffix: Teacher 모델 접미사
        student_suffix: Student 모델 접미사
    """

    def __init__(self, teacher_config: dict = None, model_suffix: str = None, student_suffix: str = None):
        """DataEnhancer를 초기화합니다.

        Args:
            teacher_config: Teacher 모델 설정 (None이면 기본 설정 사용)
            model_suffix: 출력 파일명에 사용할 Teacher 모델 접미사 (None이면 자동 생성)
            student_suffix: 출력 파일명에 사용할 Student 모델 접미사
        """
        self.teacher_config = teacher_config
        self.goal_generator = InstructionalGoalGenerator(teacher_config)
        self.analysis_generator = InstructionalAnalysis(teacher_config)
        self.model_suffix = model_suffix
        self.student_suffix = student_suffix

    def enhance_dataset(
        self,
        domain: str,
        dataset: str,
        sample_count: int = 25
    ) -> Path:
        """데이터셋 enhancement를 수행합니다.

        Args:
            domain: 도메인 이름 (math, logical, commonsense)
            dataset: 데이터셋 이름 (gsm8k, math, reclor, arc_c)
            sample_count: 학습목표 생성에 사용할 샘플 수 (기본: 25)

        Returns:
            생성된 파일 경로

        Raises:
            FileNotFoundError: 소스 파일이 존재하지 않는 경우
        """
        print(f"\n{'=' * 60}")
        print(f"Processing: {domain}/{dataset}")
        print(f"{'=' * 60}")

        # 1. 소스 데이터 로드
        source_path = self._get_source_path(domain, dataset)
        print(f"  Source: {source_path}")

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        data = self._load_json(source_path)
        print(f"  Loaded {len(data)} records")

        # 2. 학습목표 생성 (샘플 기반)
        print(f"  Generating Instructional Goal (using {sample_count} samples)...")
        samples = data[:min(sample_count, len(data))]
        goal_result = self.goal_generator.generate(
            train_samples=samples,
            domain=domain,
            dataset=dataset
        )
        instructional_goal = goal_result.get("instructional_goal", "")
        print(f"  Goal: {instructional_goal[:100]}...")

        # 3. 과제분석 생성
        print(f"  Generating Task Analysis...")
        analysis_result = self.analysis_generator.analyze(instructional_goal)
        task_analysis = analysis_result.get("raw_output", "")
        print(f"  Analysis generated ({len(task_analysis)} chars)")

        # 4. instruction 확장
        print(f"  Enhancing instructions...")
        enhanced_data = self._enhance_instructions(
            data, instructional_goal, task_analysis
        )

        # 5. 저장
        output_path = self._get_output_path(domain, dataset)
        self._save_json(enhanced_data, output_path)
        print(f"  Saved: {output_path}")

        return output_path

    def _enhance_instructions(
        self,
        data: List[Dict],
        instructional_goal: str,
        task_analysis: str
    ) -> List[Dict]:
        """데이터에 학습목표와 과제분석 메타데이터를 추가합니다.

        instruction 필드는 원본 그대로 유지하고,
        instructional_goal과 task_analysis는 metadata에 저장합니다.
        SCAFFOLDING_SYSTEM_PROMPT와의 결합은 학습 파이프라인에서 동적으로 수행됩니다.

        Args:
            data: 원본 데이터 리스트
            instructional_goal: 학습 목표
            task_analysis: 과제 분석

        Returns:
            메타데이터가 추가된 데이터 리스트
        """
        enhanced = []
        for item in data:
            new_item = {
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", ""),
                "metadata": {
                    **item.get("metadata", {}),
                    "instructional_goal": instructional_goal,
                    "task_analysis": task_analysis,
                }
            }

            enhanced.append(new_item)

        return enhanced

    def _get_source_path(self, domain: str, dataset: str) -> Path:
        """소스 데이터 파일 경로를 반환합니다."""
        return PROJECT_ROOT / "data" / domain / "train" / "data" / f"{dataset}_train.json"

    def _get_output_path(self, domain: str, dataset: str) -> Path:
        """출력 파일 경로를 반환합니다.

        출력 경로: outputs/{domain}/train/{student_short}/data/{dataset}_train_ID-MAS.json
        """
        student_short = self.student_suffix or "default"
        output_dir = PROJECT_ROOT / "outputs" / domain / "train" / student_short / "data"
        return output_dir / f"{dataset}_train_ID-MAS.json"

    def _load_json(self, path: Path) -> List[Dict]:
        """JSON 파일을 로드합니다."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_json(self, data: Any, path: Path):
        """JSON 파일을 저장합니다."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def get_all_datasets() -> List[tuple]:
    """전체 처리 대상 데이터셋 목록을 반환합니다."""
    datasets = []
    for domain, dataset_list in TRAINING_DATASETS.items():
        for dataset in dataset_list:
            datasets.append((domain, dataset))
    return datasets


def main():
    parser = argparse.ArgumentParser(
        description="ID-MAS Data Enhancement - Add instructional goals and task analysis to training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single dataset
    python scripts/enhance_data.py --domain math --dataset gsm8k

    # Process all datasets
    python scripts/enhance_data.py --all

    # Specify teacher model
    python scripts/enhance_data.py --all --teacher-model Qwen/Qwen3-32B
        """
    )

    parser.add_argument(
        "--domain",
        type=str,
        choices=["math", "logical", "commonsense"],
        help="Domain to process"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to process (e.g., gsm8k, math, reclor, arc_c)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all datasets"
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=None,
        help="Teacher model name (e.g., Qwen/Qwen3-32B)"
    )
    parser.add_argument(
        "--model-suffix",
        type=str,
        default=None,
        help="Model suffix for output file name (overrides automatic detection)"
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=25,
        help="Number of samples for instructional goal generation (default: 25)"
    )

    args = parser.parse_args()

    # 인수 검증
    if not args.all and (not args.domain or not args.dataset):
        parser.error("Either --all or both --domain and --dataset are required")

    # Teacher config 생성
    teacher_config = None
    if args.teacher_model:
        teacher_config = create_teacher_config(args.teacher_model)
        print(f"Using teacher model: {args.teacher_model}")

    # Enhancer 초기화
    enhancer = DataEnhancer(
        teacher_config=teacher_config,
        model_suffix=args.model_suffix
    )

    # 처리할 데이터셋 목록
    if args.all:
        datasets = get_all_datasets()
        print(f"Processing all {len(datasets)} datasets...")
    else:
        datasets = [(args.domain, args.dataset)]

    # 처리 실행
    results = []
    errors = []

    for domain, dataset in datasets:
        try:
            output_path = enhancer.enhance_dataset(
                domain=domain,
                dataset=dataset,
                sample_count=args.sample_count
            )
            results.append((domain, dataset, output_path))
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append((domain, dataset, str(e)))

    # 결과 요약
    print(f"\n{'=' * 60}")
    print("Enhancement Complete!")
    print(f"{'=' * 60}")

    if results:
        print(f"\nSuccessfully processed ({len(results)}):")
        for domain, dataset, path in results:
            print(f"  - {domain}/{dataset}: {path.name}")

    if errors:
        print(f"\nFailed ({len(errors)}):")
        for domain, dataset, error in errors:
            print(f"  - {domain}/{dataset}: {error}")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
