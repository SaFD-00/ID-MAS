import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import create_teacher_config, get_model_short_name
from config.domains import DATASET_TO_DOMAIN, TRAINING_DATASETS
from design_modules.instructional_goal import InstructionalGoalGenerator
from design_modules.analysis import InstructionalAnalysis


# Enhanced instruction template for high-quality SFT training data
ENHANCED_INSTRUCTION_TEMPLATE = """{original_instruction}

## Learning Objective
Your response should demonstrate: {instructional_goal}

## Problem-Solving Guidelines
Follow the structured approach below to ensure a complete and well-reasoned solution:

{task_analysis}

## Response Requirements
1. Explicitly connect each step to the relevant sub-skill or knowledge from the guidelines above
2. Verify your intermediate results before proceeding to the next step
3. Present your final answer clearly in the required format"""


class DataEnhancer:
    """데이터 Enhancement를 위한 클래스"""

    def __init__(self, teacher_config: dict = None, model_suffix: str = None):
        """
        DataEnhancer 초기화

        Args:
            teacher_config: Teacher model 설정 (None이면 기본 설정 사용)
            model_suffix: 출력 파일명에 사용할 모델 suffix (None이면 자동 생성)
        """
        self.teacher_config = teacher_config
        self.goal_generator = InstructionalGoalGenerator(teacher_config)
        self.analysis_generator = InstructionalAnalysis(teacher_config)
        self.model_suffix = model_suffix

    def enhance_dataset(
        self,
        domain: str,
        dataset: str,
        sample_count: int = 25
    ) -> Path:
        """
        데이터셋 enhancement 수행

        Args:
            domain: 도메인 이름 (math, logical, commonsense)
            dataset: 데이터셋 이름 (gsm8k, math, reclor, arc_c)
            sample_count: 학습목표 생성에 사용할 샘플 수 (기본 25)

        Returns:
            생성된 파일 경로
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

        # 6. 메타데이터 저장 (학습목표/과제분석 별도 보관)
        self._save_metadata(
            domain, dataset, instructional_goal, task_analysis, goal_result
        )

        return output_path

    def _enhance_instructions(
        self,
        data: List[Dict],
        instructional_goal: str,
        task_analysis: str
    ) -> List[Dict]:
        """
        데이터에 학습목표와 과제분석 메타데이터 추가

        중요: instruction 필드는 원본을 유지합니다.
        Enhanced instruction은 _enhanced_instruction 메타데이터로만 저장됩니다.
        이를 통해 SFT 학습 시 train-test mismatch를 방지합니다.
        """
        enhanced = []
        for item in data:
            new_item = item.copy()
            original_instruction = item.get("instruction", "")

            # instruction은 원본 유지 (train-test mismatch 방지)
            # enhanced version은 메타데이터로만 저장 (참조/디버깅용)
            enhanced_instruction = ENHANCED_INSTRUCTION_TEMPLATE.format(
                original_instruction=original_instruction,
                instructional_goal=instructional_goal,
                task_analysis=task_analysis
            )

            # 메타데이터 추가 (instruction 필드는 변경하지 않음)
            new_item["_enhanced"] = True
            new_item["_instructional_goal"] = instructional_goal
            new_item["_task_analysis"] = task_analysis
            new_item["_enhanced_instruction"] = enhanced_instruction

            enhanced.append(new_item)

        return enhanced

    def _get_source_path(self, domain: str, dataset: str) -> Path:
        """소스 데이터 파일 경로"""
        return PROJECT_ROOT / "data" / domain / "train" / "data" / f"{dataset}_train.json"

    def _get_output_path(self, domain: str, dataset: str) -> Path:
        """출력 파일 경로"""
        if self.model_suffix:
            suffix = self.model_suffix
        elif self.teacher_config:
            model_name = self.teacher_config.get("model", "unknown")
            suffix = get_model_short_name(model_name)
        else:
            suffix = "default"

        return PROJECT_ROOT / "data" / domain / "train" / "data" / f"{dataset}_train_ID-MAS_{suffix}.json"

    def _save_metadata(
        self,
        domain: str,
        dataset: str,
        instructional_goal: str,
        task_analysis: str,
        goal_result: Dict
    ):
        """메타데이터 별도 저장 (디버깅/재사용용)"""
        if self.model_suffix:
            suffix = self.model_suffix
        elif self.teacher_config:
            model_name = self.teacher_config.get("model", "unknown")
            suffix = get_model_short_name(model_name)
        else:
            suffix = "default"

        metadata_path = PROJECT_ROOT / "data" / domain / "train" / "data" / f"{dataset}_ID-MAS_metadata_{suffix}.json"

        metadata = {
            "domain": domain,
            "dataset": dataset,
            "instructional_goal": instructional_goal,
            "task_analysis": task_analysis,
            "goal_result": goal_result,
            "model_suffix": suffix
        }

        self._save_json(metadata, metadata_path)
        print(f"  Metadata saved: {metadata_path.name}")

    def _load_json(self, path: Path) -> List[Dict]:
        """JSON 파일 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_json(self, data: Any, path: Path):
        """JSON 파일 저장"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def get_all_datasets() -> List[tuple]:
    """전체 처리 대상 데이터셋 목록 반환"""
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
    python scripts/enhance_data.py --all --teacher-model Qwen/Qwen2.5-72B-Instruct
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
        help="Teacher model name (e.g., Qwen/Qwen2.5-72B-Instruct)"
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
