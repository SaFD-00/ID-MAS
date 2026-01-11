"""
Sample Data Extractor for Terminal Goal Generation

각 데이터셋에서 대표 샘플을 추출하여 Terminal Goal 생성에 사용.
Design Phase 실행 전 미리 실행하여 샘플 파일 생성.

Usage:
    # 모든 데이터셋 샘플 추출
    python -m utils.sample_extractor

    # 특정 데이터셋만 추출
    python -m utils.sample_extractor --domain math --dataset math --num-samples 20

    # 다양성 기반 샘플링 전략 지정
    python -m utils.sample_extractor --domain math --dataset gsm8k --strategy diverse
"""
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent

# 데이터 설정
DATA_CONFIG = {
    "math": {
        "gsm8k": {
            "train_file": "gsm8k_train.json",
            "samples_file": "gsm8k_samples.json",
            "num_samples": 15,
        },
        "math": {
            "train_file": "math_train.json",
            "samples_file": "math_samples.json",
            "num_samples": 20,
            # MATH는 type/level 기반 stratified sampling
            "use_stratified": True,
        },
    },
    "logical": {
        "reclor": {
            "train_file": "reclor_train.json",
            "samples_file": "reclor_samples.json",
            "num_samples": 15,
        },
    },
    "commonsense": {
        "arc_c": {
            "train_file": "arc_c_train.json",
            "samples_file": "arc_c_samples.json",
            "num_samples": 15,
        },
    },
}


def categorize_by_length(items: List[Dict], text_key: str = "input") -> Dict[str, List[Dict]]:
    """
    문제 길이로 3그룹 분류 (short/medium/long)

    Args:
        items: 데이터 아이템 리스트
        text_key: 길이를 측정할 필드 키

    Returns:
        {"short": [...], "medium": [...], "long": [...]}
    """
    if not items:
        return {"short": [], "medium": [], "long": []}

    # 길이 계산
    lengths = [(item, len(item.get(text_key, ""))) for item in items]
    lengths.sort(key=lambda x: x[1])

    # 3등분
    n = len(lengths)
    third = n // 3

    short = [item for item, _ in lengths[:third]]
    medium = [item for item, _ in lengths[third:2*third]]
    long = [item for item, _ in lengths[2*third:]]

    return {"short": short, "medium": medium, "long": long}


def extract_random_samples(
    data: List[Dict],
    num_samples: int = 15
) -> List[Dict]:
    """
    랜덤 샘플링

    Args:
        data: 전체 데이터
        num_samples: 추출할 샘플 수

    Returns:
        샘플 리스트
    """
    if len(data) <= num_samples:
        return data
    return random.sample(data, num_samples)


def extract_diverse_samples(
    data: List[Dict],
    num_samples: int = 15,
    text_key: str = "input"
) -> List[Dict]:
    """
    다양성 기반 샘플링

    1차: 60개 랜덤 추출
    2차: 문제 길이 기반으로 균등 분배

    Args:
        data: 전체 데이터
        num_samples: 최종 샘플 수
        text_key: 길이를 측정할 필드

    Returns:
        다양한 샘플 리스트
    """
    # Step 1: 60개 랜덤 추출
    pool_size = min(60, len(data))
    random_pool = random.sample(data, pool_size)

    # Step 2: 길이로 분류
    by_length = categorize_by_length(random_pool, text_key)

    # Step 3: 각 그룹에서 균등 추출
    samples_per_group = num_samples // 3
    remainder = num_samples % 3

    diverse_samples = []

    for i, (group, items) in enumerate(by_length.items()):
        # 나머지는 앞 그룹에 분배
        count = samples_per_group + (1 if i < remainder else 0)
        if items:
            selected = random.sample(items, min(count, len(items)))
            diverse_samples.extend(selected)

    return diverse_samples


def extract_math_stratified_samples(
    data: List[Dict],
    num_samples: int = 20
) -> List[Dict]:
    """
    MATH 데이터셋 Stratified Sampling

    type과 difficulty를 기반으로 균등 분배.
    output에서 난이도와 유형 정보를 추론.

    Args:
        data: MATH 훈련 데이터
        num_samples: 추출할 샘플 수

    Returns:
        stratified 샘플 리스트
    """
    # output 필드에서 문제 특성 분석
    # MATH 데이터는 LaTeX 풀이가 포함됨

    # 길이 기반으로 난이도 추론 (긴 풀이 = 어려운 문제)
    by_length = categorize_by_length(data, "output")

    # 각 난이도 그룹에서 균등 추출
    samples = []
    samples_per_group = num_samples // 3

    for group in ["short", "medium", "long"]:
        items = by_length[group]
        if items:
            count = min(samples_per_group, len(items))
            samples.extend(random.sample(items, count))

    # 부족한 샘플 수 보충
    remaining = num_samples - len(samples)
    if remaining > 0:
        all_items = [item for group in by_length.values() for item in group]
        available = [item for item in all_items if item not in samples]
        if available:
            samples.extend(random.sample(available, min(remaining, len(available))))

    return samples


def extract_samples(
    domain: str,
    dataset: str,
    num_samples: Optional[int] = None,
    strategy: str = "diverse"
) -> List[Dict]:
    """
    데이터셋에서 샘플 추출

    Args:
        domain: 도메인 이름 (math, logical, commonsense)
        dataset: 데이터셋 이름 (gsm8k, math, reclor, arc_c)
        num_samples: 샘플 수 (None이면 설정값 사용)
        strategy: 샘플링 전략 ("random", "diverse", "stratified")

    Returns:
        추출된 샘플 리스트
    """
    if domain not in DATA_CONFIG:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DATA_CONFIG.keys())}")

    if dataset not in DATA_CONFIG[domain]:
        raise ValueError(f"Unknown dataset: {dataset} for domain {domain}")

    config = DATA_CONFIG[domain][dataset]
    data_dir = PROJECT_ROOT / "data" / domain / "train" / "data"
    train_file = data_dir / config["train_file"]

    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    # 데이터 로드
    with open(train_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 샘플 수 결정
    final_num = num_samples or config["num_samples"]

    # 전략에 따른 샘플링
    if config.get("use_stratified") and strategy != "random":
        samples = extract_math_stratified_samples(data, final_num)
    elif strategy == "diverse":
        samples = extract_diverse_samples(data, final_num)
    else:
        samples = extract_random_samples(data, final_num)

    return samples


def save_samples(
    domain: str,
    dataset: str,
    samples: List[Dict],
    output_dir: Optional[Path] = None
) -> Path:
    """
    추출된 샘플을 파일로 저장

    Args:
        domain: 도메인 이름
        dataset: 데이터셋 이름
        samples: 샘플 리스트
        output_dir: 출력 디렉토리 (None이면 기본 위치)

    Returns:
        저장된 파일 경로
    """
    config = DATA_CONFIG[domain][dataset]

    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / domain / "train" / "data"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / config["samples_file"]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    return output_path


def extract_all_samples(strategy: str = "diverse") -> Dict[str, Dict[str, Path]]:
    """
    모든 데이터셋에서 샘플 추출

    Args:
        strategy: 샘플링 전략

    Returns:
        {domain: {dataset: output_path}} 형식의 결과
    """
    results = {}

    for domain, datasets in DATA_CONFIG.items():
        results[domain] = {}

        for dataset, config in datasets.items():
            try:
                print(f"\n[{domain.upper()}/{dataset.upper()}] Extracting samples...")

                samples = extract_samples(domain, dataset, strategy=strategy)
                output_path = save_samples(domain, dataset, samples)

                results[domain][dataset] = output_path
                print(f"  Extracted {len(samples)} samples -> {output_path}")

            except FileNotFoundError as e:
                print(f"  Skipped: {e}")
                results[domain][dataset] = None

    return results


def main():
    """CLI 진입점"""
    parser = argparse.ArgumentParser(
        description="Extract representative samples from datasets for Terminal Goal generation"
    )

    parser.add_argument(
        "--domain",
        type=str,
        choices=list(DATA_CONFIG.keys()),
        help="Domain to extract samples from (e.g., math, logical)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name (e.g., gsm8k, math, reclor)"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to extract (default: use config)"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["random", "diverse", "stratified"],
        default="diverse",
        help="Sampling strategy (default: diverse)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract samples from all datasets"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Sample Data Extractor for Terminal Goal Generation")
    print("=" * 60)

    if args.all or (args.domain is None and args.dataset is None):
        # 모든 데이터셋에서 추출
        results = extract_all_samples(strategy=args.strategy)

        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)

        for domain, datasets in results.items():
            for dataset, path in datasets.items():
                status = f"-> {path}" if path else "SKIPPED"
                print(f"  [{domain}/{dataset}] {status}")

    else:
        # 특정 데이터셋만 추출
        if not args.domain or not args.dataset:
            parser.error("Both --domain and --dataset are required for single extraction")

        samples = extract_samples(
            domain=args.domain,
            dataset=args.dataset,
            num_samples=args.num_samples,
            strategy=args.strategy
        )

        output_path = save_samples(args.domain, args.dataset, samples)

        print(f"\nExtracted {len(samples)} samples")
        print(f"Saved to: {output_path}")

        # 샘플 미리보기
        print("\n--- Sample Preview ---")
        for i, sample in enumerate(samples[:3]):
            input_text = sample.get("input", "")[:100]
            print(f"\n[{i+1}] {input_text}...")


if __name__ == "__main__":
    main()
