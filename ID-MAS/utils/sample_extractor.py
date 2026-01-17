"""샘플 데이터 추출기 모듈 - Instructional Goal 생성용.

이 모듈은 각 데이터셋에서 대표 샘플을 추출하여 Instructional Goal 생성에 사용합니다.
Design Phase 실행 전 미리 실행하여 샘플 파일을 생성해야 합니다.

주요 함수:
    extract_samples(): 데이터셋에서 샘플 추출
    save_samples(): 추출된 샘플 저장
    extract_all_samples(): 모든 데이터셋에서 샘플 추출

샘플링 전략:
    - stratified: 메타데이터(type > level) 기반 계층적 샘플링
    - diverse: 길이 기반 다양성 샘플링 (short/medium/long 균등)
    - random: 랜덤 샘플링

사용 예시:
    # 모든 데이터셋 샘플 추출
    >>> python -m utils.sample_extractor

    # 특정 데이터셋만 추출
    >>> python -m utils.sample_extractor --domain math --dataset math --num-samples 20

    # 메타데이터 기반 계층적 샘플링 (stratified)
    >>> python -m utils.sample_extractor --domain math --dataset math --strategy stratified

    # 길이 기반 다양성 샘플링 (diverse)
    >>> python -m utils.sample_extractor --domain math --dataset gsm8k --strategy diverse
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
            "num_samples": 20,
        },
        "math": {
            "train_file": "math_train.json",
            "samples_file": "math_samples.json",
            "num_samples": 20,
            "use_stratified": True,
        },
    },
    "logical": {
        "reclor": {
            "train_file": "reclor_train.json",
            "samples_file": "reclor_samples.json",
            "num_samples": 20,
        },
    },
    "commonsense": {
        "arc_c": {
            "train_file": "arc_c_train.json",
            "samples_file": "arc_c_samples.json",
            "num_samples": 20,
        },
    },
}


def categorize_by_length(items: List[Dict], text_key: str = "input") -> Dict[str, List[Dict]]:
    """문제 길이로 3그룹(short/medium/long)으로 분류합니다.

    Args:
        items: 데이터 아이템 리스트
        text_key: 길이를 측정할 필드 키

    Returns:
        {"short": [...], "medium": [...], "long": [...]} 형식의 딕셔너리
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
    """랜덤 샘플링을 수행합니다.

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
    """다양성 기반 샘플링을 수행합니다.

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


def extract_stratified_samples(
    data: List[Dict],
    num_samples: int,
    primary_key: str = "type",
    secondary_key: Optional[str] = "level",
    text_key: str = "input"
) -> List[Dict]:
    """메타데이터 기반 계층적 샘플링을 수행합니다 (type > level > length).

    로컬 데이터의 metadata 필드를 활용하여 다양성 있는 샘플을 추출합니다.
    메타데이터가 없거나 의미 없으면 diverse 샘플링으로 폴백합니다.

    Args:
        data: 로컬 데이터 리스트 (metadata 필드 포함)
        num_samples: 추출할 샘플 수
        primary_key: 1차 계층 키 (예: "type")
        secondary_key: 2차 계층 키 (예: "level"), None이면 1차만 사용
        text_key: 길이 측정용 필드 키

    Returns:
        다양성 있는 샘플 리스트
    """
    if not data:
        return []

    # 메타데이터 존재 여부 확인
    def get_metadata_value(item: Dict, key: str) -> Optional[str]:
        metadata = item.get("metadata", {})
        if isinstance(metadata, dict):
            return metadata.get(key)
        return None

    # 1차 키 값들 수집
    primary_values = set()
    for item in data:
        val = get_metadata_value(item, primary_key)
        if val:
            primary_values.add(val)

    # 메타데이터가 없거나 의미 없으면 fallback
    if not primary_values or len(primary_values) < 2:
        print(f"  No meaningful '{primary_key}' metadata found, falling back to diverse sampling")
        return extract_diverse_samples(data, num_samples, text_key)

    print(f"  Found {len(primary_values)} unique {primary_key} values: {sorted(primary_values)}")

    # 1차 키로 그룹화
    by_primary: Dict[str, List[Dict]] = defaultdict(list)
    for item in data:
        val = get_metadata_value(item, primary_key)
        if val:
            by_primary[val].append(item)

    # 각 1차 그룹에서 추출할 샘플 수 계산
    primary_list = sorted(primary_values)
    samples_per_primary = num_samples // len(primary_list)
    remainder = num_samples % len(primary_list)

    samples = []

    for i, primary_val in enumerate(primary_list):
        count = samples_per_primary + (1 if i < remainder else 0)
        primary_group = by_primary.get(primary_val, [])

        if not primary_group:
            continue

        # 2차 키가 있으면 2차 계층 샘플링
        if secondary_key:
            # 2차 키로 서브그룹화
            by_secondary: Dict[str, List[Dict]] = defaultdict(list)
            for item in primary_group:
                sec_val = get_metadata_value(item, secondary_key)
                if sec_val:
                    by_secondary[sec_val].append(item)

            if by_secondary and len(by_secondary) >= 2:
                # 각 2차 그룹에서 1개씩 먼저 추출
                secondary_samples = []
                for sec_val, sec_items in by_secondary.items():
                    if sec_items:
                        secondary_samples.append(random.choice(sec_items))

                # count만큼 선택
                if len(secondary_samples) >= count:
                    selected = random.sample(secondary_samples, count)
                else:
                    # 부족하면 추가 샘플링
                    selected = secondary_samples.copy()
                    remaining = [item for item in primary_group if item not in selected]
                    if remaining:
                        additional = random.sample(remaining, min(count - len(selected), len(remaining)))
                        selected.extend(additional)

                samples.extend(selected)
            else:
                # 2차 키가 없거나 의미 없으면 1차 그룹에서 직접 샘플링 (길이 다양성)
                selected = _sample_with_length_diversity(primary_group, count, text_key)
                samples.extend(selected)
        else:
            # 2차 키 없으면 1차 그룹에서 길이 다양성 기반 샘플링
            selected = _sample_with_length_diversity(primary_group, count, text_key)
            samples.extend(selected)

    return samples


def _sample_with_length_diversity(
    items: List[Dict],
    count: int,
    text_key: str = "input"
) -> List[Dict]:
    """길이 다양성을 고려한 샘플링을 수행합니다.

    Args:
        items: 샘플링 대상 아이템들
        count: 추출할 샘플 수
        text_key: 길이 측정용 필드

    Returns:
        길이 다양성이 확보된 샘플 리스트
    """
    if len(items) <= count:
        return items

    # 길이로 분류
    by_length = categorize_by_length(items, text_key)

    # 각 그룹에서 균등 추출
    samples_per_group = count // 3
    remainder = count % 3

    samples = []
    for i, (group, group_items) in enumerate(by_length.items()):
        group_count = samples_per_group + (1 if i < remainder else 0)
        if group_items:
            selected = random.sample(group_items, min(group_count, len(group_items)))
            samples.extend(selected)

    # 부족하면 남은 것에서 추가
    if len(samples) < count:
        remaining = [item for item in items if item not in samples]
        if remaining:
            additional = random.sample(remaining, min(count - len(samples), len(remaining)))
            samples.extend(additional)

    return samples


def extract_samples(
    domain: str,
    dataset: str,
    num_samples: Optional[int] = None,
    strategy: str = "diverse"
) -> List[Dict]:
    """데이터셋에서 샘플을 추출합니다.

    메타데이터가 있는 데이터셋은 계층적 샘플링을 사용합니다.
    메타데이터가 없거나 의미 없으면 diverse/random 샘플링을 사용합니다.

    Args:
        domain: 도메인 이름 (math, logical, commonsense)
        dataset: 데이터셋 이름 (gsm8k, math, reclor, arc_c)
        num_samples: 샘플 수 (None이면 설정값 사용)
        strategy: 샘플링 전략 ("random", "diverse", "stratified")

    Returns:
        추출된 샘플 리스트

    Raises:
        ValueError: 알 수 없는 도메인 또는 데이터셋인 경우
        FileNotFoundError: 학습 파일이 존재하지 않는 경우
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
    if strategy == "random":
        samples = extract_random_samples(data, final_num)
    elif config.get("use_stratified") or strategy == "stratified":
        # 메타데이터 기반 계층적 샘플링 (로컬 데이터에서 직접)
        # type > level > length 순으로 다양성 확보
        # 메타데이터가 없으면 내부에서 diverse로 fallback
        samples = extract_stratified_samples(
            data,
            final_num,
            primary_key="type",
            secondary_key="level",
            text_key="input"
        )
    else:
        # diverse: 길이 기반 다양성 샘플링
        samples = extract_diverse_samples(data, final_num)

    return samples


def save_samples(
    domain: str,
    dataset: str,
    samples: List[Dict],
    output_dir: Optional[Path] = None
) -> Path:
    """추출된 샘플을 파일로 저장합니다.

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
    """모든 데이터셋에서 샘플을 추출합니다.

    Args:
        strategy: 샘플링 전략

    Returns:
        {domain: {dataset: output_path}} 형식의 결과 딕셔너리
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
    """CLI 진입점."""
    parser = argparse.ArgumentParser(
        description="Extract representative samples from datasets for Instructional Goal generation"
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
    print("Sample Data Extractor for Instructional Goal Generation")
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
