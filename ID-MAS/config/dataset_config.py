"""데이터셋별 상세 설정 모듈.

이 모듈은 ID-MAS 시스템에서 지원하는 각 데이터셋의 상세 설정을 관리합니다.
HuggingFace 경로, split 이름, 답변 타입 등의 정보를 포함합니다.

주요 상수:
    DATASET_CONFIGS: 데이터셋별 설정 딕셔너리
    BBH_SUBTASKS: Big Bench Hard 하위 태스크 목록

주요 함수:
    get_dataset_config: 데이터셋 설정 조회
    get_datasets_by_domain: 도메인별 데이터셋 목록
    get_training_datasets: 학습 가능한 데이터셋 목록
    get_evaluation_datasets: 평가 가능한 데이터셋 목록
"""
from typing import Dict, Any, List

# 데이터셋별 상세 설정
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ==========================================================================
    # 수학 도메인 (math)
    # ==========================================================================
    "gsm8k": {
        "hf_name": "openai/gsm8k",
        "hf_config": "main",
        "train_split": "train",
        "test_split": "test",
        "answer_type": "numeric",
        "domain": "math_logic",
        "description": "Grade School Math 8K - 초등학교 수준 수학 문제",
    },
    "math": {
        "hf_name": "hendrycks/competition_math",
        "hf_config": None,
        "train_split": "train",
        "test_split": "test",
        "answer_type": "latex",
        "domain": "math_logic",
        "default_levels": [1, 2, 3],
        "description": "Competition Mathematics - 고등학교 경시대회 수준 문제",
    },
    "svamp": {
        "hf_name": "ChilleD/SVAMP",
        "hf_config": None,
        "train_split": None,  # 평가 전용
        "test_split": "test",
        "answer_type": "numeric",
        "domain": "math_logic",
        "description": "SVAMP - 수학 문제 변형을 통한 강건성 평가",
    },

    # ==========================================================================
    # 논리 도메인 (logical)
    # ==========================================================================
    "bbh": {
        "hf_name": "lukaemon/bbh",
        "hf_config": None,  # 통합 파일, subtask 정보는 metadata에
        "train_split": None,
        "test_split": "test",
        "answer_type": "text",  # 혼합이지만 기본값은 text
        "domain": "logical",
        "description": "Big Bench Hard - 논리 추론 통합 벤치마크",
    },
    "scibench": {
        "hf_name": "xw27/scibench",
        "hf_config": None,
        "train_split": "train",
        "test_split": "test",
        "answer_type": "numeric",
        "domain": "science_knowledge",
        "description": "SciBench - 대학 수준 과학 문제 (물리, 화학)",
    },
    "arc": {
        "hf_name": "allenai/ai2_arc",
        "hf_config": "ARC-Challenge",
        "train_split": "train",
        "test_split": "test",
        "answer_type": "mcq",
        "domain": "science_knowledge",
        "description": "ARC - 초등 과학 객관식 문제",
    },
    "reclor": {
        "hf_name": "sxiong/ReClor",
        "hf_config": None,
        "train_split": "train",
        "test_split": "test",
        "answer_type": "mcq",
        "domain": "logical",
        "description": "ReClor - 표준화 시험 기반 논리 추론",
    },
    "anli_r2": {
        "hf_name": "facebook/anli",
        "hf_config": None,
        "train_split": "train_r2",
        "test_split": "test_r2",
        "answer_type": "mcq",
        "domain": "logical",
        "description": "ANLI Round 2 - 적대적 자연어 추론",
    },
    "anli_r3": {
        "hf_name": "facebook/anli",
        "hf_config": None,
        "train_split": "train_r3",
        "test_split": "test_r3",
        "answer_type": "mcq",
        "domain": "logical",
        "description": "ANLI Round 3 - 적대적 자연어 추론 (고난도)",
    },

    # ==========================================================================
    # 상식 도메인 (commonsense)
    # ==========================================================================
    "arc_c": {
        "hf_name": "allenai/ai2_arc",
        "hf_config": "ARC-Challenge",
        "train_split": "train",
        "test_split": "test",
        "answer_type": "mcq",
        "domain": "commonsense",
        "description": "ARC Challenge - 초등 과학 상식 문제",
    },
    "strategyqa": {
        "hf_name": "ChilleD/StrategyQA",
        "hf_config": None,
        "train_split": None,  # train split 없음
        "test_split": "test",
        "answer_type": "boolean",
        "domain": "commonsense",
        "description": "StrategyQA - 다단계 상식 추론 Yes/No 문제",
    },
    "openbookqa": {
        "hf_name": "allenai/openbookqa",
        "hf_config": "main",
        "train_split": "train",
        "test_split": "test",
        "answer_type": "mcq",
        "domain": "commonsense",
        "description": "OpenBookQA - 오픈북 형식 초등 과학 문제",
    },
}


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """데이터셋 설정을 조회합니다.

    Args:
        dataset_name: 데이터셋명 (예: "gsm8k", "math")

    Returns:
        데이터셋 설정 딕셔너리:
        - hf_name: HuggingFace 데이터셋 경로
        - hf_config: HuggingFace config 이름
        - train_split: 학습 split 이름
        - test_split: 테스트 split 이름
        - answer_type: 답변 타입 (numeric, latex, mcq, text, boolean)
        - domain: 소속 도메인
        - description: 데이터셋 설명

    Raises:
        ValueError: 알 수 없는 데이터셋인 경우
    """
    config = DATASET_CONFIGS.get(dataset_name.lower())
    if config is None:
        available = list(DATASET_CONFIGS.keys())
        raise ValueError(
            f"알 수 없는 데이터셋: {dataset_name}. 가능한 데이터셋: {available}"
        )
    return config


def get_datasets_by_domain(domain: str) -> List[str]:
    """특정 도메인의 데이터셋 목록을 반환합니다.

    Args:
        domain: 도메인명 (예: "math_logic", "logical", "commonsense")

    Returns:
        해당 도메인의 데이터셋명 리스트
    """
    return [
        name for name, config in DATASET_CONFIGS.items()
        if config["domain"] == domain
    ]


def get_training_datasets(domain: str) -> List[str]:
    """특정 도메인에서 학습 가능한 데이터셋 목록을 반환합니다.

    train_split이 정의된 데이터셋만 반환합니다.

    Args:
        domain: 도메인명

    Returns:
        학습 가능한 데이터셋명 리스트
    """
    return [
        name for name, config in DATASET_CONFIGS.items()
        if config["domain"] == domain and config["train_split"] is not None
    ]


def get_evaluation_datasets(domain: str) -> List[str]:
    """특정 도메인에서 평가 가능한 데이터셋 목록을 반환합니다.

    test_split이 정의된 데이터셋만 반환합니다.

    Args:
        domain: 도메인명

    Returns:
        평가 가능한 데이터셋명 리스트
    """
    return [
        name for name, config in DATASET_CONFIGS.items()
        if config["domain"] == domain and config["test_split"] is not None
    ]


# Big Bench Hard 하위 태스크 목록 (참조용)
BBH_SUBTASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]
