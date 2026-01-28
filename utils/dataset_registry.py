"""데이터셋 레지스트리 모듈 - 도메인 기반 데이터 로딩.

이 모듈은 ID-MAS의 도메인 로더와 답변 추출기에 대한 중앙 집중식
접근을 제공합니다.

주요 클래스:
    DatasetRegistry: 도메인별 데이터셋 접근을 위한 레지스트리

편의 함수:
    get_domain_loader(): 도메인 로더 인스턴스 반환
    get_available_domains(): 사용 가능한 도메인 목록 반환
    get_eval_datasets_for_domain(): 도메인의 평가 데이터셋 목록 반환
    get_extractor_for_domain(): 도메인의 답변 추출기 반환

사용 예시:
    >>> from utils.dataset_registry import DatasetRegistry
    >>> loader = DatasetRegistry.get_domain_loader("math")
    >>> datasets = DatasetRegistry.get_eval_datasets_for_domain("math")
"""
from typing import Dict, List, Optional

from utils.base_loader import BaseDatasetLoader, AnswerType
from utils.answer_extractor import AnswerExtractor, get_extractor


class DatasetRegistry:
    """도메인 기반 데이터셋 로딩을 위한 레지스트리.

    현재 지원하는 도메인:
        - math: GSM8K, MATH (학습) + SVAMP, ASDiv, MAWPS (평가)
        - logical: ReClor (학습) + ANLI R2/R3, BBH (평가)
        - commonsense: ARC-C (학습) + StrategyQA, OpenBookQA (평가)

    새 도메인은 domain_loader.DomainLoader.DOMAIN_CONFIG를 확장하여 추가할 수 있습니다.
    """

    # Domain configurations
    DOMAIN_CONFIG = {
        "math": {
            "training_datasets": ["gsm8k", "math"],
            "eval_datasets": ["gsm8k", "math", "svamp", "asdiv", "mawps"],
            "default_eval": "gsm8k",
            "default_answer_type": AnswerType.LATEX,  # MATH 데이터셋의 좌표, 분수 등 지원
            "category": "math_logic"
        },
        "logical": {
            "training_datasets": ["reclor"],
            "eval_datasets": ["reclor", "anli_r2", "anli_r3", "bbh"],
            "default_eval": "reclor",
            "default_answer_type": AnswerType.MCQ,
            "category": "logical_reasoning"
        },
        "commonsense": {
            "training_datasets": ["arc_c"],
            "eval_datasets": ["arc_c", "strategyqa", "openbookqa"],
            "default_eval": "arc_c",
            "default_answer_type": AnswerType.MCQ,
            "category": "commonsense_reasoning"
        }
    }

    @classmethod
    def get_domain_loader(cls, domain: str) -> "DomainLoader":
        """도메인 기반 로더 인스턴스를 반환합니다.

        Args:
            domain: 도메인 이름 (예: "math", "logical", "commonsense")

        Returns:
            DomainLoader 인스턴스

        Raises:
            ValueError: 알 수 없는 도메인인 경우
        """
        from utils.domain_loader import DomainLoader
        return DomainLoader(domain)

    @classmethod
    def get_available_domains(cls) -> List[str]:
        """DOMAIN_CONFIG에서 사용 가능한 도메인 목록을 동적으로 반환합니다."""
        return list(cls.DOMAIN_CONFIG.keys())

    @classmethod
    def get_eval_datasets_for_domain(cls, domain: str) -> List[str]:
        """도메인의 사용 가능한 평가 데이터셋을 반환합니다.

        Args:
            domain: 도메인 이름

        Returns:
            평가 데이터셋 이름 리스트

        Raises:
            ValueError: 알 수 없는 도메인인 경우
        """
        if domain not in cls.DOMAIN_CONFIG:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(cls.DOMAIN_CONFIG.keys())}")
        return cls.DOMAIN_CONFIG[domain]["eval_datasets"]

    @classmethod
    def get_training_datasets_for_domain(cls, domain: str) -> List[str]:
        """도메인의 학습 데이터셋을 반환합니다.

        Args:
            domain: 도메인 이름

        Returns:
            학습 데이터셋 이름 리스트

        Raises:
            ValueError: 알 수 없는 도메인인 경우
        """
        if domain not in cls.DOMAIN_CONFIG:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(cls.DOMAIN_CONFIG.keys())}")
        return cls.DOMAIN_CONFIG[domain]["training_datasets"]

    @classmethod
    def get_default_eval_for_domain(cls, domain: str) -> str:
        """도메인의 기본 평가 데이터셋을 반환합니다.

        Args:
            domain: 도메인 이름

        Returns:
            기본 평가 데이터셋 이름

        Raises:
            ValueError: 알 수 없는 도메인인 경우
        """
        if domain not in cls.DOMAIN_CONFIG:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(cls.DOMAIN_CONFIG.keys())}")
        return cls.DOMAIN_CONFIG[domain]["default_eval"]

    @classmethod
    def get_answer_type_for_domain(cls, domain: str) -> AnswerType:
        """도메인의 기본 답변 타입을 반환합니다.

        Args:
            domain: 도메인 이름

        Returns:
            AnswerType 열거형 값

        Raises:
            ValueError: 알 수 없는 도메인인 경우
        """
        if domain not in cls.DOMAIN_CONFIG:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(cls.DOMAIN_CONFIG.keys())}")
        return cls.DOMAIN_CONFIG[domain]["default_answer_type"]

    @classmethod
    def get_extractor_for_domain(cls, domain: str) -> AnswerExtractor:
        """도메인의 답변 추출기를 반환합니다.

        Args:
            domain: 도메인 이름

        Returns:
            AnswerExtractor 인스턴스
        """
        answer_type = cls.get_answer_type_for_domain(domain)
        return get_extractor(answer_type)

    @classmethod
    def get_extractor_for_type(cls, answer_type: AnswerType) -> AnswerExtractor:
        """특정 답변 타입의 추출기를 반환합니다.

        Args:
            answer_type: AnswerType 열거형 값

        Returns:
            AnswerExtractor 인스턴스
        """
        return get_extractor(answer_type)

    @classmethod
    def get_domain_info(cls, domain: str) -> Dict:
        """도메인에 대한 정보를 반환합니다.

        Args:
            domain: 도메인 이름

        Returns:
            도메인 정보를 담은 딕셔너리:
                - name: 도메인 이름
                - training_datasets: 학습 데이터셋 리스트
                - eval_datasets: 평가 데이터셋 리스트
                - default_eval: 기본 평가 데이터셋
                - default_answer_type: 기본 답변 타입
                - category: 도메인 카테고리

        Raises:
            ValueError: 알 수 없는 도메인인 경우
        """
        if domain not in cls.DOMAIN_CONFIG:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(cls.DOMAIN_CONFIG.keys())}")

        config = cls.DOMAIN_CONFIG[domain]
        return {
            "name": domain,
            "training_datasets": config["training_datasets"],
            "eval_datasets": config["eval_datasets"],
            "default_eval": config["default_eval"],
            "default_answer_type": config["default_answer_type"].value,
            "category": config["category"]
        }


# 편의 함수
def get_domain_loader(domain: str) -> "DomainLoader":
    """도메인 로더 인스턴스를 반환합니다."""
    return DatasetRegistry.get_domain_loader(domain)


def get_available_domains() -> List[str]:
    """사용 가능한 도메인 목록을 반환합니다."""
    return DatasetRegistry.get_available_domains()


def get_eval_datasets_for_domain(domain: str) -> List[str]:
    """도메인의 평가 데이터셋 목록을 반환합니다."""
    return DatasetRegistry.get_eval_datasets_for_domain(domain)


def get_extractor_for_domain(domain: str) -> AnswerExtractor:
    """도메인의 답변 추출기를 반환합니다."""
    return DatasetRegistry.get_extractor_for_domain(domain)


if __name__ == "__main__":
    # Test the registry
    print("=" * 60)
    print("Dataset Registry (Domain-based)")
    print("=" * 60)

    print(f"\nAvailable domains: {DatasetRegistry.get_available_domains()}")

    for domain in DatasetRegistry.get_available_domains():
        print(f"\n--- {domain.upper()} Domain ---")
        info = DatasetRegistry.get_domain_info(domain)
        print(f"  Training datasets: {info['training_datasets']}")
        print(f"  Eval datasets: {info['eval_datasets']}")
        print(f"  Default eval: {info['default_eval']}")
        print(f"  Answer type: {info['default_answer_type']}")
