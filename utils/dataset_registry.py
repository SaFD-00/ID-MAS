"""
Dataset Registry - Domain-based data loading for ID-MAS

Provides centralized access to domain loaders and answer extractors.
"""
from typing import Dict, List, Optional

from utils.base_loader import BaseDatasetLoader, AnswerType
from utils.answer_extractor import AnswerExtractor, get_extractor


class DatasetRegistry:
    """
    Registry for domain-based dataset loading.

    Currently supports:
    - math: GSM8K, MATH (training) + SVAMP, ASDiv, MAWPS, MMLU (evaluation)

    New domains can be added by extending domain_loader.DomainLoader.DOMAIN_CONFIG.
    """

    # Domain configurations
    DOMAIN_CONFIG = {
        "math": {
            "training_datasets": ["gsm8k", "math"],
            "eval_datasets": ["gsm8k", "math", "svamp", "asdiv", "mawps", "mmlu"],
            "default_eval": "gsm8k",
            "default_answer_type": AnswerType.LATEX,  # MATH 데이터셋의 좌표, 분수 등 지원
            "category": "math_logic"
        }
    }

    @classmethod
    def get_domain_loader(cls, domain: str) -> "DomainLoader":
        """
        Get domain-based loader instance.

        Args:
            domain: Domain name (예: "math")

        Returns:
            DomainLoader instance

        Raises:
            ValueError: If domain is unknown
        """
        from utils.domain_loader import DomainLoader
        return DomainLoader(domain)

    @classmethod
    def get_available_domains(cls) -> List[str]:
        """Get list of available domains dynamically from DOMAIN_CONFIG."""
        return list(cls.DOMAIN_CONFIG.keys())

    @classmethod
    def get_eval_datasets_for_domain(cls, domain: str) -> List[str]:
        """
        Get available evaluation datasets for a domain.

        Args:
            domain: Domain name

        Returns:
            List of evaluation dataset names
        """
        if domain not in cls.DOMAIN_CONFIG:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(cls.DOMAIN_CONFIG.keys())}")
        return cls.DOMAIN_CONFIG[domain]["eval_datasets"]

    @classmethod
    def get_training_datasets_for_domain(cls, domain: str) -> List[str]:
        """
        Get training datasets for a domain.

        Args:
            domain: Domain name

        Returns:
            List of training dataset names
        """
        if domain not in cls.DOMAIN_CONFIG:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(cls.DOMAIN_CONFIG.keys())}")
        return cls.DOMAIN_CONFIG[domain]["training_datasets"]

    @classmethod
    def get_default_eval_for_domain(cls, domain: str) -> str:
        """
        Get default evaluation dataset for a domain.

        Args:
            domain: Domain name

        Returns:
            Default evaluation dataset name
        """
        if domain not in cls.DOMAIN_CONFIG:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(cls.DOMAIN_CONFIG.keys())}")
        return cls.DOMAIN_CONFIG[domain]["default_eval"]

    @classmethod
    def get_answer_type_for_domain(cls, domain: str) -> AnswerType:
        """
        Get default answer type for a domain.

        Args:
            domain: Domain name

        Returns:
            AnswerType enum value
        """
        if domain not in cls.DOMAIN_CONFIG:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(cls.DOMAIN_CONFIG.keys())}")
        return cls.DOMAIN_CONFIG[domain]["default_answer_type"]

    @classmethod
    def get_extractor_for_domain(cls, domain: str) -> AnswerExtractor:
        """
        Get answer extractor for a domain.

        Args:
            domain: Domain name

        Returns:
            AnswerExtractor instance
        """
        answer_type = cls.get_answer_type_for_domain(domain)
        return get_extractor(answer_type)

    @classmethod
    def get_extractor_for_type(cls, answer_type: AnswerType) -> AnswerExtractor:
        """
        Get answer extractor for a specific answer type.

        Args:
            answer_type: AnswerType enum value

        Returns:
            AnswerExtractor instance
        """
        return get_extractor(answer_type)

    @classmethod
    def get_domain_info(cls, domain: str) -> Dict:
        """
        Get information about a domain.

        Args:
            domain: Domain name

        Returns:
            Dict with domain information
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


# Convenience functions
def get_domain_loader(domain: str) -> "DomainLoader":
    """Get a domain loader instance."""
    return DatasetRegistry.get_domain_loader(domain)


def get_available_domains() -> List[str]:
    """Get list of available domains."""
    return DatasetRegistry.get_available_domains()


def get_eval_datasets_for_domain(domain: str) -> List[str]:
    """Get available evaluation datasets for a domain."""
    return DatasetRegistry.get_eval_datasets_for_domain(domain)


def get_extractor_for_domain(domain: str) -> AnswerExtractor:
    """Get answer extractor for a domain."""
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
