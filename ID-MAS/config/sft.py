"""SFT 모델명 매핑 및 해석 모듈.

이 모듈은 HuggingFace Hub에 업로드된 SFT 파인튜닝 모델의 이름을
생성하고 관리합니다. 기본 모델명을 SFT/SFT_ID-MAS 모델명으로 변환합니다.

주요 상수:
    MODEL_NAME_TO_SHORT: 기본 모델명 → 짧은 이름 매핑

주요 함수:
    get_sft_model_name: SFT 모델 HF Hub 이름 생성
    get_sft_idmas_model_name: SFT_ID-MAS 모델 HF Hub 이름 생성
"""

# 기본 모델명 → HuggingFace Hub 레포지토리용 짧은 이름 매핑
# SaFD-00/{short_name}-{domain} 형식으로 사용됨
MODEL_NAME_TO_SHORT = {
    "meta-llama/Llama-3.1-8B-Instruct": "llama3.1-8b",
    "meta-llama/Llama-3.1-70B-Instruct": "llama3.1-70b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama3.2-3b",
    "meta-llama/Llama-3.3-70B-Instruct": "llama3.3-70b",
    "Qwen/Qwen2.5-3B-Instruct": "qwen2.5-3b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen2.5-7b",
    "Qwen/Qwen2.5-14B-Instruct": "qwen2.5-14b",
    "Qwen/Qwen2.5-72B-Instruct": "qwen2.5-72b",
    "Qwen/Qwen3-4B-Instruct-2507": "qwen3-4b",
}


def _validate_sft_model_and_domain(base_model_name: str, domain: str):
    """SFT 모델 및 도메인 유효성을 검증합니다.

    내부 헬퍼 함수로, get_sft_model_name과 get_sft_idmas_model_name에서 호출됩니다.

    Args:
        base_model_name: 기본 모델명
        domain: 도메인명

    Raises:
        ValueError: 지원하지 않는 모델이거나 알 수 없는 도메인인 경우
    """
    # 순환 import 방지를 위해 함수 내에서 import
    from config.domains import DOMAIN_CONFIG

    if base_model_name not in MODEL_NAME_TO_SHORT:
        raise ValueError(
            f"모델 '{base_model_name}'은(는) SFT 파인튜닝을 지원하지 않습니다.\n"
            f"지원 모델: {list(MODEL_NAME_TO_SHORT.keys())}"
        )

    available_domains = list(DOMAIN_CONFIG.keys())
    if domain not in available_domains:
        raise ValueError(f"도메인은 {available_domains} 중 하나여야 합니다. 입력: {domain}")


def get_sft_model_name(base_model_name: str, domain: str) -> str:
    """SFT 파인튜닝 모델의 HuggingFace Hub 이름을 생성합니다.

    Args:
        base_model_name: 기본 모델명 (예: "Qwen/Qwen2.5-3B-Instruct")
        domain: 도메인명 (예: "math")

    Returns:
        SFT 모델 HF Hub 이름 (예: "SaFD-00/qwen2.5-3b-math")

    Raises:
        ValueError: 지원하지 않는 모델이거나 알 수 없는 도메인인 경우

    Example:
        >>> get_sft_model_name("Qwen/Qwen2.5-3B-Instruct", "math")
        'SaFD-00/qwen2.5-3b-math'
    """
    _validate_sft_model_and_domain(base_model_name, domain)

    short_name = MODEL_NAME_TO_SHORT[base_model_name]
    return f"SaFD-00/{short_name}-{domain}"


def get_sft_idmas_model_name(base_model_name: str, domain: str) -> str:
    """SFT_ID-MAS 파인튜닝 모델의 HuggingFace Hub 이름을 생성합니다.

    ID-MAS 방식으로 학습된 SFT 모델의 이름을 반환합니다.

    Args:
        base_model_name: 기본 모델명 (예: "Qwen/Qwen2.5-3B-Instruct")
        domain: 도메인명 (예: "math")

    Returns:
        SFT_ID-MAS 모델 HF Hub 이름 (예: "SaFD-00/qwen2.5-3b-math_id-mas")

    Raises:
        ValueError: 지원하지 않는 모델이거나 알 수 없는 도메인인 경우

    Example:
        >>> get_sft_idmas_model_name("Qwen/Qwen2.5-3B-Instruct", "math")
        'SaFD-00/qwen2.5-3b-math_id-mas'
    """
    _validate_sft_model_and_domain(base_model_name, domain)

    short_name = MODEL_NAME_TO_SHORT[base_model_name]
    return f"SaFD-00/{short_name}-{domain}_id-mas"
