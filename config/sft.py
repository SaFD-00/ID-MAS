"""
SFT model name mappings and resolution.
"""

# Model name mapping for SFT fine-tuned models on HuggingFace Hub
# Maps base model names to short names used in SaFD-00/{model}-{domain} repos
MODEL_NAME_TO_SHORT = {
    "Qwen/Qwen2.5-3B-Instruct": "qwen2.5-3b",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3.1-8b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen2.5-7b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama3.2-3b",
    "Qwen/Qwen3-4B-Instruct-2507": "qwen3-4b"
}


def _validate_sft_model_and_domain(base_model_name: str, domain: str):
    """
    Helper function to validate model and domain for SFT models.

    Args:
        base_model_name: Base model name
        domain: Domain name

    Raises:
        ValueError: If model or domain is not supported
    """
    # Import here to avoid circular dependency
    from config.domains import DOMAIN_CONFIG

    if base_model_name not in MODEL_NAME_TO_SHORT:
        raise ValueError(
            f"Model '{base_model_name}' not supported for SFT fine-tuning.\n"
            f"Supported models: {list(MODEL_NAME_TO_SHORT.keys())}"
        )

    available_domains = list(DOMAIN_CONFIG.keys())
    if domain not in available_domains:
        raise ValueError(f"Domain must be one of {available_domains}, got: {domain}")


def get_sft_model_name(base_model_name: str, domain: str) -> str:
    """
    Get SFT fine-tuned model name from HuggingFace Hub.

    Args:
        base_model_name: Base model name (e.g., "Qwen/Qwen2.5-3B-Instruct")
        domain: Domain name (e.g., "math")

    Returns:
        SFT model HF Hub name (e.g., "SaFD-00/qwen2.5-3b-math")

    Raises:
        ValueError: If model is not supported for SFT or domain is invalid

    Example:
        >>> get_sft_model_name("Qwen/Qwen2.5-3B-Instruct", "math")
        'SaFD-00/qwen2.5-3b-math'
    """
    _validate_sft_model_and_domain(base_model_name, domain)

    short_name = MODEL_NAME_TO_SHORT[base_model_name]
    return f"SaFD-00/{short_name}-{domain}"


def get_sft_idmas_model_name(base_model_name: str, domain: str) -> str:
    """
    Get SFT_ID-MAS fine-tuned model name from HuggingFace Hub.

    Args:
        base_model_name: Base model name (e.g., "Qwen/Qwen2.5-3B-Instruct")
        domain: Domain name (e.g., "math")

    Returns:
        SFT_ID-MAS model HF Hub name (e.g., "SaFD-00/qwen2.5-3b-math-ID-MAS")

    Raises:
        ValueError: If model is not supported or domain is invalid

    Example:
        >>> get_sft_idmas_model_name("Qwen/Qwen2.5-3B-Instruct", "math")
        'SaFD-00/qwen2.5-3b-math-ID-MAS'
    """
    _validate_sft_model_and_domain(base_model_name, domain)

    short_name = MODEL_NAME_TO_SHORT[base_model_name]
    return f"SaFD-00/{short_name}-{domain}_id-mas"
