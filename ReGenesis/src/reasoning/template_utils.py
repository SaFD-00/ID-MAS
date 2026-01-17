"""Template utilities for multi-model support.

Provides unified interface for chat templates across Llama 3.x and Qwen2.5 families.
"""

from typing import Optional, List, Dict, Any
from transformers import AutoTokenizer
from functools import lru_cache


# Llama 3 chat template
LLAMA3_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

# Qwen chat template
QWEN_TEMPLATE = """<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""

# Template mapping
TEMPLATES = {
    "llama-3": LLAMA3_TEMPLATE,
    "qwen": QWEN_TEMPLATE,
}

# Default system messages
DEFAULT_SYSTEM_MESSAGES = {
    "llama-3": "You are a helpful, respectful and honest assistant.",
    "qwen": "You are a helpful assistant.",
}


def get_chat_template(template_name: str) -> str:
    """Get chat template by name.

    Args:
        template_name: Template name ("llama-3" or "qwen")

    Returns:
        Template string
    """
    if template_name not in TEMPLATES:
        raise ValueError(
            f"Unknown template: {template_name}. Available: {list(TEMPLATES.keys())}"
        )
    return TEMPLATES[template_name]


def format_prompt(
    template_name: str,
    user_message: str,
    system_message: Optional[str] = None,
) -> str:
    """Format a prompt using the specified template.

    Args:
        template_name: Template name ("llama-3" or "qwen")
        user_message: User message content
        system_message: Optional system message (uses default if not provided)

    Returns:
        Formatted prompt string
    """
    template = get_chat_template(template_name)

    if system_message is None:
        system_message = DEFAULT_SYSTEM_MESSAGES.get(template_name, "")

    return template.format(
        system_message=system_message,
        user_message=user_message,
    )


def format_prompt_with_response(
    template_name: str,
    user_message: str,
    assistant_message: str,
    system_message: Optional[str] = None,
) -> str:
    """Format a complete prompt with response (for training).

    Args:
        template_name: Template name
        user_message: User message content
        assistant_message: Assistant response
        system_message: Optional system message

    Returns:
        Formatted prompt with response
    """
    prompt = format_prompt(template_name, user_message, system_message)

    if template_name == "llama-3":
        return prompt + assistant_message + "<|eot_id|>"
    elif template_name == "qwen":
        return prompt + assistant_message + "<|im_end|>"
    else:
        return prompt + assistant_message


@lru_cache(maxsize=16)
def get_tokenizer(model_name: str) -> AutoTokenizer:
    """Get tokenizer for model (cached).

    Args:
        model_name: HuggingFace model name

    Returns:
        AutoTokenizer instance
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    return tokenizer


def apply_chat_template(
    model_name: str,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
    tokenizer: Optional[AutoTokenizer] = None,
) -> str:
    """Apply chat template using model's native tokenizer.

    This uses the model's built-in chat template for maximum compatibility.

    Args:
        model_name: HuggingFace model name
        messages: List of message dicts with "role" and "content" keys
        add_generation_prompt: Whether to add generation prompt
        tokenizer: Optional pre-loaded tokenizer

    Returns:
        Formatted prompt string
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model_name)

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def get_stop_tokens(template_name: str) -> List[str]:
    """Get stop tokens for template.

    Args:
        template_name: Template name

    Returns:
        List of stop token strings
    """
    stop_tokens = {
        "llama-3": ["<|eot_id|>", "<|end_of_text|>"],
        "qwen": ["<|im_end|>", "<|endoftext|>"],
    }
    return stop_tokens.get(template_name, [])


def detect_template_from_model(model_name: str) -> str:
    """Detect appropriate template from model name.

    Args:
        model_name: Model name or path

    Returns:
        Template name
    """
    model_lower = model_name.lower()

    if "llama" in model_lower:
        return "llama-3"
    elif "qwen" in model_lower:
        return "qwen"
    else:
        # Default to llama-3 for unknown models
        return "llama-3"


# Reasoning-specific prompt templates
MATH_INSTRUCTION_PREFIX = (
    "Solve the following math problem step-by-step.\n"
    "Present your final answer as \\boxed{Your Answer}.\n\n"
)

LOGICAL_INSTRUCTION_PREFIX = (
    "Read the following passage and question carefully, then select the best answer.\n"
    "Present your final answer as \\boxed{Your Answer} (A, B, C, or D).\n\n"
)

COMMONSENSE_INSTRUCTION_PREFIX = (
    "Answer the following question by selecting the best option.\n"
    "Present your final answer as \\boxed{Your Answer} (A, B, C, or D).\n\n"
)


def get_instruction_prefix(dataset_type: str) -> str:
    """Get instruction prefix for dataset type.

    Args:
        dataset_type: One of "math", "logical", "commonsense"

    Returns:
        Instruction prefix string
    """
    prefixes = {
        "math": MATH_INSTRUCTION_PREFIX,
        "logical": LOGICAL_INSTRUCTION_PREFIX,
        "commonsense": COMMONSENSE_INSTRUCTION_PREFIX,
    }
    return prefixes.get(dataset_type, "")
