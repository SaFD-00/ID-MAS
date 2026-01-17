"""Prompt templates for V-STaR"""

from typing import Optional, List

# System prompts for each domain
MATH_SYSTEM_PROMPT = """You are a helpful math assistant. Solve the given problem step by step, showing your reasoning clearly.
At the end, provide the final answer in the format: The answer is \\boxed{answer}"""

LOGICAL_SYSTEM_PROMPT = """You are a helpful logical reasoning assistant. Analyze the given context and question carefully.
Consider each option systematically and provide your reasoning.
At the end, provide the final answer in the format: The answer is \\boxed{A/B/C/D}"""

COMMONSENSE_SYSTEM_PROMPT = """You are a helpful assistant for commonsense reasoning. Use your knowledge to answer the question.
Think through the problem carefully and explain your reasoning.
At the end, provide the final answer in the format: The answer is \\boxed{A/B/C/D}"""

# Domain to system prompt mapping
DOMAIN_SYSTEM_PROMPTS = {
    "math": MATH_SYSTEM_PROMPT,
    "logical": LOGICAL_SYSTEM_PROMPT,
    "commonsense": COMMONSENSE_SYSTEM_PROMPT,
}


def get_system_prompt(domain: str) -> str:
    """Get system prompt for a domain"""
    if domain not in DOMAIN_SYSTEM_PROMPTS:
        raise ValueError(f"Unknown domain: {domain}")
    return DOMAIN_SYSTEM_PROMPTS[domain]


def get_solution_prompt(
    question: str,
    domain: str,
    choices: Optional[List[str]] = None,
    include_system: bool = True
) -> str:
    """
    Generate prompt for solution generation

    Args:
        question: The question/problem text
        domain: Domain (math, logical, commonsense)
        choices: Optional list of choices for MCQ
        include_system: Whether to include system prompt

    Returns:
        Formatted prompt string
    """
    parts = []

    if include_system:
        parts.append(get_system_prompt(domain))
        parts.append("")  # Empty line

    # Add question
    parts.append("Question:")
    parts.append(question)

    # Add choices if provided
    if choices:
        parts.append("")
        parts.append("Options:")
        for i, choice in enumerate(choices):
            label = chr(ord('A') + i)
            parts.append(f"{label}. {choice}")

    parts.append("")
    parts.append("Solution:")

    return "\n".join(parts)


def get_verification_prompt(
    question: str,
    solution: str,
    domain: str,
    choices: Optional[List[str]] = None
) -> str:
    """
    Generate prompt for verification (used by DPO verifier)

    Args:
        question: The question/problem text
        solution: The candidate solution
        domain: Domain
        choices: Optional list of choices for MCQ

    Returns:
        Formatted prompt for verification scoring
    """
    parts = []

    # Question
    parts.append("Question:")
    parts.append(question)

    # Add choices if provided
    if choices:
        parts.append("")
        parts.append("Options:")
        for i, choice in enumerate(choices):
            label = chr(ord('A') + i)
            parts.append(f"{label}. {choice}")

    # Solution
    parts.append("")
    parts.append("Solution:")
    parts.append(solution)

    return "\n".join(parts)


def format_chat_messages(
    prompt: str,
    domain: str,
    response: Optional[str] = None
) -> List[dict]:
    """
    Format prompt as chat messages for chat models

    Args:
        prompt: User prompt
        domain: Domain for system prompt
        response: Optional assistant response

    Returns:
        List of message dicts
    """
    messages = [
        {"role": "system", "content": get_system_prompt(domain)},
        {"role": "user", "content": prompt},
    ]

    if response:
        messages.append({"role": "assistant", "content": response})

    return messages


def extract_answer_from_response(response: str) -> Optional[str]:
    """
    Extract answer from model response

    Looks for patterns like:
    - \\boxed{answer}
    - The answer is X
    - Answer: X

    Args:
        response: Model response text

    Returns:
        Extracted answer or None
    """
    import re

    # Try boxed format first
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Try "The answer is X" format
    answer_match = re.search(r'[Tt]he answer is[:\s]+([^\n.]+)', response)
    if answer_match:
        return answer_match.group(1).strip()

    # Try "Answer: X" format
    answer_match = re.search(r'[Aa]nswer[:\s]+([^\n.]+)', response)
    if answer_match:
        return answer_match.group(1).strip()

    return None


# GSM8K specific prompt (following V-STaR paper style)
GSM8K_PROMPT_TEMPLATE = """Solve the following math problem step by step.
Show your work clearly and provide the final numerical answer.

Problem: {question}

Solution:"""

# MATH specific prompt
MATH_PROMPT_TEMPLATE = """Solve the following mathematics problem.
Show your reasoning step by step.
Express your final answer using LaTeX notation in \\boxed{{}}.

Problem: {question}

Solution:"""

# ReClor specific prompt
RECLOR_PROMPT_TEMPLATE = """Read the following context and answer the question by selecting the best option.
Explain your reasoning before giving the final answer.

Context: {context}

Question: {question}

Options:
{options}

Solution:"""

# ARC-C specific prompt
ARC_PROMPT_TEMPLATE = """Answer the following science question by selecting the best option.
Think through the problem carefully.

Question: {question}

Options:
{options}

Solution:"""


def get_dataset_prompt(
    dataset: str,
    question: str,
    context: Optional[str] = None,
    choices: Optional[List[str]] = None
) -> str:
    """
    Get dataset-specific prompt

    Args:
        dataset: Dataset name
        question: Question text
        context: Optional context (for ReClor)
        choices: Optional list of choices

    Returns:
        Formatted prompt
    """
    if dataset == "gsm8k":
        return GSM8K_PROMPT_TEMPLATE.format(question=question)

    elif dataset == "math":
        return MATH_PROMPT_TEMPLATE.format(question=question)

    elif dataset == "reclor":
        options_str = "\n".join(
            f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices or [])
        )
        return RECLOR_PROMPT_TEMPLATE.format(
            context=context or "",
            question=question,
            options=options_str
        )

    elif dataset in ["arc_c", "openbookqa"]:
        options_str = "\n".join(
            f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices or [])
        )
        return ARC_PROMPT_TEMPLATE.format(
            question=question,
            options=options_str
        )

    else:
        # Generic prompt
        return get_solution_prompt(
            question=question,
            domain="math",  # Default
            choices=choices,
            include_system=False
        )
