"""V-STaR Prompt Templates"""

from .templates import (
    get_solution_prompt,
    get_verification_prompt,
    MATH_SYSTEM_PROMPT,
    LOGICAL_SYSTEM_PROMPT,
    COMMONSENSE_SYSTEM_PROMPT,
)

__all__ = [
    "get_solution_prompt",
    "get_verification_prompt",
    "MATH_SYSTEM_PROMPT",
    "LOGICAL_SYSTEM_PROMPT",
    "COMMONSENSE_SYSTEM_PROMPT",
]
