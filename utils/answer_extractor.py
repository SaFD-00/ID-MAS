"""
Answer Extraction Strategies for Different Answer Types

Each extractor handles:
1. extract(): Extract answer from model response
2. compare(): Compare predicted answer with ground truth
"""
import re
from abc import ABC, abstractmethod
from typing import Optional

# Import sympy for enhanced mathematical comparison
try:
    import sympy
    from sympy.parsing import sympy_parser
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    import warnings
    warnings.warn("sympy not available - enhanced math grading will fall back to basic comparison")

from utils.base_loader import AnswerType


# =============================================================================
# Enhanced Answer Extraction Helpers
# =============================================================================

def _strip_string(string: str) -> str:
    """
    Normalize LaTeX strings for comparison.

    Removes whitespace, normalizes LaTeX commands, and cleans formatting.

    Args:
        string: LaTeX or mathematical string to normalize

    Returns:
        Normalized string
    """
    string = str(string)
    # Remove newlines and LaTeX spacing commands
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    # Normalize fraction commands
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    # Remove left/right delimiters
    string = string.replace("\\left", "").replace("\\right", "")
    # Normalize decimals (ensure leading zero)
    string = string.replace(" .", " 0.").replace("{.", "{0.")

    if not string:
        return ""
    if string[0] == ".":
        string = "0" + string

    # Remove all spaces for consistent comparison
    return string.replace(" ", "")


def mathd_normalize(answer: str) -> str:
    """
    Normalize mathematical answer for comparison.

    Handles LaTeX text wrappers and applies string normalization.

    Args:
        answer: Answer string to normalize

    Returns:
        Normalized answer string
    """
    if not answer:
        return ""

    answer = answer.strip()

    # Extract content from \text{...} wrapper
    try:
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m:
            answer = m.group("text").strip()
    except Exception:
        pass

    return _strip_string(answer)


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from \\boxed{} or \\fbox{} command with proper brace matching.

    Handles nested braces correctly by tracking depth.

    Args:
        text: Text containing boxed answer

    Returns:
        Extracted answer content, or None if not found
    """
    if not text:
        return None

    # Find last occurrence of \boxed or \fbox
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
    if idx < 0:
        return None

    # Track brace depth to handle nested braces
    i = idx
    depth = 0
    start_brace = -1

    for j in range(i, len(text)):
        if text[j] == "{":
            if depth == 0:
                start_brace = j
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0 and start_brace != -1:
                # Found matching closing brace
                return text[start_brace + 1:j]

    return None


def grade_answer(pred: str, solution: str) -> bool:
    """
    Grade predicted answer against ground truth using symbolic comparison.

    Two-stage approach:
    1. String normalization and comparison
    2. Symbolic math comparison using sympy (if available)

    Args:
        pred: Predicted answer (may contain \\boxed{})
        solution: Ground truth solution (may contain \\boxed{})

    Returns:
        True if answers match, False otherwise
    """
    if not solution:
        return False

    # Stage 1: Extract boxed answers or use full text
    gt_extracted = extract_boxed_answer(solution) or solution
    pred_extracted = extract_boxed_answer(pred) or pred

    # Stage 2: Normalize both answers
    gt_norm = mathd_normalize(gt_extracted)
    pred_norm = mathd_normalize(pred_extracted)

    # Stage 3: String comparison
    if gt_norm == pred_norm:
        return True

    # Stage 4: Symbolic comparison (if sympy available)
    if not SYMPY_AVAILABLE:
        return False

    try:
        # Compute difference and parse as symbolic expression
        diff = f"({gt_norm})-({pred_norm})".replace("^", "**")
        parsed = sympy_parser.parse_expr(
            diff,
            transformations=(
                sympy_parser.standard_transformations +
                (sympy_parser.implicit_multiplication_application,)
            )
        )
        # Check if difference simplifies to zero
        if sympy.simplify(parsed) == 0:
            return True
    except Exception:
        # If symbolic parsing fails, fall back to string comparison
        pass

    return False


class AnswerExtractor(ABC):
    """Abstract base class for answer extractors"""

    @abstractmethod
    def extract(self, response: str) -> Optional[str]:
        """
        Extract answer from model response.

        Args:
            response: Model's text response

        Returns:
            Extracted answer string, or None if not found
        """
        pass

    @abstractmethod
    def compare(self, predicted: str, ground_truth: str) -> bool:
        """
        Compare predicted answer with ground truth.

        Args:
            predicted: Extracted answer from model response
            ground_truth: Correct answer

        Returns:
            True if answers match, False otherwise
        """
        pass


class MCQExtractor(AnswerExtractor):
    """
    Extract Multiple Choice answers (A/B/C/D).

    Patterns recognized:
    - \\boxed{A} (LaTeX boxed format - new)
    - "Answer: A", "answer is B"
    - "The correct answer is C"
    - "Final Answer: D"
    - Standalone letter at end of response
    """

    def extract(self, response: str) -> Optional[str]:
        # Pattern 1: \boxed{letter} (new format)
        boxed_result = extract_boxed_answer(response)
        if boxed_result:
            # Extract letter from boxed content
            letter_match = re.search(r'^([A-Da-d])$', boxed_result.strip())
            if letter_match:
                return letter_match.group(1).upper()
            # If boxed contains just a letter, return it
            if len(boxed_result.strip()) == 1 and boxed_result.strip().upper() in ['A', 'B', 'C', 'D']:
                return boxed_result.strip().upper()

        # Pattern 2: Various "answer is X" formats
        patterns = [
            r'[Aa]nswer[:\s]+([A-Da-d])',
            r'[Tt]he\s+(?:correct\s+)?answer\s+is[:\s]+([A-Da-d])',
            r'[Ff]inal\s+[Aa]nswer[:\s]+([A-Da-d])',
            r'^([A-Da-d])\.',  # Line starts with "A."
            r'\b([A-Da-d])\s+is\s+(?:the\s+)?correct',
            r'\(([A-Da-d])\)',  # Answer in parentheses
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.MULTILINE)
            if match:
                return match.group(1).upper()

        # Fallback: Check last 5 lines for standalone letter
        lines = response.strip().split('\n')
        for line in reversed(lines[-5:]):
            line = line.strip()
            if line in ['A', 'B', 'C', 'D', 'A.', 'B.', 'C.', 'D.']:
                return line[0].upper()

        return None

    def compare(self, predicted: str, ground_truth: str) -> bool:
        if predicted is None:
            return False

        # Extract from boxed format if present in ground truth
        gt_extracted = extract_boxed_answer(str(ground_truth))
        if gt_extracted:
            ground_truth = gt_extracted

        return predicted.upper().strip() == str(ground_truth).upper().strip()


class NumericExtractor(AnswerExtractor):
    """
    Extract numeric answers (integers, decimals).

    Patterns recognized:
    - "#### 25" (GSM8K style)
    - "The answer is 3.14"
    - "= 42"
    - Last number in response as fallback
    """

    def extract(self, response: str) -> Optional[str]:
        """
        Extract numeric answer from response.

        Two-stage approach:
        1. Try current method (backward compatibility - supports both #### and \\boxed{})
        2. Fall back to enhanced extraction
        """
        # Stage 1: Try current extraction method first
        # Pattern 1: GSM8K style "#### <number>" (for backward compatibility)
        match = re.search(r'####\s*([-]?\d+(?:[.,]\d+)?)', response)
        if match:
            return match.group(1).replace(',', '')

        # Pattern 2: \boxed{number} (new format)
        boxed_result = extract_boxed_answer(response)
        if boxed_result:
            # Try to extract number from boxed content
            num_match = re.search(r'([-]?\d+(?:[.,]\d+)?)', boxed_result)
            if num_match:
                return num_match.group(1).replace(',', '')
            # Return boxed content as-is if it looks numeric
            return boxed_result

        # Pattern 3: Various "answer is X" formats
        patterns = [
            r'[Aa]nswer[:\s]+\$?([-]?\d+(?:[.,]\d+)?)\$?',
            r'[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+\$?([-]?\d+(?:[.,]\d+)?)\$?',
            r'[Ff]inal\s+[Aa]nswer[:\s]+\$?([-]?\d+(?:[.,]\d+)?)\$?',
            r'=\s*\$?([-]?\d+(?:[.,]\d+)?)\$?\s*$',
            r'[Aa]nswer[:\s]+\*?\*?([-]?\d+(?:[.,]\d+)?)\*?\*?',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.MULTILINE)
            if match:
                return match.group(1).replace(',', '')

        # Fallback: Extract last number from response
        # Look for numbers that appear after common answer indicators
        answer_section = response
        if '####' in response:
            answer_section = response.split('####')[-1]
        elif 'answer' in response.lower():
            parts = re.split(r'[Aa]nswer', response)
            if len(parts) > 1:
                answer_section = parts[-1]

        numbers = re.findall(r'[-]?\d+(?:\.\d+)?', answer_section)
        if numbers:
            return numbers[-1]

        # Last resort: any number in response
        numbers = re.findall(r'[-]?\d+(?:\.\d+)?', response)
        if numbers:
            return numbers[-1]

        return None

    def compare(self, predicted: str, ground_truth: str) -> bool:
        """
        Compare numeric answers.

        Two-stage approach:
        1. Try current comparison method (backward compatibility)
        2. Fall back to enhanced symbolic grading
        """
        if predicted is None:
            return False

        # Stage 1: Try current comparison method first
        try:
            pred_val = float(predicted.replace(',', ''))
            gt_val = float(str(ground_truth).replace(',', ''))
            # Exact match for integers, small tolerance for floats
            if pred_val == gt_val:
                return True
            # Allow small floating point tolerance
            if abs(pred_val - gt_val) < 1e-6:
                return True
        except (ValueError, TypeError):
            pass

        # Stage 2: Fall back to enhanced symbolic grading
        # This handles cases like fractions, expressions, etc.
        if grade_answer(str(predicted), str(ground_truth)):
            return True

        # Stage 3: String comparison as last resort
        return str(predicted).strip() == str(ground_truth).strip()


class LaTeXExtractor(AnswerExtractor):
    """
    Extract LaTeX/mathematical answers.

    Patterns recognized:
    - \\boxed{...}
    - $...$
    - \\frac{...}{...}
    - Numeric fallback
    """

    def extract(self, response: str) -> Optional[str]:
        """
        Extract LaTeX/mathematical answer from response.

        Handles nested braces correctly (e.g., \\frac{1}{8}).
        """
        # Pattern 1: \boxed{...} or \fbox{...} with proper nested brace handling
        boxed_result = extract_boxed_answer(response)
        if boxed_result:
            return boxed_result

        # Pattern 2: Answer after "is" with LaTeX
        match = re.search(r'answer\s+is[:\s]+\$?([^$\n]+)\$?', response, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Clean up LaTeX artifacts
            answer = answer.rstrip('.')
            if answer:
                return answer

        # Fallback: Try numeric extraction
        numeric_extractor = NumericExtractor()
        return numeric_extractor.extract(response)

    def compare(self, predicted: str, ground_truth: str) -> bool:
        """
        Compare predicted answer with ground truth.

        Three-stage approach:
        1. Try current comparison method (backward compatibility)
        2. Fall back to enhanced symbolic grading
        3. Fall back to basic normalization
        """
        if predicted is None:
            return False

        # Stage 1: Try current comparison method first
        # Normalize both for comparison
        pred_normalized = self._normalize_latex(predicted)
        gt_normalized = self._normalize_latex(ground_truth)

        # Direct string comparison
        if pred_normalized == gt_normalized:
            return True

        # Try numeric comparison
        try:
            pred_val = float(pred_normalized)
            gt_val = float(gt_normalized)
            if abs(pred_val - gt_val) < 1e-6:
                return True
        except ValueError:
            pass

        # Fraction comparison
        pred_frac = self._parse_fraction(predicted)
        gt_frac = self._parse_fraction(ground_truth)
        if pred_frac is not None and gt_frac is not None:
            if abs(pred_frac - gt_frac) < 1e-6:
                return True

        # Stage 2: Fall back to enhanced symbolic grading
        if grade_answer(predicted, ground_truth):
            return True

        return False

    def _normalize_latex(self, expr: str) -> str:
        """Normalize LaTeX expression for comparison"""
        if expr is None:
            return ""
        expr = str(expr)
        # Remove common LaTeX wrappers
        expr = re.sub(r'\\(?:text|mathrm|mathbf|textbf)\{([^}]+)\}', r'\1', expr)
        expr = re.sub(r'\$([^$]+)\$', r'\1', expr)
        # Remove LaTeX commands but keep content
        expr = expr.replace('\\', '')
        expr = expr.replace('{', '').replace('}', '')
        expr = expr.replace(' ', '').strip()
        return expr

    def _parse_fraction(self, expr: str) -> Optional[float]:
        """Try to parse a fraction expression"""
        if expr is None:
            return None
        expr = str(expr)
        # Match \frac{a}{b} or a/b
        match = re.search(r'frac\{?(\d+)\}?\{?(\d+)\}?', expr)
        if match:
            try:
                return float(match.group(1)) / float(match.group(2))
            except (ValueError, ZeroDivisionError):
                pass

        match = re.search(r'(\d+)/(\d+)', expr)
        if match:
            try:
                return float(match.group(1)) / float(match.group(2))
            except (ValueError, ZeroDivisionError):
                pass

        return None


class BooleanExtractor(AnswerExtractor):
    """
    Extract Yes/No, True/False answers.

    Recognizes various forms:
    - Yes/No
    - True/False
    - Valid/Invalid
    """

    YES_VARIANTS = {'yes', 'true', 'valid', 'correct', 'y'}
    NO_VARIANTS = {'no', 'false', 'invalid', 'incorrect', 'n'}

    def extract(self, response: str) -> Optional[str]:
        response_lower = response.lower()

        # Look for explicit patterns
        patterns = [
            r'[Aa]nswer[:\s]+(Yes|No|True|False)',
            r'[Tt]he\s+answer\s+is[:\s]+(Yes|No|True|False)',
            r'^(Yes|No|True|False)[.\s]*$',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.MULTILINE | re.IGNORECASE)
            if match:
                answer = match.group(1).lower()
                return "Yes" if answer in self.YES_VARIANTS else "No"

        # Check for keywords in response
        # Look for the last occurrence as it's often the final answer
        last_yes = -1
        last_no = -1

        for variant in self.YES_VARIANTS:
            idx = response_lower.rfind(variant)
            if idx > last_yes:
                last_yes = idx

        for variant in self.NO_VARIANTS:
            idx = response_lower.rfind(variant)
            if idx > last_no:
                last_no = idx

        if last_yes > last_no:
            return "Yes"
        elif last_no > last_yes:
            return "No"

        return None

    def compare(self, predicted: str, ground_truth: str) -> bool:
        if predicted is None:
            return False

        pred_normalized = predicted.lower().strip()
        gt_normalized = str(ground_truth).lower().strip()

        # Both are "yes" variants
        if pred_normalized in self.YES_VARIANTS and gt_normalized in self.YES_VARIANTS:
            return True

        # Both are "no" variants
        if pred_normalized in self.NO_VARIANTS and gt_normalized in self.NO_VARIANTS:
            return True

        # Exact match
        return pred_normalized == gt_normalized


class TextExtractor(AnswerExtractor):
    """
    Extract free-form text answers.

    Used for BBH tasks with text outputs.
    """

    def extract(self, response: str) -> Optional[str]:
        # Pattern 1: "Answer: ..." or "The answer is ..."
        patterns = [
            r'[Aa]nswer[:\s]+(.+?)(?:\n|$)',
            r'[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\n|$)',
            r'[Ff]inal\s+[Aa]nswer[:\s]+(.+?)(?:\n|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                answer = match.group(1).strip()
                # Clean up common artifacts
                answer = answer.rstrip('.')
                if answer:
                    return answer

        # Fallback: Return last non-empty line
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        if lines:
            return lines[-1]

        return None

    def compare(self, predicted: str, ground_truth: str) -> bool:
        if predicted is None:
            return False

        # Normalize for comparison
        pred_normalized = predicted.lower().strip()
        gt_normalized = str(ground_truth).lower().strip()

        # Exact match
        if pred_normalized == gt_normalized:
            return True

        # Remove punctuation and compare
        pred_clean = re.sub(r'[^\w\s]', '', pred_normalized)
        gt_clean = re.sub(r'[^\w\s]', '', gt_normalized)

        return pred_clean == gt_clean


def get_extractor(answer_type: AnswerType) -> AnswerExtractor:
    """
    Factory function to get appropriate extractor for answer type.

    Args:
        answer_type: AnswerType enum value

    Returns:
        Corresponding AnswerExtractor instance
    """
    extractors = {
        AnswerType.MCQ: MCQExtractor(),
        AnswerType.NUMERIC: NumericExtractor(),
        AnswerType.LATEX: LaTeXExtractor(),
        AnswerType.BOOLEAN: BooleanExtractor(),
        AnswerType.TEXT: TextExtractor(),
    }
    return extractors.get(answer_type, TextExtractor())
