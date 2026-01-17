"""Answer checking and grading for V-STaR"""

import re
from typing import Any, Optional, Union
from config.domains import AnswerType


class AnswerChecker:
    """
    Answer checker for various answer types

    Supports: MCQ, NUMERIC, LATEX, TEXT, BOOLEAN
    """

    @staticmethod
    def extract_answer(response: str, answer_type: AnswerType) -> Optional[str]:
        """
        Extract answer from model response

        Args:
            response: Model response text
            answer_type: Expected answer type

        Returns:
            Extracted answer or None
        """
        if answer_type == AnswerType.MCQ:
            return MCQExtractor.extract(response)
        elif answer_type == AnswerType.NUMERIC:
            return NumericExtractor.extract(response)
        elif answer_type == AnswerType.LATEX:
            return LaTeXExtractor.extract(response)
        elif answer_type == AnswerType.BOOLEAN:
            return BooleanExtractor.extract(response)
        elif answer_type == AnswerType.TEXT:
            return TextExtractor.extract(response)
        return None

    @staticmethod
    def grade(
        predicted: Any,
        ground_truth: Any,
        answer_type: AnswerType
    ) -> bool:
        """
        Grade a predicted answer against ground truth

        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            answer_type: Answer type

        Returns:
            True if correct, False otherwise
        """
        if predicted is None:
            return False

        if answer_type == AnswerType.MCQ:
            return grade_mcq(predicted, ground_truth)
        elif answer_type == AnswerType.NUMERIC:
            return grade_numeric(predicted, ground_truth)
        elif answer_type == AnswerType.LATEX:
            return grade_latex(predicted, ground_truth)
        elif answer_type == AnswerType.BOOLEAN:
            return grade_boolean(predicted, ground_truth)
        elif answer_type == AnswerType.TEXT:
            return grade_text(predicted, ground_truth)
        return False


class MCQExtractor:
    """Extract MCQ answers (A, B, C, D)"""

    @staticmethod
    def extract(response: str) -> Optional[str]:
        # Try boxed format first
        boxed = extract_boxed_answer(response)
        if boxed and boxed.upper() in ['A', 'B', 'C', 'D']:
            return boxed.upper()

        # Try "The answer is X" format
        match = re.search(r'[Tt]he answer is[:\s]*([A-Da-d])', response)
        if match:
            return match.group(1).upper()

        # Try "Answer: X" format
        match = re.search(r'[Aa]nswer[:\s]*([A-Da-d])', response)
        if match:
            return match.group(1).upper()

        # Try to find last occurrence of single letter answer
        matches = re.findall(r'\b([A-Da-d])\b(?:\s*[.)]|\s*$)', response)
        if matches:
            return matches[-1].upper()

        return None


class NumericExtractor:
    """Extract numeric answers"""

    @staticmethod
    def extract(response: str) -> Optional[str]:
        # Try boxed format first
        boxed = extract_boxed_answer(response)
        if boxed:
            # Clean and parse
            clean = boxed.replace(',', '').replace(' ', '').replace('$', '')
            if re.match(r'^-?\d+\.?\d*$', clean):
                return clean

        # Try GSM8K format (#### answer)
        match = re.search(r'####\s*([-\d,.\s]+)', response)
        if match:
            clean = match.group(1).replace(',', '').replace(' ', '')
            if re.match(r'^-?\d+\.?\d*$', clean):
                return clean

        # Try "The answer is X" format
        match = re.search(r'[Tt]he answer is[:\s]*([-\d,.\s]+)', response)
        if match:
            clean = match.group(1).replace(',', '').replace(' ', '')
            if re.match(r'^-?\d+\.?\d*$', clean):
                return clean

        # Find the last number in the response
        matches = re.findall(r'[-]?\d+(?:,\d{3})*(?:\.\d+)?', response)
        if matches:
            clean = matches[-1].replace(',', '')
            return clean

        return None


class LaTeXExtractor:
    """Extract LaTeX answers"""

    @staticmethod
    def extract(response: str) -> Optional[str]:
        # Try boxed format
        boxed = extract_boxed_answer(response)
        if boxed:
            return normalize_latex(boxed)

        # Try to find inline math
        match = re.search(r'\$([^$]+)\$', response)
        if match:
            return normalize_latex(match.group(1))

        return None


class BooleanExtractor:
    """Extract boolean answers (Yes/No)"""

    @staticmethod
    def extract(response: str) -> Optional[str]:
        # Try boxed format
        boxed = extract_boxed_answer(response)
        if boxed:
            lower = boxed.lower()
            if lower in ['yes', 'true', '1']:
                return 'Yes'
            if lower in ['no', 'false', '0']:
                return 'No'

        # Try "The answer is X" format
        match = re.search(r'[Tt]he answer is[:\s]*(yes|no|true|false)', response, re.IGNORECASE)
        if match:
            answer = match.group(1).lower()
            return 'Yes' if answer in ['yes', 'true'] else 'No'

        # Look for Yes/No in the response
        response_lower = response.lower()
        if 'yes' in response_lower and 'no' not in response_lower:
            return 'Yes'
        if 'no' in response_lower and 'yes' not in response_lower:
            return 'No'

        return None


class TextExtractor:
    """Extract free text answers"""

    @staticmethod
    def extract(response: str) -> Optional[str]:
        # Try boxed format
        boxed = extract_boxed_answer(response)
        if boxed:
            return boxed.strip()

        # Try "The answer is X" format
        match = re.search(r'[Tt]he answer is[:\s]*([^\n.]+)', response)
        if match:
            return match.group(1).strip()

        # Return last non-empty line
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        if lines:
            return lines[-1]

        return None


def extract_boxed_answer(response: str) -> Optional[str]:
    """
    Extract answer from \\boxed{} format

    Handles nested braces correctly.
    """
    # Find \boxed{
    idx = response.find('\\boxed{')
    if idx == -1:
        return None

    # Find matching closing brace
    start = idx + 7  # len('\\boxed{')
    depth = 1
    end = start

    while end < len(response) and depth > 0:
        if response[end] == '{':
            depth += 1
        elif response[end] == '}':
            depth -= 1
        end += 1

    if depth == 0:
        return response[start:end-1].strip()

    return None


def normalize_latex(expr: str) -> str:
    """Normalize LaTeX expression for comparison"""
    # Remove common formatting
    expr = expr.strip()
    expr = expr.replace(' ', '')
    expr = expr.replace('\\left', '')
    expr = expr.replace('\\right', '')
    expr = expr.replace('\\,', '')
    expr = expr.replace('\\;', '')
    expr = expr.replace('\\!', '')
    expr = expr.replace('\\text', '')
    expr = expr.replace('\\mathrm', '')
    expr = expr.replace('\\mathbf', '')

    # Remove extra braces
    while expr.startswith('{') and expr.endswith('}'):
        expr = expr[1:-1]

    return expr


def grade_mcq(predicted: str, ground_truth: str) -> bool:
    """Grade MCQ answer"""
    return str(predicted).upper().strip() == str(ground_truth).upper().strip()


def grade_numeric(predicted: Any, ground_truth: Any) -> bool:
    """Grade numeric answer"""
    try:
        # Clean and convert
        if isinstance(predicted, str):
            predicted = predicted.replace(',', '').replace(' ', '')
        if isinstance(ground_truth, str):
            ground_truth = ground_truth.replace(',', '').replace(' ', '')

        pred_float = float(predicted)
        gt_float = float(ground_truth)

        # Check exact match first
        if pred_float == gt_float:
            return True

        # Check with tolerance for floating point
        if abs(pred_float - gt_float) < 1e-6:
            return True

        # Check integer equivalence
        if pred_float == int(pred_float) and gt_float == int(gt_float):
            return int(pred_float) == int(gt_float)

        return False
    except (ValueError, TypeError):
        return str(predicted).strip() == str(ground_truth).strip()


def grade_latex(predicted: str, ground_truth: str) -> bool:
    """Grade LaTeX answer"""
    # Normalize both
    pred_norm = normalize_latex(str(predicted))
    gt_norm = normalize_latex(str(ground_truth))

    if pred_norm == gt_norm:
        return True

    # Try symbolic comparison with sympy
    try:
        import sympy
        from sympy.parsing.latex import parse_latex

        pred_expr = parse_latex(predicted)
        gt_expr = parse_latex(ground_truth)

        # Check symbolic equality
        if sympy.simplify(pred_expr - gt_expr) == 0:
            return True
    except:
        pass

    # Try numeric comparison
    return grade_numeric(predicted, ground_truth)


def grade_boolean(predicted: Any, ground_truth: Any) -> bool:
    """Grade boolean answer"""
    # Normalize to Yes/No
    def normalize_bool(val):
        if isinstance(val, bool):
            return 'Yes' if val else 'No'
        s = str(val).lower().strip()
        if s in ['yes', 'true', '1']:
            return 'Yes'
        if s in ['no', 'false', '0']:
            return 'No'
        return s

    return normalize_bool(predicted) == normalize_bool(ground_truth)


def grade_text(predicted: str, ground_truth: str) -> bool:
    """Grade text answer"""
    # Simple exact match (case-insensitive, whitespace-normalized)
    pred_norm = ' '.join(str(predicted).lower().split())
    gt_norm = ' '.join(str(ground_truth).lower().split())
    return pred_norm == gt_norm


def grade_answer(
    predicted: Any,
    ground_truth: Any,
    answer_type: Union[AnswerType, str]
) -> bool:
    """
    Grade an answer (convenience function)

    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        answer_type: Answer type (AnswerType enum or string)

    Returns:
        True if correct
    """
    if isinstance(answer_type, str):
        answer_type = AnswerType(answer_type)

    return AnswerChecker.grade(predicted, ground_truth, answer_type)
