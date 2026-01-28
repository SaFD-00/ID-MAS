"""답변 추출 전략 모듈 - 다양한 답변 타입별 추출 전략.

이 모듈은 모델 응답에서 답변을 추출하고 정답과 비교하는 기능을 제공합니다.
각 추출기는 특정 답변 타입(MCQ, 숫자, LaTeX, 텍스트, Boolean)을 처리합니다.

주요 클래스:
    AnswerExtractor: 답변 추출기 추상 기본 클래스
    MCQExtractor: 객관식 답변 추출기
    NumericExtractor: 숫자 답변 추출기
    LaTeXExtractor: LaTeX 수식 답변 추출기
    TextExtractor: 자유 형식 텍스트 추출기
    BooleanExtractor: Yes/No, True/False 추출기

주요 함수:
    get_extractor(): 답변 타입에 맞는 추출기 반환
    extract_boxed_answer(): \\boxed{} 형식에서 답변 추출
    grade_answer(): 예측 답변과 정답의 기호적 비교

사용 예시:
    >>> from utils.answer_extractor import get_extractor
    >>> from utils.base_loader import AnswerType
    >>> extractor = get_extractor(AnswerType.NUMERIC)
    >>> answer = extractor.extract("The answer is \\\\boxed{42}")
    >>> is_correct = extractor.compare(answer, "42")
"""
import re
from abc import ABC, abstractmethod
from typing import Optional

# sympy를 사용한 향상된 수학적 비교를 위한 import
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
# 향상된 답변 추출 헬퍼 함수
# =============================================================================

def _strip_string(string: str) -> str:
    """LaTeX 문자열을 비교를 위해 정규화합니다.

    공백을 제거하고, LaTeX 명령을 정규화하며, 포맷팅을 정리합니다.

    Args:
        string: 정규화할 LaTeX 또는 수학 문자열

    Returns:
        정규화된 문자열
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
    """비교를 위해 수학 답변을 정규화합니다.

    LaTeX text 래퍼를 처리하고 문자열 정규화를 적용합니다.

    Args:
        answer: 정규화할 답변 문자열

    Returns:
        정규화된 답변 문자열
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
    """\\boxed{} 또는 \\fbox{} 명령에서 답변을 추출합니다.

    중첩된 중괄호를 깊이 추적을 통해 올바르게 처리합니다.

    Args:
        text: boxed 답변이 포함된 텍스트

    Returns:
        추출된 답변 내용, 찾지 못하면 None
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
    """기호적 비교를 사용하여 예측 답변을 채점합니다.

    2단계 접근 방식:
    1. 문자열 정규화 및 비교
    2. sympy를 사용한 기호적 수학 비교 (사용 가능한 경우)

    Args:
        pred: 예측 답변 (\\boxed{} 포함 가능)
        solution: 정답 (\\boxed{} 포함 가능)

    Returns:
        답변이 일치하면 True, 아니면 False
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
    """답변 추출기의 추상 기본 클래스.

    모든 답변 추출기가 구현해야 하는 인터페이스를 정의합니다.
    """

    @abstractmethod
    def extract(self, response: str) -> Optional[str]:
        """모델 응답에서 답변을 추출합니다.

        Args:
            response: 모델의 텍스트 응답

        Returns:
            추출된 답변 문자열, 찾지 못하면 None
        """
        pass

    @abstractmethod
    def compare(self, predicted: str, ground_truth: str) -> bool:
        """예측 답변과 정답을 비교합니다.

        Args:
            predicted: 모델 응답에서 추출한 답변
            ground_truth: 정답

        Returns:
            답변이 일치하면 True, 아니면 False
        """
        pass


class MCQExtractor(AnswerExtractor):
    """객관식 답변 추출기 (A/B/C/D).

    인식하는 패턴:
        - \\boxed{A} (LaTeX boxed 형식)
        - "Answer: A", "answer is B"
        - "The correct answer is C"
        - "Final Answer: D"
        - 응답 끝의 독립된 문자
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
    """숫자 답변 추출기 (정수, 소수).

    인식하는 패턴:
        - "#### 25" (GSM8K 스타일)
        - "The answer is 3.14"
        - "= 42"
        - 폴백으로 응답의 마지막 숫자
    """

    def extract(self, response: str) -> Optional[str]:
        """응답에서 숫자 답변을 추출합니다.

        2단계 접근 방식:
        1. 현재 메서드 시도 (하위 호환성 - #### 및 \\boxed{} 모두 지원)
        2. 향상된 추출로 폴백
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
        """숫자 답변을 비교합니다.

        2단계 접근 방식:
        1. 현재 비교 메서드 시도 (하위 호환성)
        2. 향상된 기호적 채점으로 폴백
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
    """LaTeX/수학 답변 추출기.

    인식하는 패턴:
        - \\boxed{...}
        - $...$
        - \\frac{...}{...}
        - 숫자 폴백
    """

    def extract(self, response: str) -> Optional[str]:
        """응답에서 LaTeX/수학 답변을 추출합니다.

        중첩된 중괄호를 올바르게 처리합니다 (예: \\frac{1}{8}).
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
        """예측 답변과 정답을 비교합니다.

        3단계 접근 방식:
        1. 현재 비교 메서드 시도 (하위 호환성)
        2. 향상된 기호적 채점으로 폴백
        3. 기본 정규화로 폴백
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
        """비교를 위해 LaTeX 표현식을 정규화합니다."""
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
        """분수 표현식 파싱을 시도합니다."""
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
    """Yes/No, True/False 답변 추출기.

    다양한 형식을 인식합니다:
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
    """자유 형식 텍스트 답변 추출기.

    텍스트 출력을 요구하는 BBH 태스크에 사용됩니다.
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
    """답변 타입에 맞는 추출기를 반환하는 팩토리 함수.

    Args:
        answer_type: AnswerType 열거형 값

    Returns:
        해당하는 AnswerExtractor 인스턴스
    """
    extractors = {
        AnswerType.MCQ: MCQExtractor(),
        AnswerType.NUMERIC: NumericExtractor(),
        AnswerType.LATEX: LaTeXExtractor(),
        AnswerType.BOOLEAN: BooleanExtractor(),
        AnswerType.TEXT: TextExtractor(),
    }
    return extractors.get(answer_type, TextExtractor())
