"""
Tests for answer extraction and comparison utilities.
"""
import pytest
from utils.answer_extractor import (
    get_extractor,
    NumericExtractor,
    LaTeXExtractor,
    MCQExtractor,
    BooleanExtractor,
    TextExtractor,
    extract_boxed_answer,
    mathd_normalize,
    _strip_string
)
from utils.base_loader import AnswerType


class TestAnswerExtractorFactory:
    """Test the get_extractor factory function"""

    def test_get_mcq_extractor(self):
        extractor = get_extractor(AnswerType.MCQ)
        assert isinstance(extractor, MCQExtractor)

    def test_get_numeric_extractor(self):
        extractor = get_extractor(AnswerType.NUMERIC)
        assert isinstance(extractor, NumericExtractor)

    def test_get_latex_extractor(self):
        extractor = get_extractor(AnswerType.LATEX)
        assert isinstance(extractor, LaTeXExtractor)

    def test_get_boolean_extractor(self):
        extractor = get_extractor(AnswerType.BOOLEAN)
        assert isinstance(extractor, BooleanExtractor)

    def test_get_text_extractor(self):
        extractor = get_extractor(AnswerType.TEXT)
        assert isinstance(extractor, TextExtractor)


class TestNumericExtractor:
    """Test numeric answer extraction and comparison"""

    def setup_method(self):
        self.extractor = NumericExtractor()

    def test_extract_simple_number(self):
        response = "The answer is 42."
        assert self.extractor.extract(response) == "42"

    def test_extract_boxed_number(self):
        response = "Therefore, \\boxed{3.14} is the answer."
        assert self.extractor.extract(response) == "3.14"

    def test_extract_negative_number(self):
        response = "The result is \\boxed{-15}"
        assert self.extractor.extract(response) == "-15"

    def test_extract_decimal(self):
        response = "\\boxed{0.5}"
        assert self.extractor.extract(response) == "0.5"

    def test_compare_equal_integers(self):
        assert self.extractor.compare("42", "42") is True

    def test_compare_equal_decimals(self):
        assert self.extractor.compare("3.14", "3.14") is True

    def test_compare_different_numbers(self):
        assert self.extractor.compare("42", "43") is False

    def test_compare_none_predicted(self):
        assert self.extractor.compare(None, "42") is False


class TestLaTeXExtractor:
    """Test LaTeX answer extraction and comparison"""

    def setup_method(self):
        self.extractor = LaTeXExtractor()

    def test_extract_boxed_latex(self):
        response = "The solution is \\boxed{\\frac{1}{2}}"
        result = self.extractor.extract(response)
        assert result is not None
        assert "frac" in result or "1/2" in result

    def test_extract_nested_braces(self):
        response = "\\boxed{x^{2} + y^{2} = r^{2}}"
        result = self.extractor.extract(response)
        assert result is not None

    def test_compare_equivalent_fractions(self):
        # Normalized comparison should handle equivalent forms
        result = self.extractor.compare("\\frac{1}{2}", "\\tfrac{1}{2}")
        assert result is True

    def test_compare_different_expressions(self):
        result = self.extractor.compare("\\frac{1}{2}", "\\frac{1}{3}")
        assert result is False


class TestMCQExtractor:
    """Test multiple choice answer extraction"""

    def setup_method(self):
        self.extractor = MCQExtractor()

    def test_extract_option_a(self):
        response = "The correct answer is A."
        assert self.extractor.extract(response) == "A"

    def test_extract_option_with_parentheses(self):
        response = "Answer: (B)"
        assert self.extractor.extract(response) == "B"

    def test_extract_lowercase_option(self):
        response = "The answer is c"
        result = self.extractor.extract(response)
        assert result is not None
        assert result.upper() == "C"

    def test_compare_same_option(self):
        assert self.extractor.compare("A", "A") is True

    def test_compare_case_insensitive(self):
        assert self.extractor.compare("a", "A") is True

    def test_compare_different_options(self):
        assert self.extractor.compare("A", "B") is False


class TestBooleanExtractor:
    """Test boolean answer extraction"""

    def setup_method(self):
        self.extractor = BooleanExtractor()

    def test_extract_yes(self):
        response = "Yes, that is correct."
        result = self.extractor.extract(response)
        assert result.lower() in ["yes", "true"]

    def test_extract_no(self):
        response = "No, that is incorrect."
        result = self.extractor.extract(response)
        assert result.lower() in ["no", "false"]

    def test_extract_true(self):
        response = "The answer is True"
        result = self.extractor.extract(response)
        assert result.lower() == "true"

    def test_compare_yes_variants(self):
        assert self.extractor.compare("yes", "YES") is True
        assert self.extractor.compare("True", "yes") is True

    def test_compare_no_variants(self):
        assert self.extractor.compare("no", "NO") is True
        assert self.extractor.compare("False", "no") is True

    def test_compare_different_values(self):
        assert self.extractor.compare("yes", "no") is False


class TestTextExtractor:
    """Test free-form text answer extraction"""

    def setup_method(self):
        self.extractor = TextExtractor()

    def test_extract_with_answer_prefix(self):
        response = "Answer: Paris"
        assert self.extractor.extract(response) == "Paris"

    def test_extract_with_final_answer(self):
        response = "The final answer is: London"
        result = self.extractor.extract(response)
        assert "London" in result

    def test_extract_fallback_last_line(self):
        response = "Let me think.\nAfter consideration.\nThe capital is Berlin"
        result = self.extractor.extract(response)
        assert "Berlin" in result

    def test_compare_exact_match(self):
        assert self.extractor.compare("Paris", "Paris") is True

    def test_compare_case_insensitive(self):
        assert self.extractor.compare("Paris", "paris") is True

    def test_compare_with_punctuation(self):
        assert self.extractor.compare("Paris.", "Paris") is True


class TestHelperFunctions:
    """Test helper functions for answer normalization"""

    def test_strip_string_removes_whitespace(self):
        result = _strip_string("  a  b  c  ")
        assert " " not in result
        assert result == "abc"

    def test_strip_string_normalizes_fractions(self):
        result = _strip_string("\\tfrac{1}{2}")
        assert "tfrac" not in result
        assert "frac" in result

    def test_mathd_normalize_extracts_text_wrapper(self):
        result = mathd_normalize("\\text{answer}")
        assert "answer" in result
        assert "\\text" not in result

    def test_extract_boxed_answer_simple(self):
        text = "The answer is \\boxed{42}"
        result = extract_boxed_answer(text)
        assert result == "42"

    def test_extract_boxed_answer_nested_braces(self):
        text = "\\boxed{x^{2}}"
        result = extract_boxed_answer(text)
        assert result == "x^{2}"

    def test_extract_boxed_answer_not_found(self):
        text = "No boxed answer here"
        result = extract_boxed_answer(text)
        assert result is None
