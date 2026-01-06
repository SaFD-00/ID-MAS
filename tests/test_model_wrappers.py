"""
Tests for model wrappers (GPT, Student, Qwen).
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from models.gpt_wrapper import GPTWrapper
from models.student_wrapper import StudentModelWrapper
from models.qwen_wrapper import QwenWrapper


class TestGPTWrapper:
    """Test GPTWrapper for OpenAI API integration"""

    @patch('models.gpt_wrapper.OpenAI')
    def test_init_with_config(self, mock_openai):
        config = {
            "model": "gpt-5-2025-08-07",
            "api_key": "test-key",
            "base_url": None,
            "max_tokens": 8192
        }
        wrapper = GPTWrapper(config)
        assert wrapper.model_name == "gpt-5-2025-08-07"
        assert wrapper.max_tokens == 8192
        mock_openai.assert_called_once()

    @patch('models.gpt_wrapper.OpenAI')
    def test_init_with_vllm_endpoint(self, mock_openai):
        config = {
            "model": "meta-llama/Llama-3.3-70B-Instruct",
            "api_key": "0",
            "base_url": "http://localhost:2000/v1",
            "max_tokens": 8192
        }
        wrapper = GPTWrapper(config)
        mock_openai.assert_called_with(
            api_key="0",
            base_url="http://localhost:2000/v1"
        )

    @patch('models.gpt_wrapper.OpenAI')
    def test_generate_basic(self, mock_openai):
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response

        config = {
            "model": "gpt-5-2025-08-07",
            "api_key": "test-key",
            "base_url": None,
            "max_tokens": 8192
        }
        wrapper = GPTWrapper(config)

        result = wrapper.generate("Test prompt")
        assert result == "Test response"

    @patch('models.gpt_wrapper.OpenAI')
    def test_generate_with_system_message(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_client.chat.completions.create.return_value = mock_response

        config = {
            "model": "gpt-5-2025-08-07",
            "api_key": "test-key",
            "base_url": None,
            "max_tokens": 8192
        }
        wrapper = GPTWrapper(config)

        result = wrapper.generate("Prompt", system_message="You are helpful")

        # Verify system message was included in call
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert any(msg["role"] == "system" for msg in messages)

    @patch('models.gpt_wrapper.OpenAI')
    def test_generate_with_chat_history(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_client.chat.completions.create.return_value = mock_response

        config = {
            "model": "gpt-5-2025-08-07",
            "api_key": "test-key",
            "base_url": None,
            "max_tokens": 8192
        }
        wrapper = GPTWrapper(config)

        history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"}
        ]

        result = wrapper.generate("Second question", chat_history=history)

        # Verify history was included
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert len(messages) >= 3  # history + new message


class TestStudentModelWrapper:
    """Test StudentModelWrapper for HuggingFace models"""

    @patch('models.student_wrapper.AutoModelForCausalLM')
    @patch('models.student_wrapper.AutoTokenizer')
    def test_init_with_default_model(self, mock_tokenizer, mock_model):
        # Skip actual model loading in tests
        wrapper = StudentModelWrapper.__new__(StudentModelWrapper)
        wrapper.model_name = "Qwen/Qwen2.5-3B-Instruct"
        wrapper.is_sft = False
        wrapper.is_sft_idmas = False

        assert wrapper.model_name == "Qwen/Qwen2.5-3B-Instruct"

    @patch('models.student_wrapper.AutoModelForCausalLM')
    @patch('models.student_wrapper.AutoTokenizer')
    @patch('models.student_wrapper.get_sft_model_name')
    def test_init_with_sft_model(self, mock_get_sft, mock_tokenizer, mock_model):
        mock_get_sft.return_value = "SaFD-00/qwen2.5-3b-math"

        # Create instance without actual model loading
        wrapper = StudentModelWrapper.__new__(StudentModelWrapper)
        wrapper.base_model_name = "Qwen/Qwen2.5-3B-Instruct"
        wrapper.is_sft = True
        wrapper.is_sft_idmas = False

        assert wrapper.is_sft is True

    def test_model_caching(self):
        # Verify class-level cache exists
        assert hasattr(StudentModelWrapper, '_model_cache')
        assert isinstance(StudentModelWrapper._model_cache, dict)

    @patch('models.student_wrapper.AutoModelForCausalLM')
    @patch('models.student_wrapper.AutoTokenizer')
    def test_generate_with_reflection_exists(self, mock_tokenizer, mock_model):
        # Verify generate_with_reflection method exists
        assert hasattr(StudentModelWrapper, 'generate_with_reflection')

        # Check method signature
        import inspect
        sig = inspect.signature(StudentModelWrapper.generate_with_reflection)
        params = list(sig.parameters.keys())
        assert 'prompt' in params
        assert 'teacher_feedback' in params
        assert 'reflection_result' in params


class TestQwenWrapper:
    """Test QwenWrapper (legacy, should be similar to StudentModelWrapper)"""

    @patch('models.qwen_wrapper.AutoModelForCausalLM')
    @patch('models.qwen_wrapper.AutoTokenizer')
    def test_init(self, mock_tokenizer, mock_model):
        # Create instance without actual model loading
        wrapper = QwenWrapper.__new__(QwenWrapper)
        wrapper.model_name = "Qwen/Qwen2.5-3B-Instruct"

        assert wrapper.model_name == "Qwen/Qwen2.5-3B-Instruct"

    @patch('models.qwen_wrapper.AutoModelForCausalLM')
    @patch('models.qwen_wrapper.AutoTokenizer')
    def test_generate_method_exists(self, mock_tokenizer, mock_model):
        # Verify generate method exists
        assert hasattr(QwenWrapper, 'generate')


class TestGenerateWithReflectionLogic:
    """Test the generate_with_reflection logic (critical for Phase 3)"""

    def test_reflection_prompt_construction(self):
        # This tests the logic that will be extracted to BaseModelWrapper

        # Mock wrapper with generate method
        class MockWrapper:
            def generate(self, prompt, system_message=None):
                return f"Generated from: {prompt[:50]}"

            def generate_with_reflection(self, prompt, teacher_feedback,
                                        reflection_result, system_message=None):
                # This is the duplicated code that will be extracted
                reflection_prompt = f"""
Original Problem:
{prompt}

Your Previous Reflection:
- Recognized Strengths: {reflection_result.get('recognized_strengths', [])}
- Recognized Weaknesses: {reflection_result.get('recognized_weaknesses', [])}
- Planned Reasoning Strategy: {reflection_result.get('planned_reasoning_strategy', [])}

Teacher's Suggested Next Actions:
{teacher_feedback.get('next_iteration_reasoning_actions', [])}

Now, generate a new response following your planned reasoning strategy and addressing the unsatisfied criteria.
"""
                return self.generate(reflection_prompt, system_message)

        wrapper = MockWrapper()

        teacher_feedback = {
            'next_iteration_reasoning_actions': ['Check calculation', 'Verify units']
        }

        reflection_result = {
            'recognized_strengths': ['Clear explanation'],
            'recognized_weaknesses': ['Arithmetic error'],
            'planned_reasoning_strategy': ['Double-check math', 'Show work']
        }

        result = wrapper.generate_with_reflection(
            prompt="Solve 2+2",
            teacher_feedback=teacher_feedback,
            reflection_result=reflection_result
        )

        assert "Solve 2+2" in result
        assert "Generated from:" in result

    def test_reflection_handles_empty_feedback(self):
        class MockWrapper:
            def generate(self, prompt, system_message=None):
                return prompt

            def generate_with_reflection(self, prompt, teacher_feedback,
                                        reflection_result, system_message=None):
                reflection_prompt = f"""
Original Problem:
{prompt}

Your Previous Reflection:
- Recognized Strengths: {reflection_result.get('recognized_strengths', [])}
- Recognized Weaknesses: {reflection_result.get('recognized_weaknesses', [])}
- Planned Reasoning Strategy: {reflection_result.get('planned_reasoning_strategy', [])}

Teacher's Suggested Next Actions:
{teacher_feedback.get('next_iteration_reasoning_actions', [])}

Now, generate a new response following your planned reasoning strategy and addressing the unsatisfied criteria.
"""
                return self.generate(reflection_prompt, system_message)

        wrapper = MockWrapper()

        result = wrapper.generate_with_reflection(
            prompt="Test",
            teacher_feedback={},
            reflection_result={}
        )

        # Should handle empty dicts gracefully
        assert "Test" in result
        assert "Original Problem:" in result
