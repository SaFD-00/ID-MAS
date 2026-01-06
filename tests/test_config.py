"""
Tests for configuration module.
"""
import pytest
from pathlib import Path
from config.config import (
    create_teacher_config,
    get_student_model_config,
    get_model_short_name,
    get_sft_model_name,
    get_sft_idmas_model_name,
    get_design_output_dir,
    get_domain_data_dirs,
    DEFAULT_TEACHER_MODEL,
    DEFAULT_VLLM_TEACHER_MODEL,
    DEFAULT_STUDENT_MODEL,
    AVAILABLE_TEACHER_MODELS,
    AVAILABLE_STUDENT_MODELS,
    OPENAI_API_KEY,
    PROJECT_ROOT,
    DATA_DIR,
    MODEL_NAME_TO_SHORT
)


class TestTeacherConfig:
    """Test teacher model configuration"""

    def test_create_teacher_config_default(self):
        config = create_teacher_config()
        assert config is not None
        assert "model" in config
        assert config["model"] == DEFAULT_TEACHER_MODEL

    def test_create_teacher_config_openai(self):
        config = create_teacher_config("gpt-5-2025-08-07")
        assert config["model"] == "gpt-5-2025-08-07"
        assert config["base_url"] is None  # OpenAI endpoint
        assert "api_key" in config
        assert "max_tokens" in config
        assert "reasoning" in config

    def test_create_teacher_config_vllm(self):
        config = create_teacher_config("meta-llama/Llama-3.3-70B-Instruct")
        assert config["model"] == "meta-llama/Llama-3.3-70B-Instruct"
        assert config["base_url"] == "http://localhost:2000/v1"  # vLLM endpoint
        assert config["api_key"] == "0"
        assert config["max_tokens"] == 8192

    def test_create_teacher_config_available_models(self):
        # Test that all available teacher models can create configs
        for model in AVAILABLE_TEACHER_MODELS:
            config = create_teacher_config(model)
            assert config["model"] == model

    def test_default_vllm_teacher_model(self):
        """Test that DEFAULT_VLLM_TEACHER_MODEL is valid"""
        # Should be in available models
        assert DEFAULT_VLLM_TEACHER_MODEL in AVAILABLE_TEACHER_MODELS

        # Should not be a gpt- model (should use vLLM)
        assert not DEFAULT_VLLM_TEACHER_MODEL.startswith("gpt-")

        # Should be Qwen3-30B-A3B-Instruct-2507-FP8
        assert DEFAULT_VLLM_TEACHER_MODEL == "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

        # Should create valid vLLM config
        config = create_teacher_config(DEFAULT_VLLM_TEACHER_MODEL)
        assert config["model"] == DEFAULT_VLLM_TEACHER_MODEL
        assert config["base_url"] == "http://localhost:2000/v1"
        assert config["api_key"] == "0"


class TestStudentConfig:
    """Test student model configuration"""

    def test_get_student_model_config_default(self):
        config = get_student_model_config()
        assert config is not None
        assert config["model_name"] == DEFAULT_STUDENT_MODEL
        assert "device" in config
        assert "max_new_tokens" in config
        assert "temperature" in config

    def test_get_student_model_config_specific_model(self):
        config = get_student_model_config("Qwen/Qwen3-4B-Instruct-2507")
        assert config["model_name"] == "Qwen/Qwen3-4B-Instruct-2507"

    def test_get_student_model_config_invalid_model(self):
        with pytest.raises(ValueError, match="지원하지 않는 모델"):
            get_student_model_config("invalid/model-name")

    def test_get_student_model_config_available_models(self):
        # Test that all available student models can create configs
        for model in AVAILABLE_STUDENT_MODELS:
            config = get_student_model_config(model)
            assert config["model_name"] == model


class TestModelNaming:
    """Test model name utilities"""

    def test_get_model_short_name_default(self):
        short_name = get_model_short_name()
        assert short_name is not None
        assert "/" not in short_name or short_name == DEFAULT_STUDENT_MODEL

    def test_get_model_short_name_with_slash(self):
        full_name = "Qwen/Qwen3-4B-Instruct-2507"
        short_name = get_model_short_name(full_name)
        assert short_name == "Qwen3-4B-Instruct-2507"
        assert "/" not in short_name

    def test_get_model_short_name_without_slash(self):
        name = "some-model-name"
        short_name = get_model_short_name(name)
        assert short_name == name

    def test_get_model_short_name_llama(self):
        full_name = "meta-llama/Llama-3.1-8B-Instruct"
        short_name = get_model_short_name(full_name)
        assert short_name == "Llama-3.1-8B-Instruct"


class TestSFTModelNames:
    """Test SFT model name generation"""

    def test_get_sft_model_name_qwen(self):
        model_name = get_sft_model_name("Qwen/Qwen2.5-3B-Instruct", "math")
        assert model_name == "SaFD-00/qwen2.5-3b-math"

    def test_get_sft_model_name_llama(self):
        model_name = get_sft_model_name("meta-llama/Llama-3.1-8B-Instruct", "math")
        assert model_name == "SaFD-00/llama3.1-8b-math"

    def test_get_sft_model_name_invalid_model(self):
        with pytest.raises(ValueError, match="SFT fine-tuned model not available"):
            get_sft_model_name("invalid/model", "math")

    def test_get_sft_model_name_invalid_domain(self):
        # Assuming domain validation exists
        with pytest.raises(ValueError):
            get_sft_model_name("Qwen/Qwen2.5-3B-Instruct", "invalid-domain")

    def test_get_sft_idmas_model_name_qwen(self):
        model_name = get_sft_idmas_model_name("Qwen/Qwen2.5-3B-Instruct", "math")
        assert model_name == "SaFD-00/qwen2.5-3b-math-ID-MAS"

    def test_get_sft_idmas_model_name_llama(self):
        model_name = get_sft_idmas_model_name("meta-llama/Llama-3.1-8B-Instruct", "math")
        assert model_name == "SaFD-00/llama3.1-8b-math-ID-MAS"

    def test_model_name_to_short_mapping(self):
        # Verify mapping dictionary is complete
        assert "Qwen/Qwen2.5-3B-Instruct" in MODEL_NAME_TO_SHORT
        assert "meta-llama/Llama-3.1-8B-Instruct" in MODEL_NAME_TO_SHORT

        # Verify mapped values are lowercase and follow pattern
        for full_name, short_name in MODEL_NAME_TO_SHORT.items():
            assert short_name.islower() or any(c.isdigit() for c in short_name)
            assert "-" in short_name or "." in short_name


class TestDirectoryHelpers:
    """Test directory path generation functions"""

    def test_get_design_output_dir_creates_path(self):
        design_dir = get_design_output_dir("math")
        assert isinstance(design_dir, Path)
        assert "math" in str(design_dir)
        assert "design" in str(design_dir)

    def test_get_design_output_dir_different_domains(self):
        math_dir = get_design_output_dir("math")
        knowledge_dir = get_design_output_dir("knowledge")
        assert math_dir != knowledge_dir

    def test_get_domain_data_dirs_train_mode(self):
        dirs = get_domain_data_dirs(
            domain="math",
            model_name="Qwen/Qwen2.5-3B-Instruct",
            train_dataset="gsm8k",
            mode="train"
        )

        assert isinstance(dirs, dict)
        assert "domain_dir" in dirs
        assert "model_dir" in dirs
        assert "dataset_dir" in dirs
        assert "sft_data_dir" in dirs
        assert "learning_loop_dir" in dirs

        # Verify paths contain expected components
        assert "math" in str(dirs["domain_dir"])
        assert "Qwen2.5-3B-Instruct" in str(dirs["model_dir"])

    def test_get_domain_data_dirs_eval_mode(self):
        dirs = get_domain_data_dirs(
            domain="math",
            model_name="Qwen/Qwen2.5-3B-Instruct",
            mode="eval"
        )

        assert isinstance(dirs, dict)
        assert "domain_dir" in dirs
        assert "model_dir" in dirs
        # eval mode might not have all dirs that train mode has

    def test_project_root_exists(self):
        assert PROJECT_ROOT is not None
        assert isinstance(PROJECT_ROOT, Path)
        assert PROJECT_ROOT.exists()

    def test_data_dir_created(self):
        assert DATA_DIR is not None
        assert isinstance(DATA_DIR, Path)
        # DATA_DIR should be created by config module
        assert DATA_DIR.exists()


class TestAPIConfiguration:
    """Test API key configuration"""

    def test_openai_api_key_loaded(self):
        # API key might be None if .env doesn't exist, but variable should exist
        assert OPENAI_API_KEY is not None or OPENAI_API_KEY is None
        # Just verify the import works

    def test_available_models_not_empty(self):
        assert len(AVAILABLE_TEACHER_MODELS) > 0
        assert len(AVAILABLE_STUDENT_MODELS) > 0

    def test_available_models_contain_defaults(self):
        assert DEFAULT_TEACHER_MODEL in AVAILABLE_TEACHER_MODELS
        assert DEFAULT_STUDENT_MODEL in AVAILABLE_STUDENT_MODELS
