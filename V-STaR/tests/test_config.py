"""Tests for TrainingConfig (T1 verification)"""

import pytest
from config.training import TrainingConfig, LoRAConfig, DPOConfig, SFTConfig


class TestTrainingConfig:
    """Test TrainingConfig with Algorithm 1 parameters"""

    def test_config_has_k_property(self):
        """AC5: Config has 'k' property"""
        config = TrainingConfig()
        assert hasattr(config, 'k')
        assert config.k == 16  # Default value

    def test_k_is_alias_for_samples_per_query(self):
        """k should be alias for samples_per_query"""
        config = TrainingConfig(samples_per_query=32)
        assert config.k == 32
        assert config.samples_per_query == 32

    def test_k_setter(self):
        """k setter should update samples_per_query"""
        config = TrainingConfig()
        config.k = 64
        assert config.k == 64
        assert config.samples_per_query == 64

    def test_config_has_max_pairs_per_question(self):
        """AC5: Config has max_pairs_per_question"""
        config = TrainingConfig()
        assert hasattr(config, 'max_pairs_per_question')
        assert config.max_pairs_per_question is None  # Default

    def test_max_pairs_per_question_can_be_set(self):
        """max_pairs_per_question can be set to a value"""
        config = TrainingConfig(max_pairs_per_question=10)
        assert config.max_pairs_per_question == 10

    def test_default_values(self):
        """Test default configuration values"""
        config = TrainingConfig()

        # V-STaR parameters
        assert config.num_iterations == 3
        assert config.samples_per_query == 16
        assert config.best_of_k == 64

        # Generation parameters
        assert config.temperature == 0.7

    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        config = TrainingConfig(
            num_iterations=3,
            samples_per_query=16,
            best_of_k=64,
        )
        assert config.num_iterations == 3

        # Invalid: num_iterations < 1
        with pytest.raises(AssertionError):
            TrainingConfig(num_iterations=0)

        # Invalid: samples_per_query < 1
        with pytest.raises(AssertionError):
            TrainingConfig(samples_per_query=0)

        # Invalid: temperature out of range
        with pytest.raises(AssertionError):
            TrainingConfig(temperature=0)

    def test_sub_configs(self):
        """Test sub-configurations (LoRA, SFT, DPO)"""
        config = TrainingConfig()

        # LoRA config
        assert isinstance(config.lora, LoRAConfig)
        assert config.lora.r == 8

        # SFT config
        assert isinstance(config.sft, SFTConfig)
        assert config.sft.learning_rate == 2e-5

        # DPO config
        assert isinstance(config.dpo, DPOConfig)
        assert config.dpo.beta == 0.1
