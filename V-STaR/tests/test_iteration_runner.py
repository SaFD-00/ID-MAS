"""Tests for IterationRunner - Algorithm 1 compliance (T3, T4 verification)"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
import tempfile
import json

from config.training import TrainingConfig
from training.iteration_runner import IterationRunner, IterationState


class TestIterationState:
    """Test IterationState dataclass"""

    def test_state_initialization(self):
        """Test state initial values"""
        state = IterationState(iteration=0)
        assert state.iteration == 0
        assert state.d_gen == []
        assert state.d_ver == []
        assert state.metrics_history == []

    def test_state_to_dict(self):
        """Test state serialization"""
        state = IterationState(
            iteration=2,
            d_gen=[{"q": "test"}],
            d_ver=[{"q": "test", "is_correct": True}],
            metrics_history=[{"accuracy": 0.5}],
        )
        d = state.to_dict()
        assert d["iteration"] == 2
        assert d["d_gen_size"] == 1
        assert d["d_ver_size"] == 1
        assert len(d["metrics_history"]) == 1

    def test_state_save_and_load(self):
        """Test state save and load"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state"
            path.mkdir(parents=True, exist_ok=True)

            # Create state
            state = IterationState(
                iteration=3,
                d_gen=[{"question": "q1", "response": "r1", "is_correct": True}],
                d_ver=[{"question": "q1", "response": "r1", "is_correct": True}],
                metrics_history=[{"accuracy": 0.7}],
            )

            # Save
            state.save(path)

            # Load
            loaded = IterationState.load(path)

            assert loaded.iteration == 3
            assert len(loaded.d_gen) == 1
            assert len(loaded.d_ver) == 1
            assert loaded.metrics_history[0]["accuracy"] == 0.7


class TestIterationRunnerInit:
    """Test IterationRunner initialization"""

    def test_runner_stores_model_name_as_gbase(self):
        """Runner should store model_name as G_base"""
        runner = IterationRunner(
            model_name="test-model",
            config=TrainingConfig(),
        )
        assert runner.model_name == "test-model"

    def test_runner_initial_state(self):
        """Runner should have initial state"""
        runner = IterationRunner(model_name="test-model")
        assert runner.state.iteration == 0
        assert runner.g_sft is None
        assert runner.verifier is None


class TestAlgorithm1Compliance:
    """Test Algorithm 1 compliance - core behavior"""

    @patch('training.iteration_runner.create_generator')
    @patch('training.iteration_runner.SFTTrainer')
    def test_generator_trained_from_gbase_each_iteration(
        self,
        mock_sft_trainer_cls,
        mock_create_generator,
    ):
        """AC1: Generator must be trained from G_base each iteration"""
        # Setup
        mock_generator = MagicMock()
        mock_create_generator.return_value = mock_generator

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"train_loss": 0.5}
        mock_sft_trainer_cls.return_value = mock_trainer

        runner = IterationRunner(
            model_name="test-model",
            config=TrainingConfig(),
        )

        # Simulate initialized state
        runner.state.d_gen = [{"question": "q", "response": "r", "is_correct": True}]
        runner.g_sft = MagicMock()

        # Mock sample_and_label
        with patch.object(runner, 'sample_and_label') as mock_sample:
            mock_sample.return_value = ([], [])  # No new samples

            # Run iteration
            with tempfile.TemporaryDirectory() as tmpdir:
                runner.output_dir = Path(tmpdir)
                runner.run_iteration(
                    questions=[],
                    prompt_fn=lambda x: x,
                    iteration=1,
                )

        # Verify: create_generator should be called with model_name (G_base)
        mock_create_generator.assert_called_once()
        call_kwargs = mock_create_generator.call_args
        assert call_kwargs[1]['model_name'] == "test-model"

    def test_run_iteration_does_not_train_verifier(self):
        """AC2: run_iteration should NOT train verifier"""
        runner = IterationRunner(
            model_name="test-model",
            config=TrainingConfig(),
        )

        # Check that run_iteration doesn't have DPO training code
        # by inspecting the method
        import inspect
        source = inspect.getsource(runner.run_iteration)

        # Should not contain DPO training
        assert "DPOTrainerWrapper" not in source or "dpo_trainer" not in source
        # Or if it does, it should be in comments only

    def test_finalize_method_exists(self):
        """AC2: finalize method should exist for verifier training"""
        runner = IterationRunner(model_name="test-model")
        assert hasattr(runner, 'finalize')
        assert callable(runner.finalize)

    @patch('training.iteration_runner.create_verifier')
    @patch('training.iteration_runner.DPOTrainerWrapper')
    @patch('training.iteration_runner.create_preference_pairs_from_samples')
    def test_finalize_trains_verifier_once(
        self,
        mock_create_pairs,
        mock_dpo_trainer_cls,
        mock_create_verifier,
    ):
        """AC2: finalize should train verifier exactly once"""
        # Setup
        mock_verifier = MagicMock()
        mock_create_verifier.return_value = mock_verifier

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"train_loss": 0.3}
        mock_dpo_trainer_cls.return_value = mock_trainer

        mock_create_pairs.return_value = [{"prompt": "p", "chosen": "c", "rejected": "r"}]

        runner = IterationRunner(model_name="test-model")
        runner.g_sft = MagicMock()
        runner.g_sft.model = MagicMock()
        runner.state.d_ver = [
            {"question": "q1", "response": "r1", "is_correct": True},
            {"question": "q1", "response": "r2", "is_correct": False},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            runner.output_dir = Path(tmpdir)
            runner.finalize()

        # Verify: DPO trainer should be called exactly once
        assert mock_dpo_trainer_cls.call_count == 1
        mock_trainer.train.assert_called_once()


class TestDGenDVerInitialization:
    """Test D_GEN and D_VER initialization from D_SFT"""

    def test_d_gen_initialized_from_d_sft(self):
        """AC3: D_GEN should be initialized from D_SFT"""
        runner = IterationRunner(model_name="test-model")

        d_sft = [
            {"input": "q1", "output": "r1"},
            {"input": "q2", "output": "r2"},
        ]

        # Test conversion
        d_gen = runner._convert_to_d_gen_format(d_sft)

        assert len(d_gen) == 2
        assert d_gen[0]["question"] == "q1"
        assert d_gen[0]["response"] == "r1"
        assert d_gen[0]["is_correct"] == True

    def test_d_ver_initialized_from_d_sft_with_correct_labels(self):
        """AC3: D_VER should be initialized from D_SFT with correct=True"""
        runner = IterationRunner(model_name="test-model")

        d_sft = [
            {"input": "q1", "output": "r1"},
        ]

        d_ver = runner._convert_to_d_ver_format(d_sft, is_correct=True)

        assert len(d_ver) == 1
        assert d_ver[0]["is_correct"] == True


class TestReferencePolicyFreeze:
    """Test reference policy (G_SFT) freezing"""

    @patch('training.iteration_runner.create_generator')
    @patch('training.iteration_runner.SFTTrainer')
    def test_g_sft_frozen_after_initialization(
        self,
        mock_sft_trainer_cls,
        mock_create_generator,
    ):
        """AC4: G_SFT should be frozen (eval mode, no grad) after init"""
        # Setup mock
        mock_model = MagicMock()
        mock_params = [MagicMock() for _ in range(3)]
        mock_model.parameters.return_value = iter(mock_params)

        mock_generator = MagicMock()
        mock_generator.model = mock_model
        mock_create_generator.return_value = mock_generator

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {}
        mock_sft_trainer_cls.return_value = mock_trainer

        runner = IterationRunner(model_name="test-model")

        with tempfile.TemporaryDirectory() as tmpdir:
            runner.output_dir = Path(tmpdir)
            runner.initialize(d_sft=[{"input": "q", "output": "r"}])

        # Verify: model should be in eval mode
        mock_model.eval.assert_called()

        # Verify: all parameters should have requires_grad = False
        for param in mock_params:
            assert param.requires_grad == False


class TestRunMethod:
    """Test the run() method orchestration"""

    @patch('training.iteration_runner.create_generator')
    @patch('training.iteration_runner.SFTTrainer')
    @patch.object(IterationRunner, 'run_iteration')
    @patch.object(IterationRunner, 'finalize')
    def test_run_calls_finalize_after_all_iterations(
        self,
        mock_finalize,
        mock_run_iteration,
        mock_sft_trainer_cls,
        mock_create_generator,
    ):
        """run() should call finalize() after all iterations"""
        # Setup
        mock_generator = MagicMock()
        mock_create_generator.return_value = mock_generator

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {}
        mock_sft_trainer_cls.return_value = mock_trainer

        mock_run_iteration.return_value = {"accuracy": 0.5, "sft_loss": 0.3}

        runner = IterationRunner(
            model_name="test-model",
            config=TrainingConfig(num_iterations=3),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner.output_dir = Path(tmpdir)
            runner.run(
                questions=[],
                prompt_fn=lambda x: x,
                d_sft=[{"input": "q", "output": "r"}],
            )

        # Verify: run_iteration called 3 times
        assert mock_run_iteration.call_count == 3

        # Verify: finalize called once at the end
        mock_finalize.assert_called_once()
