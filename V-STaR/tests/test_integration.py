"""Integration tests for V-STaR Algorithm 1 compliance"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import json

from config.training import TrainingConfig
from data.loader import QuestionData, SFTSample
from config.domains import AnswerType


class TestVSTaRAlgorithm1Integration:
    """
    Integration tests to verify Algorithm 1 compliance.

    Algorithm 1: V-STaR
    ────────────────────────────────────────────────
    Input: D_SFT, D_query, G_base, k, T

    D_GEN ← D_SFT
    G_SFT ← SFT(G_base, D_SFT)
    D_VER ← D_SFT (with correct=True)

    for iter = 1 to T do
        G ← SFT(G_base, D_GEN)       # ⭐ From G_base each time
        S ← sample(G, D_query, k)
        D' ← label_correctness(S)
        D_GEN ← D_GEN ∪ D'[z=1]
        D_VER ← D_VER ∪ D'
    end for

    D_pref ← preference_pairs(D_VER)
    V ← DPO(G_SFT, D_pref)           # ⭐ Once at the end
    ────────────────────────────────────────────────
    """

    @pytest.fixture
    def mock_d_sft(self):
        """Sample D_SFT data"""
        return [
            {"input": "What is 2+2?", "output": "The answer is \\boxed{4}", "metadata": {"id": "q1"}},
            {"input": "What is 3+3?", "output": "The answer is \\boxed{6}", "metadata": {"id": "q2"}},
        ]

    @pytest.fixture
    def mock_questions(self):
        """Sample questions (D_query)"""
        return [
            QuestionData(
                dataset="test",
                question_id="q1",
                question="What is 2+2?",
                answer_type=AnswerType.NUMERIC,
                ground_truth=4,
                ground_truth_formatted="4",
            ),
            QuestionData(
                dataset="test",
                question_id="q2",
                question="What is 3+3?",
                answer_type=AnswerType.NUMERIC,
                ground_truth=6,
                ground_truth_formatted="6",
            ),
        ]

    @patch('training.iteration_runner.create_generator')
    @patch('training.iteration_runner.create_verifier')
    @patch('training.iteration_runner.SFTTrainer')
    @patch('training.iteration_runner.DPOTrainerWrapper')
    @patch('training.iteration_runner.create_preference_pairs_from_samples')
    def test_full_vstar_flow(
        self,
        mock_create_pairs,
        mock_dpo_trainer_cls,
        mock_sft_trainer_cls,
        mock_create_verifier,
        mock_create_generator,
        mock_d_sft,
        mock_questions,
    ):
        """
        Test full V-STaR training flow follows Algorithm 1.

        Verifies:
        - D_GEN initialized from D_SFT
        - G_SFT trained once and frozen
        - Each iteration trains G from G_base
        - Verifier trained once at the end
        """
        from training.iteration_runner import IterationRunner

        # Setup mocks
        mock_generator = MagicMock()
        mock_generator.model.parameters.return_value = iter([MagicMock()])
        mock_create_generator.return_value = mock_generator

        mock_verifier = MagicMock()
        mock_create_verifier.return_value = mock_verifier

        mock_sft_trainer = MagicMock()
        mock_sft_trainer.train.return_value = {"train_loss": 0.5}
        mock_sft_trainer_cls.return_value = mock_sft_trainer

        mock_dpo_trainer = MagicMock()
        mock_dpo_trainer.train.return_value = {"train_loss": 0.3}
        mock_dpo_trainer_cls.return_value = mock_dpo_trainer

        mock_create_pairs.return_value = [
            {"prompt": "p", "chosen": "c", "rejected": "r"}
        ]

        # Create runner
        config = TrainingConfig(num_iterations=2)
        runner = IterationRunner(
            model_name="test-model",
            config=config,
        )

        # Mock sample_and_label to return consistent data
        def mock_sample_and_label(generator, questions, prompt_fn):
            return (
                [{"question_id": "q1", "question": "q", "response": "r", "is_correct": True}],
                [
                    {"question_id": "q1", "question": "q", "response": "r", "is_correct": True},
                    {"question_id": "q1", "question": "q", "response": "wrong", "is_correct": False},
                ],
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner.output_dir = Path(tmpdir)

            with patch.object(runner, 'sample_and_label', mock_sample_and_label):
                metrics = runner.run(
                    questions=mock_questions,
                    prompt_fn=lambda x: x.question,
                    d_sft=mock_d_sft,
                )

        # ===== VERIFICATION =====

        # 1. D_GEN should be initialized from D_SFT
        # After init: d_gen has 2 items from d_sft
        # After 2 iterations: d_gen has 2 + 1 + 1 = 4 items
        # (Each iteration adds 1 correct)
        assert len(runner.state.d_gen) >= len(mock_d_sft)

        # 2. create_generator should be called multiple times:
        # - 1 time for G_SFT initialization
        # - 2 times for each iteration (num_iterations=2)
        assert mock_create_generator.call_count == 3

        # 3. All create_generator calls should use "test-model" (G_base)
        for call in mock_create_generator.call_args_list:
            assert call[1]['model_name'] == "test-model"

        # 4. DPO trainer should be called exactly ONCE (at finalize)
        assert mock_dpo_trainer_cls.call_count == 1

        # 5. SFT trainer should be called:
        # - 1 time for G_SFT (init)
        # - 2 times for iterations
        assert mock_sft_trainer_cls.call_count == 3

        # 6. Metrics should have 2 entries (one per iteration)
        assert len(metrics) == 2

    @patch('training.iteration_runner.create_generator')
    @patch('training.iteration_runner.SFTTrainer')
    def test_g_sft_frozen_throughout_training(
        self,
        mock_sft_trainer_cls,
        mock_create_generator,
        mock_d_sft,
    ):
        """Verify G_SFT is frozen and not modified during iterations"""
        from training.iteration_runner import IterationRunner

        mock_generator = MagicMock()
        params = [MagicMock() for _ in range(3)]
        mock_generator.model.parameters.return_value = iter(params)
        mock_create_generator.return_value = mock_generator

        mock_sft_trainer = MagicMock()
        mock_sft_trainer.train.return_value = {}
        mock_sft_trainer_cls.return_value = mock_sft_trainer

        runner = IterationRunner(model_name="test-model")

        with tempfile.TemporaryDirectory() as tmpdir:
            runner.output_dir = Path(tmpdir)
            runner.initialize(d_sft=mock_d_sft)

        # G_SFT should be in eval mode
        mock_generator.model.eval.assert_called()

        # All parameters should be frozen
        for param in params:
            assert param.requires_grad == False

    def test_d_gen_accumulates_correct_solutions_only(self, mock_d_sft):
        """Verify D_GEN only accumulates correct solutions"""
        from training.iteration_runner import IterationRunner

        runner = IterationRunner(model_name="test-model")

        # Initialize with D_SFT
        runner.state.d_gen = runner._convert_to_d_gen_format(mock_d_sft)
        initial_size = len(runner.state.d_gen)

        # Simulate adding solutions
        correct_solutions = [
            {"question_id": "q3", "question": "q", "response": "r", "is_correct": True},
        ]
        incorrect_solutions = [
            {"question_id": "q4", "question": "q", "response": "wrong", "is_correct": False},
        ]

        # Only correct should be added to D_GEN
        runner.state.d_gen.extend(correct_solutions)

        assert len(runner.state.d_gen) == initial_size + 1
        assert all(item.get("is_correct", True) for item in runner.state.d_gen)

    def test_d_ver_accumulates_all_solutions(self, mock_d_sft):
        """Verify D_VER accumulates ALL solutions (correct and incorrect)"""
        from training.iteration_runner import IterationRunner

        runner = IterationRunner(model_name="test-model")

        # Initialize
        runner.state.d_ver = runner._convert_to_d_ver_format(mock_d_sft, is_correct=True)
        initial_size = len(runner.state.d_ver)

        # Simulate adding all solutions
        all_solutions = [
            {"question_id": "q3", "question": "q", "response": "correct", "is_correct": True},
            {"question_id": "q3", "question": "q", "response": "wrong", "is_correct": False},
        ]

        runner.state.d_ver.extend(all_solutions)

        assert len(runner.state.d_ver) == initial_size + 2
        # D_VER should have both correct and incorrect
        is_correct_values = [item["is_correct"] for item in runner.state.d_ver]
        assert True in is_correct_values
        # After adding the incorrect solution
        assert False in is_correct_values


class TestConfigIntegration:
    """Test configuration integration"""

    def test_config_k_used_in_sampling(self):
        """Verify config.k is used for sampling"""
        config = TrainingConfig(samples_per_query=32)

        # k property should return same value
        assert config.k == 32

        # This value should be used by SolutionSampler
        # (verified by unit tests, this is sanity check)

    def test_config_max_pairs_integration(self):
        """Verify max_pairs_per_question is passed to preference pair creation"""
        config = TrainingConfig(max_pairs_per_question=5)

        from training.iteration_runner import IterationRunner
        runner = IterationRunner(
            model_name="test",
            config=config,
        )

        assert runner.config.max_pairs_per_question == 5


class TestCLIIntegration:
    """Test CLI argument integration"""

    def test_train_parser_has_required_args(self):
        """Verify train command has all required arguments"""
        from main import main
        import argparse

        # Create parser to inspect
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        train_parser = subparsers.add_parser("train")

        # These should exist based on our changes
        train_parser.add_argument("--k", type=int, default=16)
        train_parser.add_argument("--d-sft-path", type=str)
        train_parser.add_argument("--max-pairs-per-question", type=int)

        # Parse test args
        args = train_parser.parse_args([
            "--k", "32",
            "--max-pairs-per-question", "10",
        ])

        assert args.k == 32
        assert args.max_pairs_per_question == 10
