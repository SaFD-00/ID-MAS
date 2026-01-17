"""V-STaR Iteration Runner - Algorithm 1 정확 구현"""

from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
from dataclasses import dataclass, field
import json
import copy

import torch
from tqdm import tqdm

from config.training import TrainingConfig
from data.loader import DataLoader, QuestionData, SFTSample, SFTDataLoader
from data.sampler import SolutionSampler
from data.preference_dataset import (
    create_preference_pairs_from_samples,
    PreferenceDataset,
    DGenDataset,
)
from models.generator import VSTaRGenerator, create_generator
from models.verifier import VSTaRVerifier, create_verifier
from .sft_trainer import SFTTrainer
from .dpo_trainer import DPOTrainerWrapper


@dataclass
class IterationState:
    """State of V-STaR iteration"""
    iteration: int
    d_gen: List[Dict[str, Any]] = field(default_factory=list)  # D_GEN: correct solutions only
    d_ver: List[Dict[str, Any]] = field(default_factory=list)  # D_VER: all labeled solutions
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "d_gen_size": len(self.d_gen),
            "d_ver_size": len(self.d_ver),
            "metrics_history": self.metrics_history,
        }

    def save(self, path: Path) -> None:
        """Save state to file"""
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(path / "state.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        # Save datasets
        with open(path / "d_gen.json", "w") as f:
            json.dump(self.d_gen, f, indent=2, ensure_ascii=False)

        with open(path / "d_ver.json", "w") as f:
            json.dump(self.d_ver, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "IterationState":
        """Load state from file"""
        with open(path / "state.json", "r") as f:
            metadata = json.load(f)

        with open(path / "d_gen.json", "r") as f:
            d_gen = json.load(f)

        with open(path / "d_ver.json", "r") as f:
            d_ver = json.load(f)

        return cls(
            iteration=metadata["iteration"],
            d_gen=d_gen,
            d_ver=d_ver,
            metrics_history=metadata.get("metrics_history", []),
        )


class IterationRunner:
    """
    V-STaR Iteration Runner - Algorithm 1 정확 구현

    논문 Algorithm 1:
    ────────────────────────────────────────────────
    Input: D_SFT, D_query, G_base, k, T

    D_GEN ← D_SFT                    # D_SFT로 초기화
    G_SFT ← SFT(G_base, D_SFT)       # Reference policy (고정)
    D_VER ← D_SFT (with correct=True)

    for iter = 1 to T do
        G ← SFT(G_base, D_GEN)       # ⭐ 핵심: 매번 G_base에서!
        S ← sample(G, D_query, k)
        D' ← label_correctness(S)
        D_GEN ← D_GEN ∪ D'[z=1]      # correct만
        D_VER ← D_VER ∪ D'           # 전체
    end for

    D_pref ← preference_pairs(D_VER)
    V ← DPO(G_SFT, D_pref)           # ⭐ 핵심: 마지막에 한 번!
    ────────────────────────────────────────────────
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[TrainingConfig] = None,
        output_dir: str = "./checkpoints",
        device: str = "cuda",
    ):
        """
        Initialize Iteration Runner

        Args:
            model_name: HuggingFace model name (G_base)
            config: Training configuration
            output_dir: Base output directory
            device: Device to use
        """
        self.model_name = model_name  # G_base
        self.config = config or TrainingConfig()
        self.output_dir = Path(output_dir)
        self.device = device

        # Models
        self.g_sft: Optional[VSTaRGenerator] = None  # Reference policy (고정)
        self.verifier: Optional[VSTaRVerifier] = None

        # State
        self.state = IterationState(iteration=0)

    def initialize(
        self,
        d_sft: Optional[List[Dict]] = None,
        d_sft_path: Optional[str] = None,
    ) -> None:
        """
        Algorithm 1 초기화

        Args:
            d_sft: D_SFT 데이터 (직접 전달)
            d_sft_path: D_SFT 파일 경로 (자동 로드)
        """
        print(f"Initializing V-STaR with G_base: {self.model_name}")

        # D_SFT 로드/생성
        if d_sft is None and d_sft_path:
            print(f"Loading D_SFT from: {d_sft_path}")
            with open(d_sft_path, "r") as f:
                d_sft = json.load(f)

        if d_sft:
            print(f"D_SFT size: {len(d_sft)}")

            # D_GEN ← D_SFT (논문: DGEN ← DSFT)
            self.state.d_gen = self._convert_to_d_gen_format(d_sft)

            # D_VER ← D_SFT with correct labels
            self.state.d_ver = self._convert_to_d_ver_format(d_sft, is_correct=True)

            print(f"  - D_GEN initialized with {len(self.state.d_gen)} samples")
            print(f"  - D_VER initialized with {len(self.state.d_ver)} samples")
        else:
            print("Warning: No D_SFT provided. D_GEN and D_VER start empty.")
            self.state.d_gen = []
            self.state.d_ver = []

        # G_SFT ← SFT(G_base, D_SFT) - Reference policy
        print("\n[Creating G_SFT (Reference Policy)]")
        self.g_sft = create_generator(
            model_name=self.model_name,
            config=self.config,
            use_lora=True,
            device=self.device,
        )

        if d_sft and len(d_sft) > 0:
            print(f"Training G_SFT on D_SFT ({len(d_sft)} samples)...")
            g_sft_dir = self.output_dir / "g_sft"
            g_sft_dir.mkdir(parents=True, exist_ok=True)

            sft_trainer = SFTTrainer(
                model=self.g_sft.model,
                tokenizer=self.g_sft.tokenizer,
                config=self.config,
                output_dir=str(g_sft_dir),
            )
            sft_trainer.train(train_data=d_sft)

            # Save G_SFT
            self.g_sft.save(g_sft_dir / "final")
            print(f"G_SFT saved to {g_sft_dir / 'final'}")

        # G_SFT를 Reference로 고정 (논문: GSFT는 전체 학습 동안 고정)
        print("Freezing G_SFT as reference policy...")
        self.g_sft.model.eval()
        for param in self.g_sft.model.parameters():
            param.requires_grad = False

        print("\nV-STaR initialization complete")

    def _convert_to_d_gen_format(self, data: List[Dict]) -> List[Dict]:
        """D_SFT를 D_GEN 형식으로 변환"""
        d_gen = []
        for item in data:
            # 다양한 형식 지원
            d_gen.append({
                "question_id": item.get("question_id", item.get("metadata", {}).get("id", "")),
                "question": item.get("question", item.get("input", "")),
                "response": item.get("response", item.get("output", "")),
                "is_correct": True,
            })
        return d_gen

    def _convert_to_d_ver_format(
        self,
        data: List[Dict],
        is_correct: bool = True,
    ) -> List[Dict]:
        """데이터를 D_VER 형식으로 변환"""
        d_ver = []
        for item in data:
            d_ver.append({
                "question_id": item.get("question_id", item.get("metadata", {}).get("id", "")),
                "question": item.get("question", item.get("input", "")),
                "response": item.get("response", item.get("output", "")),
                "is_correct": is_correct,
            })
        return d_ver

    def sample_and_label(
        self,
        generator: VSTaRGenerator,
        questions: List[QuestionData],
        prompt_fn: Callable[[QuestionData], str],
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Sample solutions and label them

        Args:
            generator: Current generator (이번 iteration의 G)
            questions: List of questions
            prompt_fn: Function to create prompts

        Returns:
            Tuple of (correct_solutions, all_solutions)
        """
        sampler = SolutionSampler(
            generator=generator,
            k=self.config.k,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
        )

        correct_data, all_data = sampler.sample_and_label(
            questions=questions,
            prompt_fn=prompt_fn,
            show_progress=True,
        )

        return correct_data, all_data

    def run_iteration(
        self,
        questions: List[QuestionData],
        prompt_fn: Callable[[QuestionData], str],
        iteration: int,
    ) -> Dict[str, Any]:
        """
        Algorithm 1의 단일 iteration 실행

        논문:
            G ← SFT(G_base, D_GEN)    # 매번 G_base에서!
            S ← sample(G, D_query, k)
            D' ← label_correctness(S)
            D_GEN ← D_GEN ∪ D'[z=1]
            D_VER ← D_VER ∪ D'

        NOTE: Verifier 학습은 여기서 하지 않음! finalize()에서 수행
        """
        print(f"\n{'='*60}")
        print(f"V-STaR Iteration {iteration}")
        print(f"{'='*60}")

        iteration_dir = self.output_dir / f"iteration_{iteration}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        metrics = {"iteration": iteration}

        # ⭐ Step 1: G ← SFT(G_base, D_GEN) - 매번 G_base에서 시작!
        print(f"\n[Step 1] Training Generator G from G_base on D_GEN ({len(self.state.d_gen)} samples)...")
        gen_output_dir = iteration_dir / "generator"

        # 새로운 Generator 생성 (G_base에서!)
        generator = create_generator(
            model_name=self.model_name,  # G_base
            config=self.config,
            use_lora=True,
            device=self.device,
        )

        # D_GEN으로 SFT
        if len(self.state.d_gen) > 0:
            sft_trainer = SFTTrainer(
                model=generator.model,
                tokenizer=generator.tokenizer,
                config=self.config,
                output_dir=str(gen_output_dir),
            )

            sft_metrics = sft_trainer.train(train_data=self.state.d_gen)
            metrics["sft_loss"] = sft_metrics.get("train_loss", 0)
            print(f"  - SFT Loss: {metrics['sft_loss']:.4f}")
        else:
            metrics["sft_loss"] = 0
            print("  - Skipping SFT (D_GEN is empty)")

        # Step 2: Sample k solutions per question
        print(f"\n[Step 2] Sampling {self.config.k} solutions per question...")
        correct_data, all_data = self.sample_and_label(generator, questions, prompt_fn)

        metrics["num_new_correct"] = len(correct_data)
        metrics["num_new_total"] = len(all_data)
        metrics["accuracy"] = len(correct_data) / len(all_data) if all_data else 0

        print(f"  - New correct solutions: {len(correct_data)}")
        print(f"  - New total solutions: {len(all_data)}")
        print(f"  - Accuracy: {metrics['accuracy']:.2%}")

        # Step 3: D_GEN ← D_GEN ∪ D'[z=1] (correct only)
        self.state.d_gen.extend(correct_data)
        print(f"\n[Step 3] Updated D_GEN size: {len(self.state.d_gen)}")

        # Step 4: D_VER ← D_VER ∪ D' (all labeled)
        self.state.d_ver.extend(all_data)
        print(f"[Step 4] Updated D_VER size: {len(self.state.d_ver)}")

        # Save iteration state
        self.state.iteration = iteration
        self.state.metrics_history.append(metrics)
        self.state.save(iteration_dir / "state")

        # Save generator checkpoint
        generator.save(gen_output_dir / "final")

        # Generator는 매 iteration 후 release (메모리 절약)
        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n[Iteration {iteration} Complete]")
        print(f"  - D_GEN size: {len(self.state.d_gen)}")
        print(f"  - D_VER size: {len(self.state.d_ver)}")

        return metrics

    def finalize(self) -> VSTaRVerifier:
        """
        Algorithm 1의 마지막 단계: Verifier 학습

        논문:
            D_pref ← preference_pairs(D_VER)
            V ← DPO(G_SFT, D_pref)    # 마지막에 한 번만!

        Returns:
            Trained Verifier
        """
        print(f"\n{'='*60}")
        print("V-STaR Finalization: Training Verifier")
        print(f"{'='*60}")

        ver_output_dir = self.output_dir / "verifier_final"
        ver_output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Create preference pairs from D_VER
        print(f"\n[Step 1] Creating preference pairs from D_VER ({len(self.state.d_ver)} samples)...")
        preference_pairs = create_preference_pairs_from_samples(
            all_solutions=self.state.d_ver,
            max_pairs_per_question=self.config.max_pairs_per_question,
            shuffle=True,
        )
        print(f"  - Preference pairs: {len(preference_pairs)}")

        if len(preference_pairs) == 0:
            print("Warning: No preference pairs created. Verifier training skipped.")
            return None

        # Step 2: V ← DPO(G_SFT, D_pref)
        # G_SFT를 base로 Verifier 생성
        print(f"\n[Step 2] Training Verifier with DPO (from G_SFT)...")

        # Verifier 생성 (G_SFT 기반)
        self.verifier = create_verifier(
            model_name=self.model_name,
            ref_model=self.g_sft.model,  # G_SFT as reference
            config=self.config,
            use_lora=True,
            device=self.device,
        )

        # DPO 학습
        dpo_trainer = DPOTrainerWrapper(
            model=self.verifier.model,
            ref_model=self.g_sft.model,  # G_SFT is reference
            tokenizer=self.verifier.tokenizer,
            config=self.config,
            output_dir=str(ver_output_dir),
        )

        dpo_metrics = dpo_trainer.train(train_data=preference_pairs)
        print(f"  - DPO Loss: {dpo_metrics.get('train_loss', 0):.4f}")

        # Save verifier
        self.verifier.save(ver_output_dir / "final")
        print(f"\nVerifier saved to {ver_output_dir / 'final'}")

        return self.verifier

    def run(
        self,
        questions: List[QuestionData],
        prompt_fn: Callable[[QuestionData], str],
        num_iterations: Optional[int] = None,
        d_sft: Optional[List[Dict]] = None,
        d_sft_path: Optional[str] = None,
        resume_from: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        V-STaR 전체 학습 실행 (Algorithm 1)

        Args:
            questions: Training questions (D_query)
            prompt_fn: Prompt formatting function
            num_iterations: Number of iterations (T)
            d_sft: Initial SFT data (D_SFT)
            d_sft_path: Path to D_SFT file
            resume_from: Resume from iteration number

        Returns:
            List of iteration metrics
        """
        num_iterations = num_iterations or self.config.num_iterations

        # Resume if specified
        if resume_from is not None:
            print(f"Resuming from iteration {resume_from}...")
            state_path = self.output_dir / f"iteration_{resume_from}" / "state"
            self.state = IterationState.load(state_path)

            # Load G_SFT
            g_sft_path = self.output_dir / "g_sft" / "final"
            if g_sft_path.exists():
                self.g_sft = VSTaRGenerator.load(
                    g_sft_path,
                    base_model_name=self.model_name,
                    config=self.config,
                    device=self.device,
                )
                self.g_sft.model.eval()
                for param in self.g_sft.model.parameters():
                    param.requires_grad = False

            start_iteration = resume_from + 1
        else:
            # Initialize
            self.initialize(d_sft=d_sft, d_sft_path=d_sft_path)
            start_iteration = 1

        all_metrics = []

        # Main iteration loop
        for iteration in range(start_iteration, num_iterations + 1):
            metrics = self.run_iteration(
                questions=questions,
                prompt_fn=prompt_fn,
                iteration=iteration,
            )
            all_metrics.append(metrics)

        # ⭐ Finalize: Train verifier once at the end!
        verifier = self.finalize()

        print(f"\n{'='*60}")
        print("V-STaR Training Complete!")
        print(f"{'='*60}")
        print(f"Total iterations: {num_iterations}")
        print(f"Final D_GEN size: {len(self.state.d_gen)}")
        print(f"Final D_VER size: {len(self.state.d_ver)}")

        return all_metrics

    def save_final(self, path: Optional[Path] = None) -> None:
        """Save final models"""
        path = path or self.output_dir / "final"
        path.mkdir(parents=True, exist_ok=True)

        if self.g_sft:
            self.g_sft.save(path / "g_sft")
        if self.verifier:
            self.verifier.save(path / "verifier")

        # Save final state
        self.state.save(path / "state")

        print(f"Final models saved to {path}")

    @classmethod
    def load(
        cls,
        path: Path,
        model_name: str,
        config: Optional[TrainingConfig] = None,
        device: str = "cuda",
    ) -> "IterationRunner":
        """Load runner from checkpoint"""
        runner = cls(
            model_name=model_name,
            config=config,
            output_dir=path.parent,
            device=device,
        )

        # Load state
        runner.state = IterationState.load(path / "state")

        # Load G_SFT
        runner.g_sft = VSTaRGenerator.load(
            path / "g_sft",
            base_model_name=model_name,
            config=config,
            device=device,
        )

        # Load verifier
        verifier_path = path / "verifier"
        if verifier_path.exists():
            runner.verifier = VSTaRVerifier.load(
                verifier_path,
                base_model_name=model_name,
                config=config,
                device=device,
            )

        return runner
