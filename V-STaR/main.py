#!/usr/bin/env python3
"""V-STaR Main CLI"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

from config.training import TrainingConfig
from config.models import AVAILABLE_MODELS, is_valid_model, get_model_family, get_model_size
from config.domains import DOMAIN_CONFIG


def train(args):
    """Run V-STaR training (Algorithm 1)"""
    from data.loader import DataLoader, SFTDataLoader
    from prompts.templates import create_prompt
    from training.iteration_runner import IterationRunner

    print(f"Starting V-STaR training with model: {args.model}")
    print(f"Domains: {args.domains}")
    print(f"Iterations: {args.iterations}")
    print(f"k (samples per query): {args.k}")

    # Validate model
    if not is_valid_model(args.model):
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {AVAILABLE_MODELS}")
        sys.exit(1)

    # Load config
    config = TrainingConfig(
        num_iterations=args.iterations,
        samples_per_query=args.k,
        temperature=args.temperature,
        max_pairs_per_question=args.max_pairs_per_question,
    )

    if args.learning_rate:
        config.sft.learning_rate = args.learning_rate
        config.dpo.learning_rate = args.learning_rate

    if args.batch_size:
        config.sft.per_device_train_batch_size = args.batch_size
        config.dpo.per_device_train_batch_size = args.batch_size

    # Load training data (D_query)
    domains = args.domains.split(",")
    all_questions = []
    d_sft_data = []  # D_SFT for initialization

    for domain in domains:
        domain = domain.strip()
        domain_config = DOMAIN_CONFIG.get(domain)

        if domain_config is None:
            print(f"Warning: Unknown domain '{domain}', skipping")
            continue

        for dataset_name in domain_config.datasets:
            data_path = Path(args.data_dir) / domain / "train" / f"{dataset_name}.json"

            if not data_path.exists():
                print(f"Warning: Data file not found: {data_path}")
                continue

            loader = DataLoader(data_dir=str(data_path.parent))
            questions = loader.load_dataset(dataset_name)
            all_questions.extend(questions)
            print(f"Loaded {len(questions)} questions from {dataset_name}")

            # Also collect D_SFT data (original training data with correct answers)
            import json
            with open(data_path, "r") as f:
                raw_data = json.load(f)
                d_sft_data.extend(raw_data)

    print(f"Total training questions (D_query): {len(all_questions)}")
    print(f"Total D_SFT samples: {len(d_sft_data)}")

    if not all_questions:
        print("Error: No training data loaded")
        sys.exit(1)

    # Create prompt function
    def prompt_fn(question):
        return create_prompt(question, include_cot=True)

    # Run training
    runner = IterationRunner(
        model_name=args.model,
        config=config,
        output_dir=args.output_dir,
        device=args.device,
    )

    # D_SFT: 직접 제공되었으면 사용, 아니면 자동 로드된 것 사용
    if args.d_sft_path:
        d_sft_path = args.d_sft_path
        d_sft_data = None  # Path로 로드
    else:
        d_sft_path = None
        # d_sft_data는 위에서 로드된 것 사용

    metrics = runner.run(
        questions=all_questions,
        prompt_fn=prompt_fn,
        num_iterations=args.iterations,
        d_sft=d_sft_data,
        d_sft_path=d_sft_path,
        resume_from=args.resume_from,
    )

    # Save final models
    runner.save_final()

    print("\nTraining complete!")
    print("\n" + "="*60)
    print("Iteration Summary:")
    print("="*60)
    for i, m in enumerate(metrics, 1):
        print(f"Iteration {i}: accuracy={m['accuracy']:.2%}, sft_loss={m['sft_loss']:.4f}")
    print("\nNote: Verifier was trained ONCE at the end (Algorithm 1)")


def evaluate(args):
    """Run evaluation"""
    from data.loader import DataLoader
    from prompts.templates import create_prompt
    from models.generator import VSTaRGenerator
    from models.verifier import VSTaRVerifier
    from evaluation.evaluator import (
        VSTaREvaluator,
        evaluate_generator_only,
        evaluate_with_self_consistency,
        save_evaluation_results,
    )

    print(f"Starting evaluation")
    print(f"Generator: {args.generator_path}")
    print(f"Verifier: {args.verifier_path}")

    # Load models
    generator = VSTaRGenerator.load(
        path=args.generator_path,
        base_model_name=args.base_model,
        device=args.device,
    )

    verifier = None
    if args.verifier_path:
        verifier = VSTaRVerifier.load(
            path=args.verifier_path,
            base_model_name=args.base_model,
            device=args.device,
        )

    # Load test data
    domains = args.domains.split(",")
    all_questions = []

    for domain in domains:
        domain = domain.strip()
        domain_config = DOMAIN_CONFIG.get(domain)

        if domain_config is None:
            continue

        for dataset_name in domain_config.datasets:
            data_path = Path(args.data_dir) / domain / "test" / f"{dataset_name}.json"

            if not data_path.exists():
                continue

            loader = DataLoader(
                data_dir=str(data_path.parent),
                domain=domain,
            )
            questions = loader.load_dataset(dataset_name)
            all_questions.extend(questions)
            print(f"Loaded {len(questions)} test questions from {dataset_name}")

    print(f"Total test questions: {len(all_questions)}")

    if not all_questions:
        print("Error: No test data loaded")
        sys.exit(1)

    def prompt_fn(question):
        return create_prompt(question, include_cot=True)

    # Run evaluation
    if verifier:
        # Full V-STaR evaluation with verifier
        evaluator = VSTaREvaluator(
            generator=generator,
            verifier=verifier,
        )
        results = evaluator.evaluate(
            questions=all_questions,
            prompt_fn=prompt_fn,
            num_samples=args.num_samples,
            k_values=[1, 4, 8, 16, 32, 64],
        )
    else:
        # Generator-only evaluation
        results = evaluate_generator_only(
            generator=generator,
            questions=all_questions,
            prompt_fn=prompt_fn,
            num_samples=args.num_samples,
        )

    # Self-consistency baseline
    if args.self_consistency:
        sc_results = evaluate_with_self_consistency(
            generator=generator,
            questions=all_questions,
            prompt_fn=prompt_fn,
            num_samples=args.num_samples,
        )
        results["self_consistency"] = sc_results

    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)

    metrics = results.get("metrics", results)
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Save results
    if args.output:
        save_evaluation_results(results, args.output)


def sample(args):
    """Generate solutions for questions"""
    from data.loader import DataLoader
    from prompts.templates import create_prompt
    from models.generator import VSTaRGenerator
    import json

    print(f"Generating solutions with model: {args.model_path}")

    # Load model
    generator = VSTaRGenerator.load(
        path=args.model_path,
        base_model_name=args.base_model,
        device=args.device,
    )

    # Load questions
    loader = DataLoader(
        data_dir=args.data_dir,
        domain=args.domain,
    )
    questions = loader.load_dataset(args.dataset)
    print(f"Loaded {len(questions)} questions")

    if args.max_questions:
        questions = questions[:args.max_questions]

    # Generate solutions
    results = []

    for question in questions:
        prompt = create_prompt(question, include_cot=True)

        solutions = generator.generate(
            prompt,
            k=args.num_samples,
            temperature=args.temperature,
        )

        if not isinstance(solutions, list):
            solutions = [solutions]

        results.append({
            "question_id": question.question_id,
            "question": question.question,
            "ground_truth": question.ground_truth_formatted,
            "solutions": solutions,
        })

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} results to {output_path}")


def list_models(args):
    """List available models"""
    print("Available models:")
    for model in AVAILABLE_MODELS:
        size = get_model_size(model)
        family = get_model_family(model)
        print(f"  - {model}")
        print(f"      Size: {size}, Family: {family}")


def main():
    parser = argparse.ArgumentParser(
        description="V-STaR: Training Verifiers for Self-Taught Reasoners"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run V-STaR training")
    train_parser.add_argument("--model", type=str, required=True, help="Model name")
    train_parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    train_parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Output directory")
    train_parser.add_argument("--domains", type=str, default="math,logical,commonsense", help="Domains to train on")
    train_parser.add_argument("--iterations", type=int, default=3, help="Number of iterations")
    train_parser.add_argument("--k", type=int, default=16, help="Solutions per question")
    train_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, help="Batch size")
    train_parser.add_argument("--device", type=str, default="cuda", help="Device")
    train_parser.add_argument("--d-sft-path", type=str, help="D_SFT data path (optional, auto-loaded from train data if not provided)")
    train_parser.add_argument("--max-pairs-per-question", type=int, default=None, help="Max preference pairs per question for DPO")
    train_parser.add_argument("--resume-from", type=int, help="Resume from iteration")
    train_parser.set_defaults(func=train)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate models")
    eval_parser.add_argument("--generator-path", type=str, required=True, help="Generator checkpoint path")
    eval_parser.add_argument("--verifier-path", type=str, help="Verifier checkpoint path")
    eval_parser.add_argument("--base-model", type=str, required=True, help="Base model name")
    eval_parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    eval_parser.add_argument("--domains", type=str, default="math,logical,commonsense", help="Domains to evaluate")
    eval_parser.add_argument("--num-samples", type=int, default=64, help="Solutions per question")
    eval_parser.add_argument("--self-consistency", action="store_true", help="Include self-consistency baseline")
    eval_parser.add_argument("--output", type=str, help="Output file for results")
    eval_parser.add_argument("--device", type=str, default="cuda", help="Device")
    eval_parser.set_defaults(func=evaluate)

    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Generate solutions")
    sample_parser.add_argument("--model-path", type=str, required=True, help="Model checkpoint path")
    sample_parser.add_argument("--base-model", type=str, required=True, help="Base model name")
    sample_parser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    sample_parser.add_argument("--domain", type=str, required=True, help="Domain")
    sample_parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    sample_parser.add_argument("--output", type=str, required=True, help="Output file")
    sample_parser.add_argument("--num-samples", type=int, default=16, help="Solutions per question")
    sample_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    sample_parser.add_argument("--max-questions", type=int, help="Max questions to process")
    sample_parser.add_argument("--device", type=str, default="cuda", help="Device")
    sample_parser.set_defaults(func=sample)

    # List models command
    list_parser = subparsers.add_parser("list-models", help="List available models")
    list_parser.set_defaults(func=list_models)

    # Parse and execute
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
