"""Multi-model reasoning path generation for ReGenesis.

Refactored version with configuration-based multi-model support.
Based on original reasoning_paths_gen.py (BSD 3-Clause License, Chris Taylor 2024).
"""

import json
import os
import argparse
from typing import List, Dict, Optional, Any
from tqdm import tqdm
from vllm import LLM, SamplingParams

from src.config import get_model_config, MODEL_CONFIGS
from src.config.training_config import DEFAULT_GENERATION_CONFIG
from src.reasoning.template_utils import (
    format_prompt,
    get_stop_tokens,
    detect_template_from_model,
    get_instruction_prefix,
)
from src.reasoning.read_datasets import (
    load_gsm8k,
    load_arc_c,
    load_reclor,
    load_math,
)


# 25 Seed Reasoning Modules from ReGenesis paper
REASONING_MODULES = [
    "How could I devise an experiment to help solve that problem?",
    "Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
    "How could I measure progress on this problem?",
    "How can I simplify the problem so that it is easier to solve?",
    "How can I break down this problem into smaller, more manageable parts?",
    "Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
    "Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
    "Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
    "Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
    "What is the core issue or problem that needs to be addressed?",
    "What are the potential obstacles or challenges that might arise in solving this problem?",
    "Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
    "How can progress or success in solving the problem be measured or evaluated?",
    "What indicators or metrics can be used?",
    "Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
    "Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
    "Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
    "Is the problem a design challenge that requires creative solutions and innovation?",
    "Does the problem require addressing systemic or structural issues rather than just individual instances?",
    "What kinds of solution typically are produced for this kind of problem specification?",
    "Let's think step by step.",
    "Let's make a step by step plan and implement it with good notation and explanation.",
    "Ignoring the current best solution, create an entirely new solution to the problem.",
    "Let's imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?",
    "What is the best way to modify this current best solution, given what you know about these kinds of problem specification?",
]


class ReasoningPathGenerator:
    """Multi-model reasoning path generator."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: Optional[int] = None,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
    ):
        """Initialize generator.

        Args:
            model_name: HuggingFace model name or path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum model context length
            gpu_memory_utilization: GPU memory utilization ratio
        """
        self.model_name = model_name
        self.config = get_model_config(model_name)
        self.template = self.config.template

        # Override tensor parallel if specified
        tp_size = tensor_parallel_size or self.config.tensor_parallel

        print(f"Loading model: {model_name}")
        print(f"Template: {self.template}, Tensor Parallel: {tp_size}")

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )

        self.stop_tokens = get_stop_tokens(self.template)

    def _create_sampling_params(
        self,
        temperature: float = 0.85,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> SamplingParams:
        """Create sampling parameters."""
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=self.stop_tokens,
        )

    def _format_prompt(self, user_message: str) -> str:
        """Format prompt with model-specific template."""
        return format_prompt(
            self.template,
            user_message,
            self.config.system_message,
        )

    def _generate_batch(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
    ) -> List[str]:
        """Generate responses for batch of prompts."""
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def select_reasoning_modules(
        self,
        task_descriptions: List[str],
        sampling_params: SamplingParams,
    ) -> List[str]:
        """Stage 1-1: SELECT relevant reasoning modules for tasks."""
        modules_text = "\n".join(
            f"{i}. {REASONING_MODULES[i]}"
            for i in range(len(REASONING_MODULES))
        )

        prompts = []
        for task in task_descriptions:
            prompt = (
                f"Given the task: {task}, which of the following reasoning modules are relevant?\n\n"
                f"{modules_text}\n\n"
                "Requirements: Do not elaborate on why. Only choose 2-3 reasoning modules. Do not solve this task."
            )
            prompts.append(self._format_prompt(prompt))

        return self._generate_batch(prompts, sampling_params)

    def adapt_reasoning_modules(
        self,
        selected_modules_lst: List[str],
        task_example: str,
        sampling_params: SamplingParams,
    ) -> List[str]:
        """Stage 1-2: ADAPT selected modules to be task-specific."""
        prompts = []
        for selected_modules in selected_modules_lst:
            prompt = (
                f"Without working out the full solution, adapt the following general modules "
                f"to be specific and concise to our task:\n{selected_modules}\n\n"
                f"Our task:\n{task_example}.\n\n"
                "Note: You will not work out the full solution and only adapt the general module "
                "to be specific in one or two sentences."
            )
            prompts.append(self._format_prompt(prompt))

        return self._generate_batch(prompts, sampling_params)

    def implement_reasoning_structure(
        self,
        adapted_modules_lst: List[str],
        task_description: str,
        sampling_params: SamplingParams,
    ) -> List[str]:
        """Stage 1-3: IMPLEMENT adapted modules into reasoning structure."""
        prompts = []
        for adapted_modules in adapted_modules_lst:
            prompt = (
                f"Without working out the full solution, create an actionable and concise "
                f"reasoning structure step by step for the task using the adapted reasoning module:\n"
                f"{adapted_modules}\n\n"
                f"Task Description:\n{task_description}\n\n"
                "Note: You will not work out the full solution and only create an actionable "
                "and concise reasoning structure for the task using the adapted reasoning module."
            )
            prompts.append(self._format_prompt(prompt))

        return self._generate_batch(prompts, sampling_params)

    def execute_reasoning_structure(
        self,
        reasoning_structures: List[str],
        task_instance: str,
        sampling_params: SamplingParams,
    ) -> List[str]:
        """Stage 2: Execute reasoning structure to solve task."""
        prompts = []
        for structure in reasoning_structures:
            prompt = (
                f"Using the following reasoning structure: {structure}\n\n"
                f"Solve this task step by step based on the given reasoning structure, "
                f"and present your final answer as \\boxed{{Your Answer}}.: {task_instance}"
            )
            prompts.append(self._format_prompt(prompt))

        return self._generate_batch(prompts, sampling_params)

    def generate_reasoning_paths(
        self,
        task: str,
        num_samples: int = 20,
        temperature: float = 0.85,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """Generate multiple diverse reasoning paths for a task.

        Args:
            task: Task description/question
            num_samples: Number of diverse reasoning paths to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens for generation

        Returns:
            Dictionary containing all generated paths and metadata
        """
        sampling_params = self._create_sampling_params(temperature, top_p, max_tokens)

        # Stage 1: Generate diverse reasoning structures
        selected_modules = self.select_reasoning_modules(
            [task] * num_samples, sampling_params
        )
        # Add original modules for more diversity
        all_modules = selected_modules + REASONING_MODULES

        adapted_modules = self.adapt_reasoning_modules(
            all_modules, task, sampling_params
        )

        reasoning_structures = self.implement_reasoning_structure(
            adapted_modules, task, sampling_params
        )

        # Stage 2: Execute reasoning structures
        solutions = self.execute_reasoning_structure(
            reasoning_structures, task, sampling_params
        )

        return {
            "task": task,
            "num_paths": len(solutions),
            "selected_modules": all_modules,
            "adapted_modules": adapted_modules,
            "reasoning_structures": reasoning_structures,
            "solutions": solutions,
        }


def load_dataset(dataset_name: str) -> Dict[str, List]:
    """Load dataset by name.

    Args:
        dataset_name: One of "gsm8k", "math", "reclor", "arc_c"

    Returns:
        Dictionary with "instruction" and "answer" keys
    """
    loaders = {
        "gsm8k": load_gsm8k,
        "math": load_math,
        "reclor": load_reclor,
        "arc_c": load_arc_c,
    }

    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")

    return loaders[dataset_name]()


def get_dataset_type(dataset_name: str) -> str:
    """Get dataset type for instruction prefix."""
    dataset_types = {
        "gsm8k": "math",
        "math": "math",
        "reclor": "logical",
        "arc_c": "commonsense",
    }
    return dataset_types.get(dataset_name, "math")


def generate_reasoning_data(
    model_name: str,
    dataset_name: str,
    output_dir: str,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    num_samples: int = 20,
    tensor_parallel_size: Optional[int] = None,
) -> str:
    """Generate reasoning paths for a dataset.

    Args:
        model_name: Model name or path
        dataset_name: Dataset name
        output_dir: Output directory
        start_idx: Starting index in dataset
        end_idx: Ending index (None for all)
        num_samples: Number of reasoning paths per task
        tensor_parallel_size: Override tensor parallel size

    Returns:
        Path to output file
    """
    # Load dataset
    dataset = load_dataset(dataset_name)
    dataset_type = get_dataset_type(dataset_name)

    # Initialize generator
    generator = ReasoningPathGenerator(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
    )

    # Create output directory
    model_short = model_name.split("/")[-1]
    out_dir = os.path.join(output_dir, model_short)
    os.makedirs(out_dir, exist_ok=True)

    output_file = os.path.join(
        out_dir, f"{dataset_name}_{start_idx}_{end_idx or 'all'}.jsonl"
    )

    # Process dataset
    end_idx = end_idx or len(dataset["instruction"])

    with open(output_file, "w") as f:
        for i in tqdm(range(start_idx, min(end_idx, len(dataset["instruction"])))):
            task = dataset["instruction"][i]
            truth = dataset["answer"][i]

            # Add instruction prefix
            prefix = get_instruction_prefix(dataset_type)
            full_task = prefix + task if prefix else task

            # Generate reasoning paths
            results = generator.generate_reasoning_paths(
                full_task, num_samples=num_samples
            )

            # Save each path as separate entry
            for j, solution in enumerate(results["solutions"]):
                entry = {
                    "instruction": task,
                    "answer": truth,
                    "module": results["selected_modules"][j] if j < len(results["selected_modules"]) else "",
                    "reasoning": solution,
                    "adapted_module": results["adapted_modules"][j] if j < len(results["adapted_modules"]) else "",
                    "reasoning_structure": results["reasoning_structures"][j] if j < len(results["reasoning_structures"]) else "",
                }
                f.write(json.dumps(entry) + "\n")

    print(f"Saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Generate reasoning paths with ReGenesis")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (e.g., meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["gsm8k", "math", "reclor", "arc_c"],
        help="Dataset name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/generated",
        help="Output directory",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting index",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="Ending index (None for all)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of reasoning paths per task",
    )
    parser.add_argument(
        "--tensor_parallel",
        type=int,
        default=None,
        help="Override tensor parallel size",
    )

    args = parser.parse_args()

    generate_reasoning_data(
        model_name=args.model_name,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        num_samples=args.num_samples,
        tensor_parallel_size=args.tensor_parallel,
    )


if __name__ == "__main__":
    main()
