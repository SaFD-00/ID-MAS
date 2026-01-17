"""End-to-end training pipeline for ReGenesis.

Integrates data generation, filtering, and training into a unified workflow.
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

from src.config import get_model_config, MODEL_CONFIGS, list_available_models
from src.config.training_config import get_size_category, get_training_config
from src.reasoning.multi_model_reasoning_gen import generate_reasoning_data
from src.pipeline.filtering import process_and_filter_data, merge_filtered_datasets
from src.finetune_code.multi_model_finetune import TrainingPipeline

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# Dataset configurations
DATASET_CONFIGS = {
    "gsm8k": {"answer_type": "numeric", "domain": "math"},
    "math": {"answer_type": "numeric", "domain": "math"},
    "reclor": {"answer_type": "option", "domain": "logical"},
    "arc_c": {"answer_type": "option", "domain": "commonsense"},
}


class ReGenesisPipeline:
    """End-to-end ReGenesis pipeline."""

    def __init__(
        self,
        model_name: str,
        datasets: List[str],
        output_dir: str,
        num_samples: int = 20,
        max_paths_per_task: int = 5,
        tensor_parallel_size: Optional[int] = None,
    ):
        """Initialize pipeline.

        Args:
            model_name: Model name or alias
            datasets: List of dataset names to process
            output_dir: Base output directory
            num_samples: Number of reasoning paths to generate per task
            max_paths_per_task: Maximum paths to keep after filtering
            tensor_parallel_size: Override tensor parallel size
        """
        self.model_name = model_name
        self.datasets = datasets
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.max_paths_per_task = max_paths_per_task
        self.tensor_parallel_size = tensor_parallel_size

        # Get model config
        self.model_config = get_model_config(model_name)
        self.model_short = model_name.split("/")[-1]

        # Setup directories
        self.generated_dir = os.path.join(output_dir, "generated", self.model_short)
        self.filtered_dir = os.path.join(output_dir, "filtered", self.model_short)
        self.checkpoints_dir = os.path.join(output_dir, "checkpoints", self.model_short)

        os.makedirs(self.generated_dir, exist_ok=True)
        os.makedirs(self.filtered_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def run_data_generation(
        self,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ) -> Dict[str, str]:
        """Stage 1: Generate reasoning paths for all datasets.

        Returns:
            Dictionary mapping dataset names to output file paths
        """
        logger.info("=" * 50)
        logger.info("Stage 1: Reasoning Path Generation")
        logger.info("=" * 50)

        output_files = {}

        for dataset_name in self.datasets:
            logger.info(f"\nGenerating for dataset: {dataset_name}")

            output_file = generate_reasoning_data(
                model_name=self.model_name,
                dataset_name=dataset_name,
                output_dir=os.path.join(self.output_dir, "generated"),
                start_idx=start_idx,
                end_idx=end_idx,
                num_samples=self.num_samples,
                tensor_parallel_size=self.tensor_parallel_size,
            )
            output_files[dataset_name] = output_file

        logger.info(f"\nGenerated files: {output_files}")
        return output_files

    def run_filtering(self) -> Dict[str, str]:
        """Stage 2: Filter generated reasoning paths.

        Returns:
            Dictionary mapping dataset names to filtered file paths
        """
        logger.info("=" * 50)
        logger.info("Stage 2: Data Filtering")
        logger.info("=" * 50)

        filtered_files = {}

        for dataset_name in self.datasets:
            logger.info(f"\nFiltering dataset: {dataset_name}")

            config = DATASET_CONFIGS.get(dataset_name, {"answer_type": "numeric"})

            filtered_file = process_and_filter_data(
                input_dir=self.generated_dir,
                output_dir=self.filtered_dir,
                dataset_name=dataset_name,
                answer_type=config["answer_type"],
                max_paths_per_task=self.max_paths_per_task,
            )
            filtered_files[dataset_name] = filtered_file

        logger.info(f"\nFiltered files: {filtered_files}")
        return filtered_files

    def run_merge_data(self) -> str:
        """Merge all filtered datasets into one training file.

        Returns:
            Path to merged training file
        """
        logger.info("=" * 50)
        logger.info("Merging filtered datasets")
        logger.info("=" * 50)

        merged_file = os.path.join(
            self.filtered_dir,
            f"merged_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        merge_filtered_datasets(
            filtered_dir=self.filtered_dir,
            output_path=merged_file,
            datasets=self.datasets,
        )

        return merged_file

    def run_training(
        self,
        train_data_path: str,
        use_lora: Optional[bool] = None,
        load_in_4bit: bool = False,
        **training_kwargs,
    ) -> str:
        """Stage 3: Fine-tune model on filtered data.

        Args:
            train_data_path: Path to training data JSON
            use_lora: Whether to use LoRA (auto-detect if None)
            load_in_4bit: Whether to use 4-bit quantization

        Returns:
            Path to saved checkpoint
        """
        logger.info("=" * 50)
        logger.info("Stage 3: Model Training")
        logger.info("=" * 50)

        # Determine dataset type from first dataset
        dataset_type = DATASET_CONFIGS.get(
            self.datasets[0], {"domain": "math"}
        )["domain"]

        pipeline = TrainingPipeline(
            model_name=self.model_name,
            data_path=train_data_path,
            output_dir=self.checkpoints_dir,
            dataset_type=dataset_type,
            use_lora=use_lora,
            load_in_4bit=load_in_4bit,
            **training_kwargs,
        )

        return pipeline.train()

    def run_full_pipeline(
        self,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        skip_generation: bool = False,
        skip_filtering: bool = False,
        use_lora: Optional[bool] = None,
        load_in_4bit: bool = False,
        **training_kwargs,
    ) -> Dict[str, Any]:
        """Run complete ReGenesis pipeline.

        Args:
            start_idx: Starting index for data generation
            end_idx: Ending index for data generation
            skip_generation: Skip data generation stage
            skip_filtering: Skip filtering stage
            use_lora: Use LoRA training
            load_in_4bit: Use 4-bit quantization

        Returns:
            Dictionary with pipeline results
        """
        logger.info("=" * 60)
        logger.info(f"ReGenesis Full Pipeline: {self.model_name}")
        logger.info(f"Datasets: {self.datasets}")
        logger.info("=" * 60)

        results = {
            "model_name": self.model_name,
            "datasets": self.datasets,
            "output_dir": self.output_dir,
        }

        # Stage 1: Data Generation
        if not skip_generation:
            results["generated_files"] = self.run_data_generation(start_idx, end_idx)
        else:
            logger.info("Skipping data generation stage")

        # Stage 2: Filtering
        if not skip_filtering:
            results["filtered_files"] = self.run_filtering()
        else:
            logger.info("Skipping filtering stage")

        # Merge data
        merged_file = self.run_merge_data()
        results["merged_file"] = merged_file

        # Stage 3: Training
        results["checkpoint_path"] = self.run_training(
            train_data_path=merged_file,
            use_lora=use_lora,
            load_in_4bit=load_in_4bit,
            **training_kwargs,
        )

        logger.info("=" * 60)
        logger.info("Pipeline completed!")
        logger.info(f"Results: {results}")
        logger.info("=" * 60)

        return results


def run_for_all_models(
    models: List[str],
    datasets: List[str],
    output_dir: str,
    **kwargs,
) -> Dict[str, Any]:
    """Run pipeline for multiple models.

    Args:
        models: List of model names
        datasets: List of dataset names
        output_dir: Base output directory
        **kwargs: Additional arguments for pipeline

    Returns:
        Dictionary with results for all models
    """
    all_results = {}

    for model_name in models:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"Processing model: {model_name}")
        logger.info(f"{'#' * 60}\n")

        try:
            pipeline = ReGenesisPipeline(
                model_name=model_name,
                datasets=datasets,
                output_dir=output_dir,
                **kwargs,
            )
            results = pipeline.run_full_pipeline(**kwargs)
            all_results[model_name] = {"status": "success", "results": results}
        except Exception as e:
            logger.error(f"Failed for model {model_name}: {e}")
            all_results[model_name] = {"status": "failed", "error": str(e)}

    return all_results


def main():
    parser = argparse.ArgumentParser(description="ReGenesis Training Pipeline")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (e.g., meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["gsm8k", "math", "reclor", "arc_c"],
        help="Datasets to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of reasoning paths per task",
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
        help="Ending index",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip data generation",
    )
    parser.add_argument(
        "--skip_filtering",
        action="store_true",
        help="Skip filtering",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA training",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Use 4-bit quantization",
    )

    args = parser.parse_args()

    pipeline = ReGenesisPipeline(
        model_name=args.model_name,
        datasets=args.datasets,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
    )

    pipeline.run_full_pipeline(
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        skip_generation=args.skip_generation,
        skip_filtering=args.skip_filtering,
        use_lora=args.use_lora,
        load_in_4bit=args.load_in_4bit,
    )


if __name__ == "__main__":
    main()
