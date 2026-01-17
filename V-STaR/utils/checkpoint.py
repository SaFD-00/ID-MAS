"""Checkpoint Management for V-STaR"""

from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import shutil

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint"""
    iteration: int
    timestamp: str
    model_name: str
    checkpoint_type: str  # "generator", "verifier", "state"
    metrics: Dict[str, Any]
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save metadata to file"""
        with open(path / "metadata.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CheckpointMetadata":
        """Load metadata from file"""
        with open(path / "metadata.json", "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class CheckpointManager:
    """
    Checkpoint Manager for V-STaR

    Handles saving, loading, and managing checkpoints
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        max_checkpoints: int = 5,
    ):
        """
        Initialize checkpoint manager

        Args:
            base_dir: Base directory for checkpoints
            max_checkpoints: Maximum checkpoints to keep per type
        """
        self.base_dir = Path(base_dir)
        self.max_checkpoints = max_checkpoints
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_dir(
        self,
        iteration: int,
        checkpoint_type: str,
    ) -> Path:
        """Get directory for a checkpoint"""
        return self.base_dir / f"iteration_{iteration}" / checkpoint_type

    def save_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        iteration: int,
        checkpoint_type: str,
        model_name: str,
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a model checkpoint

        Args:
            model: Model to save
            tokenizer: Tokenizer to save
            iteration: Current iteration
            checkpoint_type: Type of checkpoint (generator/verifier)
            model_name: Model name
            metrics: Optional metrics
            config: Optional config

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = self.get_checkpoint_dir(iteration, checkpoint_type)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        if isinstance(model, PeftModel):
            model.save_pretrained(checkpoint_dir)
        else:
            model.save_pretrained(checkpoint_dir)

        # Save tokenizer
        tokenizer.save_pretrained(checkpoint_dir)

        # Save metadata
        metadata = CheckpointMetadata(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            checkpoint_type=checkpoint_type,
            metrics=metrics or {},
            config=config or {},
        )
        metadata.save(checkpoint_dir)

        print(f"Checkpoint saved: {checkpoint_dir}")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(checkpoint_type)

        return checkpoint_dir

    def save_state(
        self,
        state: Dict[str, Any],
        iteration: int,
        model_name: str,
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save training state

        Args:
            state: State dictionary
            iteration: Current iteration
            model_name: Model name
            metrics: Optional metrics
            config: Optional config

        Returns:
            Path to saved state
        """
        state_dir = self.get_checkpoint_dir(iteration, "state")
        state_dir.mkdir(parents=True, exist_ok=True)

        # Save state
        with open(state_dir / "state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        # Save metadata
        metadata = CheckpointMetadata(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            checkpoint_type="state",
            metrics=metrics or {},
            config=config or {},
        )
        metadata.save(state_dir)

        print(f"State saved: {state_dir}")

        return state_dir

    def load_state(self, iteration: int) -> Dict[str, Any]:
        """Load training state"""
        state_dir = self.get_checkpoint_dir(iteration, "state")

        with open(state_dir / "state.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def get_latest_checkpoint(
        self,
        checkpoint_type: str,
    ) -> Optional[Path]:
        """Get the latest checkpoint of a type"""
        checkpoints = self.list_checkpoints(checkpoint_type)

        if not checkpoints:
            return None

        # Sort by iteration (descending)
        checkpoints.sort(key=lambda x: x["iteration"], reverse=True)

        return Path(checkpoints[0]["path"])

    def get_best_checkpoint(
        self,
        checkpoint_type: str,
        metric: str = "accuracy",
        higher_is_better: bool = True,
    ) -> Optional[Path]:
        """Get the best checkpoint based on a metric"""
        checkpoints = self.list_checkpoints(checkpoint_type)

        if not checkpoints:
            return None

        # Filter checkpoints with the metric
        valid = [c for c in checkpoints if metric in c.get("metrics", {})]

        if not valid:
            return None

        # Sort by metric
        valid.sort(
            key=lambda x: x["metrics"][metric],
            reverse=higher_is_better,
        )

        return Path(valid[0]["path"])

    def list_checkpoints(
        self,
        checkpoint_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all checkpoints

        Args:
            checkpoint_type: Filter by type (optional)

        Returns:
            List of checkpoint info dictionaries
        """
        checkpoints = []

        for iteration_dir in self.base_dir.glob("iteration_*"):
            if not iteration_dir.is_dir():
                continue

            for type_dir in iteration_dir.iterdir():
                if not type_dir.is_dir():
                    continue

                if checkpoint_type and type_dir.name != checkpoint_type:
                    continue

                metadata_path = type_dir / "metadata.json"
                if not metadata_path.exists():
                    continue

                try:
                    metadata = CheckpointMetadata.load(type_dir)
                    checkpoints.append({
                        "path": str(type_dir),
                        "iteration": metadata.iteration,
                        "timestamp": metadata.timestamp,
                        "model_name": metadata.model_name,
                        "checkpoint_type": metadata.checkpoint_type,
                        "metrics": metadata.metrics,
                    })
                except Exception as e:
                    print(f"Warning: Could not load metadata from {type_dir}: {e}")

        return checkpoints

    def _cleanup_old_checkpoints(self, checkpoint_type: str) -> None:
        """Remove old checkpoints exceeding max_checkpoints"""
        checkpoints = self.list_checkpoints(checkpoint_type)

        if len(checkpoints) <= self.max_checkpoints:
            return

        # Sort by iteration (ascending, oldest first)
        checkpoints.sort(key=lambda x: x["iteration"])

        # Remove oldest
        to_remove = len(checkpoints) - self.max_checkpoints

        for checkpoint in checkpoints[:to_remove]:
            path = Path(checkpoint["path"])
            if path.exists():
                shutil.rmtree(path)
                print(f"Removed old checkpoint: {path}")

    def delete_checkpoint(self, path: Union[str, Path]) -> bool:
        """Delete a specific checkpoint"""
        path = Path(path)

        if not path.exists():
            return False

        shutil.rmtree(path)
        print(f"Deleted checkpoint: {path}")
        return True

    def get_checkpoint_info(self, path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Get information about a checkpoint"""
        path = Path(path)

        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            return None

        try:
            metadata = CheckpointMetadata.load(path)
            return {
                "path": str(path),
                "iteration": metadata.iteration,
                "timestamp": metadata.timestamp,
                "model_name": metadata.model_name,
                "checkpoint_type": metadata.checkpoint_type,
                "metrics": metadata.metrics,
                "config": metadata.config,
            }
        except Exception as e:
            print(f"Error loading checkpoint info: {e}")
            return None


def get_checkpoint_manager(
    base_dir: str = "./checkpoints",
    max_checkpoints: int = 5,
) -> CheckpointManager:
    """Factory function for CheckpointManager"""
    return CheckpointManager(
        base_dir=base_dir,
        max_checkpoints=max_checkpoints,
    )
