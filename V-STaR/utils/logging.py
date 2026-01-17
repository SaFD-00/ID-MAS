"""Logging Utilities for V-STaR"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime
import json


# Global logger registry
_loggers: Dict[str, logging.Logger] = {}


def setup_logger(
    name: str = "vstar",
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """
    Setup a logger

    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Logging level
        console: Enable console output
        file: Enable file output

    Returns:
        Configured logger
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers = []

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def get_logger(name: str = "vstar") -> logging.Logger:
    """
    Get a logger by name

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]


class TrainingLogger:
    """
    Training logger for V-STaR

    Logs training progress, metrics, and events
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: str = "vstar",
        log_level: int = logging.INFO,
    ):
        """
        Initialize training logger

        Args:
            log_dir: Directory for logs
            experiment_name: Name of experiment
            log_level: Logging level
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = setup_logger(
            name=experiment_name,
            log_dir=str(self.log_dir),
            log_level=log_level,
        )

        # Metrics file
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.jsonl"

        # Events file
        self.events_file = self.log_dir / f"{experiment_name}_events.jsonl"

    def info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)

    def debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        iteration: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """
        Log metrics

        Args:
            metrics: Dictionary of metrics
            step: Training step
            iteration: V-STaR iteration
            prefix: Prefix for metric names
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "iteration": iteration,
            "metrics": {
                f"{prefix}{k}" if prefix else k: v
                for k, v in metrics.items()
            },
        }

        # Write to metrics file
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Log to console
        metrics_str = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        self.logger.info(f"Metrics: {metrics_str}")

    def log_event(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an event

        Args:
            event_type: Type of event
            data: Event data
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data or {},
        }

        # Write to events file
        with open(self.events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        self.logger.info(f"Event: {event_type}")

    def log_iteration_start(self, iteration: int, config: Dict[str, Any]) -> None:
        """Log start of an iteration"""
        self.log_event(
            event_type="iteration_start",
            data={"iteration": iteration, "config": config},
        )
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Starting iteration {iteration}")
        self.logger.info(f"{'='*60}")

    def log_iteration_end(
        self,
        iteration: int,
        metrics: Dict[str, Any],
    ) -> None:
        """Log end of an iteration"""
        self.log_event(
            event_type="iteration_end",
            data={"iteration": iteration, "metrics": metrics},
        )
        self.log_metrics(metrics, iteration=iteration)
        self.logger.info(f"Iteration {iteration} complete")

    def log_sampling(
        self,
        num_questions: int,
        num_correct: int,
        num_total: int,
    ) -> None:
        """Log sampling results"""
        accuracy = num_correct / num_total if num_total > 0 else 0
        self.log_metrics({
            "num_questions": num_questions,
            "num_correct": num_correct,
            "num_total": num_total,
            "sampling_accuracy": accuracy,
        }, prefix="sampling/")

    def log_sft_training(
        self,
        loss: float,
        num_samples: int,
        learning_rate: Optional[float] = None,
    ) -> None:
        """Log SFT training results"""
        metrics = {
            "loss": loss,
            "num_samples": num_samples,
        }
        if learning_rate:
            metrics["learning_rate"] = learning_rate
        self.log_metrics(metrics, prefix="sft/")

    def log_dpo_training(
        self,
        loss: float,
        num_pairs: int,
        learning_rate: Optional[float] = None,
    ) -> None:
        """Log DPO training results"""
        metrics = {
            "loss": loss,
            "num_pairs": num_pairs,
        }
        if learning_rate:
            metrics["learning_rate"] = learning_rate
        self.log_metrics(metrics, prefix="dpo/")

    def log_evaluation(
        self,
        metrics: Dict[str, Any],
        dataset: Optional[str] = None,
    ) -> None:
        """Log evaluation results"""
        self.log_event(
            event_type="evaluation",
            data={"dataset": dataset, "metrics": metrics},
        )
        self.log_metrics(metrics, prefix="eval/")

    def get_metrics_history(self) -> list:
        """Get all logged metrics"""
        if not self.metrics_file.exists():
            return []

        metrics = []
        with open(self.metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))
        return metrics

    def get_events_history(self) -> list:
        """Get all logged events"""
        if not self.events_file.exists():
            return []

        events = []
        with open(self.events_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        return events


def create_training_logger(
    log_dir: str = "./logs",
    experiment_name: str = "vstar",
) -> TrainingLogger:
    """Factory function for TrainingLogger"""
    return TrainingLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
    )
