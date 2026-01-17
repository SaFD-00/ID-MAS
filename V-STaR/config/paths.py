"""Path configuration for V-STaR"""

from pathlib import Path
from typing import Optional
from datetime import datetime

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data directory (shared with ID-MAS)
DATA_DIR = PROJECT_ROOT / "data"

# ID-MAS data directory (fallback)
IDMAS_DATA_DIR = PROJECT_ROOT.parent / "ID-MAS" / "data"

# Checkpoint directory
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir() -> Path:
    """Get data directory (prefer V-STaR, fallback to ID-MAS)"""
    if DATA_DIR.exists():
        return DATA_DIR
    if IDMAS_DATA_DIR.exists():
        return IDMAS_DATA_DIR
    raise FileNotFoundError(
        f"Data directory not found. Tried: {DATA_DIR}, {IDMAS_DATA_DIR}"
    )


def get_checkpoint_path(
    model_name: str,
    iteration: int,
    checkpoint_type: str = "generator"
) -> Path:
    """
    Get checkpoint path for a model

    Args:
        model_name: Model name (full or short)
        iteration: V-STaR iteration number
        checkpoint_type: "generator", "verifier", or "gsft"

    Returns:
        Path to checkpoint directory
    """
    from .models import get_model_short_name
    short_name = get_model_short_name(model_name)

    if checkpoint_type == "gsft":
        # G_SFT is the reference model, saved once
        path = CHECKPOINT_DIR / short_name / "gsft"
    else:
        path = CHECKPOINT_DIR / short_name / checkpoint_type / f"iter_{iteration}"

    return ensure_dir(path)


def get_output_path(
    model_name: str,
    domain: str,
    dataset: str,
    output_type: str = "results"
) -> Path:
    """
    Get output path for results

    Args:
        model_name: Model name
        domain: Domain name
        dataset: Dataset name
        output_type: "results", "generated", or "preference"

    Returns:
        Path to output directory
    """
    from .models import get_model_short_name
    short_name = get_model_short_name(model_name)

    path = OUTPUT_DIR / short_name / domain / dataset / output_type
    return ensure_dir(path)


def get_log_path(
    experiment_name: Optional[str] = None
) -> Path:
    """Get log file path"""
    ensure_dir(LOGS_DIR)

    if experiment_name:
        return LOGS_DIR / f"{experiment_name}.log"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"vstar_{timestamp}.log"


def get_generated_solutions_path(
    model_name: str,
    domain: str,
    dataset: str,
    iteration: int
) -> Path:
    """Get path for generated solutions"""
    from .models import get_model_short_name
    short_name = get_model_short_name(model_name)

    path = OUTPUT_DIR / short_name / domain / dataset / "generated"
    ensure_dir(path)
    return path / f"solutions_iter_{iteration}.json"


def get_preference_data_path(
    model_name: str,
    domain: str,
    dataset: str,
    iteration: int
) -> Path:
    """Get path for preference data (DPO training)"""
    from .models import get_model_short_name
    short_name = get_model_short_name(model_name)

    path = OUTPUT_DIR / short_name / domain / dataset / "preference"
    ensure_dir(path)
    return path / f"preference_iter_{iteration}.json"


def get_evaluation_results_path(
    model_name: str,
    domain: str,
    dataset: str,
    method: str = "best_of_k"
) -> Path:
    """Get path for evaluation results"""
    from .models import get_model_short_name
    short_name = get_model_short_name(model_name)

    path = OUTPUT_DIR / short_name / domain / dataset / "results"
    ensure_dir(path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path / f"{method}_{timestamp}.json"
