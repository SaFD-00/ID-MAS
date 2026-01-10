"""
LangGraph State Schema for ID-MAS 3-Phase Pipeline.

This module defines the state structure used throughout the pipeline:
- IDMASState: Main state schema for the graph
- QuestionResult: Result for each processed question
- DesignResult: Instructional design output

Based on the research proposal's 3-Phase Pipeline architecture.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, Optional, List, Dict, Any, Annotated, Set, Tuple
from datetime import datetime
from enum import Enum
import operator


class SFTCase(str, Enum):
    """SFT data case classification based on which phase produced the correct response."""
    A = "A"           # Phase 1 correct (initial response)
    A_FAILED = "A-Failed"  # Phase 1 failed after max iterations (reconstructed)
    B = "B"           # Phase 2 correct (fixed response with coaching)
    C = "C"           # Phase 3 (teacher modeling response)


class QuestionResultRequired(TypedDict):
    """Required fields for QuestionResult."""
    id: str                # question_id → id (필수)
    instruction: str       # 순서 변경: 두 번째 (필수)
    input: str             # question → input (필수)
    output: str            # ground_truth → output (필수)
    _problem_text: str     # 내부 처리용 (로그 저장 시 제거, 필수)


class QuestionResult(QuestionResultRequired, total=False):
    """Result for a single question processed through the pipeline."""
    # Phase 1 results
    initial_response: str
    predicted_answer: Optional[str]
    phase1_correct: bool

    # Phase 2 results (if Phase 1 incorrect)
    phase2_scores: Optional[Dict[str, Any]]
    fixed_response: Optional[str]
    fixed_predicted: Optional[str]
    phase2_correct: Optional[bool]
    coaching_db_used: bool

    # Phase 3 results (if Phase 2 incorrect)
    modeling_response: Optional[str]
    phase3_applied: bool

    # Final SFT classification
    sft_case: Optional[str]
    sft_response: Optional[str]

    # Iterative scaffolding details (Phase 1)
    iterative_scaffolding: Optional[Dict[str, Any]]
    reconstruction: Optional[Dict[str, Any]]


class DesignResult(TypedDict, total=False):
    """Instructional design output."""
    domain: str
    train_dataset: str
    identifier: str
    terminal_goal: str
    learning_objective: str
    instructional_analysis: Dict[str, Any]
    performance_objectives: Dict[str, Any]
    rubrics: Dict[str, Any]
    timestamp: str


def add_to_list(existing: List, new: Any) -> List:
    """Reducer function to append items to a list."""
    if isinstance(new, list):
        return existing + new
    return existing + [new]


class IDMASState(TypedDict, total=False):
    """
    Main state schema for the ID-MAS LangGraph pipeline.

    This state is passed through all nodes and maintains the full
    context of the learning pipeline execution.

    The state follows the 3-Phase Pipeline architecture from the research proposal:
    1. Instructional Design Phase (optional, can load existing)
    2. Phase 1: Scaffolding - Initial response generation
    3. Phase 2: Coaching - Teacher intervention for incorrect answers
    4. Phase 3: Modeling - Teacher demonstration for still-incorrect answers
    """

    # ==================== Configuration ====================
    domain: str
    train_dataset: str
    terminal_goal: str
    student_model_name: str
    teacher_model_name: str
    model_short: str
    checkpoint_interval: int
    use_iterative_scaffolding: bool
    max_iterations: int

    # ==================== Design Phase ====================
    design_result: Optional[DesignResult]
    task_analysis: str
    performance_objectives: List[Dict[str, Any]]
    rubric: Optional[Dict[str, Any]]

    # ==================== Questions ====================
    questions: List[Dict[str, Any]]
    total_questions: int
    current_question_index: int
    current_question: Optional[Dict[str, Any]]

    # ==================== Phase 1 Results ====================
    # Using Annotated with reducer for accumulating results
    phase1_results: Annotated[List[QuestionResult], add_to_list]
    phase1_processed: int
    phase1_correct_count: int
    incorrect_after_phase1: Annotated[List[QuestionResult], add_to_list]

    # Iterative scaffolding statistics
    phase1_first_attempt_correct: int
    phase1_multi_attempt_correct: int
    phase1_failed_reconstructed: int

    # ==================== Phase 2 Results ====================
    coaching_db: Optional[Dict[str, Any]]
    weak_objectives: List[Dict[str, Any]]
    phase2_results: Annotated[List[QuestionResult], add_to_list]
    phase2_processed: int
    still_incorrect_after_phase2: Annotated[List[QuestionResult], add_to_list]

    # ==================== Phase 3 Results ====================
    phase3_results: Annotated[List[QuestionResult], add_to_list]
    phase3_processed: int

    # ==================== SFT Data ====================
    sft_data: List[Dict[str, Any]]

    # ==================== Pipeline Control ====================
    current_phase: str  # "design", "phase1", "phase2", "phase3", "complete"
    is_complete: bool
    error_message: Optional[str]

    # ==================== Timestamps ====================
    started_at: Optional[str]
    updated_at: Optional[str]

    # ==================== Checkpoint ====================
    checkpoint_path: Optional[str]
    last_checkpoint_at: Optional[str]


def create_initial_state(
    domain: str,
    train_dataset: str,
    terminal_goal: str,
    student_model_name: str,
    teacher_model_name: str,
    model_short: str,
    questions: List[Dict[str, Any]],
    checkpoint_interval: int = 10,
    use_iterative_scaffolding: bool = True,
    max_iterations: int = 5,
    design_result: Optional[DesignResult] = None,
) -> IDMASState:
    """
    Create initial state for the pipeline.

    Args:
        domain: Domain name (e.g., "math")
        train_dataset: Training dataset name (e.g., "gsm8k")
        terminal_goal: Learning objective
        student_model_name: Student model name
        teacher_model_name: Teacher model name
        model_short: Short model name for file naming
        questions: List of questions to process
        checkpoint_interval: Save checkpoint every N questions
        use_iterative_scaffolding: Use iterative scaffolding in Phase 1
        max_iterations: Max iterations for iterative scaffolding
        design_result: Pre-loaded design result (optional)

    Returns:
        Initial IDMASState
    """
    return IDMASState(
        # Configuration
        domain=domain,
        train_dataset=train_dataset,
        terminal_goal=terminal_goal,
        student_model_name=student_model_name,
        teacher_model_name=teacher_model_name,
        model_short=model_short,
        checkpoint_interval=checkpoint_interval,
        use_iterative_scaffolding=use_iterative_scaffolding,
        max_iterations=max_iterations,

        # Design Phase
        design_result=design_result,
        task_analysis=design_result.get("instructional_analysis", {}).get("raw_output", "") if design_result else "",
        performance_objectives=design_result.get("performance_objectives", {}).get("performance_objectives", []) if design_result else [],
        rubric=design_result.get("rubrics") if design_result else None,

        # Questions
        questions=questions,
        total_questions=len(questions),
        current_question_index=0,
        current_question=questions[0] if questions else None,

        # Phase 1
        phase1_results=[],
        phase1_processed=0,
        phase1_correct_count=0,
        incorrect_after_phase1=[],
        phase1_first_attempt_correct=0,
        phase1_multi_attempt_correct=0,
        phase1_failed_reconstructed=0,

        # Phase 2
        coaching_db=None,
        weak_objectives=[],
        phase2_results=[],
        phase2_processed=0,
        still_incorrect_after_phase2=[],

        # Phase 3
        phase3_results=[],
        phase3_processed=0,

        # SFT Data
        sft_data=[],

        # Control
        current_phase="phase1",
        is_complete=False,
        error_message=None,

        # Timestamps
        started_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),

        # Checkpoint
        checkpoint_path=None,
        last_checkpoint_at=None,
    )


def get_statistics(state: IDMASState) -> Dict[str, Any]:
    """
    Get pipeline statistics from state.

    Args:
        state: Current pipeline state

    Returns:
        Statistics dictionary
    """
    return {
        "total_questions": state.get("total_questions", 0),
        "phase1_processed": state.get("phase1_processed", 0),
        "phase1_correct": state.get("phase1_correct_count", 0),
        "phase1_incorrect": state.get("phase1_processed", 0) - state.get("phase1_correct_count", 0),
        "phase2_processed": state.get("phase2_processed", 0),
        "phase2_fixed": len(state.get("phase2_results", [])),
        "phase3_processed": state.get("phase3_processed", 0),
        "phase3_modeling": len(state.get("phase3_results", [])),
        "sft_case_a": state.get("phase1_correct_count", 0),
        "sft_case_a_failed": state.get("phase1_failed_reconstructed", 0),
        "sft_case_b": len(state.get("phase2_results", [])),
        "sft_case_c": len(state.get("phase3_results", [])),
        "iterative_scaffolding": {
            "first_attempt_correct": state.get("phase1_first_attempt_correct", 0),
            "multi_attempt_correct": state.get("phase1_multi_attempt_correct", 0),
            "failed_reconstructed": state.get("phase1_failed_reconstructed", 0),
            "max_iterations": state.get("max_iterations", 5),
        },
    }


def load_checkpoint_from_logs(
    logs_path: Path,
) -> Tuple[Dict[str, Any], Set[str]]:
    """
    Load checkpoint state from existing logs file.

    Args:
        logs_path: Path to the logs JSON file

    Returns:
        Tuple of (checkpoint_data, processed_question_ids)
        - checkpoint_data: Dictionary with phase results and statistics
        - processed_question_ids: Set of question IDs already processed
    """
    if not logs_path.exists():
        return {}, set()

    try:
        with open(logs_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load logs from {logs_path}: {e}")
        return {}, set()

    processed_ids = set()
    checkpoint_data = {
        "phase1_results": [],
        "phase2_results": [],
        "phase3_results": [],
        "incorrect_after_phase1": [],
        "still_incorrect_after_phase2": [],
        "coaching_db": logs.get("coaching_db"),
        "phase1_processed": 0,
        "phase1_correct_count": 0,
        "phase1_first_attempt_correct": 0,
        "phase1_multi_attempt_correct": 0,
        "phase1_failed_reconstructed": 0,
        "phase2_processed": 0,
        "phase3_processed": 0,
    }

    # Process Phase 1 results
    for result in logs.get("phase1_results", []):
        qid = result.get("id")
        if qid:
            processed_ids.add(qid)
            checkpoint_data["phase1_results"].append(result)
            checkpoint_data["phase1_processed"] += 1

            if result.get("phase1_correct"):
                checkpoint_data["phase1_correct_count"] += 1
                # Track iterative scaffolding stats
                scaffolding = result.get("iterative_scaffolding", {})
                if scaffolding.get("iterations_needed", 1) == 1:
                    checkpoint_data["phase1_first_attempt_correct"] += 1
                else:
                    checkpoint_data["phase1_multi_attempt_correct"] += 1
            else:
                checkpoint_data["incorrect_after_phase1"].append(result)
                if result.get("sft_case") == SFTCase.A_FAILED.value:
                    checkpoint_data["phase1_failed_reconstructed"] += 1

    # Process Phase 2 results
    for result in logs.get("phase2_results", []):
        qid = result.get("id")
        if qid:
            processed_ids.add(qid)
            checkpoint_data["phase2_results"].append(result)
            checkpoint_data["phase2_processed"] += 1

    # Process Phase 3 results
    for result in logs.get("phase3_results", []):
        qid = result.get("id")
        if qid:
            processed_ids.add(qid)
            checkpoint_data["phase3_results"].append(result)
            checkpoint_data["phase3_processed"] += 1

    return checkpoint_data, processed_ids


def restore_state_from_checkpoint(
    initial_state: IDMASState,
    checkpoint_data: Dict[str, Any],
    processed_ids: Set[str],
) -> IDMASState:
    """
    Restore state from checkpoint data.

    Args:
        initial_state: Fresh initial state
        checkpoint_data: Checkpoint data from logs
        processed_ids: Set of already processed question IDs

    Returns:
        Restored IDMASState
    """
    if not checkpoint_data or not processed_ids:
        return initial_state

    # Filter out already processed questions
    remaining_questions = [
        q for q in initial_state.get("questions", [])
        if q.get("id") not in processed_ids
    ]

    # Merge checkpoint data into initial state
    restored = dict(initial_state)

    # Restore phase results
    restored["phase1_results"] = checkpoint_data.get("phase1_results", [])
    restored["phase2_results"] = checkpoint_data.get("phase2_results", [])
    restored["phase3_results"] = checkpoint_data.get("phase3_results", [])
    restored["incorrect_after_phase1"] = checkpoint_data.get("incorrect_after_phase1", [])
    restored["still_incorrect_after_phase2"] = checkpoint_data.get("still_incorrect_after_phase2", [])
    restored["coaching_db"] = checkpoint_data.get("coaching_db")

    # Restore counters
    restored["phase1_processed"] = checkpoint_data.get("phase1_processed", 0)
    restored["phase1_correct_count"] = checkpoint_data.get("phase1_correct_count", 0)
    restored["phase1_first_attempt_correct"] = checkpoint_data.get("phase1_first_attempt_correct", 0)
    restored["phase1_multi_attempt_correct"] = checkpoint_data.get("phase1_multi_attempt_correct", 0)
    restored["phase1_failed_reconstructed"] = checkpoint_data.get("phase1_failed_reconstructed", 0)
    restored["phase2_processed"] = checkpoint_data.get("phase2_processed", 0)
    restored["phase3_processed"] = checkpoint_data.get("phase3_processed", 0)

    # Update questions to remaining ones
    restored["questions"] = remaining_questions
    restored["total_questions"] = len(initial_state.get("questions", []))  # Keep original total
    restored["current_question_index"] = 0
    restored["current_question"] = remaining_questions[0] if remaining_questions else None

    # Determine current phase
    if remaining_questions:
        restored["current_phase"] = "phase1"
    elif restored["incorrect_after_phase1"]:
        restored["current_phase"] = "phase2_prep"
    elif restored["still_incorrect_after_phase2"]:
        restored["current_phase"] = "phase3"
    else:
        restored["current_phase"] = "finalize"

    restored["updated_at"] = datetime.now().isoformat()

    return IDMASState(**restored)
