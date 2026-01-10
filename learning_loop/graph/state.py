"""
LangGraph State Schema for ID-MAS Iterative Scaffolding Pipeline.

This module defines the state structure used throughout the pipeline:
- IDMASState: Main state schema for the graph
- QuestionResult: Result for each processed question
- DesignResult: Instructional design output

Based on the research proposal's Iterative Scaffolding architecture.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, Optional, List, Dict, Any, Annotated, Set, Tuple
from datetime import datetime
from enum import Enum
import operator


class SFTCase(str, Enum):
    """SFT data case classification based on scaffolding result."""
    A = "A"  # PO satisfied on first attempt (한번에 성공)
    B = "B"  # PO satisfied via iterative scaffolding (2-5회차 성공)
    C = "C"  # PO not satisfied after max iterations (재구성 필요)


class QuestionResultRequired(TypedDict):
    """Required fields for QuestionResult."""
    id: str                # question_id → id (필수)
    instruction: str       # 순서 변경: 두 번째 (필수)
    input: str             # question → input (필수)
    output: str            # ground_truth → output (필수)
    _problem_text: str     # 내부 처리용 (로그 저장 시 제거, 필수)


class QuestionResult(QuestionResultRequired, total=False):
    """Result for a single question processed through the pipeline."""
    # Scaffolding results
    initial_response: str
    predicted_answer: Optional[str]
    scaffolding_correct: bool  # renamed from phase1_correct

    # Final SFT classification
    sft_case: Optional[str]
    sft_response: Optional[str]

    # Iterative scaffolding details
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

    The state follows the Iterative Scaffolding architecture:
    1. Instructional Design Phase (optional, can load existing)
    2. Scaffolding - Iterative response generation with teacher guidance
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

    # ==================== Scaffolding Results ====================
    # Using Annotated with reducer for accumulating results
    scaffolding_results: Annotated[List[QuestionResult], add_to_list]
    scaffolding_processed: int
    scaffolding_correct_count: int

    # Iterative scaffolding statistics
    case_a_count: int  # 1회차 성공 (한번에 성공)
    case_b_count: int  # 2~5회차 성공 (Iterative Scaffolding 성공)
    case_c_count: int  # 5회 실패 후 재구성

    # ==================== SFT Data ====================
    sft_data: List[Dict[str, Any]]

    # ==================== Pipeline Control ====================
    current_phase: str  # "design", "scaffolding", "finalize", "complete"
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
        use_iterative_scaffolding: Use iterative scaffolding
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

        # Scaffolding
        scaffolding_results=[],
        scaffolding_processed=0,
        scaffolding_correct_count=0,
        case_a_count=0,
        case_b_count=0,
        case_c_count=0,

        # SFT Data
        sft_data=[],

        # Control
        current_phase="scaffolding",
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
    case_a = state.get("case_a_count", 0)
    case_b = state.get("case_b_count", 0)
    case_c = state.get("case_c_count", 0)

    return {
        "total_questions": state.get("total_questions", 0),
        "scaffolding_processed": state.get("scaffolding_processed", 0),
        "case_statistics": {
            "case_a": case_a,  # 한번에 성공
            "case_b": case_b,  # Iterative Scaffolding 성공
            "case_c": case_c,  # 5회 실패 후 재구성
            "success_total": case_a + case_b,
            "success_rate": (case_a + case_b) / state.get("scaffolding_processed", 1) if state.get("scaffolding_processed", 0) > 0 else 0,
        },
        "sft_data_breakdown": {
            "case_a": case_a,
            "case_b": case_b,
            "case_c": case_c,
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
        - checkpoint_data: Dictionary with scaffolding results and statistics
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
        "scaffolding_results": [],
        "scaffolding_processed": 0,
        "scaffolding_correct_count": 0,
        "case_a_count": 0,
        "case_b_count": 0,
        "case_c_count": 0,
    }

    # Process scaffolding results (supports both old and new field names)
    results_key = "scaffolding_results" if "scaffolding_results" in logs else "phase1_results"
    for result in logs.get(results_key, []):
        qid = result.get("id")
        if qid:
            processed_ids.add(qid)
            checkpoint_data["scaffolding_results"].append(result)
            checkpoint_data["scaffolding_processed"] += 1

            # Support both old and new field names
            is_correct = result.get("scaffolding_correct") or result.get("phase1_correct")
            sft_case = result.get("sft_case")

            # Legacy compatibility: old "B" case (재구성) → 새로운 "C" case로 매핑
            if sft_case == "B" and not is_correct:
                sft_case = "C"

            # Count by case
            if sft_case == SFTCase.A.value:
                checkpoint_data["case_a_count"] += 1
                checkpoint_data["scaffolding_correct_count"] += 1
            elif sft_case == SFTCase.B.value:
                checkpoint_data["case_b_count"] += 1
                checkpoint_data["scaffolding_correct_count"] += 1
            elif sft_case == SFTCase.C.value:
                checkpoint_data["case_c_count"] += 1

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

    # Restore scaffolding results
    restored["scaffolding_results"] = checkpoint_data.get("scaffolding_results", [])

    # Restore counters
    restored["scaffolding_processed"] = checkpoint_data.get("scaffolding_processed", 0)
    restored["scaffolding_correct_count"] = checkpoint_data.get("scaffolding_correct_count", 0)
    restored["case_a_count"] = checkpoint_data.get("case_a_count", 0)
    restored["case_b_count"] = checkpoint_data.get("case_b_count", 0)
    restored["case_c_count"] = checkpoint_data.get("case_c_count", 0)

    # Update questions to remaining ones
    restored["questions"] = remaining_questions
    restored["total_questions"] = len(initial_state.get("questions", []))  # Keep original total
    restored["current_question_index"] = 0
    restored["current_question"] = remaining_questions[0] if remaining_questions else None

    # Determine current phase
    if remaining_questions:
        restored["current_phase"] = "scaffolding"
    else:
        restored["current_phase"] = "finalize"

    restored["updated_at"] = datetime.now().isoformat()

    return IDMASState(**restored)
