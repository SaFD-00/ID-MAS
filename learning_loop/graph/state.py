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


class PipelineStep:
    """
    ID-MAS Iterative Scaffolding Pipeline Step 상수.

    Pipeline Flow:
    - Step 1: Initial Response (초기 응답 생성) - 1회만 실행
    - Step 2: PO Evaluation (Performance Objectives 평가) - 반복
    - Step 3: Scaffolding (스캐폴딩 아티팩트 생성) - PO 미충족 시
    - Step 4: Re-response (학생 재응답) - iteration 2~5
    - Step 5: Reconstruction (재구성) - Case A/B/C
    - Step 6: SFT Generation (SFT 데이터 생성)
    """
    # Step identifiers
    STEP1 = "step1"  # Initial Response
    STEP2 = "step2"  # PO Evaluation
    STEP3 = "step3"  # Scaffolding Artifact
    STEP4 = "step4"  # Student Re-response
    STEP5 = "step5"  # Reconstruction
    STEP6 = "step6"  # SFT Generation

    # Step names for logging
    STEP1_NAME = "initial_response"
    STEP2_NAME = "evaluation"
    STEP3_NAME = "scaffolding"
    STEP4_NAME = "reresponse"
    STEP5_NAME = "reconstruction"
    STEP6_NAME = "sft_generation"

    # Full step keys (for result/log dictionaries)
    STEP1_KEY = "step1_initial_response"
    STEP2_KEY = "step2_evaluation"
    STEP3_KEY = "step3_scaffolding"
    STEP4_KEY = "step4_reresponse"
    STEP5_KEY = "step5_reconstruction"
    STEP6_KEY = "step6_sft_generation"

    # All steps for iteration
    ALL_STEPS = [STEP1, STEP2, STEP3, STEP4, STEP5, STEP6]

    # Skip key suffixes
    SKIP_SUFFIX = "_skip"

    @classmethod
    def get_skip_key(cls, step: str) -> str:
        """Get skip key for a step (e.g., 'step2' -> 'step2_skip')."""
        return f"{step}{cls.SKIP_SUFFIX}"

    @classmethod
    def get_step_name(cls, step: str) -> str:
        """Get human-readable name for a step."""
        names = {
            cls.STEP1: cls.STEP1_NAME,
            cls.STEP2: cls.STEP2_NAME,
            cls.STEP3: cls.STEP3_NAME,
            cls.STEP4: cls.STEP4_NAME,
            cls.STEP5: cls.STEP5_NAME,
            cls.STEP6: cls.STEP6_NAME,
        }
        return names.get(step, step)


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

    # Step-based skip tracking (NEW)
    # Each step's skip is tracked in step_skips dict
    # Format: {"step2": {...}, "step3": {...}, "step5": {...}}
    step_skips: Optional[Dict[str, Dict[str, Any]]]

    # Legacy skip field (for backward compatibility)
    skip: Optional[Dict[str, Dict[str, Any]]]  # 기존 형식 유지

    # NEW: Scaffolding Artifact fields
    scaffolding_db: Optional[List[Dict[str, Any]]]  # 누적된 Scaffolding Artifacts
    db_references: Optional[List[str]]  # Student가 참조한 DB 정보 목록
    hot_count: Optional[int]  # HOT (High-Order Thinking) scaffolding count
    lot_count: Optional[int]  # LOT (Low-Order Thinking) scaffolding count

    # NEW: Skip tracking (fallback 발생 시)
    is_skipped: bool  # fallback 발생으로 skip된 경우 True
    skip_reason: Optional[str]  # skip 사유 (e.g., "evaluation_fallback")
    skip_stage: Optional[str]  # skip 발생 단계
    skip_details: Optional[Dict[str, Any]]  # 상세 skip 메타데이터


class DesignResult(TypedDict, total=False):
    """Instructional design output."""
    domain: str
    train_dataset: str
    identifier: str
    instructional_goal: str
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
    instructional_goal: str
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

    # ==================== Step-based Skip Statistics (NEW) ====================
    # Step 1: Initial Response - skip으로만 처리 (skipped_count에 포함)
    step1_skip_count: int  # Step 1 skip 수

    # Step 2: PO Evaluation
    step2_skip_count: int  # Step 2 (evaluation) skip 수

    # Step 3: Scaffolding Artifact
    step3_skip_count: int  # Step 3 (scaffolding) skip 수

    # Step 4: Student Re-response - 학생 모델이라 fallback 없음
    step4_skip_count: int  # Step 4 skip 수 (예비)

    # Step 5: Reconstruction (Case A/B/C)
    step5_skip_count: int  # Step 5 전체 skip 수
    step5_case_b_skip_count: int  # Step 5 Case B skip 수
    step5_case_c_skip_count: int  # Step 5 Case C skip 수
    step5_summarization_skip_count: int  # Step 5 대화 요약 skip 수

    # ==================== Scaffolding Artifact Statistics ====================
    hot_scaffolding_count: int  # HOT (High-Order Thinking) 스캐폴딩 생성 횟수
    lot_scaffolding_count: int  # LOT (Low-Order Thinking) 스캐폴딩 생성 횟수

    # ==================== Skip Statistics ====================
    skipped_count: int  # fallback으로 skip된 문제 수

    # ==================== Legacy Fields (Backward Compatibility) ====================
    # 이전 버전 호환을 위해 유지, 새 코드에서는 step_* 필드 사용
    case_b_fallback_count: int  # -> step5_case_b_skip_count
    case_c_fallback_count: int  # -> step5_case_c_skip_count
    evaluation_fallback_count: int  # -> step2_skip_count
    hint_fallback_count: int  # deprecated, 삭제 예정
    summarization_fallback_count: int  # -> step5_summarization_skip_count
    scaffolding_artifact_fallback_count: int  # -> step3_skip_count
    final_solution_fallback_count: int  # -> step5_case_c_skip_count

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
    instructional_goal: str,
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
        instructional_goal: Learning objective
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
        instructional_goal=instructional_goal,
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
        # Step-based skip statistics (NEW)
        step1_skip_count=0,
        step2_skip_count=0,
        step3_skip_count=0,
        step4_skip_count=0,
        step5_skip_count=0,
        step5_case_b_skip_count=0,
        step5_case_c_skip_count=0,
        step5_summarization_skip_count=0,

        # Scaffolding Artifact statistics
        hot_scaffolding_count=0,
        lot_scaffolding_count=0,

        # Skip statistics
        skipped_count=0,

        # Legacy fields (backward compatibility)
        case_b_fallback_count=0,
        case_c_fallback_count=0,
        evaluation_fallback_count=0,
        hint_fallback_count=0,
        summarization_fallback_count=0,
        scaffolding_artifact_fallback_count=0,
        final_solution_fallback_count=0,

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
        Statistics dictionary with step-based skip tracking
    """
    case_a = state.get("case_a_count", 0)
    case_b = state.get("case_b_count", 0)
    case_c = state.get("case_c_count", 0)

    # Step-based skip counts (NEW)
    step1_skip = state.get("step1_skip_count", 0)
    step2_skip = state.get("step2_skip_count", 0)
    step3_skip = state.get("step3_skip_count", 0)
    step4_skip = state.get("step4_skip_count", 0)
    step5_skip = state.get("step5_skip_count", 0)
    step5_case_b_skip = state.get("step5_case_b_skip_count", 0)
    step5_case_c_skip = state.get("step5_case_c_skip_count", 0)
    step5_summarization_skip = state.get("step5_summarization_skip_count", 0)

    # Scaffolding Artifact statistics
    hot_count = state.get("hot_scaffolding_count", 0)
    lot_count = state.get("lot_scaffolding_count", 0)

    # Skip statistics
    skipped = state.get("skipped_count", 0)

    processed = state.get("scaffolding_processed", 0)
    total_skipped = step1_skip + step2_skip + step3_skip + step4_skip + step5_skip

    return {
        "total_questions": state.get("total_questions", 0),
        "scaffolding_processed": processed,
        "case_statistics": {
            "case_a": case_a,  # 1회차 성공
            "case_b": case_b,  # 2~5회차 성공
            "case_c": case_c,  # 5회 실패 후 재구성
            "success_total": case_a + case_b,
            "success_rate": (case_a + case_b) / processed if processed > 0 else 0,
        },
        "scaffolding_artifacts": {
            "hot_count": hot_count,  # High-Order Thinking scaffolding
            "lot_count": lot_count,  # Low-Order Thinking scaffolding
            "total": hot_count + lot_count,
        },
        # Step-based skip statistics
        "skip": {
            "step1_initial_response": {
                "count": step1_skip,
                "rate": step1_skip / processed if processed > 0 else 0,
            },
            "step2_evaluation": {
                "count": step2_skip,
                "rate": step2_skip / processed if processed > 0 else 0,
            },
            "step3_scaffolding": {
                "count": step3_skip,
                "rate": step3_skip / processed if processed > 0 else 0,
            },
            "step4_reresponse": {
                "count": step4_skip,
                "rate": step4_skip / processed if processed > 0 else 0,
            },
            "step5_reconstruction": {
                "count": step5_skip,
                "rate": step5_skip / processed if processed > 0 else 0,
                "case_b": step5_case_b_skip,
                "case_c": step5_case_c_skip,
                "summarization": step5_summarization_skip,
            },
            "analysis": {
                "count": skipped,
                "rate": skipped / processed if processed > 0 else 0,
            },
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
        # Step-based skip statistics (NEW)
        "step1_skip_count": 0,
        "step2_skip_count": 0,
        "step3_skip_count": 0,
        "step4_skip_count": 0,
        "step5_skip_count": 0,
        "step5_case_b_skip_count": 0,
        "step5_case_c_skip_count": 0,
        "step5_summarization_skip_count": 0,
        # Scaffolding Artifact statistics
        "hot_scaffolding_count": 0,
        "lot_scaffolding_count": 0,
        # Skip statistics
        "skipped_count": 0,
        # Legacy fields (backward compatibility)
        "case_b_fallback_count": 0,
        "case_c_fallback_count": 0,
        "evaluation_fallback_count": 0,
        "hint_fallback_count": 0,
        "summarization_fallback_count": 0,
        "scaffolding_artifact_fallback_count": 0,
        "final_solution_fallback_count": 0,
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
                # Check for Case B reconstruction fallback (NEW: skip, OLD: failure)
                skip_data = result.get("skip", {}) or result.get("failure", {})
                # Backward compatibility: support old reconstruction_failure field
                if not skip_data and result.get("reconstruction_failure"):
                    skip_data = {"reconstruction": result["reconstruction_failure"]}
                if skip_data.get("reconstruction", {}).get("is_fallback"):
                    checkpoint_data["case_b_fallback_count"] += 1
            elif sft_case == SFTCase.C.value:
                checkpoint_data["case_c_count"] += 1
                # Check for Case C reconstruction fallback (NEW: skip, OLD: failure)
                skip_data = result.get("skip", {}) or result.get("failure", {})
                # Backward compatibility: support old reconstruction_failure field
                if not skip_data and result.get("reconstruction_failure"):
                    skip_data = {"reconstruction": result["reconstruction_failure"]}
                if skip_data.get("reconstruction", {}).get("is_fallback"):
                    checkpoint_data["case_c_fallback_count"] += 1

            # Check for step-based skips (NEW format: step_skips, OLD: step_failures)
            step_skips = result.get("step_skips", {}) or result.get("step_failures", {})
            if step_skips:
                # New format: step_skips dict
                if step_skips.get("step1", {}).get("is_fallback"):
                    checkpoint_data["step1_skip_count"] += 1
                if step_skips.get("step2", {}).get("is_fallback"):
                    checkpoint_data["step2_skip_count"] += 1
                if step_skips.get("step3", {}).get("is_fallback"):
                    checkpoint_data["step3_skip_count"] += 1
                if step_skips.get("step4", {}).get("is_fallback"):
                    checkpoint_data["step4_skip_count"] += 1
                step5 = step_skips.get("step5", {})
                if step5.get("is_fallback"):
                    checkpoint_data["step5_skip_count"] += 1
                    if step5.get("case") == "B":
                        checkpoint_data["step5_case_b_skip_count"] += 1
                    elif step5.get("case") == "C":
                        checkpoint_data["step5_case_c_skip_count"] += 1
                if step_skips.get("step5_summarization", {}).get("is_fallback"):
                    checkpoint_data["step5_summarization_skip_count"] += 1
            else:
                # Legacy format: skip dict (OLD: failure dict)
                skip_data = result.get("skip", {}) or result.get("failure", {})
                # Backward compatibility
                if not skip_data and result.get("reconstruction_failure"):
                    skip_data = {"reconstruction": result["reconstruction_failure"]}

                if skip_data.get("evaluation", {}).get("is_fallback"):
                    checkpoint_data["step2_skip_count"] += 1
                    checkpoint_data["evaluation_fallback_count"] += 1
                if skip_data.get("hint_generation"):  # list이므로 존재 여부만 확인
                    checkpoint_data["hint_fallback_count"] += 1
                if skip_data.get("summarization", {}).get("is_fallback"):
                    checkpoint_data["step5_summarization_skip_count"] += 1
                    checkpoint_data["summarization_fallback_count"] += 1
                if skip_data.get("scaffolding_artifact", {}).get("is_fallback"):
                    checkpoint_data["step3_skip_count"] += 1
                    checkpoint_data["scaffolding_artifact_fallback_count"] += 1
                if skip_data.get("final_solution", {}).get("is_fallback"):
                    checkpoint_data["step5_case_c_skip_count"] += 1
                    checkpoint_data["step5_skip_count"] += 1
                    checkpoint_data["final_solution_fallback_count"] += 1

            # Count HOT/LOT scaffolding from scaffolding_db
            scaffolding_db = result.get("scaffolding_db") or []
            for db_entry in scaffolding_db:
                for artifact in db_entry.get("artifacts", []):
                    skill_type = artifact.get("skill_type", "")
                    if skill_type == "HOT":
                        checkpoint_data["hot_scaffolding_count"] += 1
                    elif skill_type == "LOT":
                        checkpoint_data["lot_scaffolding_count"] += 1

            # NEW: Count skipped questions
            if result.get("is_skipped", False):
                checkpoint_data["skipped_count"] += 1

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
    restored["case_b_fallback_count"] = checkpoint_data.get("case_b_fallback_count", 0)
    restored["case_c_fallback_count"] = checkpoint_data.get("case_c_fallback_count", 0)
    # Step-based skip statistics (NEW: skip_count, OLD: failure_count)
    restored["step1_skip_count"] = checkpoint_data.get("step1_skip_count", 0) or checkpoint_data.get("step1_failure_count", 0)
    restored["step2_skip_count"] = checkpoint_data.get("step2_skip_count", 0) or checkpoint_data.get("step2_failure_count", 0)
    restored["step3_skip_count"] = checkpoint_data.get("step3_skip_count", 0) or checkpoint_data.get("step3_failure_count", 0)
    restored["step4_skip_count"] = checkpoint_data.get("step4_skip_count", 0) or checkpoint_data.get("step4_failure_count", 0)
    restored["step5_skip_count"] = checkpoint_data.get("step5_skip_count", 0) or checkpoint_data.get("step5_failure_count", 0)
    restored["step5_case_b_skip_count"] = checkpoint_data.get("step5_case_b_skip_count", 0) or checkpoint_data.get("step5_case_b_failure_count", 0)
    restored["step5_case_c_skip_count"] = checkpoint_data.get("step5_case_c_skip_count", 0) or checkpoint_data.get("step5_case_c_failure_count", 0)
    restored["step5_summarization_skip_count"] = checkpoint_data.get("step5_summarization_skip_count", 0) or checkpoint_data.get("step5_summarization_failure_count", 0)

    # Scaffolding Artifact statistics
    restored["hot_scaffolding_count"] = checkpoint_data.get("hot_scaffolding_count", 0)
    restored["lot_scaffolding_count"] = checkpoint_data.get("lot_scaffolding_count", 0)

    # Skip statistics
    restored["skipped_count"] = checkpoint_data.get("skipped_count", 0)

    # Legacy fields (backward compatibility)
    restored["case_b_fallback_count"] = checkpoint_data.get("case_b_fallback_count", 0)
    restored["case_c_fallback_count"] = checkpoint_data.get("case_c_fallback_count", 0)
    restored["evaluation_fallback_count"] = checkpoint_data.get("evaluation_fallback_count", 0)
    restored["hint_fallback_count"] = checkpoint_data.get("hint_fallback_count", 0)
    restored["summarization_fallback_count"] = checkpoint_data.get("summarization_fallback_count", 0)
    restored["scaffolding_artifact_fallback_count"] = checkpoint_data.get("scaffolding_artifact_fallback_count", 0)
    restored["final_solution_fallback_count"] = checkpoint_data.get("final_solution_fallback_count", 0)

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
