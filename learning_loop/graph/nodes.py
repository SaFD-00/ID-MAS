"""
LangGraph Node Functions for ID-MAS Iterative Scaffolding Pipeline.

This module implements the node functions used in the LangGraph pipeline:
- Scaffolding: process_question_scaffolding (iterative with teacher guidance)
- Utility: save_results, check_completion

Based on the research proposal's Iterative Scaffolding architecture.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from learning_loop.graph.state import (
    IDMASState,
    QuestionResult,
    SFTCase,
    get_statistics,
)


# ==================== Scaffolding ====================

def process_question_scaffolding(
    state: IDMASState,
    student_model,
    teacher_model,
    answer_extractor,
) -> Dict[str, Any]:
    """
    Process current question with iterative scaffolding.

    This node generates an initial response using the student model
    with task analysis scaffolding. If iterative scaffolding is enabled,
    the teacher provides progressive hints until correct or max iterations.

    Args:
        state: Current pipeline state
        student_model: StudentModel instance
        teacher_model: TeacherModel instance (for iterative hints)
        answer_extractor: AnswerExtractor instance

    Returns:
        State updates including scaffolding_results
    """
    question = state["current_question"]
    task_analysis = state.get("task_analysis", "")
    use_iterative = state.get("use_iterative_scaffolding", True)
    max_iterations = state.get("max_iterations", 5)
    performance_objectives = state.get("performance_objectives", [])
    terminal_goal = state.get("terminal_goal", "")

    qid = question["id"]
    print(f"\n[Scaffolding] Processing: {qid}")

    if use_iterative:
        result = _process_iterative_scaffolding(
            question=question,
            task_analysis=task_analysis,
            student_model=student_model,
            teacher_model=teacher_model,
            answer_extractor=answer_extractor,
            max_iterations=max_iterations,
            performance_objectives=performance_objectives,
            terminal_goal=terminal_goal,
        )
    else:
        result = _process_single_shot(
            question=question,
            task_analysis=task_analysis,
            student_model=student_model,
            answer_extractor=answer_extractor,
            terminal_goal=terminal_goal,
        )

    # Build state updates
    updates = {
        "scaffolding_results": [result],
        "scaffolding_processed": state.get("scaffolding_processed", 0) + 1,
        "updated_at": datetime.now().isoformat(),
    }

    if result.get("scaffolding_correct"):
        updates["scaffolding_correct_count"] = state.get("scaffolding_correct_count", 0) + 1
        print(f"  -> Correct! (Case A)")

        # Track iterative statistics
        if use_iterative:
            scaffolding_info = result.get("iterative_scaffolding", {})
            iterations_needed = scaffolding_info.get("iterations_needed", 1)
            if iterations_needed == 1:
                updates["first_attempt_correct"] = state.get("first_attempt_correct", 0) + 1
            else:
                updates["multi_attempt_correct"] = state.get("multi_attempt_correct", 0) + 1
    else:
        print(f"  -> Incorrect (predicted: {result.get('predicted_answer')})")

        if result.get("sft_case") == "B":
            updates["failed_reconstructed"] = state.get("failed_reconstructed", 0) + 1

    return updates


def _process_single_shot(
    question: Dict[str, Any],
    task_analysis: str,
    student_model,
    answer_extractor,
    terminal_goal: str = "",
) -> QuestionResult:
    """Process question with single-shot scaffolding."""
    response = student_model.generate_initial_response_with_scaffolding(
        problem_text=question["problem_text"],
        task_analysis=task_analysis,
        terminal_goal=terminal_goal,
    )

    predicted = answer_extractor.extract(response)
    is_correct = answer_extractor.compare(predicted, question["output"])

    return QuestionResult(
        id=question["id"],
        instruction=question.get("instruction", ""),
        input=question["input"],
        output=question["output"],
        _problem_text=question["problem_text"],  # 내부 처리용
        initial_response=response,
        predicted_answer=predicted,
        scaffolding_correct=is_correct,
        sft_case=SFTCase.A.value if is_correct else None,
        sft_response=response if is_correct else None,
    )


def _process_iterative_scaffolding(
    question: Dict[str, Any],
    task_analysis: str,
    student_model,
    teacher_model,
    answer_extractor,
    max_iterations: int = 5,
    performance_objectives: List[Dict] = None,
    terminal_goal: str = "",
) -> QuestionResult:
    """
    Process question with iterative scaffolding using ReAct-style PO evaluation.

    Flow:
    1. Student generates initial response with Terminal Goal emphasis
    2. Teacher evaluates with Performance Objectives + generates Socratic questions
    3. If correct answer AND all POs satisfied → success (Case A)
    4. If answer correct but PO not satisfied → continue scaffolding
    5. If not satisfied, student responds to Socratic questions
    6. Repeat until (correct AND all POs satisfied) or max iterations
    7. If failed after max iterations → reconstruction (Case B)
       - Failure can be: answer incorrect OR answer correct but PO not satisfied
    """
    conversation_history = []
    iterations = []
    is_correct = False
    all_satisfied = False
    predicted = None
    response = None
    last_correct_iteration = None
    last_correct_response = None

    for iteration in range(1, max_iterations + 1):
        # Iteration 1: Student generates initial response with scaffolding
        if iteration == 1:
            response = student_model.generate_initial_response_with_scaffolding(
                problem_text=question["problem_text"],
                task_analysis=task_analysis,
                terminal_goal=terminal_goal,
            )
        else:
            # Iterations 2+: Student responds to Socratic questions from previous evaluation
            response = student_model.respond_to_socratic_questions(
                problem_text=question["problem_text"],
                teacher_evaluation=last_evaluation,
                previous_response=last_response,
                task_analysis=task_analysis,
            )

        conversation_history.append({
            "role": "student",
            "response": response,
            "iteration": iteration,
        })

        # Check answer correctness
        predicted = answer_extractor.extract(response)
        is_correct = answer_extractor.compare(predicted, question["output"])

        # Teacher evaluates with Performance Objectives
        if performance_objectives:
            evaluation = teacher_model.evaluate_with_performance_objectives(
                student_response=response,
                performance_objectives=performance_objectives,
                problem_text=question["problem_text"],
                ground_truth=question["output"],
            )
            all_satisfied = evaluation.get("overall_assessment", {}).get("all_satisfied", False)
        else:
            evaluation = None
            all_satisfied = is_correct

        conversation_history.append({
            "role": "teacher",
            "evaluation": evaluation,
            "iteration": iteration,
        })

        iterations.append({
            "iteration_number": iteration,
            "student_response": response,
            "predicted_answer": predicted,
            "is_correct": is_correct,
            "teacher_evaluation": evaluation,
            "all_po_satisfied": all_satisfied,
            "timestamp": datetime.now().isoformat(),
        })

        # Track last correct response for potential reconstruction
        if is_correct:
            last_correct_iteration = iteration
            last_correct_response = response

        # Success condition: all POs satisfied (answer correctness is not considered)
        if all_satisfied:
            print(f"    -> Success on iteration {iteration}! (PO satisfied: True)")
            break
        else:
            # PO not satisfied - continue scaffolding
            print(f"    -> PO not satisfied on iteration {iteration}. Continuing scaffolding...")

        # Store for next iteration
        last_response = response
        last_evaluation = evaluation

    # Build result - success requires all POs satisfied (answer correctness is not considered)
    if all_satisfied:
        sft_output = _build_sft_response_from_iterations(iterations, is_success=True)
        return QuestionResult(
            id=question["id"],
            instruction=question.get("instruction", ""),
            input=question["input"],
            output=question["output"],
            _problem_text=question["problem_text"],  # 내부 처리용
            initial_response=response,
            predicted_answer=predicted,
            scaffolding_correct=True,
            sft_case=SFTCase.A.value,
            sft_response=sft_output,
            iterative_scaffolding={
                "success": True,
                "iterations_needed": len(iterations),
                "conversation_history": conversation_history,
                "iterations": iterations,
            },
        )
    else:
        # Failed after max iterations - need reconstruction (PO not satisfied)
        failure_reason = "po_not_satisfied"
        print(f"    -> Failed after {max_iterations} iterations. (Reason: {failure_reason}) Reconstructing...")

        reconstruction = teacher_model.summarize_and_reconstruct(
            problem_text=question["problem_text"],
            ground_truth=question["output"],
            task_analysis=task_analysis,
            conversation_history=conversation_history,
        )

        reconstructed_response = reconstruction.get("reconstructed_response", "")

        return QuestionResult(
            id=question["id"],
            instruction=question.get("instruction", ""),
            input=question["input"],
            output=question["output"],
            _problem_text=question["problem_text"],  # 내부 처리용
            initial_response=reconstructed_response,
            predicted_answer=predicted,
            scaffolding_correct=False,
            sft_case=SFTCase.B.value,
            sft_response=reconstructed_response,
            iterative_scaffolding={
                "success": False,
                "iterations_needed": max_iterations,
                "conversation_history": conversation_history,
                "iterations": iterations,
                "failure_reason": failure_reason,
                "last_correct_iteration": last_correct_iteration,
                "last_correct_response": last_correct_response,
            },
            reconstruction={
                "summary": reconstruction.get("summary", ""),
                "student_weaknesses": reconstruction.get("student_weaknesses", []),
                "learning_points": reconstruction.get("learning_points", []),
            },
        )


def _build_sft_response_from_iterations(
    iterations: List[Dict],
    is_success: bool = True,
) -> str:
    """Build SFT response from iteration history.

    Note: First iteration may not have teacher guidance (if correct on first try).
    Subsequent iterations have teacher_evaluation with Socratic questions.
    """
    parts = []
    for i, it in enumerate(iterations):
        # First iteration: no prior guidance, just student response
        # Subsequent iterations: include teacher guidance from previous evaluation
        if i > 0:
            # Extract guidance from teacher evaluation if available
            teacher_eval = it.get("teacher_evaluation")
            if teacher_eval and isinstance(teacher_eval, dict):
                socratic = teacher_eval.get("socratic_questions", [])
                if socratic:
                    guidance = "\n".join(f"- {q}" for q in socratic)
                    parts.append(f"[Guidance]\n{guidance}")

        parts.append(f"[Solution Attempt]\n{it['student_response']}")
        if it.get("is_correct"):
            break
    return "\n\n".join(parts)


# ==================== Utility Nodes ====================

def advance_to_next_question(state: IDMASState) -> Dict[str, Any]:
    """
    Advance to the next question in the queue.

    Args:
        state: Current pipeline state

    Returns:
        State updates with next question
    """
    current_index = state.get("current_question_index", 0)
    questions = state.get("questions", [])
    next_index = current_index + 1

    if next_index < len(questions):
        return {
            "current_question_index": next_index,
            "current_question": questions[next_index],
        }
    else:
        return {
            "current_question": None,
        }


def generate_sft_data(state: IDMASState) -> Dict[str, Any]:
    """
    Generate SFT training data from scaffolding results.

    Args:
        state: Current pipeline state

    Returns:
        State updates with sft_data
    """
    sft_data = []

    # Scaffolding results: Case A (correct) and Case B (reconstructed)
    for result in state.get("scaffolding_results", []):
        if result.get("sft_case") in (SFTCase.A.value, SFTCase.B.value):
            entry = _create_sft_entry(result)
            if entry:
                sft_data.append(entry)

    return {
        "sft_data": sft_data,
        "is_complete": True,
        "current_phase": "complete",
        "updated_at": datetime.now().isoformat(),
    }


def _create_sft_entry(result: QuestionResult) -> Optional[Dict[str, Any]]:
    """Create single SFT entry based on case."""
    case = result.get("sft_case")
    output = result.get("sft_response", "")

    if not output:
        if case in (SFTCase.A.value, SFTCase.B.value):
            output = result.get("initial_response", "")

    if not output:
        return None

    instruction = result.get("instruction", "")
    if not instruction:
        if case == SFTCase.A.value:
            instruction = "Solve the following problem with teacher guidance."
        elif case == SFTCase.B.value:
            instruction = "Solve the following problem, learning from common mistakes."
        else:
            instruction = "Solve the following problem step by step."

    question_text = result["input"]
    return {
        "instruction": instruction,
        "input": f"Question: {question_text}",
        "output": output,
        "metadata": {
            "id": result["id"],
            "sft_case": case,
            "ground_truth": result["output"],
        },
    }


def _filter_internal_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    """Remove internal fields (prefixed with _) from result for logging."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _reorder_result_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    """Reorder fields: id, instruction, input, output first, then rest."""
    ordered = {}
    # Priority fields first
    for key in ["id", "instruction", "input", "output"]:
        if key in result:
            ordered[key] = result[key]
    # Then rest
    for key, value in result.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _prepare_results_for_save(results: List[Any]) -> List[Dict[str, Any]]:
    """Filter internal fields and reorder for all results."""
    return [_reorder_result_fields(_filter_internal_fields(dict(r))) for r in results]


def save_results(
    state: IDMASState,
    output_dir: Path,
    sft_filename: str,
    logs_filename: str,
) -> Tuple[Path, Path]:
    """
    Save pipeline results and SFT data to files.

    Args:
        state: Current pipeline state
        output_dir: Output directory
        sft_filename: SFT data filename
        logs_filename: Logs filename

    Returns:
        (results_path, sft_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results (filter internal fields and reorder)
    results = {
        "scaffolding_results": _prepare_results_for_save(state.get("scaffolding_results", [])),
        "statistics": get_statistics(state),
        "timestamp": datetime.now().isoformat(),
    }

    results_path = output_dir / logs_filename
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save SFT data
    sft_path = output_dir / sft_filename
    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(state.get("sft_data", []), f, ensure_ascii=False, indent=2)

    return results_path, sft_path


def save_incremental_checkpoint(
    state: IDMASState,
    output_dir: Path,
    logs_filename: str,
) -> Path:
    """
    Save incremental checkpoint after each question processing.

    This function saves the current state to a logs file for file-based resume.
    It should be called after each question is processed.

    Args:
        state: Current pipeline state
        output_dir: Output directory
        logs_filename: Logs filename

    Returns:
        Path to the saved logs file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results with current progress (filter internal fields and reorder)
    results = {
        "scaffolding_results": _prepare_results_for_save(state.get("scaffolding_results", [])),
        "statistics": get_statistics(state),
        "is_complete": state.get("is_complete", False),
        "current_phase": state.get("current_phase", "scaffolding"),
        "timestamp": datetime.now().isoformat(),
    }

    results_path = output_dir / logs_filename
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results_path
