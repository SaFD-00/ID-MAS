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
    instructional_goal = state.get("instructional_goal", "")

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
            instructional_goal=instructional_goal,
        )
    else:
        result = _process_single_shot(
            question=question,
            task_analysis=task_analysis,
            student_model=student_model,
            answer_extractor=answer_extractor,
            instructional_goal=instructional_goal,
        )

    # NEW: Handle skipped questions (fallback occurred)
    if result.get("is_skipped", False):
        updates = {
            "scaffolding_results": [result],
            "scaffolding_processed": state.get("scaffolding_processed", 0) + 1,
            "skipped_count": state.get("skipped_count", 0) + 1,
            "updated_at": datetime.now().isoformat(),
        }
        failure_reason = result.get("failure_reason", "unknown")
        failure_stage = result.get("failure_stage", "unknown")
        print(f"  -> SKIPPED: Fallback detected at {failure_stage} ({failure_reason})")

        # Still track fallback counts for statistics
        failure = result.get("failure", {})
        if failure.get("evaluation", {}).get("is_fallback"):
            updates["evaluation_fallback_count"] = state.get("evaluation_fallback_count", 0) + 1
        if failure.get("scaffolding_artifact", {}).get("is_fallback"):
            updates["scaffolding_artifact_fallback_count"] = state.get("scaffolding_artifact_fallback_count", 0) + 1

        return updates

    # Build state updates
    updates = {
        "scaffolding_results": [result],
        "scaffolding_processed": state.get("scaffolding_processed", 0) + 1,
        "updated_at": datetime.now().isoformat(),
    }

    sft_case = result.get("sft_case")
    if sft_case == SFTCase.A.value:
        updates["case_a_count"] = state.get("case_a_count", 0) + 1
        updates["scaffolding_correct_count"] = state.get("scaffolding_correct_count", 0) + 1
        print(f"  -> Case A: Success on first attempt! (PO satisfied at iteration 1)")
    elif sft_case == SFTCase.B.value:
        updates["case_b_count"] = state.get("case_b_count", 0) + 1
        updates["scaffolding_correct_count"] = state.get("scaffolding_correct_count", 0) + 1
        iterations = result.get("iterative_scaffolding", {}).get("iterations_needed", 0)
        print(f"  -> Case B: Iterative Scaffolding succeeded! (PO satisfied at iteration {iterations})")

        # Check if Case B reconstruction used fallback
        failure = result.get("failure", {})
        if failure.get("reconstruction", {}).get("is_fallback"):
            updates["case_b_fallback_count"] = state.get("case_b_fallback_count", 0) + 1
            print(f"     [Warning] Case B reconstruction failed - using fallback response")
    elif sft_case == SFTCase.C.value:
        updates["case_c_count"] = state.get("case_c_count", 0) + 1
        print(f"  -> Case C: Reconstruction after 5 failed attempts")

        # Check if Case C reconstruction used fallback
        failure = result.get("failure", {})
        if failure.get("reconstruction", {}).get("is_fallback"):
            updates["case_c_fallback_count"] = state.get("case_c_fallback_count", 0) + 1
            print(f"     [Warning] Case C reconstruction failed - using fallback response")

    # Check for other failures (all cases)
    failure = result.get("failure", {})
    if failure.get("evaluation", {}).get("is_fallback"):
        updates["evaluation_fallback_count"] = state.get("evaluation_fallback_count", 0) + 1
        print(f"     [Warning] PO evaluation failed - using fallback")
    if failure.get("hint_generation"):  # list이므로 존재 여부만 확인
        updates["hint_fallback_count"] = state.get("hint_fallback_count", 0) + 1
        print(f"     [Warning] Hint generation failed - using fallback")
    if failure.get("summarization", {}).get("is_fallback"):
        updates["summarization_fallback_count"] = state.get("summarization_fallback_count", 0) + 1
        print(f"     [Warning] Conversation summarization failed - using fallback")
    if failure.get("scaffolding_artifact", {}).get("is_fallback"):
        updates["scaffolding_artifact_fallback_count"] = state.get("scaffolding_artifact_fallback_count", 0) + 1
        print(f"     [Warning] Scaffolding artifact generation failed - using fallback")
    if failure.get("reconstruction", {}).get("is_fallback") and sft_case == SFTCase.C.value:
        updates["final_solution_fallback_count"] = state.get("final_solution_fallback_count", 0) + 1
        print(f"     [Warning] Final solution generation failed - using fallback")

    # Update HOT/LOT scaffolding counts
    hot_count = result.get("hot_count", 0) or 0
    lot_count = result.get("lot_count", 0) or 0
    if hot_count > 0:
        updates["hot_scaffolding_count"] = state.get("hot_scaffolding_count", 0) + hot_count
    if lot_count > 0:
        updates["lot_scaffolding_count"] = state.get("lot_scaffolding_count", 0) + lot_count

    return updates


def _process_single_shot(
    question: Dict[str, Any],
    task_analysis: str,
    student_model,
    answer_extractor,
    instructional_goal: str = "",
) -> QuestionResult:
    """Process question with single-shot scaffolding."""
    response = student_model.generate_initial_response_with_scaffolding(
        problem_text=question["problem_text"],
        task_analysis=task_analysis,
        instructional_goal=instructional_goal,
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
    instructional_goal: str = "",
) -> QuestionResult:
    """
    Process question with Scaffolding Artifact-based iterative scaffolding.

    NEW Flow (Replaces Socratic Questioning):
    1. Student generates initial response with Terminal Goal emphasis
    2. Teacher evaluates with Performance Objectives
    3. If PO not satisfied → Teacher generates Scaffolding Artifact (HOT/LOT)
    4. Student responds using Scaffolding DB (must cite sources)
    5. Repeat until all POs satisfied or max iterations
    6. Case A: 1st iteration success
    7. Case B: 2-5th iteration success (reconstruct)
    8. Case C: Failed after max iterations → Teacher generates final solution
    """
    conversation_history = []
    iterations = []
    scaffolding_db = []  # NEW: Cumulative Scaffolding Artifacts
    is_correct = False
    all_satisfied = False
    predicted = None
    response = None
    last_correct_iteration = None
    last_correct_response = None
    failures = {}  # failure metadata collection
    hot_count = 0  # HOT scaffolding count
    lot_count = 0  # LOT scaffolding count

    last_response = None
    last_scaffolding_artifact = None

    for iteration in range(1, max_iterations + 1):
        # Step 1: Student generates response
        if iteration == 1:
            # Initial response with scaffolding system prompt
            response = student_model.generate_initial_response_with_scaffolding(
                problem_text=question["problem_text"],
                task_analysis=task_analysis,
                instructional_goal=instructional_goal,
            )
        else:
            # Step 4 (NEW): Student responds using Scaffolding Artifact DB
            response = student_model.respond_with_scaffolding_artifact(
                problem_text=question["problem_text"],
                previous_response=last_response,
                scaffolding_artifact=last_scaffolding_artifact,
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

        # Step 2: Teacher evaluates with Performance Objectives
        if performance_objectives:
            evaluation = teacher_model.evaluate_with_performance_objectives(
                student_response=response,
                performance_objectives=performance_objectives,
                problem_text=question["problem_text"],
                ground_truth=question["output"],
            )
            overall = evaluation.get("overall_assessment", {})
            if isinstance(overall, dict):
                all_satisfied = overall.get("all_satisfied", False)
            else:
                print(f"  Warning: overall_assessment is not dict: {type(overall)}")
                all_satisfied = False

            # Collect evaluation failure metadata
            if evaluation and evaluation.get("_failure_metadata"):
                eval_failure = evaluation["_failure_metadata"].get("evaluation")
                if eval_failure:
                    failures["evaluation"] = eval_failure

                    # NEW: Skip on evaluation fallback
                    if eval_failure.get("is_fallback"):
                        print(f"    [SKIP] Evaluation fallback detected. Skipping question {question['id']}")
                        return QuestionResult(
                            id=question["id"],
                            instruction=question.get("instruction", ""),
                            input=question["input"],
                            output=question["output"],
                            _problem_text=question["problem_text"],
                            initial_response=response,
                            predicted_answer=predicted,
                            scaffolding_correct=False,
                            sft_case=None,
                            is_skipped=True,
                            failure_reason="evaluation_fallback",
                            failure_stage="evaluation",
                            failure_details=eval_failure,
                            iterative_scaffolding={
                                "success": False,
                                "iterations_needed": iteration,
                                "conversation_history": conversation_history,
                            },
                            failure=failures,
                        )
        else:
            evaluation = None
            all_satisfied = is_correct

        # Track last correct response
        if is_correct:
            last_correct_iteration = iteration
            last_correct_response = response

        # Success condition: all POs satisfied
        if all_satisfied:
            print(f"    -> Success on iteration {iteration}! (PO satisfied: True)")

            # Add teacher entry without scaffolding artifact (success)
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
            break

        # PO not satisfied - continue scaffolding
        print(f"    -> PO not satisfied on iteration {iteration}. Generating scaffolding artifact...")

        # Step 3 (NEW): Generate Scaffolding Artifact (HOT/LOT)
        scaffolding_artifact = teacher_model.generate_scaffolding_artifact(
            problem_text=question["problem_text"],
            student_response=response,
            po_evaluation=evaluation,
            iteration_number=iteration,
            task_analysis=task_analysis,
            max_iterations=max_iterations,
        )

        # Count HOT/LOT scaffolding
        for artifact in scaffolding_artifact.get("scaffolding_artifacts", []):
            skill_type = artifact.get("skill_type", "")
            if skill_type == "HOT":
                hot_count += 1
            elif skill_type == "LOT":
                lot_count += 1

        # Collect scaffolding artifact failure metadata
        if scaffolding_artifact.get("_failure_metadata"):
            artifact_failure = scaffolding_artifact["_failure_metadata"].get("scaffolding_artifact")
            if artifact_failure and artifact_failure.get("is_fallback"):
                failures["scaffolding_artifact"] = artifact_failure

                # NEW: Skip on scaffolding artifact fallback
                print(f"    [SKIP] Scaffolding artifact fallback detected. Skipping question {question['id']}")
                return QuestionResult(
                    id=question["id"],
                    instruction=question.get("instruction", ""),
                    input=question["input"],
                    output=question["output"],
                    _problem_text=question["problem_text"],
                    initial_response=response,
                    predicted_answer=predicted,
                    scaffolding_correct=False,
                    sft_case=None,
                    is_skipped=True,
                    failure_reason="scaffolding_artifact_fallback",
                    failure_stage="scaffolding_artifact",
                    failure_details=artifact_failure,
                    iterative_scaffolding={
                        "success": False,
                        "iterations_needed": iteration,
                        "conversation_history": conversation_history,
                        "iterations": iterations,
                    },
                    scaffolding_db=scaffolding_db if scaffolding_db else None,
                    hot_count=hot_count if hot_count > 0 else None,
                    lot_count=lot_count if lot_count > 0 else None,
                    failure=failures,
                )

        # Accumulate scaffolding DB
        scaffolding_db.append({
            "iteration": iteration,
            "artifacts": scaffolding_artifact.get("scaffolding_artifacts", []),
            "summary": scaffolding_artifact.get("scaffolding_summary", ""),
        })

        conversation_history.append({
            "role": "teacher",
            "evaluation": evaluation,
            "scaffolding_artifact": scaffolding_artifact,
            "iteration": iteration,
        })

        iterations.append({
            "iteration_number": iteration,
            "student_response": response,
            "predicted_answer": predicted,
            "is_correct": is_correct,
            "teacher_evaluation": evaluation,
            "scaffolding_artifact": scaffolding_artifact,
            "all_po_satisfied": all_satisfied,
            "timestamp": datetime.now().isoformat(),
        })

        # Store for next iteration
        last_response = response
        last_scaffolding_artifact = scaffolding_artifact

    # Build result
    if all_satisfied:
        iterations_needed = len(iterations)

        reconstruction = None
        if iterations_needed == 1:
            # Case A: 1st iteration success - use student response as-is
            sft_output = response
            sft_case = SFTCase.A.value
        else:
            # Case B: 2-5th iteration success - reconstruct scaffolding process
            reconstruction = teacher_model.reconstruct_successful_scaffolding(
                problem_text=question["problem_text"],
                ground_truth=question["output"],
                task_analysis=task_analysis,
                conversation_history=conversation_history,
                final_response=response,
                iterations_needed=iterations_needed,
            )
            sft_output = reconstruction.get("reconstructed_response", response)
            sft_case = SFTCase.B.value

        # Merge Case B reconstruction failure metadata
        if reconstruction:
            failure_metadata = reconstruction.get("_failure_metadata")
            if failure_metadata:
                failures.update(failure_metadata)

        # Extract DB references from student response (if available)
        db_references = student_model.extract_db_references(response) if iterations_needed > 1 else []

        return QuestionResult(
            id=question["id"],
            instruction=question.get("instruction", ""),
            input=question["input"],
            output=question["output"],
            _problem_text=question["problem_text"],
            initial_response=response,
            predicted_answer=predicted,
            scaffolding_correct=True,
            sft_case=sft_case,
            sft_response=sft_output,
            iterative_scaffolding={
                "success": True,
                "iterations_needed": iterations_needed,
                "conversation_history": conversation_history,
                "iterations": iterations,
            },
            scaffolding_db=scaffolding_db if scaffolding_db else None,
            db_references=db_references if db_references else None,
            hot_count=hot_count if hot_count > 0 else None,
            lot_count=lot_count if lot_count > 0 else None,
            failure=failures if failures else None,
        )
    else:
        # Case C: Failed after max iterations - Teacher generates final solution
        failure_reason = "po_not_satisfied"
        print(f"    -> Failed after {max_iterations} iterations. (Reason: {failure_reason}) Generating final solution...")

        # Extract student weaknesses from conversation history
        student_weaknesses = teacher_model.extract_student_weaknesses(conversation_history)

        # NEW: Use generate_final_solution instead of summarize_and_reconstruct
        final_solution = teacher_model.generate_final_solution(
            problem_text=question["problem_text"],
            ground_truth=question["output"],
            task_analysis=task_analysis,
            scaffolding_history=scaffolding_db,
            student_weaknesses=student_weaknesses,
            max_iterations=max_iterations,
        )

        reconstructed_response = final_solution.get("solution_explanation", "")

        # Collect final solution failure metadata
        failure_metadata = final_solution.get("_failure_metadata")
        if failure_metadata:
            failures.update(failure_metadata)

        return QuestionResult(
            id=question["id"],
            instruction=question.get("instruction", ""),
            input=question["input"],
            output=question["output"],
            _problem_text=question["problem_text"],
            initial_response=reconstructed_response,
            predicted_answer=predicted,
            scaffolding_correct=False,
            sft_case=SFTCase.C.value,
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
            scaffolding_db=scaffolding_db if scaffolding_db else None,
            reconstruction={
                "addressed_weaknesses": final_solution.get("addressed_weaknesses", []),
                "key_learning_points": final_solution.get("key_learning_points", []),
                "final_answer": final_solution.get("final_answer", ""),
            },
            hot_count=hot_count if hot_count > 0 else None,
            lot_count=lot_count if lot_count > 0 else None,
            failure=failures if failures else None,
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

    # Scaffolding results: Case A, Case B, and Case C
    for result in state.get("scaffolding_results", []):
        # NEW: Skip fallback cases (is_skipped=True)
        if result.get("is_skipped", False):
            continue

        if result.get("sft_case") in (SFTCase.A.value, SFTCase.B.value, SFTCase.C.value):
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
        if case in (SFTCase.A.value, SFTCase.B.value, SFTCase.C.value):
            output = result.get("initial_response", "")

    if not output:
        return None

    instruction = result.get("instruction", "")
    if not instruction:
        if case == SFTCase.A.value:
            instruction = "Solve the following problem."
        elif case == SFTCase.B.value:
            instruction = "Solve the following problem with teacher guidance."
        elif case == SFTCase.C.value:
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
        "timestamp": datetime.now().isoformat(),
    }

    results_path = output_dir / logs_filename
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results_path
