"""LangGraph 노드 함수 모듈.

ID-MAS Iterative Scaffolding Pipeline의 노드 함수들을 구현합니다.

주요 함수:
    process_question_scaffolding: 문제 처리 (반복적 스캐폴딩)
    advance_to_next_question: 다음 문제로 이동
    generate_sft_data: SFT 데이터 생성
    save_results: 결과 저장
    save_incremental_checkpoint: 증분 체크포인트 저장

파이프라인 단계:
    Step 1: Initial Response (Student)
    Step 2: PO Evaluation (Teacher)
    Step 3: Scaffolding Artifact (Teacher)
    Step 4: Re-response (Student)
    Step 5: Reconstruction (Teacher)
    Step 6: SFT Generation

사용 예시:
    >>> from learning_loop.graph.nodes import process_question_scaffolding
    >>> updates = process_question_scaffolding(state, student, teacher, extractor)
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
from prompts.learning_prompts import SCAFFOLDING_SYSTEM_PROMPT


# ==================== Scaffolding ====================

def process_question_scaffolding(
    state: IDMASState,
    student_model,
    teacher_model,
    answer_extractor,
) -> Dict[str, Any]:
    """현재 문제를 반복적 스캐폴딩으로 처리합니다.

    학생 모델이 과제 분석 스캐폴딩과 함께 초기 응답을 생성합니다.
    반복적 스캐폴딩이 활성화된 경우, 교사가 정답을 맞추거나
    최대 반복 횟수에 도달할 때까지 점진적 힌트를 제공합니다.

    Args:
        state: 현재 파이프라인 상태
        student_model: StudentModel 인스턴스
        teacher_model: TeacherModel 인스턴스 (반복적 힌트용)
        answer_extractor: AnswerExtractor 인스턴스

    Returns:
        scaffolding_results를 포함한 상태 업데이트 딕셔너리
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
        skip_reason = result.get("skip_reason", "unknown")
        skip_stage = result.get("skip_stage", "unknown")
        print(f"  -> SKIPPED: Fallback detected at {skip_stage} ({skip_reason})")

        # Track skip counts from skip_details
        skip_details = result.get("skip_details", {})

        # Step 2 (Evaluation) skips
        if skip_details.get("step2_performance_objectives_evaluation", {}).get("is_fallback"):
            updates["step2_skip_count"] = state.get("step2_skip_count", 0) + 1
            updates["evaluation_fallback_count"] = state.get("evaluation_fallback_count", 0) + 1

        # Step 3 (Scaffolding) skips
        if skip_details.get("step3_scaffolding_artifact_generation", {}).get("is_fallback"):
            updates["step3_skip_count"] = state.get("step3_skip_count", 0) + 1
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

        # Check if Case B reconstruction used fallback (Step 5)
        skip_details = result.get("skip_details", {})
        step5_fallback = skip_details.get("step5_case_b_reconstruction", {}).get("is_fallback")
        if step5_fallback:
            updates["step5_skip_count"] = state.get("step5_skip_count", 0) + 1
            updates["step5_case_b_skip_count"] = state.get("step5_case_b_skip_count", 0) + 1
            updates["case_b_fallback_count"] = state.get("case_b_fallback_count", 0) + 1
            print(f"     [Warning] Step 5 (Case B reconstruction) skipped - using fallback response")
    elif sft_case == SFTCase.C.value:
        updates["case_c_count"] = state.get("case_c_count", 0) + 1
        print(f"  -> Case C: Reconstruction after 5 failed attempts")

        # Check if Case C reconstruction used fallback (Step 5)
        skip_details = result.get("skip_details", {})
        step5_fallback = skip_details.get("step5_case_c_final_solution", {}).get("is_fallback")
        if step5_fallback:
            updates["step5_skip_count"] = state.get("step5_skip_count", 0) + 1
            updates["step5_case_c_skip_count"] = state.get("step5_case_c_skip_count", 0) + 1
            updates["case_c_fallback_count"] = state.get("case_c_fallback_count", 0) + 1
            updates["final_solution_fallback_count"] = state.get("final_solution_fallback_count", 0) + 1
            print(f"     [Warning] Step 5 (Case C final solution) skipped - using fallback response")

    # Check for other skips (all cases) from skip_details
    skip_details = result.get("skip_details", {})

    # Step 2 (Evaluation) skips
    if skip_details.get("step2_performance_objectives_evaluation", {}).get("is_fallback"):
        updates["step2_skip_count"] = state.get("step2_skip_count", 0) + 1
        updates["evaluation_fallback_count"] = state.get("evaluation_fallback_count", 0) + 1
        print(f"     [Warning] Step 2 (PO evaluation) skipped - using fallback")

    # Step 5 summarization skips
    if skip_details.get("step5_summarization", {}).get("is_fallback"):
        updates["step5_summarization_skip_count"] = state.get("step5_summarization_skip_count", 0) + 1
        updates["summarization_fallback_count"] = state.get("summarization_fallback_count", 0) + 1
        print(f"     [Warning] Step 5 (conversation summarization) skipped - using fallback")

    # Step 3 (Scaffolding) skips
    if skip_details.get("step3_scaffolding_artifact_generation", {}).get("is_fallback"):
        updates["step3_skip_count"] = state.get("step3_skip_count", 0) + 1
        updates["scaffolding_artifact_fallback_count"] = state.get("scaffolding_artifact_fallback_count", 0) + 1
        print(f"     [Warning] Step 3 (scaffolding artifact) skipped - using fallback")

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
    """단일 시도 스캐폴딩으로 문제를 처리합니다.

    반복적 스캐폴딩 없이 한 번의 응답만 생성합니다.

    Args:
        question: 문제 정보 딕셔너리
        task_analysis: 과제 분석 결과
        student_model: StudentModel 인스턴스
        answer_extractor: AnswerExtractor 인스턴스
        instructional_goal: 학습 목표. 기본값: ""

    Returns:
        QuestionResult 객체
    """
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
    """Scaffolding Artifact 기반 반복적 스캐폴딩으로 문제를 처리합니다.

    처리 흐름 (Scaffolding Artifact 기반):
        1. 학생이 Instructional Goal 강조와 함께 초기 응답 생성
        2. 교사가 Performance Objectives로 평가
        3. PO 미충족 시 → 교사가 Scaffolding Artifact (HOT/LOT) 생성
        4. 학생이 Scaffolding Artifacts를 참조하여 응답 (출처 인용 필수)
        5. 모든 PO 충족 또는 최대 반복 도달까지 반복
        6. Case A: 1회차 성공
        7. Case B: 2-5회차 성공 (재구성)
        8. Case C: 최대 반복 후 실패 → 교사가 최종 솔루션 생성

    Args:
        question: 문제 정보 딕셔너리
        task_analysis: 과제 분석 결과
        student_model: StudentModel 인스턴스
        teacher_model: TeacherModel 인스턴스
        answer_extractor: AnswerExtractor 인스턴스
        max_iterations: 최대 반복 횟수. 기본값: 5
        performance_objectives: PO 리스트. 기본값: None
        instructional_goal: 학습 목표. 기본값: ""

    Returns:
        QuestionResult 객체
    """
    conversation_history = []
    iterations = []
    scaffolding_artifacts = []  # Cumulative Scaffolding Artifacts
    feedback_history = []  # Cumulative feedback history
    is_correct = False
    all_satisfied = False
    predicted = None
    response = None
    last_correct_iteration = None
    last_correct_response = None
    skip_details = {}  # 통합 skip 메타데이터 (키 형식: "step{N}_{stage}")
    hot_count = 0  # HOT scaffolding count
    lot_count = 0  # LOT scaffolding count

    last_feedback = None  # scaffolding_artifact에서 추출한 서술형 feedback
    previous_response = None  # 이전 iteration의 학생 응답 추적

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
            # Step 4: Student responds using teacher feedback (from scaffolding artifact)
            response = student_model.respond_to_feedback(
                problem_text=question["problem_text"],
                feedback=last_feedback,
                task_analysis=task_analysis,
                instructional_goal=instructional_goal,
                dataset_prompt=question.get("instruction", ""),
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
                iteration_number=iteration,
                previous_response=previous_response,
            )
            all_satisfied = evaluation.get("all_satisfied", False)

            # Collect evaluation skip metadata (Step 2)
            if evaluation and evaluation.get("_failure_metadata"):
                step2_meta = evaluation["_failure_metadata"].get("step2_performance_objectives_evaluation")
                if step2_meta:
                    skip_details["step2_performance_objectives_evaluation"] = step2_meta

                    # Skip on evaluation fallback
                    if step2_meta.get("is_fallback"):
                        print(f"    [SKIP] Step 2 (evaluation) fallback detected. Skipping question {question['id']}")
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
                            skip_reason="step2_evaluation_fallback",
                            skip_stage="step2_evaluation",
                            skip_details=skip_details if skip_details else None,
                            iterative_scaffolding={
                                "success": False,
                                "iterations_needed": iteration,
                                "conversation_history": conversation_history,
                            },
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

        # Collect scaffolding artifact skip metadata (Step 3)
        if scaffolding_artifact.get("_failure_metadata"):
            step3_meta = scaffolding_artifact["_failure_metadata"].get("step3_scaffolding_artifact_generation")
            if step3_meta:
                skip_details["step3_scaffolding_artifact_generation"] = step3_meta

                # Skip on scaffolding artifact fallback
                if step3_meta.get("is_fallback"):
                    print(f"    [SKIP] Step 3 (scaffolding artifact) fallback detected. Skipping question {question['id']}")
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
                        skip_reason="step3_scaffolding_fallback",
                        skip_stage="step3_scaffolding",
                        skip_details=skip_details if skip_details else None,
                        iterative_scaffolding={
                            "success": False,
                            "iterations_needed": iteration,
                            "conversation_history": conversation_history,
                            "iterations": iterations,
                        },
                        scaffolding_artifacts=scaffolding_artifacts if scaffolding_artifacts else None,
                        hot_count=hot_count if hot_count > 0 else None,
                        lot_count=lot_count if lot_count > 0 else None,
                    )

        # Accumulate scaffolding artifacts
        scaffolding_artifacts.append({
            "iteration": iteration,
            "artifacts": scaffolding_artifact.get("scaffolding_artifacts", []),
            "summary": scaffolding_artifact.get("scaffolding_summary", ""),
        })

        # Accumulate feedback history
        current_feedback = scaffolding_artifact.get("feedback", "")
        if current_feedback:
            feedback_history.append({
                "iteration": iteration,
                "feedback": current_feedback,
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

        # Store for next iteration: feedback from scaffolding artifact (not evaluation)
        last_feedback = scaffolding_artifact.get("feedback", "Review your previous response and try to address the unsatisfied objectives.")
        previous_response = response  # 다음 iteration을 위해 현재 응답 저장

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
                feedback_history=feedback_history,
            )

            # Check for Case B reconstruction fallback (Step 5)
            skip_metadata = reconstruction.get("_failure_metadata")
            if skip_metadata:
                step5_meta = skip_metadata.get("step5_case_b_reconstruction")
                if step5_meta:
                    skip_details["step5_case_b_reconstruction"] = step5_meta
                step5_sum = skip_metadata.get("step5_summarization")
                if step5_sum:
                    skip_details["step5_summarization"] = step5_sum

                # Skip on Case B reconstruction fallback
                if step5_meta and step5_meta.get("is_fallback"):
                    print(f"    [SKIP] Step 5 (Case B reconstruction) fallback detected. Skipping question {question['id']}")
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
                        skip_reason="step5_reconstruction_fallback",
                        skip_stage="step5_reconstruction",
                        skip_details=skip_details if skip_details else None,
                        iterative_scaffolding={
                            "success": True,
                            "iterations_needed": iterations_needed,
                            "conversation_history": conversation_history,
                            "iterations": iterations,
                        },
                        scaffolding_artifacts=scaffolding_artifacts if scaffolding_artifacts else None,
                        hot_count=hot_count if hot_count > 0 else None,
                        lot_count=lot_count if lot_count > 0 else None,
                    )

            sft_output = reconstruction.get("reconstructed_response", response)
            sft_case = SFTCase.B.value

        # Extract artifact references from student response (if available)
        artifact_references = student_model.extract_db_references(response) if iterations_needed > 1 else []

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
            scaffolding_artifacts=scaffolding_artifacts if scaffolding_artifacts else None,
            artifact_references=artifact_references if artifact_references else None,
            hot_count=hot_count if hot_count > 0 else None,
            lot_count=lot_count if lot_count > 0 else None,
            skip_details=skip_details if skip_details else None,
        )
    else:
        # Case C: Failed after max iterations - Teacher generates final solution
        case_c_reason = "po_not_satisfied"
        print(f"    -> Failed after {max_iterations} iterations. (Reason: {case_c_reason}) Generating final solution...")

        # Extract student weaknesses from conversation history
        student_weaknesses = teacher_model.extract_student_weaknesses(conversation_history)

        # Generate final solution with feedback history
        final_solution = teacher_model.generate_final_solution(
            problem_text=question["problem_text"],
            ground_truth=question["output"],
            task_analysis=task_analysis,
            scaffolding_history=scaffolding_artifacts,
            student_weaknesses=student_weaknesses,
            max_iterations=max_iterations,
            feedback_history=feedback_history,
        )

        reconstructed_response = final_solution.get("solution_explanation", "")

        # Collect final solution skip metadata (Step 5, Case C)
        skip_metadata = final_solution.get("_failure_metadata")
        if skip_metadata:
            step5_meta = skip_metadata.get("step5_case_c_final_solution")
            if step5_meta:
                skip_details["step5_case_c_final_solution"] = step5_meta

                # Skip on Case C reconstruction fallback
                if step5_meta.get("is_fallback"):
                    print(f"    [SKIP] Step 5 (Case C final solution) fallback detected. Skipping question {question['id']}")
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
                        skip_reason="step5_reconstruction_fallback",
                        skip_stage="step5_reconstruction",
                        skip_details=skip_details if skip_details else None,
                        iterative_scaffolding={
                            "success": False,
                            "iterations_needed": max_iterations,
                            "conversation_history": conversation_history,
                            "iterations": iterations,
                            "case_c_reason": case_c_reason,
                            "last_correct_iteration": last_correct_iteration,
                            "last_correct_response": last_correct_response,
                        },
                        scaffolding_artifacts=scaffolding_artifacts if scaffolding_artifacts else None,
                        hot_count=hot_count if hot_count > 0 else None,
                        lot_count=lot_count if lot_count > 0 else None,
                    )

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
                "case_c_reason": case_c_reason,
                "last_correct_iteration": last_correct_iteration,
                "last_correct_response": last_correct_response,
            },
            scaffolding_artifacts=scaffolding_artifacts if scaffolding_artifacts else None,
            hot_count=hot_count if hot_count > 0 else None,
            lot_count=lot_count if lot_count > 0 else None,
            skip_details=skip_details if skip_details else None,
        )


def _build_sft_response_from_iterations(
    iterations: List[Dict],
    is_success: bool = True,
) -> str:
    """반복 히스토리에서 SFT 응답을 구축합니다.

    첫 번째 반복은 교사 가이드가 없을 수 있습니다 (첫 시도에 성공한 경우).
    이후 반복에는 피드백 질문이 포함된 teacher_evaluation이 있습니다.

    Args:
        iterations: 반복 히스토리 리스트
        is_success: 성공 여부. 기본값: True

    Returns:
        구축된 SFT 응답 문자열
    """
    parts = []
    for i, it in enumerate(iterations):
        # First iteration: no prior guidance, just student response
        # Subsequent iterations: include teacher guidance from previous evaluation
        if i > 0:
            # Extract guidance from teacher evaluation if available
            teacher_eval = it.get("teacher_evaluation")
            if teacher_eval and isinstance(teacher_eval, dict):
                feedback_items = teacher_eval.get("feedback", [])
                if feedback_items:
                    guidance = "\n".join(f"- {q}" for q in feedback_items)
                    parts.append(f"[Guidance]\n{guidance}")

        parts.append(f"[Solution Attempt]\n{it['student_response']}")
        if it.get("is_correct"):
            break
    return "\n\n".join(parts)


# ==================== Utility Nodes ====================

def advance_to_next_question(state: IDMASState) -> Dict[str, Any]:
    """큐의 다음 문제로 이동합니다.

    Args:
        state: 현재 파이프라인 상태

    Returns:
        다음 문제 정보를 포함한 상태 업데이트 딕셔너리
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
    """스캐폴딩 결과에서 SFT 훈련 데이터를 생성합니다.

    Case A, B, C 결과를 SFT 데이터 형식으로 변환합니다.
    fallback으로 skip된 문제는 제외됩니다.

    Args:
        state: 현재 파이프라인 상태

    Returns:
        sft_data를 포함한 상태 업데이트 딕셔너리
    """
    sft_data = []

    instructional_goal = state.get("instructional_goal", "")
    task_analysis = state.get("task_analysis", "")

    # Scaffolding results: Case A, Case B, and Case C
    for result in state.get("scaffolding_results", []):
        # Skip fallback cases (is_skipped=True)
        if result.get("is_skipped", False):
            continue

        if result.get("sft_case") in (SFTCase.A.value, SFTCase.B.value, SFTCase.C.value):
            entry = _create_sft_entry(result, instructional_goal, task_analysis)
            if entry:
                sft_data.append(entry)

    return {
        "sft_data": sft_data,
        "is_complete": True,
        "current_phase": "complete",
        "updated_at": datetime.now().isoformat(),
    }


def _create_sft_entry(
    result: QuestionResult,
    instructional_goal: str = "",
    task_analysis: str = "",
) -> Optional[Dict[str, Any]]:
    """케이스에 따라 단일 SFT 엔트리를 생성합니다.

    instruction은 enhanced data에서 가져온 값을 우선 사용하고,
    없을 경우 SCAFFOLDING_SYSTEM_PROMPT 기반으로 생성합니다.

    Args:
        result: QuestionResult 객체
        instructional_goal: 학습 목표 (fallback instruction 생성용)
        task_analysis: 과제 분석 결과 (fallback instruction 생성용)

    Returns:
        SFT 엔트리 딕셔너리 또는 None (출력이 없는 경우)
    """
    case = result.get("sft_case")
    output = result.get("sft_response", "")

    if not output:
        if case in (SFTCase.A.value, SFTCase.B.value, SFTCase.C.value):
            output = result.get("initial_response", "")

    if not output:
        return None

    # Enhanced data에서 가져온 instruction 우선 사용
    instruction = result.get("instruction", "")
    if not instruction:
        # Fallback: SCAFFOLDING_SYSTEM_PROMPT 기반 instruction 생성
        if instructional_goal and task_analysis:
            instruction = SCAFFOLDING_SYSTEM_PROMPT.format(
                instructional_goal=instructional_goal,
                task_analysis=task_analysis
            )
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
    """로깅용으로 내부 필드(_로 시작)를 제거합니다.

    Args:
        result: 원본 결과 딕셔너리

    Returns:
        내부 필드가 제거된 딕셔너리
    """
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _reorder_result_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    """필드를 재정렬합니다: id, instruction, input, output 순서.

    Args:
        result: 원본 결과 딕셔너리

    Returns:
        재정렬된 딕셔너리
    """
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
    """모든 결과에 대해 내부 필드를 필터링하고 재정렬합니다.

    Args:
        results: 결과 리스트

    Returns:
        저장용으로 준비된 결과 리스트
    """
    return [_reorder_result_fields(_filter_internal_fields(dict(r))) for r in results]


def save_results(
    state: IDMASState,
    output_dir: Path,
    sft_filename: str,
    logs_filename: str,
) -> Tuple[Path, Path]:
    """파이프라인 결과와 SFT 데이터를 파일에 저장합니다.

    Args:
        state: 현재 파이프라인 상태
        output_dir: 출력 디렉토리
        sft_filename: SFT 데이터 파일명
        logs_filename: 로그 파일명

    Returns:
        튜플 (results_path, sft_path)
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
    """각 문제 처리 후 증분 체크포인트를 저장합니다.

    현재 상태를 로그 파일에 저장하여 파일 기반 재개를 지원합니다.
    각 문제 처리 후 호출되어야 합니다.

    Args:
        state: 현재 파이프라인 상태
        output_dir: 출력 디렉토리
        logs_filename: 로그 파일명

    Returns:
        저장된 로그 파일 경로
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
