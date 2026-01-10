"""
LangGraph StateGraph for ID-MAS 3-Phase Pipeline.

This module builds the complete LangGraph workflow:
- StateGraph construction with conditional routing
- IDMASGraphRunner for executing the pipeline
- Checkpoint support with SqliteSaver or MemorySaver

Based on the research proposal's 3-Phase Pipeline architecture.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from learning_loop.graph.state import (
    IDMASState,
    QuestionResult,
    DesignResult,
    create_initial_state,
    get_statistics,
    load_checkpoint_from_logs,
    restore_state_from_checkpoint,
)
from learning_loop.graph.nodes import (
    process_question_phase1,
    generate_coaching_db,
    process_question_phase2,
    process_question_phase3,
    advance_to_next_question,
    generate_sft_data,
    save_results,
    save_incremental_checkpoint,
)


def create_idmas_graph(
    student_model,
    teacher_model,
    answer_extractor,
) -> StateGraph:
    """
    Create the ID-MAS 3-Phase Pipeline as a LangGraph StateGraph.

    The graph implements the following flow:
    1. process_phase1 -> check if correct
       - If correct: advance to next question or finish Phase 1
       - If incorrect: add to Phase 2 queue
    2. After all Phase 1: generate_coaching_db if there are incorrect answers
    3. process_phase2 -> check if fixed
       - If fixed: mark as Case B
       - If still incorrect: add to Phase 3 queue
    4. process_phase3 -> mark as Case C (modeling)
    5. generate_sft_data -> complete

    Args:
        student_model: StudentModel instance
        teacher_model: TeacherModel instance
        answer_extractor: AnswerExtractor instance

    Returns:
        Compiled StateGraph
    """

    # ==================== Node Functions (wrapped) ====================

    def phase1_node(state: IDMASState) -> Dict[str, Any]:
        """Phase 1: Process current question with scaffolding."""
        return process_question_phase1(
            state=state,
            student_model=student_model,
            teacher_model=teacher_model,
            answer_extractor=answer_extractor,
        )

    def advance_node(state: IDMASState) -> Dict[str, Any]:
        """Advance to next question."""
        return advance_to_next_question(state)

    def coaching_db_node(state: IDMASState) -> Dict[str, Any]:
        """Generate Coaching DB for Phase 2."""
        return generate_coaching_db(
            state=state,
            teacher_model=teacher_model,
        )

    def phase2_batch_node(state: IDMASState) -> Dict[str, Any]:
        """Process all Phase 2 questions."""
        incorrect_results = list(state.get("incorrect_after_phase1", []))
        if not incorrect_results:
            return {"current_phase": "phase3_check"}

        updates = {
            "phase2_processed": 0,
            "phase2_results": [],
            "still_incorrect_after_phase2": [],
        }

        for result in incorrect_results:
            result_updates = process_question_phase2(
                state={**state, **updates},
                result=result,
                student_model=student_model,
                answer_extractor=answer_extractor,
            )
            # Merge updates
            updates["phase2_processed"] = result_updates.get(
                "phase2_processed", updates["phase2_processed"]
            )
            if "phase2_results" in result_updates:
                updates["phase2_results"] = updates.get("phase2_results", []) + result_updates["phase2_results"]
            if "still_incorrect_after_phase2" in result_updates:
                updates["still_incorrect_after_phase2"] = (
                    updates.get("still_incorrect_after_phase2", []) +
                    result_updates["still_incorrect_after_phase2"]
                )

        updates["current_phase"] = "phase3_check"
        updates["updated_at"] = datetime.now().isoformat()
        return updates

    def phase3_batch_node(state: IDMASState) -> Dict[str, Any]:
        """Process all Phase 3 questions."""
        still_incorrect = list(state.get("still_incorrect_after_phase2", []))
        if not still_incorrect:
            return {"current_phase": "finalize"}

        updates = {
            "phase3_processed": 0,
            "phase3_results": [],
        }

        for result in still_incorrect:
            result_updates = process_question_phase3(
                state={**state, **updates},
                result=result,
                teacher_model=teacher_model,
            )
            updates["phase3_processed"] = result_updates.get(
                "phase3_processed", updates["phase3_processed"]
            )
            if "phase3_results" in result_updates:
                updates["phase3_results"] = updates.get("phase3_results", []) + result_updates["phase3_results"]

        updates["current_phase"] = "finalize"
        updates["updated_at"] = datetime.now().isoformat()
        return updates

    def finalize_node(state: IDMASState) -> Dict[str, Any]:
        """Generate SFT data and mark complete."""
        return generate_sft_data(state)

    # ==================== Conditional Routing ====================

    def should_continue_phase1(state: IDMASState) -> Literal["phase1", "phase2_prep"]:
        """Check if there are more Phase 1 questions to process."""
        current_index = state.get("current_question_index", 0)
        total = state.get("total_questions", 0)

        # Check if we just processed a question (phase1_processed > current_index)
        processed = state.get("phase1_processed", 0)

        if processed < total:
            return "phase1"
        else:
            return "phase2_prep"

    def should_run_phase2(state: IDMASState) -> Literal["phase2", "phase3_check"]:
        """Check if Phase 2 is needed."""
        incorrect = state.get("incorrect_after_phase1", [])
        if incorrect:
            return "phase2"
        else:
            return "phase3_check"

    def should_run_phase3(state: IDMASState) -> Literal["phase3", "finalize"]:
        """Check if Phase 3 is needed."""
        still_incorrect = state.get("still_incorrect_after_phase2", [])
        if still_incorrect:
            return "phase3"
        else:
            return "finalize"

    # ==================== Build Graph ====================

    workflow = StateGraph(IDMASState)

    # Add nodes
    workflow.add_node("phase1", phase1_node)
    workflow.add_node("advance", advance_node)
    workflow.add_node("coaching_db", coaching_db_node)
    workflow.add_node("phase2", phase2_batch_node)
    workflow.add_node("phase3", phase3_batch_node)
    workflow.add_node("finalize", finalize_node)

    # Set entry point
    workflow.set_entry_point("phase1")

    # Add edges
    workflow.add_edge("phase1", "advance")
    workflow.add_conditional_edges(
        "advance",
        should_continue_phase1,
        {
            "phase1": "phase1",
            "phase2_prep": "coaching_db",
        }
    )
    workflow.add_conditional_edges(
        "coaching_db",
        should_run_phase2,
        {
            "phase2": "phase2",
            "phase3_check": "finalize",
        }
    )
    workflow.add_conditional_edges(
        "phase2",
        should_run_phase3,
        {
            "phase3": "phase3",
            "finalize": "finalize",
        }
    )
    workflow.add_edge("phase3", "finalize")
    workflow.add_edge("finalize", END)

    return workflow


class IDMASGraphRunner:
    """
    Runner class for the ID-MAS LangGraph pipeline.

    Handles:
    - Graph compilation with checkpointing
    - Pipeline execution
    - Result saving

    Example:
        runner = IDMASGraphRunner(
            student_model=student,
            teacher_model=teacher,
            answer_extractor=extractor,
        )
        result = runner.run(
            domain="math",
            train_dataset="gsm8k",
            questions=questions,
            design_result=design_result,
        )
    """

    def __init__(
        self,
        student_model,
        teacher_model,
        answer_extractor,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize the runner.

        Args:
            student_model: StudentModel instance
            teacher_model: TeacherModel instance
            answer_extractor: AnswerExtractor instance
            checkpoint_dir: Directory for checkpoints (uses memory if None)
        """
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.answer_extractor = answer_extractor
        self.checkpoint_dir = checkpoint_dir

        # Build graph
        self.workflow = create_idmas_graph(
            student_model=student_model,
            teacher_model=teacher_model,
            answer_extractor=answer_extractor,
        )

        # Setup checkpointer
        self.checkpointer = MemorySaver()

        # Compile graph
        self.graph = self.workflow.compile(checkpointer=self.checkpointer)

    def run(
        self,
        domain: str,
        train_dataset: str,
        terminal_goal: str,
        student_model_name: str,
        teacher_model_name: str,
        model_short: str,
        questions: List[Dict[str, Any]],
        design_result: Optional[DesignResult] = None,
        output_dir: Optional[Path] = None,
        checkpoint_interval: int = 10,
        use_iterative_scaffolding: bool = True,
        max_iterations: int = 5,
        thread_id: str = "default",
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Args:
            domain: Domain name (e.g., "math")
            train_dataset: Training dataset name (e.g., "gsm8k")
            terminal_goal: Learning objective
            student_model_name: Student model name
            teacher_model_name: Teacher model name
            model_short: Short model name for file naming
            questions: List of questions to process
            design_result: Pre-loaded design result
            output_dir: Output directory for saving results
            checkpoint_interval: Save checkpoint every N questions
            use_iterative_scaffolding: Use iterative scaffolding in Phase 1
            max_iterations: Max iterations for iterative scaffolding
            thread_id: Thread ID for checkpointing
            resume: Whether to resume from existing logs file

        Returns:
            Final state with results
        """
        print("\n" + "=" * 60)
        print("ID-MAS 3-PHASE PIPELINE (LangGraph)")
        print("=" * 60)
        print(f"Domain: {domain}")
        print(f"Dataset: {train_dataset}")
        print(f"Student Model: {student_model_name}")
        print(f"Teacher Model: {teacher_model_name}")
        print(f"Questions: {len(questions)}")
        print(f"Iterative Scaffolding: {use_iterative_scaffolding}")
        print(f"Resume Mode: {resume}")
        print("=" * 60)

        # Create initial state
        initial_state = create_initial_state(
            domain=domain,
            train_dataset=train_dataset,
            terminal_goal=terminal_goal,
            student_model_name=student_model_name,
            teacher_model_name=teacher_model_name,
            model_short=model_short,
            questions=questions,
            checkpoint_interval=checkpoint_interval,
            use_iterative_scaffolding=use_iterative_scaffolding,
            max_iterations=max_iterations,
            design_result=design_result,
        )

        # File-based resume: load checkpoint from logs file
        if resume and output_dir:
            logs_filename = f"{train_dataset}_train_id-mas_{model_short}_logs.json"
            logs_path = output_dir / logs_filename

            if logs_path.exists():
                print(f"\n[Resume] Loading checkpoint from: {logs_path}")
                checkpoint_data, processed_ids = load_checkpoint_from_logs(logs_path)

                if processed_ids:
                    print(f"[Resume] Found {len(processed_ids)} processed questions")
                    initial_state = restore_state_from_checkpoint(
                        initial_state, checkpoint_data, processed_ids
                    )
                    remaining = len(initial_state.get("questions", []))
                    print(f"[Resume] Remaining questions: {remaining}")
                    print(f"[Resume] Phase 1 correct: {initial_state.get('phase1_correct_count', 0)}")
                    print(f"[Resume] Phase 1 incorrect: {len(initial_state.get('incorrect_after_phase1', []))}")

                    # If all Phase 1 is done, check if we need to continue with Phase 2/3
                    if remaining == 0:
                        print("[Resume] All Phase 1 questions processed. Continuing with Phase 2/3...")
                else:
                    print("[Resume] No processed questions found. Starting fresh.")
            else:
                print(f"[Resume] No logs file found at {logs_path}. Starting fresh.")

        # Configuration for the run
        # recursion_limit: 문제 수 * 최대 iteration(5) * 안전 마진
        recursion_limit = max(100, len(questions) * 10)
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}

        # Prepare logs filename for incremental saving
        logs_filename = f"{train_dataset}_train_id-mas_{model_short}_logs.json"

        # Run the graph
        print("\n[Starting Pipeline Execution]")
        final_state = None
        accumulated_state = dict(initial_state)

        for event in self.graph.stream(initial_state, config):
            # event is a dict with node name as key
            for node_name, node_output in event.items():
                # Merge node output into accumulated state
                for key, value in node_output.items():
                    if key in accumulated_state and isinstance(accumulated_state[key], list) and isinstance(value, list):
                        # Extend lists (for phase1_results, etc.)
                        accumulated_state[key] = accumulated_state[key] + value
                    else:
                        accumulated_state[key] = value

                # Save incremental checkpoint after Phase 1 question processing
                if node_name == "advance" and output_dir:
                    save_incremental_checkpoint(
                        state=accumulated_state,
                        output_dir=output_dir,
                        logs_filename=logs_filename,
                    )

                # Save after Phase 2 and Phase 3 batch processing
                if node_name in ("phase2", "phase3", "coaching_db") and output_dir:
                    save_incremental_checkpoint(
                        state=accumulated_state,
                        output_dir=output_dir,
                        logs_filename=logs_filename,
                    )

                if node_name == "finalize":
                    final_state = accumulated_state
                    break

        # Get final state from checkpointer if not captured
        if final_state is None:
            checkpoint = self.checkpointer.get(config)
            if checkpoint:
                final_state = checkpoint.get("channel_values", accumulated_state)
            else:
                final_state = accumulated_state

        # Print summary
        stats = get_statistics(final_state)
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Total Questions: {stats['total_questions']}")
        print(f"Phase 1 Correct (Case A): {stats['phase1_correct']}")
        print(f"Phase 2 Fixed (Case B): {stats['phase2_fixed']}")
        print(f"Phase 3 Modeling (Case C): {stats['phase3_modeling']}")
        print(f"SFT Data Generated: {len(final_state.get('sft_data', []))}")

        # Save results if output_dir provided
        if output_dir:
            sft_filename = f"{train_dataset}_train_id-mas_{model_short}.json"
            logs_filename = f"{train_dataset}_train_id-mas_{model_short}_logs.json"

            results_path, sft_path = save_results(
                state=final_state,
                output_dir=output_dir,
                sft_filename=sft_filename,
                logs_filename=logs_filename,
            )
            print(f"\nResults saved to: {results_path}")
            print(f"SFT data saved to: {sft_path}")

            final_state["results_path"] = str(results_path)
            final_state["sft_path"] = str(sft_path)

        return final_state

    def resume(
        self,
        thread_id: str,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Resume pipeline from checkpoint.

        Args:
            thread_id: Thread ID to resume from
            output_dir: Output directory for saving results

        Returns:
            Final state with results
        """
        config = {"configurable": {"thread_id": thread_id}}

        # Get checkpoint state
        checkpoint = self.checkpointer.get(config)
        if not checkpoint:
            raise ValueError(f"No checkpoint found for thread_id: {thread_id}")

        state = checkpoint.get("channel_values", {})
        print(f"\n[Resuming from checkpoint]")
        print(f"Phase 1 processed: {state.get('phase1_processed', 0)}/{state.get('total_questions', 0)}")

        # recursion_limit 설정
        total_questions = state.get('total_questions', 100)
        config["recursion_limit"] = max(100, total_questions * 10)

        # Continue execution
        final_state = None
        for event in self.graph.stream(None, config):
            for node_name, node_output in event.items():
                if node_name == "finalize":
                    final_state = {**state, **node_output}
                    break

        if final_state is None:
            checkpoint = self.checkpointer.get(config)
            if checkpoint:
                final_state = checkpoint.get("channel_values", state)
            else:
                final_state = state

        # Save results if output_dir provided
        if output_dir and final_state.get("is_complete"):
            model_short = final_state.get("model_short", "unknown")
            train_dataset = final_state.get("train_dataset", "unknown")
            sft_filename = f"{train_dataset}_train_id-mas_{model_short}.json"
            logs_filename = f"{train_dataset}_train_id-mas_{model_short}_logs.json"

            results_path, sft_path = save_results(
                state=final_state,
                output_dir=output_dir,
                sft_filename=sft_filename,
                logs_filename=logs_filename,
            )
            print(f"\nResults saved to: {results_path}")
            print(f"SFT data saved to: {sft_path}")

        return final_state

    def get_statistics(self, state: IDMASState) -> Dict[str, Any]:
        """Get pipeline statistics from state."""
        return get_statistics(state)
