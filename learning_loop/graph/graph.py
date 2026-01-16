"""
LangGraph StateGraph for ID-MAS Iterative Scaffolding Pipeline.

This module builds the complete LangGraph workflow:
- StateGraph construction with conditional routing
- IDMASGraphRunner for executing the pipeline
- Checkpoint support with SqliteSaver or MemorySaver

Based on the research proposal's Iterative Scaffolding architecture.
"""
from __future__ import annotations

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
    process_question_scaffolding,
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
    Create the ID-MAS Iterative Scaffolding Pipeline as a LangGraph StateGraph.

    The graph implements the following flow:
    1. scaffolding -> process current question with iterative scaffolding
       - If correct (answer + PO satisfied): Case A
       - If failed after max iterations: Case B (reconstructed)
    2. advance -> check if more questions
       - If more questions: go back to scaffolding
       - If no more questions: go to finalize
    3. finalize -> generate SFT data and complete

    Args:
        student_model: StudentModel instance
        teacher_model: TeacherModel instance
        answer_extractor: AnswerExtractor instance

    Returns:
        Compiled StateGraph
    """

    # ==================== Node Functions (wrapped) ====================

    def scaffolding_node(state: IDMASState) -> Dict[str, Any]:
        """Process current question with iterative scaffolding."""
        return process_question_scaffolding(
            state=state,
            student_model=student_model,
            teacher_model=teacher_model,
            answer_extractor=answer_extractor,
        )

    def advance_node(state: IDMASState) -> Dict[str, Any]:
        """Advance to next question."""
        return advance_to_next_question(state)

    def finalize_node(state: IDMASState) -> Dict[str, Any]:
        """Generate SFT data and mark complete."""
        return generate_sft_data(state)

    # ==================== Conditional Routing ====================

    def should_continue_scaffolding(state: IDMASState) -> Literal["scaffolding", "finalize"]:
        """Check if there are more questions to process."""
        total = state.get("total_questions", 0)
        processed = state.get("scaffolding_processed", 0)

        if processed < total:
            return "scaffolding"
        else:
            return "finalize"

    # ==================== Build Graph ====================

    workflow = StateGraph(IDMASState)

    # Add nodes
    workflow.add_node("scaffolding", scaffolding_node)
    workflow.add_node("advance", advance_node)
    workflow.add_node("finalize", finalize_node)

    # Set entry point
    workflow.set_entry_point("scaffolding")

    # Add edges
    # scaffolding → advance → (conditional: more scaffolding or finalize) → END
    workflow.add_edge("scaffolding", "advance")
    workflow.add_conditional_edges(
        "advance",
        should_continue_scaffolding,
        {
            "scaffolding": "scaffolding",
            "finalize": "finalize",
        }
    )
    workflow.add_edge("finalize", END)

    return workflow


class IDMASGraphRunner:
    """
    Runner class for the ID-MAS Iterative Scaffolding Pipeline.

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
        instructional_goal: str,
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
            instructional_goal: Learning objective
            student_model_name: Student model name
            teacher_model_name: Teacher model name
            model_short: Short model name for file naming
            questions: List of questions to process
            design_result: Pre-loaded design result
            output_dir: Output directory for saving results
            checkpoint_interval: Save checkpoint every N questions
            use_iterative_scaffolding: Use iterative scaffolding
            max_iterations: Max iterations for iterative scaffolding
            thread_id: Thread ID for checkpointing
            resume: Whether to resume from existing logs file

        Returns:
            Final state with results
        """
        print("\n" + "=" * 60)
        print("ID-MAS ITERATIVE SCAFFOLDING PIPELINE (LangGraph)")
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
            instructional_goal=instructional_goal,
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
                    print(f"[Resume] Scaffolding correct: {initial_state.get('scaffolding_correct_count', 0)}")
                    print(f"[Resume] Failed reconstructed: {initial_state.get('failed_reconstructed', 0)}")

                    if remaining == 0:
                        print("[Resume] All questions processed. Finalizing...")
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

                # Save incremental checkpoint after question processing
                if node_name == "advance" and output_dir:
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
        print(f"Scaffolding Processed: {stats['scaffolding_processed']}")

        case_stats = stats.get('case_statistics', {})
        print(f"\n[Case Statistics]")
        print(f"  Case A (한번에 성공): {case_stats.get('case_a', 0)}")
        print(f"  Case B (Iterative Scaffolding 성공): {case_stats.get('case_b', 0)}")
        print(f"  Case C (5회 실패 후 재구성): {case_stats.get('case_c', 0)}")
        print(f"  ────────────────────────────")
        print(f"  Success Total (A+B): {case_stats.get('success_total', 0)}")
        print(f"  Success Rate: {case_stats.get('success_rate', 0) * 100:.1f}%")

        failures = stats.get('failures', {})
        print(f"\n[Failures]")

        reconstruction = failures.get('reconstruction', {})
        print(f"  Reconstruction:")
        print(f"    Case B: {reconstruction.get('case_b', 0)}")
        print(f"    Case C: {reconstruction.get('case_c', 0)}")
        print(f"    Subtotal: {reconstruction.get('total', 0)}")

        print(f"  Other:")
        print(f"    Evaluation: {failures.get('evaluation', 0)}")
        print(f"    Hint Generation: {failures.get('hint', 0)}")
        print(f"    Summarization: {failures.get('summarization', 0)}")

        print(f"  ────────────────────────────")
        print(f"  Total Failures: {failures.get('total_failures', 0)}")
        print(f"  Failure Rate: {failures.get('failure_rate', 0) * 100:.1f}%")

        print(f"\nSFT Data Generated: {len(final_state.get('sft_data', []))}")

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
        print(f"Scaffolding processed: {state.get('scaffolding_processed', 0)}/{state.get('total_questions', 0)}")

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
