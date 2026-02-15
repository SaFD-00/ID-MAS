"""LangGraph StateGraph 구축 모듈.

ID-MAS Iterative Scaffolding Pipeline의 완전한 LangGraph 워크플로우를 구축합니다.

주요 기능:
    - 조건부 라우팅을 포함한 StateGraph 구성
    - 파이프라인 실행을 위한 IDMASGraphRunner
    - MemorySaver를 통한 체크포인트 지원

주요 클래스:
    IDMASGraphRunner: 파이프라인 실행기

주요 함수:
    create_idmas_graph: StateGraph 생성

사용 예시:
    >>> from learning_loop.graph.graph import IDMASGraphRunner
    >>> runner = IDMASGraphRunner(student, teacher, extractor)
    >>> result = runner.run(domain="math", train_dataset="gsm8k", ...)
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
    """ID-MAS Iterative Scaffolding Pipeline을 LangGraph StateGraph로 생성합니다.

    그래프 흐름:
        1. scaffolding → 현재 문제를 반복적 스캐폴딩으로 처리
           - 정답 + PO 충족: Case A: Autonomous Mastery (1회차) / Case B: Scaffolded & Coached Mastery (2~5회차)
           - 최대 반복 후 실패: Case C: Teacher Modeling Distillation (교사 모델링 증류)
        2. advance → 더 많은 문제가 있는지 확인
           - 더 있으면: scaffolding으로 돌아감
           - 없으면: finalize로 이동
        3. finalize → SFT 데이터 생성 및 완료

    Args:
        student_model: StudentModel 인스턴스
        teacher_model: TeacherModel 인스턴스
        answer_extractor: AnswerExtractor 인스턴스

    Returns:
        컴파일된 StateGraph
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
    """ID-MAS Iterative Scaffolding Pipeline 실행기 클래스.

    주요 기능:
        - 체크포인팅을 포함한 그래프 컴파일
        - 파이프라인 실행
        - 결과 저장

    Attributes:
        student_model: StudentModel 인스턴스
        teacher_model: TeacherModel 인스턴스
        answer_extractor: AnswerExtractor 인스턴스
        workflow: StateGraph 워크플로우
        graph: 컴파일된 그래프
        checkpointer: 체크포인터 (MemorySaver)

    Example:
        >>> runner = IDMASGraphRunner(
        ...     student_model=student,
        ...     teacher_model=teacher,
        ...     answer_extractor=extractor,
        ... )
        >>> result = runner.run(
        ...     domain="math",
        ...     train_dataset="gsm8k",
        ...     questions=questions,
        ... )
    """

    def __init__(
        self,
        student_model,
        teacher_model,
        answer_extractor,
        checkpoint_dir: Optional[Path] = None,
    ):
        """IDMASGraphRunner를 초기화합니다.

        Args:
            student_model: StudentModel 인스턴스
            teacher_model: TeacherModel 인스턴스
            answer_extractor: AnswerExtractor 인스턴스
            checkpoint_dir: 체크포인트 디렉토리. None이면 메모리 사용.
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
        max_iterations: int = 3,
        thread_id: str = "default",
        resume: bool = True,
    ) -> Dict[str, Any]:
        """전체 파이프라인을 실행합니다.

        Args:
            domain: 도메인 이름 (예: "math")
            train_dataset: 훈련 데이터셋 이름 (예: "gsm8k")
            instructional_goal: 학습 목표
            student_model_name: 학생 모델 이름
            teacher_model_name: 교사 모델 이름
            model_short: 파일 이름용 짧은 모델명
            questions: 처리할 문제 리스트
            design_result: 사전 로드된 교수설계 결과. 기본값: None
            output_dir: 결과 저장 디렉토리. 기본값: None
            checkpoint_interval: 체크포인트 저장 간격. 기본값: 10
            use_iterative_scaffolding: Iterative Scaffolding 사용 여부. 기본값: True
            max_iterations: 최대 반복 횟수. 기본값: 3
            thread_id: 체크포인팅용 스레드 ID. 기본값: "default"
            resume: 기존 로그에서 재개 여부. 기본값: True

        Returns:
            결과가 포함된 최종 상태 딕셔너리
        """
        print("\n" + "=" * 60)
        print("ID-MAS ITERATIVE SCAFFOLDING PIPELINE")
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
                        # Early return - 이미 완료된 경우 그래프 실행 건너뛰기
                        final_state = dict(initial_state)
                        final_state["is_complete"] = True

                        # Print summary
                        stats = get_statistics(final_state)
                        print("\n" + "=" * 60)
                        print("PIPELINE COMPLETE (Already finished)")
                        print("=" * 60)
                        print(f"Total Questions: {stats['total_questions']}")
                        print(f"Scaffolding Processed: {stats['scaffolding_processed']}")

                        case_stats = stats.get('case_statistics', {})
                        print(f"\n[Case Statistics]")
                        print(f"  Case A: Independent Performance Mastery (독립적 수행 숙달): {case_stats.get('case_a_independent_performance_mastery', 0)}")
                        print(f"  Case B: Scaffolded & Coached Mastery (스캐폴딩 기반 숙달): {case_stats.get('case_b_scaffolded_coached_mastery', 0)}")
                        print(f"  Case C: Teacher Modeling Distillation (교사 모델링 증류): {case_stats.get('case_c_teacher_modeling_distillation', 0)}")
                        print(f"  ────────────────────────────")
                        print(f"  Success Total (Case A + Case B): {case_stats.get('success_total', 0)}")
                        print(f"  Success Rate: {case_stats.get('success_rate', 0) * 100:.1f}%")
                        print("=" * 60)

                        return final_state
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
        print(f"  Case A: Independent Performance Mastery (독립적 수행 숙달): {case_stats.get('case_a_independent_performance_mastery', 0)}")
        print(f"  Case B: Scaffolded & Coached Mastery (스캐폴딩 기반 숙달): {case_stats.get('case_b_scaffolded_coached_mastery', 0)}")
        print(f"  Case C: Teacher Modeling Distillation (교사 모델링 증류): {case_stats.get('case_c_teacher_modeling_distillation', 0)}")
        print(f"  ────────────────────────────")
        print(f"  Success Total (Case A + Case B): {case_stats.get('success_total', 0)}")
        print(f"  Success Rate: {case_stats.get('success_rate', 0) * 100:.1f}%")

        skip_stats = stats.get('skip', {})
        print(f"\n[Skip]")

        step5 = skip_stats.get('step5_reconstruction', {})
        print(f"  Reconstruction:")
        print(f"    Case B: Scaffolded & Coached Mastery: {step5.get('case_b_scaffolded_coached_mastery', 0)}")
        print(f"    Case C: Teacher Modeling Distillation: {step5.get('case_c_teacher_modeling_distillation', 0)}")
        print(f"    Subtotal: {step5.get('case_b_scaffolded_coached_mastery', 0) + step5.get('case_c_teacher_modeling_distillation', 0)}")

        print(f"  Other:")
        print(f"    Evaluation: {skip_stats.get('step2_evaluation', {}).get('count', 0)}")
        print(f"    Scaffolding: {skip_stats.get('step3_scaffolding', {}).get('count', 0)}")
        print(f"    Summarization: {step5.get('summarization', 0)}")

        analysis = skip_stats.get('analysis', {})
        print(f"  ────────────────────────────")
        print(f"  Total Skipped: {analysis.get('count', 0)}")
        print(f"  Skip Rate: {analysis.get('rate', 0) * 100:.1f}%")

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
        """체크포인트에서 파이프라인을 재개합니다.

        Args:
            thread_id: 재개할 스레드 ID
            output_dir: 결과 저장 디렉토리. 기본값: None

        Returns:
            결과가 포함된 최종 상태 딕셔너리

        Raises:
            ValueError: 해당 thread_id에 체크포인트가 없는 경우
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
        """상태에서 파이프라인 통계를 추출합니다.

        Args:
            state: 파이프라인 상태

        Returns:
            통계 딕셔너리
        """
        return get_statistics(state)
