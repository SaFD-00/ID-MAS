"""
ID-MAS 3-Phase Pipeline Controller with State Machine Support

Phase 1: Initial Response with Scaffolding
Phase 2: Fixed Response with Coaching DB (for incorrect answers)
Phase 3: Modeling (for still incorrect answers)

SFT Data Generation:
- Case A: Initial correct -> use initial response
- Case B: Fixed correct -> use fixed response with DB context
- Case C: Still incorrect -> use teacher's modeling response

State Machine Integration:
- Checkpoint-based resume support
- Progress tracking per question
- Automatic checkpoint saving every N questions
"""
from typing import Dict, Any, Optional, List, Tuple
import json
from datetime import datetime
from pathlib import Path

from learning_loop.state_machine import (
    LearningStateMachine,
    LearningState,
    LearningContext,
    QuestionProgress,
    IterationRecord,
)

# Phase 1 Iterative Scaffolding Configuration
MAX_PHASE1_ITERATIONS = 5


class IDMASPipelineController:
    """
    3-Phase Learning Pipeline Controller with State Machine

    Implements the PDF proposal pipeline:
    Phase 1 -> Phase 2 (if wrong) -> Phase 3 (if still wrong)

    Supports checkpoint-based resume functionality.
    """

    def __init__(
        self,
        student_model,
        teacher_model,
        answer_extractor,
        state_machine: LearningStateMachine,
        checkpoint_interval: int = 10
    ):
        """
        Args:
            student_model: StudentModel instance
            teacher_model: TeacherModel instance
            answer_extractor: AnswerExtractor instance for checking correctness
            state_machine: LearningStateMachine instance (required)
            checkpoint_interval: Save checkpoint every N questions (default: 10)
        """
        self.student = student_model
        self.teacher = teacher_model
        self.answer_extractor = answer_extractor
        self.state_machine = state_machine
        self.checkpoint_interval = checkpoint_interval

    @property
    def context(self) -> LearningContext:
        """Get state machine context."""
        return self.state_machine.context

    def reset(self):
        """Reset state machine for a new run."""
        self.state_machine.reset()

    # ==================== Single Question Processing ====================

    def process_single_question_phase1(
        self,
        question: Dict,
        task_analysis: str
    ) -> Dict:
        """
        Phase 1: Process a single question with scaffolding.

        Args:
            question: Question data dict
            task_analysis: Instructional analysis result

        Returns:
            Result dictionary
        """
        response = self.student.generate_initial_response_with_scaffolding(
            problem_text=question['problem_text'],
            task_analysis=task_analysis
        )

        predicted = self.answer_extractor.extract(response)
        is_correct = self.answer_extractor.compare(predicted, question['ground_truth'])

        result = {
            'question_id': question['question_id'],
            'question': question['question'],
            'problem_text': question['problem_text'],
            'ground_truth': question['ground_truth'],
            'instruction': question.get('instruction', ''),
            'initial_response': response,
            'predicted_answer': predicted,
            'phase1_correct': is_correct,
            'sft_case': 'A' if is_correct else None
        }

        # Update context
        ctx = self.context
        ctx.phase1_results.append(result)
        ctx.phase1_processed += 1

        # Update question progress
        qp = ctx.question_progress.get(
            question['question_id'],
            QuestionProgress(question['question_id'])
        )
        qp.phase1_complete = True
        qp.phase1_correct = is_correct
        if is_correct:
            qp.sft_case = 'A'
        ctx.question_progress[question['question_id']] = qp

        # Track incorrect for Phase 2
        if not is_correct:
            ctx.incorrect_after_phase1.append(result)

        return result

    def process_single_question_phase1_iterative(
        self,
        question: Dict,
        task_analysis: str
    ) -> Dict:
        """
        Phase 1 (Iterative): Process a single question with iterative scaffolding.

        Teacher provides hints, student responds. If incorrect, teacher provides
        more specific hints. Repeats up to MAX_PHASE1_ITERATIONS times.

        Args:
            question: Question data dict
            task_analysis: Instructional analysis result

        Returns:
            Result dictionary with iteration history
        """
        conversation_history = []
        iterations = []
        ctx = self.context
        qp = ctx.question_progress.get(
            question['question_id'],
            QuestionProgress(question['question_id'])
        )

        # Resume from previous iterations if any
        start_iteration = qp.phase1_iteration_count + 1
        if qp.phase1_conversation_history:
            conversation_history = qp.phase1_conversation_history.copy()
        if qp.phase1_iterations:
            iterations = [
                IterationRecord(
                    iteration_number=it['iteration_number'],
                    teacher_hint=it['teacher_hint'],
                    student_response=it['student_response'],
                    predicted_answer=it.get('predicted_answer'),
                    is_correct=it.get('is_correct', False),
                    timestamp=datetime.fromisoformat(it['timestamp']) if it.get('timestamp') else None
                )
                for it in qp.phase1_iterations
            ]

        last_response = None
        is_correct = False
        predicted = None
        response = None

        for iteration in range(start_iteration, MAX_PHASE1_ITERATIONS + 1):
            # 1. Teacher generates hint
            if iteration == 1:
                hint = self.teacher.generate_initial_hint(
                    problem_text=question['problem_text'],
                    task_analysis=task_analysis,
                    ground_truth=question['ground_truth']
                )
            else:
                hint = self.teacher.generate_progressive_hint(
                    problem_text=question['problem_text'],
                    task_analysis=task_analysis,
                    conversation_history=conversation_history,
                    last_response=last_response,
                    iteration_number=iteration,
                    ground_truth=question['ground_truth'],
                    max_iterations=MAX_PHASE1_ITERATIONS
                )

            conversation_history.append({
                "role": "teacher",
                "hint": hint,
                "iteration": iteration
            })

            # 2. Student responds with hint
            response = self.student.generate_response_with_hint(
                problem_text=question['problem_text'],
                teacher_hint=hint,
                conversation_history=conversation_history,
                task_analysis=task_analysis
            )

            conversation_history.append({
                "role": "student",
                "response": response,
                "iteration": iteration
            })

            # 3. Check correctness
            predicted = self.answer_extractor.extract(response)
            is_correct = self.answer_extractor.compare(predicted, question['ground_truth'])

            # Record iteration
            iteration_record = IterationRecord(
                iteration_number=iteration,
                teacher_hint=hint,
                student_response=response,
                predicted_answer=predicted,
                is_correct=is_correct,
                timestamp=datetime.now()
            )
            iterations.append(iteration_record)

            # Update question progress for checkpoint
            qp.phase1_iteration_count = iteration
            qp.phase1_conversation_history = conversation_history.copy()
            qp.phase1_iterations = [
                {
                    'iteration_number': it.iteration_number,
                    'teacher_hint': it.teacher_hint,
                    'student_response': it.student_response,
                    'predicted_answer': it.predicted_answer,
                    'is_correct': it.is_correct,
                    'timestamp': it.timestamp.isoformat() if it.timestamp else None
                }
                for it in iterations
            ]
            ctx.question_progress[question['question_id']] = qp

            # 4. If correct, stop
            if is_correct:
                print(f"    -> Correct on iteration {iteration}!")
                break

            last_response = response

            # 5. Periodic checkpoint (every 2 iterations)
            if iteration % 2 == 0:
                self.state_machine.save_checkpoint()

        # Build result
        if is_correct:
            # Success case
            result = self._build_phase1_success_result(
                question=question,
                iterations=iterations,
                conversation_history=conversation_history,
                final_response=response,
                predicted_answer=predicted,
                success_iteration=iterations[-1].iteration_number
            )

            # Update statistics
            if iterations[-1].iteration_number == 1:
                ctx.phase1_first_attempt_correct += 1
            else:
                ctx.phase1_multi_attempt_correct += 1

        else:
            # Failed after MAX_PHASE1_ITERATIONS - need reconstruction
            print(f"    -> Failed after {MAX_PHASE1_ITERATIONS} iterations. Reconstructing...")

            reconstruction = self.teacher.summarize_and_reconstruct(
                problem_text=question['problem_text'],
                ground_truth=question['ground_truth'],
                task_analysis=task_analysis,
                conversation_history=conversation_history
            )

            result = self._build_phase1_failed_result(
                question=question,
                iterations=iterations,
                conversation_history=conversation_history,
                reconstruction=reconstruction
            )

            ctx.phase1_failed_reconstructed += 1

        # Update context
        ctx.phase1_results.append(result)
        ctx.phase1_processed += 1

        # Update question progress
        qp.phase1_complete = True
        qp.phase1_correct = is_correct
        if result.get('sft_case'):
            qp.sft_case = result['sft_case']
        ctx.question_progress[question['question_id']] = qp

        # Track for Phase 2 only if failed after reconstruction
        if result.get('sft_case') == 'A-Failed':
            ctx.incorrect_after_phase1.append(result)

        return result

    def _build_phase1_success_result(
        self,
        question: Dict,
        iterations: List[IterationRecord],
        conversation_history: List[Dict],
        final_response: str,
        predicted_answer: str,
        success_iteration: int
    ) -> Dict:
        """Build result for Phase 1 success case."""
        # Build SFT output with all hints and final response
        sft_output = self._build_sft_response_from_iterations(iterations, is_success=True)

        return {
            'question_id': question['question_id'],
            'question': question['question'],
            'problem_text': question['problem_text'],
            'ground_truth': question['ground_truth'],
            'instruction': question.get('instruction', ''),
            'initial_response': final_response,  # For backwards compatibility
            'predicted_answer': predicted_answer,
            'phase1_correct': True,
            'sft_case': 'A',
            'sft_response': sft_output,
            # Iterative scaffolding details
            'iterative_scaffolding': {
                'success': True,
                'iterations_needed': success_iteration,
                'conversation_history': conversation_history,
                'iterations': [
                    {
                        'iteration_number': it.iteration_number,
                        'teacher_hint': it.teacher_hint,
                        'student_response': it.student_response,
                        'predicted_answer': it.predicted_answer,
                        'is_correct': it.is_correct
                    }
                    for it in iterations
                ]
            }
        }

    def _build_phase1_failed_result(
        self,
        question: Dict,
        iterations: List[IterationRecord],
        conversation_history: List[Dict],
        reconstruction: Dict
    ) -> Dict:
        """Build result for Phase 1 failure case (after MAX_PHASE1_ITERATIONS)."""
        reconstructed_response = reconstruction.get('reconstructed_response', '')

        return {
            'question_id': question['question_id'],
            'question': question['question'],
            'problem_text': question['problem_text'],
            'ground_truth': question['ground_truth'],
            'instruction': question.get('instruction', ''),
            'initial_response': reconstructed_response,  # Use reconstructed for SFT
            'predicted_answer': None,
            'phase1_correct': False,
            'sft_case': 'A-Failed',
            'sft_response': reconstructed_response,
            # Reconstruction details
            'reconstruction': {
                'summary': reconstruction.get('summary', ''),
                'student_weaknesses': reconstruction.get('student_weaknesses', []),
                'learning_points': reconstruction.get('learning_points', [])
            },
            # Iterative scaffolding details
            'iterative_scaffolding': {
                'success': False,
                'iterations_needed': MAX_PHASE1_ITERATIONS,
                'conversation_history': conversation_history,
                'iterations': [
                    {
                        'iteration_number': it.iteration_number,
                        'teacher_hint': it.teacher_hint,
                        'student_response': it.student_response,
                        'predicted_answer': it.predicted_answer,
                        'is_correct': it.is_correct
                    }
                    for it in iterations
                ]
            }
        }

    def _build_sft_response_from_iterations(
        self,
        iterations: List[IterationRecord],
        is_success: bool = True
    ) -> str:
        """
        Build SFT response from iteration history.

        Format for success case:
        [Guidance]
        First hint...

        [Solution Attempt]
        Student's response...

        ... (repeats for each iteration)

        Answer: [final answer]
        """
        parts = []

        for it in iterations:
            parts.append(f"[Guidance]\n{it.teacher_hint}")
            parts.append(f"[Solution Attempt]\n{it.student_response}")

            # If this is the successful iteration, we're done
            if it.is_correct:
                break

        return "\n\n".join(parts)

    def process_single_question_phase2(
        self,
        result: Dict,
        coaching_db: Dict,
        task_analysis: str,
        learning_objective: str
    ) -> Dict:
        """
        Phase 2: Process a single incorrect result with coaching.

        Args:
            result: Phase 1 result (incorrect)
            coaching_db: Coaching database
            task_analysis: Instructional analysis
            learning_objective: Terminal goal

        Returns:
            Updated result dictionary
        """
        fixed_response = self.student.generate_fixed_response_with_coaching(
            problem_text=result['problem_text'],
            coaching_db=coaching_db,
            task_analysis=task_analysis,
            learning_objective=learning_objective
        )

        predicted = self.answer_extractor.extract(fixed_response)
        is_correct = self.answer_extractor.compare(predicted, result['ground_truth'])

        result['fixed_response'] = fixed_response
        result['fixed_predicted'] = predicted
        result['phase2_correct'] = is_correct
        result['coaching_db_used'] = True

        # Update context
        ctx = self.context
        ctx.phase2_processed += 1

        # Update question progress
        qp = ctx.question_progress.get(
            result['question_id'],
            QuestionProgress(result['question_id'])
        )
        qp.phase2_complete = True
        qp.phase2_correct = is_correct

        if is_correct:
            result['sft_case'] = 'B'
            qp.sft_case = 'B'
            ctx.phase2_results.append(result)
        else:
            ctx.still_incorrect_after_phase2.append(result)

        ctx.question_progress[result['question_id']] = qp
        return result

    def process_single_question_phase3(
        self,
        result: Dict,
        task_analysis: str
    ) -> Dict:
        """
        Phase 3: Generate modeling response for a single question.

        Args:
            result: Phase 2 result (still incorrect)
            task_analysis: Instructional analysis

        Returns:
            Updated result dictionary
        """
        modeling_response = self.teacher.generate_modeling_response(
            problem_text=result['problem_text'],
            ground_truth=result['ground_truth'],
            task_analysis=task_analysis
        )

        result['modeling_response'] = modeling_response
        result['sft_case'] = 'C'
        result['phase3_applied'] = True

        # Update context
        ctx = self.context
        ctx.phase3_results.append(result)
        ctx.phase3_processed += 1

        # Update question progress
        qp = ctx.question_progress.get(
            result['question_id'],
            QuestionProgress(result['question_id'])
        )
        qp.phase3_complete = True
        qp.sft_case = 'C'
        ctx.question_progress[result['question_id']] = qp

        return result

    # ==================== Batch Processing with Checkpoint ====================

    def run_phase1_with_checkpoint(
        self,
        questions: List[Dict],
        task_analysis: str,
        performance_objectives: List[Dict],
        use_iterative: bool = True
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Phase 1: Generate initial responses with checkpoint support.

        Skips already processed questions and saves checkpoints periodically.

        Args:
            questions: List of question data dicts
            task_analysis: Instructional analysis result
            performance_objectives: List of performance objectives
            use_iterative: Use iterative scaffolding (default: True)

        Returns:
            (correct_results, incorrect_results)
        """
        correct = []
        incorrect = []
        ctx = self.context

        # Store task analysis and POs in context
        ctx.task_analysis = task_analysis
        ctx.performance_objectives = performance_objectives
        ctx.total_questions = len(questions)
        ctx.questions = questions

        # Get already processed question IDs
        processed_ids = {
            qid for qid, qp in ctx.question_progress.items()
            if qp.phase1_complete
        }

        # Collect already processed results
        for result in ctx.phase1_results:
            if result.get('phase1_correct'):
                correct.append(result)
            else:
                incorrect.append(result)

        mode_str = "Iterative Scaffolding" if use_iterative else "Single Shot"
        print(f"\n[Phase 1] Mode: {mode_str} (max {MAX_PHASE1_ITERATIONS} iterations)")

        for i, q in enumerate(questions):
            # Skip already processed
            if q['question_id'] in processed_ids:
                print(f"  [Phase 1] Skipping {q['question_id']} (already processed)")
                continue

            print(f"\n[Phase 1] Question {i+1}/{len(questions)}: {q['question_id']}")

            try:
                if use_iterative:
                    result = self.process_single_question_phase1_iterative(q, task_analysis)

                    if result['phase1_correct']:
                        iterations_needed = result.get('iterative_scaffolding', {}).get('iterations_needed', 1)
                        print(f"  -> Correct! (Case A, {iterations_needed} iteration(s))")
                        correct.append(result)
                    else:
                        print(f"  -> Failed after {MAX_PHASE1_ITERATIONS} iterations (Case A-Failed)")
                        incorrect.append(result)
                else:
                    # Legacy single-shot mode
                    result = self.process_single_question_phase1(q, task_analysis)

                    if result['phase1_correct']:
                        print(f"  -> Correct! (Case A)")
                        correct.append(result)
                    else:
                        gt_display = q['ground_truth'][:50] if len(q['ground_truth']) > 50 else q['ground_truth']
                        print(f"  -> Incorrect (predicted: {result['predicted_answer']}, expected: {gt_display}...)")
                        incorrect.append(result)

                # Periodic checkpoint
                if ctx.phase1_processed % self.checkpoint_interval == 0:
                    checkpoint_path = self.state_machine.save_checkpoint()
                    print(f"  [Checkpoint] Saved at question {ctx.phase1_processed}")

            except Exception as e:
                print(f"  -> Error: {e}")
                self.state_machine.set_error(str(e))
                self.state_machine.save_checkpoint()
                raise

        # Print summary
        if use_iterative:
            print(f"\n[Phase 1 Summary]")
            print(f"  First attempt correct: {ctx.phase1_first_attempt_correct}")
            print(f"  Multi-attempt correct: {ctx.phase1_multi_attempt_correct}")
            print(f"  Failed (reconstructed): {ctx.phase1_failed_reconstructed}")

        return correct, incorrect

    def run_phase2_with_checkpoint(
        self,
        incorrect_results: List[Dict],
        performance_objectives: List[Dict],
        task_analysis: str,
        learning_objective: str
    ) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Phase 2: Teacher intervention + Fixed Response with checkpoint support.

        Args:
            incorrect_results: Results from Phase 1 (incorrect only)
            performance_objectives: PO list
            task_analysis: Instructional analysis
            learning_objective: Terminal goal

        Returns:
            (fixed_correct, still_incorrect, coaching_db)
        """
        if not incorrect_results:
            return [], [], {}

        ctx = self.context

        print(f"\n[Phase 2] Processing {len(incorrect_results)} incorrect responses...")

        # Step 1: Score all incorrect responses (if not already done)
        if not ctx.coaching_db:
            print("  Step 1: Scoring by Performance Objectives...")
            all_scores = []
            for result in incorrect_results:
                if 'phase2_scores' not in result:
                    scores = self.teacher.score_by_performance_objectives(
                        student_response=result['initial_response'],
                        performance_objectives=performance_objectives,
                        ground_truth=result['ground_truth']
                    )
                    result['phase2_scores'] = scores
                all_scores.append(result.get('phase2_scores', {}))

            # Step 2: Find weak objectives
            print("  Step 2: Finding weak objectives...")
            weak_objectives = self._find_weak_objectives(
                all_scores,
                performance_objectives,
                error_threshold=0.4
            )
            print(f"    Found {len(weak_objectives)} weak objectives")

            # Step 3: Generate Coaching DB
            print("  Step 3: Generating Coaching DB...")
            if weak_objectives:
                weak_analysis = self.teacher.analyze_weak_objectives(
                    weak_objectives=weak_objectives,
                    student_responses=[r['initial_response'] for r in incorrect_results],
                    task_analysis=task_analysis
                )

                coaching_db = self.teacher.generate_coaching_db(
                    learning_objective=learning_objective,
                    task_analysis=task_analysis,
                    weak_analysis=weak_analysis
                )
            else:
                coaching_db = {
                    "learning_objective": learning_objective,
                    "task_analysis_summary": task_analysis[:500],
                    "performance_areas": [],
                    "general_tips": ["Review the problem carefully", "Check your calculations"]
                }

            ctx.coaching_db = coaching_db
        else:
            coaching_db = ctx.coaching_db

        # Step 4: Generate fixed responses
        print("  Step 4: Generating Fixed Responses with Coaching DB...")
        fixed_correct = []
        still_incorrect = []

        # Get already processed question IDs for Phase 2
        processed_ids = {
            qid for qid, qp in ctx.question_progress.items()
            if qp.phase2_complete
        }

        # Collect already processed Phase 2 results
        for result in ctx.phase2_results:
            fixed_correct.append(result)
        for result in ctx.still_incorrect_after_phase2:
            still_incorrect.append(result)

        for i, result in enumerate(incorrect_results):
            if result['question_id'] in processed_ids:
                print(f"    Skipping {result['question_id']} (already processed)")
                continue

            print(f"    Fixed Response {i+1}/{len(incorrect_results)}: {result['question_id']}")

            try:
                updated_result = self.process_single_question_phase2(
                    result, coaching_db, task_analysis, learning_objective
                )

                if updated_result.get('phase2_correct'):
                    print(f"      -> Fixed! (Case B)")
                    fixed_correct.append(updated_result)
                else:
                    print(f"      -> Still incorrect")
                    still_incorrect.append(updated_result)

                # Periodic checkpoint
                if ctx.phase2_processed % self.checkpoint_interval == 0:
                    self.state_machine.save_checkpoint()
                    print(f"      [Checkpoint] Saved at Phase 2 question {ctx.phase2_processed}")

            except Exception as e:
                print(f"      -> Error: {e}")
                self.state_machine.set_error(str(e))
                self.state_machine.save_checkpoint()
                raise

        return fixed_correct, still_incorrect, coaching_db

    def run_phase3_with_checkpoint(
        self,
        still_incorrect: List[Dict],
        task_analysis: str
    ) -> List[Dict]:
        """
        Phase 3: Modeling with checkpoint support.

        Args:
            still_incorrect: Results still incorrect after Phase 2
            task_analysis: Instructional analysis

        Returns:
            List of results with modeling responses
        """
        if not still_incorrect:
            return []

        ctx = self.context
        results = []

        print(f"\n[Phase 3] Generating Modeling for {len(still_incorrect)} questions...")

        # Get already processed question IDs for Phase 3
        processed_ids = {
            qid for qid, qp in ctx.question_progress.items()
            if qp.phase3_complete
        }

        # Collect already processed Phase 3 results
        for result in ctx.phase3_results:
            results.append(result)

        for i, result in enumerate(still_incorrect):
            if result['question_id'] in processed_ids:
                print(f"  Skipping {result['question_id']} (already processed)")
                continue

            print(f"  Modeling {i+1}/{len(still_incorrect)}: {result['question_id']}")

            try:
                updated_result = self.process_single_question_phase3(result, task_analysis)
                results.append(updated_result)

                # Periodic checkpoint
                if ctx.phase3_processed % self.checkpoint_interval == 0:
                    self.state_machine.save_checkpoint()
                    print(f"    [Checkpoint] Saved at Phase 3 question {ctx.phase3_processed}")

            except Exception as e:
                print(f"    -> Error: {e}")
                self.state_machine.set_error(str(e))
                self.state_machine.save_checkpoint()
                raise

        return results

    # ==================== Legacy Batch Methods (without checkpoint) ====================

    def run_phase1_batch(
        self,
        questions: List[Dict],
        task_analysis: str,
        performance_objectives: List[Dict],
        use_iterative: bool = True
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Phase 1: Generate initial responses for all questions with scaffolding.
        Delegates to checkpoint-enabled method.

        Args:
            questions: List of question data dicts
            task_analysis: Instructional analysis result
            performance_objectives: List of performance objectives
            use_iterative: Use iterative scaffolding (default: True)

        Returns:
            (correct_results, incorrect_results)
        """
        return self.run_phase1_with_checkpoint(
            questions, task_analysis, performance_objectives, use_iterative
        )

    def run_phase2_batch(
        self,
        incorrect_results: List[Dict],
        performance_objectives: List[Dict],
        task_analysis: str,
        learning_objective: str
    ) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Phase 2: Teacher intervention + Fixed Response generation.
        Delegates to checkpoint-enabled method.
        """
        return self.run_phase2_with_checkpoint(
            incorrect_results, performance_objectives, task_analysis, learning_objective
        )

    def run_phase3_batch(
        self,
        still_incorrect: List[Dict],
        task_analysis: str
    ) -> List[Dict]:
        """
        Phase 3: Modeling - Teacher provides correct solution.
        Delegates to checkpoint-enabled method.
        """
        return self.run_phase3_with_checkpoint(still_incorrect, task_analysis)

    # ==================== Helper Methods ====================

    def _find_weak_objectives(
        self,
        all_scores: List[Dict],
        performance_objectives: List[Dict],
        error_threshold: float = 0.4
    ) -> List[Dict]:
        """Find Performance Objectives with error rate >= threshold."""
        obj_errors = {}
        for obj in performance_objectives:
            obj_target = obj.get('target', '')
            obj_errors[obj_target] = {'total': 0, 'errors': 0}

        for scores in all_scores:
            for obj_score in scores.get('objective_scores', []):
                target = obj_score.get('objective_target', '')
                if target in obj_errors:
                    obj_errors[target]['total'] += 1
                    if obj_score.get('score', 0) < 0.6:
                        obj_errors[target]['errors'] += 1

        weak = []
        for target, counts in obj_errors.items():
            if counts['total'] > 0:
                error_rate = counts['errors'] / counts['total']
                if error_rate >= error_threshold:
                    weak.append({
                        'target': target,
                        'error_rate': error_rate,
                        'error_count': counts['errors'],
                        'total_count': counts['total']
                    })

        return weak

    # ==================== SFT Data Generation ====================

    def generate_sft_data(self) -> List[Dict]:
        """
        Generate SFT training data from all phases.

        Output format:
        {
            "instruction": "...",
            "input": "Question: ~~~~",
            "output": "..."
        }

        SFT Cases:
        - A: Correct with iterative scaffolding (all hints + final response)
        - A-Failed: Failed after max iterations (reconstructed response)
        - B: Fixed with coaching DB
        - C: Teacher modeling
        """
        sft_data = []
        ctx = self.context

        # Collect all results
        all_results = []

        # Phase 1 correct (Case A) and failed-reconstructed (Case A-Failed)
        for r in ctx.phase1_results:
            if r.get('sft_case') in ('A', 'A-Failed'):
                all_results.append(r)

        # Phase 2 fixed (Case B)
        for r in ctx.phase2_results:
            if r.get('sft_case') == 'B':
                all_results.append(r)

        # Phase 3 modeling (Case C)
        for r in ctx.phase3_results:
            if r.get('sft_case') == 'C':
                all_results.append(r)

        for result in all_results:
            sft_entry = self._create_sft_entry(result)
            if sft_entry:
                sft_data.append(sft_entry)

        return sft_data

    def _create_sft_entry(self, result: Dict) -> Optional[Dict]:
        """Create single SFT entry based on case."""
        case = result.get('sft_case')

        # Determine output based on case
        if case == 'A':
            # Check for sft_response (iterative scaffolding) or fall back to initial_response
            output = result.get('sft_response', result.get('initial_response', ''))
        elif case == 'A-Failed':
            # Reconstructed response for failed iterative scaffolding
            output = result.get('sft_response', result.get('initial_response', ''))
        elif case == 'B':
            output = result['fixed_response']
        elif case == 'C':
            output = result['modeling_response']
        else:
            return None

        # Determine instruction based on case
        instruction = result.get('instruction', '')
        if not instruction:
            if case == 'A':
                instruction = "Solve the following problem with teacher guidance."
            elif case == 'A-Failed':
                instruction = "Solve the following problem, learning from common mistakes."
            else:
                instruction = "Solve the following problem step by step."

        # Build metadata
        metadata = {
            "question_id": result['question_id'],
            "sft_case": case,
            "ground_truth": result['ground_truth']
        }

        # Add iterative scaffolding metadata if available
        if case in ('A', 'A-Failed'):
            iterative_info = result.get('iterative_scaffolding', {})
            metadata['iterations'] = iterative_info.get('iterations_needed', 1)

            if case == 'A-Failed':
                reconstruction = result.get('reconstruction', {})
                metadata['student_weaknesses'] = reconstruction.get('student_weaknesses', [])
                metadata['learning_points'] = reconstruction.get('learning_points', [])

        return {
            "instruction": instruction,
            "input": f"Question: {result['question']}",
            "output": output,
            "metadata": metadata
        }

    # ==================== Statistics and Results ====================

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        ctx = self.context
        phase1_correct = sum(1 for r in ctx.phase1_results if r.get('phase1_correct'))
        phase1_a_failed = sum(1 for r in ctx.phase1_results if r.get('sft_case') == 'A-Failed')
        phase2_fixed = len(ctx.phase2_results)
        phase3_modeling = len(ctx.phase3_results)

        stats = {
            "total_questions": ctx.total_questions,
            "phase1_processed": ctx.phase1_processed,
            "phase1_correct": phase1_correct,
            "phase1_incorrect": ctx.phase1_processed - phase1_correct,
            "phase2_processed": ctx.phase2_processed,
            "phase2_fixed": phase2_fixed,
            "phase3_processed": ctx.phase3_processed,
            "phase3_modeling": phase3_modeling,
            "sft_case_a": phase1_correct,
            "sft_case_a_failed": phase1_a_failed,
            "sft_case_b": phase2_fixed,
            "sft_case_c": phase3_modeling,
            # Iterative scaffolding statistics
            "iterative_scaffolding": {
                "first_attempt_correct": ctx.phase1_first_attempt_correct,
                "multi_attempt_correct": ctx.phase1_multi_attempt_correct,
                "failed_reconstructed": ctx.phase1_failed_reconstructed,
                "max_iterations": MAX_PHASE1_ITERATIONS
            }
        }

        return stats

    def save_results(self, output_dir: Path, sft_filename: str, logs_filename: str) -> Tuple[Path, Path]:
        """
        Save all pipeline results and SFT data.

        New file naming: {dataset}_train_id-mas_{Model}.json, {dataset}_train_id-mas_{Model}_logs.json

        Args:
            output_dir: Output directory
            sft_filename: SFT data filename (e.g., "gsm8k_train_id-mas_Qwen3-4B-Instruct-2507.json")
            logs_filename: Pipeline logs filename (e.g., "gsm8k_train_id-mas_Qwen3-4B-Instruct-2507_logs.json")

        Returns:
            (results_path, sft_path)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        ctx = self.context

        # Save detailed results
        results = {
            "phase1_results": ctx.phase1_results,
            "phase2_results": ctx.phase2_results,
            "phase3_results": ctx.phase3_results,
            "coaching_db": ctx.coaching_db,
            "statistics": self.get_statistics(),
            "state_machine_summary": self.state_machine.get_summary(),
            "timestamp": datetime.now().isoformat()
        }

        results_path = output_dir / logs_filename
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Save SFT data
        sft_data = self.generate_sft_data()
        sft_path = output_dir / sft_filename
        with open(sft_path, 'w', encoding='utf-8') as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=2)

        # Save final checkpoint
        self.state_machine.save_checkpoint()

        return results_path, sft_path
