"""
ID-MAS 메인 실행 파일 (3-Phase Pipeline with State Machine)
Domain-based Multi-Dataset Support for LLM Learning and Evaluation

학습(train)과 평가(eval)를 분리하여 실행:
- Train Mode: 설계 → 3-Phase 학습 → SFT 데이터 생성
- Eval Mode: Baseline, SFT, SFT_ID-MAS 평가

3-Phase Learning Pipeline:
- Phase 1: Initial Response with Scaffolding
- Phase 2: Fixed Response with Coaching DB
- Phase 3: Modeling (Teacher's articulate reasoning)

State Machine Features:
- Checkpoint-based resume support
- Progress tracking per question
- Automatic checkpoint saving every 10 questions

Terminal Goals:
- GSM8K: Generate coherent, step-by-step mathematical reasoning for grade-school math
- MATH: Solve advanced mathematical problems with multi-step reasoning
"""
import sys
import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List


# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from design_modules.step2_analysis import InstructionalAnalysis
from design_modules.step4_objectives import PerformanceObjectives
from design_modules.step5_test import TestItemDevelopment
from design_modules.step5_rubric import RubricDevelopment
from learning_loop.pipeline_controller import IDMASPipelineController
from learning_loop.student_model import StudentModel
from learning_loop.teacher_model import TeacherModel
from learning_loop.state_machine import (
    LearningStateMachine,
    LearningState,
    LearningContext,
    StateTransition,
)
from utils.base_loader import QuestionData
from utils.domain_loader import DomainLoader
from utils.answer_extractor import get_extractor, AnswerExtractor
from config.config import (
    DATA_DIR,
    AVAILABLE_STUDENT_MODELS, DEFAULT_STUDENT_MODEL,
    AVAILABLE_TEACHER_MODELS, DEFAULT_TEACHER_MODEL, DEFAULT_VLLM_TEACHER_MODEL,
    create_teacher_config, MODEL_NAME_TO_SHORT, get_model_short_name,
    get_domain_data_dirs, get_available_domains, get_eval_datasets_for_domain,
    get_training_datasets_for_domain, get_terminal_goal,
    get_design_output_dir
)


class IDMASPipeline:
    """
    ID-MAS 3-Phase Pipeline with State Machine (PDF Proposal Based)

    Each training dataset (GSM8K, MATH, SciBench, ARC) has its own Terminal Goal
    and is trained separately.

    Features:
    - State Machine based workflow management
    - Checkpoint-based resume support
    - Per-question progress tracking
    """

    def __init__(
        self,
        domain: str,
        train_dataset: str,
        student_model_name: Optional[str] = None,
        teacher_config: Optional[Dict] = None,
        resume: bool = False,
        checkpoint_interval: int = 10
    ):
        """
        Args:
            domain: Domain name (e.g., "math")
            train_dataset: Training dataset name (e.g., "gsm8k", "math")
            student_model_name: Student model name (None for default)
            teacher_config: Teacher model configuration (None for default GPT-5)
            resume: Whether to resume from checkpoint
            checkpoint_interval: Save checkpoint every N questions (default: 10)
        """
        self.domain = domain.lower()
        self.train_dataset = train_dataset.lower()
        self.student_model_name = student_model_name or DEFAULT_STUDENT_MODEL
        self.teacher_config = teacher_config
        self.checkpoint_interval = checkpoint_interval

        # Identifier for design files (domain_dataset)
        self.identifier = f"{self.domain}_{self.train_dataset}"

        # Get Terminal Goal for this training dataset
        self.terminal_goal = get_terminal_goal(self.train_dataset)

        # Domain loader
        self.loader = DomainLoader(domain)

        # Answer extractor based on domain
        self.answer_extractor = get_extractor(self.loader.answer_type)

        # Design modules (use teacher_config for design modules)
        self.analysis = InstructionalAnalysis(teacher_config)
        self.objectives = PerformanceObjectives(teacher_config)
        self.test_dev = TestItemDevelopment(teacher_config)
        self.rubric_dev = RubricDevelopment(teacher_config)

        # Model short name for file naming (e.g., "Qwen3-4B-Instruct-2507")
        self.model_short = get_model_short_name(self.student_model_name)

        # Model-specific directories (new structure: data/{domain}/train/{Model}/)
        self.model_dirs = get_domain_data_dirs(domain, self.student_model_name, self.train_dataset, mode="train")
        self.model_dir = self.model_dirs["model_dir"]
        self.raw_data_dir = self.model_dirs["raw_data_dir"]
        self.design_dir = self.model_dirs["design_dir"]

        # Student and Teacher models for pipeline
        self.student_model = StudentModel(model_name=self.student_model_name)
        self.teacher_model = TeacherModel(teacher_config)

        # Initialize State Machine
        self._init_state_machine(resume)

        # Initialize Pipeline Controller with State Machine
        self.pipeline = IDMASPipelineController(
            student_model=self.student_model,
            teacher_model=self.teacher_model,
            answer_extractor=self.answer_extractor,
            state_machine=self.state_machine,
            checkpoint_interval=checkpoint_interval
        )

    def _init_state_machine(self, resume: bool):
        """Initialize or restore State Machine."""
        if resume:
            checkpoint_path = LearningStateMachine.find_latest_checkpoint(
                self.domain, self.train_dataset, self.student_model_name
            )
            if checkpoint_path:
                print(f"\n[Resume] Loading checkpoint: {checkpoint_path}")
                self.state_machine = LearningStateMachine.load_checkpoint(checkpoint_path)
                print(f"[Resume] Current state: {self.state_machine.state.name}")
                print(f"[Resume] Progress: {self.state_machine.context.get_progress_summary()}")
            else:
                print("\n[Resume] No checkpoint found, starting fresh")
                self.state_machine = LearningStateMachine()
        else:
            self.state_machine = LearningStateMachine()

        # Initialize context
        ctx = self.state_machine.context
        ctx.domain = self.domain
        ctx.train_dataset = self.train_dataset
        ctx.terminal_goal = self.terminal_goal
        ctx.student_model = self.student_model_name
        ctx.teacher_model = self.teacher_config.get('model', DEFAULT_TEACHER_MODEL) if self.teacher_config else DEFAULT_TEACHER_MODEL
        if not ctx.started_at:
            ctx.started_at = datetime.now()

        # Register callbacks
        self._register_callbacks()

    def _register_callbacks(self):
        """Register State Machine callbacks."""
        sm = self.state_machine

        # Log state transitions
        def log_transition(transition: StateTransition):
            print(f"\n>>> State: {transition.from_state.name} -> {transition.to_state.name}")
            if transition.reason:
                print(f"    Reason: {transition.reason}")

        sm.on_transition(log_transition)

        # Handle error state
        def handle_error(machine, context):
            print(f"\n!!! Error occurred: {context.last_error}")
            print(f"    Error count: {context.error_count}")
            machine.save_checkpoint()

        sm.on_enter(LearningState.ERROR, handle_error)

    def run_design_phase(
        self,
        learning_objective: Optional[str] = None
    ) -> Dict:
        """
        교수 설계 단계 실행

        Args:
            learning_objective: 학습 목표 (None이면 데이터셋별 Terminal Goal 사용)

        Returns:
            설계 결과 딕셔너리
        """
        sm = self.state_machine
        ctx = sm.context

        print("\n" + "=" * 60)
        print("INSTRUCTIONAL DESIGN PHASE")
        print("=" * 60)

        # 1. 학습 목표 설정 (데이터셋별 Terminal Goal 사용)
        if learning_objective is None:
            learning_objective = self.terminal_goal

        print(f"\n[Step 1] Learning Objective (Terminal Goal for {self.train_dataset.upper()})")
        print(f"  {learning_objective}")

        # 2. 교수 분석
        sm.transition_to(LearningState.DESIGN_ANALYSIS, "Starting instructional analysis")
        print(f"\n[Step 2] Instructional Analysis")
        analysis_result = self.analysis.analyze(learning_objective)
        print(f"  Analysis completed")
        ctx.task_analysis = analysis_result["raw_output"]

        # 3. 수행목표 진술
        sm.transition_to(LearningState.DESIGN_OBJECTIVES, "Generating performance objectives")
        print(f"\n[Step 4] Performance Objectives")
        objectives_result = self.objectives.generate_objectives(
            analysis_result["raw_output"]
        )
        print(f"  Generated {len(objectives_result['performance_objectives'])} objectives")
        ctx.performance_objectives = objectives_result['performance_objectives']

        # 4. 루브릭 개발 (Essay 평가용)
        sm.transition_to(LearningState.DESIGN_RUBRIC, "Developing rubrics")
        print(f"\n[Step 5] Rubric Development")
        rubric_dev = RubricDevelopment()

        # output_type 결정 로직 (Default for math domain)
        output_type = "explanatory_text"
        # TODO: Add domain-specific output_type to DOMAIN_CONFIG for extensibility

        rubric = rubric_dev.generate_rubric(
            task_description=self.terminal_goal,
            output_type=output_type,
            performance_objectives=objectives_result
        )

        criteria_count = len(rubric.get('rubric', {}).get('criteria', []))
        print(f"  Generated rubric with {criteria_count} criteria for {output_type}")
        ctx.rubric = rubric

        # 설계 결과 저장
        design_result = {
            "domain": self.domain,
            "train_dataset": self.train_dataset,
            "identifier": self.identifier,
            "terminal_goal": self.terminal_goal,
            "learning_objective": learning_objective,
            "instructional_analysis": analysis_result,
            "performance_objectives": objectives_result,
            "rubrics": rubric,
            "timestamp": datetime.now().isoformat()
        }

        # 파일 저장
        output_dir = get_design_output_dir(self.domain)
        output_path = output_dir / f"{self.identifier}_design.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(design_result, f, ensure_ascii=False, indent=2)

        # Update context
        ctx.design_result = design_result

        sm.transition_to(LearningState.DESIGN_COMPLETE, "Design phase completed")

        print(f"\n Design phase completed")
        print(f"  Saved to: {output_path}")

        return design_result

    def run_learning_phase(
        self,
        design_result: Dict,
        num_questions: Optional[int] = None,
        resume: bool = False
    ) -> Dict:
        """
        3-Phase Learning Pipeline 실행

        Args:
            design_result: 설계 결과
            num_questions: 학습할 질문 개수 (None이면 전체)
            resume: 기존 로그에서 이어서 학습할지 여부

        Returns:
            학습 결과 딕셔너리
        """
        sm = self.state_machine
        ctx = sm.context

        print("\n" + "=" * 60)
        print("3-PHASE LEARNING PIPELINE")
        print("=" * 60)
        print(f"Training Dataset: {self.train_dataset.upper()}")
        print(f"Terminal Goal: {self.terminal_goal[:80]}...")

        # Load training data (if not already loaded)
        if not ctx.questions:
            questions = self.loader.load_training_data(
                dataset=self.train_dataset,
                limit=num_questions,
                shuffle=False
            )

            print(f"\nLoaded {len(questions)} training questions")

            # Prepare question data
            question_data = []
            for q in questions:
                question_data.append({
                    'question_id': q.question_id,
                    'question': q.question,
                    'problem_text': self.loader.format_question_as_prompt(q),
                    'ground_truth': self.loader.format_ground_truth(q),
                    'instruction': q.metadata.get('instruction', '')
                })

            ctx.questions = question_data
            ctx.total_questions = len(question_data)
        else:
            question_data = ctx.questions
            print(f"\n[Resume] Using {len(question_data)} questions from checkpoint")

        # Get design components
        task_analysis = design_result['instructional_analysis']['raw_output']
        performance_objectives = design_result['performance_objectives']['performance_objectives']

        # Phase 1: Initial Responses with Scaffolding
        if sm.state in [LearningState.DESIGN_COMPLETE, LearningState.INIT, LearningState.PAUSED]:
            if ctx.phase1_processed < ctx.total_questions:
                sm.transition_to(LearningState.PHASE1_RUNNING, "Starting Phase 1")

        if sm.state == LearningState.PHASE1_RUNNING:
            print("\n" + "=" * 60)
            print("[Phase 1] Generating Initial Responses with Scaffolding")
            print("=" * 60)
            correct_p1, incorrect_p1 = self.pipeline.run_phase1_batch(
                questions=question_data,
                task_analysis=task_analysis,
                performance_objectives=performance_objectives
            )
            print(f"\nPhase 1 Results: Correct={len(correct_p1)}, Incorrect={len(incorrect_p1)}")
            sm.transition_to(LearningState.PHASE1_COMPLETE, "Phase 1 completed")
        else:
            # Resume: get results from context
            correct_p1 = [r for r in ctx.phase1_results if r.get('phase1_correct')]
            incorrect_p1 = ctx.incorrect_after_phase1

        # Phase 2: Teacher Intervention + Fixed Responses
        if incorrect_p1 and sm.state == LearningState.PHASE1_COMPLETE:
            sm.transition_to(LearningState.PHASE2_SCORING, "Starting Phase 2")

        if sm.state in [LearningState.PHASE2_SCORING, LearningState.PHASE2_COACHING, LearningState.PHASE2_FIXING]:
            print("\n" + "=" * 60)
            print("[Phase 2] Teacher Intervention + Fixed Responses")
            print("=" * 60)

            if sm.state == LearningState.PHASE2_SCORING:
                sm.transition_to(LearningState.PHASE2_COACHING, "Scoring complete, generating coaching DB")

            if sm.state == LearningState.PHASE2_COACHING:
                sm.transition_to(LearningState.PHASE2_FIXING, "Coaching DB ready, generating fixed responses")

            fixed_correct, still_incorrect, coaching_db = self.pipeline.run_phase2_batch(
                incorrect_results=incorrect_p1,
                performance_objectives=performance_objectives,
                task_analysis=task_analysis,
                learning_objective=self.terminal_goal
            )
            print(f"\nPhase 2 Results: Fixed={len(fixed_correct)}, Still Incorrect={len(still_incorrect)}")
            sm.transition_to(LearningState.PHASE2_COMPLETE, "Phase 2 completed")
        else:
            still_incorrect = ctx.still_incorrect_after_phase2
            coaching_db = ctx.coaching_db

        # Phase 3: Modeling (if still incorrect)
        if still_incorrect and sm.state == LearningState.PHASE2_COMPLETE:
            sm.transition_to(LearningState.PHASE3_MODELING, "Starting Phase 3")

        if sm.state == LearningState.PHASE3_MODELING:
            print("\n" + "=" * 60)
            print("[Phase 3] Modeling (Teacher's Articulate Reasoning)")
            print("=" * 60)
            self.pipeline.run_phase3_batch(
                still_incorrect=still_incorrect,
                task_analysis=task_analysis
            )
            print(f"\nPhase 3 Results: Modeling applied to {len(still_incorrect)} questions")
            sm.transition_to(LearningState.PHASE3_COMPLETE, "Phase 3 completed")

        # Move to saving state
        if sm.state in [LearningState.PHASE1_COMPLETE, LearningState.PHASE2_COMPLETE, LearningState.PHASE3_COMPLETE]:
            sm.transition_to(LearningState.SAVING, "Saving results")

        # Save results and generate SFT data
        # New file naming: {dataset}_train_id-mas_{Model}.json, {dataset}_train_id-mas_{Model}_logs.json
        sft_filename = f"{self.train_dataset}_train_id-mas_{self.model_short}.json"
        logs_filename = f"{self.train_dataset}_train_id-mas_{self.model_short}_logs.json"

        results_path, sft_path = self.pipeline.save_results(
            output_dir=self.model_dir,
            sft_filename=sft_filename,
            logs_filename=logs_filename
        )

        # Get statistics
        stats = self.pipeline.get_statistics()
        sft_data = self.pipeline.generate_sft_data()

        learning_results = {
            "domain": self.domain,
            "train_dataset": self.train_dataset,
            "terminal_goal": self.terminal_goal,
            "total_questions": stats['total_questions'],
            "phase1_correct": stats['phase1_correct'],
            "phase2_fixed": stats['phase2_fixed'],
            "phase3_modeling": stats['phase3_modeling'],
            "sft_data_count": len(sft_data),
            "sft_data_path": str(sft_path),
            "results_path": str(results_path),
            "state_machine_summary": sm.get_summary(),
            "timestamp": datetime.now().isoformat()
        }

        # Save learning results summary to model directory
        summary_path = self.model_dir / f"{self.train_dataset}_train_summary_{self.model_short}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(learning_results, f, ensure_ascii=False, indent=2)

        # Transition to complete
        sm.transition_to(LearningState.COMPLETE, "Pipeline completed successfully")

        print("\n" + "=" * 60)
        print("LEARNING PHASE COMPLETED")
        print("=" * 60)
        print(f"Total Questions: {stats['total_questions']}")
        print(f"Phase 1 Correct (Case A): {stats['phase1_correct']}")
        print(f"Phase 2 Fixed (Case B): {stats['phase2_fixed']}")
        print(f"Phase 3 Modeling (Case C): {stats['phase3_modeling']}")
        print(f"SFT Data Generated: {len(sft_data)} entries")
        print(f"SFT Data Path: {sft_path}")

        return learning_results


class IDMASEvaluator:
    """
    ID-MAS 평가 전용 클래스

    Evaluation Methods:
    - baseline: Base model without any training
    - sft: SFT fine-tuned model (HuggingFace Hub)
    - sft_id-mas: SFT_ID-MAS fine-tuned model (HuggingFace Hub)
    """

    def __init__(
        self,
        domain: str,
        eval_dataset: str,
        student_model_name: Optional[str] = None,
        eval_method: str = "baseline"
    ):
        """
        Args:
            domain: Domain name (예: "math")
            eval_dataset: Evaluation dataset name
            student_model_name: Student model name (None for default)
            eval_method: Evaluation method ("baseline", "sft", "sft_id-mas")
        """
        self.domain = domain.lower()
        self.eval_dataset = eval_dataset.lower()
        self.student_model_name = student_model_name or DEFAULT_STUDENT_MODEL
        self.eval_method = eval_method

        # Domain loader
        self.loader = DomainLoader(domain)

        # Answer extractor based on domain
        self.answer_extractor = get_extractor(self.loader.answer_type)

        # Initialize student model based on method
        use_sft = (eval_method == "sft")
        use_sft_idmas = (eval_method == "sft_id-mas")

        self.student_model = StudentModel(
            model_name=self.student_model_name,
            use_sft_model=use_sft,
            use_sft_idmas_model=use_sft_idmas,
            sft_domain=domain
        )

        # Get model short name
        from config.config import get_model_short_name
        self.model_short = get_model_short_name(self.student_model_name)

        # Eval results directory (new structure: data/{domain}/eval/{Model}/)
        self.model_dirs = get_domain_data_dirs(domain, self.student_model_name, self.eval_dataset, mode="eval")
        self.eval_results_dir = self.model_dirs["model_dir"]

    def evaluate(
        self,
        num_questions: Optional[int] = None,
        resume: bool = True
    ) -> Dict:
        """
        평가 실행

        Args:
            num_questions: 평가할 질문 개수 (None이면 전체)
            resume: 기존 결과에서 이어서 평가할지 여부

        Returns:
            평가 결과 딕셔너리
        """
        # Method name for file naming
        method_name_map = {
            "baseline": "Baseline",
            "sft": "SFT",
            "sft_id-mas": "SFT_ID-MAS"
        }
        method_name = method_name_map.get(self.eval_method, self.eval_method.upper())

        print("\n" + "=" * 60)
        print(f"EVALUATION PHASE ({method_name})")
        print("=" * 60)

        print(f"\n Domain: {self.domain}")
        print(f"  Eval dataset: {self.eval_dataset}")
        print(f"  Student model: {self.student_model_name}")
        print(f"  Method: {method_name}")

        # Load evaluation data
        questions = self.loader.load_eval_data(self.eval_dataset, limit=num_questions)

        # Get appropriate extractor for evaluation dataset
        eval_extractor = self._get_extractor_for_dataset(questions)

        # 평가 결과 저장
        eval_results = {
            "domain": self.domain,
            "eval_dataset": self.eval_dataset,
            "method": method_name,
            "student_model": self.student_model_name,
            "answer_type": self.loader.answer_type.value,
            "total_questions": len(questions),
            "evaluated_questions": len(questions),
            "question_results": []
        }

        correct_count = 0

        # 결과 파일 경로 설정
        result_path = self.eval_results_dir / f"{self.eval_dataset}_eval_results-{method_name}.json"

        # Resume 모드: 기존 결과 파일에서 완료된 문제 복원
        processed_question_ids = set()
        restored_results = []

        if resume and result_path.exists():
            print("\n Resume mode: Loading existing results...")
            with open(result_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)

            for result in existing_results.get("question_results", []):
                processed_question_ids.add(result["question_id"])
                restored_results.append(result)
                if result.get("is_correct"):
                    correct_count += 1

            if processed_question_ids:
                print(f"  Found {len(processed_question_ids)} completed questions")
                print(f"  Current accuracy: {correct_count}/{len(processed_question_ids)}")
            else:
                print("  No existing results found. Starting fresh.")

        # 복원된 결과로 초기화
        eval_results["question_results"] = restored_results.copy()

        # 각 문제에 대해 1회만 시도
        for i, question in enumerate(questions):
            # Resume 모드에서 이미 처리된 질문은 스킵
            if question.question_id in processed_question_ids:
                continue

            print(f"\n[Eval] Question {i+1}/{len(questions)}")

            # 프롬프트 생성
            problem_text = self.loader.format_question_as_prompt(question)

            print(f"Question: {question.question[:100]}...")
            ground_truth_clean = re.sub(r'\\boxed\{([^}]+)\}', r'\1', question.ground_truth)
            print(f"Ground Truth: {ground_truth_clean}")

            # 학생 모델이 1회만 응답 생성
            print("\n[Student] Generating response...")
            student_response = self.student_model.generate_initial_response(
                problem_text=problem_text
            )

            print(f"\nStudent Response:\n{student_response[:500]}...")

            # 정답 추출 및 비교
            predicted_answer = eval_extractor.extract(student_response) or ""
            is_correct = eval_extractor.compare(
                predicted_answer, str(question.ground_truth)
            )

            if is_correct:
                correct_count += 1
                print(f"\n Correct! (Predicted: {predicted_answer})")
            else:
                print(f"\n Incorrect. (Predicted: {predicted_answer}, Expected: {ground_truth_clean})")

            # 결과 저장
            question_result = {
                "question_id": question.question_id,
                "question": question.question,
                "ground_truth": question.ground_truth,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "student_response": student_response
            }

            # Add MCQ-specific fields
            if question.choices:
                question_result["choices"] = question.choices

            eval_results["question_results"].append(question_result)

            # 점진적 저장
            self._save_eval_results(eval_results, result_path, correct_count)

        # 정확도 계산
        accuracy = correct_count / len(questions) if questions else 0

        print(f"\n{'=' * 60}")
        print(f"EVALUATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Correct: {correct_count}/{len(questions)}")
        print(f"Accuracy: {accuracy * 100:.2f}%")

        eval_results["correct_count"] = correct_count
        eval_results["accuracy"] = accuracy

        # 최종 결과 저장
        self._save_eval_results(eval_results, result_path, correct_count)

        print(f"\n Evaluation results saved to: {result_path}")

        return eval_results

    def _save_eval_results(
        self,
        eval_results: Dict,
        result_path: Path,
        correct_count: int
    ) -> None:
        """평가 결과를 JSON 파일로 저장"""
        evaluated = len(eval_results["question_results"])
        eval_results["evaluated_questions"] = evaluated
        eval_results["correct_count"] = correct_count
        eval_results["accuracy"] = correct_count / evaluated if evaluated > 0 else 0

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)

    def _get_extractor_for_dataset(
        self,
        questions: List[QuestionData]
    ) -> AnswerExtractor:
        """Get appropriate answer extractor for a dataset."""
        if questions:
            return get_extractor(questions[0].answer_type)
        return self.answer_extractor


def run_train_mode(args):
    """
    학습 모드 실행 (설계 → 3-Phase 학습 → SFT 데이터 생성)
    """
    # Create teacher config from CLI argument
    teacher_model_name = args.teacher_model or DEFAULT_TEACHER_MODEL
    teacher_config = create_teacher_config(teacher_model_name)

    print(f"\n{'=' * 60}")
    print(f"ID-MAS: TRAIN MODE (3-Phase Pipeline with State Machine)")
    print(f"{'=' * 60}")
    print(f"Domain: {args.domain}")
    print(f"Train Dataset: {args.train_dataset}")
    print(f"Student Model: {args.student_model or DEFAULT_STUDENT_MODEL}")
    print(f"Teacher Model: {teacher_model_name}")
    print(f"Resume: {args.resume}")
    print(f"{'=' * 60}\n")

    # Initialize pipeline with State Machine
    pipeline = IDMASPipeline(
        domain=args.domain,
        train_dataset=args.train_dataset,
        student_model_name=args.student_model,
        teacher_config=teacher_config,
        resume=args.resume,
        checkpoint_interval=10
    )

    print(f"Terminal Goal: {pipeline.terminal_goal[:80]}...")

    # 1. 교수 설계 단계
    if not args.run_design:
        design_dir = get_design_output_dir(pipeline.domain)
        design_path = design_dir / f"{pipeline.identifier}_design.json"

        # Check legacy flat structure and migrate if needed
        legacy_path = DATA_DIR / "design_outputs" / f"{pipeline.identifier}_design.json"
        if not design_path.exists() and legacy_path.exists():
            legacy_path.replace(design_path)
            print(f"Moved legacy design file from {legacy_path} to {design_path}")

        if not design_path.exists():
            print(f"Design file not found at {design_path}")
            print("Automatically generating new instructional design...")
            design_result = pipeline.run_design_phase()
        else:
            with open(design_path, 'r', encoding='utf-8') as f:
                design_result = json.load(f)
            print(f"Loaded existing design from {design_path}")

            # Update context with loaded design
            ctx = pipeline.state_machine.context
            ctx.design_result = design_result
            ctx.task_analysis = design_result['instructional_analysis']['raw_output']
            ctx.performance_objectives = design_result['performance_objectives']['performance_objectives']
    else:
        # Check if resuming and design already exists
        ctx = pipeline.state_machine.context
        if args.resume and ctx.design_result:
            design_result = ctx.design_result
            print(f"[Resume] Using design from checkpoint")
        else:
            design_result = pipeline.run_design_phase()

    # 2. 3-Phase 학습 단계
    learning_result = pipeline.run_learning_phase(
        design_result=design_result,
        resume=args.resume
    )

    print("\n" + "=" * 60)
    print("TRAIN MODE COMPLETED")
    print(f"Domain: {args.domain}")
    print(f"Train Dataset: {args.train_dataset}")
    print(f"SFT Data: {learning_result['sft_data_count']} entries")
    print(f"SFT Path: {learning_result['sft_data_path']}")
    print("=" * 60)


def run_eval_mode(args):
    """
    평가 모드 실행 (Baseline, SFT, SFT_ID-MAS)
    """
    print(f"\n{'=' * 60}")
    print(f"ID-MAS: EVAL MODE ({args.method.upper()})")
    print(f"{'=' * 60}")

    print(f"Domain: {args.domain}")
    print(f"Eval Dataset: {args.eval_dataset}")
    print(f"Model: {args.student_model or DEFAULT_STUDENT_MODEL}")
    print(f"{'=' * 60}\n")

    # Initialize evaluator
    evaluator = IDMASEvaluator(
        domain=args.domain,
        eval_dataset=args.eval_dataset,
        student_model_name=args.student_model,
        eval_method=args.method
    )

    # 평가 실행
    eval_result = evaluator.evaluate(resume=args.eval_resume)

    print("\n" + "=" * 60)
    print("EVAL MODE COMPLETED")
    print(f"Method: {args.method.upper()}")
    print(f"Eval Dataset: {args.eval_dataset}")
    print(f"Accuracy: {eval_result['accuracy'] * 100:.2f}%")
    print(f"Correct: {eval_result['correct_count']}/{eval_result['total_questions']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="ID-MAS: Instructional Design Multi-Agent System (3-Phase Pipeline with State Machine)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ========================================
  # TRAIN MODE (3-Phase Learning Pipeline)
  # ========================================

  # GSM8K로 학습
  python main.py --mode train --domain math --train-dataset gsm8k

  # Resume 학습 (중단된 곳에서 재개)
  python main.py --mode train --domain math --train-dataset gsm8k --resume True

  # 처음부터 새로 학습 (Resume 비활성화)
  python main.py --mode train --domain math --train-dataset gsm8k --resume False

  # MATH로 학습 (다른 모델)
  python main.py --mode train --domain math --train-dataset math \\
      --student-model Qwen/Qwen3-4B-Instruct-2507

  # ========================================
  # EVAL MODE - Baseline (Base Model)
  # ========================================

  # GSM8K Baseline 평가
  python main.py --mode eval --method baseline \\
      --domain math --eval-dataset gsm8k

  # ========================================
  # EVAL MODE - SFT (Fine-tuned Model)
  # ========================================

  # GSM8K SFT 평가
  python main.py --mode eval --method sft \\
      --domain math --eval-dataset gsm8k \\
      --student-model Qwen/Qwen2.5-3B-Instruct

  # ========================================
  # EVAL MODE - SFT_ID-MAS (ID-MAS Fine-tuned)
  # ========================================

  # GSM8K SFT_ID-MAS 평가
  python main.py --mode eval --method sft_id-mas \\
      --domain math --eval-dataset gsm8k \\
      --student-model Qwen/Qwen2.5-3B-Instruct

  # Cross-dataset SFT_ID-MAS 평가
  python main.py --mode eval --method sft_id-mas \\
      --domain math --eval-dataset svamp \\
      --student-model Qwen/Qwen2.5-7B-Instruct
        """
    )

    # Mode selection (required)
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "eval"],
        help="Execution mode: 'train' (learning) or 'eval' (evaluation)"
    )

    # ========================================
    # Train mode options
    # ========================================
    parser.add_argument(
        "--domain",
        type=str,
        choices=get_available_domains(),
        help="Domain name (e.g., 'math'). Required for all modes."
    )
    parser.add_argument(
        "--train-dataset",
        type=str,
        choices=["gsm8k", "math"],
        help="Training dataset. Required for train mode."
    )
    parser.add_argument(
        "--run-design",
        action="store_true",
        help="Run design phase to generate new instructional design. By default, loads existing design if available. Train mode only."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="True",
        choices=["True", "False"],
        help="Resume training from checkpoint. Train mode only. (default: True)"
    )

    # ========================================
    # Eval mode options
    # ========================================
    parser.add_argument(
        "--method",
        type=str,
        choices=["baseline", "sft", "sft_id-mas"],
        help="Evaluation method: 'baseline', 'sft', or 'sft_id-mas'. Eval mode only."
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        help="Evaluation dataset. Required for eval mode."
    )
    parser.add_argument(
        "--eval-resume",
        type=str,
        default="True",
        choices=["True", "False"],
        help="Resume evaluation from existing results. Eval mode only. (default: True)"
    )

    # ========================================
    # Common options
    # ========================================
    parser.add_argument(
        "--student-model",
        type=str,
        default=None,
        choices=AVAILABLE_STUDENT_MODELS,
        dest="student_model",
        help=f"Student model to use. Default: {DEFAULT_STUDENT_MODEL}"
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=None,
        choices=AVAILABLE_TEACHER_MODELS,
        dest="teacher_model",
        help=f"Teacher model for instructional design and evaluation. "
             f"Default: {DEFAULT_TEACHER_MODEL} (OpenAI). "
             f"For vLLM (GPU server): {DEFAULT_VLLM_TEACHER_MODEL}"
    )

    args = parser.parse_args()

    # ========================================
    # Validation
    # ========================================

    if args.mode == "train":
        # Train mode validation
        if not args.domain:
            parser.error("--domain is required for train mode")
        if not args.train_dataset:
            parser.error("--train-dataset is required for train mode")

        # Validate train-dataset belongs to selected domain
        available_train = get_training_datasets_for_domain(args.domain)
        if args.train_dataset not in available_train:
            parser.error(
                f"--train-dataset '{args.train_dataset}' is not available for {args.domain} domain. "
                f"Available: {available_train}"
            )

        # Check for invalid options in train mode
        if args.method:
            parser.error("--method is not allowed in train mode")
        if args.eval_dataset:
            parser.error("--eval-dataset is not allowed in train mode")

        # Convert --resume string to boolean
        args.resume = args.resume == "True"

        # Run train mode
        run_train_mode(args)

    elif args.mode == "eval":
        # Eval mode validation
        if not args.method:
            parser.error("--method is required for eval mode")
        if not args.eval_dataset:
            parser.error("--eval-dataset is required for eval mode")
        if not args.domain:
            parser.error("--domain is required for eval mode")

        # Check for invalid options in eval mode
        if args.train_dataset:
            parser.error("--train-dataset is not allowed in eval mode")
        if args.run_design:
            parser.error("--run-design is not allowed in eval mode")

        # Validate eval-dataset belongs to domain
        available_eval = get_eval_datasets_for_domain(args.domain)
        if args.eval_dataset not in available_eval:
            parser.error(
                f"--eval-dataset '{args.eval_dataset}' is not available for {args.domain} domain. "
                f"Available: {available_eval}"
            )

        # Validate model is supported for SFT/SFT_ID-MAS
        if args.method in ["sft", "sft_id-mas"]:
            model_to_check = args.student_model or DEFAULT_STUDENT_MODEL
            if model_to_check not in MODEL_NAME_TO_SHORT:
                parser.error(
                    f"Model '{model_to_check}' is not supported for {args.method.upper()} evaluation.\n"
                    f"Supported models: {list(MODEL_NAME_TO_SHORT.keys())}"
                )

        # Convert --eval-resume string to boolean
        args.eval_resume = args.eval_resume == "True"

        # Run eval mode
        run_eval_mode(args)


if __name__ == "__main__":
    main()
