"""
ID-MAS 메인 실행 파일 (LangGraph 기반 Iterative Scaffolding Pipeline)
Domain-based Multi-Dataset Support for LLM Learning and Evaluation

학습(train)과 평가(eval)를 분리하여 실행:
- Train Mode: 설계 → Iterative Scaffolding 학습 → SFT 데이터 생성
- Eval Mode: Baseline, SFT, SFT_ID-MAS 평가

Iterative Scaffolding Pipeline (LangGraph):
- Teacher-guided iterative response generation (max 5 iterations)
- Performance Objectives based evaluation with Socratic questions
- Case A (PO satisfied) / Case B (reconstructed) SFT data generation

LangGraph Features:
- StateGraph based workflow management
- Conditional routing for question processing
- Built-in checkpointing support

Instructional Goals:
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

from design_modules.analysis import InstructionalAnalysis
from design_modules.objectives import PerformanceObjectives
from design_modules.rubric import RubricDevelopment
from design_modules.instructional_goal import InstructionalGoalGenerator
from learning_loop.graph import IDMASGraphRunner
from learning_loop.graph.state import get_statistics
from learning_loop.student_model import StudentModel
from learning_loop.teacher_model import TeacherModel
from utils.base_loader import QuestionData
from utils.domain_loader import DomainLoader
from utils.answer_extractor import get_extractor, AnswerExtractor
from config import (
    DATA_DIR, TRAINING_DATASETS,
    AVAILABLE_STUDENT_MODELS, DEFAULT_STUDENT_MODEL,
    AVAILABLE_TEACHER_MODELS, DEFAULT_TEACHER_MODEL,
    create_teacher_config, MODEL_NAME_TO_SHORT, get_model_short_name,
    get_domain_data_dirs, get_available_domains, get_eval_datasets_for_domain,
    get_training_datasets_for_domain, get_instructional_goal,
    get_design_output_dir
)
from models.model_cache import ModelCache
from utils.dataset_enhancer import DataEnhancer, ENHANCED_INSTRUCTION_TEMPLATE


class IDMASPipeline:
    """
    ID-MAS Iterative Scaffolding Pipeline with LangGraph (Research Proposal Based)

    Each training dataset (GSM8K, MATH) has its own Instructional Goal
    and is trained separately using LangGraph StateGraph.

    Features:
    - LangGraph based workflow management
    - Iterative scaffolding with teacher guidance
    - Built-in checkpointing support
    - Per-question progress tracking
    """

    def __init__(
        self,
        domain: str,
        train_dataset: str,
        student_model_name: Optional[str] = None,
        teacher_config: Optional[Dict] = None,
        resume: bool = False,
        checkpoint_interval: int = 10,
        use_enhanced_data: bool = False,
        enhanced_model_suffix: Optional[str] = None
    ):
        """
        Args:
            domain: Domain name (e.g., "math")
            train_dataset: Training dataset name (e.g., "gsm8k", "math")
            student_model_name: Student model name (None for default)
            teacher_config: Teacher model configuration (None for default OpenAI model)
            resume: Whether to resume from checkpoint
            checkpoint_interval: Save checkpoint every N questions (default: 10)
            use_enhanced_data: Use enhanced training data with pre-generated Instructional Goal
            enhanced_model_suffix: Model suffix for enhanced data file
        """
        self.domain = domain.lower()
        self.train_dataset = train_dataset.lower()
        self.student_model_name = student_model_name or DEFAULT_STUDENT_MODEL
        self.teacher_config = teacher_config
        self.checkpoint_interval = checkpoint_interval
        self.resume = resume
        self.use_enhanced_data = use_enhanced_data
        self.enhanced_model_suffix = enhanced_model_suffix

        # Identifier for design files (domain_dataset)
        self.identifier = f"{self.domain}_{self.train_dataset}"

        # Model short name for file naming (e.g., "Qwen3-4B-Instruct-2507")
        self.model_short = get_model_short_name(self.student_model_name)

        # Teacher model name for state (정의를 먼저 해야 get_instructional_goal에서 사용 가능)
        self.teacher_model_name = teacher_config.get('model', DEFAULT_TEACHER_MODEL) if teacher_config else DEFAULT_TEACHER_MODEL

        # Get Instructional Goal from design JSON (None if not generated yet)
        self.instructional_goal = get_instructional_goal(self.train_dataset, teacher_model=self.teacher_model_name)

        # Domain loader
        self.loader = DomainLoader(domain)

        # Answer extractor based on domain
        self.answer_extractor = get_extractor(self.loader.answer_type)

        # Design modules (use teacher_config for design modules)
        self.instructional_goal_gen = InstructionalGoalGenerator(teacher_config)
        self.analysis = InstructionalAnalysis(teacher_config)
        self.objectives = PerformanceObjectives(teacher_config)
        self.rubric_dev = RubricDevelopment(teacher_config)

        # Model-specific directories (new structure: data/{domain}/train/{teacher_model}/{student_model}/)
        self.model_dirs = get_domain_data_dirs(
            domain,
            self.student_model_name,
            self.train_dataset,
            mode="train",
            teacher_model_name=self.teacher_model_name
        )
        self.model_dir = self.model_dirs["model_dir"]
        self.raw_data_dir = self.model_dirs["raw_data_dir"]
        self.design_dir = self.model_dirs["design_dir"]

        # Student and Teacher models for pipeline
        self.student_model = StudentModel(model_name=self.student_model_name)
        self.teacher_model = TeacherModel(teacher_config)

        # Initialize LangGraph Runner
        self.graph_runner = IDMASGraphRunner(
            student_model=self.student_model,
            teacher_model=self.teacher_model,
            answer_extractor=self.answer_extractor,
            checkpoint_dir=self.model_dir,
        )

    def run_design_phase(
        self,
        learning_objective: Optional[str] = None,
        regenerate_instructional_goal: bool = False
    ) -> Dict:
        """
        교수 설계 단계 실행

        Args:
            learning_objective: 학습 목표 (None이면 데이터셋별 Instructional Goal 사용)
            regenerate_instructional_goal: Instructional Goal 재생성 여부

        Returns:
            설계 결과 딕셔너리
        """
        print("\n" + "=" * 60)
        print("INSTRUCTIONAL DESIGN PHASE")
        print("=" * 60)

        # [Step 0] Instructional Goal Generation
        instructional_goal_metadata = None
        samples_path = self.raw_data_dir / f"{self.train_dataset}_samples.json"

        if samples_path.exists() and (regenerate_instructional_goal or self.instructional_goal is None):
            print(f"\n[Step 0] Instructional Goal Generation")

            with open(samples_path, 'r', encoding='utf-8') as f:
                train_samples = json.load(f)
            print(f"  Loaded {len(train_samples)} samples from {samples_path.name}")

            # Teacher Model로 Instructional Goal 생성 (3번 재시도, 실패 시 종료)
            try:
                instructional_goal_result = self.instructional_goal_gen.generate(
                    train_samples=train_samples,
                    domain=self.domain,
                    dataset=self.train_dataset,
                    max_retries=3
                )
                self.instructional_goal = instructional_goal_result["instructional_goal"]
                instructional_goal_metadata = instructional_goal_result.get("metadata", {})
                print(f"  Generated: {self.instructional_goal[:80]}...")

            except RuntimeError as e:
                # 3번 재시도 후 실패 → 프로그램 종료
                print(f"\n[FATAL] {e}")
                print(f"Please check:")
                print(f"  1. Teacher model availability")
                print(f"  2. Sample data quality ({samples_path.name})")
                print(f"  3. Network connection (if using API)")
                sys.exit(1)

            except Exception as e:
                # 기타 예외 → 프로그램 종료 (fallback 없음)
                print(f"\n[FATAL] Unexpected error during Instructional Goal generation: {e}")
                sys.exit(1)

        elif not samples_path.exists():
            print(f"\n[Step 0] Instructional Goal Generation (SKIPPED)")
            print(f"  Samples file not found: {samples_path.name}")
            print(f"  Run 'python -m utils.sample_extractor' to generate samples.")
            if not self.instructional_goal:
                print(f"  Instructional Goal: Not generated yet")
        else:
            # 기존 design JSON에서 로드된 경우
            if self.instructional_goal:
                print(f"\n[Step 0] Instructional Goal (loaded from design JSON)")
                print(f"  {self.instructional_goal[:80]}...")

        # [Step 1] 학습 목표 설정 (데이터셋별 Instructional Goal 사용)
        if learning_objective is None:
            if not self.instructional_goal:
                print(f"\n[FATAL] Instructional Goal not found for {self.train_dataset}.")
                print(f"Run design phase with --run-design flag first.")
                sys.exit(1)
            learning_objective = self.instructional_goal

        print(f"\n[Step 1] Learning Objective (Instructional Goal for {self.train_dataset.upper()})")
        print(f"  {learning_objective}")

        # Design Phase Steps (2-4) with RuntimeError handling
        try:
            # 2. 교수 분석
            print(f"\n[Step 2] Instructional Analysis")
            analysis_result = self.analysis.analyze(learning_objective, max_retries=3)
            print(f"  Analysis completed")

            # 3. 수행목표 진술
            print(f"\n[Step 3] Performance Objectives")
            objectives_result = self.objectives.generate_objectives(
                analysis_result["raw_output"],
                max_retries=3
            )
            print(f"  Generated {len(objectives_result['performance_objectives'])} objectives")

            # 4. 루브릭 개발 (Essay 평가용)
            print(f"\n[Step 4] Rubric Development")

            # output_type 결정 로직 (Default for math domain)
            output_type = "explanatory_text"

            rubric = self.rubric_dev.generate_rubric(
                task_description=self.instructional_goal,
                output_type=output_type,
                performance_objectives=objectives_result,
                max_retries=3
            )

            criteria_count = len(rubric.get('rubric', {}).get('criteria', []))
            print(f"  Generated rubric with {criteria_count} criteria for {output_type}")

        except RuntimeError as e:
            print(f"\n[FATAL] Design Phase failed: {e}")
            print(f"Please check:")
            print(f"  1. Teacher model availability")
            print(f"  2. Network connection (if using API)")
            sys.exit(1)

        # 설계 결과 저장
        design_result = {
            "domain": self.domain,
            "train_dataset": self.train_dataset,
            "identifier": self.identifier,
            "instructional_goal": self.instructional_goal,
            "instructional_goal_metadata": instructional_goal_metadata,
            "learning_objective": learning_objective,
            "instructional_analysis": analysis_result,
            "performance_objectives": objectives_result,
            "rubrics": rubric,
            "timestamp": datetime.now().isoformat()
        }

        # 파일 저장
        output_dir = get_design_output_dir(self.domain, self.teacher_model_name)
        output_path = output_dir / f"{self.identifier}_design.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(design_result, f, ensure_ascii=False, indent=2)

        print(f"\n Design phase completed")
        print(f"  Saved to: {output_path}")

        return design_result

    def generate_enhanced_data(self, design_result: Dict) -> Path:
        """
        Design 결과를 기반으로 Enhanced Training Data 생성

        중요: instruction 필드는 원본을 유지합니다.
        - SFT 학습 시 모델은 원본 instruction만 보고 학습
        - Enhanced instruction은 _enhanced_instruction 메타데이터로 저장 (참조용)
        - 이를 통해 train-test mismatch 방지

        Args:
            design_result: run_design_phase()의 결과

        Returns:
            생성된 enhanced data 파일 경로
        """
        print("\n" + "=" * 60)
        print("ENHANCED DATA GENERATION")
        print("=" * 60)

        # 1. 학습목표와 과제분석 추출
        instructional_goal = design_result.get("instructional_goal", "")
        task_analysis = design_result.get("instructional_analysis", {}).get("raw_output", "")

        if not instructional_goal:
            raise ValueError("Instructional Goal not found in design result")

        print(f"\n[Step 1] Using Instructional Goal")
        print(f"  {instructional_goal[:80]}...")

        print(f"\n[Step 2] Using Task Analysis")
        print(f"  {len(task_analysis)} characters")

        # 2. 소스 데이터 로드
        source_path = self.raw_data_dir / f"{self.train_dataset}_train.json"
        if not source_path.exists():
            raise FileNotFoundError(f"Training data not found: {source_path}")

        with open(source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"\n[Step 3] Loaded {len(data)} records from {source_path.name}")

        # 3. 메타데이터 추가 (instruction은 원본 유지)
        print(f"\n[Step 4] Adding enhancement metadata...")
        enhanced_data = []
        for item in data:
            new_item = item.copy()
            original_instruction = item.get("instruction", "")

            # Enhanced instruction 생성 (메타데이터용)
            enhanced_instruction = ENHANCED_INSTRUCTION_TEMPLATE.format(
                original_instruction=original_instruction,
                instructional_goal=instructional_goal,
                task_analysis=task_analysis
            )

            # instruction은 원본 유지, enhanced version은 메타데이터로만 저장
            new_item["_enhanced"] = True
            new_item["_instructional_goal"] = instructional_goal
            new_item["_task_analysis"] = task_analysis
            new_item["_enhanced_instruction"] = enhanced_instruction

            enhanced_data.append(new_item)

        # 4. 저장
        model_suffix = get_model_short_name(self.teacher_model_name)
        output_path = self.raw_data_dir / f"{self.train_dataset}_train_ID-MAS_{model_suffix}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, ensure_ascii=False, indent=2)

        print(f"\n[Step 5] Saved enhanced data")
        print(f"  Path: {output_path}")
        print(f"  Records: {len(enhanced_data)}")

        # 5. 메타데이터 저장
        metadata_path = self.raw_data_dir / f"{self.train_dataset}_ID-MAS_metadata_{model_suffix}.json"
        metadata = {
            "domain": self.domain,
            "dataset": self.train_dataset,
            "instructional_goal": instructional_goal,
            "task_analysis": task_analysis,
            "teacher_model": self.teacher_model_name,
            "model_suffix": model_suffix,
            "timestamp": datetime.now().isoformat()
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"\n Enhanced data generation completed")

        # 자동으로 enhanced data 사용 설정
        self.use_enhanced_data = True
        self.enhanced_model_suffix = model_suffix

        return output_path

    def run_learning_phase(
        self,
        design_result: Dict,
        num_questions: Optional[int] = None,
        resume: bool = False
    ) -> Dict:
        """
        LangGraph 기반 Iterative Scaffolding Pipeline 실행

        Args:
            design_result: 설계 결과
            num_questions: 학습할 질문 개수 (None이면 전체)
            resume: 기존 로그에서 이어서 학습할지 여부

        Returns:
            학습 결과 딕셔너리
        """
        # Load training data (enhanced or regular)
        if self.use_enhanced_data and self.enhanced_model_suffix:
            print(f"\n[Enhanced Data] Loading enhanced training data...")
            print(f"  Model suffix: {self.enhanced_model_suffix}")
            questions = self.loader.load_enhanced_training_data(
                dataset=self.train_dataset,
                model_suffix=self.enhanced_model_suffix,
                limit=num_questions,
                shuffle=False
            )
            print(f"\nLoaded {len(questions)} enhanced training questions")
        else:
            questions = self.loader.load_training_data(
                dataset=self.train_dataset,
                limit=num_questions,
                shuffle=False
            )
            print(f"\nLoaded {len(questions)} training questions")

        # Prepare question data (new field names: id, instruction, input, output)
        question_data = []
        for q in questions:
            question_data.append({
                'id': q.question_id,
                'instruction': q.metadata.get('instruction', ''),
                'input': q.question,
                'output': q.metadata.get('full_output', q.ground_truth_formatted),
                'problem_text': self.loader.format_question_as_prompt(q),
            })

        # Run LangGraph pipeline
        thread_id = f"{self.domain}_{self.train_dataset}_{self.model_short}"

        final_state = self.graph_runner.run(
            domain=self.domain,
            train_dataset=self.train_dataset,
            instructional_goal=self.instructional_goal,
            student_model_name=self.student_model_name,
            teacher_model_name=self.teacher_model_name,
            model_short=self.model_short,
            questions=question_data,
            design_result=design_result,
            output_dir=self.model_dir,
            checkpoint_interval=self.checkpoint_interval,
            use_iterative_scaffolding=True,
            max_iterations=5,
            thread_id=thread_id,
            resume=resume,
        )

        # Extract statistics
        stats = get_statistics(final_state)

        learning_results = {
            "domain": self.domain,
            "train_dataset": self.train_dataset,
            "instructional_goal": self.instructional_goal,
            "total_questions": stats['total_questions'],
            "scaffolding_processed": stats['scaffolding_processed'],
            "scaffolding_correct": stats['scaffolding_correct'],
            "sft_case_a": stats['sft_case_a'],
            "sft_case_a_failed": stats['sft_case_a_failed'],
            "iterative_scaffolding": stats.get('iterative_scaffolding', {}),
            "sft_data_count": len(final_state.get('sft_data', [])),
            "sft_data_path": final_state.get('sft_path', ''),
            "results_path": final_state.get('results_path', ''),
            "timestamp": datetime.now().isoformat()
        }

        # Save learning results summary to model directory
        summary_path = self.model_dir / f"{self.train_dataset}_train_summary_{self.model_short}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(learning_results, f, ensure_ascii=False, indent=2)

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
    학습 모드 실행 (설계 → Iterative Scaffolding 학습 → SFT 데이터 생성)
    """
    # Create teacher config from CLI argument
    teacher_model_name = args.teacher_model or DEFAULT_TEACHER_MODEL
    student_model_name = args.student_model or DEFAULT_STUDENT_MODEL
    teacher_config = create_teacher_config(teacher_model_name)

    print(f"\n{'=' * 60}")
    print(f"ID-MAS: TRAIN MODE (Iterative Scaffolding Pipeline)")
    print(f"{'=' * 60}")
    print(f"Domain: {args.domain}")
    print(f"Train Dataset: {args.train_dataset}")
    print(f"Student Model: {student_model_name}")
    print(f"Teacher Model: {teacher_model_name}")
    print(f"Resume: {args.resume}")
    if getattr(args, 'use_enhanced_data', False):
        print(f"Enhanced Data: Yes (suffix: {args.enhanced_model_suffix})")

    # Teacher와 Student 모델이 동일한지 확인 (로컬 모델인 경우)
    is_local_teacher = not (
        teacher_model_name.startswith("gpt-") or
        teacher_model_name.startswith("o1") or
        teacher_model_name.startswith("o3")
    )
    if is_local_teacher and teacher_model_name == student_model_name:
        print(f"\n[Model Sharing] Teacher and Student use same model: {teacher_model_name}")
        print(f"[Model Sharing] Model will be loaded ONCE and shared (memory optimized)")

    print(f"{'=' * 60}\n")

    # Initialize pipeline with State Machine
    pipeline = IDMASPipeline(
        domain=args.domain,
        train_dataset=args.train_dataset,
        student_model_name=args.student_model,
        teacher_config=teacher_config,
        resume=args.resume,
        checkpoint_interval=10,
        use_enhanced_data=getattr(args, 'use_enhanced_data', False),
        enhanced_model_suffix=getattr(args, 'enhanced_model_suffix', None)
    )

    if pipeline.instructional_goal:
        print(f"Instructional Goal: {pipeline.instructional_goal}")
    else:
        print(f"Instructional Goal: Not generated yet")
        print(f"  Run with --run-design flag to generate Instructional Goal first.")

    # 1. 교수 설계 단계
    if not args.run_design:
        design_dir = get_design_output_dir(pipeline.domain, pipeline.teacher_model_name)
        design_path = design_dir / f"{pipeline.identifier}_design.json"

        # Check legacy flat structure and migrate if needed
        legacy_path = DATA_DIR / "design_outputs" / f"{pipeline.identifier}_design.json"
        if not design_path.exists() and legacy_path.exists():
            legacy_path.replace(design_path)
            print(f"Moved legacy design file from {legacy_path} to {design_path}")

        if not design_path.exists():
            print(f"Design file not found at {design_path}")
            print("Automatically generating new instructional design...")
            # resume=False이면 Instructional Goal도 새로 생성
            design_result = pipeline.run_design_phase(
                regenerate_instructional_goal=(not args.resume)
            )
        else:
            with open(design_path, 'r', encoding='utf-8') as f:
                design_result = json.load(f)
            print(f"Loaded existing design from {design_path}")
    else:
        # Run design phase
        # resume=False이면 Instructional Goal도 새로 생성
        design_result = pipeline.run_design_phase(
            regenerate_instructional_goal=(not args.resume)
        )

    # 2. Enhanced Data 생성 (design phase 결과 기반)
    # --use-enhanced-data 플래그가 없어도 자동으로 생성 및 사용
    if not getattr(args, 'use_enhanced_data', False):
        # Enhanced data가 이미 존재하는지 확인
        model_suffix = get_model_short_name(pipeline.teacher_model_name)
        enhanced_path = pipeline.raw_data_dir / f"{pipeline.train_dataset}_train_ID-MAS_{model_suffix}.json"

        if enhanced_path.exists() and args.resume:
            print(f"\n[Enhanced Data] Using existing enhanced data: {enhanced_path.name}")
            pipeline.use_enhanced_data = True
            pipeline.enhanced_model_suffix = model_suffix
        else:
            # Enhanced data 생성
            pipeline.generate_enhanced_data(design_result)

    # 3. Iterative Scaffolding 학습 단계
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

    # ModelCache 상태 출력
    loaded_models = ModelCache.get_loaded_models()
    if loaded_models:
        print(f"\n[Model Cache] Loaded models: {len(loaded_models)}")
        for model_name, device in loaded_models:
            print(f"  - {model_name} @ {device}")

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
        description="ID-MAS: Instructional Design Multi-Agent System (Iterative Scaffolding Pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ========================================
  # TRAIN MODE (Iterative Scaffolding Pipeline)
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
    # Get all available training datasets across all domains
    all_train_datasets = sorted(set(ds for datasets in TRAINING_DATASETS.values() for ds in datasets))
    parser.add_argument(
        "--train-dataset",
        type=str,
        choices=all_train_datasets,
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
    parser.add_argument(
        "--use-enhanced-data",
        action="store_true",
        dest="use_enhanced_data",
        help="Use enhanced training data with pre-generated Instructional Goal and Task Analysis. Train mode only."
    )
    parser.add_argument(
        "--enhanced-model-suffix",
        type=str,
        default=None,
        dest="enhanced_model_suffix",
        help="Model suffix for enhanced data file (e.g., 'Qwen2.5-72B-Instruct'). Required when --use-enhanced-data is set."
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
             f"Default: {DEFAULT_TEACHER_MODEL}"
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

        # Validate --use-enhanced-data requires --enhanced-model-suffix
        if args.use_enhanced_data and not args.enhanced_model_suffix:
            parser.error("--enhanced-model-suffix is required when using --use-enhanced-data")

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
