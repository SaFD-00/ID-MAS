"""ID-MAS 메인 실행 모듈.

이 모듈은 ID-MAS(Instructional Design Multi-Agent System) 파이프라인의
메인 진입점으로, 학습(train)과 평가(eval) 모드를 제공합니다.

LangGraph 기반 Iterative Scaffolding Pipeline을 사용하여:
- 교사 모델의 가이드 하에 학생 모델의 반복적 응답 개선 (최대 5회)
- Performance Objectives 기반 평가와 소크라테스식 질문 생성
- Case A: Independent Performance Mastery(PO 충족)/Case B: Scaffolded & Coached Mastery(스캐폴딩 후 PO 충족)/Case C: Teacher Modeling Distillation(교사 시범) SFT 데이터 생성

주요 클래스:
    IDMASPipeline: 학습 모드 파이프라인 (설계 → 학습 → SFT 데이터 생성)
    IDMASEvaluator: 평가 모드 실행기 (Baseline, SFT, SFT_ID-MAS)

주요 함수:
    run_train_mode: 학습 모드 실행
    run_eval_mode: 평가 모드 실행
    main: CLI 인터페이스 진입점

사용 예시:
    # 학습 모드
    python main.py --mode train --domain math --train-dataset gsm8k

    # 평가 모드
    python main.py --mode eval --method baseline --domain math --eval-dataset gsm8k
"""
import sys
import os
import json
import re
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List


# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from design_modules.analysis import InstructionalAnalysis
from design_modules.objectives import PerformanceObjectives
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
    get_design_output_dir, normalize_gpu_ids
)
from models.model_cache import ModelCache


class IDMASPipeline:
    """ID-MAS 학습 파이프라인 클래스.

    LangGraph StateGraph 기반의 Iterative Scaffolding 파이프라인을 구현합니다.
    각 학습 데이터셋(GSM8K, MATH 등)은 고유한 Instructional Goal을 가지며
    별도로 학습됩니다.

    Attributes:
        domain: 도메인명 (예: "math", "logical", "commonsense")
        train_dataset: 학습 데이터셋명 (예: "gsm8k", "math")
        student_model_name: 학생 모델 이름
        teacher_config: 교사 모델 설정 딕셔너리
        instructional_goal: 데이터셋별 학습 목표
        loader: 도메인 데이터 로더
        answer_extractor: 답변 추출기
        graph_runner: LangGraph 파이프라인 실행기

    Example:
        >>> pipeline = IDMASPipeline(
        ...     domain="math",
        ...     train_dataset="gsm8k",
        ...     student_model_name="Qwen/Qwen3-1.7B"
        ... )
        >>> design_result = pipeline.run_design_phase()
        >>> learning_result = pipeline.run_learning_phase(design_result)
    """

    def __init__(
        self,
        domain: str,
        train_dataset: str,
        student_model_name: Optional[str] = None,
        teacher_config: Optional[Dict] = None,
        resume: bool = False,
        checkpoint_interval: int = 10,
        student_gpu_ids=None,
        max_iterations: int = 5
    ):
        """IDMASPipeline 인스턴스를 초기화합니다.

        Args:
            domain: 도메인명 (예: "math", "logical", "commonsense")
            train_dataset: 학습 데이터셋명 (예: "gsm8k", "math")
            student_model_name: 학생 모델명. None이면 기본 모델 사용
            teacher_config: 교사 모델 설정. None이면 기본 OpenAI 모델 사용
            resume: 체크포인트에서 이어서 학습할지 여부
            checkpoint_interval: 체크포인트 저장 간격 (문제 수 기준)
            student_gpu_ids: Student GPU 인덱스 tuple (예: (0,), (0,1,2)).
                None이면 자동 할당.
            max_iterations: Iterative Scaffolding 최대 반복 횟수 (기본값: 5)
        """
        self.domain = domain.lower()
        self.train_dataset = train_dataset.lower()
        self.student_model_name = student_model_name or DEFAULT_STUDENT_MODEL
        self.teacher_config = teacher_config
        self.checkpoint_interval = checkpoint_interval
        self.max_iterations = max_iterations
        self.resume = resume

        # 설계 파일용 식별자 (domain_dataset 형식)
        self.identifier = f"{self.domain}_{self.train_dataset}"

        # 파일명용 모델 약칭 (예: "Qwen3-4B")
        self.model_short = get_model_short_name(self.student_model_name)

        # 교사 모델명 (Instructional Goal 로딩에 필요하므로 먼저 정의)
        self.teacher_model_name = teacher_config.get('model', DEFAULT_TEACHER_MODEL) if teacher_config else DEFAULT_TEACHER_MODEL

        # 설계 JSON에서 Instructional Goal 로드 (미생성 시 None)
        self.instructional_goal = get_instructional_goal(self.train_dataset, teacher_model=self.teacher_model_name)

        # 도메인 데이터 로더 초기화
        self.loader = DomainLoader(domain)

        # 도메인 기반 답변 추출기 초기화
        self.answer_extractor = get_extractor(self.loader.answer_type)

        # 교수설계 모듈 초기화 (교사 모델 사용)
        self.instructional_goal_gen = InstructionalGoalGenerator(teacher_config)
        self.analysis = InstructionalAnalysis(teacher_config)
        self.objectives = PerformanceObjectives(teacher_config)

        # 모델별 디렉토리 설정 (data/{domain}/train/{teacher}/{student}/)
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
        self.enhanced_data_dir = self.model_dirs["enhanced_data_dir"]

        # 파이프라인용 학생/교사 모델 초기화
        self.student_model = StudentModel(model_name=self.student_model_name, gpu_ids=student_gpu_ids)
        self.teacher_model = TeacherModel(teacher_config)

        # LangGraph 파이프라인 실행기 초기화
        self.graph_runner = IDMASGraphRunner(
            student_model=self.student_model,
            teacher_model=self.teacher_model,
            answer_extractor=self.answer_extractor,
        )

    def run_design_phase(
        self,
        learning_objective: Optional[str] = None,
        regenerate_instructional_goal: bool = False
    ) -> Dict:
        """교수설계 단계를 실행합니다.

        Dick & Carey 모델 기반의 5단계 교수설계를 수행합니다:
        - Step 0: Instructional Goal 생성 (샘플 기반)
        - Step 1: 학습 목표 설정
        - Step 2: 교수 분석 (서브스킬 추출)
        - Step 3: 수행목표 진술
        - Step 4: 루브릭 개발

        Args:
            learning_objective: 커스텀 학습 목표. None이면 자동 생성된
                Instructional Goal 사용
            regenerate_instructional_goal: True면 기존 Goal이 있어도 재생성

        Returns:
            설계 결과 딕셔너리. 키:
            - domain: 도메인명
            - train_dataset: 데이터셋명
            - instructional_goal: 생성된 학습 목표
            - instructional_analysis: 교수 분석 결과
            - performance_objectives: 수행목표 목록
            - timestamp: 생성 시각

        Raises:
            RuntimeError: 교사 모델 호출 실패 시 (3회 재시도 후)
            SystemExit: 치명적 오류 발생 시 프로그램 종료
        """
        print("\n" + "=" * 60)
        print("INSTRUCTIONAL DESIGN PHASE")
        print("=" * 60)

        # [Step 0] Instructional Goal 생성
        instructional_goal_metadata = None
        samples_path = self.raw_data_dir / f"{self.train_dataset}_samples.json"

        if samples_path.exists() and (regenerate_instructional_goal or self.instructional_goal is None):
            print(f"\n[Step 0] Instructional Goal Generation")

            with open(samples_path, 'r', encoding='utf-8') as f:
                train_samples = json.load(f)
            print(f"  Loaded {len(train_samples)} samples from {samples_path.name}")

            # 교사 모델로 Instructional Goal 생성 (최대 3회 재시도)
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
                # 3회 재시도 실패 시 프로그램 종료
                print(f"\n[FATAL] {e}")
                print(f"Please check:")
                print(f"  1. Teacher model availability")
                print(f"  2. Sample data quality ({samples_path.name})")
                print(f"  3. Network connection (if using API)")
                sys.exit(1)

            except Exception as e:
                # 예상치 못한 예외 발생 시 프로그램 종료
                print(f"\n[FATAL] Unexpected error during Instructional Goal generation: {e}")
                sys.exit(1)

        elif not samples_path.exists():
            print(f"\n[Step 0] Instructional Goal Generation (SKIPPED)")
            print(f"  Samples file not found: {samples_path.name}")
            print(f"  Run 'python -m utils.sample_extractor' to generate samples.")
            if not self.instructional_goal:
                print(f"  Instructional Goal: Not generated yet")
        else:
            # 기존 설계 JSON에서 로드된 경우
            if self.instructional_goal:
                print(f"\n[Step 0] Instructional Goal (loaded from design JSON)")
                print(f"  {self.instructional_goal[:80]}...")

        # [Step 1] 학습 목표 설정
        if learning_objective is None:
            if not self.instructional_goal:
                print(f"\n[FATAL] Instructional Goal not found for {self.train_dataset}.")
                print(f"Run design phase with --run-design flag first.")
                sys.exit(1)
            learning_objective = self.instructional_goal

        print(f"\n[Step 1] Learning Objective (Instructional Goal for {self.train_dataset.upper()})")
        print(f"  {learning_objective}")

        # [Step 2-4] 교수설계 단계 실행 (RuntimeError 처리 포함)
        try:
            # Step 2: 교수 분석
            print(f"\n[Step 2] Instructional Analysis")
            analysis_result = self.analysis.analyze(learning_objective, max_retries=3)
            print(f"  Analysis completed")

            # Step 3: 수행목표 진술
            print(f"\n[Step 3] Performance Objectives")
            objectives_result = self.objectives.generate_objectives(
                analysis_result["raw_output"],
                max_retries=3
            )
            print(f"  Generated {len(objectives_result['performance_objectives'])} objectives")

        except RuntimeError as e:
            print(f"\n[FATAL] Design Phase failed: {e}")
            print(f"Please check:")
            print(f"  1. Teacher model availability")
            print(f"  2. Network connection (if using API)")
            sys.exit(1)

        # 설계 결과 구성
        design_result = {
            "domain": self.domain,
            "train_dataset": self.train_dataset,
            "identifier": self.identifier,
            "instructional_goal": self.instructional_goal,
            "instructional_goal_metadata": instructional_goal_metadata,
            "learning_objective": learning_objective,
            "instructional_analysis": analysis_result,
            "performance_objectives": objectives_result,
            "timestamp": datetime.now().isoformat()
        }

        # 설계 결과 JSON 파일로 저장
        output_dir = get_design_output_dir(self.domain, self.teacher_model_name)
        output_path = output_dir / f"{self.identifier}_design.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(design_result, f, ensure_ascii=False, indent=2)

        print(f"\n Design phase completed")
        print(f"  Saved to: {output_path}")

        return design_result

    def generate_enhanced_data(self, design_result: Dict) -> Path:
        """설계 결과 기반 Enhanced Training Data를 생성합니다.

        원본 학습 데이터의 instruction은 그대로 유지하고,
        metadata에 Instructional Goal과 Task Analysis를 추가합니다.

        출력 필드:
        - instruction: 원본 유지
        - input: 원본 유지
        - output: 원본 유지
        - metadata: 원본 + instructional_goal, task_analysis 추가

        Args:
            design_result: run_design_phase()에서 반환된 설계 결과

        Returns:
            생성된 enhanced data 파일 경로 (Path 객체)

        Raises:
            ValueError: Instructional Goal이 설계 결과에 없을 때
            FileNotFoundError: 원본 학습 데이터 파일이 없을 때
        """
        print("\n" + "=" * 60)
        print("ENHANCED DATA GENERATION")
        print("=" * 60)

        # 1. 설계 결과에서 학습목표와 과제분석 추출
        instructional_goal = design_result.get("instructional_goal", "")
        task_analysis = design_result.get("instructional_analysis", {}).get("raw_output", "")

        if not instructional_goal:
            raise ValueError("Instructional Goal not found in design result")

        print(f"\n[Step 1] Using Instructional Goal")
        print(f"  {instructional_goal[:80]}...")

        print(f"\n[Step 2] Using Task Analysis")
        print(f"  {len(task_analysis)} characters")

        # 2. 원본 학습 데이터 로드
        source_path = self.raw_data_dir / f"{self.train_dataset}_train.json"
        if not source_path.exists():
            raise FileNotFoundError(f"Training data not found: {source_path}")

        with open(source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"\n[Step 3] Loaded {len(data)} records from {source_path.name}")

        # 3. instruction은 원본 유지, metadata에 설계 정보 추가
        print(f"\n[Step 4] Enhancing instructions...")
        enhanced_data = []
        for item in data:
            new_item = {
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", ""),
                "metadata": {
                    **item.get("metadata", {}),
                    "instructional_goal": instructional_goal,
                    "task_analysis": task_analysis,
                }
            }

            enhanced_data.append(new_item)

        # 4. 파일 저장
        self.enhanced_data_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.enhanced_data_dir / f"{self.train_dataset}_train_ID-MAS.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, ensure_ascii=False, indent=2)

        print(f"\n[Step 5] Saved enhanced data")
        print(f"  Path: {output_path}")
        print(f"  Records: {len(enhanced_data)}")

        print(f"\n Enhanced data generation completed")

        return output_path

    def run_learning_phase(
        self,
        design_result: Dict,
        num_questions: Optional[int] = None,
        resume: bool = False
    ) -> Dict:
        """LangGraph 기반 Iterative Scaffolding 학습을 실행합니다.

        6단계 파이프라인:
        1. 학생 모델 초기 응답 생성
        2. 교사 모델 PO 평가 (소크라테스식 질문)
        3. 스캐폴딩 아티팩트 생성
        4. 학생 재응답
        5. 응답 재구성 (Case A: Independent Performance Mastery / Case B: Scaffolded & Coached Mastery / Case C: Teacher Modeling Distillation)
        6. SFT 데이터 생성

        Args:
            design_result: run_design_phase()에서 반환된 설계 결과
            num_questions: 학습할 문제 수. None이면 전체
            resume: True면 기존 로그에서 이어서 학습

        Returns:
            학습 결과 딕셔너리. 키:
            - total_questions: 전체 문제 수
            - scaffolding_processed: 스캐폴딩 처리된 문제 수
            - scaffolding_correct: 정답 처리된 문제 수
            - case_a_independent_performance_mastery: Case A: Independent Performance Mastery SFT 데이터 수
            - sft_data_count: 총 SFT 데이터 수
            - sft_data_path: SFT 데이터 저장 경로
        """
        # Enhanced 학습 데이터 로드
        print(f"\n[Enhanced Data] Loading enhanced training data...")
        print(f"  Enhanced data dir: {self.enhanced_data_dir}")
        questions = self.loader.load_enhanced_training_data(
            dataset=self.train_dataset,
            enhanced_data_dir=self.enhanced_data_dir,
            limit=num_questions,
            shuffle=False
        )
        print(f"\nLoaded {len(questions)} enhanced training questions")

        # 질문 데이터 준비 (새 필드명: id, instruction, input, output)
        question_data = []
        for q in questions:
            question_data.append({
                'id': q.question_id,
                'instruction': q.metadata.get('instruction', ''),
                'input': q.question,
                'output': q.metadata.get('full_output', q.ground_truth_formatted),
            })

        # LangGraph 파이프라인 실행
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
            max_iterations=self.max_iterations,
            resume=resume,
        )

        # 통계 추출
        stats = get_statistics(final_state)

        case_stats = stats.get('case_statistics', {})
        learning_results = {
            "domain": self.domain,
            "train_dataset": self.train_dataset,
            "instructional_goal": self.instructional_goal,
            "total_questions": stats['total_questions'],
            "scaffolding_processed": stats['scaffolding_processed'],
            "scaffolding_correct": case_stats.get('success_total', 0),
            "case_a_independent_performance_mastery": case_stats.get('case_a_independent_performance_mastery', 0),
            "case_c_teacher_modeling_distillation": case_stats.get('case_c_teacher_modeling_distillation', 0),
            "case_statistics": case_stats,
            "sft_data_count": len(final_state.get('sft_data', [])),
            "sft_data_path": final_state.get('sft_path', ''),
            "results_path": final_state.get('results_path', ''),
            "timestamp": datetime.now().isoformat()
        }

        # 학습 결과 요약 저장
        summary_path = self.model_dir / f"{self.train_dataset}_train_summary_{self.model_short}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(learning_results, f, ensure_ascii=False, indent=2)

        return learning_results


class IDMASEvaluator:
    """ID-MAS 평가 전용 클래스.

    세 가지 평가 방법을 지원합니다:
    - baseline: 파인튜닝 없는 기본 모델 평가
    - sft: SFT 파인튜닝 모델 평가 (HuggingFace Hub)
    - sft_id-mas: ID-MAS 방식 SFT 모델 평가 (HuggingFace Hub)

    Attributes:
        domain: 도메인명
        eval_dataset: 평가 데이터셋명
        student_model_name: 평가할 학생 모델명
        eval_method: 평가 방법
        loader: 도메인 데이터 로더
        answer_extractor: 답변 추출기
        student_model: 초기화된 학생 모델

    Example:
        >>> evaluator = IDMASEvaluator(
        ...     domain="math",
        ...     eval_dataset="gsm8k",
        ...     eval_method="sft_id-mas"
        ... )
        >>> result = evaluator.evaluate()
        >>> print(f"Accuracy: {result['accuracy']:.2%}")
    """

    def __init__(
        self,
        domain: str,
        eval_dataset: str,
        student_model_name: Optional[str] = None,
        eval_method: str = "baseline",
        student_gpu_ids=None
    ):
        """IDMASEvaluator 인스턴스를 초기화합니다.

        Args:
            domain: 도메인명 (예: "math", "logical", "commonsense")
            eval_dataset: 평가 데이터셋명 (예: "gsm8k", "svamp")
            student_model_name: 학생 모델명. None이면 기본 모델 사용
            eval_method: 평가 방법. "baseline", "sft", "sft_id-mas" 중 하나
            student_gpu_ids: Student GPU 인덱스 tuple (예: (0,), (0,1,2)).
                None이면 자동 할당.
        """
        self.domain = domain.lower()
        self.eval_dataset = eval_dataset.lower()
        self.student_model_name = student_model_name or DEFAULT_STUDENT_MODEL
        self.eval_method = eval_method

        # 도메인 데이터 로더 초기화
        self.loader = DomainLoader(domain)

        # 도메인 기반 답변 추출기 초기화
        self.answer_extractor = get_extractor(self.loader.answer_type)

        # 평가 방법에 따른 학생 모델 초기화
        use_sft = (eval_method == "sft")
        use_sft_idmas = (eval_method == "sft_id-mas")

        self.student_model = StudentModel(
            model_name=self.student_model_name,
            use_sft_model=use_sft,
            use_sft_idmas_model=use_sft_idmas,
            sft_domain=domain,
            gpu_ids=student_gpu_ids,
        )

        # 파일명용 모델 약칭
        from config.models import get_model_short_name
        self.model_short = get_model_short_name(self.student_model_name)

        # 평가 결과 디렉토리 (data/{domain}/eval/{Model}/)
        self.model_dirs = get_domain_data_dirs(domain, self.student_model_name, self.eval_dataset, mode="eval")
        self.eval_results_dir = self.model_dirs["model_dir"]

    def evaluate(
        self,
        num_questions: Optional[int] = None,
        resume: bool = True
    ) -> Dict:
        """평가를 실행합니다.

        각 문제에 대해 학생 모델이 1회 응답을 생성하고,
        답변 추출기로 정답 여부를 판단합니다.

        Args:
            num_questions: 평가할 문제 수. None이면 전체
            resume: True면 기존 결과 파일에서 이어서 평가

        Returns:
            평가 결과 딕셔너리. 키:
            - domain: 도메인명
            - eval_dataset: 데이터셋명
            - method: 평가 방법명
            - total_questions: 전체 문제 수
            - correct_count: 정답 수
            - accuracy: 정확도 (0~1)
            - question_results: 문제별 상세 결과 리스트
        """
        # 파일명용 방법명 매핑
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

        # 평가 데이터 로드
        questions = self.loader.load_eval_data(self.eval_dataset, limit=num_questions)

        # 데이터셋에 맞는 답변 추출기 획득
        eval_extractor = self._get_extractor_for_dataset(questions)

        # 평가 결과 초기화
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
        result_path = self.eval_results_dir / f"{self.eval_dataset}_eval_results-{method_name}.json"

        # Resume 모드: 기존 결과에서 완료된 문제 복원
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

        eval_results["question_results"] = restored_results.copy()

        # 문제별 평가 실행 (1회 시도)
        for i, question in enumerate(questions):
            # 이미 처리된 문제는 스킵
            if question.question_id in processed_question_ids:
                continue

            print(f"\n[Eval] Question {i+1}/{len(questions)}")

            # 프롬프트 생성 (instruction → system_message, input → user_message)
            instruction = question.metadata.get('instruction', '')
            problem_input = question.question

            print(f"Question: {question.question[:100]}...")
            ground_truth_clean = re.sub(r'\\boxed\{([^}]+)\}', r'\1', question.ground_truth)
            print(f"Ground Truth: {ground_truth_clean}")

            # 학생 모델 응답 생성
            print("\n[Student] Generating response...")
            student_response = self.student_model.generate_initial_response(
                problem_text=problem_input,
                system_message=instruction if instruction else None
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

            # 문제 결과 저장
            question_result = {
                "question_id": question.question_id,
                "question": question.question,
                "ground_truth": question.ground_truth,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "student_response": student_response
            }

            # 선택지 문제인 경우 추가 필드
            if question.choices:
                question_result["choices"] = question.choices

            eval_results["question_results"].append(question_result)

            # 점진적 저장 (문제마다 저장)
            self._save_eval_results(eval_results, result_path, correct_count)

        # 최종 정확도 계산
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
        """평가 결과를 JSON 파일로 저장합니다.

        Args:
            eval_results: 평가 결과 딕셔너리
            result_path: 저장 경로
            correct_count: 현재까지 정답 수
        """
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
        """데이터셋에 맞는 답변 추출기를 반환합니다.

        Args:
            questions: 질문 데이터 리스트

        Returns:
            해당 데이터셋의 답변 타입에 맞는 추출기
        """
        if questions:
            return get_extractor(questions[0].answer_type)
        return self.answer_extractor


def _parse_gpu_ids(value: str):
    """CLI GPU 인자 문자열을 파싱합니다.

    Args:
        value: GPU 인자 문자열 (예: "0", "0,1,2")

    Returns:
        단일 GPU면 int, 다중 GPU면 tuple.
    """
    parts = [int(p.strip()) for p in value.split(",")]
    if len(parts) == 1:
        return parts[0]
    return tuple(parts)


def _resolve_gpu_allocation(args, student_model_name, teacher_model_name):
    """GPU 할당을 결정합니다.

    다중 GPU tensor parallel을 지원합니다.

    Args:
        args: CLI 인자 (student_gpu, teacher_gpu 속성 포함)
        student_model_name: Student 모델명
        teacher_model_name: Teacher 모델명

    Returns:
        (student_gpu_ids, teacher_gpu_ids) 튜플.
        각각 Optional[Tuple[int, ...]] 타입.
    """
    raw_student = getattr(args, 'student_gpu', None)
    raw_teacher = getattr(args, 'teacher_gpu', None)

    # normalize_gpu_ids로 통일 (None, int, tuple → Optional[Tuple[int, ...]])
    student_gpu_ids = normalize_gpu_ids(raw_student)
    teacher_gpu_ids = normalize_gpu_ids(raw_teacher)

    is_api_teacher = teacher_model_name.startswith("gpt-")
    is_local_teacher = not is_api_teacher

    # 시나리오 1: teacher-gpu 지정 + API teacher → 경고, teacher-gpu 무시
    if teacher_gpu_ids is not None and is_api_teacher:
        print(f"\n[GPU] WARNING: --teacher-gpu={raw_teacher} ignored (teacher '{teacher_model_name}' is an API model)")
        teacher_gpu_ids = None

    # 시나리오 2: 같은 로컬 모델 + 다른 GPU → 경고, student-gpu로 통일 (ModelCache 공유)
    if (is_local_teacher
            and teacher_model_name == student_model_name
            and student_gpu_ids is not None and teacher_gpu_ids is not None
            and student_gpu_ids != teacher_gpu_ids):
        print(f"\n[GPU] WARNING: Same model '{teacher_model_name}' on different GPUs ({student_gpu_ids} vs {teacher_gpu_ids})")
        print(f"[GPU] Using student-gpu={student_gpu_ids} for both (ModelCache sharing)")
        teacher_gpu_ids = student_gpu_ids

    # 시나리오 3: 같은 로컬 모델 + student-gpu만 지정 → teacher-gpu도 동일하게 설정 (ModelCache 공유)
    if (is_local_teacher
            and teacher_model_name == student_model_name
            and student_gpu_ids is not None and teacher_gpu_ids is None):
        print(f"\n[GPU] Same model '{teacher_model_name}': setting teacher-gpu={student_gpu_ids} (ModelCache sharing)")
        teacher_gpu_ids = student_gpu_ids

    # GPU 겹침 검증 (다른 모델이 같은 GPU 사용 시 경고)
    if (is_local_teacher
            and teacher_model_name != student_model_name
            and student_gpu_ids is not None and teacher_gpu_ids is not None):
        student_set = set(student_gpu_ids)
        teacher_set = set(teacher_gpu_ids)
        overlap = student_set & teacher_set
        if overlap:
            print(f"\n[GPU] WARNING: GPU overlap detected {overlap} between student ({student_gpu_ids}) and teacher ({teacher_gpu_ids})")

    # CUDA_VISIBLE_DEVICES 자동 설정
    all_gpu_ids = set()
    if student_gpu_ids is not None:
        all_gpu_ids.update(student_gpu_ids)
    if teacher_gpu_ids is not None:
        all_gpu_ids.update(teacher_gpu_ids)

    if all_gpu_ids and "CUDA_VISIBLE_DEVICES" not in os.environ:
        cuda_visible = ",".join(str(g) for g in sorted(all_gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
        print(f"[GPU] CUDA_VISIBLE_DEVICES set to: {cuda_visible}")

    return student_gpu_ids, teacher_gpu_ids


def run_train_mode(args):
    """학습 모드를 실행합니다.

    설계 → Iterative Scaffolding 학습 → SFT 데이터 생성의
    전체 파이프라인을 실행합니다.

    Args:
        args: argparse로 파싱된 CLI 인자. 필수 속성:
            - domain: 도메인명
            - train_dataset: 학습 데이터셋명
            - student_model: 학생 모델명 (선택)
            - teacher_model: 교사 모델명 (선택)
            - resume: 이어서 학습 여부
            - run_design: 설계 단계 강제 실행 여부
            - max_iterations: Iterative Scaffolding 최대 반복 횟수 (기본값: 5)
            - student_gpu: Student GPU 인덱스 (선택)
            - teacher_gpu: Teacher GPU 인덱스 (선택)
    """
    # CLI 인자에서 설정 생성
    teacher_model_name = args.teacher_model or DEFAULT_TEACHER_MODEL
    student_model_name = args.student_model or DEFAULT_STUDENT_MODEL

    # GPU 할당 결정
    student_gpu_ids, teacher_gpu_ids = _resolve_gpu_allocation(args, student_model_name, teacher_model_name)

    teacher_config = create_teacher_config(teacher_model_name, gpu_ids=teacher_gpu_ids)

    print(f"\n{'=' * 60}")
    print(f"ID-MAS: TRAIN MODE (Iterative Scaffolding Pipeline)")
    print(f"{'=' * 60}")
    print(f"Domain: {args.domain}")
    print(f"Train Dataset: {args.train_dataset}")
    print(f"Student Model: {student_model_name}")
    print(f"Teacher Model: {teacher_model_name}")
    if student_gpu_ids is not None:
        print(f"Student GPU: {student_gpu_ids}")
    if teacher_gpu_ids is not None:
        print(f"Teacher GPU: {teacher_gpu_ids}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"Resume: {args.resume}")

    # 로컬 모델인 경우 Teacher/Student 동일 여부 확인
    is_local_teacher = not teacher_model_name.startswith("gpt-")
    if is_local_teacher and teacher_model_name == student_model_name:
        print(f"\n[Model Sharing] Teacher and Student use same model: {teacher_model_name}")
        print(f"[Model Sharing] Model will be loaded ONCE and shared (memory optimized)")

    print(f"{'=' * 60}\n")

    # 파이프라인 초기화
    pipeline = IDMASPipeline(
        domain=args.domain,
        train_dataset=args.train_dataset,
        student_model_name=args.student_model,
        teacher_config=teacher_config,
        resume=args.resume,
        checkpoint_interval=10,
        student_gpu_ids=student_gpu_ids,
        max_iterations=args.max_iterations,
    )

    if pipeline.instructional_goal:
        print(f"Instructional Goal: {pipeline.instructional_goal}")
    else:
        print(f"Instructional Goal: Not generated yet")
        print(f"  Run with --run-design flag to generate Instructional Goal first.")

    # 1. 교수설계 단계
    if not args.run_design:
        design_dir = get_design_output_dir(pipeline.domain, pipeline.teacher_model_name)
        design_path = design_dir / f"{pipeline.identifier}_design.json"

        # 레거시 플랫 구조 확인 및 마이그레이션
        legacy_path = DATA_DIR / "design_outputs" / f"{pipeline.identifier}_design.json"
        if not design_path.exists() and legacy_path.exists():
            legacy_path.replace(design_path)
            print(f"Moved legacy design file from {legacy_path} to {design_path}")

        if not design_path.exists():
            print(f"Design file not found at {design_path}")
            print("Automatically generating new instructional design...")
            design_result = pipeline.run_design_phase(
                regenerate_instructional_goal=(not args.resume)
            )
        else:
            with open(design_path, 'r', encoding='utf-8') as f:
                design_result = json.load(f)
            print(f"Loaded existing design from {design_path}")
    else:
        # 설계 단계 강제 실행
        design_result = pipeline.run_design_phase(
            regenerate_instructional_goal=(not args.resume)
        )

    # 2. Enhanced Data 확인 및 생성
    enhanced_path = pipeline.enhanced_data_dir / f"{pipeline.train_dataset}_train_ID-MAS.json"

    if enhanced_path.exists() and args.resume:
        print(f"\n[Enhanced Data] Using existing enhanced data: {enhanced_path.name}")
    else:
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

    # 모델 캐시 상태 출력
    loaded_models = ModelCache.get_loaded_models()
    if loaded_models:
        print(f"\n[Model Cache] Loaded models: {len(loaded_models)}")
        for model_name, device, gpu_ids in loaded_models:
            gpu_info = f" (gpu_ids={gpu_ids})" if gpu_ids is not None else ""
            print(f"  - {model_name} @ {device}{gpu_info}")

    print("=" * 60)


def run_eval_mode(args):
    """평가 모드를 실행합니다.

    Baseline, SFT, SFT_ID-MAS 세 가지 방법으로 평가합니다.

    Args:
        args: argparse로 파싱된 CLI 인자. 필수 속성:
            - domain: 도메인명
            - eval_dataset: 평가 데이터셋명
            - method: 평가 방법
            - student_model: 학생 모델명 (선택)
            - eval_resume: 이어서 평가 여부
            - student_gpu: Student GPU 인덱스 (선택)
    """
    raw_student_gpu = getattr(args, 'student_gpu', None)
    student_gpu_ids = normalize_gpu_ids(raw_student_gpu)

    # CUDA_VISIBLE_DEVICES 자동 설정
    if student_gpu_ids is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        cuda_visible = ",".join(str(g) for g in student_gpu_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
        print(f"[GPU] CUDA_VISIBLE_DEVICES set to: {cuda_visible}")

    print(f"\n{'=' * 60}")
    print(f"ID-MAS: EVAL MODE ({args.method.upper()})")
    print(f"{'=' * 60}")

    print(f"Domain: {args.domain}")
    print(f"Eval Dataset: {args.eval_dataset}")
    print(f"Model: {args.student_model or DEFAULT_STUDENT_MODEL}")
    if student_gpu_ids is not None:
        print(f"Student GPU: {student_gpu_ids}")
    print(f"{'=' * 60}\n")

    # 평가기 초기화
    evaluator = IDMASEvaluator(
        domain=args.domain,
        eval_dataset=args.eval_dataset,
        student_model_name=args.student_model,
        eval_method=args.method,
        student_gpu_ids=student_gpu_ids,
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
    """CLI 진입점 함수.

    argparse를 사용하여 명령줄 인자를 파싱하고
    적절한 모드(train/eval)를 실행합니다.
    """
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
      --student-model Qwen/Qwen3-4B

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
      --student-model Qwen/Qwen3-1.7B

  # ========================================
  # EVAL MODE - SFT_ID-MAS (ID-MAS Fine-tuned)
  # ========================================

  # GSM8K SFT_ID-MAS 평가
  python main.py --mode eval --method sft_id-mas \\
      --domain math --eval-dataset gsm8k \\
      --student-model Qwen/Qwen3-1.7B

  # Cross-dataset SFT_ID-MAS 평가
  python main.py --mode eval --method sft_id-mas \\
      --domain math --eval-dataset svamp \\
      --student-model Qwen/Qwen3-4B

  # ========================================
  # GPU 할당 (단일/다중 GPU)
  # ========================================

  # 단일 GPU 할당
  python main.py --mode train --domain math --train-dataset gsm8k \\
      --student-model Qwen/Qwen3-4B --teacher-model Qwen/Qwen3-32B \\
      --student-gpu 0 --teacher-gpu 1

  # 다중 GPU tensor parallel (Qwen3-32B 등 대형 모델)
  python main.py --mode train --domain math --train-dataset gsm8k \\
      --student-model Qwen/Qwen3-32B --teacher-model gpt-5.2 \\
      --student-gpu 0,1,2

  # Student/Teacher 모두 다중 GPU
  python main.py --mode train --domain math --train-dataset gsm8k \\
      --student-model Qwen/Qwen3-8B --teacher-model Qwen/Qwen3-32B \\
      --student-gpu 0,1 --teacher-gpu 2,3,4,5

  # API Teacher + Student GPU 지정
  python main.py --mode train --domain math --train-dataset gsm8k \\
      --student-model Qwen/Qwen3-8B --teacher-model gpt-5.2 \\
      --student-gpu 0
        """
    )

    # 모드 선택 (필수)
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "eval"],
        help="실행 모드: 'train'(학습) 또는 'eval'(평가)"
    )

    # ========================================
    # 학습 모드 옵션
    # ========================================
    parser.add_argument(
        "--domain",
        type=str,
        choices=get_available_domains(),
        help="도메인명 (예: 'math'). 모든 모드에서 필수"
    )

    # 전체 도메인의 학습 데이터셋 수집
    all_train_datasets = sorted(set(ds for datasets in TRAINING_DATASETS.values() for ds in datasets))
    parser.add_argument(
        "--train-dataset",
        type=str,
        choices=all_train_datasets,
        help="학습 데이터셋. 학습 모드에서 필수"
    )
    parser.add_argument(
        "--run-design",
        action="store_true",
        help="설계 단계 실행 (기존 설계가 있어도 새로 생성). 학습 모드 전용"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="True",
        choices=["True", "False"],
        help="체크포인트에서 이어서 학습. 학습 모드 전용 (기본값: True)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        dest="max_iterations",
        help="Iterative Scaffolding 최대 반복 횟수. 학습 모드 전용 (기본값: 5)"
    )

    # ========================================
    # 평가 모드 옵션
    # ========================================
    parser.add_argument(
        "--method",
        type=str,
        choices=["baseline", "sft", "sft_id-mas"],
        help="평가 방법. 평가 모드에서 필수"
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        help="평가 데이터셋. 평가 모드에서 필수"
    )
    parser.add_argument(
        "--eval-resume",
        type=str,
        default="True",
        choices=["True", "False"],
        help="기존 결과에서 이어서 평가. 평가 모드 전용 (기본값: True)"
    )

    # ========================================
    # 공통 옵션
    # ========================================
    parser.add_argument(
        "--student-model",
        type=str,
        default=None,
        choices=AVAILABLE_STUDENT_MODELS,
        dest="student_model",
        help=f"학생 모델. 기본값: {DEFAULT_STUDENT_MODEL}"
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=None,
        choices=AVAILABLE_TEACHER_MODELS,
        dest="teacher_model",
        help=f"교사 모델 (설계 및 평가용). 기본값: {DEFAULT_TEACHER_MODEL}"
    )

    # ========================================
    # GPU 할당 옵션
    # ========================================
    parser.add_argument(
        "--student-gpu",
        type=_parse_gpu_ids,
        default=None,
        dest="student_gpu",
        help="Student 모델 GPU (예: 0, '0,1,2'). 다중 GPU는 쉼표로 구분. 미지정 시 CUDA_VISIBLE_DEVICES 사용"
    )
    parser.add_argument(
        "--teacher-gpu",
        type=_parse_gpu_ids,
        default=None,
        dest="teacher_gpu",
        help="Teacher 모델 GPU (예: 1, '3,4,5,6'). 다중 GPU는 쉼표로 구분. 미지정 시 CUDA_VISIBLE_DEVICES 사용. API 모델은 무시"
    )

    args = parser.parse_args()

    # ========================================
    # 인자 검증
    # ========================================

    if args.mode == "train":
        # 학습 모드 검증
        if not args.domain:
            parser.error("--domain은 학습 모드에서 필수입니다")
        if not args.train_dataset:
            parser.error("--train-dataset은 학습 모드에서 필수입니다")

        # 선택한 도메인에서 학습 데이터셋 유효성 검증
        available_train = get_training_datasets_for_domain(args.domain)
        if args.train_dataset not in available_train:
            parser.error(
                f"--train-dataset '{args.train_dataset}'은(는) {args.domain} 도메인에서 "
                f"사용할 수 없습니다. 가능한 데이터셋: {available_train}"
            )

        # 학습 모드에서 사용할 수 없는 옵션 검증
        if args.method:
            parser.error("--method는 학습 모드에서 사용할 수 없습니다")
        if args.eval_dataset:
            parser.error("--eval-dataset은 학습 모드에서 사용할 수 없습니다")

        # --resume 문자열을 boolean으로 변환
        args.resume = args.resume == "True"

        # 학습 모드 실행
        run_train_mode(args)

    elif args.mode == "eval":
        # 평가 모드 검증
        if not args.method:
            parser.error("--method는 평가 모드에서 필수입니다")
        if not args.eval_dataset:
            parser.error("--eval-dataset은 평가 모드에서 필수입니다")
        if not args.domain:
            parser.error("--domain은 평가 모드에서 필수입니다")

        # 평가 모드에서 사용할 수 없는 옵션 검증
        if args.train_dataset:
            parser.error("--train-dataset은 평가 모드에서 사용할 수 없습니다")
        if args.run_design:
            parser.error("--run-design은 평가 모드에서 사용할 수 없습니다")

        # 선택한 도메인에서 평가 데이터셋 유효성 검증
        available_eval = get_eval_datasets_for_domain(args.domain)
        if args.eval_dataset not in available_eval:
            parser.error(
                f"--eval-dataset '{args.eval_dataset}'은(는) {args.domain} 도메인에서 "
                f"사용할 수 없습니다. 가능한 데이터셋: {available_eval}"
            )

        # SFT/SFT_ID-MAS 모델 지원 여부 검증
        if args.method in ["sft", "sft_id-mas"]:
            model_to_check = args.student_model or DEFAULT_STUDENT_MODEL
            if model_to_check not in MODEL_NAME_TO_SHORT:
                parser.error(
                    f"모델 '{model_to_check}'은(는) {args.method.upper()} 평가를 지원하지 않습니다.\n"
                    f"지원 모델: {list(MODEL_NAME_TO_SHORT.keys())}"
                )

        # --eval-resume 문자열을 boolean으로 변환
        args.eval_resume = args.eval_resume == "True"

        # 평가 모드 실행
        run_eval_mode(args)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
