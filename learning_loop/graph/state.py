"""LangGraph 상태 스키마 모듈.

ID-MAS Iterative Scaffolding Pipeline의 상태 구조를 정의합니다.

주요 클래스:
    IDMASState: 그래프 메인 상태 스키마
    QuestionResult: 개별 문제 처리 결과
    DesignResult: 교수설계 출력 결과
    SFTCase: SFT 데이터 케이스 분류 (A/B/C)
    PipelineStep: 파이프라인 단계 상수

주요 함수:
    create_initial_state: 초기 상태 생성
    get_statistics: 파이프라인 통계 조회
    load_checkpoint_from_logs: 로그에서 체크포인트 로드
    restore_state_from_checkpoint: 체크포인트에서 상태 복원

사용 예시:
    >>> from learning_loop.graph.state import IDMASState, create_initial_state
    >>> state = create_initial_state(domain="math", train_dataset="gsm8k", ...)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, Optional, List, Dict, Any, Set, Tuple
from datetime import datetime
from enum import Enum


class SFTCase(str, Enum):
    """스캐폴딩 결과에 따른 SFT 데이터 케이스 분류.

    Attributes:
        INDEPENDENT_PERFORMANCE_MASTERY: Case A: Independent Performance Mastery — 독립적 수행 숙달 (1회차 PO 충족)
        SCAFFOLDED_COACHED_MASTERY: Case B: Scaffolded & Coached Mastery — 스캐폴딩 기반 숙달 (2~5회차 PO 충족)
        TEACHER_MODELING_DISTILLATION: Case C: Teacher Modeling Distillation — 교사 모델링 증류 (최대 반복 후 교사 시범)
    """
    INDEPENDENT_PERFORMANCE_MASTERY = "case_a_independent_performance_mastery"
    SCAFFOLDED_COACHED_MASTERY = "case_b_scaffolded_coached_mastery"
    TEACHER_MODELING_DISTILLATION = "case_c_teacher_modeling_distillation"
    SKIPPED = "skipped"  # 처리 실패로 건너뛴 문제


class PipelineStep:
    """ID-MAS Iterative Scaffolding Pipeline 단계 상수.

    파이프라인 흐름:
        Step 1: Initial Response (초기 응답 생성) - 1회만 실행
        Step 2: PO Evaluation (Performance Objectives 평가) - 반복
        Step 3: Scaffolding (스캐폴딩 아티팩트 생성) - PO 미충족 시
        Step 4: Re-response (학생 재응답) - iteration 2~5
        Step 5: Reconstruction (재구성) - Case A: Independent Performance Mastery / Case B: Scaffolded & Coached Mastery / Case C: Teacher Modeling Distillation
        Step 6: SFT Generation (SFT 데이터 생성)

    Attributes:
        STEP1~STEP6: 단계 식별자
        STEP1_NAME~STEP6_NAME: 로깅용 단계 이름
        STEP1_KEY~STEP6_KEY: 결과 딕셔너리용 전체 키
        ALL_STEPS: 모든 단계 리스트
    """
    # Step identifiers
    STEP1 = "step1"  # Initial Response
    STEP2 = "step2"  # PO Evaluation
    STEP3 = "step3"  # Scaffolding Artifact
    STEP4 = "step4"  # Student Re-response
    STEP5 = "step5"  # Reconstruction
    STEP6 = "step6"  # SFT Generation

    # Step names for logging
    STEP1_NAME = "initial_response"
    STEP2_NAME = "evaluation"
    STEP3_NAME = "scaffolding"
    STEP4_NAME = "reresponse"
    STEP5_NAME = "reconstruction"
    STEP6_NAME = "sft_generation"

    # Full step keys (for result/log dictionaries)
    STEP1_KEY = "step1_initial_response"
    STEP2_KEY = "step2_evaluation"
    STEP3_KEY = "step3_scaffolding"
    STEP4_KEY = "step4_reresponse"
    STEP5_KEY = "step5_reconstruction"
    STEP6_KEY = "step6_sft_generation"

    # All steps for iteration
    ALL_STEPS = [STEP1, STEP2, STEP3, STEP4, STEP5, STEP6]

    @classmethod
    def get_step_name(cls, step: str) -> str:
        """단계의 읽기 쉬운 이름을 반환합니다.

        Args:
            step: 단계 식별자

        Returns:
            단계 이름 (예: "evaluation")
        """
        names = {
            cls.STEP1: cls.STEP1_NAME,
            cls.STEP2: cls.STEP2_NAME,
            cls.STEP3: cls.STEP3_NAME,
            cls.STEP4: cls.STEP4_NAME,
            cls.STEP5: cls.STEP5_NAME,
            cls.STEP6: cls.STEP6_NAME,
        }
        return names.get(step, step)


class QuestionResultRequired(TypedDict):
    """QuestionResult 필수 필드 정의.

    Attributes:
        id: 문제 고유 ID
        instruction: 지시문 (데이터셋별 instruction)
        input: 문제 텍스트 (순수 input)
        output: 정답 (ground truth)
    """
    id: str
    instruction: str
    input: str
    output: str


class QuestionResult(QuestionResultRequired, total=False):
    """파이프라인을 통해 처리된 단일 문제의 결과.

    QuestionResultRequired의 필수 필드를 상속받고
    선택적 필드들을 추가로 정의합니다.
    """
    # 스캐폴딩 결과
    initial_response: str
    predicted_answer: Optional[str]
    scaffolding_correct: bool  # renamed from phase1_correct

    # Final SFT classification
    sft_case: Optional[str]
    sft_response: Optional[str]

    # Iterative scaffolding details
    iterative_scaffolding: Optional[Dict[str, Any]]
    reconstruction: Optional[Dict[str, Any]]

    # NEW: Scaffolding Artifact fields
    scaffolding_artifacts: Optional[List[Dict[str, Any]]]  # 누적된 Scaffolding Artifacts
    artifact_references: Optional[List[str]]  # Student가 참조한 Artifact 정보 목록
    hot_count: Optional[int]  # HOT (High-Order Thinking) scaffolding count
    lot_count: Optional[int]  # LOT (Low-Order Thinking) scaffolding count

    # Skip 정보
    skip_reason: Optional[str]  # skip 사유 (에러 메시지)



class DesignResult(TypedDict, total=False):
    """교수설계 출력 결과.

    Attributes:
        domain: 도메인 (math, logical, commonsense)
        train_dataset: 훈련 데이터셋 이름
        identifier: 고유 식별자
        instructional_goal: 학습 목표
        learning_objective: 학습 목표 상세
        instructional_analysis: 교수 분석 결과
        performance_objectives: 수행목표 리스트
        timestamp: 생성 시간
    """
    domain: str
    train_dataset: str
    identifier: str
    instructional_goal: str
    learning_objective: str
    instructional_analysis: Dict[str, Any]
    performance_objectives: Dict[str, Any]
    timestamp: str


class IDMASState(TypedDict, total=False):
    """ID-MAS LangGraph 파이프라인 메인 상태 스키마.

    모든 노드를 통과하며 학습 파이프라인 실행의 전체 컨텍스트를 유지합니다.

    상태 구조는 Iterative Scaffolding 아키텍처를 따릅니다:
        1. Instructional Design Phase (선택적, 기존 결과 로드 가능)
        2. Scaffolding - 교사 가이드를 통한 반복적 응답 생성

    섹션별 필드:
        - Configuration: 설정 정보
        - Design Phase: 교수설계 결과
        - Questions: 처리할 문제 목록
        - Scaffolding Results: 스캐폴딩 결과 및 통계
        - SFT Data: 생성된 SFT 데이터
        - Pipeline Control: 파이프라인 제어
        - Timestamps: 시간 정보
        - Checkpoint: 체크포인트 정보
    """

    # ==================== Configuration ====================
    domain: str
    train_dataset: str
    instructional_goal: str
    student_model_name: str
    teacher_model_name: str
    model_short: str
    checkpoint_interval: int
    use_iterative_scaffolding: bool
    max_iterations: int

    # ==================== Design Phase ====================
    design_result: Optional[DesignResult]
    task_analysis: str
    performance_objectives: List[Dict[str, Any]]

    # ==================== Questions ====================
    questions: List[Dict[str, Any]]
    total_questions: int
    current_question_index: int
    current_question: Optional[Dict[str, Any]]

    # ==================== Scaffolding Results ====================
    scaffolding_results: List[QuestionResult]
    scaffolding_processed: int
    scaffolding_correct_count: int

    # Iterative scaffolding statistics
    case_a_independent_performance_mastery_count: int  # Case A: Independent Performance Mastery — 독립적 수행 숙달 (1회차 PO 충족)
    case_b_scaffolded_coached_mastery_count: int  # Case B: Scaffolded & Coached Mastery — 스캐폴딩 기반 숙달 (2~5회차 PO 충족)
    case_c_teacher_modeling_distillation_count: int  # Case C: Teacher Modeling Distillation — 교사 모델링 증류 (최대 반복 후 교사 시범)

    # ==================== Scaffolding Artifact Statistics ====================
    hot_scaffolding_count: int  # HOT (High-Order Thinking) 스캐폴딩 생성 횟수
    lot_scaffolding_count: int  # LOT (Low-Order Thinking) 스캐폴딩 생성 횟수
    skipped_count: int  # 처리 실패로 건너뛴 문제 수

    # ==================== SFT Data ====================
    sft_data: List[Dict[str, Any]]

    # ==================== Pipeline Control ====================
    current_phase: str  # "design", "scaffolding", "finalize", "complete"
    is_complete: bool
    error_message: Optional[str]

    # ==================== Timestamps ====================
    started_at: Optional[str]
    updated_at: Optional[str]

    # ==================== Checkpoint ====================
    checkpoint_path: Optional[str]
    last_checkpoint_at: Optional[str]
    logs_file_path: Optional[str]  # finalize에서 결과 로드용 로그 파일 경로


def create_initial_state(
    domain: str,
    train_dataset: str,
    instructional_goal: str,
    student_model_name: str,
    teacher_model_name: str,
    model_short: str,
    questions: List[Dict[str, Any]],
    checkpoint_interval: int = 10,
    use_iterative_scaffolding: bool = True,
    max_iterations: int = 3,
    design_result: Optional[DesignResult] = None,
) -> IDMASState:
    """파이프라인 초기 상태를 생성합니다.

    Args:
        domain: 도메인 이름 (예: "math")
        train_dataset: 훈련 데이터셋 이름 (예: "gsm8k")
        instructional_goal: 학습 목표
        student_model_name: 학생 모델 이름
        teacher_model_name: 교사 모델 이름
        model_short: 파일 이름용 짧은 모델명
        questions: 처리할 문제 리스트
        checkpoint_interval: 체크포인트 저장 간격 (문제 수). 기본값: 10
        use_iterative_scaffolding: Iterative Scaffolding 사용 여부. 기본값: True
        max_iterations: 최대 반복 횟수. 기본값: 3
        design_result: 사전 로드된 교수설계 결과. 기본값: None

    Returns:
        초기화된 IDMASState
    """
    return IDMASState(
        # Configuration
        domain=domain,
        train_dataset=train_dataset,
        instructional_goal=instructional_goal,
        student_model_name=student_model_name,
        teacher_model_name=teacher_model_name,
        model_short=model_short,
        checkpoint_interval=checkpoint_interval,
        use_iterative_scaffolding=use_iterative_scaffolding,
        max_iterations=max_iterations,

        # Design Phase
        design_result=design_result,
        task_analysis=design_result.get("instructional_analysis", {}).get("raw_output", "") if design_result else "",
        performance_objectives=design_result.get("performance_objectives", {}).get("performance_objectives", []) if design_result else [],

        # Questions
        questions=questions,
        total_questions=len(questions),
        current_question_index=0,
        current_question=questions[0] if questions else None,

        # Scaffolding
        scaffolding_results=[],
        scaffolding_processed=0,
        scaffolding_correct_count=0,
        case_a_independent_performance_mastery_count=0,
        case_b_scaffolded_coached_mastery_count=0,
        case_c_teacher_modeling_distillation_count=0,

        # Scaffolding Artifact statistics
        hot_scaffolding_count=0,
        lot_scaffolding_count=0,
        skipped_count=0,

        # SFT Data
        sft_data=[],

        # Control
        current_phase="scaffolding",
        is_complete=False,
        error_message=None,

        # Timestamps
        started_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),

        # Checkpoint
        checkpoint_path=None,
        last_checkpoint_at=None,
        logs_file_path=None,
    )


def get_statistics(state: IDMASState) -> Dict[str, Any]:
    """파이프라인 상태에서 통계를 추출합니다.

    Args:
        state: 현재 파이프라인 상태

    Returns:
        통계 딕셔너리:
            - total_questions: 전체 문제 수
            - scaffolding_processed: 처리된 문제 수
            - case_statistics: 케이스별 통계 (Case A: Independent Performance Mastery / Case B: Scaffolded & Coached Mastery / Case C: Teacher Modeling Distillation)
            - scaffolding_artifacts: HOT/LOT 스캐폴딩 통계
    """
    case_a = state.get("case_a_independent_performance_mastery_count", 0)
    case_b = state.get("case_b_scaffolded_coached_mastery_count", 0)
    case_c = state.get("case_c_teacher_modeling_distillation_count", 0)

    skipped = state.get("skipped_count", 0)

    # Scaffolding Artifact statistics
    hot_count = state.get("hot_scaffolding_count", 0)
    lot_count = state.get("lot_scaffolding_count", 0)

    processed = state.get("scaffolding_processed", 0)

    return {
        "total_questions": state.get("total_questions", 0),
        "scaffolding_processed": processed,
        "case_statistics": {
            "case_a_independent_performance_mastery": case_a,  # Case A: Independent Performance Mastery (독립적 수행 숙달)
            "case_b_scaffolded_coached_mastery": case_b,  # Case B: Scaffolded & Coached Mastery (스캐폴딩 기반 숙달)
            "case_c_teacher_modeling_distillation": case_c,  # Case C: Teacher Modeling Distillation (교사 모델링 증류)
            "skipped": skipped,  # 처리 실패로 건너뛴 문제
            "success_total": case_a + case_b,
            "success_rate": (case_a + case_b) / processed if processed > 0 else 0,
        },
        "scaffolding_artifacts": {
            "hot_count": hot_count,  # High-Order Thinking scaffolding
            "lot_count": lot_count,  # Low-Order Thinking scaffolding
            "total": hot_count + lot_count,
        },
    }


def load_checkpoint_from_logs(
    logs_path: Path,
) -> Tuple[Dict[str, Any], Set[str]]:
    """기존 로그 파일에서 체크포인트 상태를 로드합니다.

    JSONL (신규) → JSON (레거시) 순으로 시도합니다.
    JSONL은 한 줄씩 스트리밍 읽기하여 메모리 O(processed_ids)만 사용합니다.

    Args:
        logs_path: 로그 JSON 파일 경로 (*.json)
                   내부에서 *.jsonl 존재 여부를 먼저 확인합니다.

    Returns:
        튜플 (checkpoint_data, processed_question_ids):
            - checkpoint_data: 스캐폴딩 결과 및 통계 딕셔너리
            - processed_question_ids: 이미 처리된 문제 ID 집합
    """
    # JSONL 형식 우선 시도 (신규 포맷, 스트리밍 읽기)
    jsonl_path = logs_path.with_suffix(".jsonl")
    if jsonl_path.exists():
        return _load_checkpoint_from_jsonl(jsonl_path)

    # 레거시 JSON 폴백
    if not logs_path.exists():
        return {}, set()

    return _load_checkpoint_from_json(logs_path)


def _accumulate_result_stats(
    result: Dict[str, Any],
    checkpoint_data: Dict[str, Any],
    processed_ids: Set[str],
) -> None:
    """단일 결과에서 통계를 추출하여 checkpoint_data에 누적합니다.

    JSONL 스트리밍과 레거시 JSON 양쪽에서 공통 사용합니다.
    """
    LEGACY_CASE_MAP = {
        "A": SFTCase.INDEPENDENT_PERFORMANCE_MASTERY.value,
        "B": SFTCase.SCAFFOLDED_COACHED_MASTERY.value,
        "C": SFTCase.TEACHER_MODELING_DISTILLATION.value,
    }

    qid = result.get("id")
    if not qid:
        return

    processed_ids.add(qid)
    checkpoint_data["scaffolding_processed"] += 1

    # Support both old and new field names
    is_correct = result.get("scaffolding_correct") or result.get("phase1_correct")
    sft_case = result.get("sft_case")

    # Legacy compatibility: old "B" case (재구성) → Teacher Modeling Distillation로 매핑
    if sft_case == "B" and not is_correct:
        sft_case = SFTCase.TEACHER_MODELING_DISTILLATION.value

    # Legacy case value mapping (기존 "A"/"B"/"C" → 새 값)
    if sft_case in LEGACY_CASE_MAP:
        sft_case = LEGACY_CASE_MAP[sft_case]

    # Count by case
    if sft_case == SFTCase.INDEPENDENT_PERFORMANCE_MASTERY.value:
        checkpoint_data["case_a_independent_performance_mastery_count"] += 1
        checkpoint_data["scaffolding_correct_count"] += 1
    elif sft_case == SFTCase.SCAFFOLDED_COACHED_MASTERY.value:
        checkpoint_data["case_b_scaffolded_coached_mastery_count"] += 1
        checkpoint_data["scaffolding_correct_count"] += 1
    elif sft_case == SFTCase.TEACHER_MODELING_DISTILLATION.value:
        checkpoint_data["case_c_teacher_modeling_distillation_count"] += 1
    elif sft_case == SFTCase.SKIPPED.value:
        checkpoint_data["skipped_count"] += 1

    # Count HOT/LOT scaffolding from scaffolding_artifacts
    scaffolding_artifacts_data = result.get("scaffolding_artifacts") or []
    for artifact_entry in scaffolding_artifacts_data:
        for artifact in artifact_entry.get("artifacts", []):
            skill_type = artifact.get("skill_type", "")
            if skill_type == "HOT":
                checkpoint_data["hot_scaffolding_count"] += 1
            elif skill_type == "LOT":
                checkpoint_data["lot_scaffolding_count"] += 1


def _new_checkpoint_data() -> Dict[str, Any]:
    """빈 체크포인트 데이터를 생성합니다."""
    return {
        "scaffolding_processed": 0,
        "scaffolding_correct_count": 0,
        "case_a_independent_performance_mastery_count": 0,
        "case_b_scaffolded_coached_mastery_count": 0,
        "case_c_teacher_modeling_distillation_count": 0,
        "hot_scaffolding_count": 0,
        "lot_scaffolding_count": 0,
        "skipped_count": 0,
    }


def _load_checkpoint_from_jsonl(
    jsonl_path: Path,
) -> Tuple[Dict[str, Any], Set[str]]:
    """JSONL 파일에서 스트리밍으로 체크포인트를 로드합니다.

    한 줄씩 읽어 ID와 통계만 추출합니다.
    메모리 사용: O(processed_ids) — 결과 데이터는 메모리에 유지하지 않습니다.

    Args:
        jsonl_path: JSONL 파일 경로

    Returns:
        튜플 (checkpoint_data, processed_ids)
    """
    processed_ids: Set[str] = set()
    checkpoint_data = _new_checkpoint_data()

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    continue
                _accumulate_result_stats(result, checkpoint_data, processed_ids)
    except IOError as e:
        print(f"Warning: Could not load JSONL from {jsonl_path}: {e}")
        return {}, set()

    return checkpoint_data, processed_ids


def _load_checkpoint_from_json(
    logs_path: Path,
) -> Tuple[Dict[str, Any], Set[str]]:
    """레거시 JSON 파일에서 체크포인트를 로드합니다.

    기존 동작과 동일: 전체 JSON을 메모리에 로드합니다.
    마이그레이션 전 또는 레거시 파일 지원용입니다.

    Args:
        logs_path: JSON 파일 경로

    Returns:
        튜플 (checkpoint_data, processed_ids)
    """
    try:
        with open(logs_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load logs from {logs_path}: {e}")
        return {}, set()

    processed_ids: Set[str] = set()
    checkpoint_data = _new_checkpoint_data()

    # Process scaffolding results (supports both old and new field names)
    results_key = "scaffolding_results" if "scaffolding_results" in logs else "phase1_results"
    for result in logs.get(results_key, []):
        _accumulate_result_stats(result, checkpoint_data, processed_ids)

    return checkpoint_data, processed_ids


def restore_state_from_checkpoint(
    initial_state: IDMASState,
    checkpoint_data: Dict[str, Any],
    processed_ids: Set[str],
) -> IDMASState:
    """체크포인트 데이터에서 상태를 복원합니다.

    이미 처리된 문제를 제외하고 나머지 문제만 포함하여
    상태를 복원합니다.

    Args:
        initial_state: 초기 상태
        checkpoint_data: 로그에서 로드한 체크포인트 데이터
        processed_ids: 이미 처리된 문제 ID 집합

    Returns:
        복원된 IDMASState
    """
    if not checkpoint_data or not processed_ids:
        return initial_state

    # Filter out already processed questions
    remaining_questions = [
        q for q in initial_state.get("questions", [])
        if q.get("id") not in processed_ids
    ]

    # Merge checkpoint data into initial state
    restored = dict(initial_state)

    # Scaffolding results는 파일에만 보관, 메모리에서는 비어있음
    restored["scaffolding_results"] = []

    # Restore counters
    restored["scaffolding_processed"] = checkpoint_data.get("scaffolding_processed", 0)
    restored["scaffolding_correct_count"] = checkpoint_data.get("scaffolding_correct_count", 0)
    restored["case_a_independent_performance_mastery_count"] = checkpoint_data.get("case_a_independent_performance_mastery_count", 0)
    restored["case_b_scaffolded_coached_mastery_count"] = checkpoint_data.get("case_b_scaffolded_coached_mastery_count", 0)
    restored["case_c_teacher_modeling_distillation_count"] = checkpoint_data.get("case_c_teacher_modeling_distillation_count", 0)

    # Scaffolding Artifact statistics
    restored["hot_scaffolding_count"] = checkpoint_data.get("hot_scaffolding_count", 0)
    restored["lot_scaffolding_count"] = checkpoint_data.get("lot_scaffolding_count", 0)

    # Update questions to remaining ones
    restored["questions"] = remaining_questions
    restored["total_questions"] = len(initial_state.get("questions", []))  # Keep original total
    restored["current_question_index"] = 0
    restored["current_question"] = remaining_questions[0] if remaining_questions else None

    # Determine current phase
    if remaining_questions:
        restored["current_phase"] = "scaffolding"
    else:
        restored["current_phase"] = "finalize"

    restored["updated_at"] = datetime.now().isoformat()

    return IDMASState(**restored)
