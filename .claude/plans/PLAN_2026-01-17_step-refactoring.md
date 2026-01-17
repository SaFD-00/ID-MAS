# ID-MAS Pipeline Step Refactoring Plan

## 1. Problem Analysis

### 현재 문제점
1. **로그 구조 불일치**: `failure` 부분에서 `reconstruction`은 묶여있지만, `evaluation`, `hint`, `summarization`, `scaffolding_artifact`, `final_solution`은 분리되어 있음
2. **단계 표현 부재**: 파이프라인의 각 단계에 명확한 step 번호가 없음
3. **변수명 불일관**: 코드, 로그, 문서에서 단계를 나타내는 방식이 일관되지 않음

### Root Cause
- 초기 설계 시 step 기반 구조를 고려하지 않음
- 점진적으로 기능이 추가되면서 일관성이 깨짐

---

## 2. Requirements Specification

### 새로운 Step 구조

| Step | 이름 | 설명 | 함수 |
|------|------|------|------|
| **Step 1** | Initial Response | 초기 응답 생성 | `generate_initial_response_with_scaffolding()` |
| **Step 2** | PO Evaluation | Performance Objectives 평가 | `evaluate_with_performance_objectives()` |
| **Step 3** | Scaffolding | 스캐폴딩 아티팩트 생성 | `generate_scaffolding_artifact()` |
| **Step 4** | Re-response | 학생 재응답 | `respond_with_scaffolding_artifact()` |
| **Step 5** | Reconstruction | 최종 응답 재구성 | `reconstruct_*()` / `generate_final_solution()` |
| **Step 6** | SFT Generation | SFT 데이터 생성 | `generate_sft_data()` |

### Acceptance Criteria

1. **AC1**: 모든 변수명이 `step{N}_` 접두사를 사용
2. **AC2**: 로그의 `failure` 구조가 `step{N}_failures` 형태로 통합
3. **AC3**: `ARCHITECTURE.md`에 새로운 step 구조가 반영
4. **AC4**: `README.md`에 새로운 step 구조가 반영
5. **AC5**: 기존 로그 파일과의 호환성 유지 (optional)
6. **AC6**: 모든 테스트 통과

---

## 3. Architecture Design

### 3.0 Step 기반 파이프라인 흐름도

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ID-MAS Iterative Scaffolding Pipeline                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Step 1: Initial Response (초기 응답 생성) - 1회만 실행              │    │
│  │   └─ student.generate_initial_response_with_scaffolding()           │    │
│  │   └─ Fallback: 빈 응답 + step1_failure 기록                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    ITERATION LOOP (최대 5회)                         │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │ Step 2: PO Evaluation (Performance Objectives 평가)           │  │    │
│  │  │   └─ teacher.evaluate_with_performance_objectives()           │  │    │
│  │  │   └─ Fallback: all_satisfied=False + step2_failure 기록       │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                           ↓                                          │    │
│  │                    all_satisfied?                                    │    │
│  │                    ├─ YES → Exit Loop → Step 5                       │    │
│  │                    └─ NO ↓                                           │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │ Step 3: Scaffolding Artifact (스캐폴딩 생성)                  │  │    │
│  │  │   └─ teacher.generate_scaffolding_artifact()                  │  │    │
│  │  │   └─ HOT/LOT 분류 → scaffolding_db에 누적                     │  │    │
│  │  │   └─ Fallback: 기본 힌트 + step3_failure 기록                 │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                           ↓                                          │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │ Step 4: Student Re-response (학생 재응답)                     │  │    │
│  │  │   └─ student.respond_with_scaffolding_artifact()              │  │    │
│  │  │   └─ scaffolding_db 참조하여 개선된 응답 생성                  │  │    │
│  │  │   └─ Fallback: 이전 응답 유지 + step4_failure 기록            │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                           ↓                                          │    │
│  │                 iteration < 5? ──YES──→ Step 2로 돌아감             │    │
│  │                         │                                            │    │
│  │                        NO (5회 도달)                                 │    │
│  │                         ↓                                            │    │
│  │                    Exit Loop → Step 5 (Case C)                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Step 5: Reconstruction (재구성)                                     │    │
│  │   ├─ Case A (iteration=1, all_satisfied=True)                       │    │
│  │   │   └─ 원본 학생 응답 그대로 사용                                  │    │
│  │   │                                                                  │    │
│  │   ├─ Case B (iteration=2~5, all_satisfied=True)                     │    │
│  │   │   └─ teacher.reconstruct_successful_scaffolding()               │    │
│  │   │   └─ 대화 기반 재구성                                           │    │
│  │   │   └─ Fallback: 마지막 학생 응답 사용 + step5_failure 기록       │    │
│  │   │                                                                  │    │
│  │   └─ Case C (iteration=5, all_satisfied=False)                      │    │
│  │       └─ teacher.generate_final_solution()                          │    │
│  │       └─ 정답 기반 최종 솔루션 생성                                  │    │
│  │       └─ Fallback: 정답만 포함 + step5_failure 기록                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Step 6: SFT Data Generation (SFT 데이터 생성)                       │    │
│  │   └─ generate_sft_data()                                            │    │
│  │   └─ Case A/B/C 결과를 SFT 형식으로 변환                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.0.1 반복 로직 (Iteration Logic)

| Iteration | 실행 Step | 조건 |
|-----------|----------|------|
| 1 | Step 1 → Step 2 | 항상 실행 |
| 1 | Step 2 → Step 5 | `all_satisfied=True` (Case A) |
| 1 | Step 2 → Step 3 → Step 4 | `all_satisfied=False` |
| 2~4 | Step 2 → Step 5 | `all_satisfied=True` (Case B) |
| 2~4 | Step 2 → Step 3 → Step 4 | `all_satisfied=False` |
| 5 | Step 2 → Step 5 | `all_satisfied=True` (Case B) |
| 5 | Step 2 → Step 5 | `all_satisfied=False` (Case C) |

### 3.0.2 Fallback 로직 (Step별 실패 처리)

| Step | 실패 원인 | Fallback 동작 | 기록 위치 |
|------|----------|---------------|----------|
| **Step 1** | API 호출 실패, 파싱 오류 | 빈 응답 반환, 문제 skip | `step1_failure` |
| **Step 2** | API 호출 실패, JSON 파싱 오류 | `all_satisfied=False` 가정, 계속 진행 | `step2_failure` |
| **Step 3** | API 호출 실패, 파싱 오류 | 기본 힌트 메시지 사용 | `step3_failure` |
| **Step 4** | API 호출 실패 | 이전 iteration 응답 유지 | `step4_failure` |
| **Step 5 (B)** | 재구성 실패 | 마지막 학생 응답 사용 | `step5_failure` (case_b) |
| **Step 5 (C)** | 최종 솔루션 생성 실패 | 정답만 포함한 응답 | `step5_failure` (case_c) |

### 3.0.3 Failure Metadata 구조 (새로운 형식)

**각 Step의 _failure_metadata:**

```python
# Step 2 (PO Evaluation) 실패 예시
"_failure_metadata": {
    "step2": {
        "is_fallback": True,
        "failure_reason": "json_parse_error",
        "last_error": "JSONDecodeError: ...",
        "attempts_needed": 3,
        "max_retries_exceeded": True,
        "stage": "po_evaluation"
    }
}

# Step 3 (Scaffolding) 실패 예시
"_failure_metadata": {
    "step3": {
        "is_fallback": True,
        "failure_reason": "scaffolding_generation_failed",
        "last_error": "API timeout",
        "attempts_needed": 3,
        "max_retries_exceeded": True,
        "stage": "scaffolding_artifact"
    }
}

# Step 5 (Reconstruction) 실패 예시 - Case B
"_failure_metadata": {
    "step5": {
        "is_fallback": True,
        "failure_reason": "reconstruction_failed",
        "case": "B",
        "last_error": "...",
        "stage": "reconstruction"
    },
    "step5_summarization": {  # 부가 실패 (대화 요약)
        "is_fallback": True,
        "failure_reason": "summarization_failed"
    }
}

# Step 5 (Reconstruction) 실패 예시 - Case C
"_failure_metadata": {
    "step5": {
        "is_fallback": True,
        "failure_reason": "final_solution_failed",
        "case": "C",
        "last_error": "...",
        "stage": "final_solution"
    }
}
```

### 3.0.4 통합 통계 구조 (새로운 형식)

```json
{
  "statistics": {
    "total_questions": 100,
    "scaffolding_processed": 95,

    "case_statistics": {
      "case_a": 30,
      "case_b": 50,
      "case_c": 15,
      "success_total": 80,
      "success_rate": 0.842
    },

    "iteration_statistics": {
      "avg_iterations": 2.3,
      "distribution": {
        "1": 30,  // Case A
        "2": 25,
        "3": 15,
        "4": 7,
        "5": 18   // Case B(3) + Case C(15)
      }
    },

    "scaffolding_artifacts": {
      "hot_count": 45,
      "lot_count": 120,
      "total": 165
    },

    "step_failures": {
      "step1": {"count": 0, "rate": 0.0},
      "step2": {"count": 1, "rate": 0.01},
      "step3": {"count": 2, "rate": 0.02},
      "step4": {"count": 0, "rate": 0.0},
      "step5": {
        "count": 3,
        "rate": 0.03,
        "case_b": 2,
        "case_c": 1,
        "summarization": 1  // 부가 실패
      }
    },

    "skipped": {
      "count": 5,
      "rate": 0.053,
      "reasons": {
        "step1_failure": 3,
        "critical_error": 2
      }
    },

    "total_failures": 6,
    "failure_rate": 0.063
  }
}
```

### 3.1 새로운 로그 구조

**Before:**
```json
{
  "failure": {
    "evaluation": {...},
    "hint": 0,
    "summarization": 0,
    "scaffolding_artifact": 0,
    "final_solution": 0
  }
}
```

**After:**
```json
{
  "step1_initial_response": {...},
  "step2_evaluation": {...},
  "step3_scaffolding": {...},
  "step4_reresponse": {...},
  "step5_reconstruction": {...},
  "failures": {
    "step1": {"count": 0, "details": []},
    "step2": {"count": 1, "details": [...]},
    "step3": {"count": 0, "details": []},
    "step4": {"count": 0, "details": []},
    "step5": {"count": 0, "details": []}
  }
}
```

### 3.2 Statistics 구조 변경

**Before:**
```json
"failures": {
  "reconstruction": {"case_b": 2, "case_c": 1, "total": 3},
  "evaluation": 1,
  "hint": 0,
  "summarization": 0,
  "scaffolding_artifact": 2,
  "final_solution": 0
}
```

**After:**
```json
"step_failures": {
  "step1": {"count": 0, "rate": 0.0},
  "step2": {"count": 1, "rate": 0.01},
  "step3": {"count": 2, "rate": 0.02},
  "step4": {"count": 0, "rate": 0.0},
  "step5": {"count": 3, "rate": 0.03, "case_b": 2, "case_c": 1}
},
"total_failures": 6,
"failure_rate": 0.06
```

---

## 4. Task Decomposition

### Task List

| ID | Task | 파일 | 우선순위 | 의존성 |
|----|------|------|----------|--------|
| **T1** | Step 상수 정의 | `learning_loop/graph/state.py` | P0 | - |
| **T2** | IDMASState 스키마 변경 | `learning_loop/graph/state.py` | P0 | T1 |
| **T3** | Teacher 함수 반환값 변경 | `learning_loop/teacher_model.py` | P0 | T1 |
| **T4** | Student 함수 반환값 변경 | `learning_loop/student_model.py` | P0 | T1 |
| **T5** | Nodes 로직 리팩토링 | `learning_loop/graph/nodes.py` | P1 | T2, T3, T4 |
| **T6** | Statistics 구조 변경 | `learning_loop/graph/nodes.py` | P1 | T5 |
| **T7** | 로그 저장 형식 변경 | `learning_loop/graph/nodes.py` | P1 | T5, T6 |
| **T8** | ARCHITECTURE.md 업데이트 | `ARCHITECTURE.md` | P2 | T1-T7 |
| **T9** | README.md 업데이트 | `README.md` | P2 | T1-T7 |
| **T10** | 통합 테스트 | - | P2 | T1-T9 |

### Dependency Graph

```
T1 (Step 상수) ─────────────────────────────────────────────────┐
    │                                                           │
    ├── T2 (State 스키마) ──┐                                   │
    │                       │                                   │
    ├── T3 (Teacher) ───────┼── T5 (Nodes) ── T6 (Stats) ───┐  │
    │                       │       │                        │  │
    └── T4 (Student) ───────┘       │                        │  │
                                    │                        │  │
                                    └── T7 (로그 저장) ──────┤  │
                                                             │  │
                                            T8 (ARCH.md) ────┼──┘
                                            T9 (README.md) ──┘
                                                             │
                                            T10 (테스트) ────┘
```

---

## 5. Implementation Strategy

### 5.1 Code Writing Method (TDD)

각 Task는 다음 순서로 진행:

1. **RED**: 실패하는 테스트 작성 (해당되는 경우)
2. **GREEN**: 최소한의 코드로 테스트 통과
3. **REFACTOR**: 코드 정리 및 개선

### 5.2 Incremental Verification Steps

| Stage | 검증 내용 | 도구 | 통과 기준 |
|-------|----------|------|----------|
| T1-T4 완료 후 | 임포트 오류 없음 | `python -c "from learning_loop import *"` | 오류 없음 |
| T5 완료 후 | 파이프라인 실행 | `python main.py --mode train --domain math --train-dataset gsm8k --limit 5` | 5개 문제 처리 |
| T6-T7 완료 후 | 로그 구조 확인 | 로그 파일 검사 | 새 구조 반영 |
| T8-T9 완료 후 | 문서 일관성 | 수동 검토 | 코드와 문서 일치 |
| T10 | 전체 통합 | 전체 테스트 | 모든 테스트 통과 |

### 5.3 Per-Task Implementation Guide

#### T1: Step 상수 정의

**파일**: `learning_loop/graph/state.py`

```python
# Step 상수 정의
class PipelineStep:
    """ID-MAS Pipeline Step 상수"""
    STEP1_INITIAL_RESPONSE = "step1_initial_response"
    STEP2_EVALUATION = "step2_evaluation"
    STEP3_SCAFFOLDING = "step3_scaffolding"
    STEP4_RERESPONSE = "step4_reresponse"
    STEP5_RECONSTRUCTION = "step5_reconstruction"
    STEP6_SFT_GENERATION = "step6_sft_generation"

    # 편의 상수
    ALL_STEPS = [
        STEP1_INITIAL_RESPONSE,
        STEP2_EVALUATION,
        STEP3_SCAFFOLDING,
        STEP4_RERESPONSE,
        STEP5_RECONSTRUCTION,
        STEP6_SFT_GENERATION,
    ]
```

#### T2: IDMASState 스키마 변경

**주요 변경**:
- `evaluation_fallback_count` → `step2_failure_count`
- `scaffolding_artifact_fallback_count` → `step3_failure_count`
- `final_solution_fallback_count` → `step5_failure_count`

#### T3-T4: Teacher/Student 함수 반환값 변경

**주요 변경**:
- `_failure_metadata["evaluation"]` → `_failure_metadata["step2"]`
- `_failure_metadata["scaffolding_artifact"]` → `_failure_metadata["step3"]`
- `_failure_metadata["final_solution"]` → `_failure_metadata["step5"]`
- `_failure_metadata["reconstruction"]` → `_failure_metadata["step5"]`

#### T5: Nodes 로직 리팩토링

**주요 변경**:
- 모든 결과 딕셔너리 키를 step 기반으로 변경
- failure 집계 로직 수정

#### T6: Statistics 구조 변경

**주요 변경**:
```python
# Before
"failures": {
    "evaluation": count,
    "scaffolding_artifact": count,
    ...
}

# After
"step_failures": {
    "step1": {"count": 0, "rate": 0.0},
    "step2": {"count": count, "rate": rate},
    ...
}
```

#### T7: 로그 저장 형식 변경

**주요 변경**:
- 각 문제 결과에 step 기반 키 사용
- 이전 버전 로그와의 호환성 고려 (마이그레이션 함수)

#### T8-T9: 문서 업데이트

**ARCHITECTURE.md 변경 사항**:
- "Iterative Scaffolding Pipeline" 섹션에 Step 표 추가
- 데이터 흐름 다이어그램 업데이트
- 로그 구조 예시 업데이트

**README.md 변경 사항**:
- 파이프라인 설명에 Step 구조 추가
- 출력 파일 형식 설명 업데이트

---

## 6. Parallel Agent Execution Plan

### 6.1 Execution Mode

- [x] **Sequential** (의존성 있음, 단계별 검증 필요)
- [ ] Parallel
- [ ] Competitive

### 6.2 Agent Assignment

| Task Group | Agent | 모델 | 예상 출력 |
|------------|-------|------|----------|
| T1-T4 | Single Agent | sonnet | 상수 및 스키마 정의 |
| T5-T7 | Single Agent | sonnet | 노드 로직 리팩토링 |
| T8-T9 | Single Agent | haiku | 문서 업데이트 |
| T10 | verifier | sonnet | 테스트 실행 및 검증 |

### 6.3 Execution Order

```
Group 1 (sequential): [T1] → [T2, T3, T4] → [T5] → [T6, T7]
  └── 상수 정의 → 스키마/함수 변경 → 노드 로직 → 통계/로그

Group 2 (sequential): [T8, T9]
  └── 문서 업데이트

Group 3: [T10]
  └── 통합 테스트 및 검증
```

---

## 7. Quality Gates

### Phase 1: Context & Problem
- [x] 코드베이스 컨텍스트 수집 완료
- [x] 문제/목표 명확히 식별
- [x] 근본 원인 분석 완료

### Phase 2: Requirements Clarification
- [x] 명확한 목표 정의 (사용자 확인)
- [x] 범위 경계 명시적 정의
- [x] 제약 조건 식별

### Phase 3: Spec & Architecture
- [x] 각 요구사항에 대한 테스트 가능한 AC 존재
- [x] 아키텍처 접근 방식 선택 (근거 포함)
- [x] 데이터 모델 정의

### Phase 4: Task & Execution
- [x] 작업 분해 완료 (Least-to-Most)
- [x] 의존성 매핑 완료
- [x] 작업별 TDD 전략 정의
- [x] 검증 단계 정의
- [x] 에이전트 할당 최적화

### Implementation Readiness
- [x] 모든 작업에 명확한 입력/출력
- [x] 각 작업에 대한 테스트 파일 식별
- [x] 통합 전략 정의
- [x] 롤백 계획 (git을 통한 복원)

---

## 8. Risk & Mitigation

| 리스크 | 영향 | 완화 방안 |
|--------|------|----------|
| 기존 로그 호환성 | 중 | 마이그레이션 함수 작성 또는 버전 구분 |
| 다른 코드의 의존성 | 높 | 단계별 검증으로 조기 발견 |
| 문서-코드 불일치 | 낮 | T10에서 최종 검증 |

---

## 9. Estimated Changes

| 파일 | 변경 유형 | 예상 라인 수 |
|------|----------|-------------|
| `learning_loop/graph/state.py` | 수정 | ~50 |
| `learning_loop/graph/nodes.py` | 수정 | ~100 |
| `learning_loop/teacher_model.py` | 수정 | ~30 |
| `learning_loop/student_model.py` | 수정 | ~20 |
| `ARCHITECTURE.md` | 수정 | ~50 |
| `README.md` | 수정 | ~30 |

---

## 10. Next Steps

1. 사용자 승인 후 T1부터 순차적으로 진행
2. 각 Task 완료 후 점진적 검증
3. 모든 Task 완료 후 전체 통합 테스트
