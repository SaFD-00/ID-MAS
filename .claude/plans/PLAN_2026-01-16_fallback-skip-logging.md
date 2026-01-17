# Development Plan: Fallback 발생 시 Skip + Failure 로깅

**생성일**: 2026-01-16
**상태**: Draft

---

## 1. Problem Analysis

### 1.1 현재 상황
- Teacher 모델의 7개 inference 함수에 이미 fallback 메커니즘 구현됨
- fallback 응답은 `_failure_metadata.is_fallback: true`로 표시됨
- **문제점**: fallback 응답이 SFT 데이터로 사용되면 학습 품질 저하

### 1.2 Root Cause
- fallback 발생 시 해당 문제를 계속 처리하여 저품질 데이터가 학습 데이터에 포함됨
- failure 발생 문제를 별도로 추적하지 않아 분석이 어려움

### 1.3 요구사항
1. **Skip**: fallback 발생 시 해당 문제 skip
2. **Log**: failure 정보는 로그 파일에 기록 (현재 수준 유지)
3. **Exclude**: SFT 데이터에서 skipped 문제 제외
4. **Track**: 통계에 skipped_count 추가

---

## 2. Requirements Specification

### 2.1 Acceptance Criteria

| ID | Criteria | Verification |
|----|----------|--------------|
| AC1 | fallback 발생 시 해당 문제가 skip되어야 함 | 로그에서 is_skipped: true 확인 |
| AC2 | skipped 문제는 로그 파일에 failure 정보와 함께 기록되어야 함 | logs.json에 failure 필드 확인 |
| AC3 | skipped 문제는 SFT 데이터에 포함되지 않아야 함 | sft_data에서 skipped 문제 없음 |
| AC4 | statistics에 skipped_count가 집계되어야 함 | statistics.failures.skipped_count 확인 |
| AC5 | 기존 통과 케이스는 영향받지 않아야 함 | Case A/B/C 정상 동작 확인 |

### 2.2 Fallback 감지 조건

다음 조건 중 하나라도 만족 시 해당 문제 skip:

| Stage | Fallback 조건 | 파일:라인 |
|-------|---------------|----------|
| PO Evaluation | `evaluation._failure_metadata.evaluation.is_fallback == True` | teacher_model.py:127 |
| Scaffolding Artifact | `artifact._failure_metadata.scaffolding_artifact.is_fallback == True` | teacher_model.py:645 |
| Case B Reconstruction | `result._failure_metadata.reconstruction.is_fallback == True` | teacher_model.py:358 |
| Case C Reconstruction | `result._failure_metadata.reconstruction.is_fallback == True` | teacher_model.py:469 |
| Final Solution | `result._failure_metadata.final_solution.is_fallback == True` | teacher_model.py:754 |

---

## 3. Architecture Design

### 3.1 접근 방식 비교

| 접근 방식 | 장점 | 단점 | 선택 |
|----------|-----|-----|------|
| **A: Early Return** | 즉시 skip, 리소스 절약 | 일부 정보 손실 가능 | |
| **B: Flag + Filter** | 전체 정보 보존, 유연함 | 약간의 리소스 낭비 | ✅ |
| **C: Exception 기반** | 명확한 흐름 제어 | 기존 구조 대폭 변경 필요 | |

### 3.2 선택: B - Flag + Filter 방식

**이유**:
1. 기존 코드 구조 최소 변경
2. failure 정보 완전 보존 (로깅 목적)
3. 유연한 필터링 가능

### 3.3 Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ _process_iterative_scaffolding()                                 │
│   │                                                             │
│   ├─ Teacher inference → fallback 발생?                          │
│   │   └─ YES: is_skipped=True, failure 정보 수집, early return   │
│   │   └─ NO: 정상 처리 계속                                       │
│   │                                                             │
│   └─ QuestionResult 반환 (is_skipped 포함)                       │
├─────────────────────────────────────────────────────────────────┤
│ generate_sft_data()                                              │
│   │                                                             │
│   └─ is_skipped == True? → EXCLUDE from sft_data                │
├─────────────────────────────────────────────────────────────────┤
│ save_incremental_checkpoint()                                    │
│   │                                                             │
│   └─ ALL results (including skipped) → logs.json                │
├─────────────────────────────────────────────────────────────────┤
│ statistics                                                       │
│   │                                                             │
│   └─ skipped_count 집계 추가                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Task Decomposition

### Task List

| ID | Task | Dependencies | Files | Priority |
|----|------|--------------|-------|----------|
| T1 | QuestionResult에 is_skipped 필드 추가 | - | state.py | P0 |
| T2 | IDMASState에 skipped_count 필드 추가 | - | state.py | P0 |
| T2.1 | create_initial_state()에 skipped_count 초기화 추가 | T2 | state.py | P0 |
| T2.2 | get_statistics()에 skipped 통계 추가 | T1, T2 | state.py | P0 |
| T2.3 | load_checkpoint_from_logs()에 skipped_count 복원 | T1, T2 | state.py | P0 |
| T2.4 | restore_state_from_checkpoint()에 skipped_count 복원 | T1, T2 | state.py | P0 |
| T3 | fallback 감지 헬퍼 함수 추가 | - | nodes.py | P0 |
| T3.1 | _process_iterative_scaffolding에 evaluation fallback 감지 | T1, T3 | nodes.py | P0 |
| T3.2 | _process_iterative_scaffolding에 scaffolding_artifact fallback 감지 | T1, T3 | nodes.py | P0 |
| T3.3 | process_question_scaffolding에 skipped 처리 로직 추가 | T1, T2 | nodes.py | P0 |
| T4 | generate_sft_data에서 skipped 문제 제외 | T1 | nodes.py | P0 |
| T5 | 단위 테스트 작성 | T1-T4 | tests/ | P1 |
| T6 | 통합 테스트 및 검증 | T5 | - | P1 |

### Dependency Graph

```
T1 ──────────────────────────────────────────┐
     │                                        │
     ├──► T2.2 ──┐                           │
T2 ──┼──► T2.3 ──┼──► T3.1 ──┐              │
     ├──► T2.4 ──┤           │              │
     └──► T2.1 ──┘           ├──► T5 ──► T6 │
                             │              │
T3 ──────────────► T3.2 ─────┤              │
                             │              │
T1 ──────────────► T3.3 ─────┘              │
     │                                       │
     └──────────────► T4 ────────────────────┘
```

---

## 5. Implementation Strategy

### 5.1 TDD Approach

각 Task별 RED → GREEN → REFACTOR:

#### T1: QuestionResult에 is_skipped 필드 추가

**RED** (테스트 먼저):
```python
# tests/test_state.py
def test_question_result_has_is_skipped_field():
    result = QuestionResult(
        id="test_1",
        instruction="...",
        input="...",
        output="...",
        is_skipped=True,
        failure_reason="evaluation_fallback"
    )
    assert result.is_skipped == True
    assert result.failure_reason == "evaluation_fallback"
```

**GREEN** (최소 구현):
```python
# learning_loop/graph/state.py
class QuestionResult(TypedDict, total=False):
    # ... existing fields ...
    is_skipped: bool  # NEW: fallback 발생 시 True
    failure_reason: Optional[str]  # NEW: skip 사유
```

#### T3: fallback 감지 로직 추가

**구현 위치**: [nodes.py:_process_iterative_scaffolding()](learning_loop/graph/nodes.py#L172)

```python
# nodes.py의 _process_iterative_scaffolding 함수 내부

def _check_and_handle_fallback(
    evaluation: Optional[Dict],
    scaffolding_artifact: Optional[Dict],
    question_id: str
) -> Tuple[bool, Optional[str]]:
    """Check if fallback occurred and return skip status."""

    # Check evaluation fallback
    if evaluation:
        eval_meta = evaluation.get("_failure_metadata", {}).get("evaluation", {})
        if eval_meta.get("is_fallback"):
            return True, "evaluation_fallback"

    # Check scaffolding artifact fallback
    if scaffolding_artifact:
        artifact_meta = scaffolding_artifact.get("_failure_metadata", {}).get("scaffolding_artifact", {})
        if artifact_meta.get("is_fallback"):
            return True, "scaffolding_artifact_fallback"

    return False, None
```

### 5.2 Incremental Verification Steps

| Stage | What to Verify | Tool | Pass Criteria |
|-------|---------------|------|---------------|
| After T1 | QuestionResult 필드 추가 | pytest | TypedDict 검증 통과 |
| After T2 | IDMASState 필드 추가 | pytest | State 생성 검증 통과 |
| After T3 | Fallback 감지 동작 | pytest | fallback 시 is_skipped=True |
| After T4 | SFT 데이터 필터링 | pytest | skipped 문제 제외 확인 |
| After T5 | 통계 집계 | pytest | skipped_count 정확 |
| After T7 | 전체 통합 | 실제 실행 | 로그 파일 검증 |

---

## 6. Detailed Implementation Guide

### 6.1 state.py 변경

**파일**: [learning_loop/graph/state.py](learning_loop/graph/state.py)

#### 변경 1: QuestionResult에 필드 추가 (line ~80)

```python
class QuestionResult(TypedDict, total=False):
    # ... existing fields ...

    # NEW: Skip tracking
    is_skipped: bool  # fallback 발생으로 skip된 경우 True
    failure_reason: Optional[str]  # skip 사유 (e.g., "evaluation_fallback")
    failure_stage: Optional[str]  # 실패 발생 단계
    failure_details: Optional[Dict[str, Any]]  # 상세 failure 메타데이터
```

#### 변경 2: IDMASState에 skipped_count 추가 (line ~130)

```python
class IDMASState(TypedDict, total=False):
    # ... existing fields ...

    # Failure statistics (기존)
    evaluation_fallback_count: int
    # ...

    # NEW: Skip statistics
    skipped_count: int  # fallback으로 skip된 문제 수
```

#### 변경 3: create_initial_state()에 초기값 추가 (line ~230)

```python
# learning_loop/graph/state.py - create_initial_state() 함수 내
skipped_count=0,
```

#### 변경 4: get_statistics()에 skipped 통계 추가 (line ~279, 297)

```python
# line 279 근처
skipped = state.get("skipped_count", 0)

# line 297 failures dict에 추가
"failures": {
    # ... existing ...
    "skipped_count": skipped,
    "skipped_rate": skipped / processed if processed > 0 else 0,
}
```

#### 변경 5: load_checkpoint_from_logs()에 skipped_count 복원 (line ~356)

```python
# line 356 근처 checkpoint_data 초기화에 추가
checkpoint_data = {
    # ... existing ...
    "skipped_count": 0,
}

# line 360-426 루프 내에 추가
if result.get("is_skipped", False):
    checkpoint_data["skipped_count"] += 1
```

#### 변경 6: restore_state_from_checkpoint()에 skipped_count 복원 (line ~476)

```python
# line 476 근처에 추가
restored["skipped_count"] = checkpoint_data.get("skipped_count", 0)
```

### 6.2 nodes.py 변경

**파일**: [learning_loop/graph/nodes.py](learning_loop/graph/nodes.py)

#### 변경 1: fallback 감지 헬퍼 함수 추가 (line ~170 위치)

```python
def _check_fallback_in_metadata(metadata: Optional[Dict], stage: str) -> bool:
    """Check if fallback occurred in the given metadata."""
    if not metadata:
        return False
    stage_meta = metadata.get("_failure_metadata", {}).get(stage, {})
    return stage_meta.get("is_fallback", False)


def _collect_failure_info(
    evaluation: Optional[Dict] = None,
    scaffolding_artifact: Optional[Dict] = None,
    reconstruction: Optional[Dict] = None,
    final_solution: Optional[Dict] = None,
) -> Tuple[bool, Optional[str], Optional[str], Optional[Dict]]:
    """
    Collect failure information from all stages.

    Returns:
        (is_skipped, failure_reason, failure_stage, failure_details)
    """
    checks = [
        (evaluation, "evaluation", "evaluation_fallback"),
        (scaffolding_artifact, "scaffolding_artifact", "scaffolding_artifact_fallback"),
        (reconstruction, "reconstruction", "reconstruction_fallback"),
        (final_solution, "final_solution", "final_solution_fallback"),
    ]

    for data, stage, reason in checks:
        if _check_fallback_in_metadata(data, stage):
            details = data.get("_failure_metadata", {}).get(stage, {}) if data else None
            return True, reason, stage, details

    return False, None, None, None
```

#### 변경 2: _process_iterative_scaffolding 수정 (line ~172)

기존 함수 내 여러 위치에서 fallback 감지 추가:

**2-1. Evaluation fallback 감지 (line ~254 이후)**:
```python
# Step 2: Teacher evaluates with Performance Objectives
if performance_objectives:
    evaluation = teacher_model.evaluate_with_performance_objectives(...)

    # NEW: Check for evaluation fallback → skip
    if evaluation.get("_failure_metadata", {}).get("evaluation", {}).get("is_fallback"):
        print(f"    [SKIP] Evaluation fallback detected. Skipping question {qid}")
        return QuestionResult(
            id=qid,
            instruction=question.get("instruction", ""),
            input=question["problem_text"],
            output=question["output"],
            is_skipped=True,
            failure_reason="evaluation_fallback",
            failure_stage="evaluation",
            failure_details=evaluation.get("_failure_metadata", {}).get("evaluation"),
            scaffolding_correct=False,
            sft_case=None,
        )
```

**2-2. Scaffolding artifact fallback 감지 (line ~315 이후)**:
```python
# Step 3 (NEW): Generate Scaffolding Artifact (HOT/LOT)
scaffolding_artifact = teacher_model.generate_scaffolding_artifact(...)

# NEW: Check for scaffolding_artifact fallback → skip
if scaffolding_artifact.get("_failure_metadata", {}).get("scaffolding_artifact", {}).get("is_fallback"):
    print(f"    [SKIP] Scaffolding artifact fallback detected. Skipping question {qid}")
    return QuestionResult(
        id=qid,
        instruction=question.get("instruction", ""),
        input=question["problem_text"],
        output=question["output"],
        is_skipped=True,
        failure_reason="scaffolding_artifact_fallback",
        failure_stage="scaffolding_artifact",
        failure_details=scaffolding_artifact.get("_failure_metadata", {}).get("scaffolding_artifact"),
        scaffolding_correct=False,
        sft_case=None,
        # 현재까지의 정보 보존
        initial_response=response if iteration == 1 else None,
        iterative_scaffolding={
            "success": False,
            "iterations_needed": iteration,
            "conversation_history": conversation_history,
        },
    )
```

#### 변경 3: process_question_scaffolding에 skipped 처리 추가 (line ~86)

```python
def process_question_scaffolding(state: IDMASState) -> Dict[str, Any]:
    # ... existing code ...

    result = _process_iterative_scaffolding(...)

    # NEW: Handle skipped questions
    if result.get("is_skipped", False):
        updates = {
            "scaffolding_results": state.get("scaffolding_results", []) + [result],
            "skipped_count": state.get("skipped_count", 0) + 1,
            "scaffolding_processed": state.get("scaffolding_processed", 0) + 1,
            "updated_at": datetime.now().isoformat(),
        }
        print(f"  -> SKIPPED: Fallback detected ({result.get('failure_reason', 'unknown')})")
        return updates

    # ... continue with existing case handling ...
```

#### 변경 3: generate_sft_data에서 skipped 제외 (line ~512)

```python
def generate_sft_data(state: IDMASState) -> Dict[str, Any]:
    sft_data = []
    skipped_count = 0

    for result in state.get("scaffolding_results", []):
        # NEW: Skip fallback cases
        if result.get("is_skipped", False):
            skipped_count += 1
            continue

        if result.get("sft_case") in (SFTCase.A.value, SFTCase.B.value, SFTCase.C.value):
            entry = _create_sft_entry(result)
            if entry:
                sft_data.append(entry)

    return {
        "sft_data": sft_data,
        "skipped_count": state.get("skipped_count", 0) + skipped_count,  # 누적
        "is_complete": True,
        "current_phase": "complete",
        "updated_at": datetime.now().isoformat(),
    }
```

#### 변경 4: statistics에 skipped_count 추가

`get_statistics()` 함수 또는 통계 집계 부분에 추가:

```python
"failures": {
    # ... existing ...
    "skipped_count": state.get("skipped_count", 0),
    "skipped_rate": state.get("skipped_count", 0) / state.get("scaffolding_processed", 1),
}
```

---

## 7. Quality Gates

### Phase 1: Context & Problem
- [x] Codebase context gathered
- [x] Problem/goal identified
- [x] Root cause analyzed

### Phase 2: Requirements Clarification
- [x] Clear goal definition confirmed with user
- [x] Scope boundaries explicitly defined
- [x] Constraints and limitations identified
- [x] Ambiguous requirements clarified

### Phase 3: Spec & Architecture
- [x] Testable AC exist for each requirement
- [x] Architecture approach selected with rationale
- [x] Data model defined

### Phase 4: Task & Execution
- [x] Tasks decomposed
- [x] Dependencies mapped
- [x] TDD strategy defined per task
- [x] Verification steps defined

### Implementation Readiness
- [ ] All tasks have clear inputs/outputs
- [ ] Test files identified for each task
- [ ] Integration strategy defined

---

## 8. Verification Plan

### 8.1 Unit Tests

```python
# tests/test_fallback_skip.py

def test_fallback_detection_evaluation():
    """evaluation fallback 감지 테스트"""
    evaluation = {
        "_failure_metadata": {
            "evaluation": {"is_fallback": True, "failure_reason": "json_parse_error"}
        }
    }
    is_skipped, reason, stage, details = _collect_failure_info(evaluation=evaluation)
    assert is_skipped == True
    assert reason == "evaluation_fallback"


def test_skipped_excluded_from_sft():
    """skipped 문제가 SFT 데이터에서 제외되는지 테스트"""
    state = {
        "scaffolding_results": [
            {"id": "q1", "is_skipped": True, "sft_case": "A"},
            {"id": "q2", "is_skipped": False, "sft_case": "A", "sft_response": "..."},
        ]
    }
    result = generate_sft_data(state)
    assert len(result["sft_data"]) == 1
    assert result["sft_data"][0]["metadata"]["id"] == "q2"
```

### 8.2 Integration Test

실제 fallback 상황 시뮬레이션:
1. Teacher API를 mock하여 fallback 응답 반환
2. 전체 파이프라인 실행
3. 로그 파일에서 is_skipped=True 확인
4. SFT 데이터에서 해당 문제 제외 확인

### 8.3 Manual Verification

```bash
# 실행 후 로그 파일 검증
jq '.scaffolding_results[] | select(.is_skipped == true)' \
  data/math/train/.../logs.json

# SFT 데이터에 skipped 없음 확인
jq '.[] | select(.metadata.id | contains("skipped"))' \
  data/math/train/.../.json

# 통계 확인
jq '.statistics.failures.skipped_count' \
  data/math/train/.../logs.json
```

---

## 9. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| 기존 테스트 깨짐 | Medium | 점진적 변경, 기존 테스트 우선 통과 확인 |
| 로그 파일 포맷 호환성 | Low | is_skipped 필드 optional로 설정 |
| Resume 기능 영향 | Medium | load_checkpoint_from_logs 업데이트 필요 |

---

## Appendix: File Changes Summary

| File | Changes | Lines Affected |
|------|---------|----------------|
| learning_loop/graph/state.py | QuestionResult, IDMASState 필드 추가 | ~80, ~130 |
| learning_loop/graph/nodes.py | fallback 감지 로직, SFT 필터링 | ~170, ~512 |
| tests/test_fallback_skip.py | 새 테스트 파일 | NEW |
