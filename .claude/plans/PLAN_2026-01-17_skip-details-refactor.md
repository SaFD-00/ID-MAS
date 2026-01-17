# Skip Details 구조 리팩토링 계획

## 1. 문제 분석

### 현재 상태
현재 skip 관련 메타데이터가 **3가지 중복 구조**로 저장되고 있음:

```json
{
  "step_skips": {
    "step2": { "is_fallback": false, "attempts_needed": 1, "stage": "..." },
    "step3": { "is_fallback": true, "failure_reason": "...", "last_error": "...", "stage": "..." }
  },
  "skip": {
    "evaluation": { "is_fallback": false, "attempts_needed": 1, "stage": "..." },
    "scaffolding_artifact": { "is_fallback": true, "failure_reason": "...", "last_error": "...", "stage": "..." }
  },
  "skip_details": { ... }
}
```

### 문제점
1. 동일한 정보가 3개 필드에 중복 저장
2. `stage` 필드가 키와 값 모두에 존재 (중복)
3. `last_error`가 마지막 에러만 저장 (이전 에러 손실)

### 목표 상태
```json
{
  "skip_details": {
    "step2_performance_objectives_evaluation": {
      "is_fallback": false,
      "attempts_needed": 1
    },
    "step3_scaffolding_artifact_generation": {
      "is_fallback": true,
      "failure_reason": "scaffolding_artifact_generation_failed",
      "last_error": ["Error 1", "Error 2", "Error 3"],
      "max_retries_exceeded": 3
    }
  }
}
```

---

## 2. 요구사항 명세

### 변경 사항
| 항목 | 현재 | 변경 후 |
|------|------|---------|
| 필드 구조 | `step_skips`, `skip`, `skip_details` | `skip_details`만 사용 |
| 키 형식 | `step2`, `evaluation` 등 | `step{N}_{stage}` |
| `stage` 필드 | 값에 포함 | 키에만 포함 (값에서 제거) |
| `last_error` | 문자열 (마지막만) | 배열 (모든 에러) |

### 키 이름 매핑
| 현재 키 | 새 키 |
|---------|-------|
| `step2` / `evaluation` | `step2_performance_objectives_evaluation` |
| `step3` / `scaffolding_artifact` | `step3_scaffolding_artifact_generation` |
| `step5` (Case B) / `reconstruction` | `step5_case_b_reconstruction` |
| `step5` (Case C) / `final_solution` | `step5_case_c_final_solution` |
| `step5_summarization` / `summarization` | `step5_summarization` |

---

## 3. 아키텍처 설계

### 수정 대상 파일

```
learning_loop/
├── graph/
│   ├── state.py          # QuestionResult TypedDict 수정
│   └── nodes.py          # skip 데이터 수집 및 QuestionResult 생성 수정
└── teacher_model.py      # _failure_metadata 생성 로직 수정
```

### 선택된 접근 방식
**점진적 리팩토링**: 기존 코드와의 호환성을 유지하면서 단계적으로 수정

---

## 4. 태스크 분해

### Task 1: teacher_model.py - _failure_metadata 구조 변경

| 세부 작업 | 파일 위치 | 설명 |
|-----------|-----------|------|
| T1.1 | 라인 107-177 | `evaluate_with_performance_objectives()` - step2 메타데이터 수정 |
| T1.2 | 라인 327-397 | `reconstruct_successful_scaffolding()` - step5 Case B 수정 |
| T1.3 | 라인 665-740 | `generate_scaffolding_artifact()` - step3 수정 |
| T1.4 | 라인 786-863 | `generate_final_solution()` - step5 Case C 수정 |

**변경 내용**:
- `last_error`를 리스트로 변경하고 모든 에러 수집
- Legacy 키 제거 (evaluation, scaffolding_artifact 등)
- `stage` 필드 제거
- 새 키 형식 사용: `step{N}_{stage_name}`

### Task 2: graph/state.py - TypedDict 수정

| 세부 작업 | 파일 위치 | 설명 |
|-----------|-----------|------|
| T2.1 | 라인 164 | `step_skips` 필드 제거 |
| T2.2 | 라인 167 | `skip` 필드 제거 |
| T2.3 | 라인 179 | `skip_details` 타입 주석 업데이트 |

### Task 3: graph/nodes.py - skip 데이터 수집 로직 수정

| 세부 작업 | 파일 위치 | 설명 |
|-----------|-----------|------|
| T3.1 | 라인 297-298 | `skips`, `step_skips` 변수 → `skip_details` 단일 변수로 변경 |
| T3.2 | 라인 348-383 | Step 2 evaluation skip 처리 수정 |
| T3.3 | 라인 436-474 | Step 3 scaffolding skip 처리 수정 |
| T3.4 | 라인 525-566 | Step 5 Case B reconstruction skip 처리 수정 |
| T3.5 | 라인 618-659 | Step 5 Case C final solution skip 처리 수정 |
| T3.6 | 전체 | QuestionResult 생성 시 `step_skips`, `skip` 제거, `skip_details` 사용 |

---

## 5. 구현 전략

### TDD 접근
1. **RED**: 새 구조에 대한 테스트 작성 (기존 테스트 수정)
2. **GREEN**: 최소한의 코드로 테스트 통과
3. **REFACTOR**: 코드 정리

### 구현 순서
```
T1 (teacher_model.py) → T2 (state.py) → T3 (nodes.py)
```

의존성: T3은 T1, T2 완료 후 진행

### 검증 단계
| 단계 | 검증 내용 | 도구 |
|------|-----------|------|
| 코드 수정 후 | 타입 체크 | mypy / pyright |
| 통합 테스트 | 파이프라인 실행 | 샘플 데이터로 테스트 |
| 로그 확인 | 새 구조 생성 확인 | JSON 로그 파일 검사 |

---

## 6. 상세 구현 가이드

### T1.1: evaluate_with_performance_objectives() 수정

**현재 코드** (라인 107-177):
```python
# 에러 수집
last_error = None
for attempt in range(1, max_retries + 1):
    try:
        ...
    except Exception as e:
        last_error = e
```

**변경 코드**:
```python
# 에러 수집 (배열)
errors = []
for attempt in range(1, max_retries + 1):
    try:
        ...
    except Exception as e:
        errors.append(str(e))
```

**성공 시 메타데이터**:
```python
result['_failure_metadata'] = {
    "step2_performance_objectives_evaluation": {
        "is_fallback": False,
        "attempts_needed": attempt
    }
}
```

**실패 시 메타데이터**:
```python
"_failure_metadata": {
    "step2_performance_objectives_evaluation": {
        "is_fallback": True,
        "failure_reason": "json_parse_error",
        "last_error": errors,  # 배열
        "max_retries_exceeded": max_retries
    }
}
```

### T1.2-T1.4: 동일 패턴 적용

각 메서드에서:
1. `last_error = None` → `errors = []`
2. `last_error = e` → `errors.append(str(e))`
3. `"last_error": str(last_error)` → `"last_error": errors`
4. Legacy 키 제거
5. `stage` 필드 제거

### T3: nodes.py 수정 패턴

**현재 코드**:
```python
skips = {}
step_skips = {}

# Step 2 수집
if step2_skip:
    step_skips[PipelineStep.STEP2] = step2_skip
if eval_skip:
    skips["evaluation"] = eval_skip
```

**변경 코드**:
```python
skip_details = {}

# Step 2 수집
if step2_meta := evaluation.get("_failure_metadata", {}).get("step2_performance_objectives_evaluation"):
    skip_details["step2_performance_objectives_evaluation"] = step2_meta
```

---

## 7. Quality Gates

### Phase 1: 코드 수정
- [ ] teacher_model.py 모든 메서드 수정 완료
- [ ] state.py TypedDict 수정 완료
- [ ] nodes.py skip 처리 로직 수정 완료

### Phase 2: 검증
- [ ] 타입 체크 통과
- [ ] 샘플 데이터로 파이프라인 실행 성공
- [ ] 로그 파일에 새 구조 정상 생성 확인

### Phase 3: 정리
- [ ] 사용하지 않는 Legacy 코드 제거
- [ ] 주석 업데이트

---

## 8. 예상 결과

### 변경 전 로그
```json
{
  "step_skips": { "step2": {...}, "step3": {...} },
  "skip": { "evaluation": {...}, "scaffolding_artifact": {...} },
  "skip_details": {...}
}
```

### 변경 후 로그
```json
{
  "skip_details": {
    "step2_performance_objectives_evaluation": {
      "is_fallback": false,
      "attempts_needed": 1
    },
    "step3_scaffolding_artifact_generation": {
      "is_fallback": true,
      "failure_reason": "scaffolding_artifact_generation_failed",
      "last_error": [
        "Error 1: JSON parse failed",
        "Error 2: Timeout",
        "Error 3: Invalid response"
      ],
      "max_retries_exceeded": 3
    }
  }
}
```
