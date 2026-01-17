# Plan: failure → skip 용어 통일 리팩토링

## 1. 문제 분석

### 배경
- 모든 step에서 fallback 발생 시 해당 질문이 skip됨
- `failure.step{N}.count`의 합계 = `failure.analysis.count` (skip된 총 질문 수)
- 따라서 "failure"라는 용어가 실제로는 "skip"을 의미함

### 목표
로그 및 코드에서 "failure" 용어를 "skip"으로 대체하여 의미를 명확히 함

### 변경 범위
| 변경 대상 | 변경 전 | 변경 후 |
|-----------|---------|---------|
| State 필드 | `step_failures` | `step_skips` |
| State 필드 | `failure` (dict) | `skip` |
| State 필드 | `*_failure_count` | `*_skip_count` |
| QuestionResult 필드 | `failure_reason` | `skip_reason` |
| QuestionResult 필드 | `failure_stage` | `skip_stage` |
| QuestionResult 필드 | `failure_details` | `skip_details` |
| 상수 | `FAILURE_SUFFIX` | `SKIP_SUFFIX` |
| 메서드 | `get_failure_key()` | `get_skip_key()` |
| 로그 출력 | `[Failure]` | `[Skip]` |
| 통계 키 | `"failure"` | `"skip"` |

### 유지 대상 (변경하지 않음)
| 항목 | 이유 |
|------|------|
| `failure_analysis` | 교사가 학생의 실패 원인을 분석한 교육적 피드백 |
| `_failure_metadata` 내부 | fallback 메커니즘 설명용 (is_fallback, failure_reason 등) |
| 프롬프트 템플릿 | LLM 출력 형식에 정의된 필드명 |

---

## 2. 작업 계획

### Task 1: state.py 수정

**파일**: `learning_loop/graph/state.py`

| 라인 | 변경 전 | 변경 후 |
|------|---------|---------|
| 68 | `FAILURE_SUFFIX = "_failure"` | `SKIP_SUFFIX = "_skip"` |
| 71-72 | `get_failure_key()` | `get_skip_key()` |
| 113-116 | `step_failures: Optional[Dict]` | `step_skips: Optional[Dict]` |
| 118-119 | `failure: Optional[Dict]` | `skip: Optional[Dict]` |
| 129-131 | `failure_reason`, `failure_stage`, `failure_details` | `skip_reason`, `skip_stage`, `skip_details` |
| 202-217 | `step{N}_failure_count` | `step{N}_skip_count` |
| 374-392 | `step{N}_failure` 변수 | `step{N}_skip` 변수 |
| 409-438 | `"failure": {...}` | `"skip": {...}` |
| 539-559 | `add_result_to_checkpoint()` 내부 | 변수명 및 키 변경 |
| 639-647 | `restore_from_checkpoint()` 내부 | 변수명 및 키 변경 |

### Task 2: nodes.py 수정

**파일**: `learning_loop/graph/nodes.py`

| 위치 | 변경 내용 |
|------|----------|
| `_update_for_skipped_question()` | `failure_*` → `skip_*` 변수명 |
| `process_question_scaffolding()` | `step_failures` → `step_skips`, `failures` → `skips` |
| QuestionResult 생성 시 | `failure=`, `step_failures=` → `skip=`, `step_skips=` |
| 업데이트 딕셔너리 | `step{N}_failure_count` → `step{N}_skip_count` |

### Task 3: graph.py 수정

**파일**: `learning_loop/graph/graph.py`

| 라인 | 변경 전 | 변경 후 |
|------|---------|---------|
| 333-334 | `failure = stats.get('failure', {})`, `[Failure]` | `skip = stats.get('skip', {})`, `[Skip]` |
| 336-350 | `failure.get(...)` | `skip.get(...)` |

### Task 4: QuestionResult 타입 수정 (해당되는 경우)

QuestionResult가 TypedDict나 dataclass로 정의되어 있다면 필드명 변경

### Task 5: ARCHITECTURE.md 문서 업데이트

문서에서 `failure` → `skip` 용어 업데이트

---

## 3. 구현 전략

### TDD 접근
1. **RED**: 기존 테스트가 있다면 새 필드명으로 업데이트
2. **GREEN**: 코드 변경 적용
3. **REFACTOR**: 일관성 검증

### 변경 순서
1. `state.py` - 핵심 State 정의
2. `nodes.py` - 로직 처리
3. `graph.py` - 출력
4. 문서 업데이트

### 주의사항
- **Backward Compatibility**: 기존 로그 파일은 `failure` 키를 사용하므로, `restore_from_checkpoint()`에서 두 키 모두 지원 필요
- **Legacy Fields**: 기존 `*_fallback_count` 필드는 그대로 유지 (별도 목적)

---

## 4. Quality Gates

### 완료 기준
- [ ] 모든 `*_failure_count` → `*_skip_count` 변경
- [ ] 모든 `step_failures` → `step_skips` 변경
- [ ] 로그 출력에서 `[Failure]` → `[Skip]` 변경
- [ ] 통계 구조에서 `"failure"` → `"skip"` 변경
- [ ] Python syntax check 통과
- [ ] 기존 로그 파일 호환성 유지

### 검증 방법
```bash
# Syntax check
python3 -m py_compile learning_loop/graph/state.py learning_loop/graph/nodes.py learning_loop/graph/graph.py

# 용어 변경 확인
grep -r "failure_count" learning_loop/  # 결과 없어야 함
grep -r "skip_count" learning_loop/     # 결과 있어야 함
```

---

## 5. 예상 최종 로그 구조

```json
{
  "statistics": {
    "skip": {
      "step1_initial_response": { "count": 0, "rate": 0.0 },
      "step2_evaluation": { "count": 0, "rate": 0.0 },
      "step3_scaffolding": { "count": 0, "rate": 0.0 },
      "step4_reresponse": { "count": 0, "rate": 0.0 },
      "step5_reconstruction": {
        "count": 0, "rate": 0.0,
        "case_b": 0, "case_c": 0, "summarization": 0
      },
      "analysis": { "count": 0, "rate": 0.0 }
    }
  }
}
```

---

## 6. 예상 소요 시간

| Task | 예상 복잡도 |
|------|------------|
| state.py | 높음 (많은 필드) |
| nodes.py | 중간 |
| graph.py | 낮음 |
| 문서 | 낮음 |
| 검증 | 낮음 |
