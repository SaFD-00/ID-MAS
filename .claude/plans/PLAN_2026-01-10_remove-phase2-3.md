# Phase 2-3 코드 삭제 계획

## 목표
ID-MAS 프로젝트에서 Phase 2 (Coaching)와 Phase 3 (Modeling) 관련 코드를 삭제하고, Phase 1 (Scaffolding)만 남긴다.

## 사용자 결정 사항
- **A-Failed 유지**: 5회 실패 시 대화 분석 후 재구성 로직 유지
- **incorrect_after_phase1 필드 삭제**: Phase 2로 넘어가지 않으므로 불필요

## 영향 범위 분석

### 삭제 대상 파일 및 코드

| 파일 | 삭제할 코드 | 비고 |
|------|------------|------|
| `learning_loop/graph/nodes.py` | Phase 2/3 함수들 | 약 200줄 |
| `learning_loop/graph/graph.py` | Phase 2/3 노드/엣지 | 약 80줄 |
| `learning_loop/graph/state.py` | Phase 2/3 상태 필드 | 약 50줄 |
| `learning_loop/teacher_model.py` | Phase 2/3 메서드 | 약 150줄 |
| `prompts/learning_prompts.py` | Phase 2/3 프롬프트 | 약 120줄 |
| `main.py` | Phase 2/3 관련 출력 | 소수 |
| `README.md` | Phase 2/3 설명 | 문서 |
| `ARCHITECTURE.md` | Phase 2/3 설명 | 문서 |

---

## 구현 계획

### Step 1: learning_loop/graph/nodes.py 수정

**삭제할 함수들:**
- `generate_coaching_db()` (line 344-415)
- `_find_weak_objectives()` (line 418-449)
- `process_question_phase2()` (line 452-509)
- `process_question_phase3()` (line 514-552)

**수정할 함수:**
- `process_question_phase1()`: `incorrect_after_phase1` 업데이트 로직 제거
- `generate_sft_data()`: Phase 2/3 결과 수집 부분 제거 (Case B, C 제거)
- `save_results()`: Phase 2/3 결과 저장 부분 제거
- `save_incremental_checkpoint()`: Phase 2/3 결과 저장 부분 제거

---

### Step 2: learning_loop/graph/graph.py 수정

**삭제할 import:**
```python
from learning_loop.graph.nodes import (
    generate_coaching_db,       # 삭제
    process_question_phase2,    # 삭제
    process_question_phase3,    # 삭제
)
```

**삭제할 노드 함수:**
- `coaching_db_node()` (line 85-90)
- `phase2_batch_node()` (line 92-125)
- `phase3_batch_node()` (line 127-152)

**삭제할 조건부 라우팅:**
- `should_run_phase2()` (line 173-179)
- `should_run_phase3()` (line 181-187)

**수정할 그래프 구성:**
- Phase 1 완료 후 바로 `finalize`로 이동하도록 변경
- `coaching_db`, `phase2`, `phase3` 노드 제거
- 관련 조건부 엣지 제거

**새로운 흐름:**
```
phase1 → advance → (더 있으면 phase1, 없으면 finalize) → END
```

---

### Step 3: learning_loop/graph/state.py 수정

**삭제할 SFTCase:**
- `B = "B"` (Phase 2)
- `C = "C"` (Phase 3)

**삭제할 QuestionResult 필드:**
```python
# Phase 2 results
phase2_scores: Optional[Dict[str, Any]]
fixed_response: Optional[str]
fixed_predicted: Optional[str]
phase2_correct: Optional[bool]
coaching_db_used: bool

# Phase 3 results
modeling_response: Optional[str]
phase3_applied: bool
```

**삭제할 IDMASState 필드:**
```python
# Phase 1 → Phase 2 전달용 (삭제)
incorrect_after_phase1: Annotated[List[QuestionResult], add_to_list]

# Phase 2
coaching_db: Optional[Dict[str, Any]]
weak_objectives: List[Dict[str, Any]]
phase2_results: Annotated[List[QuestionResult], add_to_list]
phase2_processed: int
still_incorrect_after_phase2: Annotated[List[QuestionResult], add_to_list]

# Phase 3
phase3_results: Annotated[List[QuestionResult], add_to_list]
phase3_processed: int
```

**수정할 함수:**
- `create_initial_state()`: Phase 2/3 초기화 제거
- `get_statistics()`: Phase 2/3 통계 제거
- `load_checkpoint_from_logs()`: Phase 2/3 복원 로직 제거
- `restore_state_from_checkpoint()`: Phase 2/3 복원 로직 제거

---

### Step 4: learning_loop/teacher_model.py 수정

**삭제할 import:**
```python
from prompts.learning_prompts import (
    PERFORMANCE_SCORING_PROMPT,      # 삭제
    WEAK_OBJECTIVE_ANALYSIS_PROMPT,  # 삭제
    COACHING_DB_GENERATION_PROMPT,   # 삭제
    MODELING_PROMPT,                 # 삭제
)
```

**삭제할 메서드:**
- `score_by_performance_objectives()` (line 36-86)
- `analyze_weak_objectives()` (line 157-202)
- `generate_coaching_db()` (line 204-250)
- `generate_modeling_response()` (line 252-291)

---

### Step 5: prompts/learning_prompts.py 수정

**삭제할 프롬프트:**
- `PERFORMANCE_SCORING_PROMPT` (line 42-74)
- `WEAK_OBJECTIVE_ANALYSIS_PROMPT` (line 169-201)
- `COACHING_DB_GENERATION_PROMPT` (line 208-274)
- `COACHING_RESPONSE_PROMPT` (line 281-309)
- `MODELING_PROMPT` (line 316-346)

---

### Step 6: main.py 수정

**수정할 내용:**
- 모듈 docstring에서 Phase 2/3 설명 제거
- `run_train_mode()` 출력에서 Phase 2/3 통계 제거
- `IDMASGraphRunner.run()` 호출 후 통계 출력 수정

---

### Step 7: README.md 수정

**수정할 내용:**
- "3-Phase Pipeline" 설명 → "Scaffolding Pipeline" 또는 "Iterative Scaffolding Pipeline"
- Phase 2/3 관련 설명 제거
- SFT Case B, C 설명 제거
- 데이터 구조에서 Phase 2/3 관련 필드 제거
- 로그 형식에서 `phase2_results`, `phase3_results`, `coaching_db` 필드 제거

---

### Step 8: ARCHITECTURE.md 수정

**수정할 내용:**
- "3-Phase Pipeline" 설명 → "Scaffolding Pipeline"
- Phase 2 (Coaching), Phase 3 (Modeling) 섹션 제거
- 그래프 흐름도 수정 (phase1 → finalize)
- SFT Case 표에서 B, C 제거
- Mermaid 다이어그램 수정

---

### Step 9: 로그 형식 수정

**`*_logs.json` 형식 변경:**

기존:
```json
{
  "phase1_results": [...],
  "phase2_results": [...],
  "phase3_results": [...],
  "coaching_db": {...},
  "statistics": {...}
}
```

변경 후:
```json
{
  "phase1_results": [...],
  "statistics": {...}
}
```

**`statistics` 필드에서 제거:**
- `phase2_processed`, `phase2_fixed`
- `phase3_processed`, `phase3_modeling`
- `sft_case_b`, `sft_case_c`

---

### Step 10: 검증 및 Git Push

1. **구문 검증**: `python -m py_compile main.py`
2. **import 검증**: `python -c "from learning_loop.graph import IDMASGraphRunner"`
3. **실행 테스트** (선택):
   ```bash
   python main.py --mode train --domain math --train-dataset gsm8k --resume False
   ```
4. **Git Push**: `/git:git-push` 실행

---

## 검증 방법

1. **구문 검증**: `python -m py_compile main.py`
2. **import 검증**: `python -c "from learning_loop.graph import IDMASGraphRunner"`
3. **실행 테스트**:
   ```bash
   python main.py --mode train --domain math --train-dataset gsm8k --resume False
   ```
4. **결과 확인**: Phase 1만 실행되고 SFT 데이터가 Case A, A-Failed만 포함하는지 확인

---

## 주의사항

1. **Phase 1 로직 유지**: `_process_iterative_scaffolding()`과 관련 함수는 그대로 유지
2. **A-Failed 케이스 유지**: Phase 1 실패 후 재구성 로직 유지
3. **Teacher 모델 일부 기능 유지**:
   - `evaluate_with_performance_objectives()` - Phase 1 Iterative에서 사용
   - `summarize_and_reconstruct()` - A-Failed 케이스에서 사용
   - `generate_initial_hint()`, `generate_progressive_hint()` - Iterative scaffolding에서 사용

---

## 파일 수정 순서

1. `prompts/learning_prompts.py` - 프롬프트 삭제
2. `learning_loop/teacher_model.py` - Phase 2/3 메서드 삭제
3. `learning_loop/graph/state.py` - 상태 스키마 수정
4. `learning_loop/graph/nodes.py` - 노드 함수 수정
5. `learning_loop/graph/graph.py` - 그래프 구성 수정
6. `main.py` - 출력 수정
7. `README.md` - 문서 수정
8. `ARCHITECTURE.md` - 문서 수정
9. 검증 실행
10. `/git:git-push` 실행
