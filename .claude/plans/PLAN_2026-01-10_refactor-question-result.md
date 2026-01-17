# Plan: QuestionResult 필드 리팩토링

**Date**: 2026-01-10
**Domain**: coding
**Status**: Draft

## 목표

로그 파일(`*_logs.json`) 및 관련 코드에서 `QuestionResult` 구조 변경:
1. `problem_text`, `ground_truth` 필드 삭제
2. `question_id` → `id`, `question` → `input` 명칭 변경
3. 필드 순서: `id`, `instruction`, `input`, 나머지

## 영향 범위 분석

### 변경 대상 파일

| 파일 | 변경 내용 | 우선순위 |
|------|----------|---------|
| `learning_loop/graph/state.py` | `QuestionResult` TypedDict 정의 수정 | 1 |
| `learning_loop/graph/nodes.py` | `QuestionResult` 생성/사용 로직 수정 | 2 |
| `main.py` | `question_data` 생성 시 필드명 변경 | 3 |

### 상세 변경 사항

#### 1. `state.py` - QuestionResult 정의

**Before:**
```python
class QuestionResult(TypedDict, total=False):
    question_id: str
    question: str
    problem_text: str
    ground_truth: str
    instruction: str
    # ... 나머지
```

**After:**
```python
class QuestionResult(TypedDict, total=False):
    id: str              # question_id → id
    instruction: str     # 순서 변경
    input: str           # question → input
    # problem_text 삭제
    # ground_truth 삭제
    # ... 나머지
```

#### 2. `nodes.py` - QuestionResult 생성 함수들

수정 대상 함수:
- `_process_single_shot()` (line 109-137)
- `_process_iterative_scaffolding()` (line 140-288)
- `process_question_phase2()` (line 429-485)
- `process_question_phase3()` (line 490-527)
- `_create_sft_entry()` (line 598-632)

**변경 패턴:**
```python
# Before
QuestionResult(
    question_id=question["question_id"],
    question=question["question"],
    problem_text=question["problem_text"],
    ground_truth=question["ground_truth"],
    ...
)

# After
QuestionResult(
    id=question["id"],
    instruction=question.get("instruction", ""),
    input=question["input"],
    ...
)
```

#### 3. `main.py` - question_data 생성

**Before (line 241-249):**
```python
question_data.append({
    'question_id': q.question_id,
    'question': q.question,
    'problem_text': self.loader.format_question_as_prompt(q),
    'ground_truth': self.loader.format_ground_truth(q),
    'instruction': q.metadata.get('instruction', '')
})
```

**After:**
```python
question_data.append({
    'id': q.question_id,
    'instruction': q.metadata.get('instruction', ''),
    'input': q.question,
    # problem_text, ground_truth는 내부 처리용으로 별도 저장
    '_problem_text': self.loader.format_question_as_prompt(q),
    '_ground_truth': self.loader.format_ground_truth(q),
})
```

> **참고**: `problem_text`와 `ground_truth`는 학습 로직에서 필요하므로 언더스코어 prefix로 내부용 필드로 유지하거나, 별도 처리 필요

#### 4. `state.py` - load_checkpoint_from_logs 함수

로그 파일에서 데이터 로드 시 새 필드명 사용:
```python
# Before
qid = result.get("question_id")

# After
qid = result.get("id")
```

## 실행 순서

### Step 1: QuestionResult 타입 정의 수정
- [ ] `learning_loop/graph/state.py` 수정
- [ ] 필드 순서: `id`, `instruction`, `input`, 나머지

### Step 2: nodes.py 수정
- [ ] `_process_single_shot()` 수정
- [ ] `_process_iterative_scaffolding()` 수정
- [ ] `process_question_phase2()` 수정
- [ ] `process_question_phase3()` 수정
- [ ] `_create_sft_entry()` 수정

### Step 3: main.py 수정
- [ ] `run_learning_phase()` 내 question_data 생성 로직 수정

### Step 4: load_checkpoint_from_logs 수정
- [ ] `state.py`의 체크포인트 로드 함수 수정

### Step 5: 테스트
- [ ] 기존 로그 파일과의 호환성 확인 (마이그레이션 필요 여부)
- [ ] 새 학습 실행 테스트

## 주의 사항

### 1. 내부 처리용 필드 분리
`problem_text`와 `ground_truth`는 학습 로직에서 필요:
- 정답 비교 (`answer_extractor.compare()`)
- Phase 2/3 처리

**해결 방안**:
- 로그 파일에는 저장하지 않지만, 런타임에서는 사용
- `QuestionResult`에서 제거하되, 처리 함수에서 별도 파라미터로 전달

### 2. 기존 로그 파일 호환성
- 기존 `*_logs.json` 파일과 새 형식 비호환
- 마이그레이션 스크립트 또는 하위 호환 로드 로직 필요

## 리스크

| 리스크 | 영향 | 대응 |
|--------|-----|------|
| 기존 로그 파일 비호환 | Resume 실패 | 마이그레이션 스크립트 작성 |
| ground_truth 제거로 정답 비교 불가 | 학습 실패 | 내부 처리용 필드로 분리 |

## 질문 사항

1. `problem_text`와 `ground_truth`를 로그에서 완전히 제거할지, 언더스코어 prefix로 내부용으로 유지할지?
2. 기존 로그 파일 마이그레이션이 필요한지?
