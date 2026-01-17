# Plan: Enhanced Instruction 데이터 형식 변경

## 1. Problem Analysis

### 현재 상황
- `dataset_enhancer.py`에서 생성되는 데이터 구조:
```json
{
  "instruction": "원본 instruction",
  "input": "문제 텍스트",
  "output": "정답",
  "metadata": {},
  "_enhanced": true,
  "_instructional_goal": "학습 목표",
  "_task_analysis": "과제 분석",
  "_enhanced_instruction": "Enhanced된 instruction"
}
```

### 문제점
- `_enhanced_instruction` 필드가 별도로 존재하여 실제 학습에 사용되지 않음
- 메타데이터 필드들이 불필요하게 데이터셋에 포함됨

### 목표 데이터 구조
```json
{
  "instruction": "_enhanced_instruction 값 (Enhanced된 instruction)",
  "input": "문제 텍스트",
  "output": "정답",
  "metadata": {}
}
```

---

## 2. Requirements Specification

### Acceptance Criteria
- [ ] AC1: 생성된 데이터셋의 `instruction` 필드에 enhanced instruction 값이 직접 들어감
- [ ] AC2: `_enhanced`, `_instructional_goal`, `_task_analysis`, `_enhanced_instruction` 필드가 제거됨
- [ ] AC3: 기존 ID-MAS 데이터 파일들이 새 형식으로 재생성됨
- [ ] AC4: 데이터 로더(`domain_loader.py`) 및 학습 코드(`main.py`) 수정 불필요

### 범위
- **In Scope**: `utils/dataset_enhancer.py` 수정, 기존 데이터 파일 재생성
- **Out of Scope**: `domain_loader.py`, `main.py` 수정

---

## 3. Architecture Design

### 선택된 접근법: 데이터 생성 시점 수정

**이유**:
- 가장 단순하고 깔끔한 해결책
- 로더 및 학습 코드 변경 없음
- 데이터 구조가 표준 SFT 형식과 일치

### 변경 영역
```
utils/dataset_enhancer.py
└── _enhance_instructions() 메서드 (라인 116-150)
    ├── instruction 필드에 enhanced_instruction 값 대입
    └── 메타데이터 필드들 제거
```

---

## 4. Task Decomposition

### Task List

| ID | Task | Dependencies | Files | Priority |
|----|------|--------------|-------|----------|
| T1 | `_enhance_instructions()` 메서드 수정 | - | utils/dataset_enhancer.py | P0 |
| T2 | 기존 ID-MAS 데이터 파일 재생성 | T1 | data/math/train/data/*.json | P1 |
| T3 | 결과 검증 | T2 | - | P1 |

### Dependency Graph
```
T1 ──► T2 ──► T3
```

---

## 5. Implementation Strategy

### T1: `_enhance_instructions()` 메서드 수정

**수정 위치**: `utils/dataset_enhancer.py:116-150`

**Before** (라인 129-148):
```python
enhanced = []
for item in data:
    new_item = item.copy()
    original_instruction = item.get("instruction", "")

    enhanced_instruction = ENHANCED_INSTRUCTION_TEMPLATE.format(...)

    # 메타데이터 추가 (instruction 필드는 변경하지 않음)
    new_item["_enhanced"] = True
    new_item["_instructional_goal"] = instructional_goal
    new_item["_task_analysis"] = task_analysis
    new_item["_enhanced_instruction"] = enhanced_instruction

    enhanced.append(new_item)
```

**After**:
```python
enhanced = []
for item in data:
    original_instruction = item.get("instruction", "")

    enhanced_instruction = ENHANCED_INSTRUCTION_TEMPLATE.format(...)

    # 깔끔한 구조로 새 아이템 생성
    new_item = {
        "instruction": enhanced_instruction,  # Enhanced instruction 직접 사용
        "input": item.get("input", ""),
        "output": item.get("output", ""),
        "metadata": item.get("metadata", {})
    }

    enhanced.append(new_item)
```

**주석 수정** (라인 122-127):
```python
"""
데이터에 학습목표와 과제분석이 포함된 enhanced instruction 적용

instruction 필드에 enhanced instruction을 직접 적용하여
SFT 학습 시 향상된 프롬프트가 사용되도록 합니다.
"""
```

### T2: 기존 데이터 파일 재생성

```bash
# 모든 ID-MAS 데이터 재생성
python utils/dataset_enhancer.py --all --model-suffix Qwen2.5-7B-Instruct
```

### T3: 결과 검증

1. JSON 파일 구조 확인:
   - `instruction` 필드에 enhanced 내용 포함 여부
   - `_enhanced`, `_instructional_goal` 등 필드 제거 여부

2. 검증 명령:
```bash
# 첫 번째 레코드 구조 확인
python -c "import json; d=json.load(open('data/math/train/data/gsm8k_train_ID-MAS_Qwen2.5-7B-Instruct.json')); print(list(d[0].keys()))"
# 기대 결과: ['instruction', 'input', 'output', 'metadata']
```

---

## 6. Execution Plan

### 실행 순서
1. **Sequential**: T1 → T2 → T3

### 예상 소요
- T1: 코드 수정 (단일 파일, 단일 메서드)
- T2: 데이터 재생성 (API 호출 필요 - 시간 소요)
- T3: 검증 (자동화)

---

## 7. Quality Gates

### Phase 1: 코드 수정 (T1)
- [ ] `_enhance_instructions()` 메서드 수정 완료
- [ ] 주석 업데이트 완료

### Phase 2: 데이터 재생성 (T2)
- [ ] 재생성 명령 실행
- [ ] 오류 없이 완료

### Phase 3: 검증 (T3)
- [ ] JSON 구조 확인: 4개 필드만 존재 (instruction, input, output, metadata)
- [ ] instruction 필드에 "## Learning Objective" 포함
- [ ] `_enhanced` 등 메타데이터 필드 없음

---

## 8. Rollback Plan

필요 시 원복:
1. `dataset_enhancer.py` git checkout
2. 기존 데이터 파일은 재생성으로 복구 가능 (소스 데이터 보존됨)
