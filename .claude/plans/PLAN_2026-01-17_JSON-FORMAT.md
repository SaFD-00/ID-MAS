# JSON 데이터 형식 변환 계획

## 1. Problem Analysis

### 현재 상태
기존 생성된 파일들이 8개 필드를 포함:
```json
{
  "instruction": "원본 instruction",
  "input": "...",
  "output": "...",
  "metadata": {...},
  "_enhanced": true,
  "_instructional_goal": "...",
  "_task_analysis": "...",
  "_enhanced_instruction": "확장된 instruction"
}
```

### 원하는 형식
4개 필드만 유지:
```json
{
  "instruction": "_enhanced_instruction 값 사용",
  "input": "...",
  "output": "...",
  "metadata": {...}
}
```

### Root Cause
- 기존 파일들이 이전 버전 코드로 생성됨
- 현재 `dataset_enhancer.py`는 이미 올바른 형식 사용 (라인 136-144)

---

## 2. Requirements Specification

### Acceptance Criteria
- [ ] AC1: 변환된 파일은 4개 필드만 포함 (instruction, input, output, metadata)
- [ ] AC2: instruction 필드는 `_enhanced_instruction` 값 사용
- [ ] AC3: metadata는 원본 유지 (level, type 등)
- [ ] AC4: 3B, 7B 두 파일 모두 변환 완료
- [ ] AC5: JSON 파일 유효성 검증 통과

### 대상 파일
1. `data/math/train/data/math_train_ID-MAS_Qwen2.5-3B-Instruct_Qwen2.5-3B-Instruct.json`
2. `data/math/train/data/math_train_ID-MAS_Qwen2.5-7B-Instruct_Qwen2.5-7B-Instruct.json`

---

## 3. Architecture Design

### 접근 방법: 변환 스크립트

기존 파일을 읽어서 새 형식으로 변환하는 간단한 Python 스크립트 사용.

**이유:**
- `dataset_enhancer.py`는 이미 올바른 형식 → 수정 불필요
- 기존 파일만 변환하면 됨
- API 호출 없이 로컬에서 즉시 처리 가능

---

## 4. Task Decomposition

### Task List
| ID | Task | Dependencies | Files | Priority |
|----|------|--------------|-------|----------|
| T1 | 변환 스크립트 작성 | - | scripts/convert_json_format.py | P0 |
| T2 | 3B 파일 변환 | T1 | math_train_ID-MAS_*_3B_*.json | P0 |
| T3 | 7B 파일 변환 | T1 | math_train_ID-MAS_*_7B_*.json | P0 |
| T4 | 결과 검증 | T2, T3 | - | P0 |

### Dependency Graph
```
T1 ──┬──► T2 ──┐
     │         ├──► T4
     └──► T3 ──┘
```

---

## 5. Implementation Strategy

### 5.1 변환 로직
```python
def convert_item(item):
    return {
        "instruction": item.get("_enhanced_instruction", item.get("instruction", "")),
        "input": item.get("input", ""),
        "output": item.get("output", ""),
        "metadata": item.get("metadata", {})
    }
```

### 5.2 실행 순서
1. 변환 스크립트 작성
2. 3B, 7B 파일 병렬 변환
3. 변환 결과 검증 (레코드 수, 필드 구조)

---

## 6. Parallel Agent Execution Plan

### 6.1 Execution Mode
- [x] Sequential (단일 스크립트로 두 파일 처리)

### 6.2 예상 결과
- 입력: 7,500 레코드 × 2 파일
- 출력: 동일 레코드 수, 4개 필드

---

## 7. Quality Gates

### Phase 1: 스크립트 작성
- [ ] 변환 로직 정확성

### Phase 2: 변환 실행
- [ ] 3B 파일 변환 완료
- [ ] 7B 파일 변환 완료
- [ ] 레코드 수 일치 확인

### Phase 3: 검증
- [ ] 필드 구조 확인 (4개만)
- [ ] JSON 유효성 검증
- [ ] instruction 내용 확인

### Implementation Readiness
- [ ] 모든 AC 충족
- [ ] 기존 파일 백업 여부 확인
