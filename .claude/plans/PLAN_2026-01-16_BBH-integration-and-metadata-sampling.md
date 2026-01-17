# Development Plan: BBH Integration & Metadata-based Sampling

**Date**: 2026-01-16
**Status**: Pending Approval

---

## 1. Problem Analysis

### 1.1 Current State

**Task 1: BBH Dataset Fragmentation**
- `dataset_preparer.py`에서 BBH를 9개 subtask별 파일로 분리 저장
- 현재 파일들: `bbh_{subtask}_test.json` (9개)
- `domains.py`에서 각 subtask를 개별 eval dataset으로 등록
- 평가 시 9번의 개별 inference 필요 → 비효율적

**Task 2: Sample Extraction Strategy**
- `sample_extractor.py`에서 MATH만 Hugging Face에서 직접 로드하여 메타데이터(type, level) 기반 stratified sampling
- 다른 데이터셋은 길이 기반 diverse sampling 또는 random sampling
- 이제 모든 데이터셋에 메타데이터가 포함됨 → 활용 필요

### 1.2 Root Cause (Five Whys)

**BBH 분리 저장 문제**:
1. Why: 평가 시 9번 inference 필요 → 비효율적
2. Why: 각 subtask가 개별 파일로 저장되어 있음
3. Why: 초기 설계에서 subtask별 세분화 분석 목적
4. Why: 하지만 통합 평가가 더 효율적
5. Why: subtask 정보는 metadata로 유지 가능

**메타데이터 미활용 문제**:
1. Why: MATH 외 데이터셋에서 메타데이터 기반 샘플링 미구현
2. Why: dataset_preparer.py가 메타데이터 포함하도록 수정됨 (최근)
3. Why: sample_extractor.py가 아직 이를 활용하지 않음

---

## 2. Requirements Specification

### 2.1 Functional Requirements

| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| FR1 | BBH 통합 저장 | 9개 subtask가 단일 `bbh_test.json` 파일로 저장됨 |
| FR2 | BBH metadata 유지 | 각 문제의 `metadata.subtask`에 원래 subtask 이름 포함 |
| FR3 | BBH instruction 유지 | 각 문제는 해당 subtask의 원래 instruction 유지 |
| FR4 | 평가 통합 | `domains.py`에서 `bbh`를 단일 eval dataset으로 등록 |
| FR5 | 메타데이터 샘플링 | type > level > length 순으로 계층적 샘플링 |
| FR6 | 메타데이터 없는 경우 fallback | 기존 diverse/random 샘플링으로 fallback |

### 2.2 Non-Functional Requirements

| ID | Requirement | Criteria |
|----|-------------|----------|
| NFR1 | 하위 호환성 | 기존 개별 BBH 파일 처리 로직 유지 (optional) |
| NFR2 | 코드 단순화 | 중복 코드 최소화 |

---

## 3. Architecture Design

### 3.1 Approach Comparison

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **A. 완전 통합** | BBH를 단일 파일로만 저장, 개별 파일 제거 | 단순, 깔끔 | 기존 호환성 손실 |
| **B. 하이브리드** | 통합 파일 생성 + 개별 파일도 유지 | 하위 호환성 | 중복 저장 |
| **C. 동적 로딩** | 개별 파일 유지, 로딩 시 통합 | 저장 구조 유지 | 로딩 복잡성 증가 |

### 3.2 Selected Approach: A (완전 통합)

**선택 이유**:
- 단순하고 유지보수 용이
- 기존 개별 파일 참조 코드는 적음 (domains.py, domain_loader.py만 수정 필요)
- metadata.subtask로 필요시 subtask별 분석 가능

---

## 4. Task Decomposition

### 4.1 Task List

| ID | Task | Dependencies | Files | Priority |
|----|------|--------------|-------|----------|
| T1 | BBH 통합 저장 함수 수정 | - | `utils/dataset_preparer.py` | P0 |
| T2 | domains.py BBH 설정 통합 | T1 | `config/domains.py` | P0 |
| T3 | domain_loader.py BBH 설정 통합 | T1 | `utils/domain_loader.py` | P0 |
| T3.1 | dataset_config.py BBH 설정 통합 | T1 | `config/dataset_config.py` | P0 |
| T4 | 메타데이터 기반 샘플링 함수 추가 | - | `utils/sample_extractor.py` | P0 |
| T5 | MATH 직접 로드 제거 | T4 | `utils/sample_extractor.py` | P1 |
| T6 | 기존 BBH 개별 파일 정리 | T1-T3.1 | `data/logical/eval/data/` | P2 |

### 4.2 Dependency Graph

```
T1 (BBH 통합) ──┬──► T2 (domains.py)
               ├──► T3 (domain_loader.py)
               └──► T6 (파일 정리)

T4 (메타데이터 샘플링) ──► T5 (MATH 직접 로드 제거)
```

---

## 5. Implementation Strategy

### 5.1 Task T1: BBH 통합 저장 함수 수정

**File**: `utils/dataset_preparer.py`

**Changes**:
```python
# Before: process_bbh() - subtask별 파일 저장
def process_bbh(eval_dir: Path, subtasks: List[str]):
    for subtask in subtasks:
        # ... 각 subtask를 별도 파일로 저장
        save_json(records, eval_dir / f"bbh_{subtask}_test.json")

# After: process_bbh() - 통합 파일 저장
def process_bbh(eval_dir: Path, subtasks: List[str]):
    all_records = []
    for subtask in subtasks:
        # ... subtask별 데이터 수집
        for item in data:
            records.append({
                "instruction": prompt,  # subtask별 instruction 유지
                "input": input_text,
                "output": format_output(target),
                "metadata": {"subtask": subtask}  # subtask 정보 유지
            })
        all_records.extend(records)

    # 단일 파일로 저장
    save_json(all_records, eval_dir / "bbh_test.json")
```

### 5.2 Task T2-T3: Config 파일 수정

**File**: `config/domains.py`

```python
# Before
"eval_datasets": [
    "reclor", "anli_r2", "anli_r3",
    "bbh_boolean_expressions", "bbh_formal_fallacies",
    # ... 9개 개별 항목
]

# After
"eval_datasets": [
    "reclor", "anli_r2", "anli_r3",
    "bbh"  # 단일 항목
]
```

**File**: `utils/domain_loader.py`

```python
# Before: 9개 개별 BBH 설정
"bbh_boolean_expressions": {"filename": "bbh_boolean_expressions_test.json", ...},
# ...

# After: 단일 BBH 설정
"bbh": {"filename": "bbh_test.json", "answer_type": AnswerType.TEXT},
```

### 5.3 Task T4: 메타데이터 기반 샘플링 함수

**File**: `utils/sample_extractor.py`

```python
def extract_stratified_samples(
    data: List[Dict],
    num_samples: int,
    primary_key: str = "type",      # 1차 계층
    secondary_key: str = "level",   # 2차 계층 (optional)
    text_key: str = "input"         # 길이 측정용
) -> List[Dict]:
    """
    메타데이터 기반 계층적 샘플링

    Strategy: type > level > length
    1. primary_key로 그룹화하여 균등 분배
    2. 각 그룹 내에서 secondary_key로 다시 균등 분배
    3. 최하위에서 length 다양성 확보

    메타데이터가 없거나 의미 없으면 기존 diverse 샘플링으로 fallback
    """
    # 메타데이터 존재 여부 확인
    has_primary = any(item.get("metadata", {}).get(primary_key) for item in data)

    if not has_primary:
        # fallback to diverse sampling
        return extract_diverse_samples(data, num_samples, text_key)

    # 계층적 샘플링 로직
    ...
```

### 5.4 Task T5: MATH 직접 로드 제거

```python
# Before: MATH는 Hugging Face에서 직접 로드
def extract_math_stratified_samples(data, num_samples):
    hf_data = load_math_from_huggingface()  # 제거
    ...

# After: 로컬 데이터의 메타데이터 활용
def extract_samples(domain, dataset, ...):
    # 로컬 데이터 로드 (이미 metadata 포함)
    with open(train_file) as f:
        data = json.load(f)

    # 메타데이터 기반 샘플링 (통합 함수 사용)
    if config.get("use_stratified"):
        samples = extract_stratified_samples(
            data, num_samples,
            primary_key="type",
            secondary_key="level"
        )
```

---

## 6. Parallel Agent Execution Plan

### 6.1 Execution Mode

- [x] **Sequential** (dependencies exist between tasks)
- [ ] Parallel
- [ ] Competitive

### 6.2 Execution Order

```
Phase 1: [T1, T4] - 병렬 가능 (독립적)
  T1: BBH 통합 저장
  T4: 메타데이터 샘플링 함수

Phase 2: [T2, T3, T5] - 병렬 가능
  T2: domains.py 수정 (T1 의존)
  T3: domain_loader.py 수정 (T1 의존)
  T5: MATH 직접 로드 제거 (T4 의존)

Phase 3: [T6]
  T6: 기존 BBH 파일 정리 (T1-T3 완료 후)
```

---

## 7. Quality Gates

### Phase 1 Verification
- [ ] `process_bbh()` 함수가 단일 `bbh_test.json` 생성
- [ ] 각 레코드의 `metadata.subtask`에 subtask 이름 포함
- [ ] 각 레코드의 `instruction`이 해당 subtask의 prompt 유지
- [ ] `extract_stratified_samples()` 함수 구현 완료

### Phase 2 Verification
- [ ] `domains.py`의 `eval_datasets`에 단일 "bbh" 항목
- [ ] `domain_loader.py`의 `eval_datasets`에 단일 "bbh" 항목
- [ ] `sample_extractor.py`에서 MATH Hugging Face 로드 제거
- [ ] 기존 테스트 통과

### Phase 3 Verification
- [ ] `data/logical/eval/data/bbh_*.json` 개별 파일 삭제됨
- [ ] 전체 시스템 동작 확인

### Final Verification
- [ ] `python utils/dataset_preparer.py` 실행 성공
- [ ] `python -m utils.sample_extractor --domain math --dataset math` 성공
- [ ] 평가 시 BBH 단일 inference 확인

---

## 8. Rollback Plan

실패 시 복구 방법:
1. Git으로 변경사항 revert
2. 기존 BBH 개별 파일은 `dataset_preparer.py` 재실행으로 복구 가능

---

## 9. File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `utils/dataset_preparer.py` | Modify | `process_bbh()` 통합 저장으로 수정 |
| `config/domains.py` | Modify | BBH 개별 항목 → 단일 "bbh" |
| `utils/domain_loader.py` | Modify | BBH 개별 설정 → 단일 설정 |
| `config/dataset_config.py` | Modify | BBH 개별 config 항목 통합/정리 |
| `utils/sample_extractor.py` | Modify | 메타데이터 기반 샘플링 추가, MATH 직접 로드 제거 |
| `data/logical/eval/data/bbh_*_test.json` | Delete | 9개 개별 파일 삭제 |
| `data/logical/eval/data/bbh_test.json` | Create | 통합 파일 (dataset_preparer 실행 시) |
