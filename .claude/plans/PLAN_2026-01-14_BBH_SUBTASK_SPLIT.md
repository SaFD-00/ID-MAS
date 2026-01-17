# BBH 서브태스크별 분리 저장 및 평가 구현 계획

**작성일**: 2026-01-14
**상태**: Draft

---

## 1. Problem Analysis

### 1.1 현재 상태
- BBH(Big Bench Hard) 데이터셋의 9개 논리 추론 서브태스크가 단일 파일(`bbh_test.json`)로 통합 저장됨
- 평가 시 2,250개 레코드(9개 서브태스크 × ~250개)가 한꺼번에 평가됨
- 서브태스크별 개별 성능 분석이 불가능

### 1.2 목표 상태
- 각 서브태스크를 개별 JSON 파일로 저장
- 서브태스크별로 독립적인 평가 가능
- 서브태스크별 개별 평가 결과 파일 생성

### 1.3 대상 서브태스크 (9개)
| 서브태스크 | 파일명 |
|-----------|--------|
| boolean_expressions | bbh_boolean_expressions_test.json |
| formal_fallacies | bbh_formal_fallacies_test.json |
| logical_deduction_three_objects | bbh_logical_deduction_three_objects_test.json |
| logical_deduction_five_objects | bbh_logical_deduction_five_objects_test.json |
| logical_deduction_seven_objects | bbh_logical_deduction_seven_objects_test.json |
| tracking_shuffled_objects_three_objects | bbh_tracking_shuffled_objects_three_objects_test.json |
| tracking_shuffled_objects_five_objects | bbh_tracking_shuffled_objects_five_objects_test.json |
| tracking_shuffled_objects_seven_objects | bbh_tracking_shuffled_objects_seven_objects_test.json |
| web_of_lies | bbh_web_of_lies_test.json |

---

## 2. Requirements Specification

### 2.1 Acceptance Criteria

**AC1: 데이터셋 준비 (dataset_preparer.py)**
- [ ] `process_bbh()` 함수가 각 서브태스크를 개별 JSON 파일로 저장
- [ ] 파일명 형식: `bbh_{subtask_name}_test.json`
- [ ] 각 파일에 해당 서브태스크의 레코드만 포함
- [ ] 기존 통합 파일(`bbh_test.json`) 생성 제거

**AC2: 데이터셋 등록 (domain_loader.py)**
- [ ] DOMAIN_CONFIG["logical"]["eval_datasets"]에 9개 서브태스크 등록
- [ ] 각 서브태스크가 독립적인 eval 데이터셋으로 인식
- [ ] 기존 `bbh` 엔트리 제거

**AC3: 평가 실행**
- [ ] `--eval-dataset=bbh_boolean_expressions` 식으로 개별 평가 가능
- [ ] 각 평가가 해당 서브태스크 데이터만 로드

**AC4: 결과 저장**
- [ ] 결과 파일명: `bbh_{subtask}_eval_results-{Method}.json`
- [ ] 각 서브태스크별 독립적인 정확도 계산

---

## 3. Architecture Design

### 3.1 접근 방식 비교

| 접근법 | 장점 | 단점 |
|-------|------|------|
| **A. 개별 파일 분리 (선택)** | 단순, 명확, 기존 로직 재사용 | 파일 수 증가 |
| B. 파라미터 기반 필터링 | 파일 하나 유지 | 로직 복잡, 메타데이터 필요 |
| C. 동적 서브셋 로딩 | 유연성 | 큰 구조 변경 필요 |

### 3.2 선택: 접근법 A (개별 파일 분리)
- 기존 코드 구조와 일관성 유지
- 수정 범위 최소화
- 명확한 파일-데이터셋 매핑

### 3.3 데이터 흐름

```
[HuggingFace: lukaemon/bbh]
         │
         ├── boolean_expressions ──► bbh_boolean_expressions_test.json
         ├── formal_fallacies ──► bbh_formal_fallacies_test.json
         ├── logical_deduction_three_objects ──► bbh_logical_deduction_three_objects_test.json
         ├── logical_deduction_five_objects ──► bbh_logical_deduction_five_objects_test.json
         ├── logical_deduction_seven_objects ──► bbh_logical_deduction_seven_objects_test.json
         ├── tracking_shuffled_objects_three_objects ──► bbh_tracking_shuffled_objects_three_objects_test.json
         ├── tracking_shuffled_objects_five_objects ──► bbh_tracking_shuffled_objects_five_objects_test.json
         ├── tracking_shuffled_objects_seven_objects ──► bbh_tracking_shuffled_objects_seven_objects_test.json
         └── web_of_lies ──► bbh_web_of_lies_test.json
```

---

## 4. Task Decomposition

### Task List

| ID | Task | Dependencies | Files | Priority |
|----|------|--------------|-------|----------|
| T1 | `process_bbh()` 함수 수정 - 서브태스크별 개별 저장 | - | utils/dataset_preparer.py | P0 |
| T2 | DOMAIN_CONFIG 수정 - 9개 서브태스크 등록 | - | utils/domain_loader.py | P0 |
| T3 | 기존 bbh_test.json 삭제 | T1 | data/logical/eval/data/ | P1 |
| T4 | 데이터셋 재생성 실행 | T1 | - | P1 |
| T5 | 평가 테스트 | T1, T2, T4 | - | P2 |

### Dependency Graph

```
T1 ──┬──► T3
     │
     ├──► T4 ──► T5
     │         ↑
T2 ──┴─────────┘
```

---

## 5. Implementation Strategy

### 5.1 Code Writing Method

각 태스크는 다음 순서로 진행:
1. 기존 코드 백업/확인
2. 최소한의 수정으로 구현
3. 동작 검증

### 5.2 Per-Task Implementation Guide

#### T1: process_bbh() 함수 수정

**파일**: `utils/dataset_preparer.py` (라인 696-737)

**현재 코드**:
```python
def process_bbh(eval_dir: Path, subtasks: List[str]):
    all_records = []  # 모든 서브태스크 통합
    for subtask in subtasks:
        # ... 로드 로직 ...
        all_records.append(...)

    # 단일 파일로 저장
    save_json(all_records, eval_dir / "bbh_test.json")
```

**수정 후 코드**:
```python
def process_bbh(eval_dir: Path, subtasks: List[str]):
    for subtask in subtasks:
        records = []  # 서브태스크별 개별 리스트
        # ... 로드 로직 ...
        records.append(...)

        # 개별 파일로 저장
        save_json(records, eval_dir / f"bbh_{subtask}_test.json")
```

#### T2: DOMAIN_CONFIG 수정

**파일**: `utils/domain_loader.py` (라인 76-81)

**현재 코드**:
```python
"eval_datasets": {
    "reclor": {"filename": "reclor_test.json", "answer_type": AnswerType.MCQ},
    "anli_r2": {"filename": "anli_r2_test.json", "answer_type": AnswerType.MCQ},
    "anli_r3": {"filename": "anli_r3_test.json", "answer_type": AnswerType.MCQ},
    "bbh": {"filename": "bbh_test.json", "answer_type": AnswerType.TEXT},
},
```

**수정 후 코드**:
```python
"eval_datasets": {
    "reclor": {"filename": "reclor_test.json", "answer_type": AnswerType.MCQ},
    "anli_r2": {"filename": "anli_r2_test.json", "answer_type": AnswerType.MCQ},
    "anli_r3": {"filename": "anli_r3_test.json", "answer_type": AnswerType.MCQ},
    # BBH 서브태스크별 개별 등록
    "bbh_boolean_expressions": {"filename": "bbh_boolean_expressions_test.json", "answer_type": AnswerType.TEXT},
    "bbh_formal_fallacies": {"filename": "bbh_formal_fallacies_test.json", "answer_type": AnswerType.TEXT},
    "bbh_logical_deduction_three_objects": {"filename": "bbh_logical_deduction_three_objects_test.json", "answer_type": AnswerType.TEXT},
    "bbh_logical_deduction_five_objects": {"filename": "bbh_logical_deduction_five_objects_test.json", "answer_type": AnswerType.TEXT},
    "bbh_logical_deduction_seven_objects": {"filename": "bbh_logical_deduction_seven_objects_test.json", "answer_type": AnswerType.TEXT},
    "bbh_tracking_shuffled_objects_three_objects": {"filename": "bbh_tracking_shuffled_objects_three_objects_test.json", "answer_type": AnswerType.TEXT},
    "bbh_tracking_shuffled_objects_five_objects": {"filename": "bbh_tracking_shuffled_objects_five_objects_test.json", "answer_type": AnswerType.TEXT},
    "bbh_tracking_shuffled_objects_seven_objects": {"filename": "bbh_tracking_shuffled_objects_seven_objects_test.json", "answer_type": AnswerType.TEXT},
    "bbh_web_of_lies": {"filename": "bbh_web_of_lies_test.json", "answer_type": AnswerType.TEXT},
},
```

#### T3: 기존 bbh_test.json 삭제

```bash
rm data/logical/eval/data/bbh_test.json
```

#### T4: 데이터셋 재생성 실행

```bash
python utils/dataset_preparer.py
```

#### T5: 평가 테스트

```bash
# 서브태스크 개별 평가 테스트
python main.py evaluate --domain logical --eval-dataset bbh_boolean_expressions --limit 5
```

---

## 6. Parallel Agent Execution Plan

### 6.1 Execution Mode
- [x] Sequential (dependencies, shared state)
- 태스크 간 의존성이 있어 순차 실행 필요

### 6.2 Execution Order

```
T1 (process_bbh 수정)
    │
    ▼
T2 (DOMAIN_CONFIG 수정) ─ 병렬 가능하나 순차 권장
    │
    ▼
T3 (기존 파일 삭제)
    │
    ▼
T4 (데이터셋 재생성)
    │
    ▼
T5 (평가 테스트)
```

---

## 7. Quality Gates

### Phase 1: Context & Problem
- [x] 코드베이스 컨텍스트 수집
- [x] 문제/목표 식별
- [x] 현재 BBH 처리 로직 분석

### Phase 2: Requirements Clarification
- [x] 사용자와 요구사항 확인
- [x] 파일 구조: 완전 대체
- [x] 평가 방식: 개별 평가만
- [x] 결과 저장: 서브태스크별 개별 파일

### Phase 3: Spec & Architecture
- [x] Testable AC 정의
- [x] 아키텍처 접근법 선택 (개별 파일 분리)

### Phase 4: Task & Execution
- [x] 태스크 분해 완료
- [x] 의존성 매핑
- [x] 구현 가이드 작성

### Implementation Readiness
- [x] 모든 태스크에 명확한 입력/출력 정의
- [x] 수정할 파일과 라인 식별
- [ ] 롤백 계획: git으로 복원 가능

---

## 8. Usage After Implementation

### 평가 실행 예시

```bash
# 개별 서브태스크 평가
python main.py evaluate \
    --domain logical \
    --eval-dataset bbh_boolean_expressions \
    --student-model llama-2-7b \
    --eval-method baseline

# 다른 서브태스크
python main.py evaluate --domain logical --eval-dataset bbh_formal_fallacies
python main.py evaluate --domain logical --eval-dataset bbh_web_of_lies
```

### 결과 파일 위치

```
data/logical/eval/{model_short}/
├── bbh_boolean_expressions_eval_results-Baseline.json
├── bbh_formal_fallacies_eval_results-Baseline.json
├── bbh_logical_deduction_three_objects_eval_results-Baseline.json
├── ...
└── bbh_web_of_lies_eval_results-Baseline.json
```

---

## 9. Rollback Plan

문제 발생 시:
```bash
git checkout -- utils/dataset_preparer.py utils/domain_loader.py
```

데이터 복원:
```bash
python utils/dataset_preparer.py  # 원본 코드로 재생성
```
