# Plan: Teacher Model 계층 추가 (디렉토리 구조 변경)

## Overview

학습 결과물의 저장 경로에 Teacher Model 계층을 추가하여 동일한 Student Model이 다른 Teacher Model로 학습한 결과를 구분할 수 있도록 합니다.

### 변경 요약

| 항목 | 현재 | 변경 후 |
|------|------|---------|
| Train 경로 | `data/{domain}/train/{student_model}/` | `data/{domain}/train/{teacher_model}/{student_model}/` |
| Eval 경로 | `data/{domain}/eval/{student_model}/` | **변경 없음** |

### 영향 범위

- Train 모드만 변경
- 기존 데이터 마이그레이션 불필요 (새로 학습 시작)

---

## Implementation Steps

### Step 1: `config/config.py` - `get_domain_data_dirs()` 함수 수정

**파일**: `config/config.py:337-380`

**변경 내용**:
1. `teacher_model_name` 파라미터 추가
2. Train 모드에서 경로 생성 시 teacher_model 계층 포함
3. Eval 모드는 기존 유지

**변경 전**:
```python
def get_domain_data_dirs(domain: str, model_name: str = None, train_dataset: str = None, mode: str = "train") -> dict:
    ...
    if mode == "train":
        model_dir = DATA_DIR / domain / "train" / model_short
```

**변경 후**:
```python
def get_domain_data_dirs(domain: str, model_name: str = None, train_dataset: str = None, mode: str = "train", teacher_model_name: str = None) -> dict:
    ...
    if mode == "train":
        teacher_short = get_model_short_name(teacher_model_name) if teacher_model_name else "default-teacher"
        model_dir = DATA_DIR / domain / "train" / teacher_short / model_short
```

**완료 기준**:
- [ ] 함수 시그니처에 `teacher_model_name` 파라미터 추가
- [ ] Train 모드에서 teacher_model 디렉토리 계층 추가
- [ ] Eval 모드는 기존 로직 유지
- [ ] 독스트링 업데이트

---

### Step 2: `config/paths.py` - 동일 함수 업데이트 (하위 호환성)

**파일**: `config/paths.py:88-141`

**변경 내용**:
- `config/config.py`와 동일한 변경 적용
- 이 파일은 deprecated 버전으로 보이지만 아직 사용되고 있을 수 있음

**완료 기준**:
- [ ] `config/config.py`와 동일하게 수정
- [ ] 함수 시그니처 동기화

---

### Step 3: `main.py` - `IDMASPipeline` 클래스 수정

**파일**: `main.py:71-136`

**변경 내용**:
1. `get_domain_data_dirs()` 호출 시 `teacher_model_name` 전달
2. `teacher_model_name`은 이미 `self.teacher_model_name`으로 저장되어 있음 (line 128)

**변경 전** (line 118):
```python
self.model_dirs = get_domain_data_dirs(domain, self.student_model_name, self.train_dataset, mode="train")
```

**변경 후**:
```python
self.model_dirs = get_domain_data_dirs(
    domain,
    self.student_model_name,
    self.train_dataset,
    mode="train",
    teacher_model_name=self.teacher_model_name
)
```

**완료 기준**:
- [ ] `get_domain_data_dirs()` 호출에 teacher_model_name 전달
- [ ] 정상 동작 확인

---

### Step 4: `main.py` - `get_design_output_dir()` 검토

**파일**: `config/config.py:106-122`

**검토 내용**:
- Design 출력은 `data/{domain}/train/instructional-design/`에 저장됨
- Teacher Model별로 다른 설계가 필요한지 확인 필요

**결정**: Design은 Terminal Goal 기반이므로 Teacher Model과 무관 - **변경 불필요**

---

### Step 5: 테스트

**테스트 시나리오**:

1. **경로 생성 테스트**:
   ```bash
   python -c "
   from config.config import get_domain_data_dirs, DEFAULT_TEACHER_MODEL
   dirs = get_domain_data_dirs('math', 'Qwen/Qwen2.5-3B-Instruct', 'gsm8k', 'train', DEFAULT_TEACHER_MODEL)
   print(dirs['model_dir'])
   # Expected: data/math/train/gpt-5-2025-08-07/Qwen2.5-3B-Instruct/
   "
   ```

2. **Train 모드 실행 테스트**:
   ```bash
   python main.py --mode train --domain math --train-dataset gsm8k --resume False
   ```

3. **Eval 모드 확인** (변경 없음):
   ```bash
   python main.py --mode eval --method baseline --domain math --eval-dataset gsm8k
   ```

**완료 기준**:
- [ ] 경로 생성이 예상대로 동작
- [ ] Train 모드에서 새 경로에 파일 저장
- [ ] Eval 모드는 기존대로 동작

---

## File Changes Summary

| 파일 | 변경 유형 | 변경량 |
|------|----------|--------|
| `config/config.py` | 수정 | ~15줄 |
| `config/paths.py` | 수정 | ~15줄 |
| `main.py` | 수정 | ~5줄 |

---

## Risks & Mitigations

| 리스크 | 영향도 | 완화 방안 |
|--------|--------|----------|
| 기존 체크포인트 호환성 | 낮음 | 새 경로에서 학습 시작, 기존 데이터 수동 관리 |
| 다른 코드에서 경로 직접 사용 | 중간 | Grep으로 `data/math/train` 패턴 검색하여 확인 |

---

## Definition of Done

- [ ] `config/config.py` 수정 완료
- [ ] `config/paths.py` 수정 완료
- [ ] `main.py` 수정 완료
- [ ] 경로 생성 테스트 통과
- [ ] Train 모드 실행 테스트 통과
- [ ] Eval 모드 동작 확인
