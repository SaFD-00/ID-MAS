# Plan: Enhanced Data 파일명 패턴 변경

## 1. 문제 분석

### 현재 상태
- 파일명 패턴: `{dataset}_train_ID-MAS_{teacher_model}.json`
- 예: `gsm8k_train_ID-MAS_Qwen2.5-7B-Instruct.json`

### 목표 상태
- 파일명 패턴: `{dataset}_train_ID-MAS_{teacher_model}_{student_model}.json`
- 예: `gsm8k_train_ID-MAS_Qwen2.5-7B-Instruct_Qwen2.5-3B-Instruct.json`

### 추가 요구사항
- 메타데이터 파일 삭제 및 관련 코드 제거
- 기존 파일 삭제 (마이그레이션 불필요)

---

## 2. 요구사항 명세

### Acceptance Criteria
- [ ] AC1: Enhanced data 파일이 `{dataset}_train_ID-MAS_{teacher}_{student}.json` 패턴으로 생성됨
- [ ] AC2: 메타데이터 저장 코드가 제거됨
- [ ] AC3: Enhanced data 로드 시 새 패턴으로 파일을 찾음
- [ ] AC4: 파일명 파싱 로직이 새 패턴을 올바르게 파싱함
- [ ] AC5: 기존 단일 모델 패턴 파일은 무시됨

---

## 3. 아키텍처 설계

### 변경 대상 파일

| 파일 | 변경 유형 | 영향도 |
|------|----------|--------|
| `main.py` | 수정 | 높음 |
| `utils/dataset_enhancer.py` | 수정 | 중간 |
| `utils/domain_loader.py` | 수정 | 중간 |

### 접근 방식
**단일 접근 방식 선택**: 파일명 패턴에 student_model suffix 추가

- 장점: 명확한 구분, teacher/student 조합별 고유 파일
- 구분자: 언더스코어(`_`) 유지
- 패턴: `{dataset}_train_ID-MAS_{teacher}_{student}.json`

---

## 4. 태스크 분해

### Task List

| ID | Task | 의존성 | 파일 | 우선순위 |
|----|------|--------|------|----------|
| T1 | dataset_enhancer.py - student_suffix 지원 추가 | - | utils/dataset_enhancer.py | P0 |
| T2 | dataset_enhancer.py - 메타데이터 코드 제거 | - | utils/dataset_enhancer.py | P0 |
| T3 | main.py - Enhanced data 생성 경로 변경 | T1 | main.py | P0 |
| T4 | main.py - 메타데이터 저장 코드 제거 | - | main.py | P0 |
| T5 | main.py - Enhanced data 로드 경로 변경 | T3 | main.py | P0 |
| T6 | domain_loader.py - 파일명 패턴 및 파싱 변경 | - | utils/domain_loader.py | P0 |
| T7 | main.py - load_enhanced_training_data 호출 수정 | T6 | main.py | P0 |

### 의존성 그래프
```
T1 ──► T3 ──► T5
T2 (독립)
T4 (독립)
T6 ──► T7
```

---

## 5. 구현 전략

### 5.1 파일별 상세 변경 사항

#### T1: utils/dataset_enhancer.py - student_suffix 지원 추가

**변경 위치**: `__init__` 메서드 및 `_get_output_path` 메서드

```python
# 변경 전 (__init__)
def __init__(self, teacher_config: dict = None, model_suffix: str = None):
    self.model_suffix = model_suffix

# 변경 후
def __init__(self, teacher_config: dict = None, model_suffix: str = None, student_suffix: str = None):
    self.model_suffix = model_suffix
    self.student_suffix = student_suffix
```

```python
# 변경 전 (_get_output_path)
return PROJECT_ROOT / "data" / domain / "train" / "data" / f"{dataset}_train_ID-MAS_{suffix}.json"

# 변경 후
if self.student_suffix:
    return PROJECT_ROOT / "data" / domain / "train" / "data" / f"{dataset}_train_ID-MAS_{suffix}_{self.student_suffix}.json"
else:
    return PROJECT_ROOT / "data" / domain / "train" / "data" / f"{dataset}_train_ID-MAS_{suffix}.json"
```

#### T2: utils/dataset_enhancer.py - 메타데이터 코드 제거

**삭제 대상**: `_save_metadata` 메서드 전체 (라인 167-198)
**삭제 대상**: `enhance_dataset`에서 `_save_metadata` 호출 부분

#### T3: main.py - Enhanced data 생성 경로 변경

**변경 위치**: 라인 352-353

```python
# 변경 전
model_suffix = get_model_short_name(self.teacher_model_name)
output_path = self.raw_data_dir / f"{self.train_dataset}_train_ID-MAS_{model_suffix}.json"

# 변경 후
teacher_suffix = get_model_short_name(self.teacher_model_name)
student_suffix = get_model_short_name(self.student_model_name)
output_path = self.raw_data_dir / f"{self.train_dataset}_train_ID-MAS_{teacher_suffix}_{student_suffix}.json"
```

#### T4: main.py - 메타데이터 저장 코드 제거

**삭제 대상**: 라인 362-376 (메타데이터 저장 블록 전체)

#### T5: main.py - Enhanced data 로드 경로 변경

**변경 위치**: 라인 780-781

```python
# 변경 전
model_suffix = get_model_short_name(pipeline.teacher_model_name)
enhanced_path = pipeline.raw_data_dir / f"{pipeline.train_dataset}_train_ID-MAS_{model_suffix}.json"

# 변경 후
teacher_suffix = get_model_short_name(pipeline.teacher_model_name)
student_suffix = get_model_short_name(pipeline.student_model_name)
enhanced_path = pipeline.raw_data_dir / f"{pipeline.train_dataset}_train_ID-MAS_{teacher_suffix}_{student_suffix}.json"
```

#### T6: utils/domain_loader.py - 파일명 패턴 및 파싱 변경

**변경 위치 1**: `load_enhanced_training_data` 메서드 시그니처 및 파일명 생성

```python
# 변경 전
def load_enhanced_training_data(self, dataset: str, model_suffix: str, ...) -> List[Dict]:
    filename = f"{dataset}_train_ID-MAS_{model_suffix}.json"

# 변경 후
def load_enhanced_training_data(self, dataset: str, teacher_suffix: str, student_suffix: str, ...) -> List[Dict]:
    filename = f"{dataset}_train_ID-MAS_{teacher_suffix}_{student_suffix}.json"
```

**변경 위치 2**: `get_available_enhanced_data` 메서드 파싱 로직

```python
# 변경 전
pattern = "*_train_ID-MAS_*.json"
parts = name.split("_train_ID-MAS_")
if len(parts) == 2:
    ds_name, model_suffix = parts

# 변경 후
pattern = "*_train_ID-MAS_*_*.json"
parts = name.split("_train_ID-MAS_")
if len(parts) == 2:
    ds_name = parts[0]
    suffix_parts = parts[1].split("_", 1)  # teacher_student 분리
    if len(suffix_parts) == 2:
        teacher_suffix, student_suffix = suffix_parts
```

#### T7: main.py - load_enhanced_training_data 호출 수정

**변경 위치**: 라인 402-412 (run_learning_phase 메서드 내)

```python
# 변경 전
if model_suffix is None:
    model_suffix = get_model_short_name(self.teacher_model_name)
questions = self.loader.load_enhanced_training_data(
    dataset=self.train_dataset,
    model_suffix=model_suffix,
    limit=num_questions,
    shuffle=False
)

# 변경 후
teacher_suffix = get_model_short_name(self.teacher_model_name)
student_suffix = get_model_short_name(self.student_model_name)
questions = self.loader.load_enhanced_training_data(
    dataset=self.train_dataset,
    teacher_suffix=teacher_suffix,
    student_suffix=student_suffix,
    limit=num_questions,
    shuffle=False
)
```

---

## 6. 병렬 에이전트 실행 계획

### 6.1 실행 모드
- [x] Sequential (의존성 있음, 순서 중요)

### 6.2 실행 순서
```
Group 1 (순차): [T1, T2] → dataset_enhancer.py 수정
Group 2 (순차): [T6] → domain_loader.py 수정
Group 3 (순차): [T3, T4, T5, T7] → main.py 수정
Group 4: 검증
```

### 6.3 통합 전략
1. 각 파일별 변경 완료 후 구문 검증 (`python -m py_compile`)
2. 전체 변경 완료 후 통합 테스트

---

## 7. Quality Gates

### Phase 1: Context & Problem
- [x] 코드베이스 컨텍스트 수집 완료
- [x] 문제/목표 식별 완료
- [x] 변경 위치 파악 완료

### Phase 2: Requirements Clarification
- [x] 목표 정의 확인 (사용자 확인 완료)
- [x] 범위 정의: Enhanced data 파일명 + 메타데이터 제거
- [x] 제약 사항: 기존 파일 삭제, 마이그레이션 불필요

### Phase 3: Spec & Architecture
- [x] 테스트 가능한 AC 존재
- [x] 아키텍처 접근 방식 선정 완료
- [x] 파일별 변경 사항 정의 완료

### Phase 4: Task & Execution
- [x] 태스크 분해 완료 (6개 태스크)
- [x] 의존성 매핑 완료
- [x] 검증 단계 정의 완료

### Implementation Readiness
- [x] 모든 태스크에 명확한 입력/출력 있음
- [x] 통합 전략 정의 완료
- [ ] 구현 대기 중

---

## 8. 구현 체크리스트

### utils/dataset_enhancer.py
- [ ] `__init__`에 `student_suffix` 매개변수 추가
- [ ] `_get_output_path`에 student_suffix 포함 로직 추가
- [ ] `_save_metadata` 메서드 삭제
- [ ] `enhance_dataset`에서 `_save_metadata` 호출 제거

### main.py (TrainPipeline.generate_enhanced_data)
- [ ] 파일명에 teacher_suffix + student_suffix 모두 포함
- [ ] 메타데이터 저장 코드 블록 삭제 (라인 362-376)

### main.py (train 함수)
- [ ] Enhanced data 로드 경로에 student_suffix 추가

### main.py (TrainPipeline.run_learning_phase)
- [ ] `load_enhanced_training_data` 호출 시 teacher_suffix, student_suffix 전달

### utils/domain_loader.py
- [ ] `load_enhanced_training_data`에 student_suffix 매개변수 추가
- [ ] 파일명 생성 패턴 변경
- [ ] `get_available_enhanced_data` 파싱 로직 변경

### 검증
- [ ] 모든 파일 구문 검증 통과
- [ ] 실제 파이프라인 실행 테스트
