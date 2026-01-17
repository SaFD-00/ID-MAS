# ReClor 오류 수정 및 MMLU 제거 계획

**작성일**: 2026-01-10
**목표**:
1. ReClor 데이터셋 로딩 오류 수정 (Dataset scripts 지원 중단 문제)
2. MMLU 데이터셋 관련 코드 완전 제거

---

## 📋 작업 개요

### 문제 1: ReClor 데이터셋 로딩 실패
**오류 메시지**:
```
RuntimeError: Dataset scripts are no longer supported, but found reclor.py
```

**원인**: `community-datasets/reclor`는 구형 dataset script 방식을 사용하며, 최신 HuggingFace datasets 라이브러리에서 지원하지 않음

**해결 방안**: `sxiong/ReClor` 데이터셋으로 변경 (2025년 11월 업데이트, 표준 JSON 포맷)

### 문제 2: MMLU 제거
**이유**: Math 도메인에서 MMLU 평가 제외

**영향 범위**: 10개 파일 수정 필요

---

## 🎯 Task 1: ReClor 데이터셋 경로 변경

### 1.1 config/dataset_config.py 수정

**위치**: Line 79-87

**변경 전**:
```python
"reclor": {
    "hf_name": "community-datasets/reclor",
    "hf_config": None,
    "train_split": "train",
    "test_split": "test",
    "answer_type": "mcq",
    "domain": "logical",
    "description": "ReClor - Logical reasoning from standardized tests",
},
```

**변경 후**:
```python
"reclor": {
    "hf_name": "sxiong/ReClor",
    "hf_config": None,
    "train_split": "train",
    "test_split": "test",
    "answer_type": "mcq",
    "domain": "logical",
    "description": "ReClor - Logical reasoning from standardized tests",
},
```

### 1.2 utils/dataset_preparer.py 수정

**위치**: Line 512 (process_reclor 함수)

**변경 전**:
```python
def process_reclor(train_dir: Path, eval_dir: Path):
    print("\n[ReClor] Processing...")
    dataset_id = "community-datasets/reclor"
```

**변경 후**:
```python
def process_reclor(train_dir: Path, eval_dir: Path):
    print("\n[ReClor] Processing...")
    dataset_id = "sxiong/ReClor"
```

**검증 방법**:
```bash
# 데이터 다운로드 테스트
python3 -m utils.dataset_preparer --domain logical
```

---

## 🗑️ Task 2: MMLU 완전 제거

### 2.1 utils/dataset_preparer.py

**삭제할 부분**:

1. **DATASET_PROMPTS에서 MMLU 프롬프트 삭제** (Line 66-70):
```python
# 삭제
    # MMLU - Multiple choice questions
    "mmlu": """You are a helpful math assistant.
This is a multiple choice question. Solve the problem step by step, then select the correct answer from the given options (A, B, C, or D).
Your final answer MUST be a single letter (A, B, C, or D) within \\boxed{}.
Example: \\boxed{A}""",
```

2. **MMLU_MATH_SUBJECTS 리스트 삭제** (Line 136-143):
```python
# 삭제
# MMLU Math subjects
MMLU_MATH_SUBJECTS = [
    "abstract_algebra",
    "college_mathematics",
    "elementary_mathematics",
    "high_school_mathematics",
    "high_school_statistics",
]
```

3. **process_mmlu 함수 삭제** (Line 340-376):
```python
# 전체 함수 삭제
def process_mmlu(output_dir: Path, subjects: List[str], output_filename: str, is_math_domain: bool = False):
    ...
```

4. **main() 함수에서 MMLU 호출 제거** (Line 773-774):
```python
# 삭제
    # 3. MMLU Math subjects (test only)
    process_mmlu(eval_dir, MMLU_MATH_SUBJECTS, "mmlu_test.json", is_math_domain=True)
```

### 2.2 config/dataset_config.py

**삭제할 부분**:

1. **MMLU 설정 삭제** (Line 68-76):
```python
# 삭제
    "mmlu": {
        "hf_name": "cais/mmlu",
        "hf_config": "required",  # Subject required
        "train_split": "validation",  # MMLU uses validation for training
        "test_split": "test",
        "answer_type": "mcq",
        "domain": "science_knowledge",
        "description": "Massive Multitask Language Understanding - 57 subjects",
    },
```

2. **MMLU_STEM_SUBJECTS 리스트 삭제** (Line 320-341):
```python
# 삭제 (전체 리스트)
# MMLU STEM subjects for reference
MMLU_STEM_SUBJECTS = [
    ...
]
```

### 2.3 config/domains.py

**수정 위치**: Line 46

**변경 전**:
```python
"eval_datasets": ["gsm8k", "math", "svamp", "asdiv", "mawps", "mmlu"],
```

**변경 후**:
```python
"eval_datasets": ["gsm8k", "math", "svamp", "asdiv", "mawps"],
```

### 2.4 config/config.py

**수정 위치**: Line 323

**변경 전**:
```python
        "eval_datasets": ["gsm8k", "math", "svamp", "asdiv", "mawps", "mmlu"],
```

**변경 후**:
```python
        "eval_datasets": ["gsm8k", "math", "svamp", "asdiv", "mawps"],
```

### 2.5 utils/domain_loader.py

**삭제할 부분**:

1. **주석에서 MMLU 제거** (Line 32):
```python
# 변경 전
    - math: GSM8K, MATH (training) + SVAMP, ASDiv, MAWPS, MMLU (evaluation)
# 변경 후
    - math: GSM8K, MATH (training) + SVAMP, ASDiv, MAWPS (evaluation)
```

2. **DOMAIN_CONFIG에서 MMLU 제거** (Line 64):
```python
# 삭제
                "mmlu": {"filename": "mmlu_test.json", "answer_type": AnswerType.MCQ},
```

### 2.6 utils/dataset_registry.py

**수정할 부분**:

1. **주석에서 MMLU 제거** (Line 17):
```python
# 변경 전
    - math: GSM8K, MATH (training) + SVAMP, ASDiv, MAWPS, MMLU (evaluation)
# 변경 후
    - math: GSM8K, MATH (training) + SVAMP, ASDiv, MAWPS (evaluation)
```

2. **DOMAIN_CONFIG에서 MMLU 제거** (Line 26):
```python
# 변경 전
            "eval_datasets": ["gsm8k", "math", "svamp", "asdiv", "mawps", "mmlu"],
# 변경 후
            "eval_datasets": ["gsm8k", "math", "svamp", "asdiv", "mawps"],
```

### 2.7 README.md

**수정할 부분**:

1. **평가 데이터셋 표** (Line 26):
```markdown
<!-- 변경 전 -->
| **Math** | GSM8K, MATH | SVAMP, ASDiv, MAWPS, MMLU |
<!-- 변경 후 -->
| **Math** | GSM8K, MATH | SVAMP, ASDiv, MAWPS |
```

2. **디렉토리 구조** (Line 353):
```markdown
<!-- 삭제 -->
│       │   └── mmlu_test.json
```

3. **스크립트 설명** (Line 470):
```markdown
<!-- 변경 전 -->
- **Math 도메인**: GSM8K, MATH, SVAMP, ASDiv, MAWPS, MMLU (수학 과목)
<!-- 변경 후 -->
- **Math 도메인**: GSM8K, MATH, SVAMP, ASDiv, MAWPS
```

4. **Pipeline 로그 형식 - 통계 필드** (Line 432-433):
```markdown
<!-- 변경 전 -->
    "sft_case_a": 80,
    "sft_case_a_failed": 20
<!-- 변경 후 -->
    "sft_case_a": 80,
    "sft_case_b": 20
```
> **이유**: 이전 커밋(4bfdbeb)에서 "A-Failed" 용어를 "Case B"로 통일했으나 README 누락됨

### 2.8 ARCHITECTURE.md

**수정할 부분**:

1. **평가 데이터셋 표** (Line 24):
```markdown
<!-- 변경 전 -->
| **Math** | GSM8K, MATH | GSM8K, MATH, SVAMP, ASDiv, MAWPS, MMLU |
<!-- 변경 후 -->
| **Math** | GSM8K, MATH | GSM8K, MATH, SVAMP, ASDiv, MAWPS |
```

2. **예시 코드** (Line 60):
```python
# 변경 전
# → ["gsm8k", "math", "svamp", "asdiv", "mawps", "mmlu"]
# 변경 후
# → ["gsm8k", "math", "svamp", "asdiv", "mawps"]
```

3. **디렉토리 구조** (Line 441):
```markdown
<!-- 삭제 -->
        │   └── mmlu_test.json
```

4. **DOMAIN_CONFIG 예시** (Line 864):
```python
# 변경 전
        "eval_datasets": ["gsm8k", "math", "svamp", "asdiv", "mawps", "mmlu"],
# 변경 후
        "eval_datasets": ["gsm8k", "math", "svamp", "asdiv", "mawps"],
```

---

## ✅ 검증 계획

### Task 1: ReClor 검증
```bash
# 1. 데이터셋 로딩 테스트
python3 -c "from datasets import load_dataset; ds = load_dataset('sxiong/ReClor', split='train'); print(f'Loaded {len(ds)} examples')"

# 2. 전체 logical 도메인 데이터 준비 테스트
python3 -m utils.dataset_preparer --domain logical

# 3. 결과 파일 확인
ls -lh data/logical/train/reclor_train.json
ls -lh data/logical/eval/reclor_test.json
```

### Task 2: MMLU 제거 검증
```bash
# 1. MMLU 참조 확인 (결과 없어야 함)
grep -r "mmlu" --include="*.py" .

# 2. Math 도메인 데이터 준비 테스트
python3 -m utils.dataset_preparer --domain math

# 3. MMLU 파일이 생성되지 않았는지 확인
ls data/math/eval/ | grep mmlu  # 결과 없어야 함
```

---

## 📝 실행 순서

1. **ReClor 수정** (우선순위 높음 - 데이터 준비 블로킹 이슈)
   - [ ] config/dataset_config.py 수정
   - [ ] utils/dataset_preparer.py 수정
   - [ ] 검증 실행

2. **MMLU 제거**
   - [ ] utils/dataset_preparer.py 수정 (3곳)
   - [ ] config 파일 수정 (4개 파일)
   - [ ] utils 파일 수정 (2개 파일)
   - [ ] 문서 파일 수정 (2개 파일)
   - [ ] 검증 실행

3. **Git 커밋 및 푸시**
   - [ ] git add
   - [ ] git commit (명확한 커밋 메시지)
   - [ ] git push

---

## 📌 예상 변경 파일 목록

**수정 파일** (10개):
1. config/dataset_config.py
2. config/domains.py
3. config/config.py
4. utils/dataset_preparer.py
5. utils/domain_loader.py
6. utils/dataset_registry.py
7. README.md
8. ARCHITECTURE.md
9. config/paths.py (주석만)
10. utils/base_loader.py (주석만)

**커밋 메시지 제안**:
```
fix: ReClor 데이터셋 경로 변경 및 MMLU 제거

- ReClor: community-datasets/reclor → sxiong/ReClor로 변경
  (Dataset scripts 지원 중단 문제 해결)
- MMLU 관련 코드 및 설정 완전 제거
  (Math 도메인 평가 데이터셋에서 제외)
- 문서 업데이트 (README.md, ARCHITECTURE.md)
```

---

## 🔍 참고 자료

**ReClor 대체 데이터셋 정보**:
- [sxiong/ReClor - HuggingFace](https://huggingface.co/datasets/sxiong/ReClor)
- 최근 업데이트: 2025년 11월
- 표준 JSON 포맷 (train.json, test.json, val.json)
