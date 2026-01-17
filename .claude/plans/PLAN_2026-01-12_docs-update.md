# Documentation & Code Update Plan

## Overview

**목표**:
1. `config/models.py` 코드 수정 (LLaMA-Factory API 제거 → 로컬 HuggingFace 직접 로드)
2. USAGE.md, ARCHITECTURE.md, README.md 문서 업데이트

**도메인**: coding + writing (코드 수정 + 문서 업데이트)

**분석 결과**: 코드 내 불일치 및 문서와 실제 동작 간의 불일치 발견

---

## Phase 1: Context Analysis (완료)

### 발견된 주요 불일치 사항

#### 1. 코드 불일치: config/models.py vs teacher_wrapper.py

**config/models.py** (`create_teacher_config()`, lines 44-62):
```python
# LLaMA-Factory API 모델 (OpenAI-compatible endpoint)
else:
    base_url = os.getenv("LLAMA_FACTORY_BASE_URL", "http://localhost:2000/v1")
    return {
        "model": model_name,
        "base_url": base_url,  # ← 사용되지 않음!
        "api_key": "0",
        "max_tokens": 8192
    }
```

**teacher_wrapper.py** (`_is_api_model()`, lines 170-178 & `__init__()`, lines 206-222):
```python
def _is_api_model(model_name: str) -> bool:
    return (
        model_name.startswith("gpt-") or
        model_name.startswith("o1") or
        model_name.startswith("o3")
    )

# __init__에서:
if self._use_api:
    self._init_api_client()  # OpenAI API 사용
else:
    # 로컬 모델: ModelCache를 통해 직접 로드
    cached = ModelCache.get_or_load(self.model_name, self.device)
    self.model = cached["model"]
    self.tokenizer = cached["tokenizer"]
```

**결론**: `teacher_wrapper.py`는 이미 로컬 HuggingFace 모델을 `ModelCache`로 직접 로드하도록 구현됨.
`config/models.py`의 LLaMA-Factory 설정은 **실제로 무시됨**!

#### 코드 수정 필요 (Task 0)

`config/models.py`의 `create_teacher_config()` 함수를 정리하여:
1. LLaMA-Factory API 관련 코드 제거
2. 로컬 모델에 적합한 설정 반환 (device, max_new_tokens 등)
3. 주석 업데이트

#### 2. 문서 불일치 (ARCHITECTURE.md, README.md)

**문서 (현재)**:
- "로컬 모델: HuggingFace 모델을 `ModelCache`를 통해 로드 및 추론" ← **정확함 (실제 동작)**
- "o1-*, o3-*로 시작하는 모델은 OpenAI API" ← **정확함**

**수정 필요**:
- 문서는 실제 동작(`teacher_wrapper.py`)을 기준으로 이미 맞음
- `config/models.py`의 코드와 주석을 문서/실제 동작에 맞게 수정해야 함

#### 2. BBH 데이터셋 불일치 (README.md, ARCHITECTURE.md)

**문서**: "9개 태스크가 `bbh_test.json` 하나로 통합"

**실제 코드** (`config/domains.py:55-63`):
```python
"eval_datasets": [
    "reclor", "anli_r2", "anli_r3",
    "bbh_boolean_expressions", "bbh_formal_fallacies",
    "bbh_logical_deduction_three_objects", "bbh_logical_deduction_five_objects",
    "bbh_logical_deduction_seven_objects",
    "bbh_tracking_shuffled_objects_three_objects",
    "bbh_tracking_shuffled_objects_five_objects",
    "bbh_tracking_shuffled_objects_seven_objects",
    "bbh_web_of_lies"
]
```

vs `utils/domain_loader.py:80`:
```python
"bbh": {"filename": "bbh_test.json", "answer_type": AnswerType.TEXT},
```

**수정 필요**:
- config/domains.py와 domain_loader.py 간의 차이 확인 후 문서에 정확히 반영
- BBH가 통합 파일인지 개별 태스크인지 명확히 기술

#### 3. USAGE.md 내용 부족

**현재**: 단순 학습 명령어 예시 5개만 존재

**필요**:
- 전체 지원 모델 목록 (Teacher/Student)
- 전체 데이터셋 목록 (도메인별 학습/평가)
- CLI 옵션 상세 설명
- 사용 예제 확장

---

## Phase 2: Update Plan

### Task 0: config/models.py 코드 수정 (우선순위: 최고)

**목적**: LLaMA-Factory API 관련 코드 제거, 로컬 HuggingFace 직접 로드에 맞는 설정으로 정리

**수정 내용**:

```python
# 변경 전 (현재 코드)
def create_teacher_config(model_name: str = None) -> dict:
    if model_name is None:
        model_name = DEFAULT_TEACHER_MODEL

    # OpenAI 모델 (gpt-로 시작)
    if model_name.startswith("gpt-"):
        return {
            "model": model_name,
            "base_url": None,
            "api_key": OPENAI_API_KEY,
            ...
        }
    # LLaMA-Factory API 모델 (OpenAI-compatible endpoint)  ← 삭제 필요
    else:
        import os
        base_url = os.getenv("LLAMA_FACTORY_BASE_URL", "http://localhost:2000/v1")
        return {
            "model": model_name,
            "base_url": base_url,  # ← 실제로 사용되지 않음
            "api_key": "0",
            "max_tokens": 8192
        }

# 변경 후
def create_teacher_config(model_name: str = None) -> dict:
    if model_name is None:
        model_name = DEFAULT_TEACHER_MODEL

    # API 모델 판단 (OpenAI 모델)
    is_openai_model = (
        model_name.startswith("gpt-") or
        model_name.startswith("o1") or
        model_name.startswith("o3")
    )

    # OpenAI API 모델
    if is_openai_model:
        return {
            "model": model_name,
            "base_url": None,
            "api_key": OPENAI_API_KEY,
            "reasoning": {"effort": "medium"},
            "text": {"verbosity": "medium"},
            "max_tokens": 8192
        }

    # 로컬 HuggingFace 모델 (ModelCache를 통해 직접 로드)
    return {
        "model": model_name,
        "device": "cuda",
        "max_new_tokens": 8192,
        "temperature": 0.7,
        "do_sample": True
    }
```

**파일**: `config/models.py` (lines 30-62)

**영향 분석**:
- `teacher_wrapper.py`: 변경 없음 (이미 로컬 모델 직접 로드 구현됨)
- `main.py`: 변경 없음 (config만 전달)
- 문서: ARCHITECTURE.md, README.md는 이미 로컬 직접 로드로 설명되어 있으므로 정확해짐

---

### Task 1: USAGE.md 전면 개편

**예상 작업량**: 높음

**업데이트 내용**:

1. **지원 모델 섹션 추가**
   - Teacher 모델 (OpenAI + LLaMA-Factory API)
   - Student 모델 (로컬 HuggingFace)
   - SFT 모델 매핑 (HuggingFace Hub)

2. **지원 데이터셋 섹션 추가**
   - 도메인별 학습 데이터셋
   - 도메인별 평가 데이터셋 (In-Domain + OOD)
   - Answer Type별 분류

3. **CLI 사용법 상세화**
   - 학습 모드 전체 옵션
   - 평가 모드 전체 옵션
   - 환경 변수 설정

4. **예제 확장**
   - 각 도메인별 학습/평가 예제
   - 로컬 Teacher 모델 사용 예제
   - Resume/Checkpoint 관련 예제

**참조 소스**:
- `config/models.py`: `AVAILABLE_TEACHER_MODELS`, `AVAILABLE_STUDENT_MODELS`
- `config/domains.py`: `DOMAIN_CONFIG`, `TERMINAL_GOALS`
- `config/sft.py`: `MODEL_NAME_TO_SHORT`
- `utils/domain_loader.py`: `DOMAIN_CONFIG`

### Task 2: ARCHITECTURE.md 수정

**예상 작업량**: 낮음

**업데이트 내용**:

1. **모델 설정 섹션** (lines 111-118, 765-834)
   - ✅ Teacher 모델 설정: 이미 "로컬 HuggingFace 직접 로드" 설명 있음 → 변경 불필요
   - ✅ `o1-`, `o3-` 관련 내용: 이미 정확함 → 변경 불필요
   - ✅ `ModelCache` 역할: Teacher/Student 공유로 이미 설명됨 → 변경 불필요

2. **BBH 데이터셋 설명 수정**
   - config/domains.py (개별 태스크) vs domain_loader.py (통합 파일) 확인 후 정확한 설명으로 수정

3. **CLI 사용법 업데이트** (lines 700-760)
   - 최신 옵션 반영
   - LLaMA-Factory 관련 예제 제거 또는 수정

### Task 3: README.md 수정

**예상 작업량**: 낮음

**업데이트 내용**:

1. **교사 모델 섹션** (lines 84-107)
   - ✅ 이미 "로컬 모델: HuggingFace 직접 로드" 설명 있음 → 변경 불필요
   - ⚠️ LLaMA-Factory 서버 관련 예시 제거 필요 (lines 203-205, 212-227)

2. **BBH 데이터셋 설명 수정** (lines 31-32, 388)
   - 실제 구현에 맞게 수정

3. **로컬 교사 모델 사용 예시 수정** (lines 212-227)
   - ❌ "localhost:2000/v1에서 서버 실행 필요" 문구 제거
   - ❌ `LLAMA_FACTORY_BASE_URL` 환경변수 언급 제거
   - ✅ ModelCache를 통한 직접 로드로 설명 수정

---

## Phase 3: Implementation Steps

### Step 0: config/models.py 코드 수정 (우선순위: 최고)

**작업 순서**:

1. `config/models.py` 파일 수정
   - LLaMA-Factory API 관련 코드 제거
   - 로컬 모델 설정 추가 (device, max_new_tokens, temperature, do_sample)
   - o1-*, o3-* 모델도 OpenAI API로 분류하도록 수정

2. 주석 업데이트
   - "LLaMA-Factory API 모델" → "로컬 HuggingFace 모델"

**검증**:
- `teacher_wrapper.py`와의 일관성 확인
- 기존 테스트 실행 (있는 경우)

---

### Step 1: USAGE.md 전면 개편 (우선순위: 높음)

```markdown
# USAGE.md 새 구조

1. 지원 모델
   1.1 Teacher 모델
       - OpenAI 모델 (gpt-5-2025-08-07)
       - LLaMA-Factory API 모델 (localhost:2000/v1)
   1.2 Student 모델
       - 로컬 HuggingFace 모델
   1.3 SFT 모델
       - HuggingFace Hub 매핑

2. 지원 데이터셋
   2.1 Math 도메인
       - 학습: gsm8k, math
       - 평가: gsm8k, math, svamp, asdiv, mawps
   2.2 Logical 도메인
       - 학습: reclor
       - 평가: reclor, anli_r2, anli_r3, bbh_*
   2.3 Commonsense 도메인
       - 학습: arc_c
       - 평가: arc_c, strategyqa, openbookqa

3. CLI 사용법
   3.1 학습 모드 (--mode train)
   3.2 평가 모드 (--mode eval)
   3.3 환경 변수

4. 사용 예제
   4.1 도메인별 학습 예제
   4.2 도메인별 평가 예제
   4.3 고급 사용법
```

### Step 2: ARCHITECTURE.md 수정 (우선순위: 중간)

**수정 영역**:
1. Line 111-118: 모델 설정 개요
2. Line 765-834: 상세 모델 설정
3. BBH 관련 설명

### Step 3: README.md 수정 (우선순위: 중간)

**수정 영역**:
1. Line 84-107: 교사 모델 섹션
2. Line 31-32: BBH 참고 설명
3. Line 212-227: 로컬 모델 사용 예시

---

## Phase 4: Quality Gates

### Checklist

- [ ] USAGE.md
  - [ ] 모든 Teacher 모델 목록 포함
  - [ ] 모든 Student 모델 목록 포함
  - [ ] 모든 도메인/데이터셋 목록 포함
  - [ ] CLI 옵션 완전히 문서화
  - [ ] 예제 명령어 실행 가능 확인

- [ ] ARCHITECTURE.md
  - [ ] Teacher 모델 설정 설명이 `config/models.py`와 일치
  - [ ] BBH 데이터셋 설명이 실제 구현과 일치
  - [ ] 다이어그램/Mermaid 코드 최신화

- [ ] README.md
  - [ ] ARCHITECTURE.md와 일관성 유지
  - [ ] 빠른 시작 가이드 검증
  - [ ] 환경 변수 섹션 최신화

### 검증 방법

1. **코드 대조**: 각 설명이 실제 코드와 일치하는지 확인
2. **명령어 테스트**: 예제 명령어가 실제로 동작하는지 확인
3. **일관성 검사**: 세 문서 간 중복 내용이 일치하는지 확인

---

## Risks & Considerations

### Risk 1: config/domains.py vs domain_loader.py 불일치

**현황**: BBH 데이터셋 정의가 두 파일에서 다름
- `config/domains.py`: 개별 태스크 9개
- `domain_loader.py`: 통합 파일 1개

**대응**:
- 실제 런타임 동작 확인 필요
- 코드 분석 후 어느 것이 실제로 사용되는지 파악
- 문서에는 실제 동작 기준으로 기술

### Risk 2: SFT 모델 가용성

**현황**: HuggingFace Hub의 SFT 모델 존재 여부 불확실
- `SaFD-00/{model}-{domain}` 형식의 모델들

**대응**:
- 문서에 "해당 모델이 HuggingFace Hub에 업로드되어 있어야 함" 명시

---

## Summary

| 순서 | 파일 | 작업량 | 주요 변경 |
|------|------|--------|----------|
| 0 | config/models.py | 낮음 | LLaMA-Factory API 제거, 로컬 HuggingFace 설정으로 변경 |
| 1 | USAGE.md | 높음 | 전면 개편 - 모델/데이터셋/CLI 상세 추가 |
| 2 | ARCHITECTURE.md | 낮음 | BBH 설명 수정 (Teacher 모델 설명은 이미 정확) |
| 3 | README.md | 낮음 | BBH 설명 수정, LLaMA-Factory 언급 제거 |

**총 예상 작업**:
- 1개 코드 파일 수정 (`config/models.py`)
- 3개 문서 파일 수정 (USAGE.md, ARCHITECTURE.md, README.md)

**핵심 인사이트**:
- `teacher_wrapper.py`는 이미 로컬 HuggingFace 모델을 `ModelCache`로 직접 로드하도록 구현됨
- `config/models.py`의 LLaMA-Factory 설정이 실제로 무시되고 있었음
- 코드를 문서/실제 동작에 맞게 정리하면 일관성 확보
