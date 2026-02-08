# ID-MAS 사용 가이드

이 문서는 ID-MAS (Instructional Design Multi-Agent System)의 상세 사용법을 제공합니다.

## 목차

1. [지원 모델](#지원-모델)
2. [지원 데이터셋](#지원-데이터셋)
3. [CLI 사용법](#cli-사용법)
4. [환경 변수](#환경-변수)
5. [사용 예제](#사용-예제)

---

## 지원 모델

### Teacher 모델 (설계 및 평가)

Teacher 모델은 교수 설계 및 Iterative Scaffolding 평가에 사용됩니다.

| 유형 | 모델 | 비고 |
|------|------|------|
| **OpenAI API** | `gpt-5.2` | 기본값 (OPENAI_API_KEY 필요) |
| **로컬 HuggingFace** | `Qwen/Qwen3-1.7B` | GPU 직접 로드 |
| **로컬 HuggingFace** | `Qwen/Qwen3-4B` | GPU 직접 로드 |
| **로컬 HuggingFace** | `Qwen/Qwen3-8B` | GPU 직접 로드 |
| **로컬 HuggingFace** | `Qwen/Qwen3-32B` | GPU 직접 로드 |

**모델 선택 로직**:
- `gpt-*`로 시작하는 모델 → OpenAI API 사용
- 그 외 모델 → `ModelCache`를 통해 로컬 HuggingFace 직접 로드

### Student 모델 (학습 대상)

Student 모델은 Iterative Scaffolding 학습의 대상입니다.

| 모델 | 비고 |
|------|------|
| `Qwen/Qwen3-1.7B` | **기본값** |
| `Qwen/Qwen3-4B` | Qwen3 4B |
| `Qwen/Qwen3-8B` | Qwen3 8B |
| `Qwen/Qwen3-32B` | Qwen3 32B |

**메모리 공유**: Teacher/Student 동일 모델 사용 시 `ModelCache`가 메모리를 공유합니다.

### SFT 모델 (HuggingFace Hub)

Fine-tuning된 모델은 HuggingFace Hub에서 로드됩니다.

| 베이스 모델 | SFT 모델 | SFT_ID-MAS 모델 |
|-------------|----------|-----------------|
| `Qwen/Qwen3-1.7B` | `SaFD-00/qwen3-1.7b-{domain}` | `SaFD-00/qwen3-1.7b-{domain}_id-mas` |
| `Qwen/Qwen3-4B` | `SaFD-00/qwen3-4b-{domain}` | `SaFD-00/qwen3-4b-{domain}_id-mas` |
| `Qwen/Qwen3-8B` | `SaFD-00/qwen3-8b-{domain}` | `SaFD-00/qwen3-8b-{domain}_id-mas` |
| `Qwen/Qwen3-32B` | `SaFD-00/qwen3-32b-{domain}` | `SaFD-00/qwen3-32b-{domain}_id-mas` |

---

## 지원 데이터셋

### Math 도메인

| 구분 | 데이터셋 | 답변 유형 | 비고 |
|------|----------|-----------|------|
| **학습** | `gsm8k` | Numeric | 초등 수학 문제 |
| **학습** | `math` | LaTeX | 고급 수학 문제 |
| **평가 (In-Domain)** | `gsm8k` | Numeric | |
| **평가 (In-Domain)** | `math` | LaTeX | |
| **평가 (OOD)** | `svamp` | Numeric | Cross-dataset |
| **평가 (OOD)** | `asdiv` | Numeric | Cross-dataset |
| **평가 (OOD)** | `mawps` | Numeric | Cross-dataset |

### Logical 도메인

| 구분 | 데이터셋 | 답변 유형 | 비고 |
|------|----------|-----------|------|
| **학습** | `reclor` | MCQ | 논리 추론 |
| **평가 (In-Domain)** | `reclor` | MCQ | |
| **평가 (OOD)** | `anli_r2` | MCQ | Adversarial NLI |
| **평가 (OOD)** | `anli_r3` | MCQ | Adversarial NLI |
| **평가 (OOD)** | `bbh` | Text | BBH |

### Commonsense 도메인

| 구분 | 데이터셋 | 답변 유형 | 비고 |
|------|----------|-----------|------|
| **학습** | `arc_c` | MCQ | ARC-Challenge |
| **평가 (In-Domain)** | `arc_c` | MCQ | |
| **평가 (OOD)** | `strategyqa` | Boolean | Yes/No 질문 |
| **평가 (OOD)** | `openbookqa` | MCQ | 상식 과학 |

---

## CLI 사용법

### 공통 옵션

| 옵션 | 설명 | 값 |
|------|------|-----|
| `--mode` | 실행 모드 | `train`, `eval` (필수) |
| `--model` | 학생 모델 선택 | `Qwen/Qwen3-1.7B` (기본값) |
| `--teacher-model` | 교사/설계 모델 선택 | `gpt-5.2` (기본값) |

### 학습 모드 (--mode train)

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--domain` | 도메인 | `math`, `logical`, `commonsense` (필수) |
| `--train-dataset` | 학습 데이터셋 | `gsm8k`, `math`, `reclor`, `arc_c` (필수) |
| `--run-design` | 새로운 설계 생성 강제 | `False` |
| `--resume` | 기존 로그에서 이어서 학습 | `True` |

**참고**: `--resume False` 사용 시 Instructional Goal도 샘플 데이터에서 재생성됩니다.

### 평가 모드 (--mode eval)

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--method` | 평가 방법 | `baseline`, `sft`, `sft_id-mas` (필수) |
| `--domain` | 도메인 (데이터셋 검증용) | (필수) |
| `--eval-dataset` | 평가 데이터셋 | (필수) |
| `--eval-resume` | 기존 결과에서 이어서 평가 | `True` |

### 평가 방법

| Method | 설명 |
|--------|------|
| `baseline` | 베이스 모델로 평가 (순수 성능 측정) |
| `sft` | HuggingFace Hub에서 SFT 모델 로드하여 평가 |
| `sft_id-mas` | ID-MAS Pipeline으로 학습된 SFT 모델 평가 |

---

## 사용

### 학습

#### Math 도메인
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --domain math --train-dataset gsm8k \
    --student-model Qwen/Qwen3-1.7B \
    --teacher-model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode train --domain math --train-dataset gsm8k \
    --student-model Qwen/Qwen3-4B \
    --teacher-model Qwen/Qwen3-4B

CUDA_VISIBLE_DEVICES=0 python main.py --mode train --domain math --train-dataset gsm8k \
    --student-model Qwen/Qwen3-32B \
    --teacher-model Qwen/Qwen3-32B

CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain math --train-dataset math \
    --student-model Qwen/Qwen3-1.7B \
    --teacher-model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain math --train-dataset math \
    --student-model Qwen/Qwen3-4B \
    --teacher-model Qwen/Qwen3-4B

CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain math --train-dataset math \
    --student-model Qwen/Qwen3-32B \
    --teacher-model Qwen/Qwen3-32B
```

#### Logical 도메인

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --domain logical --train-dataset reclor \
    --student-model Qwen/Qwen3-1.7B \
    --teacher-model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode train --domain logical --train-dataset reclor \
    --student-model Qwen/Qwen3-4B \
    --teacher-model Qwen/Qwen3-4B

CUDA_VISIBLE_DEVICES=0 python main.py --mode train --domain logical --train-dataset reclor \
    --student-model Qwen/Qwen3-32B \
    --teacher-model Qwen/Qwen3-32B
```

#### Commonsense 도메인

```bash
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain commonsense --train-dataset arc_c \
    --student-model Qwen/Qwen3-1.7B \
    --teacher-model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain commonsense --train-dataset arc_c \
    --student-model Qwen/Qwen3-4B \
    --teacher-model Qwen/Qwen3-4B

CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain commonsense --train-dataset arc_c \
    --student-model Qwen/Qwen3-32B \
    --teacher-model Qwen/Qwen3-32B
```

### 평가

#### Math 도메인

```bash
# Baseline 평가 (In-Domain)
CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method baseline \
    --domain math --eval-dataset gsm8k \
    --model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method baseline \
    --domain math --eval-dataset gsm8k \
    --model Qwen/Qwen3-4B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method baseline \
    --domain math --eval-dataset gsm8k \
    --model Qwen/Qwen3-32B

# SFT 평가 (In-Domain)
CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft \
    --domain math --eval-dataset gsm8k \
    --model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft \
    --domain math --eval-dataset gsm8k \
    --model Qwen/Qwen3-4B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft \
    --domain math --eval-dataset gsm8k \
    --model Qwen/Qwen3-32B

# SFT_ID-MAS 평가 (In-Domain)
CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft_id-mas \
    --domain math --eval-dataset gsm8k \
    --model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft_id-mas \
    --domain math --eval-dataset gsm8k \
    --model Qwen/Qwen3-4B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft_id-mas \
    --domain math --eval-dataset gsm8k \
    --model Qwen/Qwen3-32B

# OOD 평가 (SVAMP)
CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method baseline \
    --domain math --eval-dataset svamp \
    --model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft \
    --domain math --eval-dataset svamp \
    --model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft_id-mas \
    --domain math --eval-dataset svamp \
    --model Qwen/Qwen3-1.7B
```

#### Logical 도메인

```bash
# Baseline 평가 (In-Domain)
CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method baseline \
    --domain logical --eval-dataset reclor \
    --model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method baseline \
    --domain logical --eval-dataset reclor \
    --model Qwen/Qwen3-4B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method baseline \
    --domain logical --eval-dataset reclor \
    --model Qwen/Qwen3-32B

# SFT 평가 (In-Domain)
CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft \
    --domain logical --eval-dataset reclor \
    --model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft \
    --domain logical --eval-dataset reclor \
    --model Qwen/Qwen3-4B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft \
    --domain logical --eval-dataset reclor \
    --model Qwen/Qwen3-32B

# SFT_ID-MAS 평가 (In-Domain)
CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft_id-mas \
    --domain logical --eval-dataset reclor \
    --model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft_id-mas \
    --domain logical --eval-dataset reclor \
    --model Qwen/Qwen3-4B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft_id-mas \
    --domain logical --eval-dataset reclor \
    --model Qwen/Qwen3-32B

# OOD 평가 (ANLI-R2)
CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method baseline \
    --domain logical --eval-dataset anli_r2 \
    --model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft \
    --domain logical --eval-dataset anli_r2 \
    --model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft_id-mas \
    --domain logical --eval-dataset anli_r2 \
    --model Qwen/Qwen3-1.7B

# OOD 평가 (BBH 서브태스크 - 개별 평가)
# 사용 가능한 서브태스크:
# - bbh_boolean_expressions
# - bbh_formal_fallacies
# - bbh_logical_deduction_three_objects
# - bbh_logical_deduction_five_objects
# - bbh_logical_deduction_seven_objects
# - bbh_tracking_shuffled_objects_three_objects
# - bbh_tracking_shuffled_objects_five_objects
# - bbh_tracking_shuffled_objects_seven_objects
# - bbh_web_of_lies

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method baseline \
    --domain logical --eval-dataset bbh_boolean_expressions \
    --model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method baseline \
    --domain logical --eval-dataset bbh_web_of_lies \
    --model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft \
    --domain logical --eval-dataset bbh_formal_fallacies \
    --model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft_id-mas \
    --domain logical --eval-dataset bbh_logical_deduction_three_objects \
    --model Qwen/Qwen3-1.7B
```

#### Commonsense 도메인

```bash
# Baseline 평가 (In-Domain)
CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method baseline \
    --domain commonsense --eval-dataset arc_c \
    --model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method baseline \
    --domain commonsense --eval-dataset arc_c \
    --model Qwen/Qwen3-4B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method baseline \
    --domain commonsense --eval-dataset arc_c \
    --model Qwen/Qwen3-32B

# SFT 평가 (In-Domain)
CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft \
    --domain commonsense --eval-dataset arc_c \
    --model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft \
    --domain commonsense --eval-dataset arc_c \
    --model Qwen/Qwen3-4B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft \
    --domain commonsense --eval-dataset arc_c \
    --model Qwen/Qwen3-32B

# SFT_ID-MAS 평가 (In-Domain)
CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft_id-mas \
    --domain commonsense --eval-dataset arc_c \
    --model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft_id-mas \
    --domain commonsense --eval-dataset arc_c \
    --model Qwen/Qwen3-4B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft_id-mas \
    --domain commonsense --eval-dataset arc_c \
    --model Qwen/Qwen3-32B

# OOD 평가 (StrategyQA)
CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method baseline \
    --domain commonsense --eval-dataset strategyqa \
    --model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft \
    --domain commonsense --eval-dataset strategyqa \
    --model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode eval --method sft_id-mas \
    --domain commonsense --eval-dataset strategyqa \
    --model Qwen/Qwen3-1.7B
```

---

## 기타

```bash
## [1]
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --domain math --train-dataset gsm8k \
    --student-model Qwen/Qwen3-1.7B \
    --teacher-model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=0 python main.py --mode train --domain logical --train-dataset reclor \
    --student-model Qwen/Qwen3-1.7B \
    --teacher-model Qwen/Qwen3-1.7B

## [2]
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --domain math --train-dataset gsm8k \
    --student-model Qwen/Qwen3-4B \
    --teacher-model Qwen/Qwen3-4B

CUDA_VISIBLE_DEVICES=0 python main.py --mode train --domain logical --train-dataset reclor \
    --student-model Qwen/Qwen3-4B \
    --teacher-model Qwen/Qwen3-4B

## [3]
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain math --train-dataset math \
    --student-model Qwen/Qwen3-1.7B \
    --teacher-model Qwen/Qwen3-1.7B

CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain commonsense --train-dataset arc_c \
    --student-model Qwen/Qwen3-1.7B \
    --teacher-model Qwen/Qwen3-1.7B

## [4]
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --domain math --train-dataset math \
    --student-model Qwen/Qwen3-4B \
    --teacher-model Qwen/Qwen3-4B

CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain commonsense --train-dataset arc_c \
    --student-model Qwen/Qwen3-4B \
    --teacher-model Qwen/Qwen3-4B
```
