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
| **OpenAI API** | `gpt-5-2025-08-07` | 기본값 (OPENAI_API_KEY 필요) |
| **OpenAI API** | `o1-*`, `o3-*` | OpenAI Reasoning 모델 |
| **로컬 HuggingFace** | `meta-llama/Llama-3.1-8B-Instruct` | GPU 직접 로드 |
| **로컬 HuggingFace** | `meta-llama/Llama-3.1-70B-Instruct` | GPU 직접 로드 |
| **로컬 HuggingFace** | `meta-llama/Llama-3.2-3B-Instruct` | GPU 직접 로드 |
| **로컬 HuggingFace** | `meta-llama/Llama-3.3-70B-Instruct` | GPU 직접 로드 |
| **로컬 HuggingFace** | `Qwen/Qwen2.5-3B-Instruct` | GPU 직접 로드 |
| **로컬 HuggingFace** | `Qwen/Qwen2.5-7B-Instruct` | GPU 직접 로드 |
| **로컬 HuggingFace** | `Qwen/Qwen2.5-14B-Instruct` | GPU 직접 로드 |
| **로컬 HuggingFace** | `Qwen/Qwen2.5-72B-Instruct` | GPU 직접 로드 |
| **로컬 HuggingFace** | `Qwen/Qwen3-4B-Instruct-2507` | GPU 직접 로드 |

**모델 선택 로직**:
- `gpt-*`, `o1-*`, `o3-*`로 시작하는 모델 → OpenAI API 사용
- 그 외 모델 → `ModelCache`를 통해 로컬 HuggingFace 직접 로드

### Student 모델 (학습 대상)

Student 모델은 Iterative Scaffolding 학습의 대상입니다.

| 모델 | 비고 |
|------|------|
| `meta-llama/Llama-3.1-8B-Instruct` | Llama 3.1 8B |
| `meta-llama/Llama-3.1-70B-Instruct` | Llama 3.1 70B |
| `meta-llama/Llama-3.2-3B-Instruct` | Llama 3.2 3B |
| `meta-llama/Llama-3.3-70B-Instruct` | Llama 3.3 70B |
| `Qwen/Qwen2.5-3B-Instruct` | **기본값** |
| `Qwen/Qwen2.5-7B-Instruct` | Qwen2.5 7B |
| `Qwen/Qwen2.5-14B-Instruct` | Qwen2.5 14B |
| `Qwen/Qwen2.5-72B-Instruct` | Qwen2.5 72B |
| `Qwen/Qwen3-4B-Instruct-2507` | Qwen3 4B (최신) |

**메모리 공유**: Teacher/Student 동일 모델 사용 시 `ModelCache`가 메모리를 공유합니다.

### SFT 모델 (HuggingFace Hub)

Fine-tuning된 모델은 HuggingFace Hub에서 로드됩니다.

| 베이스 모델 | SFT 모델 | SFT_ID-MAS 모델 |
|-------------|----------|-----------------|
| `Qwen/Qwen2.5-3B-Instruct` | `SaFD-00/qwen2.5-3b-{domain}` | `SaFD-00/qwen2.5-3b-{domain}_id-mas` |
| `Qwen/Qwen2.5-7B-Instruct` | `SaFD-00/qwen2.5-7b-{domain}` | `SaFD-00/qwen2.5-7b-{domain}_id-mas` |
| `Qwen/Qwen2.5-14B-Instruct` | `SaFD-00/qwen2.5-14b-{domain}` | `SaFD-00/qwen2.5-14b-{domain}_id-mas` |
| `Qwen/Qwen2.5-72B-Instruct` | `SaFD-00/qwen2.5-72b-{domain}` | `SaFD-00/qwen2.5-72b-{domain}_id-mas` |
| `Qwen/Qwen3-4B-Instruct-2507` | `SaFD-00/qwen3-4b-{domain}` | `SaFD-00/qwen3-4b-{domain}_id-mas` |
| `meta-llama/Llama-3.1-8B-Instruct` | `SaFD-00/llama3.1-8b-{domain}` | `SaFD-00/llama3.1-8b-{domain}_id-mas` |
| `meta-llama/Llama-3.1-70B-Instruct` | `SaFD-00/llama3.1-70b-{domain}` | `SaFD-00/llama3.1-70b-{domain}_id-mas` |
| `meta-llama/Llama-3.2-3B-Instruct` | `SaFD-00/llama3.2-3b-{domain}` | `SaFD-00/llama3.2-3b-{domain}_id-mas` |
| `meta-llama/Llama-3.3-70B-Instruct` | `SaFD-00/llama3.3-70b-{domain}` | `SaFD-00/llama3.3-70b-{domain}_id-mas` |

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
| **평가 (OOD)** | `bbh_boolean_expressions` | Text | BBH 서브태스크 |
| **평가 (OOD)** | `bbh_formal_fallacies` | Text | BBH 서브태스크 |
| **평가 (OOD)** | `bbh_logical_deduction_three_objects` | Text | BBH 서브태스크 |
| **평가 (OOD)** | `bbh_logical_deduction_five_objects` | Text | BBH 서브태스크 |
| **평가 (OOD)** | `bbh_logical_deduction_seven_objects` | Text | BBH 서브태스크 |
| **평가 (OOD)** | `bbh_tracking_shuffled_objects_three_objects` | Text | BBH 서브태스크 |
| **평가 (OOD)** | `bbh_tracking_shuffled_objects_five_objects` | Text | BBH 서브태스크 |
| **평가 (OOD)** | `bbh_tracking_shuffled_objects_seven_objects` | Text | BBH 서브태스크 |
| **평가 (OOD)** | `bbh_web_of_lies` | Text | BBH 서브태스크 |

### Commonsense 도메인

| 구분 | 데이터셋 | 답변 유형 | 비고 |
|------|----------|-----------|------|
| **학습** | `arc_c` | MCQ | ARC-Challenge |
| **평가 (In-Domain)** | `arc_c` | MCQ | |
| **평가 (OOD)** | `strategyqa` | Boolean | Yes/No 질문 |
| **평가 (OOD)** | `openbookqa` | MCQ | 상식 과학 |

### Terminal Goal (학습 목표)

| 데이터셋 | Terminal Goal |
|----------|---------------|
| `gsm8k` | Generate coherent, step-by-step mathematical reasoning in natural language that leads to a correct numerical answer for grade-school level math problems. |
| `math` | Solve advanced mathematical problems by selecting appropriate mathematical concepts and constructing logically valid, multi-step reasoning that leads to a correct solution. |
| `reclor` | Analyze logical reasoning problems by comprehending complex passages, identifying logical relationships, and selecting the most appropriate conclusion based on formal reasoning principles. |
| `arc_c` | Apply commonsense scientific knowledge to solve elementary science problems by understanding fundamental concepts and selecting the correct answer from multiple choices. |

---

## CLI 사용법

### 공통 옵션

| 옵션 | 설명 | 값 |
|------|------|-----|
| `--mode` | 실행 모드 | `train`, `eval` (필수) |
| `--model` | 학생 모델 선택 | `Qwen/Qwen2.5-3B-Instruct` (기본값) |
| `--teacher-model` | 교사/설계 모델 선택 | `gpt-5-2025-08-07` (기본값) |

### 학습 모드 (--mode train)

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--domain` | 도메인 | `math`, `logical`, `commonsense` (필수) |
| `--train-dataset` | 학습 데이터셋 | `gsm8k`, `math`, `reclor`, `arc_c` (필수) |
| `--run-design` | 새로운 설계 생성 강제 | `False` |
| `--resume` | 기존 로그에서 이어서 학습 | `True` |

**참고**: `--resume False` 사용 시 Terminal Goal도 샘플 데이터에서 재생성됩니다.

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

## 환경 변수

| 변수 | 설명 | 필수 여부 |
|------|------|----------|
| `OPENAI_API_KEY` | OpenAI API 키 | gpt-*/o1-*/o3-* 모델 사용 시 필수 |
| `HF_TOKEN` | HuggingFace 토큰 | 로컬 모델 사용 시 필수 |
| `IDMAS_DEBUG_API` | API 디버그 로그 출력 | 선택 (기본값: 0) |

```bash
# .env 파일 설정 예시
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
IDMAS_DEBUG_API=0
```

---

## 사용 예제

### 학습 예제

#### Math 도메인

```bash
# GSM8K로 학습 (기본 설정)
python main.py --mode train --domain math --train-dataset gsm8k

# MATH로 학습
python main.py --mode train --domain math --train-dataset math

# 다른 학생 모델로 학습
python main.py --mode train --domain math --train-dataset gsm8k \
    --student-model Qwen/Qwen3-4B-Instruct-2507

# 로컬 Teacher 모델 사용 (GPU 필요)
python main.py --mode train --domain math --train-dataset gsm8k \
    --teacher-model Qwen/Qwen2.5-7B-Instruct

# Teacher/Student 동일 모델 사용 (메모리 공유)
python main.py --mode train --domain math --train-dataset gsm8k \
    --teacher-model Qwen/Qwen2.5-3B-Instruct \
    --student-model Qwen/Qwen2.5-3B-Instruct

# 처음부터 새로 학습 (Resume 비활성화 + Terminal Goal 재생성)
python main.py --mode train --domain math --train-dataset gsm8k --resume False
```

#### Logical 도메인

```bash
# ReClor로 학습
python main.py --mode train --domain logical --train-dataset reclor

# 다른 학생 모델로 학습
python main.py --mode train --domain logical --train-dataset reclor \
    --student-model meta-llama/Llama-3.1-8B-Instruct
```

#### Commonsense 도메인

```bash
# ARC-Challenge로 학습
python main.py --mode train --domain commonsense --train-dataset arc_c
```

### 평가 예제

#### Baseline 평가

```bash
# Math - GSM8K Baseline 평가
python main.py --mode eval --method baseline \
    --domain math --eval-dataset gsm8k

# Math - SVAMP OOD 평가
python main.py --mode eval --method baseline \
    --domain math --eval-dataset svamp

# Logical - ReClor Baseline 평가
python main.py --mode eval --method baseline \
    --domain logical --eval-dataset reclor

# Logical - ANLI-R2 OOD 평가
python main.py --mode eval --method baseline \
    --domain logical --eval-dataset anli_r2

# Commonsense - StrategyQA OOD 평가
python main.py --mode eval --method baseline \
    --domain commonsense --eval-dataset strategyqa

# 처음부터 새로 평가 (Resume 비활성화)
python main.py --mode eval --method baseline \
    --domain math --eval-dataset gsm8k --eval-resume False
```

#### SFT 평가

```bash
# GSM8K SFT 평가
python main.py --mode eval --method sft \
    --domain math --eval-dataset gsm8k \
    --model Qwen/Qwen2.5-3B-Instruct

# SVAMP Cross-dataset SFT 평가
python main.py --mode eval --method sft \
    --domain math --eval-dataset svamp \
    --model Qwen/Qwen2.5-7B-Instruct

# 다른 학생 모델로 평가
python main.py --mode eval --method sft \
    --domain math --eval-dataset math \
    --model meta-llama/Llama-3.1-8B-Instruct
```

#### SFT_ID-MAS 평가

```bash
# GSM8K SFT_ID-MAS 평가
python main.py --mode eval --method sft_id-mas \
    --domain math --eval-dataset gsm8k \
    --model Qwen/Qwen2.5-3B-Instruct

# MATH SFT_ID-MAS 평가
python main.py --mode eval --method sft_id-mas \
    --domain math --eval-dataset math \
    --model meta-llama/Llama-3.1-8B-Instruct
```

### 고급 사용법

#### GPU 선택

```bash
# 특정 GPU 사용
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --domain math --train-dataset gsm8k

# 여러 GPU 중 선택
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain math --train-dataset math \
    --student-model Qwen/Qwen2.5-7B-Instruct
```

#### 디버그 모드

```bash
# LLM API raw response 출력
IDMAS_DEBUG_API=1 python main.py --mode train --domain math --train-dataset gsm8k
```

#### 데이터 준비

```bash
# 전체 데이터셋 다운로드 및 전처리
python -m utils.dataset_preparer

# 샘플 데이터 추출 (Terminal Goal 동적 생성용)
python -m utils.sample_extractor

# 특정 도메인/데이터셋만 추출
python -m utils.sample_extractor --domain math --dataset gsm8k --strategy diverse
```

---

## 참고

- 더 자세한 시스템 구조는 [ARCHITECTURE.md](ARCHITECTURE.md)를 참고하세요.
- 빠른 시작 가이드는 [README.md](README.md)를 참고하세요.
