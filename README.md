# ID-MAS: Instructional Design Multi-Agent System

LLM 학습을 위한 Dick & Carey 모델 기반 교수 설계 시스템

## 주요 기능

- **데이터셋별 분리 학습**: 각 데이터셋(GSM8K, MATH)별 고유 Terminal Goal로 학습
- **Iterative Scaffolding Pipeline**: Performance Objectives 기반 평가 + Socratic 질문을 통한 반복 학습 (최대 5회)
- **LangGraph 기반 워크플로우**: StateGraph 상태 관리, 조건부 라우팅, 체크포인트 기반 Resume 지원
- **다양한 평가 방법**: Baseline, SFT, SFT_ID-MAS 모델 평가 지원
- **유연한 도메인 구조**: 설정 파일만 수정하여 새로운 도메인 쉽게 추가 가능

## 지원 도메인 및 Terminal Goal

| 도메인 | 학습 데이터셋 | Terminal Goal |
|--------|--------------|---------------|
| **Math** | GSM8K | Generate coherent, step-by-step mathematical reasoning in natural language that leads to a correct numerical answer for grade-school level math problems. |
| **Math** | MATH | Solve advanced mathematical problems by selecting appropriate mathematical concepts and constructing logically valid, multi-step reasoning that leads to a correct solution. |
| **Logical** | ReClor | Analyze logical reasoning problems by comprehending complex passages, identifying logical relationships, and selecting the most appropriate conclusion based on formal reasoning principles. |
| **Commonsense** | ARC-Challenge | Apply commonsense scientific knowledge to solve elementary science problems by understanding fundamental concepts and selecting the correct answer from multiple choices. |

### 평가 데이터셋

| 도메인 | In-Domain 평가 | OOD 평가 |
|--------|---------------|----------|
| **Math** | GSM8K, MATH | SVAMP, ASDiv, MAWPS |
| **Logical** | ReClor | ANLI-R2, ANLI-R3, BBH (9개 서브태스크) |
| **Commonsense** | ARC-Challenge | StrategyQA, OpenBookQA |

**참고**:
- BBH는 9개의 논리 추론 서브태스크(boolean expressions, formal fallacies, logical deduction 등)로 개별 평가됩니다.
- ReClor는 로컬 JSON 파일(`.claude/references/data/reclor_data/`)을 사용하며, HuggingFace 대신 직접 준비된 데이터를 활용합니다.

## 시스템 구조

```
ID-MAS/
├── design_modules/          # 교수 설계 단계별 모듈
│   ├── terminal_goal.py    # Step 0: Terminal Goal 동적 생성
│   ├── analysis.py         # Step 1: 교수 분석 (Goal & Sub-skills)
│   ├── objectives.py       # Step 2: 수행목표 진술 (B-C-CR)
│   ├── test.py             # Step 3: Test item 개발
│   └── rubric.py           # Step 4: 루브릭 개발 (Essay형)
├── prompts/                 # 프롬프트 템플릿
│   └── terminal_goal_prompts.py  # Terminal Goal 생성 프롬프트
├── learning_loop/           # Iterative Scaffolding Pipeline (LangGraph 기반)
│   ├── graph/              # LangGraph StateGraph 구현
│   │   ├── __init__.py     # 모듈 export
│   │   ├── state.py        # 상태 스키마 (IDMASState, QuestionResult)
│   │   ├── nodes.py        # 노드 함수 (scaffolding, advance, finalize)
│   │   └── graph.py        # StateGraph 구성 및 IDMASGraphRunner
│   ├── student_model.py    # Ms: 학생 모델
│   └── teacher_model.py    # Mt: 교사 모델
├── utils/                   # 유틸리티
│   ├── base_loader.py      # 데이터셋 로더 베이스 클래스
│   ├── dataset_preparer.py # 데이터셋 다운로드 및 전처리
│   ├── domain_loader.py    # 도메인 기반 데이터 로더
│   ├── dataset_registry.py # 도메인 레지스트리
│   ├── answer_extractor.py # 답변 추출기 (5가지 유형)
│   ├── sample_extractor.py # Terminal Goal용 대표 샘플 추출
│   └── reparse_eval_results.py  # 평가 결과 재처리
├── models/                  # 모델 래퍼
│   ├── base_wrapper.py     # 모델 래퍼 베이스 클래스
│   ├── teacher_wrapper.py  # Teacher 모델 래퍼 (API + 로컬)
│   ├── student_wrapper.py  # Student 모델 래퍼 (로컬)
│   ├── model_cache.py      # 글로벌 모델 캐시 (메모리 공유)
│   └── local_model_mixin.py # 로컬 모델 생성 믹스인
├── config/                  # 설정 모듈 (서브모듈 구조)
│   ├── __init__.py         # 통합 인터페이스 (backward compatibility)
│   ├── api.py              # API 키 및 인증 (OPENAI_API_KEY, HF_TOKEN)
│   ├── models.py           # Teacher/Student 모델 설정
│   ├── sft.py              # SFT 모델 매핑
│   ├── domains.py          # 도메인 및 데이터셋 설정
│   ├── paths.py            # 디렉토리 경로 헬퍼
│   └── config.py           # 레거시 호환성 유지
├── data/                    # 데이터 저장
│   └── math/               # Math 도메인 데이터
├── main.py                 # 메인 실행 파일 (LangGraph 기반)
└── requirements.txt        # 의존성
```

## 사용 모델

### 교사 모델 (설계 및 평가)

- **기본값 (OpenAI)**: `gpt-5-2025-08-07` (OPENAI_API_KEY 필요)
- **로컬 모델**: HuggingFace 모델 직접 로드 (GPU 필요)
- CLI에서 `--teacher-model`로 선택 (`config.create_teacher_config()`가 설정 자동 생성)
- `gpt-`, `o1-`, `o3-`로 시작하는 모델은 OpenAI API, 그 외 모델은 로컬 HuggingFace 직접 로드
- Teacher/Student 동일 모델 사용 시 `ModelCache`로 메모리 공유
- 설계 모듈과 교사 모델 모두 동일한 `teacher_config`를 공유

지원 모델 (`config.AVAILABLE_TEACHER_MODELS`)

| 유형 | 모델 | 비고 |
|------|------|------|
| OpenAI | gpt-5-2025-08-07 | 기본값 (API) |
| 로컬 | meta-llama/Llama-3.1-8B-Instruct | HuggingFace 직접 로드 |
| 로컬 | meta-llama/Llama-3.1-70B-Instruct | HuggingFace 직접 로드 |
| 로컬 | meta-llama/Llama-3.2-3B-Instruct | HuggingFace 직접 로드 |
| 로컬 | meta-llama/Llama-3.3-70B-Instruct | HuggingFace 직접 로드 |
| 로컬 | Qwen/Qwen2.5-3B-Instruct | HuggingFace 직접 로드 |
| 로컬 | Qwen/Qwen2.5-7B-Instruct | HuggingFace 직접 로드 |
| 로컬 | Qwen/Qwen2.5-14B-Instruct | HuggingFace 직접 로드 |
| 로컬 | Qwen/Qwen2.5-72B-Instruct | HuggingFace 직접 로드 |
| 로컬 | Qwen/Qwen3-4B-Instruct-2507 | HuggingFace 직접 로드 |

#### 로컬 모델 사용 예시

```bash
# 로컬 Teacher 모델 사용 (GPU 필요)
python main.py --mode train --domain math --train-dataset gsm8k \
    --teacher-model Qwen/Qwen2.5-7B-Instruct

# Teacher/Student 동일 모델 사용 (메모리 공유)
python main.py --mode train --domain math --train-dataset gsm8k \
    --teacher-model Qwen/Qwen2.5-3B-Instruct \
    --student-model Qwen/Qwen2.5-3B-Instruct
```

### 학생 모델 (선택 가능)
| 모델 | 설명 |
|------|------|
| `meta-llama/Llama-3.1-8B-Instruct` | Llama 3.1 8B |
| `meta-llama/Llama-3.1-70B-Instruct` | Llama 3.1 70B |
| `meta-llama/Llama-3.2-3B-Instruct` | Llama 3.2 3B |
| `meta-llama/Llama-3.3-70B-Instruct` | Llama 3.3 70B |
| `Qwen/Qwen2.5-3B-Instruct` | Qwen2.5 3B (기본값) |
| `Qwen/Qwen2.5-7B-Instruct` | Qwen2.5 7B |
| `Qwen/Qwen2.5-14B-Instruct` | Qwen2.5 14B |
| `Qwen/Qwen2.5-72B-Instruct` | Qwen2.5 72B |
| `Qwen/Qwen3-4B-Instruct-2507` | Qwen3 4B (최신) |

## 빠른 시작

```bash
# 1. Conda 환경 생성 및 활성화
conda create -n ID-MAS python=3.11 -y
conda activate ID-MAS

# 2. 의존성 설치
pip install -r requirements.txt

# 3. .env 파일 설정
cp .env.example .env
# .env 파일을 열어 OPENAI_API_KEY와 HF_TOKEN 설정

# 4. 데이터 준비 (HuggingFace에서 다운로드)
python -m utils.dataset_preparer

# 4-1. 샘플 데이터 추출 (Terminal Goal 동적 생성용, 선택사항)
python -m utils.sample_extractor

# 5. 학습 실행 (Iterative Scaffolding Pipeline)
python main.py --mode train --domain math --train-dataset gsm8k
# 교사 모델 사용 시: --teacher-model meta-llama/Llama-3.3-70B-Instruct 등 추가

# 6. 평가 실행
python main.py --mode eval --method baseline \
    --domain math --eval-dataset gsm8k
```

## 디버그 로그

LLM API 응답 원문을 확인하려면 `IDMAS_DEBUG_API`를 설정합니다.

```bash
# LLM API raw response 출력
IDMAS_DEBUG_API=1 python main.py --mode train --domain math --train-dataset gsm8k
```

## 환경변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 키 | (필수, gpt-*/o1-*/o3-* 모델 사용 시) |
| `HF_TOKEN` | HuggingFace 토큰 | (필수, 로컬 모델 사용 시) |
| `IDMAS_DEBUG_API` | API 디버그 로그 출력 | `0` |

## 실행 예제

### 학습 모드 (--mode train)

학습을 수행하고 SFT 데이터를 생성합니다.

```bash
# Math 도메인 - GSM8K로 학습
python main.py --mode train --domain math --train-dataset gsm8k

# Math 도메인 - MATH로 학습
python main.py --mode train --domain math --train-dataset math

# Logical 도메인 - ReClor로 학습
python main.py --mode train --domain logical --train-dataset reclor

# Commonsense 도메인 - ARC-Challenge로 학습
python main.py --mode train --domain commonsense --train-dataset arc_c

# 다른 학생 모델로 학습
python main.py --mode train --domain math --train-dataset gsm8k \
    --student-model Qwen/Qwen3-4B-Instruct-2507

# 로컬 Teacher 모델 사용 (GPU 필요)
python main.py --mode train --domain math --train-dataset gsm8k \
    --teacher-model meta-llama/Llama-3.3-70B-Instruct

# 처음부터 새로 학습 (Resume 비활성화 + Terminal Goal 재생성)
# --resume False 사용 시 샘플 데이터 기반으로 Terminal Goal을 다시 생성합니다
python main.py --mode train --domain math --train-dataset gsm8k --resume False
```

### 로컬 교사 모델 사용 예시

GPU 서버에서 HuggingFace 로컬 모델 사용: 

```bash
# 로컬 모델 직접 로드 (GPU 필요)
python main.py --mode train --domain math --train-dataset gsm8k \
    --teacher-model Qwen/Qwen2.5-7B-Instruct

# Teacher/Student 동일 모델 사용 시 메모리 공유
python main.py --mode train --domain math --train-dataset gsm8k \
    --teacher-model Qwen/Qwen2.5-3B-Instruct \
    --student-model Qwen/Qwen2.5-3B-Instruct
```

**참고**: 로컬 모델 사용 시 충분한 GPU 메모리가 필요합니다. `ModelCache`가 동일 모델을 공유하여 메모리를 절약합니다.

### 평가 모드 (--mode eval)

#### Baseline 평가

베이스 모델의 순수 성능을 측정합니다.

```bash
# GSM8K Baseline 평가
python main.py --mode eval --method baseline \
    --domain math --eval-dataset gsm8k

# SVAMP Cross-dataset 평가
python main.py --mode eval --method baseline \
    --domain math --eval-dataset svamp

# Logical 도메인 - ReClor Baseline 평가
python main.py --mode eval --method baseline \
    --domain logical --eval-dataset reclor

# Logical 도메인 - ANLI-R2 OOD 평가
python main.py --mode eval --method baseline \
    --domain logical --eval-dataset anli_r2

# Logical 도메인 - BBH 서브태스크 OOD 평가 (개별 평가)
python main.py --mode eval --method baseline \
    --domain logical --eval-dataset bbh_boolean_expressions

python main.py --mode eval --method baseline \
    --domain logical --eval-dataset bbh_web_of_lies

# Commonsense 도메인 - StrategyQA OOD 평가
python main.py --mode eval --method baseline \
    --domain commonsense --eval-dataset strategyqa

# 처음부터 새로 평가 (Resume 비활성화)
python main.py --mode eval --method baseline \
    --domain math --eval-dataset gsm8k --eval-resume False
```

#### SFT 평가

HuggingFace Hub에서 fine-tuned 모델을 로드하여 평가합니다.

```bash
# GSM8K SFT 평가
python main.py --mode eval --method sft \
    --domain math --eval-dataset gsm8k \
    --model Qwen/Qwen2.5-3B-Instruct

# Cross-dataset SFT 평가 (Math 모델로 SVAMP 평가)
python main.py --mode eval --method sft \
    --domain math --eval-dataset svamp \
    --model Qwen/Qwen2.5-7B-Instruct

# 다른 학생 모델로 평가
python main.py --mode eval --method sft \
    --domain math --eval-dataset math \
    --model meta-llama/Llama-3.1-8B-Instruct
```

**사용 가능한 SFT 모델:**
- Math: `SaFD-00/qwen2.5-3b-math`, `SaFD-00/qwen2.5-7b-math`, `SaFD-00/qwen2.5-14b-math`, `SaFD-00/qwen3-4b-math`, `SaFD-00/llama3.1-8b-math`, `SaFD-00/llama3.2-3b-math`

#### SFT_ID-MAS 평가

ID-MAS Iterative Scaffolding Pipeline으로 학습된 SFT 모델을 평가합니다.

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

**사용 가능한 SFT_ID-MAS 모델:**
- Math: `SaFD-00/qwen2.5-3b-math_id-mas`, `SaFD-00/qwen2.5-7b-math_id-mas`, `SaFD-00/qwen2.5-14b-math_id-mas`, `SaFD-00/qwen3-4b-math_id-mas`, `SaFD-00/llama3.1-8b-math_id-mas`, `SaFD-00/llama3.2-3b-math_id-mas`

## CLI 옵션

### 공통 옵션

| 옵션 | 설명 | 값 |
|------|------|-----|
| `--mode` | 실행 모드 | `train`, `eval` (필수) |
| `--model` | 학생 모델 선택 | Qwen/Qwen2.5-3B-Instruct (기본값) |
| `--teacher-model` | 교사/설계 모델 선택 | `config.AVAILABLE_TEACHER_MODELS` (기본값: gpt-5-2025-08-07) |

### 학습 모드 전용 옵션 (--mode train)

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--domain` | 도메인: `math`, `logical`, `commonsense` | (필수) |
| `--train-dataset` | 학습 데이터셋: `gsm8k`, `math`, `reclor`, `arc_c` | (필수) |
| `--run-design` | 새로운 설계 생성 (기본값: 기존 설계 로드 또는 자동 생성) | False |
| `--resume` | 기존 로그에서 이어서 학습 (`True`/`False`). `False` 시 Terminal Goal도 재생성 | `True` |

### 평가 모드 전용 옵션 (--mode eval)

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--method` | 평가 방법: `baseline`, `sft`, `sft_id-mas` | (필수) |
| `--domain` | 도메인 (데이터셋 검증용) | (필수) |
| `--eval-dataset` | 평가 데이터셋 | (필수) |
| `--eval-resume` | 기존 결과에서 이어서 평가 (`True`/`False`) | `True` |

## 평가 방법

| Method | 설명 | 특징 |
|--------|------|------|
| `baseline` | 베이스 모델로 평가 | 순수 베이스 모델 성능 측정 |
| `sft` | Fine-tuned 모델로 평가 | HuggingFace Hub에서 SFT 모델 로드하여 평가 |
| `sft_id-mas` | ID-MAS SFT 모델로 평가 | Iterative Scaffolding Pipeline으로 학습된 SFT 모델 평가 |

### SFT 평가

SFT 방법은 HuggingFace Hub에서 fine-tuned 모델을 로드하여 평가:
- `SaFD-00/{model}-{domain}` (예: `SaFD-00/qwen2.5-3b-math`)

### SFT_ID-MAS 평가

ID-MAS Iterative Scaffolding Pipeline으로 생성된 SFT 데이터로 학습된 모델을 평가:
- `SaFD-00/{model}-{domain}_id-mas` (예: `SaFD-00/qwen2.5-3b-math_id-mas`)

## 데이터 구조

```
data/
├── math/                                           # Math 도메인
│   ├── train/                                      # 학습 데이터
│   │   ├── data/                                   # 원본 학습 데이터
│   │   │   ├── gsm8k_train.json                   # GSM8K 학습 데이터
│   │   │   ├── gsm8k_samples.json                 # GSM8K 샘플 (Terminal Goal용)
│   │   │   ├── math_train.json                    # MATH 학습 데이터
│   │   │   └── math_samples.json                  # MATH 샘플 (Terminal Goal용)
│   │   │
│   │   └── {Teacher-Model}/                        # Teacher 모델별
│   │       ├── instructional-design/               # 설계 결과
│   │       └── {Student-Model}/                    # Student 모델별 SFT 데이터
│   │
│   └── eval/                                       # 평가 데이터
│       ├── data/                                   # 원본 평가 데이터
│       │   ├── gsm8k_test.json
│       │   ├── math_test.json
│       │   ├── svamp_test.json
│       │   ├── asdiv_test.json
│       │   └── mawps_test.json
│       └── {Model}/                                # 모델별 평가 결과
│
├── logical/                                        # Logical 도메인
│   ├── train/
│   │   ├── data/
│   │   │   ├── reclor_train.json                  # 로컬 데이터에서 생성
│   │   │   └── reclor_samples.json                # ReClor 샘플 (Terminal Goal용)
│   │   └── {Teacher-Model}/
│   │       ├── instructional-design/
│   │       └── {Student-Model}/
│   └── eval/
│       ├── data/
│       │   ├── reclor_test.json                   # In-Domain (로컬 데이터)
│       │   ├── anli_r2_test.json                  # OOD
│       │   ├── anli_r3_test.json                  # OOD
│       │   └── bbh_*.json                         # OOD (9개 서브태스크 개별 파일)
│       └── {Model}/
│
└── commonsense/                                    # Commonsense 도메인
    ├── train/
    │   ├── data/
    │   │   ├── arc_c_train.json
    │   │   └── arc_c_samples.json                 # ARC-C 샘플 (Terminal Goal용)
    │   └── {Teacher-Model}/
    │       ├── instructional-design/
    │       └── {Student-Model}/
    └── eval/
        ├── data/
        │   ├── arc_c_test.json                    # In-Domain
        │   ├── strategyqa_test.json               # OOD
        │   └── openbookqa_test.json               # OOD
        └── {Model}/
```

## 데이터 형식

### 원본 학습/평가 데이터 형식

```json
{
  "instruction": "You are a helpful math assistant.\nSolve the problem step by step and provide your final answer within \\boxed{}.",
  "input": "문제 텍스트",
  "output": "풀이 과정... \\boxed{정답}"
}
```

### Pipeline 로그 형식 (`*_logs.json`)

```json
{
  "scaffolding_results": [
    {
      "id": "question_id",
      "instruction": "시스템 프롬프트",
      "input": "문제 텍스트",
      "output": "정답 (ground truth)",
      "initial_response": "학생 모델 응답",
      "predicted_answer": "추출된 답",
      "scaffolding_correct": true,
      "sft_case": "A",
      "sft_response": "SFT용 응답",
      "iterative_scaffolding": {
        "success": true,
        "iterations_needed": 1,
        "conversation_history": [...]
      }
    }
  ],
  "statistics": {
    "total_questions": 100,
    "scaffolding_processed": 100,
    "scaffolding_correct": 80,
    "sft_case_a": 80,
    "sft_case_b": 20
  }
}
```

### SFT 데이터 형식 (`*_id-mas_{Model}.json`)

```json
[
  {
    "instruction": "시스템 프롬프트",
    "input": "문제 텍스트",
    "output": "SFT용 응답"
  }
]
```

### SFT Case 분류

| Case | 설명 | 응답 출처 |
|------|------|-----------|
| `A` | Iterative Scaffolding 성공 (PO 충족) | 학생 모델 응답 (1회 또는 다중 시도) |
| `B` | 5회 실패 후 재구성 (PO 미충족) | AI 기반 대화 분석 후 재구성된 응답 |

**성공 조건**: 모든 수행목표(PO)가 충족되면(`all_satisfied=True`) Case A로 처리됩니다. 정답 여부(`is_correct`)는 SFT 케이스 분류에 영향을 주지 않습니다.

**B 처리**: 최대 5회 Iterative Scaffolding 후에도 PO 충족 조건을 만족하지 못한 경우, Teacher 모델이 대화 히스토리를 AI 기반으로 축약/분석하여 학생의 약점을 파악하고, 이를 보완하는 정답 솔루션을 재구성합니다.

## 데이터 준비

### 자동 데이터 준비

HuggingFace에서 데이터셋을 다운로드하고 전처리합니다:

```bash
python -m utils.dataset_preparer
```

이 스크립트는 다음 작업을 수행합니다:
- HuggingFace에서 데이터셋 다운로드
- 로컬 ReClor 데이터 변환 (`.claude/references/data/reclor_data/`)
- BBH 9개 서브태스크를 개별 파일로 저장 (`bbh_*_test.json`)
- 통일된 JSON 형식으로 변환

다운로드되는 데이터셋:
- **Math 도메인**: GSM8K, MATH, SVAMP, ASDiv, MAWPS
- **Logical 도메인**: ReClor (로컬 데이터 활용), ANLI-R2, ANLI-R3, BBH (논리 추론 9개 태스크 통합)
- **Commonsense 도메인**: ARC-Challenge, StrategyQA, OpenBookQA

### 데이터셋 소스

| 도메인 | 데이터셋 | 소스 | 비고 |
|--------|---------|------|------|
| Math | GSM8K, MATH, SVAMP, ASDiv, MAWPS | HuggingFace | - |
| Logical | ANLI-R2, ANLI-R3 | HuggingFace | - |
| Logical | BBH | HuggingFace (lukaemon/bbh) | 9개 서브태스크 개별 파일 (`bbh_*_test.json`) |
| Logical | ReClor | 로컬 파일 (`.claude/references/data/reclor_data/`) | train/test 로컬 JSON 활용 |
| Commonsense | ARC-Challenge | HuggingFace | - |
| Commonsense | StrategyQA | HuggingFace (ChilleD/StrategyQA) | - |
| Commonsense | OpenBookQA | HuggingFace | - |

### ReClor 로컬 데이터 준비

ReClor 데이터셋은 로컬 파일을 사용합니다. 다음 파일이 필요합니다:

```
.claude/references/data/reclor_data/
├── train.json (4.7MB)
├── test.json (1MB)
├── val.json (533KB, 선택적)
├── question_type_names.json
├── source_list.txt
└── use_items.txt
```

**필수 파일**: `train.json`, `test.json`
**JSON 형식**:
```json
{
  "context": "지문 텍스트",
  "question": "질문 텍스트",
  "answers": ["선택지 A", "선택지 B", "선택지 C", "선택지 D"],
  "label": 1,
  "id_string": "train_0"
}
```

`dataset_preparer.py` 실행 시 이 파일들이 자동으로 읽혀 다음과 같은 형식으로 변환됩니다:

```json
{
  "instruction": "You are a logical reasoning assistant...",
  "input": "Context:\n[context]\nQuestion: [question]\nOptions:\nA. [option A]\nB. [option B]\nC. [option C]\nD. [option D]",
  "output": "\\boxed{B}"
}
```

## HuggingFace 토큰 설정

학생 모델(Qwen, Llama)을 사용하려면 HuggingFace 토큰이 필요합니다.

### 토큰 발급

1. [HuggingFace](https://huggingface.co)에 가입/로그인
2. [Settings > Access Tokens](https://huggingface.co/settings/tokens)로 이동
3. "New token" 클릭 → 이름 입력 → "Read" 권한 선택 → "Generate"
4. 생성된 토큰 복사 (hf_xxxxx 형식)

### 토큰 등록

```bash
# .env 파일에 설정 (권장)
HF_TOKEN=hf_your_token_here
```

### Llama 모델 접근 권한 (필수)

Meta의 Llama 모델은 gated model로, 별도 접근 승인이 필요합니다:

1. [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 페이지 방문
2. "Access repository" 버튼 클릭
3. Meta 라이선스 동의 후 제출
4. 승인 이메일 수신 후 사용 가능 (보통 즉시~수분 내)

> **참고**: Qwen 모델은 별도 승인 없이 바로 사용 가능합니다.

## 새로운 도메인 추가하기

ID-MAS는 설정 기반의 유연한 도메인 구조를 제공합니다. 새로운 도메인을 추가하려면:

### 1. 데이터 준비

새 도메인 디렉토리와 데이터 파일 구조 생성:

```bash
# 디렉토리 구조 생성
mkdir -p data/{domain}/train/data
mkdir -p data/{domain}/eval/data

# 학습 데이터 배치
# data/{domain}/train/data/{dataset}_train.json

# 평가 데이터 배치
# data/{domain}/eval/data/{dataset}_test.json
```

### 2. 설정 파일 업데이트

**config/domains.py**에 도메인 정보 추가:

```python
# Terminal Goals 추가
TERMINAL_GOALS = {
    "gsm8k": "...",
    "math": "...",
    "your_dataset": "Your terminal goal description here"
}

# Dataset to domain mapping 추가
DATASET_TO_DOMAIN = {
    "gsm8k": "math",
    "math": "math",
    "your_dataset": "your_domain"
}

# Training datasets 추가
TRAINING_DATASETS = {
    "math": ["gsm8k", "math"],
    "your_domain": ["your_dataset"]
}

# Domain config 추가
DOMAIN_CONFIG = {
    "math": {...},
    "your_domain": {
        "data_dir": DATA_DIR / "your_domain",
        "training_datasets": ["your_dataset"],
        "eval_datasets": ["your_eval_dataset"],
        "default_eval": "your_eval_dataset"
    }
}
```

### 3. 실행 및 테스트

```bash
# 새 도메인으로 학습 실행
python main.py --mode train --domain your_domain --train-dataset your_dataset

# 새 도메인으로 평가 실행
python main.py --mode eval --method baseline --domain your_domain --eval-dataset your_eval_dataset
```

### 주의사항

- Terminal Goal은 학습 데이터셋별로 명확하게 정의해야 합니다
- 데이터 파일 형식은 기존 도메인(math)의 JSON 구조를 참고하세요
- 새 도메인 추가 시 코드 수정은 불필요합니다 (설정만 수정)
- 더 자세한 가이드는 [ARCHITECTURE.md](ARCHITECTURE.md)를 참고하세요