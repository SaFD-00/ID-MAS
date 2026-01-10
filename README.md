# ID-MAS: Instructional Design Multi-Agent System

LLM 학습을 위한 Dick & Carey 모델 기반 교수 설계 시스템

## 주요 기능

- **데이터셋별 분리 학습**: 각 데이터셋(GSM8K, MATH)별 고유 Terminal Goal로 학습
- **3-Phase Pipeline 학습**: Scaffolding → Coaching → Modeling 단계별 학습
- **LangGraph 기반 워크플로우**: StateGraph 상태 관리, 조건부 라우팅, 체크포인트 기반 Resume 지원
- **다양한 평가 방법**: Baseline, SFT, SFT_ID-MAS 모델 평가 지원
- **유연한 도메인 구조**: 설정 파일만 수정하여 새로운 도메인 쉽게 추가 가능

## 지원 도메인 및 Terminal Goal

| 도메인 | 학습 데이터셋 | Terminal Goal |
|--------|--------------|---------------|
| **Math** | GSM8K | Generate coherent, step-by-step mathematical reasoning in natural language that leads to a correct numerical answer for grade-school level math problems. |
| **Math** | MATH | Solve advanced mathematical problems by selecting appropriate mathematical concepts and constructing logically valid, multi-step reasoning that leads to a correct solution. |

### 평가 데이터셋

| 도메인 | 평가 데이터셋 |
|--------|---------------|
| **Math** | GSM8K, MATH, SVAMP, ASDiv, MAWPS, MMLU |

## 시스템 구조

```
ID-MAS/
├── design_modules/          # 교수 설계 단계별 모듈
│   ├── analysis.py         # 교수 분석 (Goal & Sub-skills)
│   ├── objectives.py       # 수행목표 진술 (B-C-CR)
│   ├── test.py             # Test item 개발
│   └── rubric.py           # 루브릭 개발 (Essay형)
├── learning_loop/           # 3-Phase 학습 파이프라인 (LangGraph 기반)
│   ├── graph/              # LangGraph StateGraph 구현
│   │   ├── __init__.py     # 모듈 export
│   │   ├── state.py        # 상태 스키마 (IDMASState, QuestionResult)
│   │   ├── nodes.py        # 노드 함수 (phase1, phase2, phase3 등)
│   │   └── graph.py        # StateGraph 구성 및 IDMASGraphRunner
│   ├── student_model.py    # Ms: 학생 모델
│   └── teacher_model.py    # Mt: 교사 모델
├── utils/                   # 유틸리티
│   ├── base_loader.py      # 데이터셋 로더 베이스 클래스
│   ├── dataset_preparer.py # 데이터셋 다운로드 및 전처리
│   ├── domain_loader.py    # 도메인 기반 데이터 로더
│   ├── dataset_registry.py # 도메인 레지스트리
│   ├── answer_extractor.py # 답변 추출기 (5가지 유형)
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
| 로컬 | Qwen/Qwen2.5-3B-Instruct | HuggingFace 직접 로드 |
| 로컬 | Qwen/Qwen2.5-7B-Instruct | HuggingFace 직접 로드 |
| 로컬 | Qwen/Qwen2.5-14B-Instruct | HuggingFace 직접 로드 |
| 로컬 | Qwen/Qwen3-4B-Instruct-2507 | HuggingFace 직접 로드 |
| 로컬 | meta-llama/Llama-3.1-8B-Instruct | HuggingFace 직접 로드 |
| 로컬 | meta-llama/Llama-3.2-3B-Instruct | HuggingFace 직접 로드 |

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
| `Qwen/Qwen3-4B-Instruct-2507` | Qwen3 4B (최신) |
| `Qwen/Qwen2.5-3B-Instruct` | Qwen2.5 3B (기본값) |
| `Qwen/Qwen2.5-7B-Instruct` | Qwen2.5 7B |
| `Qwen/Qwen2.5-14B-Instruct` | Qwen2.5 14B |
| `meta-llama/Llama-3.1-8B-Instruct` | Llama 3.1 8B |
| `meta-llama/Llama-3.2-3B-Instruct` | Llama 3.2 3B |

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

# 5. 학습 실행 (3-Phase Pipeline)
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

# 다른 학생 모델로 학습
python main.py --mode train --domain math --train-dataset gsm8k \
    --student-model Qwen/Qwen3-4B-Instruct-2507

# 교사 모델로 학습 (localhost:2000/v1에서 서버 실행 필요)
python main.py --mode train --domain math --train-dataset gsm8k \
    --teacher-model meta-llama/Llama-3.3-70B-Instruct

# 처음부터 새로 학습 (Resume 비활성화)
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

ID-MAS 3-Phase Pipeline으로 학습된 SFT 모델을 평가합니다.

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
| `--teacher-model` | 교사/설계 모델 선택 | `config.AVAILABLE_TEACHER_MODELS` (기본값: gpt-5-2025-08-07, LLaMA-Factory: localhost:2000/v1) |

### 학습 모드 전용 옵션 (--mode train)

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--domain` | 도메인: `math` | (필수) |
| `--train-dataset` | 학습 데이터셋: `gsm8k`, `math` | (필수) |
| `--run-design` | 새로운 설계 생성 (기본값: 기존 설계 로드 또는 자동 생성) | False |
| `--resume` | 기존 로그에서 이어서 학습 (`True`/`False`) | `True` |

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
| `sft_id-mas` | ID-MAS SFT 모델로 평가 | 3-Phase Pipeline으로 학습된 SFT 모델 평가 |

### SFT 평가

SFT 방법은 HuggingFace Hub에서 fine-tuned 모델을 로드하여 평가:
- `SaFD-00/{model}-{domain}` (예: `SaFD-00/qwen2.5-3b-math`)

### SFT_ID-MAS 평가

ID-MAS 3-Phase Pipeline으로 생성된 SFT 데이터로 학습된 모델을 평가:
- `SaFD-00/{model}-{domain}_id-mas` (예: `SaFD-00/qwen2.5-3b-math_id-mas`)

## 데이터 구조

```
data/
└── math/                                           # Math 도메인
    ├── train/                                      # 학습 데이터
    │   ├── data/                                   # 원본 학습 데이터
    │   │   ├── gsm8k_train.json                   # GSM8K 학습 데이터
    │   │   └── math_train.json                    # MATH 학습 데이터
    │   │
    │   └── {Teacher-Model}/                        # Teacher 모델별 (예: gpt-5-2025-08-07)
    │       ├── instructional-design/               # 설계 결과
    │       │   ├── math_gsm8k_design.json         # GSM8K 설계
    │       │   └── math_math_design.json          # MATH 설계
    │       │
    │       └── {Student-Model}/                    # Student 모델별 (예: Qwen3-4B-Instruct-2507)
    │           ├── gsm8k_train_id-mas_{Model}.json     # SFT 데이터
    │           ├── gsm8k_train_id-mas_{Model}_logs.json # Pipeline 로그
    │           ├── gsm8k_checkpoint_{timestamp}.json   # 체크포인트
    │           └── gsm8k_train_summary_{Model}.json    # 학습 요약
    │
    └── eval/                                       # 평가 데이터
        ├── data/                                   # 원본 평가 데이터
        │   ├── gsm8k_test.json                    # GSM8K 평가 데이터
        │   ├── math_test.json                     # MATH 평가 데이터
        │   ├── svamp_test.json                    # SVAMP 평가 데이터
        │   ├── asdiv_test.json                    # ASDiv 평가 데이터
        │   ├── mawps_test.json                    # MAWPS 평가 데이터
        │   └── mmlu_test.json                     # MMLU (수학) 평가 데이터
        │
        └── {Model}/                                # 모델별 평가 결과
            ├── gsm8k_eval_results-Baseline.json   # Baseline 평가
            ├── gsm8k_eval_results-SFT.json        # SFT 평가
            └── gsm8k_eval_results-SFT_ID-MAS.json # SFT_ID-MAS 평가
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
  "phase1_results": [
    {
      "id": "question_id",
      "instruction": "시스템 프롬프트",
      "input": "문제 텍스트",
      "output": "정답 (ground truth)",
      "initial_response": "학생 모델 응답",
      "predicted_answer": "추출된 답",
      "phase1_correct": true,
      "sft_case": "A",
      "sft_response": "SFT용 응답",
      "iterative_scaffolding": {
        "success": true,
        "iterations_needed": 1,
        "conversation_history": [...]
      }
    }
  ],
  "phase2_results": [...],
  "phase3_results": [...],
  "coaching_db": {...}
}
```

### SFT 데이터 형식 (`*_id-mas_{Model}.json`)

```json
[
  {
    "instruction": "시스템 프롬프트",
    "input": "문제 텍스트",
    "output": "SFT용 응답 (Phase 1/2/3 결과)"
  }
]
```

### SFT Case 분류

| Case | 설명 | 응답 출처 |
|------|------|-----------|
| `A` | Phase 1 정답 | 학생 모델 초기 응답 |
| `A-Failed` | Phase 1 실패 후 재구성 | 재구성된 응답 |
| `B` | Phase 2 정답 | Coaching 후 수정된 응답 |
| `C` | Phase 3 모델링 | 교사 모델 응답 |

## 데이터 준비

HuggingFace에서 데이터셋을 다운로드하고 전처리합니다:

```bash
python -m utils.dataset_preparer
```

이 스크립트는 다음 데이터셋을 다운로드합니다:
- **Math 도메인**: GSM8K, MATH, SVAMP, ASDiv, MAWPS, MMLU (수학 과목)

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

**참고**: config 모듈은 리팩토링되어 5개의 서브모듈로 분리되었습니다. backward compatibility를 위해 `from config import ...` 형태의 기존 import는 그대로 작동합니다.

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

## 참고 자료

- **Dick & Carey 모델**: PDF 참고 (prompt 초안.pdf)
- **OpenAI API**: https://platform.openai.com/docs
- **GSM8K**: https://huggingface.co/datasets/openai/gsm8k
- **MATH**: https://huggingface.co/datasets/EleutherAI/hendrycks_math
- **ARC**: https://huggingface.co/datasets/allenai/ai2_arc
- **MMLU**: https://huggingface.co/datasets/cais/mmlu
