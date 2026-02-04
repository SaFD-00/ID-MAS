# ID-MAS: Instructional Design Multi-Agent System

LLM 학습을 위한 Dick & Carey 모델 기반 교수 설계 시스템

## 개요

ID-MAS는 교수 설계(Instructional Design) 이론을 LLM 학습에 적용한 Multi-Agent 시스템입니다.

**핵심 특징:**
- **데이터셋별 분리 학습**: GSM8K, MATH 각각 고유한 Instructional Goal로 독립 학습
- **Iterative Scaffolding Pipeline**: Performance Objectives 기반 평가 + HOT/LOT Scaffolding을 통한 6-Step 반복 학습 (최대 5회 반복)
- **LangGraph 기반 워크플로우**: StateGraph 상태 관리, 체크포인트 기반 Resume 지원

## 빠른 시작

```bash
# 1. 환경 설정
conda create -n ID-MAS python=3.11 -y
conda activate ID-MAS
pip install -r requirements.txt

# 2. 환경변수 설정
cp .env.example .env
# .env 파일을 열어 OPENAI_API_KEY, HF_TOKEN 설정

# 3. 데이터 준비
python -m utils.dataset_preparer
python -m utils.sample_extractor

# 4. 학습 실행
python main.py --mode train --domain math --train-dataset gsm8k

# 5. 평가 실행
python main.py --mode eval --method baseline --domain math --eval-dataset gsm8k
```

### 환경변수

| 변수 | 설명 | 필수 여부 |
|------|------|----------|
| `OPENAI_API_KEY` | OpenAI API 키 | gpt-* 모델 사용 시 |
| `HF_TOKEN` | HuggingFace 토큰 | 로컬 모델 사용 시 |
| `IDMAS_DEBUG_API` | API 디버그 로그 출력 | 선택 (기본값: 0) |

## 디렉토리 구조

```
ID-MAS/
├── main.py                      # 메인 실행 파일
├── config/                      # 설정 모듈
├── design_modules/              # 교수 설계 단계
├── learning_loop/               # Iterative Scaffolding Pipeline
├── models/                      # 모델 래퍼
├── prompts/                     # 프롬프트 템플릿
├── utils/                       # 유틸리티
├── data/                        # 원본 데이터 (train/data/, eval/data/)
└── outputs/                     # 학습 결과물
    └── {domain}/train/{teacher}/{student}/
```

## 문서

| 문서 | 설명 |
|------|------|
| [사용 가이드](USAGE.md) | 지원 모델, 데이터셋, CLI 상세 사용법, 실행 예제 |

## 시스템 아키텍처

ID-MAS는 Dick & Carey 교수 설계 모델을 기반으로 LLM을 학습시키는 Multi-Agent 시스템입니다. LangGraph 기반 Iterative Scaffolding Pipeline을 통해 Performance Objectives 기반 평가와 Socratic 질문으로 SFT 학습 데이터를 생성합니다.

### 시스템 구성

#### 1. Config 모듈 (`config/`)

설정 관리를 위한 서브모듈 구조입니다.

| 파일 | 역할 |
|------|------|
| `__init__.py` | 통합 인터페이스 (backward compatibility) |
| `api.py` | API 키 관리 (OPENAI_API_KEY, HF_TOKEN) |
| `models.py` | Teacher/Student 모델 설정 |
| `domains.py` | 도메인-데이터셋 매핑 |
| `sft.py` | SFT 모델 매핑 |
| `paths.py` | 경로 헬퍼 함수 |

**도메인 설정 (`domains.py`):**

```python
DOMAIN_CONFIG = {
    "math": {
        "training_datasets": ["gsm8k", "math"],
        "eval_datasets": ["gsm8k", "math", "svamp", "asdiv", "mawps"],
    },
    "logical": {
        "training_datasets": ["reclor"],
        "eval_datasets": ["reclor", "anli_r2", "anli_r3", "bbh"],
    },
    "commonsense": {
        "training_datasets": ["arc_c"],
        "eval_datasets": ["arc_c", "strategyqa", "openbookqa"],
    }
}
```

#### 2. 교수 설계 모듈 (`design_modules/`)

Dick & Carey 모델 기반 4단계 설계 프로세스입니다.

| 단계 | 모듈 | 역할 |
|------|------|------|
| Step 0 | `instructional_goal.py` | 샘플 데이터 분석 → Instructional Goal 자동 생성 |
| Step 1 | `analysis.py` | 학습 목표 → Subskills, Subtasks 분해 |
| Step 2 | `objectives.py` | Performance Objectives (B-C-CR) 생성 |
| Step 3 | `rubric.py` | Essay형 평가 루브릭 생성 |

**Instructional Goal 생성 (Step 0):**

```python
# InstructionalGoalGenerator 출력 형식
{
    "instructional_goal": "The model will generate coherent, step-by-step...",
    "cognitive_level": "Apply",  # Bloom's Taxonomy
    "primary_verb": "generate",
    "rationale": "..."
}
```

#### 3. 학습 파이프라인 (`learning_loop/`)

LangGraph 기반 Iterative Scaffolding Pipeline입니다.

```
learning_loop/
├── graph/
│   ├── state.py      # IDMASState (TypedDict 기반 상태 스키마)
│   ├── nodes.py      # scaffolding, advance, finalize 노드
│   └── graph.py      # StateGraph 구성 및 IDMASGraphRunner
├── student_model.py  # 학생 모델 (응답 생성)
└── teacher_model.py  # 교사 모델 (평가, Scaffolding Artifact)
```

**Student Model 주요 기능:**

| 메서드 | 역할 |
|--------|------|
| `generate_initial_response_with_scaffolding()` | Task Analysis와 함께 초기 응답 생성 |
| `respond_with_scaffolding_artifact()` | Scaffolding Artifact 참조하여 개선 응답 생성 |

**Teacher Model 주요 기능:**

| 메서드 | 역할 |
|--------|------|
| `evaluate_with_performance_objectives()` | PO 기반 평가 |
| `generate_scaffolding_artifact()` | HOT/LOT Scaffolding 생성 |
| `reconstruct_successful_scaffolding()` | Case B 응답 재구성 |
| `generate_final_solution()` | Case C 최종 풀이 생성 |

#### 4. Models 패키지 (`models/`)

모델 래퍼 계층 구조입니다.

```
BaseModelWrapper (추상 클래스)
├── TeacherModelWrapper
│   ├── API 모델 (gpt-*, o1-*, o3-*)
│   └── 로컬 HuggingFace 모델
└── StudentModelWrapper
    └── 로컬 HuggingFace 모델
```

**ModelCache:**
- 글로벌 싱글톤 캐시
- Teacher/Student 동일 모델 사용 시 메모리 공유
- 메서드: `get_or_load()`, `is_loaded()`, `clear()`

#### 5. Utils (`utils/`)

| 파일 | 역할 |
|------|------|
| `dataset_preparer.py` | HuggingFace 데이터 다운로드/전처리 |
| `sample_extractor.py` | Instructional Goal용 샘플 추출 |
| `domain_loader.py` | 도메인별 데이터 로더 |
| `answer_extractor.py` | 5가지 답변 타입 추출기 |

**AnswerExtractor 지원 타입:**

| 타입 | 패턴 | 예시 |
|------|------|------|
| `NUMERIC` | `#### 25`, `answer is 3.14` | GSM8K |
| `LATEX` | `\boxed{...}` | MATH |
| `MCQ` | `Answer: A` | ReClor, ARC |
| `BOOLEAN` | `Yes/No`, `True/False` | StrategyQA |
| `TEXT` | 마지막 줄 | BBH |

### Iterative Scaffolding Pipeline

#### Pipeline Step 정의

| Step | 명칭 | 모듈 | 역할 |
|------|------|------|------|
| **Step 1** | Initial Response | `StudentModel` | Task Analysis 기반 초기 응답 생성 |
| **Step 2** | PO Evaluation | `TeacherModel` | Performance Objectives 기반 응답 평가 |
| **Step 3** | Scaffolding Artifact | `TeacherModel` | HOT/LOT Scaffolding 생성 |
| **Step 4** | Student Re-response | `StudentModel` | Scaffolding 참조 개선 응답 |
| **Step 5** | Reconstruction | `TeacherModel` | Case B/C 최종 응답 재구성 |
| **Step 6** | SFT Generation | `nodes.py` | SFT 학습 데이터 생성 |

#### 전체 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                    Design Phase (Step 0-3)                   │
├─────────────────────────────────────────────────────────────┤
│  Samples → Instructional Goal → Analysis → PO → Rubric      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Iterative Scaffolding Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Iteration 1                                          │   │
│  │   [Step 1] Student Initial Response                  │   │
│  │       ↓                                              │   │
│  │   [Step 2] Teacher PO Evaluation ──→ 성공 ──→ Case A │   │
│  │       ↓ (실패)                                       │   │
│  │   [Step 3] Scaffolding Artifact                      │   │
│  │       ↓                                              │   │
│  │   [Step 4] Student Re-response                       │   │
│  └───────────────────────┬──────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Iteration 2-5 (반복)                                 │   │
│  │   [Step 2] Teacher PO Evaluation ──→ 성공 ──→ Case B │   │
│  │       ↓ (실패)                                       │   │
│  │   [Step 3] Scaffolding Artifact                      │   │
│  │       ↓                                              │   │
│  │   [Step 4] Student Re-response                       │   │
│  └───────────────────────┬──────────────────────────────┘   │
│                          ↓ (5회 반복 후 실패)               │
│                       Case C                                 │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Finalization                                         │   │
│  │   [Step 5] Reconstruction (Case B/C만)               │   │
│  │       ↓                                              │   │
│  │   [Step 6] SFT Data Generation                       │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Step별 Fallback 처리

| Step | 실패 원인 | Fallback 동작 |
|------|-----------|---------------|
| **Step 2** | API 에러, 파싱 실패 | 최대 3회 재시도 → 보수적 평가 (all_satisfied=False) |
| **Step 3** | API 에러, 생성 실패 | 최대 3회 재시도 → 기본 Scaffolding 생성 |
| **Step 5** | 재구성 실패 | 최대 3회 재시도 → ground_truth 기반 응답 생성 |

#### Case 분류

| Case | 조건 | SFT 응답 | Step 5 필요 |
|------|------|----------|-------------|
| **A** | 1회차 PO 충족 | 학생 응답 그대로 사용 | No |
| **B** | 2~5회차 PO 충족 | Teacher가 대화 기반 재구성 | Yes |
| **C** | 5회 후 PO 미충족 | Teacher가 정답 기반 재구성 | Yes |

**성공 조건:** 모든 Performance Objectives가 충족되면 (`all_satisfied=True`) 성공으로 처리됩니다.

#### HOT/LOT Scaffolding Artifact

| 유형 | 대상 | 내용 |
|------|------|------|
| **HOT** (High-Order Thinking) | 분석/평가/창조 | strategy_suggestion, partial_example, socratic_question |
| **LOT** (Low-Order Thinking) | 기억/이해/적용 | missed_concept, brief_explanation, key_attention_points |

#### Skip Metadata 구조

Step별 skip 정보는 `skip_details` 딕셔너리에 통합 저장됩니다:

```python
skip_details = {
    "step2_performance_objectives_evaluation": {  # 키 형식: step{N}_{stage}
        "is_fallback": False,
        "attempts_needed": 1
    },
    "step3_scaffolding_artifact_generation": {
        "is_fallback": True,
        "failure_reason": "scaffolding_artifact_generation_failed",
        "last_error": [  # 배열: 모든 에러 수집
            "Error 1: JSON parse failed",
            "Error 2: Timeout",
            "Error 3: Invalid response"
        ],
        "max_retries_exceeded": 3
    },
    "step5_case_b_reconstruction": {
        "is_fallback": True,
        "failure_reason": "reconstruction_failed",
        "last_error": ["Error message"],
        "max_retries_exceeded": 3
    }
}
```

**키 형식 규칙:**
| Step | 키 이름 |
|------|---------|
| Step 2 | `step2_performance_objectives_evaluation` |
| Step 3 | `step3_scaffolding_artifact_generation` |
| Step 5 (Case B) | `step5_case_b_reconstruction` |
| Step 5 (Case C) | `step5_case_c_final_solution` |
| Step 5 (요약) | `step5_summarization` |

### 데이터 흐름

#### 설계 단계

```
Sample Data (*_samples.json)
    ↓
[Step 0: Instructional Goal Generation]
    ↓
Instructional Goal
    ↓
[Step 1: Instructional Analysis]
    ↓
[Step 2: Performance Objectives]
    ↓
[Step 3: Rubric Development]
    ↓
Design Output JSON
```

#### 학습 단계

```
Training Data ({dataset}_train.json)
    ↓
[DomainLoader.load_training_data()]
    ↓
[IDMASGraphRunner.run()]
    ↓
[Iterative Scaffolding per question]
  - Initial Response (Task Analysis 포함)
  - PO Evaluation
  - Scaffolding Artifact (필요시)
  - Student 재응답 (최대 5회)
    ↓
[SFT Data Generation]
    ↓
{dataset}_sft_{model}.json
```

#### 평가 단계

```
Test Data ({dataset}_test.json)
    ↓
[Model Selection]
  - Baseline: 베이스 모델
  - SFT: HuggingFace SFT 모델
  - SFT_ID-MAS: ID-MAS SFT 모델
    ↓
[Student Response Generation]
    ↓
[AnswerExtractor] → 답변 추출
    ↓
[정답 비교]
    ↓
{dataset}_eval_results-{method}.json
```

### 데이터 구조

```
data/                                          # 원본 데이터
├── math/
│   ├── train/data/                            # 학습용 원본 데이터
│   │   ├── gsm8k_train.json
│   │   ├── gsm8k_samples.json                 # Instructional Goal용 샘플
│   │   ├── gsm8k_train_reasoning.json         # Reasoning 버전
│   │   ├── math_train.json
│   │   └── math_samples.json
│   │
│   └── eval/data/                             # 평가용 원본 데이터
│       ├── gsm8k_test.json
│       ├── math_test.json
│       ├── svamp_test.json
│       ├── asdiv_test.json
│       └── mawps_test.json
│
├── logical/
│   ├── train/data/
│   │   ├── reclor_train.json
│   │   └── reclor_samples.json
│   └── eval/data/
│       ├── reclor_test.json
│       ├── anli_r2_test.json
│       ├── anli_r3_test.json
│       └── bbh_test.json                      # 9개 서브태스크 통합
│
└── commonsense/
    ├── train/data/
    │   ├── arc_c_train.json
    │   └── arc_c_samples.json
    └── eval/data/
        ├── arc_c_test.json
        ├── strategyqa_test.json
        └── openbookqa_test.json

outputs/                                       # 학습 결과물
├── math/
│   └── train/
│       └── {Teacher-Model}/                   # Teacher 모델별
│           ├── instructional-design/
│           │   ├── math_gsm8k_design.json
│           │   └── math_math_design.json
│           │
│           └── {Student-Model}/               # Student 모델별
│               ├── gsm8k_sft_{model}.json
│               └── gsm8k_train_summary_{model}.json
│
├── logical/
│   └── train/{Teacher-Model}/{Student-Model}/
│
└── commonsense/
    └── train/{Teacher-Model}/{Student-Model}/

data/{domain}/eval/{Student-Model}/            # 평가 결과
├── gsm8k_eval_results-Baseline.json
├── gsm8k_eval_results-SFT.json
└── gsm8k_eval_results-SFT_ID-MAS.json
```

### JSON 데이터 형식

**학습/평가 데이터:**
```json
{
  "instruction": "You are a helpful math assistant...",
  "input": "문제 텍스트",
  "output": "풀이 과정...\n\nThe answer is \\boxed{42}",
  "metadata": {}
}
```

**설계 결과:**
```json
{
  "domain": "math",
  "train_dataset": "gsm8k",
  "instructional_goal": "Generate coherent, step-by-step...",
  "instructional_analysis": { ... },
  "performance_objectives": { ... },
  "rubrics": { ... },
  "timestamp": "2026-01-16T..."
}
```

**SFT 데이터:**
```json
{
  "instruction": "Solve the following problem.",
  "input": "Question: ...",
  "output": "SFT용 응답",
  "metadata": {
    "id": "gsm8k_train_0",
    "sft_case": "A",
    "ground_truth": "42"
  }
}
```

### 새로운 도메인 추가 가이드

#### Step 1: 데이터 준비

```bash
# 원본 데이터 디렉토리 생성
mkdir -p data/{domain}/train/data
mkdir -p data/{domain}/eval/data

# 학습 데이터: data/{domain}/train/data/{dataset}_train.json
# 평가 데이터: data/{domain}/eval/data/{dataset}_test.json

# 학습 결과물은 자동으로 outputs/{domain}/train/{teacher}/{student}/ 에 저장됨
```

#### Step 2: Config 업데이트

**`config/domains.py`:**

```python
# 1. DATASET_TO_DOMAIN 추가
DATASET_TO_DOMAIN = {
    "your_dataset": "your_domain",
}

# 2. TRAINING_DATASETS 추가
TRAINING_DATASETS = {
    "your_domain": ["your_dataset"],
}

# 3. DOMAIN_CONFIG 추가
DOMAIN_CONFIG = {
    "your_domain": {
        "data_dir": DATA_DIR / "your_domain",
        "training_datasets": ["your_dataset"],
        "eval_datasets": ["your_dataset", "your_eval_dataset"],
        "default_eval": "your_dataset"
    }
}
```

#### Step 3: 실행 및 검증

```bash
# 학습 테스트
python main.py --mode train --domain your_domain --train-dataset your_dataset

# 평가 테스트
python main.py --mode eval --method baseline \
    --domain your_domain --eval-dataset your_dataset
```

#### 주의사항

1. **데이터 형식**: `instruction`, `input`, `output` 필드 필수
2. **Answer Type**: `utils/answer_extractor.py`에서 적절한 추출기 선택
3. **코드 수정 불필요**: 설정 파일만 수정하면 자동으로 동작

## 참고 문헌

1. Dick, W., Carey, L., & Carey, J. O. (2015). The systematic design of instruction (8th ed.). Pearson.
2. Anderson, L. W., & Krathwohl, D. R. (2001). A taxonomy for learning, teaching, and assessing. Longman.
