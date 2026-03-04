# ID-MAS: Instructional Design Multi-Agent System

LLM 학습을 위한 Dick & Carey 모델 기반 교수 설계 시스템

## 개요

ID-MAS는 교수 설계(Instructional Design) 이론을 LLM 학습에 적용한 Multi-Agent 시스템입니다. Dick & Carey의 체계적 교수 설계 모델을 기반으로, Teacher-Student 상호작용을 통해 고품질 SFT(Supervised Fine-Tuning) 데이터를 자동 생성합니다.

**핵심 특징:**
- **데이터셋별 분리 학습**: GSM8K, MATH 각각 고유한 Instructional Goal로 독립 학습
- **Iterative Scaffolding Pipeline**: Performance Objectives 기반 평가 + HOT/LOT Scaffolding + Self-Refinement을 통한 반복 학습 (최대 5회 반복)
- **LangGraph 기반 워크플로우**: StateGraph 상태 관리, 체크포인트 기반 Resume 지원

---

## 시스템 아키텍처

### 3-Phase Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ID-MAS Pipeline Overview                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: Instructional Design                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Samples → Instructional Goal → Analysis → Performance Objectives     │  │
│  │                                    ↓                                 │  │
│  │                        Enhanced Training Data                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    ↓                                        │
│  Phase 2: Adaptive Scaffolding (SFT Data Generation)                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ For each question:                                                   │  │
│  │   Student Response → Teacher PO Evaluation → Scaffolding Artifact    │  │
│  │       ↕ (최대 5회 반복)                                               │  │
│  │   Case A/Case B/Case C 분류 → SFT 데이터 생성                         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    ↓                                        │
│  Phase 3: Instructional Delivery (SFT)                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ SFT Fine-tuning → Evaluation (Baseline / SFT / SFT_ID-MAS)          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Multi-Agent 구성

| Agent | 역할 | 모델 |
|-------|------|------|
| **Teacher** | 교수 설계, PO 평가, Scaffolding 생성, Positive Feedback, 응답 재구성 | OpenAI API (`gpt-5.2`) 또는 로컬 HuggingFace (기본: `Qwen/Qwen3-4B`) |
| **Student** | 초기 응답 생성, Scaffolding 참조 재응답, Self-Refinement | 로컬 HuggingFace (기본: `Qwen/Qwen3-4B`) |

---

### Phase 1: Instructional Design

#### 목적

데이터셋별 학습 목표를 자동 생성하고, 교수 분석을 통해 Performance Objectives를 도출합니다. 이 결과물은 Phase 2의 평가 기준으로 사용됩니다.

#### 설계 단계

```
Sample Data (*_samples.json)
    ↓
[Step 0] Instructional Goal Generation ─── instructional_goal.py
    ↓
[Step 1] Learning Objective 설정
    ↓
[Step 2] Instructional Analysis ─────────── analysis.py
    ↓
[Step 3] Performance Objectives ─────────── objectives.py
    ↓
Design Result JSON ({domain}_{dataset}_design.json)
    ↓
[Enhanced Data Generation] ──────────────── dataset_enhancer.py
    ↓
Enhanced Training Data (outputs/{domain}/train/{student}/data/{dataset}_train_ID-MAS.json)
```

#### Step별 상세

| Step | 명칭 | 모듈 | 입력 | 출력 |
|------|------|------|------|------|
| **Step 0** | Instructional Goal Generation | `design_modules/instructional_goal.py` | `{dataset}_samples.json` (20개 대표 샘플) | `instructional_goal`, `cognitive_level`, `primary_verb` |
| **Step 1** | Learning Objective | — | Instructional Goal | Learning Objective (= Instructional Goal) |
| **Step 2** | Instructional Analysis | `design_modules/analysis.py` | Learning Objective | 트리 구조 (Subskills/Subtasks) |
| **Step 3** | Performance Objectives | `design_modules/objectives.py` | Instructional Analysis | PO 배열 (`target` + `performance_objective`) |
| **Enhanced** | Enhanced Data Generation | `utils/dataset_enhancer.py` | 원본 학습 데이터 + Design Result | `metadata`에 `instructional_goal`, `task_analysis` 추가 |

#### 설계 결과 JSON

```json
{
    "domain": "math",
    "train_dataset": "gsm8k",
    "identifier": "math_gsm8k",
    "instructional_goal": "Generate coherent, step-by-step...",
    "instructional_goal_metadata": { "cognitive_level": "Apply", "primary_verb": "generate" },
    "learning_objective": "...",
    "instructional_analysis": {
        "learning_objective": "...",
        "raw_output": "트리 구조 텍스트",
        "parsed": { }
    },
    "performance_objectives": {
        "performance_objectives": [
            { "target": "Problem Interpretation", "performance_objective": "Correctly identify..." }
        ]
    },
    "timestamp": "2026-01-16T..."
}
```

**저장 경로**: `outputs/{domain}/train/{teacher_short}/instructional-design/{domain}_{dataset}_design.json`

---

### Phase 2: Adaptive Scaffolding — SFT Data Generation

#### 목적

LangGraph 기반 Iterative Scaffolding Pipeline을 통해 각 학습 문제에 대해 교사-학생 반복 상호작용을 수행하고, 그 결과로 SFT 학습 데이터를 생성합니다.

#### LangGraph StateGraph 구조

```
                    ┌──────────────┐
                    │  Entry Point │
                    └──────┬───────┘
                           ↓
                 ┌─────────────────────┐
            ┌──→ │    scaffolding      │ ← 현재 문제를 반복적 스캐폴딩으로 처리
            │    │  (Step 1~5 내부)     │
            │    └─────────┬───────────┘
            │              ↓
            │    ┌─────────────────────┐
            │    │      advance        │ ← 다음 문제로 이동
            │    └─────────┬───────────┘
            │              ↓
            │    ┌─────────────────────┐
            │    │ should_continue?    │ ← 조건부 라우팅
            │    └──┬──────────────┬───┘
            │       │              │
            │  더 있음           완료
            └───────┘              │
                                   ↓
                         ┌─────────────────┐
                         │    finalize      │ ← SFT 데이터 생성
                         └─────────┬───────┘
                                   ↓
                                  END
```

- **모듈**: `learning_loop/graph/graph.py` → `create_idmas_graph()`, `IDMASGraphRunner`
- **상태**: `learning_loop/graph/state.py` → `IDMASState` (TypedDict)
- **노드**: `learning_loop/graph/nodes.py` → `process_question_scaffolding`, `advance_to_next_question`, `generate_sft_data`
- **체크포인터**: `MemorySaver` (LangGraph 내장)

#### Iterative Scaffolding Pipeline (6-Step)

각 문제에 대해 최대 5회 반복하며, 교사의 Scaffolding을 통해 학생의 응답을 개선합니다.

| Step | 명칭 | Agent | 모듈 | 역할 |
|------|------|-------|------|------|
| **Step 1** | Initial Response | Student | `student_model.py` | Task Analysis 기반 초기 응답 생성 |
| **Step 2** | PO Evaluation | Teacher | `teacher_model.py` | Performance Objectives 기반 응답 평가 |
| **Step 3** | Scaffolded Corrective Feedback | Teacher | `teacher_model.py` | HOT/LOT Scaffolding 생성 |
| **Step 4** | Teacher-Supported Reattempt | Student | `student_model.py` | Scaffolding Artifact 참조 개선 응답 |
| **Step 5a-1** | Positive Reinforcement | Teacher | `teacher_model.py` | Case A/B: PO 충족 시 강점 + 개선점 피드백 |
| **Step 5a-2** | Feedback-Driven Elaboration | Student | `student_model.py` | Case A/B: Positive Feedback 기반 응답 개선 |
| **Step 5b** | Teacher Modeling | Teacher | `teacher_model.py` | Case C: 최종 솔루션 생성 |
| **Step 6** | SFT Generation | — | `nodes.py` | SFT 학습 데이터 생성 |

#### 전체 흐름

```
┌─────────────────────────────────────────────────────────────┐
│              Iterative Scaffolding Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Iteration 1                                          │   │
│  │   [Step 1] Student Initial Response                  │   │
│  │       ↓                                              │   │
│  │   [Step 2] Teacher PO Evaluation ──→ 성공 ── Case A  │   │
│  │       ↓ (실패)                                       │   │
│  │   [Step 3] Scaffolding Artifact (HOT/LOT)            │   │
│  │       ↓                                              │   │
│  │   [Step 4] Student Re-response                       │   │
│  └───────────────────────┬──────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Iteration 2-5 (반복)                                 │   │
│  │   [Step 2] Teacher PO Evaluation ──→ 성공 ── Case B  │   │
│  │       ↓ (실패)                                       │   │
│  │   [Step 3] Scaffolding Artifact (HOT/LOT)            │   │
│  │       ↓                                              │   │
│  │   [Step 4] Student Re-response                       │   │
│  └───────────────────────┬──────────────────────────────┘   │
│                          ↓ (5회 반복 후 실패)                │
│                       Case C                                 │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Self-Refinement (Case A / Case B)                    │   │
│  │   [Step 5a-1] Teacher Positive Reinforcement         │   │
│  │       ↓                                              │   │
│  │   [Step 5a-2] Student Feedback-Driven Elaboration    │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ Finalization                                         │   │
│  │   [Step 5b] Teacher Modeling (Case C만)              │   │
│  │       ↓                                              │   │
│  │   [Step 6] SFT Data Generation                       │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Case 분류

| Case | 정식 명칭 | 조건 | SFT 응답 소스 | Self-Refinement |
|------|----------|------|--------------|-----------------|
| **Case A** | Independent Performance Mastery | 1회차 PO 충족 | Student Self-Refined Response | Yes (Step 5a-1→5a-2) |
| **Case B** | Scaffolded & Coached Mastery | 2~5회차 PO 충족 | Student Self-Refined Response | Yes (Step 5a-1→5a-2) |
| **Case C** | Teacher Modeling Distillation | 5회 후 PO 미충족 | Teacher 최종 솔루션 | No (Step 5b) |

**성공 조건**: 모든 Performance Objectives가 충족되면 (`all_satisfied == True`) 성공으로 처리됩니다.

#### HOT/LOT Scaffolding Artifact

Bloom's Taxonomy 기반으로 인지 수준별 차별화된 Scaffolding을 제공합니다.

| 유형 | 대상 인지 수준 | 제공 내용 |
|------|--------------|----------|
| **HOT** (High-Order Thinking) | 분석/평가/창조 | `strategy_suggestion`, `partial_example`, `key_attention_points` |
| **LOT** (Low-Order Thinking) | 기억/이해/적용 | `missed_concept`, `brief_explanation` |

#### Checkpoint & Resume

- **증분 체크포인트**: 각 문제 처리 후 `save_incremental_checkpoint()` 호출 → 로그 파일 업데이트
- **파일 기반 Resume**: `{dataset}_train_id-mas_{model}_logs.json`에서 `processed_ids` 추출 → 미처리 문제만 재실행
- **상태 복원**: `load_checkpoint_from_logs()` → `restore_state_from_checkpoint()`
- **조기 종료**: 모든 문제 처리 완료 시 그래프 실행 건너뛰기

#### SFT 데이터 출력 형식

```json
[
    {
        "instruction": "Solve the following problem.",
        "input": "Question: ...",
        "output": "SFT용 응답",
        "metadata": {
            "id": "gsm8k_train_0",
            "sft_case": "case_a_independent_performance_mastery",
            "ground_truth": "42"
        }
    }
]
```

**저장 경로**: `outputs/{domain}/train/{teacher_short}/{student_short}/{dataset}_train_id-mas_{model}.json`

---

### Phase 3: Instructional Delivery — SFT

#### 목적

Phase 2에서 생성된 SFT 데이터로 학생 모델을 파인튜닝하고, 다양한 평가 데이터셋에서 성능을 측정합니다.

#### LlamaFactory LoRA SFT

[LlamaFactory](https://github.com/hiyouga/LlamaFactory)를 사용하여 LoRA Fine-tuning을 수행합니다.

**SFT 설정:**

| Parameter | Value |
|-----------|-------|
| Method | LoRA (rank=8, target=all) |
| Template | Qwen3 |
| Epochs | 3 |
| LR Scheduler | Cosine (warmup 0.1) |
| Learning Rate | 1e-4 |
| Precision | bf16 |
| Cutoff Length | 2048 |

**모델별 배치 설정** (Effective Batch Size = 16):

| Model | Batch Size | Gradient Accumulation | DeepSpeed |
|-------|-----------|----------------------|-----------|
| Qwen3-0.6B | 8 | 2 | - |
| Qwen3-1.7B | 4 | 4 | - |
| Qwen3-4B | 2 | 8 | - |
| Qwen3-8B | 1 | 16 | - |
| Qwen3-14B | 1 | 16 | - |
| Qwen3-32B | 1 | 16 | ZeRO-3 |

**자동화 파이프라인** (`ID-MAS.ipynb` Section 2.0):
1. ID-MAS 출력 SFT 데이터를 LlamaFactory `data/` 디렉토리로 복사
2. `dataset_info.json`에 데이터셋 등록
3. 실험별 `train.yaml` / `merge.yaml` 자동 생성
4. LoRA 학습 → 어댑터 병합 → HuggingFace Hub 업로드

#### 평가 방법

`--student-model`에 HuggingFace Hub 모델 ID를 직접 지정하여 평가합니다.

| 방법 | `--student-model` 예시 | 설명 |
|------|----------------------|------|
| **Baseline** | `Qwen/Qwen3-0.6B` | 파인튜닝 없는 기본 성능 |
| **SFT_ID-MAS** | `SaFD-00/qwen3-0.6b-id-mas-math-gsm8k` | ID-MAS 방식 SFT 모델 |

#### Answer Extractor

`utils/answer_extractor.py`에서 5가지 답변 타입을 지원합니다.

| 타입 | 추출기 | 패턴 | 데이터셋 |
|------|--------|------|---------|
| `NUMERIC` | `NumericExtractor` | `#### 25`, `answer is 3.14` | GSM8K, SVAMP, ASDiv |
| `LATEX` | `LaTeXExtractor` | `\boxed{...}` | MATH, MAWPS |
| `MCQ` | `MCQExtractor` | `Answer: A` | ReClor, ARC-C, OpenBookQA |
| `BOOLEAN` | `BooleanExtractor` | `Yes/No`, `True/False` | StrategyQA |
| `TEXT` | `TextExtractor` | 마지막 줄 | BBH |

- **기호적 비교**: LaTeX/Numeric에서 `sympy` 기반 수학적 동치성 검증 (예: `\frac{1}{2}` == `0.5`)

#### 평가 결과 JSON 형식

```json
{
    "domain": "math",
    "eval_dataset": "gsm8k",
    "student_model": "Qwen/Qwen3-4B",
    "answer_type": "numeric",
    "total_questions": 1319,
    "evaluated_questions": 1319,
    "correct_count": 456,
    "accuracy": 0.3457,
    "question_results": [ ]
}
```

**저장 경로**: `outputs/{domain}/eval/{student_short}/{dataset}_eval_results.json`

---

## 동작 예시

> GSM8K 데이터셋 + Qwen3-4B 모델 기반 동작 흐름을 요약합니다.

### Phase 1 예시: Instructional Design

**Step 0 — Instructional Goal 생성**: 20개 GSM8K 샘플을 분석하여 다음과 같은 학습 목표를 자동 생성합니다.

```json
{
    "instructional_goal": "The model will solve complex mathematical word problems by applying arithmetic operations, logical reasoning, and mathematical concepts to determine the correct numerical answer in real-world contexts.",
    "cognitive_level": "Apply",
    "primary_verb": "generate"
}
```

**Step 2 — Instructional Analysis**: Learning Objective를 Subskills/Subtasks 트리로 분해합니다.

```
Instructional Goal: ... (Apply)
 ├── Understand Problem Context (Understand)
 │   ├── Identify key information (Remember)
 │   └── Determine relevant concepts (Understand)
 ├── Apply Arithmetic Operations (Apply)
 ├── Apply Logical Reasoning (Apply)
 └── Verify Final Answer (Evaluate)
```

**Step 3 — Performance Objectives**: 각 Subskill에 대해 행동(Behavior), 조건(Condition), 기준(Criterion)을 통합한 PO를 생성합니다 (예: 6개 PO).

### Phase 2 예시: Adaptive Scaffolding

각 학습 문제에 대해 아래 3가지 Case 중 하나로 분류됩니다.

#### Case A: Independent Performance Mastery (1회차 PO 충족)

```
Student Initial Response → Teacher PO Evaluation (전체 PO 충족)
    → Teacher Positive Reinforcement (강점 + 개선점 피드백)
    → Student Feedback-Driven Elaboration (Self-Refined Response)
    → SFT 데이터 생성 (output = Self-Refined Response)
```

Student가 첫 시도에서 모든 Performance Objectives를 충족합니다. Teacher가 강점과 개선 방향을 피드백하고, Student는 이를 반영하여 응답을 정교화합니다.

#### Case B: Scaffolded & Coached Mastery (2~5회차 PO 충족)

```
Student Initial Response → Teacher PO Evaluation (일부 PO 미충족)
    → [반복] Scaffolding Artifact (HOT/LOT) → Student Re-response → PO Evaluation
    → (N회차에서 전체 PO 충족)
    → Teacher Positive Reinforcement → Student Feedback-Driven Elaboration
    → SFT 데이터 생성 (output = Self-Refined Response)
```

Teacher의 HOT/LOT Scaffolding을 통해 미충족 PO를 하나씩 해결합니다. 각 반복마다 인지 수준별 차별화된 피드백이 누적됩니다.

#### Case C: Teacher Modeling Distillation (5회 반복 실패)

```
Student Initial Response → [5회 반복 Scaffolding] → (전체 PO 여전히 미충족)
    → Teacher Modeling (전체 iteration_summaries 기반 최종 솔루션 생성)
    → SFT 데이터 생성 (output = Teacher 최종 솔루션)
```

5회 반복 후에도 PO를 충족하지 못한 경우, Teacher가 직접 모범 풀이를 생성합니다. 이 경우 Self-Refinement은 수행되지 않습니다.

---

## 실험 구성

### 실험 목록

| ID | Student Model | Teacher Model | Type | 설명 |
|----|--------------|---------------|------|------|
| [1] | Qwen3-0.6B | Qwen3-0.6B | Self-Distillation | 최소 모델 자기증류 |
| [2] | Qwen3-1.7B | Qwen3-1.7B | Self-Distillation | |
| [3] | Qwen3-4B | Qwen3-4B | Self-Distillation | |
| [4] | Qwen3-8B | Qwen3-8B | Self-Distillation | |
| [5] | Qwen3-14B | Qwen3-14B | Self-Distillation | |
| [6] | Qwen3-32B | Qwen3-32B | Self-Distillation | 최대 모델 자기증류 |
| [7] | Qwen3-4B | GPT-5.2 | Cross-Model | 대형→소형 전이 |
| [8] | Qwen3-8B | GPT-5.2 | Cross-Model | 대형→소형 전이 |

### 평가 도메인 & 데이터셋

| Domain | Training (In-domain) | Evaluation (+ Out-of-domain) |
|--------|---------------------|------------------------------|
| Math | GSM8K, MATH | GSM8K, MATH, SVAMP, ASDiv, MAWPS |
| Logical | ReClor | ReClor, ANLI-R2, ANLI-R3, BBH |
| Commonsense | ARC-C | ARC-C, StrategyQA, OpenBookQA |

### HuggingFace 모델 명명 규칙

SFT 모델은 HuggingFace Hub에 다음 형식으로 업로드됩니다:

```
SaFD-00/{model}-id-mas-{domain}-{dataset}          # Self-Distillation
SaFD-00/{model}-id-mas-{domain}-{dataset}-gpt52    # Cross-Model (GPT-5.2 Teacher)
```

**예시:**
- `SaFD-00/qwen3-0.6b-id-mas-math-gsm8k` — Qwen3-0.6B, GSM8K 학습, Self-Distillation
- `SaFD-00/qwen3-4b-id-mas-logical-reclor-gpt52` — Qwen3-4B, ReClor 학습, GPT-5.2 Teacher

---

## 빠른 시작

### CLI 방식

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

# 5. 멀티 GPU 학습 (Student: GPU 0, Teacher: GPU 1)
python main.py --mode train --domain math --train-dataset gsm8k \
    --student-model Qwen/Qwen3-4B --teacher-model Qwen/Qwen3-32B \
    --student-gpu 0 --teacher-gpu 1

# 6. 평가 실행
# Baseline (원본 모델)
python main.py --mode eval --domain math --eval-dataset gsm8k \
    --student-model Qwen/Qwen3-0.6B --student-gpu 0

# SFT 모델 (HF Hub에서 로드)
python main.py --mode eval --domain math --eval-dataset gsm8k \
    --student-model SaFD-00/qwen3-0.6b-id-mas-math-gsm8k --student-gpu 0
```

### Notebook 방식 (Google Colab)

`ID-MAS.ipynb`를 사용하여 전체 파이프라인을 실행할 수 있습니다.

```
ID-MAS.ipynb 구조:
├── 0. Environment Setup        ← BASE_DIR 설정 (경로 1곳만 수정)
├── 1. ID-MAS Training          ← SFT 데이터 생성 (8개 실험 x 4개 데이터셋)
│   ├── 1.1 Self-Distillation   ← 실험 [1]~[6]
│   └── 1.2 Cross-Model         ← 실험 [7]~[8]
├── 2. LlamaFactory SFT         ← LoRA Fine-tuning & HF Upload
│   ├── 2.0 Setup               ← LlamaFactory 설치, 데이터/YAML 자동 생성
│   ├── 2.1 Self-Distillation   ← LoRA SFT [1]~[6]
│   ├── 2.2 Cross-Model         ← LoRA SFT [7]~[8]
│   └── 2.3 Merge & Upload      ← LoRA 병합 → HuggingFace Hub 업로드
└── 3. Evaluation               ← Baseline vs SFT 성능 비교
    ├── 3.1 Self-Distillation   ← 평가 [1]~[6]
    └── 3.2 Cross-Model         ← 평가 [7]~[8]
```

**사용법:**
1. `ID-MAS.ipynb`를 Google Colab에서 열기
2. 상단의 `BASE_DIR` 변수를 프로젝트 경로로 수정
3. 셀을 순서대로 실행

### 환경변수

| 변수 | 설명 | 필수 여부 |
|------|------|----------|
| `OPENAI_API_KEY` | OpenAI API 키 | gpt-* 모델 사용 시 |
| `HF_TOKEN` | HuggingFace 토큰 | 로컬 모델 사용 시 |
| `IDMAS_DEBUG_API` | API 디버그 로그 출력 | 선택 (기본값: 0) |

---

## 프로젝트 구조

### 디렉토리 구조

```
ID-MAS/
├── main.py                      # 메인 실행 파일
├── ID-MAS.ipynb                 # 전체 파이프라인 노트북 (Colab)
├── config/                      # 설정 모듈
│   ├── api.py                   # API 키 관리, PROJECT_ROOT
│   ├── models.py                # Teacher/Student 모델 설정
│   ├── domains.py               # 도메인-데이터셋 매핑
│   ├── sft.py                   # SFT 모델명 매핑
│   └── paths.py                 # 경로 헬퍼
├── design_modules/              # 교수 설계 단계
│   ├── instructional_goal.py    # Step 0: Instructional Goal 생성
│   ├── analysis.py              # Step 2: Instructional Analysis
│   └── objectives.py            # Step 3: Performance Objectives
├── learning_loop/               # Iterative Scaffolding Pipeline
│   ├── student_model.py         # Student Agent (Step 1, 4, 5a-2)
│   ├── teacher_model.py         # Teacher Agent (Step 2, 3, 5a-1, 5b)
│   └── graph/                   # LangGraph 워크플로우
│       ├── graph.py             # StateGraph 정의, IDMASGraphRunner
│       ├── nodes.py             # 노드 함수, SFT 데이터 생성
│       └── state.py             # IDMASState (TypedDict)
├── models/                      # 모델 래퍼
│   ├── base_wrapper.py          # 추상 기본 클래스
│   ├── teacher_wrapper.py       # Teacher 모델 래퍼 (OpenAI API + 로컬)
│   ├── student_wrapper.py       # Student 모델 래퍼 (로컬)
│   ├── model_cache.py           # 글로벌 싱글톤 캐시
│   ├── local_model_mixin.py     # 로컬 모델 공용 기능
│   └── remote_model.py          # RemoteLLMProxy (GPU 격리 subprocess)
├── prompts/                     # 프롬프트 상수 (템플릿)
│   ├── design_prompts.py        # Phase 1 프롬프트
│   └── learning_prompts.py      # Phase 2 프롬프트
├── utils/                       # 유틸리티
│   ├── prompt_helpers.py        # 프롬프트 구성 헬퍼
│   ├── base_loader.py           # 데이터 로더, AnswerType enum
│   ├── domain_loader.py         # 도메인별 JSON 데이터 로더
│   ├── answer_extractor.py      # 5가지 답변 타입 추출기
│   ├── dataset_preparer.py      # HuggingFace 데이터 다운로드/전처리
│   ├── sample_extractor.py      # Instructional Goal용 샘플 추출
│   ├── dataset_enhancer.py      # 학습 데이터 instruction 강화
│   └── dataset_registry.py      # 데이터셋 메타데이터 레지스트리
├── data/                        # 원본 데이터 (train/data/, eval/data/)
├── outputs/                     # 학습 결과물
│   └── {domain}/train/{teacher}/{student}/
└── LlamaFactory/                # LlamaFactory (LoRA SFT)
    ├── data/                    # SFT 데이터 + dataset_info.json
    ├── examples/train_custom/   # 실험별 train.yaml, merge.yaml
    └── saves/                   # LoRA 체크포인트 + 병합 모델
```

### GPU 할당 아키텍처

`--student-gpu`와 `--teacher-gpu` CLI 옵션을 통해 모델별 GPU를 지정할 수 있습니다.

```
시나리오 A: 같은 로컬 모델 (ModelCache 공유)
┌──────────────────────────────────────────────────┐
│                   Main Process                    │
│  ┌─────────┐                                     │
│  │ vLLM LLM │ ← ModelCache에서 공유              │
│  │ (GPU 0)  │                                     │
│  └────┬─────┘                                     │
│       ├── StudentModelWrapper                     │
│       └── TeacherModelWrapper                     │
└──────────────────────────────────────────────────┘

시나리오 B: 다른 로컬 모델 (GPU 분리)
┌──────────────────────────────────────────────────┐
│                   Main Process                    │
│  ┌─────────────┐    ┌──────────────────────┐     │
│  │ vLLM LLM    │    │ RemoteLLMProxy       │     │
│  │ Student     │    │ Teacher (Pipe IPC)    │     │
│  │ (GPU 0)     │    └──────────┬───────────┘     │
│  └─────────────┘               │                  │
└────────────────────────────────┼──────────────────┘
                                 │
                    ┌────────────┴─────────────┐
                    │      Child Process        │
                    │  CUDA_VISIBLE_DEVICES=1   │
                    │  ┌─────────────┐          │
                    │  │ vLLM LLM    │          │
                    │  │ Teacher     │          │
                    │  │ (GPU 1)     │          │
                    │  └─────────────┘          │
                    └──────────────────────────┘

시나리오 C: API Teacher
┌──────────────────────────────────────────────────┐
│                   Main Process                    │
│  ┌─────────────┐    ┌──────────────────────┐     │
│  │ vLLM LLM    │    │ OpenAI API Client    │     │
│  │ Student     │    │ Teacher (gpt-5.2)    │     │
│  │ (GPU 0)     │    │ (no GPU needed)      │     │
│  └─────────────┘    └──────────────────────┘     │
└──────────────────────────────────────────────────┘
```

**ModelCache**: 글로벌 싱글톤으로 `(model_name, device, gpu_ids)` 키 기반 관리. Teacher/Student 동일 모델 사용 시 메모리 공유.

**RemoteLLMProxy** (`models/remote_model.py`): `CUDA_VISIBLE_DEVICES` 설정 후 자식 프로세스에서 vLLM 로드, `multiprocessing.Pipe`로 chat 요청 수신/응답.

---

## CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--mode` | 실행 모드 (`train` / `eval`) | 필수 |
| `--domain` | 도메인 (`math` / `logical` / `commonsense`) | 필수 |
| `--train-dataset` | 학습 데이터셋 (train 모드) | — |
| `--eval-dataset` | 평가 데이터셋 (eval 모드) | — |
| `--student-model` | Student 모델 (HuggingFace Hub 모델 ID) | `Qwen/Qwen3-4B` |
| `--teacher-model` | Teacher 모델 | `Qwen/Qwen3-4B` |
| `--student-gpu` | Student 모델 GPU 인덱스 (쉼표로 다중 GPU 지정) | 자동 (CUDA_VISIBLE_DEVICES) |
| `--teacher-gpu` | Teacher 모델 GPU 인덱스 (API 모델은 무시) | 자동 (CUDA_VISIBLE_DEVICES) |
| `--run-design` | 설계 단계 강제 재실행 (train 전용) | `False` |
| `--max-iterations` | Iterative Scaffolding 최대 반복 횟수 (train 전용) | `5` |
| `--resume` | 체크포인트에서 이어서 학습 (train 전용) | `True` |
| `--eval-resume` | 기존 결과에서 이어서 평가 (eval 전용) | `True` |

> 상세 사용법은 [사용 가이드](USAGE.md) 참조

---

## 데이터 구조

### 디렉토리 구조

```
data/                                          # 원본 데이터
├── math/
│   ├── train/data/                            # 학습용 원본 데이터
│   │   ├── gsm8k_train.json
│   │   ├── gsm8k_samples.json                 # Instructional Goal용 샘플
│   │   ├── math_train.json
│   │   └── math_samples.json
│   └── eval/data/                             # 평가용 원본 데이터
│       ├── gsm8k_test.json
│       ├── math_test.json
│       ├── svamp_test.json
│       ├── asdiv_test.json
│       └── mawps_test.json
├── logical/
│   ├── train/data/
│   │   ├── reclor_train.json
│   │   └── reclor_samples.json
│   └── eval/data/
│       ├── reclor_test.json
│       ├── anli_r2_test.json
│       ├── anli_r3_test.json
│       └── bbh_test.json                      # 9개 서브태스크 통합
└── commonsense/
    ├── train/data/
    │   ├── arc_c_train.json
    │   └── arc_c_samples.json
    └── eval/data/
        ├── arc_c_test.json
        ├── strategyqa_test.json
        └── openbookqa_test.json

outputs/                                       # 학습 결과물
├── {domain}/
│   ├── train/
│   │   ├── {Teacher-Model}/
│   │   │   ├── instructional-design/
│   │   │   │   └── {domain}_{dataset}_design.json
│   │   │   └── {Student-Model}/
│   │   │       ├── {dataset}_train_id-mas_{model}.json      # SFT 데이터
│   │   │       └── {dataset}_train_id-mas_{model}_logs.json  # 학습 로그
│   │   └── {Student-Model}/data/
│   │       └── {dataset}_train_ID-MAS.json                   # Enhanced 데이터
│   └── eval/
│       └── {Student-Model}/
│           └── {dataset}_eval_results.json                   # 평가 결과
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

---

## 새로운 도메인 추가 가이드

### Step 1: 데이터 준비

```bash
mkdir -p data/{domain}/train/data
mkdir -p data/{domain}/eval/data

# 학습 데이터: data/{domain}/train/data/{dataset}_train.json
# 평가 데이터: data/{domain}/eval/data/{dataset}_test.json
```

### Step 2: Config 업데이트

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

### Step 3: 실행 및 검증

```bash
# 학습
python main.py --mode train --domain your_domain --train-dataset your_dataset

# 평가
python main.py --mode eval --domain your_domain --eval-dataset your_dataset
```

### 주의사항

1. **데이터 형식**: `instruction`, `input`, `output` 필드 필수
2. **Answer Type**: `utils/answer_extractor.py`에서 적절한 추출기 선택
3. **코드 수정 불필요**: 설정 파일만 수정하면 자동으로 동작

---

## 문서

| 문서 | 설명 |
|------|------|
| [ID-MAS.ipynb](ID-MAS.ipynb) | 전체 파이프라인 노트북 (Training → SFT → Evaluation) |
| [사용 가이드](USAGE.md) | 지원 모델, 데이터셋, CLI 상세 사용법, 실행 예제 |

## 참고 문헌

1. Dick, W., Carey, L., & Carey, J. O. (2015). The systematic design of instruction (8th ed.). Pearson.
2. Anderson, L. W., & Krathwohl, D. R. (2001). A taxonomy for learning, teaching, and assessing. Longman.
