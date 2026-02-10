# ID-MAS: System Architecture

## 시스템 개요

ID-MAS는 Dick & Carey 교수 설계 모델을 기반으로 LLM을 학습시키는 Multi-Agent 시스템입니다. 전체 파이프라인은 3개 Phase로 구성됩니다.

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
│  │   Case A/B/C 분류 → SFT 데이터 생성                                   │  │
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
| **Teacher** | 교수 설계, PO 평가, Scaffolding 생성, 응답 재구성 | OpenAI API (`gpt-5.2`) 또는 로컬 HuggingFace |
| **Student** | 초기 응답 생성, Scaffolding 참조 재응답 | 로컬 HuggingFace (기본: `Qwen/Qwen3-1.7B`) |

---

## Phase 1: Instructional Design Phase

### 목적

데이터셋별 학습 목표를 자동 생성하고, 교수 분석을 통해 Performance Objectives를 도출합니다. 이 결과물은 Phase 2의 평가 기준으로 사용됩니다.

### 설계 단계

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
Enhanced Training Data ({dataset}_train_ID-MAS_{teacher}_{student}.json)
```

### Step 0: Instructional Goal Generation

샘플 데이터를 분석하여 데이터셋 고유의 학습 목표를 자동 생성합니다.

- **모듈**: `design_modules/instructional_goal.py` → `InstructionalGoalGenerator`
- **입력**: `{dataset}_samples.json` (20개 대표 샘플, `utils/sample_extractor.py`로 추출)
- **출력**:
  ```json
  {
      "instructional_goal": "The model will generate coherent, step-by-step...",
      "cognitive_level": "Apply",
      "primary_verb": "generate",
      "rationale": "..."
  }
  ```
- **재시도**: 최대 3회, 실패 시 프로그램 종료

### Step 1: Learning Objective

Instructional Goal을 Learning Objective로 설정합니다. 커스텀 목표를 직접 지정할 수도 있습니다.

### Step 2: Instructional Analysis

학습 목표를 Subskills과 Subtasks로 분해합니다.

- **모듈**: `design_modules/analysis.py` → `InstructionalAnalysis`
- **출력**: 트리 구조 (├──, └── 형식) + 파싱된 서브스킬 딕셔너리
- **재시도**: 최대 3회

### Step 3: Performance Objectives

ABCD 모델 기반 수행목표를 생성합니다. Phase 2에서 학생 응답 평가의 기준이 됩니다.

- **모듈**: `design_modules/objectives.py` → `PerformanceObjectives`
- **출력**:
  ```json
  {
      "performance_objectives": [
          {
              "target": "Problem Interpretation",
              "Behavior": "Correctly identify key quantities and relationships",
              "Condition": "Given a word problem with numerical data",
              "Criterion": "All relevant information extracted without errors"
          }
      ]
  }
  ```
- **검증**: `validate_objectives()` 함수로 형식 검증
- **재시도**: 최대 3회

### Enhanced Data Generation

원본 학습 데이터의 instruction을 Instructional Goal + Task Analysis로 강화합니다.

- **모듈**: `utils/dataset_enhancer.py` → `DataEnhancer`, `ENHANCED_INSTRUCTION_TEMPLATE`
- **입력**: `{dataset}_train.json` (원본) + Design Result
- **출력**: `{dataset}_train_ID-MAS_{teacher}_{student}.json`
- **변환**: `instruction` 필드만 enhanced instruction으로 교체, `input`/`output`/`metadata` 유지

### 설계 결과 저장

```json
{
    "domain": "math",
    "train_dataset": "gsm8k",
    "identifier": "math_gsm8k",
    "instructional_goal": "Generate coherent, step-by-step...",
    "instructional_goal_metadata": { ... },
    "learning_objective": "...",
    "instructional_analysis": {
        "learning_objective": "...",
        "raw_output": "트리 구조 텍스트",
        "parsed": { ... }
    },
    "performance_objectives": {
        "performance_objectives": [ ... ]
    },
    "timestamp": "2026-01-16T..."
}
```

**저장 경로**: `outputs/{domain}/train/{teacher_short}/instructional-design/{domain}_{dataset}_design.json`

---

## Phase 2: Adaptive Scaffolding Phase — SFT Data Generation

### 목적

LangGraph 기반 Iterative Scaffolding Pipeline을 통해 각 학습 문제에 대해 교사-학생 반복 상호작용을 수행하고, 그 결과로 SFT(Supervised Fine-Tuning) 학습 데이터를 생성합니다.

### LangGraph StateGraph 구조

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

### Iterative Scaffolding Pipeline (6-Step)

각 문제에 대해 최대 5회 반복하며, 교사의 Scaffolding을 통해 학생의 응답을 개선합니다.

#### Pipeline Step 정의

| Step | 명칭 | Agent | 모듈 | 역할 |
|------|------|-------|------|------|
| **Step 1** | Initial Response | Student | `student_model.py` | Task Analysis 기반 초기 응답 생성 |
| **Step 2** | PO Evaluation | Teacher | `teacher_model.py` | Performance Objectives 기반 응답 평가 |
| **Step 3** | Scaffolding Artifact | Teacher | `teacher_model.py` | HOT/LOT Scaffolding 생성 |
| **Step 4** | Student Re-response | Student | `student_model.py` | Scaffolding Artifact 참조 개선 응답 |
| **Step 5** | Reconstruction | Teacher | `teacher_model.py` | Case B/C 최종 응답 재구성 |
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
│  │   [Step 2] Teacher PO Evaluation ──→ 성공 ──→ Case A │   │
│  │       ↓ (실패)                                       │   │
│  │   [Step 3] Scaffolding Artifact (HOT/LOT)            │   │
│  │       ↓                                              │   │
│  │   [Step 4] Student Re-response                       │   │
│  └───────────────────────┬──────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Iteration 2-5 (반복)                                 │   │
│  │   [Step 2] Teacher PO Evaluation ──→ 성공 ──→ Case B │   │
│  │       ↓ (실패)                                       │   │
│  │   [Step 3] Scaffolding Artifact (HOT/LOT)            │   │
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

#### Step 1: Initial Response (Student)

- **함수**: `StudentModel.generate_initial_response_with_scaffolding()`
- **입력**: `problem_text`, `task_analysis`, `instructional_goal`
- **동작**: Task Analysis를 시스템 프롬프트에 포함하여 체계적 풀이 유도
- **1회차에만 실행** (2회차부터는 Step 4로 대체)

#### Step 2: PO Evaluation (Teacher)

- **함수**: `TeacherModel.evaluate_with_performance_objectives()`
- **입력**: `student_response`, `performance_objectives`, `problem_text`, `ground_truth`
- **동작**: 각 PO에 대해 `is_satisfied` 판단 + 피드백 질문 생성
- **출력**:
  ```json
  {
      "performance_evaluation": [
          {
              "objective_content": "PO 내용",
              "is_satisfied": true,
              "reason_for_unmet_objective": "",
              "feedback_question": ""
          }
      ],
      "overall_assessment": {
          "objectives_met": "3 of 3",
          "all_satisfied": true,
          "primary_weakness": "",
          "recommended_focus": ""
      }
  }
  ```
- **성공 조건**: `all_satisfied == True` → 반복 종료
- **Fallback**: 최대 3회 JSON 파싱 재시도, 실패 시 `all_satisfied=False`로 보수적 처리

#### Step 3: Scaffolding Artifact — HOT/LOT (Teacher)

- **함수**: `TeacherModel.generate_scaffolding_artifact()`
- **조건**: `all_satisfied == False`인 경우에만 호출
- **동작**: 미충족 PO에 대해 인지 수준별 Scaffolding 생성
- **출력**:
  ```json
  {
      "scaffolding_artifacts": [
          {
              "target_objective": "대상 PO",
              "skill_type": "HOT",
              "cognitive_level": "Analysis",
              "failure_analysis": "학생 오류 분석",
              "scaffolding_content": {
                  "strategy_suggestion": "...",
                  "partial_example": "...",
                  "feedback_question": "..."
              }
          }
      ],
      "scaffolding_summary": "3~5문장 요약"
  }
  ```
- **누적**: `scaffolding_artifacts`에 iteration별로 누적 저장
- **Fallback**: 최대 3회 재시도, 실패 시 기본 LOT Scaffolding 생성

#### Step 4: Student Re-response (Student)

- **함수**: `StudentModel.respond_with_scaffolding_artifact()`
- **조건**: iteration >= 2
- **입력**: `problem_text`, `previous_response`, `scaffolding_artifact`, `task_analysis`
- **동작**: Scaffolding Artifact를 참조하여 개선된 응답 생성 (출처 인용 포함)
- **DB 참조 추출**: `StudentModel.extract_artifact_references()` — "Information Retrieved from Scaffolding Artifact:" 섹션 파싱

#### Step 5: Reconstruction (Teacher)

Case에 따라 다른 재구성 전략을 적용합니다.

**Case A** (iteration == 1, `all_satisfied == True`):
- 재구성 불필요, 학생 원본 응답 그대로 사용

**Case B** (iteration >= 2, `all_satisfied == True`):
- **함수**: `TeacherModel.reconstruct_successful_scaffolding()`
- **동작**: 대화 히스토리를 AI 기반으로 요약 → 스캐폴딩 과정을 통합한 정제 응답 생성
- **출력**: `reconstructed_response`, `key_learning_points`, `improvement_summary`
- **Fallback**: 요약 실패 시 truncation fallback, 재구성 실패 시 최종 학생 응답 사용

**Case C** (iteration == max_iterations, `all_satisfied == False`):
- **함수**: `TeacherModel.generate_final_solution()`
- **동작**: `extract_student_weaknesses()`로 약점 추출 → 약점을 보완한 교육적 정답 풀이 생성
- **출력**: `solution_explanation`, `addressed_weaknesses`, `key_learning_points`, `final_answer`
- **Fallback**: 최대 3회 재시도, 실패 시 ground_truth 기반 응답 생성

#### Step 6: SFT Data Generation

- **함수**: `generate_sft_data()` (`nodes.py`)
- **동작**: `scaffolding_results`에서 `is_skipped == False`인 항목을 SFT 형식으로 변환
- **변환 규칙**:
  - `instruction`: Case별 기본값 또는 enhanced instruction
  - `input`: `"Question: {문제 텍스트}"`
  - `output`: Case A → 학생 응답, Case B → 재구성 응답, Case C → 최종 풀이
  - `metadata`: `id`, `sft_case`, `ground_truth`

### Case 분류

| Case | 조건 | SFT 응답 소스 | Step 5 필요 |
|------|------|--------------|-------------|
| **A** | 1회차 PO 충족 | 학생 응답 그대로 사용 | No |
| **B** | 2~5회차 PO 충족 | Teacher가 대화 기반 재구성 | Yes |
| **C** | 5회 후 PO 미충족 | Teacher가 정답 기반 재구성 | Yes |

**성공 조건**: 모든 Performance Objectives가 충족되면 (`all_satisfied == True`) 성공으로 처리됩니다.

### HOT/LOT Scaffolding Artifact

Bloom's Taxonomy 기반으로 인지 수준별 차별화된 Scaffolding을 제공합니다.

| 유형 | 대상 인지 수준 | 제공 내용 |
|------|--------------|----------|
| **HOT** (High-Order Thinking) | 분석/평가/창조 | `strategy_suggestion`, `partial_example`, `feedback_question` |
| **LOT** (Low-Order Thinking) | 기억/이해/적용 | `missed_concept`, `brief_explanation`, `key_attention_points` |

### Skip/Fallback 처리

각 Step에서 API 에러나 파싱 실패 발생 시, 최대 3회 재시도 후 Fallback 처리됩니다.

#### Step별 Fallback 동작

| Step | 실패 원인 | Fallback 동작 |
|------|-----------|---------------|
| **Step 2** | API 에러, JSON 파싱 실패 | 보수적 평가 (`all_satisfied=False`) → 질문 Skip |
| **Step 3** | API 에러, 생성 실패 | 기본 LOT Scaffolding 생성 → 질문 Skip |
| **Step 5** | 재구성 실패 | 학생 최종 응답(B) 또는 ground_truth 기반 응답(C) 사용 |

#### Skip Metadata 구조

Step별 skip 정보는 `skip_details` 딕셔너리에 통합 저장됩니다:

```python
skip_details = {
    "step2_performance_objectives_evaluation": {
        "is_fallback": False,
        "attempts_needed": 1
    },
    "step3_scaffolding_artifact_generation": {
        "is_fallback": True,
        "failure_reason": "scaffolding_artifact_generation_failed",
        "last_error": [
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

### Checkpoint & Resume

- **증분 체크포인트**: 각 문제 처리 후 `save_incremental_checkpoint()` 호출 → 로그 파일 업데이트
- **파일 기반 Resume**: `{dataset}_train_id-mas_{model}_logs.json`에서 `processed_ids` 추출 → 미처리 문제만 재실행
- **상태 복원**: `load_checkpoint_from_logs()` → `restore_state_from_checkpoint()`
- **조기 종료**: 모든 문제 처리 완료 시 그래프 실행 건너뛰기

### SFT 데이터 출력 형식

```json
[
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
]
```

**저장 경로**: `outputs/{domain}/train/{teacher_short}/{student_short}/{dataset}_train_id-mas_{model}.json`

---

## Phase 3: Instructional Delivery Phase — SFT

### 목적

Phase 2에서 생성된 SFT 데이터로 학생 모델을 파인튜닝하고, 다양한 평가 데이터셋에서 성능을 측정합니다.

### 평가 방법

| 방법 | CLI 옵션 | 모델 소스 | 설명 |
|------|---------|----------|------|
| **Baseline** | `--method baseline` | 베이스 모델 그대로 | 파인튜닝 없는 기본 성능 |
| **SFT** | `--method sft` | HuggingFace Hub SFT 모델 | 일반 SFT 파인튜닝 모델 |
| **SFT_ID-MAS** | `--method sft_id-mas` | HuggingFace Hub ID-MAS 모델 | ID-MAS 방식 SFT 모델 |

SFT 모델명 생성: `config/sft.py` → `get_sft_model_name()`, `get_sft_idmas_model_name()`

예시:
- SFT: `SaFD-00/qwen3-1.7b-math`
- SFT_ID-MAS: `SaFD-00/qwen3-1.7b-math_id-mas`

### Answer Extractor

`utils/answer_extractor.py`에서 5가지 답변 타입을 지원합니다.

| 타입 | 추출기 | 패턴 | 데이터셋 |
|------|--------|------|---------|
| `NUMERIC` | `NumericExtractor` | `#### 25`, `answer is 3.14` | GSM8K, SVAMP, ASDiv |
| `LATEX` | `LaTeXExtractor` | `\boxed{...}` | MATH, MAWPS |
| `MCQ` | `MCQExtractor` | `Answer: A` | ReClor, ARC-C, OpenBookQA |
| `BOOLEAN` | `BooleanExtractor` | `Yes/No`, `True/False` | StrategyQA |
| `TEXT` | `TextExtractor` | 마지막 줄 | BBH |

- **기호적 비교**: LaTeX/Numeric에서 `sympy` 기반 수학적 동치성 검증 (예: `\frac{1}{2}` == `0.5`)
- **Fallback**: `sympy` 미설치 시 문자열 정규화 비교

### 평가 흐름

```
Test Data ({dataset}_test.json)
    ↓
[Model Selection]
  - Baseline: 베이스 모델
  - SFT: HuggingFace SFT 모델
  - SFT_ID-MAS: ID-MAS SFT 모델
    ↓
For each question:
  [Student Response Generation] (1회 시도)
      ↓
  [AnswerExtractor.extract()] → 답변 추출
      ↓
  [AnswerExtractor.compare()] → 정답 비교
      ↓
  점진적 저장 (매 문제마다 결과 파일 업데이트)
    ↓
{dataset}_eval_results-{Method}.json
```

- **Resume 모드**: 기존 결과 파일에서 `processed_question_ids` 추출 → 미평가 문제만 재실행
- **모듈**: `main.py` → `IDMASEvaluator`

### 평가 결과 JSON 형식

```json
{
    "domain": "math",
    "eval_dataset": "gsm8k",
    "method": "Baseline",
    "student_model": "Qwen/Qwen3-1.7B",
    "answer_type": "numeric",
    "total_questions": 1319,
    "evaluated_questions": 1319,
    "correct_count": 456,
    "accuracy": 0.3457,
    "question_results": [
        {
            "question_id": "gsm8k_test_0",
            "question": "문제 텍스트...",
            "ground_truth": "74",
            "predicted_answer": "74",
            "is_correct": true,
            "student_response": "풀이 과정...",
            "choices": null
        }
    ]
}
```

**저장 경로**: `outputs/{domain}/eval/{student_short}/{dataset}_eval_results-{Method}.json`

---

## 시스템 구성 모듈

### Config (`config/`)

| 파일 | 역할 |
|------|------|
| `__init__.py` | 통합 인터페이스 (backward compatibility) |
| `api.py` | API 키 관리 (`OPENAI_API_KEY`, `HF_TOKEN`), `PROJECT_ROOT` |
| `models.py` | Teacher/Student 모델 설정, `create_teacher_config()`, `get_model_short_name()` |
| `domains.py` | 도메인-데이터셋 매핑, `DOMAIN_CONFIG`, `get_domain_data_dirs()`, `get_instructional_goal()` |
| `sft.py` | SFT 모델명 매핑 (`get_sft_model_name()`, `get_sft_idmas_model_name()`) |
| `paths.py` | 경로 헬퍼 (`get_design_output_dir()`) |
| `config.py` | 레거시 호환성 (과도기용) |
| `dataset_config.py` | 데이터셋별 답변 타입 설정 |

**도메인 설정 (`DOMAIN_CONFIG`):**

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

### Models (`models/`)

```
BaseModelWrapper (추상 클래스)
├── TeacherModelWrapper
│   ├── API 모델 (gpt-*)
│   └── 로컬 HuggingFace 모델
└── StudentModelWrapper
    └── 로컬 HuggingFace 모델
```

| 파일 | 역할 |
|------|------|
| `base_wrapper.py` | 추상 기본 클래스 |
| `teacher_wrapper.py` | Teacher 모델 래퍼 (OpenAI API + 로컬) |
| `student_wrapper.py` | Student 모델 래퍼 (로컬) |
| `model_cache.py` | 글로벌 싱글톤 캐시 |
| `local_model_mixin.py` | 로컬 모델 공용 기능 |

**ModelCache:**
- 글로벌 싱글톤으로 `(model_name, device)` 튜플 키로 관리
- Teacher/Student 동일 모델 사용 시 메모리 공유
- 메서드: `get_or_load()`, `is_loaded()`, `clear()`, `memory_usage()`

### Prompts (`prompts/`)

| 파일 | 역할 |
|------|------|
| `design_prompts.py` | Instructional Goal, Instructional Analysis, Performance Objectives 프롬프트 |
| `learning_prompts.py` | Scaffolding, PO Evaluation, Reconstruction 프롬프트 |

**주요 프롬프트 상수 (`learning_prompts.py`):**

| 상수 | 용도 |
|------|------|
| `SCAFFOLDING_SYSTEM_PROMPT` | Student 초기 응답 시스템 프롬프트 |
| `TEACHER_INTERVENTION_PROMPT` | Teacher PO 평가 |
| `SCAFFOLDING_ARTIFACT_PROMPT` | Teacher HOT/LOT Scaffolding 생성 |
| `STUDENT_WITH_ARTIFACT_PROMPT` | Student 재응답 (DB 참조) |
| `SUCCESSFUL_SCAFFOLDING_RECONSTRUCTION_PROMPT` | Teacher Case B 재구성 |
| `TEACHER_FINAL_SOLUTION_PROMPT` | Teacher Case C 최종 풀이 |
| `CONVERSATION_SUMMARIZATION_PROMPT` | Teacher 대화 요약 |

### Utils (`utils/`)

| 파일 | 역할 |
|------|------|
| `base_loader.py` | 데이터 로더 추상 클래스, `AnswerType` enum, `QuestionData` dataclass |
| `domain_loader.py` | 도메인별 JSON 데이터 로더 (`DomainLoader`) |
| `answer_extractor.py` | 5가지 답변 타입 추출기, `get_extractor()` |
| `dataset_preparer.py` | HuggingFace 데이터 다운로드/전처리 |
| `sample_extractor.py` | Instructional Goal용 대표 샘플 추출 (random/diverse/stratified) |
| `dataset_enhancer.py` | 학습 데이터 instruction 강화 |
| `dataset_registry.py` | 데이터셋 메타데이터 레지스트리 |
| `dataset_config.py` | 데이터셋별 설정 |
| `reparse_eval_results.py` | 평가 결과 재분석 유틸리티 |

---

## 데이터 구조

### 디렉토리 구조

```
data/                                          # 원본 데이터
├── math/
│   ├── train/data/                            # 학습용 원본 데이터
│   │   ├── gsm8k_train.json
│   │   ├── gsm8k_samples.json                 # Instructional Goal용 샘플
│   │   ├── gsm8k_train_ID-MAS_*.json          # Enhanced 학습 데이터
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
│   ├── train/
│   │   └── {Teacher-Model}/                   # Teacher 모델별
│   │       ├── instructional-design/
│   │       │   ├── math_gsm8k_design.json
│   │       │   └── math_math_design.json
│   │       │
│   │       └── {Student-Model}/               # Student 모델별
│   │           ├── gsm8k_train_id-mas_{model}.json      # SFT 데이터
│   │           ├── gsm8k_train_id-mas_{model}_logs.json  # 학습 로그
│   │           └── gsm8k_train_summary_{model}.json      # 학습 통계
│   │
│   └── eval/
│       └── {Student-Model}/                   # 평가 결과
│           ├── gsm8k_eval_results-Baseline.json
│           ├── gsm8k_eval_results-SFT.json
│           └── gsm8k_eval_results-SFT_ID-MAS.json
│
├── logical/
│   └── ...
│
└── commonsense/
    └── ...
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
# 결과물은 자동으로 outputs/{domain}/train/{teacher}/{student}/ 에 저장
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
python main.py --mode eval --method baseline \
    --domain your_domain --eval-dataset your_dataset
```

### 주의사항

1. **데이터 형식**: `instruction`, `input`, `output` 필드 필수
2. **Answer Type**: `utils/answer_extractor.py`에서 적절한 추출기 선택
3. **코드 수정 불필요**: 설정 파일만 수정하면 자동으로 동작
