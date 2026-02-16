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
Enhanced Training Data (outputs/{domain}/train/{student}/data/{dataset}_train_ID-MAS.json)
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

행동(Behavior), 조건(Condition), 기준(Criterion)을 통합한 수행목표를 생성합니다. Phase 2에서 학생 응답 평가의 기준이 됩니다.

- **모듈**: `design_modules/objectives.py` → `PerformanceObjectives`
- **출력**:
  ```json
  {
      "performance_objectives": [
          {
              "target": "Problem Interpretation",
              "performance_objective": "Correctly identify key quantities and relationships, given a word problem with numerical data, extracting all relevant information without errors"
          }
      ]
  }
  ```
- **검증**: `validate_objectives()` 함수로 형식 검증
- **재시도**: 최대 3회

### Enhanced Data Generation

원본 학습 데이터에 Instructional Goal + Task Analysis를 metadata로 추가합니다.

- **모듈**: `utils/dataset_enhancer.py` → `DataEnhancer`
- **입력**: `{dataset}_train.json` (원본) + Design Result
- **출력**: `outputs/{domain}/train/{student}/data/{dataset}_train_ID-MAS.json`
- **변환**: `instruction`은 원본 유지, `metadata`에 `instructional_goal`과 `task_analysis` 추가

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
| **Step 3** | Scaffolded Corrective Feedback | Teacher | `teacher_model.py` | HOT/LOT Scaffolding 생성 |
| **Step 4** | Teacher-Supported Reattempt | Student | `student_model.py` | Scaffolding Artifact 참조 개선 응답 |
| **Step 5a-1** | Positive Reinforcement | Teacher | `teacher_model.py` | Case A: Independent Performance Mastery / Case B: Scaffolded & Coached Mastery: PO 충족 시 강점 + 개선점 피드백 |
| **Step 5a-2** | Feedback-Driven Elaboration | Student | `student_model.py` | Case A: Independent Performance Mastery / Case B: Scaffolded & Coached Mastery: Positive Feedback 기반 응답 개선 |
| **Step 5b** | Teacher Modeling | Teacher | `teacher_model.py` | Case C: Teacher Modeling Distillation: 최종 솔루션 생성 |
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
│  │   [Step 2] Teacher PO Evaluation ──→ 성공 ──┐                          │   │
│  │       ↓ (실패)                              │ Case A: Independent Performance Mastery │   │
│  │   [Step 3] Scaffolding Artifact (HOT/LOT)   │             │   │
│  │       ↓                                     │             │   │
│  │   [Step 4] Student Re-response              │             │   │
│  └───────────────────────┬─────────────────────┘             │   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Iteration 2-5 (반복)                                 │   │
│  │   [Step 2] Teacher PO Evaluation ──→ 성공 ──┐                                │   │
│  │       ↓ (실패)                              │ Case B: Scaffolded & Coached Mastery │   │
│  │   [Step 3] Scaffolding Artifact (HOT/LOT)   │        │   │
│  │       ↓                                     │        │   │
│  │   [Step 4] Student Re-response              │        │   │
│  └───────────────────────┬─────────────────────┘        │   │
│                          ↓ (5회 반복 후 실패)               │
│                       Case C: Teacher Modeling Distillation   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Self-Refinement (Case A: Independent Performance Mastery / Case B: Scaffolded & Coached Mastery)  │   │
│  │   [Step 5a-1] Teacher Positive Reinforcement          │   │
│  │       ↓                                              │   │
│  │   [Step 5a-2] Student Feedback-Driven Elaboration    │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ Finalization                                         │   │
│  │   [Step 5b] Teacher Modeling (Case C: Teacher Modeling Distillation만)  │   │
│  │       ↓                                              │   │
│  │   [Step 6] SFT Data Generation                       │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Step 1: Initial Response (Student)

- **함수**: `StudentModel.generate_initial_response_with_scaffolding()`
- **입력**: `problem_text`, `task_analysis`, `instructional_goal`, `dataset_prompt`
- **동작**: Task Analysis를 시스템 프롬프트에 포함하여 체계적 풀이 유도
- **1회차에만 실행** (2회차부터는 Step 4로 대체)

#### Step 2: PO Evaluation (Teacher)

- **함수**: `TeacherModel.evaluate_with_performance_objectives()`
- **입력**: `student_response`, `performance_objectives`, `problem_text`, `ground_truth`
- **동작**: 각 PO에 대해 `is_satisfied` 판단 + 판단 근거 및 개선 방향 reasoning 작성
- **출력**:
  ```json
  {
      "performance_evaluation": [
          {
              "objective_content": "PO 내용",
              "is_satisfied": true,
              "reasoning": "판단 근거 (WHY) + 개선/정교화 방향 (HOW)"
          }
      ]
  }
  ```
- **성공 조건**: `all_satisfied == True` → 반복 종료
- **재시도**: 최대 3회, 실패 시 예외 전파

#### Step 3: Scaffolded Corrective Feedback — HOT/LOT (Teacher)

- **함수**: `TeacherModel.generate_scaffolding_artifact()`
- **조건**: `all_satisfied == False`인 경우에만 호출
- **동작**: 미충족 PO에 대해 인지 수준별 Scaffolding 생성 (구조화된 마크다운 출력 → 정규식 파싱)
- **출력**: 구조화된 마크다운 텍스트를 정규식으로 파싱하여 딕셔너리 생성
  ```
  [Instructional Goal]
  ...
  [Instructional Analysis]
  ...
  [Scaffolding for Task [1] (High Order Skill)]:
  - Target Objective: ...
  - Cognitive Level: Analyze
  - Failure Analysis: ...
  - Suggested Strategy: ...
  - Key Attention Points: ...

  [Scaffolding for Task [2] (Low Order Skill)]:
  - Target Objective: ...
  - Cognitive Level: Understand
  - Failure Analysis: ...
  - Missed Concept/Information: ...
  - Brief Explanation: ...

  [Feedback]
  통합 서술형 피드백 단락

  [Iteration Summary]
  3-5문장 요약
  ```
- **파싱 결과**: `scaffolding_artifacts` (HOT/LOT 리스트), `feedback`, `iteration_summary`, `_raw_text` (전체 텍스트 → 학생 전달용)
- **누적**: `scaffolding_artifacts`에 iteration별로 누적 저장
- **재시도**: 최대 3회, 실패 시 예외 전파

#### Step 4: Teacher-Supported Reattempt (Student)

- **함수**: `StudentModel.respond_to_feedback()`
- **조건**: iteration >= 2
- **입력**: `problem_text`, `scaffolding_text` (전체 Scaffolding Artifact `_raw_text`), `task_analysis`, `instructional_goal`, `dataset_prompt`
- **동작**: `TEACHER_SUPPORTED_REATTEMPT_SYSTEM_PROMPT` / `_USER_PROMPT`으로 system에 지침, user에 Scaffolding Artifact + 문제를 분리하여 개선된 응답 생성
- **DB 참조 추출**: `StudentModel.extract_db_references()` — "Information Retrieved from Scaffolding Artifact:" 섹션 파싱

#### Step 5a-1: Positive Reinforcement (Teacher) — Case A: Independent Performance Mastery / Case B: Scaffolded & Coached Mastery

- **함수**: `TeacherModel.generate_positive_feedback()`
- **조건**: `all_satisfied == True` (Case A: Independent Performance Mastery 또는 Case B: Scaffolded & Coached Mastery)
- **동작**: 모든 PO가 충족된 Student 응답의 강점을 분석하고, 구체적인 개선 제안을 생성
- **출력**: `feedback_text` (강점 + 개선 제안 + 통합 가이드)
- **재시도**: 최대 3회, 실패 시 예외 전파

#### Step 5a-2: Feedback-Driven Elaboration (Student) — Case A: Independent Performance Mastery / Case B: Scaffolded & Coached Mastery

- **함수**: `StudentModel.self_refine_response()`
- **조건**: Step 5a-1에서 positive feedback이 생성된 경우
- **동작**: Teacher의 Positive Feedback을 참조하여 기존 응답의 reasoning을 강화한 개선된 응답 생성
- **입력**: `problem_text`, `positive_feedback`, `task_analysis`, `instructional_goal`
- **출력**: Self-Refined Response → SFT output으로 사용
- **설계 의도**: PO 평가의 강점/개선점을 기존 풀이에 녹여 SFT 데이터 품질을 향상

#### Step 5b: Teacher Modeling (Teacher) — Case C: Teacher Modeling Distillation

- **함수**: `TeacherModel.generate_final_solution()`
- **조건**: `iteration == max_iterations` AND `all_satisfied == False`
- **동작**: 전체 `iteration_summaries`를 기반으로 Step 1과 동일한 형식의 정답 풀이를 평문 텍스트로 생성
- **출력**: `solution_explanation` (평문 텍스트)
- **재시도**: 최대 3회, 실패 시 예외 전파

#### Step 6: SFT Data Generation

- **함수**: `generate_sft_data()` (`nodes.py`)
- **동작**: `scaffolding_results`에서 Case A: Independent Performance Mastery / Case B: Scaffolded & Coached Mastery / Case C: Teacher Modeling Distillation 항목을 SFT 형식으로 변환
- **변환 규칙**:
  - `instruction`: 원본 instruction + `LEARNING_TASK_SYSTEM_PROMPT` 동적 결합 (Task Analysis 포함)
  - `input`: `"Question: {문제 텍스트}"`
  - `output`: Case A: Independent Performance Mastery → Student Self-Refined Response, Case B: Scaffolded & Coached Mastery → Student Self-Refined Response, Case C: Teacher Modeling Distillation → Teacher 최종 풀이
  - `metadata`: `id`, `sft_case`, `ground_truth`

### Case 분류

| Case | 조건 | SFT 응답 소스 | Self-Refinement | Step 5b 필요 |
|------|------|--------------|-----------------|-------------|
| **Case A: Independent Performance Mastery** | 1회차 PO 충족 | Teacher Positive Reinforcement → Student Feedback-Driven Elaboration | Yes (Step 5a-1→5a-2) | No |
| **Case B: Scaffolded & Coached Mastery** | 2~5회차 PO 충족 | Teacher Positive Reinforcement → Student Feedback-Driven Elaboration | Yes (Step 5a-1→5a-2) | No |
| **Case C: Teacher Modeling Distillation** | 5회 후 PO 미충족 | Teacher가 정답 기반 재구성 | No | Yes |

**성공 조건**: 모든 Performance Objectives가 충족되면 (`all_satisfied == True`) 성공으로 처리됩니다.
**Self-Refinement**: Case A: Independent Performance Mastery / Case B: Scaffolded & Coached Mastery에서 PO 충족 후 Teacher의 Positive Feedback(강점 + 개선점)을 기반으로 Student가 응답을 self-refine하여 SFT 데이터 품질을 향상시킵니다.

### HOT/LOT Scaffolding Artifact

Bloom's Taxonomy 기반으로 인지 수준별 차별화된 Scaffolding을 제공합니다.

| 유형 | 대상 인지 수준 | 제공 내용 |
|------|--------------|----------|
| **HOT** (High-Order Thinking) | 분석/평가/창조 | `strategy_suggestion`, `partial_example`, `key_attention_points` |
| **LOT** (Low-Order Thinking) | 기억/이해/적용 | `missed_concept`, `brief_explanation` |

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
            "sft_case": "case_a_independent_performance_mastery",
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
- SFT: `SaFD-00/qwen3-4b-math`
- SFT_ID-MAS: `SaFD-00/qwen3-4b-math_id-mas`

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
    "student_model": "Qwen/Qwen3-4B",
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

**도메인 설정 (`DOMAIN_CONFIG`):**

```python
DOMAIN_CONFIG = {
    "math": {
        "data_dir": DATA_DIR / "math",
        "training_datasets": ["gsm8k", "math"],
        "eval_datasets": ["gsm8k", "math", "svamp", "asdiv", "mawps"],
        "default_eval": "gsm8k"
    },
    "logical": {
        "data_dir": DATA_DIR / "logical",
        "training_datasets": ["reclor"],
        "eval_datasets": ["reclor", "anli_r2", "anli_r3", "bbh"],
        "default_eval": "reclor"
    },
    "commonsense": {
        "data_dir": DATA_DIR / "commonsense",
        "training_datasets": ["arc_c"],
        "eval_datasets": ["arc_c", "strategyqa", "openbookqa"],
        "default_eval": "arc_c"
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
| `remote_model.py` | RemoteLLMProxy + subprocess worker (GPU 격리) |

**ModelCache:**
- 글로벌 싱글톤으로 `(model_name, device, gpu_ids)` 튜플 키로 관리 (`gpu_ids`는 `Optional[Tuple[int, ...]]`)
- Teacher/Student 동일 모델 사용 시 메모리 공유
- `gpu_ids`가 지정되면 `RemoteLLMProxy`를 통해 subprocess에서 모델 로드
- `gpu_ids`가 None이면 기존 인프로세스 vLLM 로드 (CUDA_VISIBLE_DEVICES 기반)
- 메서드: `get_or_load()`, `get_loaded_models()`

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

**RemoteLLMProxy** (`models/remote_model.py`):
- `_remote_model_worker()`: 자식 프로세스 함수. `CUDA_VISIBLE_DEVICES` 설정 후 vLLM `LLM` 로드, `multiprocessing.Pipe`로 chat 요청 수신/응답
- `RemoteLLMProxy`: 메인 프로세스에서 사용하는 프록시. `llm.chat()`과 동일한 인터페이스 제공
- `_RemoteOutput`/`_RemoteCompletionOutput`: vLLM output 호환 래퍼

### Prompts (`prompts/`)

프롬프트 상수(문자열 템플릿)만 정의합니다. 프롬프트 구성 헬퍼 함수는 `utils/prompt_helpers.py`에 위치합니다.

| 파일 | 역할 |
|------|------|
| `design_prompts.py` | Instructional Goal, Instructional Analysis, Performance Objectives 프롬프트 상수 |
| `learning_prompts.py` | Scaffolding, PO Evaluation, Reconstruction 프롬프트 상수 |

**주요 프롬프트 상수 (`learning_prompts.py`):**

| 상수 | 용도 |
|------|------|
| `LEARNING_TASK_SYSTEM_PROMPT` | Student 초기 응답 시스템 프롬프트 (Task Analysis 포함) |
| `FORMATIVE_ASSESSMENT_SYSTEM_PROMPT` / `_USER_PROMPT` | Teacher PO 평가 (system: 역할 정의, user: 평가 데이터) |
| `SCAFFOLDED_CORRECTIVE_FEEDBACK_SYSTEM_PROMPT` / `_USER_PROMPT` | Teacher HOT/LOT Scaffolding 생성 (system: 역할 정의, user: 입력 데이터) |
| `TEACHER_SUPPORTED_REATTEMPT_SYSTEM_PROMPT` / `_USER_PROMPT` | Student 재응답 (system/user 분리, Scaffolding Artifact 참조) |
| `POSITIVE_REINFORCEMENT_SYSTEM_PROMPT` / `_USER_PROMPT` | Teacher PO 충족 시 강점 + 개선점 피드백 생성 (Case A: Independent Performance Mastery / Case B: Scaffolded & Coached Mastery) |
| `FEEDBACK_DRIVEN_ELABORATION_SYSTEM_PROMPT` / `_USER_PROMPT` | Student Positive Feedback 기반 응답 Self-Refinement (system/user 분리, Case A: Independent Performance Mastery / Case B: Scaffolded & Coached Mastery) |
| `TEACHER_MODELING_SYSTEM_PROMPT` / `_USER_PROMPT` | Case C: Teacher Modeling Distillation 최종 풀이 (system: 역할 정의, user: 문제/이력 데이터) |

### Utils (`utils/`)

| 파일 | 역할 |
|------|------|
| `prompt_helpers.py` | 프롬프트 구성 헬퍼 (`format_samples_for_prompt`, `get_instructional_goal_prompt`) |
| `base_loader.py` | 데이터 로더 추상 클래스, `AnswerType` enum, `QuestionData` dataclass |
| `domain_loader.py` | 도메인별 JSON 데이터 로더 (`DomainLoader`) |
| `answer_extractor.py` | 5가지 답변 타입 추출기, `get_extractor()` |
| `dataset_preparer.py` | HuggingFace 데이터 다운로드/전처리 |
| `sample_extractor.py` | Instructional Goal용 대표 샘플 추출 (random/diverse/stratified) |
| `dataset_enhancer.py` | 학습 데이터 instruction 강화 |
| `dataset_registry.py` | 데이터셋 메타데이터 레지스트리 |

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
│   │   ├── {Teacher-Model}/                   # Teacher 모델별
│   │   │   ├── instructional-design/
│   │   │   │   ├── math_gsm8k_design.json
│   │   │   │   └── math_math_design.json
│   │   │   │
│   │   │   └── {Student-Model}/               # Student 모델별
│   │   │       ├── gsm8k_train_id-mas_{model}.json      # SFT 데이터
│   │   │       ├── gsm8k_train_id-mas_{model}_logs.json  # 학습 로그
│   │   │       └── gsm8k_train_summary_{model}.json      # 학습 통계
│   │   │
│   │   └── {Student-Model}/                   # Student 모델별 Enhanced Data
│   │       └── data/
│   │           └── {dataset}_train_ID-MAS.json  # Enhanced 학습 데이터
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
