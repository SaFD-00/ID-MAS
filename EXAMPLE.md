# ID-MAS 동작 예시: Phase 1~3 상세 설명

> GSM8K 데이터셋 + Qwen3-8B 모델 기반 실제 실행 로그를 통해 ID-MAS의 3-Phase 파이프라인이 어떻게 작동하는지 Case별로 설명합니다.

---

## 목차

1. [전체 파이프라인 개요](#1-전체-파이프라인-개요)
2. [Phase 1: Instructional Design](#2-phase-1-instructional-design)
3. [Phase 2: Adaptive Scaffolding](#3-phase-2-adaptive-scaffolding)
   - [Case A 예시](#case-a-1회차-성공)
   - [Case B 예시](#case-b-2회차-이상-성공)
   - [Case C 예시](#case-c-최대-반복-후-실패)
4. [Phase 3: Instructional Delivery (SFT)](#4-phase-3-instructional-delivery)
5. [통계 요약](#5-통계-요약)

---

## 1. 전체 파이프라인 개요

```
Phase 1: Instructional Design (1회 실행, 데이터셋 단위)
  ├── Step 0: Instructional Goal 생성
  ├── Step 1: Learning Objective 설정
  ├── Step 2: Instructional Analysis (Task 분해)
  └── Step 3: Performance Objectives 생성
         ↓
Phase 2: Adaptive Scaffolding (문제별 반복)
  ├── Step 1: Student 초기 응답
  ├── Step 2: Teacher PO 평가 → 성공이면 Case A
  ├── Step 3: Scaffolding Artifact 생성 (HOT/LOT)
  ├── Step 4: Student 재응답 (Scaffolding DB 참조)
  ├── (Step 2~4 반복, 최대 5회)
  ├── Step 5: Reconstruction (Case B/C)
  └── Step 6: SFT 데이터 생성
         ↓
Phase 3: Instructional Delivery
  └── SFT 학습 데이터로 Student 모델 Fine-tuning → 평가
```

---

## 2. Phase 1: Instructional Design

Phase 1은 데이터셋 단위로 **1회만** 실행됩니다. GSM8K 데이터셋에 대해 Teacher 모델(Qwen3-8B)이 교수 설계를 수행합니다.

### Step 0: Instructional Goal 생성

20개의 대표 샘플(`gsm8k_samples.json`)을 분석하여 데이터셋 고유의 학습 목표를 자동 생성합니다.

**생성 결과:**
```json
{
  "instructional_goal": "The model will solve complex mathematical word problems by setting up and solving equations, interpreting relationships, and performing calculations.",
  "cognitive_level": "Apply",
  "primary_verb": "solve"
}
```

### Step 1: Learning Objective 설정

Instructional Goal을 그대로 Learning Objective로 설정합니다.

### Step 2: Instructional Analysis

Learning Objective를 Subskills와 Subtasks의 계층 구조로 분해합니다.

**생성 결과 (Task Analysis Tree):**
```
Terminal Goal: Solve complex mathematical word problems by setting up and solving equations,
              interpreting relationships, and performing calculations. (Apply – Procedural Knowledge)
 ├── [1] Setting up equations from word problems (Apply – Procedural Knowledge)
 │   ├── [1-1] Identifying key information and variables from the problem
 │   │         (Understand – Conceptual Knowledge)
 │   ├── [1-2] Determining the relationship between variables
 │   │         (Analyze – Conceptual Knowledge)
 │   └── [1-3] Formulating equations based on identified relationships
 │             (Apply – Procedural Knowledge)
 ├── [2] Solving equations (Apply – Procedural Knowledge)
 │   ├── [2-1] Applying appropriate algebraic operations to solve for unknowns
 │   │         (Apply – Procedural Knowledge)
 │   └── [2-2] Checking the solution by substituting it back into the original equation
 │             (Evaluate – Procedural Knowledge)
 ├── [3] Interpreting relationships (Understand – Conceptual Knowledge)
 │   ├── [3-1] Understanding the meaning of variables and constants in the context
 │   │         (Understand – Conceptual Knowledge)
 │   └── [3-2] Interpreting the solution in the context of the real-world scenario
 │             (Apply – Procedural Knowledge)
 └── [4] Performing calculations (Apply – Procedural Knowledge)
     ├── [4-1] Carrying out arithmetic operations accurately
     │         (Apply – Procedural Knowledge)
     └── [4-2] Using appropriate tools or methods for complex calculations
               (Apply – Procedural Knowledge)
```

### Step 3: Performance Objectives 생성

ABCD 모델 기반으로 각 Subskill에 대한 측정 가능한 수행목표를 생성합니다. 이 PO들이 **Phase 2에서 학생 응답 평가의 기준**이 됩니다.

**생성 결과 (14개 PO 중 일부):**

| # | Target | Behavior | Condition | Criterion |
|---|--------|----------|-----------|-----------|
| 1 | Terminal Goal | Solve complex mathematical word problems | Given access to basic mathematical functions | 90% accuracy |
| 2 | Subskill 1-1 | Identify key information and variables | Given a word problem | 85% accuracy |
| 3 | Subskill 2-1 | Apply algebraic operations to solve for unknowns | Given equations based on identified relationships | 90% accuracy |
| 4 | Subskill 4-1 | Carry out arithmetic operations accurately | Given the planned solution steps | 95% accuracy |

### Enhanced Data 생성

Phase 1의 결과물(Instructional Goal + Task Analysis)을 원본 학습 데이터의 `instruction` 필드에 주입합니다.

**변환 전 (원본):**
```
"instruction": "Solve this math problem."
```

**변환 후 (Enhanced):**
```
"instruction": "You are a helpful math assistant.
Solve this mathematical problem step by step...

## Learning Objective
The model will solve mathematical word problems step-by-step...

## Problem-Solving Guidelines
### Instructional Analysis Results
Terminal Goal: Solve mathematical word problems... (Apply – Procedural Knowledge)
 ├── [1] Identify key information...
 │   ├── [1-1] Read and comprehend the problem statement...
 │   ├── [1-2] Recognize and extract numerical values...
 ...

## Response Requirements
1. Explicitly connect each step to the relevant sub-skill
2. Verify your intermediate results
3. Present your final answer clearly in the required format"
```

> **핵심**: 학생 모델은 이 Enhanced Instruction을 보고 Task Analysis 구조에 맞춰 체계적으로 풀이를 생성합니다.

---

## 3. Phase 2: Adaptive Scaffolding

Phase 2는 **각 문제별로** 실행됩니다. 교사-학생 반복 상호작용을 통해 SFT 학습 데이터를 생성합니다.

### Case 분류 기준

| Case | 조건 | SFT 응답 소스 |
|------|------|--------------|
| **A** | 1회차에 모든 PO 충족 | Student 원본 응답 그대로 |
| **B** | 2~5회차에 모든 PO 충족 | Teacher가 대화 히스토리 기반 재구성 |
| **C** | 5회 반복 후에도 PO 미충족 | Teacher가 정답 기반 교육적 풀이 생성 |

---

### Case A: 1회차 성공

> **문제 ID**: `gsm8k_train_0`
> **문제**: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
> **정답**: 72

#### Step 1: Student 초기 응답

Student 모델이 Enhanced Instruction(Task Analysis 포함)을 참고하여 체계적 풀이를 생성합니다.

```
#### Step-by-Step Reasoning

1. **Identify Key Information and Relevant Mathematical Concepts**
   - In April, Natalia sold 48 clips.
   - In May, she sold half as many clips as in April.

2. **Formulate a Plan to Solve the Problem**
   - Calculate the number of clips sold in May.
   - Add the number of clips sold in April and May to get the total.

3. **Execute the Solution Plan Accurately**
   - Clips sold in May = 48/2 = 24
   - Total clips sold = 48 + 24 = 72

4. **Communicate the Solution Clearly**
   - The total number of clips Natalia sold is 72.

### Final Answer
\boxed{72}
```

**추출된 답**: 72 (정답과 일치)

#### Step 2: Teacher PO 평가

Teacher가 Performance Objectives 기준으로 학생 응답을 평가합니다.

```json
{
  "overall_assessment": {
    "objectives_met": "12 of 12 objectives satisfied",
    "all_satisfied": true,
    "primary_weakness": null,
    "recommended_focus": null
  }
}
```

**모든 PO 충족** → 반복 종료 → **Case A 확정**

#### Step 5 & 6: SFT 데이터 생성

Case A는 재구성 불필요. 학생 원본 응답을 그대로 SFT 학습 데이터로 사용합니다.

```json
{
  "instruction": "[Enhanced Instruction with Task Analysis]",
  "input": "Question: Natalia sold clips to 48 of her friends in April...",
  "output": "[Student의 원본 응답 그대로]",
  "metadata": {
    "id": "gsm8k_train_0",
    "sft_case": "A",
    "ground_truth": "72"
  }
}
```

**최종 로그 요약:**
- `iterations_needed`: 1
- `scaffolding_db`: null (Scaffolding 불필요)
- `hot_count`: null
- `lot_count`: null

---

### Case B: 2회차 이상 성공

> **문제 ID**: `gsm8k_train_116`
> **문제**: "Valerie needs to put stamps on the envelopes she is about to mail. She has thank you cards for each of her grandmother, uncle and aunt for the birthday presents they sent. She also has to pay the water bill and the electric bill separately. She wants to send three more mail-in rebates than she does bills and she has twice as many job applications as rebates to mail. How many stamps does she need if everything needs 1 stamp except the electric bill, which needs 2?"
> **정답**: 21

#### Iteration 1: Student 초기 응답 → PO 평가 실패

**Step 1 — Student 초기 응답:**

```
1. Thank you cards: 3 (grandmother, uncle, aunt)
2. Bills: 2 (water, electric)
3. Mail-in rebates = bills + 3 = 5
4. Job applications = 2 × rebates = 10

Total items needing 1 stamp: 3 + 2 + 5 + 10 = 20
Stamps for electric bill: 2
Total stamps: 20 + 2 = 22      ← 오답!

\boxed{22}
```

**Step 2 — Teacher PO 평가:**

```json
{
  "overall_assessment": {
    "objectives_met": "3 of 8 objectives satisfied",
    "all_satisfied": false,
    "primary_weakness": "Arithmetic calculations and total stamp count",
    "recommended_focus": "Revisiting the arithmetic calculations"
  }
}
```

**미충족 PO 예시:**

| PO | 미충족 이유 | Socratic 질문 |
|----|-----------|--------------|
| Terminal Goal | 총 스탬프 수 오계산 | "How can you determine the number of job applications?" |
| Algebraic Operations (90%) | 총 스탬프 수 오계산 | "Can you walk me through the calculation of the total stamps?" |
| Arithmetic Accuracy (95%) | 산술 오류 | "Let's go through the arithmetic again." |

**Step 3 — Scaffolding Artifact 생성 (HOT/LOT):**

Teacher가 미충족 PO별로 차별화된 Scaffolding을 생성합니다.

**HOT Artifact (고차 사고):**
```json
{
  "target_objective": "Terminal Goal",
  "skill_type": "HOT",
  "cognitive_level": "Apply",
  "failure_analysis": "The student did not correctly calculate the number of job applications and the total number of stamps needed.",
  "scaffolding_content": {
    "strategy_suggestion": "Break down the problem into smaller parts and set up equations based on the relationships given.",
    "partial_example": "Identify that the number of job applications is twice the number of mail-in rebates. If R is the number of rebates, then job applications = 2R.",
    "socratic_question": "If the number of mail-in rebates is R, how would you express the number of job applications in terms of R?"
  }
}
```

**LOT Artifact (저차 사고):**
```json
{
  "target_objective": "Arithmetic Accuracy (95%)",
  "skill_type": "LOT",
  "cognitive_level": "Apply",
  "failure_analysis": "The student made errors in the arithmetic calculations.",
  "scaffolding_content": {
    "missed_concept": "The student may have missed the fact that the electric bill requires 2 stamps instead of 1.",
    "brief_explanation": "Ensure you account for the different number of stamps required for each item.",
    "key_attention_points": "Be careful with the number of stamps required for each item, especially the electric bill."
  }
}
```

> **HOT vs LOT 차이**: HOT는 분석/평가/창조 수준의 사고를 위한 전략 제안과 Socratic 질문을, LOT는 기억/이해/적용 수준의 누락 개념과 핵심 주의사항을 제공합니다.

#### Iteration 2: Student 재응답 → PO 평가 실패

**Step 4 — Student 재응답 (Scaffolding DB 참조):**

```
### Information Retrieved from Scaffolding DB:
- [1-2] Determining the relationship between variables
- [2-1] Applying appropriate algebraic operations to solve for unknowns
- [4-1] Carrying out arithmetic operations accurately

### Improved Reasoning:
1. Thank you cards: 3
2. Bills: 2 (water, electric)
3. Total items to mail: 5 + 3R
4. Total stamps = 3 + 1 + 2 + R + 2R = 6 + 3R

Since R = 3 (assumption)...     ← R=3으로 잘못 가정!
Total = 6 + 9 = 15

\boxed{15}
```

**Step 2 — Teacher PO 평가:**
```json
{
  "objectives_met": "4 of 9 objectives satisfied",
  "all_satisfied": false,
  "primary_weakness": "Incorrect handling of variable R and final calculation accuracy"
}
```

**Step 3 — 2차 Scaffolding Artifact:**

```json
{
  "skill_type": "HOT",
  "cognitive_level": "Analyze",
  "failure_analysis": "The student assumed R = 3 without justification.",
  "scaffolding_content": {
    "strategy_suggestion": "Identify the relationship between the number of rebates and job applications, and solve for R using the given conditions.",
    "socratic_question": "How can you express the number of job applications in terms of the number of rebates, and what equation can you set up to find R?"
  }
}
```

#### Iteration 3: Student 재응답 → PO 평가 성공

**Step 4 — Student 재응답:**

```
### Information Retrieved from Scaffolding DB:
- [1-2] Determining the relationship between variables
- [2-1] Applying appropriate algebraic operations to solve for unknowns
- [4-1] Carrying out arithmetic operations accurately

### Improved Reasoning:
1. Thank you cards: 3
2. Bills: 2
3. R = 2 + 3 = 5 (bills + 3)     ← 관계식을 올바르게 적용!
4. Job applications = 2R = 10

Total stamps:
- Thank you: 3×1 = 3
- Bills: 2×1 = 2
- Rebates: 5×1 = 5
- Applications: 10×1 = 10
- Electric bill extra: 2×2 = 4

Total = 3 + 2 + 5 + 10 + 4 = 24

\boxed{24}
```

**Step 2 — Teacher PO 평가:**
```json
{
  "objectives_met": "All objectives satisfied",
  "all_satisfied": true
}
```

**모든 PO 충족** → 반복 종료 → **Case B 확정** (3회차 성공)

> **참고**: 학생의 최종 답 24는 실제 정답 21과 다르지만, PO 평가에서 모든 목표가 충족되었다고 판단되어 Case B로 분류되었습니다. 이는 PO 평가가 풀이 과정의 논리적 완성도를 기준으로 하기 때문입니다.

#### Step 5: Reconstruction (Case B)

Teacher가 3회에 걸친 대화 히스토리를 분석하여, Scaffolding 과정에서의 학습 포인트를 통합한 정제된 응답을 생성합니다.

```
sft_response: "A clean, comprehensive solution that incorporates the learning
              from scaffolding...\nThe answer is \boxed{21}"
```

#### 최종 로그 요약

```json
{
  "id": "gsm8k_train_116",
  "sft_case": "B",
  "iterative_scaffolding": {
    "success": true,
    "iterations_needed": 3
  },
  "scaffolding_db": [
    { "iteration": 1, "artifacts": [1 HOT + 3 LOT] },
    { "iteration": 2, "artifacts": [1 HOT + 1 LOT] }
  ],
  "db_references": [
    "[1-2] Determining the relationship between variables",
    "[2-1] Applying appropriate algebraic operations to solve for unknowns",
    "[4-1] Carrying out arithmetic operations accurately"
  ],
  "hot_count": 2,
  "lot_count": 4,
  "skip_details": {
    "step2_performance_objectives_evaluation": { "is_fallback": false },
    "step3_scaffolding_artifact_generation": { "is_fallback": false },
    "step5_case_b_reconstruction": { "is_fallback": false, "case": "B" }
  }
}
```

**Case B 흐름 요약:**
```
Iteration 1: Student(22) → Teacher(3/8 PO) → Scaffolding(1 HOT + 3 LOT)
Iteration 2: Student(15) → Teacher(4/9 PO) → Scaffolding(1 HOT + 1 LOT)
Iteration 3: Student(24) → Teacher(All PO ✓) → Case B 확정
     ↓
Step 5: Teacher가 대화 히스토리 기반 재구성 → SFT 응답 생성
```

---

### Case C: 최대 반복 후 실패

> **문제 ID**: `gsm8k_train_187`
> **문제**: "Madeline has 5 boxes with 24 crayons in each. She noticed that 5/8 of the crayons in the 2 boxes were not yet used. In the 2 other boxes, only 2/3 of the crayons were used while the last box was not entirely used. How many unused crayons did Madeline have?"
> **정답**: 70

#### Iteration 1: Student 초기 응답 → 실패

**Step 1 — Student 초기 응답:**

```
1. 2 boxes: 24×2 = 48 crayons, unused = 48 × 5/8 = 30
2. 2 other boxes: 24×2 = 48 crayons, unused = 48 × 1/3 = 16
3. Last box: not entirely used → x < 24
   Assume x = 8 (since 1/3 × 24 = 8)     ← 근거 없는 가정!
4. Total: 30 + 16 + 8 = 54

\boxed{54}
```

**Step 2 — Teacher PO 평가:**

```json
{
  "objectives_met": "3 of 6 objectives satisfied",
  "all_satisfied": false,
  "primary_weakness": "Assumed x = 8 without justification"
}
```

**미충족 PO:**

| PO | 미충족 이유 | Socratic 질문 |
|----|-----------|--------------|
| Equation Formulation | x = 8을 근거 없이 가정 | "How can you determine the exact number of unused crayons in the last box without making assumptions?" |
| Algebraic Operations | 마지막 상자의 미사용 크레용 계산/검증 없음 | "Can you show your calculations for the last box?" |
| Real-world Interpretation | 총 미사용 크레용 수 해석 오류 | — |

**Step 3 — Scaffolding Artifact 생성**

#### Iteration 2~5: 반복 실패

| Iteration | Student 답 | PO 충족 | 핵심 문제 |
|-----------|-----------|---------|----------|
| 1 | 54 | 3/6 | 마지막 상자에 x=8 근거 없이 가정 |
| 2 | 52 | 4/12 | 분수 1/4를 근거 없이 적용 |
| 3 | — | — | 반복적 계산, 답 추출 불가 |
| 4 | 52 | 3/7 | 여전히 임의 가정 |
| 5 | 52 | 실패 | 최대 반복 도달 |

> **핵심 관찰**: Student가 "마지막 상자가 완전히 사용되지 않았다"는 조건을 "완전히 사용되지 않았다 = 전부 미사용(24개)"으로 해석하지 못하고, 매 반복에서 임의의 분수를 가정하는 패턴에서 벗어나지 못했습니다.

5회 반복 후에도 모든 PO 미충족 → **Case C 확정**

#### Step 5: Final Solution (Case C)

Teacher가 Student의 약점을 분석한 뒤, 정답(70)을 기반으로 교육적 풀이를 생성합니다.

```
[Understanding the Problem]
Madeline has 5 boxes with 24 crayons in each...

[Step-by-Step Solution]
1. 2 boxes (5/8 unused): 48 × 5/8 = 30 unused
2. 2 boxes (2/3 used): 48 × 1/3 = 16 unused
3. Last box (not entirely used): 24 unused     ← 핵심: "not entirely used" = 전부 미사용

Total: 30 + 16 + 24 = 70

[Common Pitfalls Addressed]
- The student assumed x = 8 without providing a clear rationale
- The student did not explain why x = 8 is a reasonable assumption
- The student did not fully account for the unused crayons in the last box

Answer: \boxed{70}
```

#### 최종 로그 요약

```json
{
  "id": "gsm8k_train_187",
  "sft_case": "C",
  "predicted_answer": "52",
  "scaffolding_correct": false,
  "iterative_scaffolding": {
    "success": false,
    "iterations_needed": 5
  }
}
```

**Case C 흐름 요약:**
```
Iteration 1: Student(54) → Teacher(3/6 PO) → Scaffolding
Iteration 2: Student(52) → Teacher(4/12 PO) → Scaffolding
Iteration 3: Student(?) → Teacher(평가 불가) → Scaffolding
Iteration 4: Student(52) → Teacher(3/7 PO) → Scaffolding
Iteration 5: Student(52) → Teacher(실패) → 최대 반복 도달
     ↓
Step 5: Teacher가 정답 기반 교육적 풀이 생성 (약점 보완)
     → "[Common Pitfalls Addressed]" 섹션 포함
```

---

## 4. Phase 3: Instructional Delivery

Phase 2에서 생성된 SFT 데이터로 Student 모델을 Fine-tuning하고, 평가 데이터셋에서 성능을 측정합니다.

### SFT 데이터 형식

```json
{
  "instruction": "[Enhanced Instruction with Task Analysis]",
  "input": "Question: [문제 텍스트]",
  "output": "[Case별 SFT 응답]",
  "metadata": {
    "id": "gsm8k_train_XXX",
    "sft_case": "A|B|C",
    "ground_truth": "[정답]"
  }
}
```

### Case별 SFT 응답 소스

| Case | SFT `output` 소스 | 특징 |
|------|-------------------|------|
| **A** | Student 원본 응답 | 자체 능력으로 정답 도출 |
| **B** | Teacher 재구성 응답 | Scaffolding 학습 과정을 통합한 정제 응답 |
| **C** | Teacher 최종 풀이 | 학생 약점을 보완한 교육적 정답 풀이 |

### 평가 방법

| Method | 설명 |
|--------|------|
| Baseline | 파인튜닝 없는 기본 모델 |
| SFT | 일반 SFT 파인튜닝 |
| SFT_ID-MAS | ID-MAS 방식 SFT (Phase 2 데이터 활용) |

---

## 5. 통계 요약

### GSM8K + Qwen3-8B 실행 결과

| 항목 | 값 |
|------|-----|
| **전체 문제 수** | 7,473 |
| **처리 완료** | 7,473 (100%) |
| **Scaffolding 성공** | 7,374 (98.67%) |

### Case 분포

| Case | 건수 | 비율 | 설명 |
|------|------|------|------|
| **A** | 6,935 | 92.80% | 1회차 성공 |
| **B** | 439 | 5.88% | 2~5회차 성공 |
| **C** | 75 | 1.00% | 최대 반복 후 실패 |
| **Skip** | 24 | 0.32% | API 오류 등으로 건너뜀 |

```
Case A ████████████████████████████████████████████ 92.80%
Case B ███                                          5.88%
Case C █                                            1.00%
Skip   ▏                                            0.32%
```

> **해석**: Qwen3-8B 모델은 GSM8K 문제의 92.8%를 Task Analysis 기반 Enhanced Instruction만으로 1회에 해결했습니다. 5.88%는 Teacher의 Scaffolding을 통해 개선되었고, 1.0%만이 최대 반복 후에도 해결하지 못했습니다.

---

## 부록: 주요 개념 정리

### HOT vs LOT Scaffolding

| 유형 | 대상 인지 수준 | 제공 내용 |
|------|--------------|----------|
| **HOT** (High-Order Thinking) | 분석/평가/창조 | `strategy_suggestion`, `partial_example`, `socratic_question` |
| **LOT** (Low-Order Thinking) | 기억/이해/적용 | `missed_concept`, `brief_explanation`, `key_attention_points` |

### Scaffolding DB

- 각 iteration에서 생성된 Scaffolding Artifact가 **누적** 저장됩니다
- Student는 재응답 시 DB를 참조하여 `"Information Retrieved from Scaffolding DB:"` 섹션에 인용합니다
- 이를 통해 이전 피드백을 반영한 개선된 응답을 생성합니다

### Skip/Fallback 처리

| Step | 실패 원인 | Fallback 동작 |
|------|-----------|---------------|
| Step 2 (PO 평가) | API 에러, JSON 파싱 실패 | 보수적 평가 → Skip |
| Step 3 (Scaffolding) | API 에러, 생성 실패 | 기본 LOT Scaffolding → Skip |
| Step 5 (재구성) | 재구성 실패 | Case B: 학생 최종 응답 / Case C: ground_truth 기반 |
