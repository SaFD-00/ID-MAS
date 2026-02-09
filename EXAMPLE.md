# ID-MAS 동작 예시: Phase 1~3 상세 설명

> GSM8K 데이터셋 + Qwen3-4B 모델 기반 실제 실행 로그를 통해 ID-MAS의 3-Phase 파이프라인이 어떻게 작동하는지 Case별로 설명합니다.
> **Student와 Teacher 모두 Qwen3-4B 모델**을 사용하며, 각 Step에서 LLM에 입력되는 실제 프롬프트를 포함하여 파이프라인 동작을 프롬프트 수준까지 설명합니다.

---

## 목차

1. [전체 파이프라인 개요](#1-전체-파이프라인-개요)
2. [Phase 1: Instructional Design](#2-phase-1-instructional-design)
   - [Step 0: Instructional Goal 생성](#step-0-instructional-goal-생성)
   - [Step 2: Instructional Analysis](#step-2-instructional-analysis)
   - [Step 3: Performance Objectives 생성](#step-3-performance-objectives-생성)
   - [Enhanced Data 생성](#enhanced-data-생성)
3. [Phase 2: Adaptive Scaffolding](#3-phase-2-adaptive-scaffolding)
   - [Case A 예시](#case-a-1회차-성공)
     - [Step 1: Student 초기 응답 프롬프트](#case-a-step-1-student-초기-응답)
     - [Step 2: Teacher PO 평가 프롬프트](#case-a-step-2-teacher-po-평가)
   - [Case B 예시](#case-b-2회차-이상-성공)
     - [Step 3: Scaffolding Artifact 생성 프롬프트](#case-b-step-3-scaffolding-artifact-생성)
     - [Step 4: Student 재응답 프롬프트](#case-b-step-4-student-재응답)
     - [Step 5: Reconstruction 프롬프트](#case-b-step-5-reconstruction)
   - [Case C 예시](#case-c-최대-반복-후-실패)
     - [Step 5: Final Solution 프롬프트](#case-c-step-5-final-solution)
4. [Phase 3: Instructional Delivery (SFT)](#4-phase-3-instructional-delivery)
5. [통계 요약](#5-통계-요약)
6. [부록 A: 주요 개념 정리](#부록-a-주요-개념-정리)
7. [부록 B: 프롬프트 상수 참조 테이블](#부록-b-프롬프트-상수-참조-테이블)

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
  ├── Step 4: Student 재응답 (Scaffolding Artifact 참조)
  ├── (Step 2~4 반복, 최대 5회)
  ├── Step 5: Reconstruction (Case B/C)
  └── Step 6: SFT 데이터 생성
         ↓
Phase 3: Instructional Delivery
  └── SFT 학습 데이터로 Student 모델 Fine-tuning → 평가
```

---

## 2. Phase 1: Instructional Design

Phase 1은 데이터셋 단위로 **1회만** 실행됩니다. GSM8K 데이터셋에 대해 Teacher 모델(Qwen3-4B)이 교수 설계를 수행합니다.

### Step 0: Instructional Goal 생성

20개의 대표 샘플(`gsm8k_samples.json`)을 분석하여 데이터셋 고유의 학습 목표를 자동 생성합니다.

**생성 결과:**
```json
{
  "instructional_goal": "The model will solve multi-step mathematical problems by applying arithmetic operations, ratios, and proportional reasoning to real-world scenarios.",
  "cognitive_level": "Apply",
  "primary_verb": "solve"
}
```

#### 실제 프롬프트

**System Message** (`INSTRUCTIONAL_GOAL_SYSTEM_MESSAGE`):
```
You are an expert in instructional design and educational assessment.
Your role is to analyze learning materials and derive clear, measurable performance objectives.

Principles:
- Objectives must be Specific, Measurable, Achievable, and Relevant
- Focus on observable behaviors that can be assessed
- Consider the cognitive complexity required by the tasks

Respond with valid JSON only.
```

**User Message** (`INSTRUCTIONAL_GOAL_PROMPT`):

> Placeholder: `{sample_count}` → 20, `{train_data}` → `format_samples_for_prompt()` 출력

```
You are given a sample of items representing a specific task domain. These items are used to evaluate the student you are teaching. Your mission is to analyze the entire test set and determine a core instructional requirement that defines the instructional goal.

## Instructions
1. **Analyze the input test items** to identify the ultimate action the model must demonstrate to provide appropriate answers. Focus on observable and transferable results.
2. **Identify the highest cognitive level** required by the specific nature of the given data, based on the framework of Bloom's Taxonomy.
3. **Avoid describing individual test items** or listing sub-skills, learning steps, or evaluation criteria.
4. **Focus exclusively on deriving a single, comprehensive Instructional Goal** that encapsulates the core requirement across the entire set.


## Output Requirements
1. Write only one Instructional Goal statement.
2. Describe what the model does in real or applied contexts.
3. Begin with: 'The model will...'.
4. Use an observable verb that LLMs can do.
5. Reflect the highest cognitive level without explicitly mentioning the theory's name.
6. Clarify available resources, knowledge, and specific skills to achieve instructional goal.
7. Use only one verb.


## Reference Examples
- "The model will generate coherent, step-by-step mathematical reasoning in natural language that leads to a correct numerical answer for grade-school level math problems."
- "The model will evaluate argumentative texts by identifying, integrating, and judging the logical relationships among claims, evidence, assumptions, and conclusions to determine which inference, critique, or completion is logically warranted in applied reasoning contexts."
- "The model should be able to apply common knowledge to solve a variety of problems related to natural phenomena, human behavior, and environmental interactions."


## Input Data
Below are 20 representative samples from the dataset:

### Sample 1
You are a helpful math assistant.
Solve this mathematical problem step by step. Show your reasoning clearly and use proper mathematical notation.

## Response Format
Your final answer MUST be within \boxed{}.
Example: \boxed{42}
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

### Sample 2
You are a helpful math assistant.
Solve this mathematical problem step by step...
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?

[... 18개 추가 샘플 ...]


## Output (JSON)
{
  "pattern_analysis": "Brief summary of common patterns found in samples",
  "cognitive_demands": ["list", "of", "required", "cognitive", "processes"],
  "instructional_goal": "The model will ...",
  "cognitive_level": "Remember|Understand|Apply|Analyze|Evaluate|Create",
  "primary_verb": "the main action verb used",
  "rationale": "Why this goal was chosen based on the analysis"
}
```

> **`{train_data}` 치환 예시**: `format_samples_for_prompt()` 함수는 각 샘플의 `instruction`(최대 200자)과 `input`(최대 500자)만 추출하여 `### Sample N` 형식으로 포맷합니다. `output` 필드는 학습목표 도출 편향을 방지하기 위해 의도적으로 제외됩니다.

### Step 1: Learning Objective 설정

Instructional Goal을 그대로 Learning Objective로 설정합니다.

### Step 2: Instructional Analysis

Learning Objective를 Subskills와 Subtasks의 계층 구조로 분해합니다.

> **`<think>` 블록 참고**: Qwen3-4B는 응답 시 `<think>...</think>` 형태의 thinking trace를 먼저 생성합니다. 이 thinking trace는 모델의 내부 추론 과정이며, Task Analysis의 `raw_output`에 포함됩니다. Enhanced Instruction 생성 시 이 `<think>` 블록이 함께 instruction에 포함되어 학생 모델에 전달됩니다.

**생성 결과 (Task Analysis Tree):**
```
Instructional Goal: Solve multi-step mathematical problems by applying arithmetic operations,
              ratios, and proportional reasoning to real-world scenarios. (Apply)
 ├── Apply arithmetic operations to multi-step problems (Apply)
 │   ├── Select appropriate arithmetic operations (Understand)
 │   ├── Perform calculations using selected operations (Apply)
 ├── Set up and simplify ratios (Apply)
 │   ├── Identify ratios in real-world scenarios (Understand)
 │   ├── Simplify ratios to their lowest terms (Apply)
 ├── Solve proportional reasoning problems (Apply)
 │   ├── Establish proportional relationships (Understand)
 │   ├── Solve proportions using cross-multiplication (Apply)
 │   ├── Evaluate the reasonableness of solutions (Evaluate)
```

#### 실제 프롬프트

**User Message** (`INSTRUCTIONAL_ANALYSIS_PROMPT`, system=None):

> Placeholder: `{learning_objective}` → 실제 GSM8K Instructional Goal

```
You are an instructional design expert. Perform the Instructional Analysis step of the Dick & Carey model for the learning objective provided below.

[Learning objective]: The model will solve multi-step mathematical problems by applying arithmetic operations, ratios, and proportional reasoning to real-world scenarios.

[Instructions]
Perform the Instructional Analysis and construct a hierarchical structure in the form of: Instructional Goal → Subskills → Subtasks.
Present the instructional analysis results as a text-based tree structure.
Write all skill statements concisely using an action verb + object format.
Include only the minimum number of Subskills and Subtasks that are essential to achieving the Instructional Goal. For every function or sub-function, indicate the learning outcome based on Bloom's revised Taxonomy: Remember / Understand / Apply / Analyze / Evaluate / Create.
The final output must follow the structure and labels in the Output Format below. Do not change the wording, ordering, line breaks, or section titles. The Output Format example is provided ONLY to specify formatting and structure. Determine all subskills and subtasks strictly based on the given Learning Goal.

[Output Format]
### Instructional Analysis Results
Instructional Goal: [Learning objective statement] (learning outcome)
 ├── [Subskill statements] (learning outcome)
 │   ├── [Subtask statements, if needed] (learning outcome)

[Output Format Description]
- Use consistent numbering (e.g., [1], [1-1])
- Use tree characters (├──, │, └──) where applicable
```

### Step 3: Performance Objectives 생성

ABCD 모델 기반으로 각 Subskill에 대한 측정 가능한 수행목표를 생성합니다. 이 PO들이 **Phase 2에서 학생 응답 평가의 기준**이 됩니다.

**생성 결과 (11개 PO 중 일부):**

| # | Target | Behavior | Condition | Criterion |
|---|--------|----------|-----------|-----------|
| 1 | Instructional Goal | Solve multi-step mathematical problems by applying arithmetic operations, ratios, and proportional reasoning | Given a multi-step problem involving arithmetic operations, ratios, and proportional reasoning | Accurately execute all steps and arrive at a mathematically valid solution |
| 2 | Subskill [1] | Select appropriate arithmetic operations to solve multi-step problems | Given a multi-step problem requiring arithmetic operations | Accurately choose and apply the correct arithmetic operations in sequence |
| 3 | Subskill [2] | Set up and simplify ratios in real-world scenarios | Given a problem involving ratios | Correctly establish and simplify ratios to their lowest terms |
| 4 | Subskill [3] | Solve proportional reasoning problems using cross-multiplication | Given a proportional relationship | Correctly apply cross-multiplication and verify the solution's validity |

#### 실제 프롬프트

**User Message** (`PERFORMANCE_OBJECTIVES_PROMPT`, system=None):

> Placeholder: `{instructional_analysis}` → 실제 GSM8K Task Analysis Tree

```
You are an instructional designer specializing in the Dick and Carey instructional design model, and a researcher in LLM learning methodologies.
Based on the provided Instructional Goal and Instructional Analysis Result, generate a set of Performance Objectives that will serve as the criteria for evaluating the observable performance within the LLM's reasoning process.

Performance objectives should be written using the guidelines provided in Anderson & Krathwohl's Taxonomy for Learning.
Specifically, they should be created using information from the learning outcomes identified in the Instructional Analysis Results.

[Input Data]
Instructional Analysis Result: ### Instructional Analysis Results
Instructional Goal: Solve multi-step mathematical problems by applying arithmetic operations,
              ratios, and proportional reasoning to real-world scenarios. (Apply)
 ├── Apply arithmetic operations to multi-step problems (Apply)
 │   ├── Select appropriate arithmetic operations (Understand)
 │   ├── Perform calculations using selected operations (Apply)
 ├── Set up and simplify ratios (Apply)
 │   ├── Identify ratios in real-world scenarios (Understand)
 │   ├── Simplify ratios to their lowest terms (Apply)
 ├── Solve proportional reasoning problems (Apply)
 │   ├── Establish proportional relationships (Understand)
 │   ├── Solve proportions using cross-multiplication (Apply)
 │   ├── Evaluate the reasonableness of solutions (Evaluate)

[Instructions]
- For each Subskill and Subtask in the instructional analysis, you must create at least one Performance Objective.
- You may create more than one PO per Subskill/Subtask if it involves multiple evaluable aspects. However, each PO must evaluate exactly ONE specific aspect of performance. Do NOT combine multiple evaluation criteria into a single PO.
- Each PO must directly map to a specific Subskill or Subtask from the analysis. Include the "target" field to clearly indicate which item it corresponds to (e.g., "Subskill [1]", "Subtask [1-1]").
- Every Performance Objective must include all three components—Behavior, Condition, and Criterion—and each component must be explicitly stated.
- Behavior: This is a description of LLM's intellectual skill including actions, content, and concepts. Write each PO Behavior as a complete, self-contained statement that can be directly used as objective_content in evaluation. The Behavior must be specific enough to be independently evaluable without requiring additional context.
- Condition: This is a description of the tools and resources that will be available to the learner when performing the skill. Write the conditions based solely on the data given in the problem or generated during the reasoning process. It should ALWAYS begin with 'Given ~'.
- Criterion: This is a description of acceptable performance of the skill. The Criterion component must be tailored to the nature of the task: for tasks with correct answers, it must include a clear and measurable standard such as accuracy requirements, acceptable error ranges, or the number of correct responses; whereas for tasks with no single correct answer, it must specify the information or features that must be present for an acceptable response.
- Furthermore, these criteria must be formulated to evaluate the observable reasoning process within a single problem-solving task.
- You must not add content that does not appear in the Instructional Analysis Result.

IMPORTANT: The Behavior field of each PO will be used verbatim as the evaluation criterion (objective_content) during the teacher evaluation step. Therefore:
1. Write Behavior as a clear, complete sentence describing a single observable action.
2. Maintain consistent writing style across all POs (use the same sentence structure pattern).
3. Each Behavior must be self-explanatory without needing to read the Condition or Criterion.

[Anderson & Krathwohl's Taxonomy Reference]
[... Taxonomy 참조 내용 ...]

[Output Format]
Your output must be formatted as JSON, following this structure and no other form of explanation or commentary:
{
  "performance_objectives": [
    {
      "target": "Instructional Goal",
      "Behavior": "...",
      "Condition": "...",
      "Criterion": "..."
    },
    {
      "target": "Subskill [X]",
      "Behavior": "...",
      "Condition": "...",
      "Criterion": "..."
    },
    {
      "target": "Subtask [X-Y]",
      "Behavior": "...",
      "Condition": "...",
      "Criterion": "..."
    }
  ]
}

Output ONLY the JSON object above. Do not include any additional text, explanation, or commentary outside the JSON structure.
```

### Enhanced Data 생성

Phase 1의 결과물(Instructional Goal + Task Analysis)을 원본 학습 데이터의 `instruction` 필드에 주입합니다.

#### 실제 템플릿

**`ENHANCED_INSTRUCTION_TEMPLATE`** (LLM 호출 없음, 단순 문자열 치환):

> Placeholder: `{original_instruction}`, `{instructional_goal}`, `{task_analysis}`

```
{original_instruction}

## Learning Objective
Your response should demonstrate: {instructional_goal}

## Problem-Solving Guidelines
Follow the structured approach below to ensure a complete and well-reasoned solution:

{task_analysis}

## Response Requirements
1. Explicitly connect each step to the relevant sub-skill or knowledge from the guidelines above
2. Verify your intermediate results before proceeding to the next step
3. Present your final answer clearly in the required format
```

#### 치환 결과 예시

**변환 전 (원본):**
```
"instruction": "You are a helpful math assistant.\nSolve this mathematical problem step by step. Show your reasoning clearly and use proper mathematical notation.\n\n## Response Format\nYour final answer MUST be within \\boxed{}.\nExample: \\boxed{42}"
```

**변환 후 (Enhanced):**
```
You are a helpful math assistant.
Solve this mathematical problem step by step. Show your reasoning clearly and use proper mathematical notation.

## Response Format
Your final answer MUST be within \boxed{}.
Example: \boxed{42}

## Learning Objective
Your response should demonstrate: The model will solve multi-step mathematical problems by applying arithmetic operations, ratios, and proportional reasoning to real-world scenarios.

## Problem-Solving Guidelines
Follow the structured approach below to ensure a complete and well-reasoned solution:

<think>
[Qwen3-4B의 thinking trace — Task Analysis 생성 과정의 내부 추론]
</think>

### Instructional Analysis Results
Instructional Goal: Solve multi-step mathematical problems by applying arithmetic operations,
              ratios, and proportional reasoning to real-world scenarios. (Apply)
 ├── Apply arithmetic operations to multi-step problems (Apply)
 │   ├── Select appropriate arithmetic operations (Understand)
 │   ├── Perform calculations using selected operations (Apply)
 ├── Set up and simplify ratios (Apply)
 │   ├── Identify ratios in real-world scenarios (Understand)
 │   ├── Simplify ratios to their lowest terms (Apply)
 ├── Solve proportional reasoning problems (Apply)
 │   ├── Establish proportional relationships (Understand)
 │   ├── Solve proportions using cross-multiplication (Apply)
 │   ├── Evaluate the reasonableness of solutions (Evaluate)

## Response Requirements
1. Explicitly connect each step to the relevant sub-skill or knowledge from the guidelines above
2. Verify your intermediate results before proceeding to the next step
3. Present your final answer clearly in the required format
```

> **핵심**: Qwen3-4B의 Task Analysis `raw_output`에는 `<think>` 블록이 포함되어 있으며, 이것이 Enhanced Instruction에 그대로 포함됩니다. 학생 모델은 이 Enhanced Instruction을 보고 Task Analysis 구조에 맞춰 체계적으로 풀이를 생성합니다.

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

#### Case A Step 1: Student 초기 응답

Student 모델이 Enhanced Instruction(Task Analysis 포함)을 참고하여 체계적 풀이를 생성합니다.

##### 실제 프롬프트

**System Message** (`SCAFFOLDING_SYSTEM_PROMPT`):

> Placeholder: `{instructional_goal}` → GSM8K Instructional Goal, `{task_analysis}` → Task Analysis Tree

```
The purpose of your response is to demonstrate the attainment of the Instructional Goal: The model will solve multi-step mathematical problems by applying arithmetic operations, ratios, and proportional reasoning to real-world scenarios.

You must adhere to the specific performance procedures and required knowledge/skills outlined in the Instructional Analysis results below. Ensure that your solution describes the full reasoning process using all provided steps and resources before arriving at the final answer.

[Instructional Analysis (Learning Structure)]
### Instructional Analysis Results
Instructional Goal: Solve multi-step mathematical problems by applying arithmetic operations,
              ratios, and proportional reasoning to real-world scenarios. (Apply)
 ├── Apply arithmetic operations to multi-step problems (Apply)
 │   ├── Select appropriate arithmetic operations (Understand)
 │   ├── Perform calculations using selected operations (Apply)
 ├── Set up and simplify ratios (Apply)
 │   ├── Identify ratios in real-world scenarios (Understand)
 │   ├── Simplify ratios to their lowest terms (Apply)
 ├── Solve proportional reasoning problems (Apply)
 │   ├── Establish proportional relationships (Understand)
 │   ├── Solve proportions using cross-multiplication (Apply)
 │   ├── Evaluate the reasonableness of solutions (Evaluate)

[Instructions]
1. Identify which skills and sub-skills from the instructional analysis are relevant to this problem
2. Plan your problem-solving strategy based on the instructional goal and subskills
3. Execute each step systematically, demonstrating the required performance behaviors
4. Ensure your solution describes the full reasoning process using all provided steps and resources
5. Provide your final answer clearly

[Output Format]
Problem-solving strategy and flow:
- Instructional goal alignment: [how this solution demonstrates the instructional goal]
- Relevant skills applied: [list the relevant skills from instructional analysis]
- Step-by-step reasoning: [your detailed solution following the instructional structure]

Answer: [your final answer]
```

**User Message** = `problem_text` (문제 원문):
```
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
```

##### Student 응답 결과

```
<think>
Okay, let's solve this problem step by step. First, I need to figure out how many clips Natalia sold in May. The problem says she sold half as many in May as in April. Since she sold 48 clips in April, I should calculate half of 48 for May.

Half of 48 is 24, so she sold 24 clips in May. Now, to find the total number of clips sold over both months, I need to add the clips from April and May together. That would be 48 (April) plus 24 (May).

Adding those together gives 72. Let me check if I applied the arithmetic operations correctly. Yes, I used division to find half of 48 and then addition to sum both amounts. This aligns with the sub-skill of applying arithmetic operations.

I also need to make sure I set up the ratios correctly. The problem mentions selling half as many in May, which is a ratio of 1:2 (May:April). Simplifying that ratio confirms it's 1/2, so my calculation for May's clips is correct.

Finally, evaluating the reasonableness of the solution: selling 48 in April and 24 in May totals 72, which makes sense as half of April's sales are in May. The answer seems logical.

\boxed{72}
```

**추출된 답**: 72 (정답과 일치)

#### Case A Step 2: Teacher PO 평가

Teacher가 Performance Objectives 기준으로 학생 응답을 평가합니다.

##### 실제 프롬프트

**User Message** (`TEACHER_INTERVENTION_PROMPT`):

> Placeholder: `{problem_text}` → 문제 원문, `{student_response}` → 위 Student 응답, `{performance_objectives}` → Phase 1에서 생성된 PO JSON, `{ground_truth}` → "72"

```
You are a teacher supporting the learning of a student.

Your role is NOT to provide correct answers, but to generate a reasoning state that guides the student's next response. You must monitor the student's reasoning steps to ensure they meet the established performance objectives.

In cases of non-compliance or error, you must generate tailored, specific feedback to guide the student toward the desired outcome. Your feedback functions as an intermediate thought in a ReAct-style learning loop and must guide the student's next reasoning action.

[Input Data]
- Problem: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
- Student response: <think>
  Okay, let's solve this problem step by step...
  [... 전체 응답 ...]
- Performance objectives: [11개 PO JSON 전체]
- Ground truth (FOR REFERENCE ONLY - DO NOT REVEAL): 72

[Instructions]
1. Assess student performance according to each performance objective.
2. Use the Criterion defined in each performance objective as the evaluation standard.
3. DO NOT reveal correct answers or model solutions.
4. Analyze the student response and determine which performance objectives are satisfied and which are not.
5. All judgments must be grounded in observable reasoning behaviors in the student response, such as how claims are justified, how relationships are analyzed, or how judgments are formed.
6. Avoid vague or abstract evaluations.

For UNSATISFIED performance objectives, provide structured feedback with ALL four components:
  (a) Error Analysis: Identify EXACTLY what area the student got wrong and WHY, referencing specific parts of their actual response.
  (b) Improvement Direction: Suggest a CONCRETE direction and strategy for how to correct and improve.
  (c) Response Comment: Provide a specific comment on the student's previous response process.
  (d) Metacognitive Prompt: Ask a question that prompts self-reflection.

HOT/LOT Differentiation for Feedback Depth:
- For HOT (High-Order Thinking: Analyze, Evaluate, Create) objectives: Provide MORE detailed feedback.
- For LOT (Low-Order Thinking: Remember, Understand, Apply) objectives: Provide CONCISE feedback.

For SATISFIED performance objectives:
- Provide a brief positive comment acknowledging what the student did well.

IMPORTANT: When describing student errors or suggesting improvements, use SPECIFIC and CONCRETE vocabulary.
- BAD: "Your approach needs improvement" / "Think more carefully"
- GOOD: "You failed to isolate the variable x by dividing both sides by 3" / "Apply the distributive property to expand (a+b)^2"

CRITICAL: The "objective_content" field MUST contain the EXACT text from the input performance objectives.
Do NOT generate new descriptions. Copy the Behavior text from the provided Performance Objectives word-for-word.

[Output Format - JSON]
{
  "performance_evaluation": [
    {
      "objective_content": "MUST be the EXACT text from the performance objectives. Copy the Behavior field verbatim. Do NOT paraphrase, summarize, or rewrite.",
      "is_satisfied": true or false,
      "reason_for_unmet_objective": "Detailed description of the cause if false; null if true",
      "feedback": {
        "error_analysis": "What specific area the student got wrong and why, referencing their actual response (if false; null if true)",
        "improvement_direction": "Concrete direction and strategy for how to correct and improve (if false; null if true)",
        "response_comment": "Specific comment on the student's previous response process (if false; positive comment if true)",
        "metacognitive_prompt": "Question to prompt self-reflection, e.g., 'Did you consider using X? Think about why it is needed.' (if false; null if true)"
      }
    }
  ],
  "overall_assessment": {
    "objectives_met": "X of Y objectives satisfied",
    "all_satisfied": true or false,
    "primary_weakness": "Main area needing improvement if any; null if all satisfied",
    "recommended_focus": "What the student should focus on next if not all satisfied; null if complete"
  }
}

Output ONLY the JSON object above.
```

##### Teacher 평가 결과

```json
{
  "overall_assessment": {
    "objectives_met": "11 of 11 objectives satisfied",
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
  "output": "[Student의 원본 응답 그대로 — <think> 블록 포함]",
  "metadata": {
    "id": "gsm8k_train_0",
    "sft_case": "A",
    "ground_truth": "72"
  }
}
```

> **참고**: `sft_response`에는 Student의 `<think>` 블록이 포함됩니다. Qwen3-4B는 thinking trace를 자동 생성하므로, SFT 데이터의 output에도 이 패턴이 반영됩니다.

**최종 로그 요약:**
- `iterations_needed`: 1
- `scaffolding_artifacts`: null (Scaffolding 불필요)
- `hot_count`: null
- `lot_count`: null
- `_failure_metadata`: `{"step2_performance_objectives_evaluation": {"is_fallback": false, "attempts_needed": 1}}`

---

### Case B: 2회차 이상 성공

> **문제 ID**: `gsm8k_train_3`
> **문제**: "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?"
> **정답**: 42

#### Iteration 1: Student 초기 응답 → PO 평가 실패

**Step 1 — Student 초기 응답:**

(System Message = `SCAFFOLDING_SYSTEM_PROMPT`, User Message = 위 문제 원문 — Case A Step 1과 동일한 프롬프트 구조)

```
<think>
Okay, let's tackle this problem step by step. First, I need to understand what's being asked. Julie has a 120-page book. She read 12 pages yesterday. Today, she read twice as many as yesterday. So today she read 24 pages. Now, I need to figure out how many pages she has left and then determine how many she should read tomorrow if she wants to read half of the remaining pages.

First, let's calculate the total pages she's read so far. Yesterday: 12 pages. Today: 2 times 12, which is 24. So total read is 12 + 24 = 36 pages.

The book has 120 pages total, so the remaining pages would be 120 - 36. Let me do that subtraction: 120 minus 36 is 84 pages left.

Now, the problem says she wants to read half of the remaining pages tomorrow. So half of 84 is 42. Therefore, she should read 42 pages tomorrow.
</think>

### Step-by-Step Reasoning
1. Pages read yesterday: 12
2. Pages read today: 2 × 12 = 24
3. Total read: 12 + 24 = 36
4. Remaining: 120 - 36 = 84
5. Half of remaining: 84 / 2 = 42

\boxed{42}
```

**추출된 답**: 42 (정답과 일치)

**Step 2 — Teacher PO 평가:**

```json
{
  "overall_assessment": {
    "objectives_met": "7 of 8 objectives satisfied",
    "all_satisfied": false,
    "primary_weakness": "Subskill [2]: Set up and simplify ratios in real-world scenarios",
    "recommended_focus": "Practice problems that explicitly require ratio setup and simplification."
  }
}
```

**미충족 PO:**

| PO | 미충족 이유 | 피드백 |
|----|-----------|--------|
| Set up and simplify ratios | 문제에 ratio가 필요하지 않았지만, PO는 ratio 적용을 기대 | "Did you consider how ratios might be used to describe the relationship between pages read and total pages?" |

> **핵심 관찰**: Student는 정답(42)을 도출했으나, ratio 관련 PO를 시연하지 않아 미충족 판정을 받았습니다.

#### Case B Step 3: Scaffolding Artifact 생성

Teacher가 미충족 PO별로 차별화된 Scaffolding을 생성합니다.

##### 실제 프롬프트

**User Message** (`SCAFFOLDING_ARTIFACT_PROMPT`):

> Placeholder: `{problem_text}` → 문제 원문, `{student_response}` → Student의 응답, `{po_evaluation}` → Teacher PO 평가 JSON, `{failed_objectives}` → 미충족 PO 목록, `{task_analysis}` → Task Analysis Tree, `{iteration_number}` → 1, `{max_iterations}` → 5

```
You are an instructional design expert (Dick & Carey model) creating a Scaffolding Artifact to help a student improve.

Your role is to design pedagogical scaffolding for Performance Objectives that the student failed to meet. This scaffolding will be stored as a "Scaffolding Artifact" that the student can reference in their next attempt.

[Input Data]
- Problem: Julie is reading a 120-page book...
- Student's Response: [Student 응답 전체]
- Performance Objectives Evaluation: [Teacher PO 평가 JSON 전체]
- Failed Performance Objectives: [미충족 PO 목록]
- Instructional Analysis: [Task Analysis Tree 전체]
- Iteration Number: 1 of 5

[Instructions]
1. **Select scaffolding targets**: Focus on Performance Objectives with high failure rates that are critical for achieving the Instructional Goal.

2. **Classify skill level**: For each unmet PO, determine if it requires:
   - **HOT (High-Order Thinking)**: Analyze, Evaluate, Create
   - **LOT (Low-Order Thinking)**: Remember, Understand, Apply

3. **Design appropriate scaffolding**:

   For **HOT skills**:
   - Strategy suggestion: Propose an approach or reasoning strategy
   - Partial worked example: Show partial reasoning (stop before the final answer)
   - Feedback question: Guide thinking without revealing the answer
   - Key attention points: What the student should focus on

   For **LOT skills**:
   - Missed concept/information: Explicitly state what the student missed
   - Brief explanation: Provide a concise explanation to minimize cognitive load

4. **Do NOT reveal correct answers** - guide reasoning, don't solve.

[Output Format - JSON]
{
  "scaffolding_artifacts": [
    {
      "target_objective": "The specific unmet Performance Objective",
      "skill_type": "HOT" or "LOT",
      "cognitive_level": "...",
      "failure_analysis": "Why the student failed this objective",
      "scaffolding_content": {
        "strategy_suggestion": "Suggested approach (for HOT) or null",
        "partial_example": "Partial worked example showing key reasoning (for HOT) or null",
        "feedback": "Guiding feedback (for HOT) or null",
        "missed_concept": "Concept the student missed (for LOT) or null",
        "brief_explanation": "Concise explanation (for LOT) or null",
        "key_attention_points": "What to focus on in next attempt"
      }
    }
  ],
  "scaffolding_summary": "A 3-5 sentence summary synthesizing the key guidance for the student's next attempt. This should be actionable and reference the specific strategies or concepts without revealing answers."
}

CRITICAL INSTRUCTIONS:
1. Your response MUST be ONLY valid JSON - no additional text
2. Do NOT reveal correct answers or complete solutions
3. Focus on guiding the reasoning process
4. The scaffolding_summary should be directly usable by the student

Output ONLY the JSON object above.
```

##### Scaffolding 결과

**HOT Artifact (고차 사고):**
```json
{
  "target_objective": "Set up and simplify ratios in real-world scenarios",
  "skill_type": "HOT",
  "cognitive_level": "Analyze",
  "failure_analysis": "The problem involved proportional reasoning (halving remaining pages) but not ratios, so the subskill of setting up ratios was not directly applicable.",
  "scaffolding_content": {
    "strategy_suggestion": "Ask yourself: 'Are there two quantities being compared that could form a ratio?' Look for phrases like 'ratio of...', 'proportional to', or 'in the same proportion'.",
    "partial_example": "If 12 pages were read out of 120, the ratio of pages read to total pages is 12:120. Simplify this ratio by dividing both numbers by 12: 1:10.",
    "feedback": "Check if the problem contains a comparison between two quantities. If so, try expressing that relationship as a ratio before solving."
  }
}
```

#### Case B Step 4: Student 재응답

Student가 Teacher의 피드백과 Scaffolding Artifact를 참조하여 개선된 응답을 생성합니다.

##### 실제 프롬프트

**User Message** (`STUDENT_WITH_ARTIFACT_PROMPT`):

> Placeholder: `{problem_text}` → 문제 원문, `{teacher_feedback}` → Teacher 평가 결과, `{scaffolding_summary}` → Scaffolding 요약, `{scaffolding_artifacts}` → Scaffolding Artifact JSON, `{task_analysis}` → Task Analysis Tree

```
You are a student learning to solve problems with scaffolding support.

Your teacher has evaluated your previous attempt and provided feedback and scaffolding guidance to help you improve. You must carefully use this information to generate a better solution.

[Problem]
Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?

[Teacher's Feedback on Your Previous Response]
[Teacher 평가 JSON — 미충족 PO에 대한 구조화된 피드백 포함]

[Scaffolding Artifact]
The following scaffolding information has been prepared to help you:
To improve ratio application, focus on identifying comparisons between quantities in the problem. If a ratio exists, express it in simplest form before solving...

[Detailed Scaffolding Artifacts]
[HOT Artifact JSON 전체]

[Instructional Analysis (Learning Structure)]
[Task Analysis Tree 전체]

[Instructions]
1. Carefully read your teacher's feedback to understand what you got wrong and why
2. Review the Scaffolding Artifact and identify which guidance applies to your mistakes
3. Apply the strategies, concepts, or explanations from the scaffolding to improve your solution
4. Show your improved reasoning step by step
5. Provide your final answer clearly

[Output Format]
Improved Reasoning:
- Applying scaffolding guidance: [explain how you are using the scaffolding]
- Step-by-step solution: [your detailed improved solution]

Answer: [your final answer]
```

##### Iteration 2~4 Student 응답 요약

| Iteration | Student 답 | PO 충족 | 핵심 변화 |
|-----------|-----------|---------|----------|
| 1 | 42 ✓ | 7/8 | 정답 도출했으나 ratio 미시연 |
| 2 | 42 ✓ | 3/4 | ratio 84:120→7:10으로 표현 시도, 일부 PO 여전히 미충족 |
| 3 | 42 ✓ | 3/4 | ratio 및 proportion 단계를 추가했으나 cross-multiplication 미충족 |
| 4 | 42 ✓ | 7/7 ✓ | 모든 PO 충족 — ratio 84:120→7:10 + proportion 설정 완료 |

**Iteration 4 — Student 최종 응답:**

```
<think>
... The correct approach is to first simplify the ratio of remaining to total pages (84:120 → 7:10). Then, since she wants to read half of the remaining pages, that's 84 ÷ 2 = 42. But the teacher wanted the ratio step and proportion. So maybe I should show that 7/10 of the total pages is 84, so half of that would be 7/10 * 1/2 = 7/20 of the total. Then 7/20 of 120 is 42.
</think>

Improved Reasoning:
To solve this problem, I first calculate the remaining pages Julie needs to read:
120 total pages - (12 + 24) pages read = **84 pages remaining**.

Next, I define the ratio of remaining pages to total pages:
84:120 → Simplify by dividing both numbers by 12: **7:10**.

Using proportional reasoning, I set up a proportion to find half of the remaining pages:
84/120 = x/120 (where x is half of 84)
Cross-multiplying: 84 × 120 = 120 × x → x = 84
However, since the question asks for **half of the remaining pages**, I calculate:
84/2 = **42**

Answer: \boxed{42}
```

**Step 2 — Teacher PO 평가 (Iteration 4):**
```json
{
  "objectives_met": "7 of 7 objectives satisfied",
  "all_satisfied": true
}
```

**모든 PO 충족** → 반복 종료 → **Case B 확정** (4회차 성공)

#### Case B Step 5: Reconstruction

Teacher가 4회에 걸친 대화 히스토리를 분석하여, Scaffolding 과정에서의 학습 포인트를 통합한 정제된 응답을 생성합니다.

##### 실제 프롬프트 (1) — 대화 요약

**User Message** (`CONVERSATION_SUMMARIZATION_PROMPT`):

> Placeholder: `{problem_text}` → 문제 원문, `{ground_truth}` → "42", `{conversation_history}` → 4회 반복 전체 대화 히스토리

```
You are a teacher analyzing a tutoring session where a student struggled with a problem.

[Problem]
Julie is reading a 120-page book...

[Correct Answer]
42

[Full Conversation History]
[Iteration 1~4의 Student 응답 + Teacher 평가 + Scaffolding Artifact 전체]

[Your Task]
Summarize this tutoring session concisely, focusing on what's important for understanding the student's learning gaps.

[Output Format]
Keep your summary under 1000 characters. Use this structure:

ATTEMPT SUMMARY:
- Iter 1: [approach] → [specific error] → Answer: [answer]
- Iter 2: [approach] → [specific error] → Answer: [answer]
...

KEY PATTERNS:
- Main weakness: [specific skill/concept gap]
- Recurring error: [pattern across attempts]

Do NOT include lengthy explanations. Be telegraphic and specific.
```

##### 실제 프롬프트 (2) — 성공 재구성

**User Message** (`SUCCESSFUL_SCAFFOLDING_RECONSTRUCTION_PROMPT`):

> Placeholder: `{problem_text}` → 문제 원문, `{ground_truth}` → "42", `{task_analysis}` → Task Analysis Tree, `{iterations_needed}` → 4, `{conversation_summary}` → 위 대화 요약 결과, `{final_response}` → Student 4회차 최종 응답

```
You are an expert teacher reconstructing a successful learning outcome into clean SFT training data.

[Problem]
Julie is reading a 120-page book...

[Correct Answer]
42

[Task Analysis]
[Task Analysis Tree 전체]

[Scaffolding Process Summary]
The student succeeded after 4 iterations.
[대화 요약 결과]

[Final Successful Response]
[Student 4회차 응답]

[Your Task]
Reconstruct the student's learning journey into a single, clean response that:
1. Incorporates the key insights gained through scaffolding
2. Presents a clear, step-by-step solution
3. Naturally integrates the guidance that led to success
4. Is suitable for SFT training (no explicit mention of scaffolding process)

The reconstructed response should be what an ideal student would produce after having learned from this scaffolding experience.

[Output Format - JSON]
{
    "reconstructed_response": "A clean, comprehensive solution that incorporates the learning from scaffolding...",
    "key_learning_points": ["Point 1", "Point 2", "Point 3"],
    "improvement_summary": "How the student improved through the scaffolding process..."
}

Output ONLY the JSON object above. Do not include any additional text, explanation, or commentary outside the JSON structure.
```

##### Reconstruction 결과

```
sft_response: "Julie read 12 pages yesterday and 24 pages today, totaling 36 pages. The remaining pages are 120 - 36 = 84. The ratio of remaining pages to total pages is 84:120, which simplifies to 7:10. To find half of the remaining pages, calculate 84 ÷ 2 = 42. Thus, Julie should read 42 pages tomorrow."
```

#### 최종 로그 요약

```json
{
  "id": "gsm8k_train_3",
  "sft_case": "B",
  "iterative_scaffolding": {
    "success": true,
    "iterations_needed": 4
  },
  "skip_details": {
    "step2_performance_objectives_evaluation": { "is_fallback": false, "attempts_needed": 1 },
    "step3_scaffolding_artifact_generation": { "is_fallback": false, "attempts_needed": 1 },
    "step5_case_b_reconstruction": { "is_fallback": false }
  }
}
```

**Case B 흐름 요약:**
```
Iteration 1: Student(42✓) → Teacher(7/8 PO) → Scaffolding(1 HOT)
Iteration 2: Student(42✓) → Teacher(3/4 PO) → Scaffolding(1 LOT)
Iteration 3: Student(42✓) → Teacher(3/4 PO) → Scaffolding(2 HOT)
Iteration 4: Student(42✓) → Teacher(7/7 PO ✓) → Case B 확정
     ↓
Step 5: Teacher가 대화 히스토리 기반 재구성 → SFT 응답 생성
```

> **핵심 관찰**: Student는 매 반복에서 정답(42)을 맞췄지만, ratio/proportional reasoning 관련 PO를 시연하지 못해 반복이 필요했습니다. 4회차에서 ratio 84:120→7:10 단순화와 proportion 설정을 명시적으로 포함하여 모든 PO를 충족했습니다.

---

### Case C: 최대 반복 후 실패

> **문제 ID**: `gsm8k_train_6`
> **문제**: "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?"
> **정답**: 48

#### Iteration 1: Student 초기 응답 → 실패

**Step 1 — Student 초기 응답:**

```
<think>
Okay, let's see. Albert wants to know how many pizza slices he can eat in a day. He buys 2 large pizzas and 2 small ones. I need to calculate the total number of slices.

First, I should figure out how many slices each type of pizza has. A large pizza has 16 slices, so 2 large pizzas would be 2 times 16. Let me calculate that: 2 * 16 = 32 slices from the large pizzas.

Then, the small pizzas. Each small pizza has 8 slices, and there are 2 of them. So 2 * 8 = 16 slices from the small pizzas.

Now, add the slices from both large and small pizzas together. 32 (large) + 16 (small) = 48 slices total.
</think>

1. Large pizzas: 2 × 16 = 32 slices
2. Small pizzas: 2 × 8 = 16 slices
3. Total: 32 + 16 = 48 slices

\boxed{48}
```

**추출된 답**: 48 (정답과 일치!)

**Step 2 — Teacher PO 평가:**

```json
{
  "overall_assessment": {
    "objectives_met": "6 of 8 objectives satisfied",
    "all_satisfied": false,
    "primary_weakness": "Limited application of ratio and proportional reasoning skills",
    "recommended_focus": "Practice identifying when ratio and proportional reasoning are needed in different problem contexts"
  }
}
```

**미충족 PO:**

| PO | 미충족 이유 |
|----|-----------|
| Identify ratios in real-world scenarios | 문제가 ratio 분석을 필요로 하지 않지만, PO는 해당 기술의 적용을 기대 |
| Simplify ratios to their lowest terms | 동일 |
| Establish proportional relationships | 동일 |
| Solve proportions using cross-multiplication | 동일 |
| Evaluate the reasonableness of solutions | 풀이에 합리성 검증 미포함 |

#### Iteration 2~5: 반복 실패

| Iteration | Student 답 | PO 충족 | 핵심 문제 |
|-----------|-----------|---------|----------|
| 1 | 48 ✓ | 6/8 | ratio/proportion PO 미충족 |
| 2 | 48 ✓ | 7/9 | 산술 정확하나 ratio/proportion 여전히 미적용 |
| 3 | 48 ✓ | 3/4 | ratio/proportion 불필요함을 올바르게 인식했으나 PO 미충족 |
| 4 | 48 ✓ | 4/5 | 동일 패턴 반복 |
| 5 | 48 ✓ | 3/6 | 최대 반복 도달 |

> **핵심 관찰**: Student는 매 반복에서 정답(48)을 정확히 계산했습니다. 그러나 이 문제 자체가 ratio/proportional reasoning을 필요로 하지 않는 단순 산술 문제이므로, 해당 PO를 충족시킬 수 없는 **기술 미스매치(skill mismatch)** 가 발생했습니다. Student는 "ratio와 proportion이 이 문제에 불필요하다"고 올바르게 판단했지만, PO 평가 시스템은 이를 미충족으로 처리했습니다.

5회 반복 후에도 모든 PO 미충족 → **Case C 확정**

#### Case C Step 5: Final Solution

Teacher가 Student의 약점을 분석한 뒤, 정답(48)을 기반으로 교육적 풀이를 생성합니다.

##### 실제 프롬프트

**User Message** (`TEACHER_FINAL_SOLUTION_PROMPT`):

> Placeholder: `{max_iterations}` → 5, `{problem_text}` → 문제 원문, `{ground_truth}` → "48", `{task_analysis}` → Task Analysis Tree, `{iterations_count}` → 5, `{scaffolding_history}` → 5회 반복 Scaffolding 히스토리, `{student_weaknesses}` → 반복 실패 분석 결과

```
You are a teacher providing a complete, correct solution after the student failed to solve the problem after 5 attempts.

[Problem]
Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?

[Correct Answer]
48

[Instructional Analysis]
[Task Analysis Tree 전체]

[Scaffolding History]
The following scaffolding was provided across 5 iterations:
[5회 반복 Scaffolding Artifact + Teacher 평가 히스토리]

[Student's Persistent Weaknesses]
Based on the failed attempts, the student consistently struggled with:
- Recognizing when ratio and proportional reasoning skills are applicable vs. not applicable
- The problem itself did not require ratio analysis, creating a mismatch between problem type and learning objectives

[Instructions]
Generate a complete, educational solution that:
1. Directly addresses each of the student's identified weaknesses
2. Demonstrates the correct reasoning process step by step
3. Highlights the key concepts and strategies the student missed
4. Explains WHY each step is necessary (not just WHAT to do)
5. Serves as an ideal learning example for SFT training

The solution should be what an expert student would produce - clear, complete, and pedagogically valuable.

[Output Format - JSON]
{
  "solution_explanation": "Complete step-by-step solution with clear reasoning. Format:\n[Understanding the Problem]\n...\n[Key Concepts Applied]\n...\n[Step-by-Step Solution]\n...\n[Common Pitfalls Addressed]\n...\nAnswer: [correct answer]",
  "addressed_weaknesses": [
    "How weakness 1 was addressed in the solution",
    "How weakness 2 was addressed in the solution"
  ],
  "key_learning_points": [
    "Key takeaway 1 from this problem",
    "Key takeaway 2 from this problem",
    "Key takeaway 3 from this problem"
  ],
  "final_answer": "The correct answer"
}

CRITICAL INSTRUCTIONS:
1. Your response MUST be ONLY valid JSON - no additional text
2. The solution_explanation should be comprehensive and educational
3. Explicitly connect the solution to the student's weaknesses
4. Ensure all brackets and quotes are properly closed

Output ONLY the JSON object above.
```

##### Teacher 최종 풀이 결과

```
[Understanding the Problem]
Albert buys 2 large pizzas (16 slices each) and 2 small pizzas (8 slices each). The problem asks for the total number of slices he eats if he eats all of them.

[Key Concepts Applied]
This problem requires arithmetic operations (multiplication and addition) to calculate total slices. Ratios or proportional reasoning are not needed here since the question is about summing quantities, not comparing or scaling them.

[Step-by-Step Solution]
1. Calculate slices from large pizzas: 2 pizzas * 16 slices/pizza = 32 slices
2. Calculate slices from small pizzas: 2 pizzas * 8 slices/pizza = 16 slices
3. Add both totals: 32 slices + 16 slices = 48 slices

[Common Pitfalls Addressed]
- Recognizing that ratios/proportions are not required for this problem
- Avoiding errors in multiplication (e.g., confusing 16*2 with 8*2)
- Ensuring correct addition of totals

Answer: \boxed{48}
```

#### 최종 로그 요약

```json
{
  "id": "gsm8k_train_6",
  "sft_case": "C",
  "predicted_answer": "48",
  "scaffolding_correct": false,
  "iterative_scaffolding": {
    "success": false,
    "iterations_needed": 5
  },
  "_failure_metadata": {
    "step2_performance_objectives_evaluation": {
      "is_fallback": false,
      "attempts_needed": 1
    }
  }
}
```

**Case C 흐름 요약:**
```
Iteration 1: Student(48✓) → Teacher(6/8 PO) → Scaffolding(4 HOT)
Iteration 2: Student(48✓) → Teacher(7/9 PO) → Scaffolding(4 HOT)
Iteration 3: Student(48✓) → Teacher(3/4 PO) → Scaffolding(2 LOT)
Iteration 4: Student(48✓) → Teacher(4/5 PO) → Scaffolding(1 LOT)
Iteration 5: Student(48✓) → Teacher(3/6 PO) → 최대 반복 도달
     ↓
Step 5: Teacher가 정답 기반 교육적 풀이 생성
     → "[Key Concepts Applied]", "[Common Pitfalls Addressed]" 섹션 포함
```

> **분석**: 이 Case C는 Student의 수학적 오류가 아니라 **문제 유형과 학습 목표 간의 미스매치**에 의해 발생했습니다. 문제 자체가 ratio/proportional reasoning을 요구하지 않음에도 해당 PO가 평가 기준에 포함되어 있어, Student가 올바르게 "불필요하다"고 판단해도 PO를 충족시킬 수 없었습니다.

---

## 4. Phase 3: Instructional Delivery

Phase 2에서 생성된 SFT 데이터로 Student 모델을 Fine-tuning하고, 평가 데이터셋에서 성능을 측정합니다.

### SFT 데이터 형식

```json
{
  "instruction": "[Enhanced Instruction with Task Analysis — <think> 블록 포함]",
  "input": "Question: [문제 텍스트]",
  "output": "[Case별 SFT 응답 — <think> 블록 포함]",
  "metadata": {
    "id": "gsm8k_train_XXX",
    "sft_case": "A|B|C",
    "ground_truth": "[정답]"
  }
}
```

> **`<think>` 블록**: Qwen3-4B는 응답 시 `<think>...</think>` thinking trace를 자동 생성합니다. SFT 데이터의 `output` 필드에도 이 패턴이 그대로 포함되며, Fine-tuning 시 Student 모델이 이 사고 과정을 학습합니다.

### Case별 SFT 엔트리 예시

#### Case A 엔트리 (gsm8k_train_0)

```json
{
  "instruction": "You are a helpful math assistant.\nSolve this mathematical problem step by step...\n\n## Learning Objective\nYour response should demonstrate: The model will solve multi-step mathematical problems by applying arithmetic operations, ratios, and proportional reasoning to real-world scenarios.\n\n## Problem-Solving Guidelines\n...\n<think>\n[Qwen3-4B thinking trace]\n</think>\n### Instructional Analysis Results\n...",
  "input": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "output": "<think>\nOkay, let's solve this problem step by step. First, I need to figure out how many clips Natalia sold in May...\n\n\\boxed{72}",
  "metadata": {
    "id": "gsm8k_train_0",
    "sft_case": "A",
    "ground_truth": "72"
  }
}
```

> **Case A**: Student 원본 응답(`<think>` 블록 포함)을 그대로 사용. 1회차에 모든 PO를 충족했으므로 추가 가공 없음.

#### Case B 엔트리 (gsm8k_train_3)

```json
{
  "instruction": "[Enhanced Instruction with Task Analysis — Case A와 동일한 구조]",
  "input": "Question: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
  "output": "Julie read 12 pages yesterday and 24 pages today, totaling 36 pages. The remaining pages are 120 - 36 = 84. The ratio of remaining pages to total pages is 84:120, which simplifies to 7:10. To find half of the remaining pages, calculate 84 ÷ 2 = 42. Thus, Julie should read 42 pages tomorrow.",
  "metadata": {
    "id": "gsm8k_train_3",
    "sft_case": "B",
    "ground_truth": "42"
  }
}
```

> **Case B**: Teacher가 4회 반복의 학습 과정을 통합하여 재구성한 응답. Scaffolding 언급 없이 이상적인 학생 응답 형태로 변환됨.

#### Case C 엔트리 (gsm8k_train_6)

```json
{
  "instruction": "[Enhanced Instruction with Task Analysis — Case A와 동일한 구조]",
  "input": "Question: Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
  "output": "[Understanding the Problem]\nAlbert buys 2 large pizzas (16 slices each) and 2 small pizzas (8 slices each). The problem asks for the total number of slices he eats if he eats all of them.\n\n[Key Concepts Applied]\nThis problem requires arithmetic operations (multiplication and addition) to calculate total slices. Ratios or proportional reasoning are not needed here since the question is about summing quantities, not comparing or scaling them.\n\n[Step-by-Step Solution]\n1. Calculate slices from large pizzas: 2 pizzas * 16 slices/pizza = 32 slices\n2. Calculate slices from small pizzas: 2 pizzas * 8 slices/pizza = 16 slices\n3. Add both totals: 32 slices + 16 slices = 48 slices\n\n[Common Pitfalls Addressed]\n- Recognizing that ratios/proportions are not required for this problem\n- Avoiding errors in multiplication (e.g., confusing 16*2 with 8*2)\n- Ensuring correct addition of totals\n\nAnswer: \\boxed{48}",
  "metadata": {
    "id": "gsm8k_train_6",
    "sft_case": "C",
    "ground_truth": "48"
  }
}
```

> **Case C**: Teacher가 교육적 풀이를 생성. `[Key Concepts Applied]`, `[Common Pitfalls Addressed]` 섹션이 포함되어 이 문제에서 ratio/proportion이 불필요함을 명시적으로 설명합니다.

### Case별 SFT 응답 소스

| Case | SFT `output` 소스 | 특징 |
|------|-------------------|------|
| **A** | Student 원본 응답 | 자체 능력으로 정답 도출, `<think>` 블록 포함 |
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

### GSM8K + Qwen3-4B 실행 결과

| 항목 | 값 |
|------|-----|
| **전체 문제 수** | 290 |
| **처리 완료** | 290 (100%) |

### Case 분포

| Case | 건수 | 비율 | 설명 |
|------|------|------|------|
| **A** | 172 | 59.3% | 1회차 성공 |
| **B** | 86 | 29.7% | 2~5회차 성공 |
| **C** | 32 | 11.0% | 최대 반복 후 실패 |

```
Case A ██████████████████████████████ 59.3%
Case B ███████████████               29.7%
Case C ██████                        11.0%
```

> **해석**: Qwen3-4B 모델은 GSM8K 문제의 59.3%를 Task Analysis 기반 Enhanced Instruction만으로 1회에 해결했습니다. 29.7%는 Teacher의 Scaffolding을 통해 개선되었고, 11.0%가 최대 반복 후에도 PO를 충족하지 못했습니다.
>
> **Qwen2.5-7B-Instruct와 비교**: 이전 Qwen2.5-7B-Instruct 실행에서는 Case A가 92.8%였으나, 더 작은 모델인 Qwen3-4B에서는 59.3%로 크게 낮아졌습니다. 이는 모델 크기가 작을수록 Task Analysis만으로는 PO를 충족시키기 어려우며, 더 많은 Scaffolding이 필요함을 보여줍니다. 특히 Case C의 11.0%는 앞서 Case C 예시에서 관찰된 것처럼 문제 유형과 학습 목표 간의 기술 미스매치가 주요 원인입니다.

---

## 부록 A: 주요 개념 정리

### HOT vs LOT Scaffolding

| 유형 | 대상 인지 수준 | 제공 내용 |
|------|--------------|----------|
| **HOT** (High-Order Thinking) | 분석/평가/창조 | `strategy_suggestion`, `partial_example`, `feedback` |
| **LOT** (Low-Order Thinking) | 기억/이해/적용 | `missed_concept`, `brief_explanation`, `key_attention_points` |

### Scaffolding Artifact

- 각 iteration에서 생성된 Scaffolding Artifact가 **누적** 저장됩니다
- Student는 재응답 시 Teacher의 피드백과 Scaffolding Artifact를 참조하여 개선된 응답을 생성합니다
- `feedback` 필드는 4요소 구조: `error_analysis`, `improvement_direction`, `response_comment`, `metacognitive_prompt`

### Skip/Fallback 처리

| Step | 실패 원인 | Fallback 동작 |
|------|-----------|---------------|
| Step 2 (PO 평가) | API 에러, JSON 파싱 실패 | 보수적 평가 → Skip |
| Step 3 (Scaffolding) | API 에러, 생성 실패 | 기본 LOT Scaffolding → Skip |
| Step 5 (재구성) | 재구성 실패 | Case B: 학생 최종 응답 / Case C: ground_truth 기반 |

**`_failure_metadata` 필드:**
- `is_fallback`: Fallback 처리 여부
- `attempts_needed`: 성공까지 필요했던 API 호출 횟수

---

## 부록 B: 프롬프트 상수 참조 테이블

| 상수명 | 소스 파일 | 용도 | Phase/Step | 메시지 역할 | 출력 형식 |
|--------|----------|------|-----------|------------|----------|
| `INSTRUCTIONAL_GOAL_SYSTEM_MESSAGE` | `prompts/instructional_goal_prompts.py` | 교수설계 전문가 역할 설정 | Phase 1 / Step 0 | System | — |
| `INSTRUCTIONAL_GOAL_PROMPT` | `prompts/instructional_goal_prompts.py` | 데이터셋 분석 → Instructional Goal 도출 | Phase 1 / Step 0 | User | JSON |
| `INSTRUCTIONAL_ANALYSIS_PROMPT` | `prompts/design_prompts.py` | Learning Objective → Task Analysis Tree 분해 | Phase 1 / Step 2 | User (system=None) | Text (Tree) |
| `PERFORMANCE_OBJECTIVES_PROMPT` | `prompts/design_prompts.py` | Task Analysis → Performance Objectives 생성 | Phase 1 / Step 3 | User (system=None) | JSON |
| `ENHANCED_INSTRUCTION_TEMPLATE` | `utils/dataset_enhancer.py` | 원본 instruction + Goal + Analysis 주입 | Phase 1 / Enhanced Data | — (LLM 미호출) | Text |
| `SCAFFOLDING_SYSTEM_PROMPT` | `prompts/learning_prompts.py` | Student 문제 해결 시스템 프롬프트 | Phase 2 / Step 1 | System | Text |
| `TEACHER_INTERVENTION_PROMPT` | `prompts/learning_prompts.py` | Teacher PO 평가 + 4요소 구조화 피드백 (`error_analysis`, `improvement_direction`, `response_comment`, `metacognitive_prompt`) | Phase 2 / Step 2 | User | JSON |
| `SCAFFOLDING_ARTIFACT_PROMPT` | `prompts/learning_prompts.py` | 미충족 PO별 HOT/LOT Scaffolding 생성 (`feedback` 필드) | Phase 2 / Step 3 | User | JSON |
| `STUDENT_WITH_ARTIFACT_PROMPT` | `prompts/learning_prompts.py` | Student: Teacher 피드백 + Scaffolding Artifact 참조 재응답 | Phase 2 / Step 4 | User | Text |
| `CONVERSATION_SUMMARIZATION_PROMPT` | `prompts/learning_prompts.py` | 튜터링 세션 대화 요약 | Phase 2 / Step 5 (Case B) | User | Text |
| `SUCCESSFUL_SCAFFOLDING_RECONSTRUCTION_PROMPT` | `prompts/learning_prompts.py` | 성공 학습 과정 → SFT 응답 재구성 | Phase 2 / Step 5 (Case B) | User | JSON |
| `TEACHER_FINAL_SOLUTION_PROMPT` | `prompts/learning_prompts.py` | 최대 반복 실패 후 교육적 풀이 생성 (`[Key Concepts Applied]` 섹션 포함) | Phase 2 / Step 5 (Case C) | User | JSON |
