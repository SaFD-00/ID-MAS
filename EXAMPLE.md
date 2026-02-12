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
     - [Step 5: SFT 응답 결정](#case-b-step-5-sft-응답-결정)
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
  ├── Step 2: Teacher PO 평가 (평가 전용) → 성공이면 Case A/B
  ├── Step 3: Scaffolding Artifact + 서술형 피드백 생성
  ├── Step 4: Student 재응답 (Teacher 피드백 참조)
  ├── (Step 2~4 반복, 최대 5회)
  ├── Step 5a: Teacher Positive Feedback (Case A/B) — 강점 + 개선점
  ├── Step 5b: Student Self-Refinement (Case A/B) — 응답 개선
  ├── Step 5c: Final Solution (Case C만) — 교육적 풀이 평문 텍스트 출력
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
  "instructional_goal": "The model will solve complex mathematical problems by analyzing relationships between quantities, performing multi-step calculations, and applying mathematical reasoning to arrive at accurate solutions in real-world contexts.",
  "cognitive_level": "Create",
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

> Placeholder: `{sample_count}` → 20, `{train_data}` → `utils/prompt_helpers.py`의 `format_samples_for_prompt()` 출력

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

> **`{train_data}` 치환 예시**: `utils/prompt_helpers.py`의 `format_samples_for_prompt()` 함수는 각 샘플의 `instruction`(최대 200자)과 `input`(최대 500자)만 추출하여 `### Sample N` 형식으로 포맷합니다. `output` 필드는 학습목표 도출 편향을 방지하기 위해 의도적으로 제외됩니다.

### Step 1: Learning Objective 설정

Instructional Goal을 그대로 Learning Objective로 설정합니다.

### Step 2: Instructional Analysis

Learning Objective를 Subskills와 Subtasks의 계층 구조로 분해합니다.

**생성 결과 (Task Analysis Tree):**
```
Instructional Goal: The model will solve complex mathematical problems by analyzing
              relationships between quantities, performing multi-step calculations,
              and applying mathematical reasoning to arrive at accurate solutions
              in real-world contexts. (Create)
 ├── Analyze relationships between quantities (Analyze)
 │   ├── Identify relationships between variables in real-world scenarios (Analyze)
 │   ├── Interpret mathematical models to represent real-world situations (Analyze)
 ├── Perform multi-step calculations (Apply)
 │   ├── Execute calculations involving multiple operations and steps (Apply)
 │   ├── Maintain accuracy throughout multi-step problem-solving processes (Apply)
 ├── Apply mathematical reasoning (Apply)
 │   ├── Use logical reasoning to determine appropriate strategies for problem-solving (Apply)
 │   ├── Justify solutions using mathematical principles and evidence (Evaluate)
```

#### 실제 프롬프트

**System Message** (`INSTRUCTIONAL_ANALYSIS_SYSTEM_PROMPT`):
```
You are an instructional design expert. Perform the Instructional Analysis step of the Dick & Carey model for the learning objective provided below.
```

**User Message** (`INSTRUCTIONAL_ANALYSIS_USER_PROMPT`):

> Placeholder: `{learning_objective}` → 실제 GSM8K Instructional Goal

```
[Learning objective]: The model will solve complex mathematical problems by analyzing relationships between quantities, performing multi-step calculations, and applying mathematical reasoning to arrive at accurate solutions in real-world contexts.

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

행동(Behavior), 조건(Condition), 기준(Criterion)을 하나의 문장으로 통합한 수행목표를 각 Subskill에 대해 생성합니다. 이 PO들이 **Phase 2에서 학생 응답 평가의 기준**이 됩니다.

**생성 결과 (10개 PO):**

| # | Target | Performance Objective |
|---|--------|----------------------|
| 1 | Instructional Goal | Given a complex mathematical problem involving real-world contexts, the LLM will solve the problem by analyzing relationships between quantities, performing multi-step calculations, and applying mathematical reasoning to arrive at an accurate solution. |
| 2 | Analyze relationships between quantities | Given a real-world scenario with multiple variables, the LLM will identify the relationships between the variables by interpreting the context and translating it into a mathematical representation. |
| 3 | Identify relationships between variables | Given a real-world problem with multiple variables, the LLM will identify the relationships between the variables by recognizing patterns, dependencies, and functional connections within the scenario. |
| 4 | Interpret mathematical models | Given a mathematical model, the LLM will interpret it to represent a real-world situation by explaining the meaning of variables, equations, and relationships within the context of the problem. |
| 5 | Perform multi-step calculations | Given a multi-step mathematical problem requiring multiple operations, the LLM will execute the calculations in the correct order, maintaining accuracy throughout the process. |
| 6 | Execute calculations involving multiple operations | Given a multi-step calculation problem with multiple operations, the LLM will execute the calculations correctly, following the order of operations and maintaining accuracy at each step. |
| 7 | Maintain accuracy throughout multi-step processes | Given a multi-step problem requiring sequential calculations, the LLM will maintain accuracy by ensuring that each step is performed correctly and that any intermediate results are used appropriately in subsequent steps. |
| 8 | Apply mathematical reasoning | Given a mathematical problem, the LLM will use logical reasoning to determine an appropriate strategy for solving the problem by selecting the most effective method based on the problem's structure and constraints. |
| 9 | Use logical reasoning for strategy | Given a complex mathematical problem, the LLM will use logical reasoning to determine an appropriate strategy for solving the problem by selecting and justifying the most effective approach based on the problem's characteristics. |
| 10 | Justify solutions using mathematical principles | Given a solved mathematical problem, the LLM will justify the solution using mathematical principles and evidence by explaining the reasoning behind each step and citing relevant mathematical concepts and formulas. |

#### 실제 프롬프트

**System Message** (`PERFORMANCE_OBJECTIVES_SYSTEM_PROMPT`):
```
You are an instructional designer specializing in the Dick and Carey instructional design model, and a researcher in LLM learning methodologies.
Based on the provided Instructional Goal and Instructional Analysis Result, generate a set of Performance Objectives that will serve as the criteria for evaluating the observable performance within the LLM's reasoning process.
```

**User Message** (`PERFORMANCE_OBJECTIVES_USER_PROMPT`):

> Placeholder: `{instructional_analysis}` → 실제 GSM8K Task Analysis Tree

```
Specifically, they should be created using information from the learning outcomes identified in the Instructional Analysis Results. specializing in the Dick and Carey instructional design model, and a researcher in LLM learning methodologies.
Based on the provided Instructional Goal and Instructional Analysis Result, generate a set of Performance Objectives that will serve as the criteria for evaluating the observable performance within the LLM's reasoning process.
Specifically, they should be created using information from the learning outcomes identified in the Instructional Analysis Results.

[Input Data]
Instructional Analysis Result: ### Instructional Analysis Results
Instructional Goal: The model will solve complex mathematical problems by analyzing
              relationships between quantities, performing multi-step calculations,
              and applying mathematical reasoning to arrive at accurate solutions
              in real-world contexts. (Create)
 ├── Analyze relationships between quantities (Analyze)
 │   ├── Identify relationships between variables in real-world scenarios (Analyze)
 │   ├── Interpret mathematical models to represent real-world situations (Analyze)
 ├── Perform multi-step calculations (Apply)
 │   ├── Execute calculations involving multiple operations and steps (Apply)
 │   ├── Maintain accuracy throughout multi-step problem-solving processes (Apply)
 ├── Apply mathematical reasoning (Apply)
 │   ├── Use logical reasoning to determine appropriate strategies for problem-solving (Apply)
 │   ├── Justify solutions using mathematical principles and evidence (Evaluate)

[Instructions]
For each Subskills and Subtasks in the instructional analysis, you must create at least one Performance Objective. You can create multiple performance objectives for subskills or subtasks that have more than one requirement.
Every Performance Objective must include all three components—Behavior, Condition, and Criterion—and each component must be explicitly stated in one sentence.
- Behavior: This is a description of LLM's intellectual skill including actions, content, and concepts.
- Condition: This is a description of the tools and resources that will be available to the learner when performing the skill. Write the conditions based solely on the data given in the problem or generated during the reasoning process. And it should always begin with 'given ~'.
- Criterion: This is a description of acceptable performance of the skill. The Criterion component must be tailored to the nature of the task: for tasks with correct answers, it must include a clear and measurable standard such as accuracy requirements, acceptable error ranges, or the number of correct responses; whereas for tasks with no single correct answer, it must specify the information or features that must be present for an acceptable response. Furthermore, these criteria must be formulated to evaluate the observable reasoning process within a single problem-solving task.
Each Performance Objective must correspond directly to a single Subskill and Subtask, and you must not add content that does not appear in the Instructional Analysis Result. Each performance objective must start with an action verb and must not include an explicit subject.

[Output Format]
Your output must be formatted as JSON, following this structure and no other form of explanation or commentary:

{
  "performance_objectives": [
    {
      "target": "Instructional Goal",
      "performance_objective": "A single sentence integrating behavior, condition, and criteria"
    },
    {
      "target": "Subskill X",
      "performance_objective": "A single sentence integrating behavior, condition, and criteria"
    },
    {
      "target": "Subtask X",
      "performance_objective": "A single sentence integrating behavior, condition, and criteria"
    }
  ]
}

Output ONLY the JSON object above. Do not include any additional text, explanation, or commentary outside the JSON structure.
```

### Enhanced Data 생성

Phase 1의 결과물(Instructional Goal + Task Analysis)을 원본 학습 데이터의 `metadata`에 추가합니다. `instruction` 필드는 원본 그대로 유지됩니다.

#### 동작 방식

`utils/dataset_enhancer.py`의 `DataEnhancer`가 원본 데이터를 변환합니다 (LLM 호출 없음):

- `instruction`: 원본 유지
- `input`: 원본 유지
- `output`: 원본 유지
- `metadata`: 원본 + `instructional_goal`, `task_analysis` 추가

#### 변환 결과 예시

**변환 전 (원본):**
```json
{
  "instruction": "You are a helpful math assistant.\nSolve this mathematical problem step by step...",
  "input": "Natalia sold clips to 48 of her friends in April...",
  "output": "...\nThe answer is \\boxed{72}",
  "metadata": {}
}
```

**변환 후 (Enhanced):**
```json
{
  "instruction": "You are a helpful math assistant.\nSolve this mathematical problem step by step...",
  "input": "Natalia sold clips to 48 of her friends in April...",
  "output": "...\nThe answer is \\boxed{72}",
  "metadata": {
    "instructional_goal": "The model will solve complex mathematical problems by analyzing relationships between quantities...",
    "task_analysis": "### Instructional Analysis Results\nInstructional Goal: ..."
  }
}
```

> **핵심**: `instruction`은 원본 그대로 유지되며, Phase 2에서 `SCAFFOLDING_SYSTEM_PROMPT`와 동적으로 결합됩니다. SFT 데이터 생성 시에도 `original_instruction + SCAFFOLDING_SYSTEM_PROMPT`로 동적 결합합니다.

---

## 3. Phase 2: Adaptive Scaffolding

Phase 2는 **각 문제별로** 실행됩니다. 교사-학생 반복 상호작용을 통해 SFT 학습 데이터를 생성합니다.

### Case 분류 기준

| Case | 조건 | SFT 응답 소스 |
|------|------|--------------|
| **A** | 1회차에 모든 PO 충족 | Teacher Positive Feedback → Student Self-Refinement → Refined Response |
| **B** | 2~5회차에 모든 PO 충족 | Teacher Positive Feedback → Student Self-Refinement → Refined Response |
| **C** | 5회 반복 후에도 PO 미충족 | Teacher가 정답 기반 교육적 풀이 생성 (평문 텍스트) |

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
The purpose of your response is to demonstrate the attainment of the Instructional Goal: The model will solve complex mathematical problems by analyzing relationships between quantities, performing multi-step calculations, and applying mathematical reasoning to arrive at accurate solutions in real-world contexts.

You must adhere to the specific performance procedures and required knowledge/skills outlined in the Instructional Analysis results below. Ensure that your solution describes the full reasoning process using all provided steps and resources before arriving at the final answer.

[Instructional Analysis]
### Instructional Analysis Results
Instructional Goal: The model will solve complex mathematical problems by analyzing
              relationships between quantities, performing multi-step calculations,
              and applying mathematical reasoning to arrive at accurate solutions
              in real-world contexts. (Create)
 ├── Analyze relationships between quantities (Analyze)
 │   ├── Identify relationships between variables in real-world scenarios (Analyze)
 │   ├── Interpret mathematical models to represent real-world situations (Analyze)
 ├── Perform multi-step calculations (Apply)
 │   ├── Execute calculations involving multiple operations and steps (Apply)
 │   ├── Maintain accuracy throughout multi-step problem-solving processes (Apply)
 ├── Apply mathematical reasoning (Apply)
 │   ├── Use logical reasoning to determine appropriate strategies for problem-solving (Apply)
 │   ├── Justify solutions using mathematical principles and evidence (Evaluate)

[Instructions]
1. Identify which skills and sub-skills from the instructional analysis are relevant to this problem
2. Plan your problem-solving strategy based on the instructional goal and subskills
3. Execute each step systematically, demonstrating the required performance behaviors
4. Ensure your solution describes the full reasoning process using all provided steps and resources
5. Provide your final answer clearly

[Output Format]
- Instructional goal alignment: [how this solution demonstrates the instructional goal]
- Step-by-step reasoning: [your detailed solution following the instructional structure]
- Final answer: "The answer is \boxed{your final answer}"
```

**User Message** = `question["input"]` (문제 원문):
```
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
```

##### Student 응답 결과

```
- Instructional goal alignment: This solution demonstrates the instructional goal by analyzing
  the relationship between the number of clips sold in April and May, performing multi-step
  calculations to find the total number of clips sold, and applying mathematical reasoning to
  arrive at an accurate solution in a real-world context.

- Step-by-step reasoning:
  1. **Identify the quantities and relationships**:
     - In April, Natalia sold 48 clips.
     - In May, she sold half as many clips as in April.
     - The goal is to find the total number of clips sold in both months.

  2. **Calculate the number of clips sold in May**:
     Since Natalia sold half as many clips in May as in April, we calculate:
     $$
     \text{Clips sold in May} = \frac{1}{2} \times 48 = 24
     $$

  3. **Add the number of clips sold in April and May**:
     $$
     \text{Total clips sold} = 48 + 24 = 72
     $$

  4. **Verify the solution**:
     The calculation is accurate and follows logical reasoning. The total number of clips sold
     in April and May is 72.

- Final answer: "The answer is \boxed{72}"
```

**추출된 답**: 72 (정답과 일치)

#### Case A Step 2: Teacher PO 평가

Teacher가 Performance Objectives 기준으로 학생 응답을 **평가만** 수행합니다. 피드백은 생성하지 않습니다.

##### 실제 프롬프트

**System Message** (`TEACHER_INTERVENTION_SYSTEM_PROMPT`):
```
You are a teacher supporting the learning of a student.

Your role is to evaluate the student's response against the established performance objectives. You must monitor the student's reasoning steps to ensure they meet the performance objectives.
```

**User Message** (`TEACHER_INTERVENTION_USER_PROMPT`):

> Placeholder: `{problem_text}` → 문제 원문, `{student_response}` → 위 Student 응답, `{performance_objectives}` → Phase 1에서 생성된 PO JSON, `{ground_truth}` → "72"

```

Your role is to evaluate the student's response against the established performance objectives. You must monitor the student's reasoning steps to ensure they meet the performance objectives.

[Input Data]
- Problem: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
- Student response:
  - Instructional goal alignment: This solution demonstrates the instructional goal by analyzing
    the relationship between the number of clips sold in April and May...
  [... 전체 응답 ...]
- Performance objectives: [10개 PO JSON 전체]
- Ground truth (FOR REFERENCE ONLY - DO NOT REVEAL): 72

[Instructions]
Evaluate the student model's response according to the following rules.
1. Assess student performance according to the performance objectives. Use the criterion embedded in each performance objective as the evaluation standard. Do not reveal correct answers or model solutions.
2. Analyze the student response and determine which performance objectives are satisfied and which are not. All judgments must be grounded in observable reasoning behaviors in the student response, such as how claims are justified, how relationships are analyzed, or how judgments are formed. Avoid vague or abstract evaluations.
3. For each PO, write a "feedback" that references the student's actual response:
   - If satisfied: describe specific strengths observed (e.g., which reasoning steps, strategies, or expressions demonstrate mastery).
   - If NOT satisfied: explain the specific reason the objective was not met, citing what the student wrote or omitted.

[Output Format - JSON]
{
  "performance_evaluation": [
    {
      "objective_content": "Copy the performance_objective field from performance objectives VERBATIM",
      "is_satisfied": true or false,
      "feedback": "If satisfied: specific strengths observed in the student's response. If NOT satisfied: specific reason this objective was not met, referencing what the student wrote or omitted."
    }
  ]
}

Output ONLY valid JSON.
```

##### Teacher 평가 결과

```json
{
  "performance_evaluation": [
    {
      "objective_content": "Given a complex mathematical problem involving real-world contexts, the LLM will solve the problem by analyzing relationships between quantities, performing multi-step calculations, and applying mathematical reasoning to arrive at an accurate solution.",
      "is_satisfied": true,
      "feedback": "The student correctly analyzed the relationship between the number of clips sold in April and May, performed multi-step calculations to find the total, and applied mathematical reasoning to arrive at the accurate solution."
    },
    {
      "objective_content": "Given a real-world scenario with multiple variables, the LLM will identify the relationships between the variables by interpreting the context and translating it into a mathematical representation.",
      "is_satisfied": true,
      "feedback": "The student identified the relationship between the number of clips sold in April and May, translating the context into a mathematical representation by calculating half of 48 for May."
    },
    {
      "objective_content": "Given a multi-step mathematical problem requiring multiple operations, the LLM will execute the calculations in the correct order, maintaining accuracy throughout the process.",
      "is_satisfied": true,
      "feedback": "The student executed the calculations in the correct order, first calculating the number of clips sold in May and then adding it to the number sold in April, maintaining accuracy throughout the process."
    }
  ]
}
```

> **평가 전용**: 현재 `TEACHER_INTERVENTION_PROMPT`는 `objective_content`, `is_satisfied`, `feedback`만 출력합니다. `overall_assessment`는 포함되지 않습니다.

**모든 PO 충족 (10/10)** → 반복 종료 → **Case A 확정**

#### Case A Step 3: Teacher Positive Feedback

모든 PO가 충족되었으므로 Teacher가 학생 응답의 강점과 개선점을 분석하는 Positive Feedback을 생성합니다.

##### 실제 프롬프트

**System Message** (`TEACHER_POSITIVE_FEEDBACK_SYSTEM_PROMPT`):
```
You are a teacher providing constructive feedback to strengthen a student's already satisfactory response.
```

**User Message** (`TEACHER_POSITIVE_FEEDBACK_USER_PROMPT`):

> Placeholder: `{problem_text}` → 문제 원문, `{student_response}` → Student 응답, `{po_evaluation}` → PO 평가 JSON

``` to strengthen a student's already satisfactory response.

[Problem]
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

[Student's Response]
[위 Student 응답 전체]

[Performance Objectives Evaluation]
[10개 PO 평가 JSON — 모두 is_satisfied: true]

[Instructions]
The student's response has satisfied all Performance Objectives. Provide feedback that will help the student further strengthen their response:

1. Strengths Summary: For each satisfied PO, briefly describe what the student did well, citing specific parts of their response.

2. Enhancement Suggestions: Identify 2-3 concrete ways the student could improve their response quality:
   - Clearer reasoning structure or explanations
   - More explicit connections between steps
   - Better demonstration of the underlying concepts
   - More rigorous justification of key steps

3. Integration Guidance: Explain how to incorporate these improvements naturally into the existing solution without changing the core logic or final answer.

[Output Format]

[Strengths]
- PO 1: <what was done well>
- PO 2: <what was done well>
...

[Enhancement Suggestions]
1. <specific improvement suggestion>
2. <specific improvement suggestion>
3. <specific improvement suggestion>

[Integration Guidance]
<How to naturally incorporate these improvements into the response>

Output ONLY the structured text above.
```

#### Case A Step 4: Student Self-Refinement

Student가 Teacher의 Positive Feedback을 참조하여 응답을 개선합니다.

##### 실제 프롬프트

**System Message** (`STUDENT_SELF_REFINEMENT_PROMPT`):

> Placeholder: `{scaffolding_system_prompt}` → `SCAFFOLDING_SYSTEM_PROMPT` (채워진 상태), `{positive_feedback}` → Teacher의 Positive Feedback 텍스트

```
[SCAFFOLDING_SYSTEM_PROMPT — Task Analysis 포함]

[Teacher's Feedback on Your Response]
Your teacher has evaluated your response and confirmed that it meets all performance objectives.
The following feedback highlights your strengths and suggests ways to further improve your response:

[Strengths]
- PO 1: The student correctly identified the relationship between April and May sales...
- PO 2: ...

[Enhancement Suggestions]
1. Add explicit reasoning about why "half as many" translates to division by 2
2. Strengthen the verification step with a check against the problem constraints
3. ...

[Integration Guidance]
Incorporate these improvements by...

[Instructions]
1. Keep your correct reasoning and final answer unchanged.
2. Integrate the enhancement suggestions naturally into your solution.
3. Strengthen the clarity, structure, and justification of your reasoning.
4. Demonstrate deeper understanding of the underlying concepts.
5. Your improved response should be a complete, standalone solution (not a diff or list of changes).

[Output Format]
Write your complete improved solution following the original output format:
- Instructional goal alignment: [how this solution demonstrates the instructional goal]
- Step-by-step reasoning: [your improved detailed solution]
- Final answer: "The answer is \boxed{your final answer}"
```

**User Message** = `question["input"]` (문제 원문)

#### Step 5 & 6: SFT 데이터 생성

Case A는 Self-Refinement 후의 **Refined Response**를 SFT 학습 데이터로 사용합니다.

```json
{
  "instruction": "[Original Instruction + SCAFFOLDING_SYSTEM_PROMPT (동적 결합)]",
  "input": "Question: Natalia sold clips to 48 of her friends in April...",
  "output": "[Student의 Self-Refined 응답]",
  "metadata": {
    "id": "gsm8k_train_0",
    "sft_case": "A",
    "ground_truth": "72"
  }
}
```

**최종 로그 요약:**
- `iterations_needed`: 1
- `scaffolding_artifacts`: null (Scaffolding 불필요)
- `self_refinement`: `{"positive_feedback": {...}, "original_response": "...", "refined_response": "..."}`
- `hot_count`: null
- `lot_count`: null
- `_failure_metadata`: `{"step2_performance_objectives_evaluation": {"is_fallback": false, "attempts_needed": 1}}`

---

### Case B: 2회차 이상 성공

> **문제 ID**: `gsm8k_train_4`
> **문제**: "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"
> **정답**: 624

#### Iteration 1: Student 초기 응답 → PO 평가 실패

**Step 1 — Student 초기 응답:**

(System Message = `SCAFFOLDING_SYSTEM_PROMPT`, User Message = 위 문제 원문 — Case A Step 1과 동일한 프롬프트 구조)

```
- Instructional goal alignment: This solution demonstrates the instructional goal by analyzing the
  relationship between the number of letters, pages, friends, and time, performing multi-step
  calculations to determine the total pages written in a year, and applying mathematical reasoning
  to arrive at an accurate solution.

- Step-by-step reasoning:
  1. **Identify the quantities and their relationships**:
     - James writes **3 pages** per letter.
     - He writes **2 different letters** per week.
     - He writes **twice a week**.
     - We need to find out how many **pages he writes in a year**.

  2. **Determine the total number of letters written per week**:
     - He writes 2 letters per week.
     - So, the total number of letters written per week is:
       $$
       2 \text{ letters/week}
       $$

  3. **Determine the total number of letters written in a year**:
     - There are 52 weeks in a year.
     - So, the total number of letters written in a year is:
       $$
       2 \text{ letters/week} \times 52 \text{ weeks} = 104 \text{ letters}
       $$

  4. **Determine the total number of pages written in a year**:
     - Each letter is 3 pages.
     - So, the total number of pages written in a year is:
       $$
       104 \text{ letters} \times 3 \text{ pages/letter} = 312 \text{ pages}
       $$

- Final answer: "The answer is \boxed{312}"
```

**추출된 답**: 312 (정답 624와 불일치)

**Step 2 — Teacher PO 평가:**

```json
{
  "performance_evaluation": [
    {
      "objective_content": "Given a multi-step mathematical problem requiring multiple operations, the LLM will execute the calculations in the correct order, maintaining accuracy throughout the process.",
      "is_satisfied": false,
      "feedback": "The student's calculation is incorrect. The correct total number of letters written in a year should be 2 letters/week × 52 weeks = 104 letters. However, the student multiplied 104 letters × 3 pages/letter = 312 pages, which is incorrect. The correct answer should be 624 pages."
    },
    {
      "objective_content": "Given a multi-step calculation problem with multiple operations, the LLM will execute the calculations correctly, following the order of operations and maintaining accuracy at each step.",
      "is_satisfied": false,
      "feedback": "The student's final answer of 312 pages is wrong. The correct answer should be 624 pages, indicating that the student's final answer is incorrect."
    },
    {
      "objective_content": "Given a multi-step problem requiring sequential calculations, the LLM will maintain accuracy by ensuring that each step is performed correctly and that any intermediate results are used appropriately in subsequent steps.",
      "is_satisfied": false,
      "feedback": "The student's final answer is incorrect, indicating that the intermediate steps were not maintained with accuracy. The correct answer should be 624 pages, but the student arrived at 312 pages."
    }
  ]
}
```

**미충족 PO:**

| PO | 미충족 이유 |
|----|-----------
| Execute calculations in correct order | "twice a week" 미반영: 주당 2통이 아닌 4통(2×2) |
| Execute calculations correctly | 최종 답 312 vs 정답 624 |
| Maintain accuracy throughout | 중간 결과(주당 편지 수)에서 "twice" 누락 |

> **핵심 관찰**: Student는 "twice a week"을 반영하지 않아 주당 편지 수를 2통으로 계산했습니다. 실제로는 2명에게 × 주 2회 = 주당 4통이 맞습니다. 이는 문제 해석 오류로 인한 **실제 계산 오류**입니다.

#### Case B Step 3: Scaffolding Artifact 생성

Teacher가 미충족 PO별로 차별화된 Scaffolding과 **서술형 피드백**을 생성합니다.

##### 실제 프롬프트

**System Message** (`SCAFFOLDING_ARTIFACT_SYSTEM_PROMPT`):
```
You are an instructional design expert (Dick & Carey model) creating a Scaffolding Artifact to help a student improve.

Your role is to design pedagogical scaffolding for Performance Objectives that the student failed to meet. This scaffolding will be stored as a "Scaffolding Artifact" that the student can reference in their next attempt.
```

**User Message** (`SCAFFOLDING_ARTIFACT_USER_PROMPT`):

> Placeholder: `{problem_text}` → 문제 원문, `{student_response}` → Student의 응답, `{po_evaluation}` → Teacher PO 평가 JSON, `{previous_iteration_summaries}` → 이전 반복 요약 목록, `{instructional_goal}` → Instructional Goal, `{task_analysis}` → Task Analysis Tree

``` to help a student improve.

Your role is to design pedagogical scaffolding for Performance Objectives that the student failed to meet. This scaffolding will be stored as a "Scaffolding Artifact" that the student can reference in their next attempt.

[Input Data]
- Problem: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
- Student's Response: [Student 응답 전체]
- Performance Objectives Evaluation: [Teacher PO 평가 JSON 전체]

[Previous Iteration Summaries]
(No previous iteration summaries. This is the first attempt.)

[Instructions]
1. **Select scaffolding targets**: Focus on Performance Objectives with high failure rates that are critical for achieving the Instructional Goal.

2. **Classify skill level**: For each unmet PO, determine if it requires:
   - **HOT (High-Order Thinking)**: Analyze, Evaluate, Create
   - **LOT (Low-Order Thinking)**: Remember, Understand, Apply

3. **Design appropriate scaffolding**:

   For **HOT skills**:
   - Strategy suggestion: Propose an approach or reasoning strategy
   - Partial worked example: Show partial reasoning (stop before the final answer)
   - Key attention points: What the student should focus on

   For **LOT skills**:
   - Missed concept/information: Explicitly state what the student missed
   - Brief explanation: Provide a concise explanation to minimize cognitive load

4. **Generate integrated narrative feedback**: Write a single cohesive feedback paragraph...

5. **Do NOT reveal correct answers** - guide reasoning, don't solve.

6. **Generate iteration summary**: Write a concise summary of THIS iteration...

[Output Format - Structured Text]

[Instructional Goal]
{instructional_goal}

[Instructional Analysis]
{task_analysis}

[Scaffolding for Task [1] (High Order Skill)]:
- Target Objective: ...
- Cognitive Level: ...
- Failure Analysis: ...
- Suggested Strategy: ...
- Key Attention Points: ...

[Feedback]
<integrated narrative paragraph>

[Iteration Summary]
<3-5 sentence summary>

CRITICAL INSTRUCTIONS:
1. Do NOT reveal correct answers or complete solutions
2. Focus on guiding the reasoning process
3. The [Feedback] section is the primary feedback delivered to the student
4. The [Iteration Summary] must capture BOTH the student's attempt AND the scaffolding provided

Output ONLY the structured text above. Do NOT include JSON formatting.
```

##### Scaffolding 결과

```
[Instructional Goal]
The model will solve complex mathematical problems by analyzing relationships between quantities,
performing multi-step calculations, and applying mathematical reasoning to arrive at accurate
solutions in real-world contexts.

[Instructional Analysis]
### Instructional Analysis Results
Instructional Goal: The model will solve complex mathematical problems by analyzing
              relationships between quantities, performing multi-step calculations,
              and applying mathematical reasoning to arrive at accurate solutions
              in real-world contexts. (Create)
 ├── Analyze relationships between quantities (Analyze)
 │   ├── Identify relationships between variables in real-world scenarios (Analyze)
 │   ├── Interpret mathematical models to represent real-world situations (Analyze)
 ├── Perform multi-step calculations (Apply)
 │   ├── Execute calculations involving multiple operations and steps (Apply)
 │   ├── Maintain accuracy throughout multi-step problem-solving processes (Apply)
 ├── Apply mathematical reasoning (Apply)
 │   ├── Use logical reasoning to determine appropriate strategies for problem-solving (Apply)
 │   ├── Justify solutions using mathematical principles and evidence (Evaluate)

[Scaffolding for Task [1] (High Order Skill)]:
- Target Objective: Perform multi-step calculations and maintain accuracy throughout the process
- Cognitive Level: Apply
- Failure Analysis: The student performed the calculations correctly but arrived at the wrong
  final answer, indicating a miscalculation in the final step
- Suggested Strategy:
  (a) Strategy 1: Break down the problem into smaller steps and verify each step before moving
      to the next
      - Partial worked example (stop before the final answer):
        1. James writes 3 pages per letter.
        2. He writes 2 different letters per week.
        3. He writes twice a week.
        4. Total number of pages written in a year is:
           104 letters × 3 pages/letter = ?
           *(Stop here – verify this multiplication step before proceeding)*

  (b) Strategy 2: Use a different approach to verify the result
      - Teacher's reasoning clarification:
        If you multiply 104 letters by 3 pages per letter, you should get 312 pages. However,
        if you consider that James writes **2 letters twice a week**, that means he writes
        **4 letters per week**. Then, 4 letters/week × 52 weeks = 208 letters, and 208 × 3 = 624
        pages. This alternative approach highlights a potential misunderstanding of the problem's
        structure.

- Key Attention Points: Ensure that you correctly interpret the problem's wording and that all
  multiplications are performed accurately.

[Scaffolding for Task [2] (Low Order Skill)]:
- Target Objective: Maintain accuracy throughout multi-step calculations
- Cognitive Level: Apply
- Failure Analysis: The student multiplied 104 letters by 3 pages/letter but arrived at an
  incorrect final answer
- Missed Concept/Information: The problem states that James writes **2 different letters twice
  a week**, which means he writes **4 letters per week**, not 2
- Brief Explanation: The problem says he writes 2 letters **twice a week**, so the total number
  of letters per week is 2 × 2 = 4, not 2. This is a critical detail that affects the final
  calculation.

[Feedback]
The student correctly identified the relationships between the quantities and followed a logical
approach to solve the problem. However, the final answer was incorrect, which suggests a
miscalculation in the multiplication step. The key issue is that the student may have
misunderstood the problem's wording, specifically that James writes 2 letters **twice a week**,
meaning he writes 4 letters per week. To improve, the student should carefully re-read the
problem, verify each step of the calculation, and double-check the multiplication. A good way to
verify is to calculate the total number of letters per week again and ensure that each step is
performed accurately before moving to the next.

[Iteration Summary]
The student attempted to solve the problem by breaking it down into smaller steps and correctly
identified the relationships between the variables. However, the final answer was incorrect,
indicating a miscalculation in the multiplication step. The scaffolding provided focuses on
clarifying the problem's wording and emphasizing the importance of verifying each step of the
calculation. The student was guided to re-examine the problem's structure and ensure accuracy in
all steps.
```

> **핵심 변경**: 출력 형식이 JSON에서 구조화된 마크다운으로 변경되었습니다. `[Feedback]` 섹션은 서술형 단락으로, (1) 오류 분석, (2) 개선 방향, (3) 검증 단계를 통합합니다. 전체 Scaffolding Artifact 텍스트(`_raw_text`)가 학생에게 직접 전달됩니다.

#### Case B Step 4: Student 재응답

Student가 Teacher의 **서술형 피드백**을 참조하여 개선된 응답을 생성합니다.

##### 실제 프롬프트

**System Message** (`STUDENT_FEEDBACK_RESPONSE_PROMPT`):

> Placeholder: `{dataset_prompt}` → 원본 데이터셋 instruction, `{scaffolding_system_prompt}` → `SCAFFOLDING_SYSTEM_PROMPT` (채워진 상태), `{scaffolding_artifact}` → Scaffolding Artifact 전체 텍스트 (`_raw_text`)

```
You are a helpful math assistant.
Solve this mathematical problem step by step. Show your reasoning clearly and use proper mathematical notation.

## Response Format
Your final answer MUST be within \boxed{}.
Example: \boxed{42}

The purpose of your response is to demonstrate the attainment of the Instructional Goal: The model will solve complex mathematical problems by analyzing relationships between quantities, performing multi-step calculations, and applying mathematical reasoning to arrive at accurate solutions in real-world contexts.

You must adhere to the specific performance procedures and required knowledge/skills outlined in the Instructional Analysis results below. Ensure that your solution describes the full reasoning process using all provided steps and resources before arriving at the final answer.

[Instructional Analysis]
### Instructional Analysis Results
Instructional Goal: The model will solve complex mathematical problems by analyzing
              relationships between quantities, performing multi-step calculations,
              and applying mathematical reasoning to arrive at accurate solutions
              in real-world contexts. (Create)
 ├── Analyze relationships between quantities (Analyze)
 │   ├── Identify relationships between variables in real-world scenarios (Analyze)
 │   ├── Interpret mathematical models to represent real-world situations (Analyze)
 [... 전체 Task Analysis ...]

[Instructions]
1. Identify which skills and sub-skills from the instructional analysis are relevant to this problem
[... SCAFFOLDING_SYSTEM_PROMPT의 나머지 ...]

[Output Format]
- Instructional goal alignment: [how this solution demonstrates the instructional goal]
- Step-by-step reasoning: [your detailed solution following the instructional structure]
- Final answer: "The answer is \boxed{your final answer}"

[Scaffolding Artifact]
Your teacher has evaluated your previous response and designed the following scaffolding to guide your improvement:

[Instructional Goal]
The model will solve complex mathematical problems by analyzing relationships between quantities,
performing multi-step calculations, and applying mathematical reasoning to arrive at accurate
solutions in real-world contexts.

[Instructional Analysis]
### Instructional Analysis Results
[... Task Analysis Tree ...]

[Scaffolding for Task [1] (High Order Skill)]:
- Target Objective: Perform multi-step calculations and maintain accuracy throughout the process
- Cognitive Level: Apply
- Failure Analysis: The student performed the calculations correctly but arrived at the wrong
  final answer, indicating a miscalculation in the final step
- Suggested Strategy:
  (a) Strategy 1: Break down the problem into smaller steps and verify each step...
  (b) Strategy 2: Use a different approach to verify the result...
- Key Attention Points: Ensure that you correctly interpret the problem's wording...

[Scaffolding for Task [2] (Low Order Skill)]:
- Target Objective: Maintain accuracy throughout multi-step calculations
- Missed Concept/Information: The problem states that James writes **2 different letters twice
  a week**, which means he writes **4 letters per week**, not 2
- Brief Explanation: The problem says he writes 2 letters **twice a week**, so the total number
  of letters per week is 2 × 2 = 4, not 2.

[Feedback]
The student correctly identified the relationships between the quantities and followed a logical
approach to solve the problem. However, the final answer was incorrect...

[Iteration Summary]
The student attempted to solve the problem by breaking it down into smaller steps...

[Instructions]
1. Carefully study the scaffolding artifact above, including the strategies and examples provided
2. For High Order Skills: follow the suggested strategies and reasoning approaches
3. For Low Order Skills: review the missed concepts and explanations
4. Pay special attention to the Key Attention Points and Feedback sections
5. Address each unsatisfied performance objective systematically
6. Show your improved thinking step by step
7. Provide your final answer clearly
```

**User Message** = `question["input"]` (문제 원문):
```
James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?
```

> **핵심 변경**: Student는 `dataset_prompt` + `SCAFFOLDING_SYSTEM_PROMPT` + Teacher의 **전체 Scaffolding Artifact 텍스트**를 참조합니다. 이전에는 서술형 `feedback` 필드만 전달되었으나, 이제 HOT/LOT 스캐폴딩, 전략, 부분 예시, Feedback, Iteration Summary 등 모든 교수적 지원이 학생에게 제공됩니다.

##### Iteration 2 Student 응답

```
- Instructional goal alignment: This solution demonstrates the instructional goal by analyzing
  relationships between quantities (letters, pages, and weeks), performing multi-step calculations
  (multiplying letters by pages and weeks), and applying mathematical reasoning to correctly
  interpret and solve the problem.

- Step-by-step reasoning:
  1. **Identify the quantities and their relationships**:
     - James writes **3 pages** per letter.
     - He writes **2 different letters** per week.
     - He writes **twice a week**, meaning he writes **2 letters twice** per week.
     - We need to find out how many **pages he writes in a year**.

  2. **Determine the total number of letters written per week**:
     - He writes 2 letters **twice a week**, so:
       $$
       2 \text{ letters} \times 2 = 4 \text{ letters/week}
       $$

  3. **Determine the total number of letters written in a year**:
     - There are **52 weeks** in a year.
     - So, the total number of letters written in a year is:
       $$
       4 \text{ letters/week} \times 52 \text{ weeks} = 208 \text{ letters}
       $$

  4. **Determine the total number of pages written in a year**:
     - Each letter is **3 pages**.
     - So, the total number of pages written in a year is:
       $$
       208 \text{ letters} \times 3 \text{ pages/letter} = 624 \text{ pages}
       $$

- Final answer: "The answer is \boxed{624}"
```

**추출된 답**: 624 (정답과 일치)

**Step 2 — Teacher PO 평가 (Iteration 2):**
```json
{
  "performance_evaluation": [
    {
      "objective_content": "Given a multi-step mathematical problem requiring multiple operations, the LLM will execute the calculations in the correct order, maintaining accuracy throughout the process.",
      "is_satisfied": true,
      "feedback": "The student executed the calculations in the correct order (letters per week, letters per year, pages per year) and maintained accuracy throughout the process."
    },
    {
      "objective_content": "Given a multi-step calculation problem with multiple operations, the LLM will execute the calculations correctly, following the order of operations and maintaining accuracy at each step.",
      "is_satisfied": true,
      "feedback": "The student followed the order of operations correctly (multiplying letters per week by weeks per year, then multiplying by pages per letter) and maintained accuracy at each step."
    },
    {
      "objective_content": "Given a multi-step problem requiring sequential calculations, the LLM will maintain accuracy by ensuring that each step is performed correctly and that any intermediate results are used appropriately in subsequent steps.",
      "is_satisfied": true,
      "feedback": "The student maintained accuracy by performing each step correctly and using intermediate results (letters per week, letters per year) appropriately in subsequent calculations."
    }
  ]
}
```

**모든 PO 충족 (10/10)** → 반복 종료 → **Case B 확정** (2회차 성공)

#### Case B Step 5: Self-Refinement

Case B에서도 Case A와 동일하게, PO가 모든 충족된 후 **Teacher Positive Feedback → Student Self-Refinement** 과정을 수행합니다.

```python
# nodes.py
# Case A/B: PO 충족 후 Self-Refinement
positive_feedback_result = teacher_model.generate_positive_feedback(
    problem_text=question["input"],
    student_response=response,
    po_evaluation=evaluation,
)
refined_response = student_model.self_refine_response(
    problem_text=question["input"],
    positive_feedback=positive_feedback_result["feedback_text"],
    task_analysis=task_analysis,
    instructional_goal=instructional_goal,
)
sft_output = refined_response  # Refined Response가 SFT output
sft_case = SFTCase.B.value
```

**SFT output = Iteration 2의 Student Self-Refined 응답**

> **설계 의도**: Scaffolding 과정을 거쳐 Student가 자력으로 PO를 충족시킨 후, Teacher의 Positive Feedback을 통해 응답의 reasoning을 더욱 강화합니다. PO 평가의 강점/개선점을 기존 풀이에 녹여 SFT 데이터 품질을 향상시킵니다.

#### 최종 로그 요약

```json
{
  "id": "gsm8k_train_4",
  "sft_case": "B",
  "iterative_scaffolding": {
    "success": true,
    "iterations_needed": 2
  },
  "hot_count": 1,
  "lot_count": 1,
  "skip_details": {
    "step2_performance_objectives_evaluation": { "is_fallback": false, "attempts_needed": 1 },
    "step3_scaffolding_artifact_generation": { "is_fallback": false, "attempts_needed": 1 }
  }
}
```

**Case B 흐름 요약:**
```
Iteration 1: Student(312✗) → Teacher(7/10 PO) → Scaffolding(1 HOT + 1 LOT) + 서술형 피드백
Iteration 2: Student(624✓) → Teacher(10/10 PO ✓) → Case B 확정
     ↓
Self-Refinement: Teacher Positive Feedback → Student Self-Refine → Refined Response
     ↓
Case B: Refined Response를 SFT 데이터로 사용
```

> **핵심 관찰**: Student는 1회차에서 "twice a week"을 주당 편지 수에 반영하지 않아 312라는 오답을 냈습니다. Teacher의 LOT Scaffolding에서 "2 letters **twice a week** means 4 letters per week"이라는 핵심 개념을 지적한 후, 2회차에서 올바르게 4 letters/week로 수정하여 624라는 정답을 도출했습니다. 이는 **실제 문제 해석 오류**에 의한 Case B로, 교육적 의미가 있는 학습 개선 사례입니다.

---

### Case C: 최대 반복 후 실패

> **참고**: 이번 Qwen3-4B 실행에서는 Case C가 **0건** 발생했습니다. 아래는 파이프라인 동작 설명을 위한 **가상 예시**입니다. 실제 Case C가 발생할 경우의 프롬프트와 처리 흐름을 보여줍니다.

> **가상 문제**: "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?"
> **정답**: 48

5회 반복 후에도 모든 PO를 충족하지 못한 경우 → **Case C 확정**

#### Case C Step 5: Final Solution

Teacher가 Student의 약점을 분석한 뒤, 정답(48)을 기반으로 교육적 풀이를 **평문 텍스트**로 생성합니다.

##### 실제 프롬프트

**System Message** (`TEACHER_FINAL_SOLUTION_SYSTEM_PROMPT`):
```
You are a teacher providing a complete, correct solution after the student failed to solve the problem after 5 attempts.
```

**User Message** (`TEACHER_FINAL_SOLUTION_USER_PROMPT`):

> Placeholder: `{max_iterations}` → 5, `{problem_text}` → 문제 원문, `{ground_truth}` → "48", `{task_analysis}` → Task Analysis Tree (최대 1500자 제한), `{last_iteration_summary}` → 마지막 iteration의 summary 텍스트, `{student_weaknesses}` → `extract_student_weaknesses()`로 추출한 약점 목록 (최대 5개)

``` after the student failed to solve the problem after 5 attempts.

[Problem]
Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?

[Correct Answer]
48

[Instructional Analysis]
[Task Analysis Tree 전체]

[Last Iteration Summary]
The following is a summary of the student's last attempt and the scaffolding provided:
[마지막 iteration의 summary 텍스트]

[Student's Persistent Weaknesses]
Based on the failed attempts, the student consistently struggled with:
- [약점 목록]

[Instructions]
Generate a complete, educational solution that:
1. Directly addresses each of the student's identified weaknesses
2. Demonstrates the correct reasoning process step by step
3. Highlights the key concepts and strategies the student missed
4. Explains WHY each step is necessary (not just WHAT to do)
5. Serves as an ideal learning example for SFT training

The solution should be what an expert student would produce - clear, complete, and pedagogically valuable.

[Output Format]
Write your response as plain text (NOT JSON). Structure your solution clearly and end with the boxed answer.

Example format:
[Understanding the Problem]
Let me analyze this problem step by step...

[Key Concepts Applied]
The key insight here is...

[Step-by-Step Solution]
Step 1: ...
Step 2: ...

The answer is \boxed{correct answer}

Output ONLY the solution as plain text. Do not include any JSON, metadata, or commentary.
```

##### Teacher 최종 풀이 결과 (가상)

```
[Understanding the Problem]
Albert buys 2 large pizzas (16 slices each) and 2 small pizzas (8 slices each). The problem asks
for the total number of slices he eats if he eats all of them.

[Key Concepts Applied]
This problem requires arithmetic operations (multiplication and addition) to calculate total slices.

[Step-by-Step Solution]
Step 1: Calculate slices from large pizzas: 2 pizzas × 16 slices/pizza = 32 slices
Step 2: Calculate slices from small pizzas: 2 pizzas × 8 slices/pizza = 16 slices
Step 3: Add both totals: 32 slices + 16 slices = 48 slices

The answer is \boxed{48}
```

> **참고**: 이 예시는 가상입니다. 실제 Qwen3-4B 실행에서는 모든 453건이 Case A(437) 또는 Case B(16)로 처리되어 Case C가 발생하지 않았습니다.

---

## 4. Phase 3: Instructional Delivery

Phase 2에서 생성된 SFT 데이터로 Student 모델을 Fine-tuning하고, 평가 데이터셋에서 성능을 측정합니다.

### SFT 데이터 형식

```json
{
  "instruction": "[Original Instruction + SCAFFOLDING_SYSTEM_PROMPT (동적 결합)]",
  "input": "Question: [문제 텍스트]",
  "output": "[Case별 SFT 응답]",
  "metadata": {
    "id": "gsm8k_train_XXX",
    "sft_case": "A|B|C",
    "ground_truth": "[정답]"
  }
}
```

> **instruction 동적 결합**: SFT 데이터 생성 시 `_create_sft_entry()`에서 `original_instruction + "\n\n" + SCAFFOLDING_SYSTEM_PROMPT`를 런타임에 결합합니다.

### Case별 SFT 엔트리 예시

#### Case A 엔트리 (gsm8k_train_0)

```json
{
  "instruction": "You are a helpful math assistant.\nSolve this mathematical problem step by step...\n\nThe purpose of your response is to demonstrate the attainment of the Instructional Goal: The model will solve complex mathematical problems by analyzing relationships between quantities, performing multi-step calculations, and applying mathematical reasoning to arrive at accurate solutions in real-world contexts.\n\nYou must adhere to the specific performance procedures...\n\n[Instructional Analysis]\n### Instructional Analysis Results\n...\n\n[Instructions]\n1. Identify which skills...\n...\n\n[Output Format]\n- Instructional goal alignment: ...\n- Step-by-step reasoning: ...\n- Final answer: \"The answer is \\boxed{your final answer}\"",
  "input": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "output": "- Instructional goal alignment: This solution demonstrates the instructional goal by analyzing the relationship between the number of clips sold in April and May...\n- Step-by-step reasoning:\n  1. **Identify the quantities and relationships**: In April, Natalia sold 48 clips...\n  2. **Calculate the number of clips sold in May**: 48 / 2 = 24\n  3. **Add the number of clips sold**: 48 + 24 = 72\n  4. **Verify the solution**: The calculation is accurate...\n- Final answer: \"The answer is \\boxed{72}\"",
  "metadata": {
    "id": "gsm8k_train_0",
    "sft_case": "A",
    "ground_truth": "72"
  }
}
```

> **Case A**: PO 충족 후 Teacher Positive Feedback → Student Self-Refinement → Refined Response를 SFT output으로 사용.

#### Case B 엔트리 (gsm8k_train_4)

```json
{
  "instruction": "Original Instruction + SCAFFOLDING_SYSTEM_PROMPT (동적 결합) — Case A와 동일한 구조",
  "input": "Question: James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?",
  "output": "- Instructional goal alignment: This solution demonstrates the instructional goal by analyzing relationships between quantities (letters, pages, and weeks), performing multi-step calculations...\n- Step-by-step reasoning:\n  1. **Identify the quantities and their relationships**: James writes 3 pages per letter, 2 different letters, twice a week.\n  2. **Determine the total number of letters written per week**: 2 letters × 2 = 4 letters/week\n  3. **Determine the total number of letters written in a year**: 4 letters/week × 52 weeks = 208 letters\n  4. **Determine the total number of pages written in a year**: 208 letters × 3 pages/letter = 624 pages\n- Final answer: \"The answer is \\boxed{624}\"",
  "metadata": {
    "id": "gsm8k_train_4",
    "sft_case": "B",
    "ground_truth": "624"
  }
}
```

> **Case B**: Scaffolding을 통해 PO를 충족한 후 Teacher Positive Feedback → Student Self-Refinement → Refined Response를 SFT output으로 사용.

#### Case C 엔트리 (가상 예시)

```json
{
  "instruction": "Original Instruction + SCAFFOLDING_SYSTEM_PROMPT (동적 결합) — Case A와 동일한 구조",
  "input": "Question: Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
  "output": "[Understanding the Problem]\nAlbert buys 2 large pizzas (16 slices each) and 2 small pizzas (8 slices each). The problem asks for the total number of slices he eats if he eats all of them.\n\n[Key Concepts Applied]\nThis problem requires arithmetic operations (multiplication and addition) to calculate total slices.\n\n[Step-by-Step Solution]\nStep 1: Calculate slices from large pizzas: 2 pizzas × 16 slices/pizza = 32 slices\nStep 2: Calculate slices from small pizzas: 2 pizzas × 8 slices/pizza = 16 slices\nStep 3: Add both totals: 32 slices + 16 slices = 48 slices\n\nThe answer is \\boxed{48}",
  "metadata": {
    "id": "hypothetical",
    "sft_case": "C",
    "ground_truth": "48"
  }
}
```

> **Case C (가상)**: Teacher가 교육적 풀이를 평문 텍스트로 생성. 실제 Qwen3-4B 실행에서는 Case C가 발생하지 않았음.

### Case별 SFT 응답 소스

| Case | SFT `output` 소스 | 특징 |
|------|-------------------|------|
| **A** | Student Self-Refined 응답 | PO 충족 후 Teacher Positive Feedback 기반 Self-Refinement |
| **B** | Student Self-Refined 응답 | Scaffolding + PO 충족 후 Teacher Positive Feedback 기반 Self-Refinement |
| **C** | Teacher 최종 풀이 (평문 텍스트) | 학생 약점을 보완한 교육적 정답 풀이 |

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
| **전체 SFT 데이터** | 7,473 |
| **Scaffolding 처리 건수** | 453 |
| **처리 완료** | 453 (100%) |

> **참고**: 전체 GSM8K train 데이터(7,473건) 중 453건에 대해 Phase 2 Scaffolding 상호작용이 수행되었습니다. 나머지 데이터는 Enhanced Instruction만 적용되어 SFT 데이터로 사용됩니다.

### Case 분포 (Scaffolding 453건 기준)

| Case | 건수 | 비율 | 설명 |
|------|------|------|------|
| **A** | 437 | 96.5% | 1회차 성공 |
| **B** | 16 | 3.5% | 2~5회차 성공 |
| **C** | 0 | 0.0% | 최대 반복 후 실패 |

```
Case A █████████████████████████████████████████████████ 96.5%
Case B ██                                               3.5%
Case C                                                   0.0%
```

### Case B 상세 분포

| Iterations | 건수 | 비율 |
|-----------|------|------|
| 2회차 성공 | 12 | 75.0% |
| 3회차 성공 | 3 | 18.8% |
| 4회차 성공 | 1 | 6.2% |

> **해석**: Qwen3-4B 모델은 GSM8K 문제의 96.5%를 Task Analysis 기반 Enhanced Instruction만으로 1회에 해결했습니다. 3.5%만이 Teacher의 Scaffolding을 필요로 했으며, 모든 문제가 최대 4회 이내에 해결되어 Case C는 발생하지 않았습니다.
>
> Case B의 주된 실패 원인은 **문제 해석 오류**(예: "twice a week" 미반영)와 **계산 실수**로, Scaffolding을 통해 빠르게 교정되었습니다.

---

## 부록 A: 주요 개념 정리

### HOT vs LOT Scaffolding

| 유형 | 대상 인지 수준 | 제공 내용 |
|------|--------------|----------|
| **HOT** (High-Order Thinking) | 분석/평가/창조 | `strategy_suggestion`, `partial_example`, `key_attention_points` |
| **LOT** (Low-Order Thinking) | 기억/이해/적용 | `missed_concept`, `brief_explanation` |

### Scaffolding Artifact 및 피드백

- 각 iteration에서 생성된 Scaffolding Artifact가 **누적** 저장됩니다
- `SCAFFOLDING_ARTIFACT_PROMPT`는 **구조화된 마크다운**으로 출력합니다 (JSON 미사용)
- 출력은 `[Instructional Goal]`, `[Scaffolding for Task [N]]`, `[Feedback]`, `[Iteration Summary]` 섹션으로 구성됩니다
- `[Feedback]` 섹션은 (1) 오류 분석, (2) 개선 방향, (3) 검증 단계를 하나의 자연스러운 서술로 통합합니다
- Student는 재응답 시 **전체 Scaffolding Artifact 텍스트**를 참조합니다 (HOT/LOT 스캐폴딩, 전략, Feedback 등 모든 교수적 지원 포함)

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

모든 Teacher 프롬프트는 `_SYSTEM_PROMPT` / `_USER_PROMPT` 쌍으로 분리되어 있습니다.

| 상수명 | 소스 파일 | 용도 | Phase/Step | 메시지 역할 | 출력 형식 |
|--------|----------|------|-----------|------------|----------|
| `INSTRUCTIONAL_GOAL_SYSTEM_MESSAGE` | `prompts/design_prompts.py` | 교수설계 전문가 역할 설정 | Phase 1 / Step 0 | System | — |
| `INSTRUCTIONAL_GOAL_PROMPT` | `prompts/design_prompts.py` | 데이터셋 분석 → Instructional Goal 도출 | Phase 1 / Step 0 | User | JSON |
| `INSTRUCTIONAL_ANALYSIS_SYSTEM_PROMPT` | `prompts/design_prompts.py` | 교수 분석 전문가 역할 설정 | Phase 1 / Step 2 | System | — |
| `INSTRUCTIONAL_ANALYSIS_USER_PROMPT` | `prompts/design_prompts.py` | Learning Objective → Task Analysis Tree 분해 | Phase 1 / Step 2 | User | Text (Tree) |
| `PERFORMANCE_OBJECTIVES_SYSTEM_PROMPT` | `prompts/design_prompts.py` | PO 생성 전문가 역할 설정 | Phase 1 / Step 3 | System | — |
| `PERFORMANCE_OBJECTIVES_USER_PROMPT` | `prompts/design_prompts.py` | Task Analysis → Performance Objectives 생성 | Phase 1 / Step 3 | User | JSON |
| `SCAFFOLDING_SYSTEM_PROMPT` | `prompts/learning_prompts.py` | Student 문제 해결 시스템 프롬프트 | Phase 2 / Step 1 | System | Text |
| `TEACHER_INTERVENTION_SYSTEM_PROMPT` | `prompts/learning_prompts.py` | Teacher 평가 역할 설정 | Phase 2 / Step 2 | System | — |
| `TEACHER_INTERVENTION_USER_PROMPT` | `prompts/learning_prompts.py` | Teacher PO 평가 (평가 전용) | Phase 2 / Step 2 | User | JSON |
| `SCAFFOLDING_ARTIFACT_SYSTEM_PROMPT` | `prompts/learning_prompts.py` | Scaffolding 생성 역할 설정 | Phase 2 / Step 3 | System | — |
| `SCAFFOLDING_ARTIFACT_USER_PROMPT` | `prompts/learning_prompts.py` | 미충족 PO별 HOT/LOT Scaffolding + Feedback 생성 | Phase 2 / Step 3 | User | Structured Text |
| `STUDENT_FEEDBACK_RESPONSE_PROMPT` | `prompts/learning_prompts.py` | Student: `dataset_prompt` + `SCAFFOLDING_SYSTEM_PROMPT` + Scaffolding Artifact 기반 재응답 | Phase 2 / Step 4 | System | Text |
| `TEACHER_POSITIVE_FEEDBACK_SYSTEM_PROMPT` | `prompts/learning_prompts.py` | Positive Feedback 역할 설정 | Phase 2 / Step 5 (Case A/B) | System | — |
| `TEACHER_POSITIVE_FEEDBACK_USER_PROMPT` | `prompts/learning_prompts.py` | PO 충족 후 강점 + 개선점 피드백 생성 | Phase 2 / Step 5 (Case A/B) | User | Structured Text |
| `STUDENT_SELF_REFINEMENT_PROMPT` | `prompts/learning_prompts.py` | Positive Feedback 기반 응답 개선 | Phase 2 / Step 5 (Case A/B) | System | Text |
| `TEACHER_FINAL_SOLUTION_SYSTEM_PROMPT` | `prompts/learning_prompts.py` | Final Solution 역할 설정 | Phase 2 / Step 5 (Case C) | System | — |
| `TEACHER_FINAL_SOLUTION_USER_PROMPT` | `prompts/learning_prompts.py` | 최대 반복 실패 후 교육적 풀이 생성 | Phase 2 / Step 5 (Case C) | User | Text (평문) |
