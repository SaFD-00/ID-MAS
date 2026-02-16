# ID-MAS 동작 예시: Phase 1~3 상세 설명

> GSM8K 데이터셋 + Qwen3-4B 모델 기반 실제 실행 로그를 통해 ID-MAS의 3-Phase 파이프라인이 어떻게 작동하는지 Case별로 설명합니다.
> **Student와 Teacher 모두 Qwen3-4B 모델**을 사용하며, 각 Step에서 LLM에 입력되는 실제 프롬프트를 포함하여 파이프라인 동작을 프롬프트 수준까지 설명합니다.

---

## 목차

1. [전체 파이프라인 개요](#1-전체-파이프라인-개요)
2. [Phase 1: Instructional Design](#2-phase-1-instructional-design)
   - [Step 0: Instructional Goal 생성](#step-0-instructional-goal-생성)
   - [Step 1: Learning Objective 설정](#step-1-learning-objective-설정)
   - [Step 2: Instructional Analysis](#step-2-instructional-analysis)
   - [Step 3: Performance Objectives 생성](#step-3-performance-objectives-생성)
   - [Enhanced Data 생성](#enhanced-data-생성)
3. [Phase 2: Adaptive Scaffolding](#3-phase-2-adaptive-scaffolding)
   - [Case A: Independent Performance Mastery 예시](#case-a-independent-performance-mastery-독립적-수행-숙달)
   - [Case B: Scaffolded & Coached Mastery 예시](#case-b-scaffolded--coached-mastery-스캐폴딩-기반-숙달)
   - [Case C: Teacher Modeling Distillation 예시](#case-c-teacher-modeling-distillation-교사-모델링-증류)
4. [Phase 3: Instructional Delivery (SFT)](#4-phase-3-instructional-delivery)
5. [부록 A: 주요 개념 정리](#부록-a-주요-개념-정리)
6. [부록 B: 프롬프트 상수 참조 테이블](#부록-b-프롬프트-상수-참조-테이블)

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
  ├── Step 1: Student Initial Response (Student 초기 응답)
  ├── Step 2: Teacher PO Evaluation (평가 전용) → 성공이면 Case A: Independent Performance Mastery/Case B: Scaffolded & Coached Mastery
  ├── Step 3: Scaffolding Artifact Generation (Scaffolding Artifact + 서술형 피드백 생성)
  ├── Step 4: Student Reattempt (Teacher 피드백 참조)
  ├── (Step 2~4 반복, 최대 N회 — `--max-iterations`로 설정, 기본값 5)
  ├── Step 5a-1: Teacher Positive Reinforcement (Case A: Independent Performance Mastery/Case B: Scaffolded & Coached Mastery) — 강점 + 개선점
  ├── Step 5a-2: Student Feedback-Driven Elaboration (Case A: Independent Performance Mastery/Case B: Scaffolded & Coached Mastery) — 응답 개선
  ├── Step 5b: Teacher Final Solution (Case C: Teacher Modeling Distillation만) — 교육적 풀이 평문 텍스트 출력
  └── Step 6: SFT Data Generation (SFT 데이터 생성)
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
  "instructional_goal": "The model will solve complex mathematical word problems by applying arithmetic operations, logical reasoning, and mathematical concepts to determine the correct numerical answer in real-world contexts.",
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

**User Message** (`INSTRUCTIONAL_GOAL_USER_PROMPT`):

> Placeholder: `{sample_count}` → 20, `{train_data}` → `utils/prompt_helpers.py`의 `format_samples_for_prompt()` 출력

```
You are given a sample of items representing a specific task domain. These items are used to evaluate the student you are teaching. Your mission is to analyze the entire test set and determine a core instructional requirement that defines the instructional goal.

## Instructions
1. Analyze the input test items to identify the ultimate action the model must demonstrate to provide appropriate answers. Focus on observable and transferable results.
2. Identify the highest cognitive level required by the specific nature of the given data, based on the framework of Bloom's Taxonomy.
3. Avoid describing individual test items or listing sub-skills, learning steps, or evaluation criteria.
4. Focus exclusively on deriving a single, comprehensive Instructional Goal that encapsulates the core requirement across the entire set.


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
Instructional Goal: The model will solve complex mathematical word problems by applying
              arithmetic operations, logical reasoning, and mathematical concepts to
              determine the correct numerical answer in real-world contexts. (Apply)

 ├── Understand Problem Context (Understand)
 │   ├── Identify key information in the problem (Remember)
 │   └── Determine relevant mathematical concepts (Understand)
 ├── Apply Arithmetic Operations (Apply)
 │   ├── Perform multi-step arithmetic calculations (Apply)
 │   └── Use appropriate operations (Add, Subtract, Multiply, Divide) (Apply)
 ├── Apply Logical Reasoning (Apply)
 │   ├── Analyze relationships between variables (Analyze)
 │   └── Make logical deductions (Apply)
 └── Verify Final Answer (Evaluate)
     ├── Substitute answer back into problem constraints (Evaluate)
     └── Check for consistency with all problem conditions (Evaluate)
```

#### 실제 프롬프트

**System Message** (`INSTRUCTIONAL_ANALYSIS_SYSTEM_PROMPT`):
```
You are an instructional design expert. Perform the Instructional Analysis step of the Dick & Carey model for the learning objective provided below.
```

**User Message** (`INSTRUCTIONAL_ANALYSIS_USER_PROMPT`):

> Placeholder: `{learning_objective}` → 실제 GSM8K Instructional Goal

```
[Learning objective]: The model will solve complex mathematical word problems by applying arithmetic operations, logical reasoning, and mathematical concepts to determine the correct numerical answer in real-world contexts.

[Instructions]
Perform the Instructional Analysis and construct a hierarchical structure in the form of: Instructional Goal → Subskills → Subtasks.
Present the instructional analysis results as a text-based tree structure.
Write all skill statements concisely using an action verb + object format.
Include only the minimum number of Subskills and Subtasks that are essential to achieving the Instructional Goal. For every function or sub-function, indicate the learning outcome based on Bloom's revised Taxonomy: Remember / Understand / Apply / Analyze / Evaluate / Create.
The analysis MUST include a "Verify Final Answer" subskill (Evaluate level) that requires cross-checking the computed result against all problem constraints. This is distinct from computational or reasoning subskills — it specifically requires substituting the answer back into the original problem conditions to confirm correctness and consistency.
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

> **핵심 변경**: `INSTRUCTIONAL_ANALYSIS_USER_PROMPT`에 "Verify Final Answer" subskill 필수 포함 규칙이 추가되었습니다. 이는 최종 답의 정확성 검증을 교수분석 수준에서 명시적으로 요구합니다.

### Step 3: Performance Objectives 생성

행동(Behavior), 조건(Condition), 기준(Criterion)을 하나의 문장으로 통합한 수행목표를 각 Subskill에 대해 생성합니다. 이 PO들이 **Phase 2에서 학생 응답 평가의 기준**이 됩니다.

**생성 결과 (6개 PO):**

| # | Target | Performance Objective |
|---|--------|----------------------|
| 1 | Instructional Goal | Given a complex mathematical word problem, the model will solve it by applying arithmetic operations, logical reasoning, and mathematical concepts to determine the correct numerical answer in real-world contexts with 100% accuracy. |
| 2 | Understand Problem Context | Given a mathematical word problem, the model will identify key information in the problem by extracting relevant numerical data, units, and contextual clues with 100% accuracy. |
| 3 | Determine relevant mathematical concepts | Given a mathematical word problem, the model will determine relevant mathematical concepts by identifying the appropriate operations, formulas, and principles needed to solve the problem with 100% accuracy. |
| 4 | Apply Arithmetic Operations | Given a mathematical word problem requiring multi-step arithmetic calculations, the model will perform the calculations using appropriate operations (Add, Subtract, Multiply, Divide) with 100% accuracy. |
| 5 | Apply Logical Reasoning | Given a mathematical word problem, the model will analyze relationships between variables and make logical deductions to ensure the solution is consistent with the problem's structure and constraints with 100% accuracy. |
| 6 | Verify Final Answer | Given a mathematical word problem, the model will substitute the computed final answer back into the original problem constraints, check for consistency with all problem conditions, and ensure the answer is correct and consistent with every condition stated in the problem with 100% accuracy. |

#### 실제 프롬프트

**System Message** (`PERFORMANCE_OBJECTIVES_SYSTEM_PROMPT`):
```
You are an instructional designer specializing in the Dick and Carey instructional design model, and a researcher in LLM learning methodologies.
Based on the provided Instructional Goal and Instructional Analysis Result, generate a set of Performance Objectives that will serve as the criteria for evaluating the observable performance within the LLM's reasoning process.
Specifically, they should be created using information from the learning outcomes identified in the Instructional Analysis Results.
```

**User Message** (`PERFORMANCE_OBJECTIVES_USER_PROMPT`):

> Placeholder: `{instructional_analysis}` → 실제 GSM8K Task Analysis Tree

```
[Input Data]
Instructional Analysis Result: ### Instructional Analysis Results
Instructional Goal: The model will solve complex mathematical word problems by applying
              arithmetic operations, logical reasoning, and mathematical concepts to
              determine the correct numerical answer in real-world contexts. (Apply)

 ├── Understand Problem Context (Understand)
 │   ├── Identify key information in the problem (Remember)
 │   └── Determine relevant mathematical concepts (Understand)
 ├── Apply Arithmetic Operations (Apply)
 │   ├── Perform multi-step arithmetic calculations (Apply)
 │   └── Use appropriate operations (Add, Subtract, Multiply, Divide) (Apply)
 ├── Apply Logical Reasoning (Apply)
 │   ├── Analyze relationships between variables (Analyze)
 │   └── Make logical deductions (Apply)
 └── Verify Final Answer (Evaluate)
     ├── Substitute answer back into problem constraints (Evaluate)
     └── Check for consistency with all problem conditions (Evaluate)

[Instructions]
For each Subskills and Subtasks in the instructional analysis, you must create at least one Performance Objective. You can create multiple performance objectives for subskills or subtasks that have more than one requirement.
Every Performance Objective must include all three components—Behavior, Condition, and Criterion—and each component must be explicitly stated in one sentence.
- Behavior: This is a description of LLM's intellectual skill including actions, content, and concepts.
- Condition: This is a description of the tools and resources that will be available to the learner when performing the skill. Write the conditions based solely on the data given in the problem or generated during the reasoning process. And it should always begin with 'given ~'.
- Criterion: This is a description of acceptable performance of the skill. The Criterion component must be tailored to the nature of the task: for tasks with correct answers, it must include a clear and measurable standard such as accuracy requirements, acceptable error ranges, or the number of correct responses; whereas for tasks with no single correct answer, it must specify the information or features that must be present for an acceptable response. Furthermore, these criteria must be formulated to evaluate the observable reasoning process within a single problem-solving task.
Additionally, at least one Performance Objective MUST explicitly target the "Verify Final Answer" subskill from the Instructional Analysis. This PO must require the student to cross-check the computed final answer against ALL original problem constraints (e.g., substituting the answer back into the original conditions, verifying boundary values, confirming unit consistency, or checking that the selected option satisfies all given criteria). The Criterion for this PO must state that the final answer must be correct and consistent with every condition stated in the problem.
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

> **핵심 변경**: `PERFORMANCE_OBJECTIVES_USER_PROMPT`에 "Verify Final Answer" PO 필수 규칙이 추가되었습니다. 이 PO는 최종 답을 원래 문제 조건에 대입하여 정확성과 일관성을 검증하는 것을 요구합니다.

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
    "instructional_goal": "The model will solve complex mathematical word problems by applying arithmetic operations, logical reasoning, and mathematical concepts to determine the correct numerical answer in real-world contexts.",
    "task_analysis": "### Instructional Analysis Results\nInstructional Goal: ..."
  }
}
```

> **핵심**: `instruction`은 원본 그대로 유지되며, Phase 2에서 `LEARNING_TASK_SYSTEM_PROMPT`와 동적으로 결합됩니다. SFT 데이터 생성 시에도 `original_instruction + LEARNING_TASK_SYSTEM_PROMPT`로 동적 결합합니다.

---

## 3. Phase 2: Adaptive Scaffolding

Phase 2는 **각 문제별로** 실행됩니다. 교사-학생 반복 상호작용을 통해 SFT 학습 데이터를 생성합니다.

### Case 분류 기준

| 분류 명칭 | 조건 | SFT 응답 소스 |
|------|------|--------------|
| **Case A: Independent Performance Mastery** | 1회차에 모든 PO 충족 | Teacher Positive Reinforcement → Student Feedback-Driven Elaboration → Refined Response |
| **Case B: Scaffolded & Coached Mastery** | 2~5회차에 모든 PO 충족 | Teacher Positive Reinforcement → Student Feedback-Driven Elaboration → Refined Response |
| **Case C: Teacher Modeling Distillation** | 5회 반복 후에도 PO 미충족 | Teacher가 정답 기반 교육적 풀이 생성 (평문 텍스트) |

---

### Case A: Independent Performance Mastery: 독립적 수행 숙달

> **문제 ID**: `gsm8k_train_0`
> **문제**: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
> **정답**: 72

#### Case A Step 1: Student Initial Response

Student 모델이 Enhanced Instruction(Task Analysis 포함)을 참고하여 체계적 풀이를 생성합니다.

##### 실제 프롬프트

**System Message** (`LEARNING_TASK_SYSTEM_PROMPT`):

> Placeholder: `{instructional_goal}` → GSM8K Instructional Goal, `{task_analysis}` → Task Analysis Tree

```
The purpose of your response is to demonstrate the attainment of the Instructional Goal: The model will solve complex mathematical word problems by applying arithmetic operations, logical reasoning, and mathematical concepts to determine the correct numerical answer in real-world contexts.

You must adhere to the specific performance procedures and required knowledge/skills outlined in the Instructional Analysis results below. Ensure that your solution describes the full reasoning process using all provided steps and resources before arriving at the final answer.

[Instructional Analysis]
### Instructional Analysis Results
Instructional Goal: The model will solve complex mathematical word problems by applying
              arithmetic operations, logical reasoning, and mathematical concepts to
              determine the correct numerical answer in real-world contexts. (Apply)

 ├── Understand Problem Context (Understand)
 │   ├── Identify key information in the problem (Remember)
 │   └── Determine relevant mathematical concepts (Understand)
 ├── Apply Arithmetic Operations (Apply)
 │   ├── Perform multi-step arithmetic calculations (Apply)
 │   └── Use appropriate operations (Add, Subtract, Multiply, Divide) (Apply)
 ├── Apply Logical Reasoning (Apply)
 │   ├── Analyze relationships between variables (Analyze)
 │   └── Make logical deductions (Apply)
 └── Verify Final Answer (Evaluate)
     ├── Substitute answer back into problem constraints (Evaluate)
     └── Check for consistency with all problem conditions (Evaluate)

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
- Instructional goal alignment: This solution demonstrates the instructional goal by applying
  arithmetic operations (addition and division), understanding the problem context, and verifying
  the final answer through logical reasoning and consistency checks.

- Step-by-step reasoning:

1. **Identify key information in the problem:**
   - Natalia sold clips to 48 friends in April.
   - She sold half as many clips in May as she did in April.
   - We are asked to find the total number of clips sold in April and May.

2. **Determine relevant mathematical concepts:**
   - Addition: To find the total number of clips sold.
   - Division: To find half of a number (for May's sales).
   - Logical reasoning: To understand the relationship between the number of clips sold in April and May.

3. **Apply arithmetic operations:**
   - Clips sold in April: 48
   - Clips sold in May: $ \frac{1}{2} \times 48 = 24 $
   - Total clips sold: $ 48 + 24 = 72 $

4. **Analyze relationships between variables:**
   - The number of clips sold in May is half of the number sold in April.
   - This relationship is used to calculate May's sales before summing both months.

5. **Verify final answer:**
   - Total clips sold in April: 48
   - Total clips sold in May: 24
   - Total: $ 48 + 24 = 72 $
   - The answer is consistent with the problem's conditions.

- **Final answer:** "The answer is \boxed{72}"
```

**추출된 답**: 72 (정답과 일치)

#### Case A Step 2: Teacher PO Evaluation

Teacher가 Performance Objectives 기준으로 학생 응답을 **평가만** 수행합니다. 피드백은 생성하지 않습니다.

##### 실제 프롬프트

**System Message** (`FORMATIVE_ASSESSMENT_SYSTEM_PROMPT`):
```
You are a teacher supporting the learning of a student.

Your role is to evaluate the student's response against the established performance objectives. You must monitor the student's reasoning steps to ensure they meet the performance objectives.
```

**User Message** (`FORMATIVE_ASSESSMENT_USER_PROMPT`):

> Placeholder: `{problem_text}` → 문제 원문, `{student_response}` → 위 Student 응답, `{performance_objectives}` → Phase 1에서 생성된 6개 PO JSON, `{ground_truth}` → "72"

```
[Input Data]
- Problem: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
- Student response: [위 Student 응답 전체]
- Performance objectives: [6개 PO JSON 전체]
- Ground truth (FOR REFERENCE ONLY - DO NOT REVEAL): The answer is \boxed{72}

[Instructions]
Evaluate the student model's response according to the following rules.
1. Assess student performance according to the performance objectives. Use the criterion embedded in each performance objective as the evaluation standard. Do not reveal correct answers or model solutions.
2. Analyze the student response and determine which performance objectives are satisfied and which are not. All judgments must be grounded in observable reasoning behaviors in the student response, such as how claims are justified, how relationships are analyzed, or how judgments are formed. Avoid vague or abstract evaluations.
3. For each PO, write a "reasoning" that explains your evaluation:
   - State WHY you determined the objective is satisfied or not, citing specific evidence from the student's response (e.g., which reasoning steps, strategies, expressions, or omissions led to your judgment).
   - Then describe HOW the student could improve or elaborate: if satisfied, suggest ways to strengthen or deepen the demonstrated reasoning; if NOT satisfied, explain what specific changes or additions would help meet the objective.
4. CRITICAL — Answer Verification Rule: For any Performance Objective related to answer verification, solution validation, or final answer correctness, you MUST compare the student's final answer against the ground truth provided above. If the student's final answer does NOT match the ground truth, that PO MUST be marked as is_satisfied: false, regardless of how well the intermediate reasoning steps appear. Do NOT rationalize an incorrect final answer as satisfying an answer verification objective.

[Output Format - JSON]
{
  "performance_evaluation": [
    {
      "objective_content": "Copy the performance_objective field from performance objectives VERBATIM",
      "is_satisfied": true or false,
      "reasoning": "WHY: Evidence-based explanation of why this objective is/is not satisfied, referencing the student's actual response. HOW: Specific suggestions for improvement or elaboration."
    }
  ]
}

Output ONLY valid JSON.
```

> **핵심 변경**: `FORMATIVE_ASSESSMENT_USER_PROMPT`에 **CRITICAL Answer Verification Rule**이 추가되었습니다. 최종 답이 ground truth와 불일치하면, 답 검증 관련 PO는 반드시 `is_satisfied: false`로 평가해야 합니다.

##### Teacher 평가 결과

```json
{
  "performance_evaluation": [
    {
      "objective_content": "Given a complex mathematical word problem, the model will solve it by applying arithmetic operations, logical reasoning, and mathematical concepts to determine the correct numerical answer in real-world contexts with 100% accuracy.",
      "is_satisfied": true,
      "reasoning": "The student correctly applied arithmetic operations (division and addition) and logical reasoning to solve the problem. The final answer of 72 matches the ground truth, demonstrating accurate problem-solving in a real-world context. HOW: The student could further elaborate on the real-world implications of the answer, such as how many clips Natalia sold on average per month."
    },
    {
      "objective_content": "Given a mathematical word problem, the model will identify key information in the problem by extracting relevant numerical data, units, and contextual clues with 100% accuracy.",
      "is_satisfied": true,
      "reasoning": "The student clearly identified the key information: 48 clips sold in April, half as many in May, and the need to find the total. The student correctly extracted numerical data and contextual clues. HOW: The student could improve by explicitly stating the units (e.g., 'clips') in the explanation."
    },
    {
      "objective_content": "Given a mathematical word problem, the model will determine relevant mathematical concepts by identifying the appropriate operations, formulas, and principles needed to solve the problem with 100% accuracy.",
      "is_satisfied": true,
      "reasoning": "The student correctly identified the relevant mathematical concepts: addition and division. The student explained how these operations apply to the problem. HOW: The student could further elaborate on why these operations are appropriate in this context."
    },
    {
      "objective_content": "Given a mathematical word problem requiring multi-step arithmetic calculations, the model will perform the calculations using appropriate operations (Add, Subtract, Multiply, Divide) with 100% accuracy.",
      "is_satisfied": true,
      "reasoning": "The student correctly performed the calculations: 48 divided by 2 equals 24, and 48 plus 24 equals 72. The calculations are accurate. HOW: The student could provide more detailed steps or explain the calculation process in more depth."
    },
    {
      "objective_content": "Given a mathematical word problem, the model will analyze relationships between variables and make logical deductions to ensure the solution is consistent with the problem's structure and constraints with 100% accuracy.",
      "is_satisfied": true,
      "reasoning": "The student correctly analyzed the relationship between the number of clips sold in April and May, and made logical deductions to ensure the solution was consistent with the problem's structure. HOW: The student could further explain how the relationship between the two months supports the final answer."
    },
    {
      "objective_content": "Given a mathematical word problem, the model will substitute the computed final answer back into the original problem constraints, check for consistency with all problem conditions, and ensure the answer is correct and consistent with every condition stated in the problem with 100% accuracy.",
      "is_satisfied": true,
      "reasoning": "The student verified the final answer by substituting it back into the problem and checking for consistency. The final answer of 72 is correct and matches the ground truth. HOW: The student could further explain how each step of the verification process confirms the answer's correctness."
    }
  ]
}
```

**모든 PO 충족 (6/6)** → 반복 종료 → **Case A: Independent Performance Mastery 확정**

#### Case A Step 5a-1: Teacher Positive Reinforcement

모든 PO가 충족되었으므로 Teacher가 학생 응답의 강점과 개선점을 분석하는 Positive Reinforcement를 생성합니다.

##### 실제 프롬프트

**System Message** (`POSITIVE_REINFORCEMENT_SYSTEM_PROMPT`):
```
You are a teacher providing constructive feedback to strengthen a student's already satisfactory response.
```

**User Message** (`POSITIVE_REINFORCEMENT_USER_PROMPT`):

> Placeholder: `{problem_text}` → 문제 원문, `{student_response}` → Student 응답, `{po_evaluation}` → PO 평가 JSON

```
[Problem]
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

[Student's Response]
[위 Student 응답 전체]

[Performance Objectives Evaluation]
[6개 PO 평가 JSON — 모두 is_satisfied: true]

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

##### Teacher Positive Reinforcement 결과

```
[Strengths]
- PO 1: The student clearly demonstrated the instructional goal by applying arithmetic operations
  (addition and division) and logical reasoning to solve the problem, resulting in the correct
  final answer of 72.
- PO 2: The student effectively identified key information from the problem, including the number
  of clips sold in April, the relationship between April and May sales, and the need to find the
  total, showing strong comprehension of the problem context.

[Enhancement Suggestions]
1. Add more explicit connections between the steps, such as explaining why division is used to find
   May's sales and how addition is used to find the total.
2. Provide more detailed explanations of the mathematical concepts, such as why addition and division
   are appropriate in this context.
3. Elaborate on the real-world implications of the answer, such as how many clips Natalia sold on
   average per month.

[Integration Guidance]
To incorporate these improvements naturally, the student can add a brief explanation after each
calculation step, such as "We use division to find half of 48 because the problem states she sold
half as many in May." They can also add a sentence at the end explaining the real-world relevance,
like "Natalia sold an average of 36 clips per month, which helps us understand her sales pattern."
These additions will enhance clarity without altering the core logic or final answer.
```

#### Case A Step 5a-2: Student Feedback-Driven Elaboration

Student가 Teacher의 Positive Reinforcement를 참조하여 응답을 정교화합니다.

##### 실제 프롬프트

**System Message** (`FEEDBACK_DRIVEN_ELABORATION_SYSTEM_PROMPT`):

> Placeholder: `{scaffolding_system_prompt}` → `LEARNING_TASK_SYSTEM_PROMPT` (채워진 상태)

```
[LEARNING_TASK_SYSTEM_PROMPT — Task Analysis 포함]

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

**User Message** (`FEEDBACK_DRIVEN_ELABORATION_USER_PROMPT`):

> Placeholder: `{positive_feedback}` → Teacher의 Positive Feedback 텍스트, `{problem_text}` → 문제 원문

```
[Teacher's Feedback on Your Response]
Your teacher has evaluated your response and confirmed that it meets all performance objectives.
The following feedback highlights your strengths and suggests ways to further improve your response:

[Strengths]
- PO 1: The student clearly demonstrated the instructional goal by applying arithmetic operations...
[... 전체 Positive Feedback ...]

[Problem]
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
```

##### Feedback-Driven Elaboration 응답 결과

```
- Instructional goal alignment: This solution demonstrates the instructional goal by applying
  arithmetic operations (addition and division) and logical reasoning to solve the problem,
  while also explaining the mathematical concepts and real-world implications of the answer.

- Step-by-step reasoning:
  1. **Identify key information**: The problem states that Natalia sold 48 clips in April and
     half as many in May. We are asked to find the total number of clips sold in both months.
  2. **Determine relevant mathematical concepts**: To find the number of clips sold in May, we
     need to calculate half of 48, which involves division. To find the total number of clips
     sold, we add the number of clips sold in April and May, which involves addition.
  3. **Apply arithmetic operations**:
     - To find the number of clips sold in May:
       $$
       \text{May sales} = \frac{48}{2} = 24
       $$
       We use division to find half of 48 because the problem states that Natalia sold half as
       many clips in May as she did in April.
     - To find the total number of clips sold:
       $$
       \text{Total sales} = 48 + 24 = 72
       $$
       We use addition to combine the sales from both months.
  4. **Apply logical reasoning**: The problem provides a clear relationship between April and May
     sales (May is half of April), which allows us to calculate May's sales first. Then, we
     combine both amounts to find the total.
  5. **Verify final answer**: The calculations are consistent with the problem's constraints.
     April sales (48) plus May sales (24) equals 72, which matches the total.
  6. **Real-world implications**: Natalia sold an average of 36 clips per month (72 ÷ 2), which
     helps us understand her sales pattern over the two months.

- Final answer: "The answer is \boxed{72}"
```

#### Case A Step 6: SFT 데이터 생성

Case A: Independent Performance Mastery는 Feedback-Driven Elaboration 후의 **Refined Response**를 SFT 학습 데이터로 사용합니다.

```json
{
  "instruction": "[Original Instruction + LEARNING_TASK_SYSTEM_PROMPT (동적 결합)]",
  "input": "Question: Natalia sold clips to 48 of her friends in April...",
  "output": "[Student의 Feedback-Driven Elaboration 응답]",
  "metadata": {
    "id": "gsm8k_train_0",
    "sft_case": "case_a_independent_performance_mastery",
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

**Case A: Independent Performance Mastery 흐름 요약:**
```
Step 1: Student(72✓) → Step 2: Teacher(6/6 PO ✓) → Case A 확정
     ↓
Step 5a-1: Teacher Positive Reinforcement → Step 5a-2: Student Feedback-Driven Elaboration → Refined Response
     ↓
Step 6: Refined Response를 SFT 데이터로 사용
```

---

### Case B: Scaffolded & Coached Mastery: 스캐폴딩 기반 숙달

> **문제 ID**: `gsm8k_train_4`
> **문제**: "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"
> **정답**: 624

#### Iteration 1: Student 초기 응답 → PO 평가 부분 실패

**Step 1 — Student Initial Response:**

(System Message = `LEARNING_TASK_SYSTEM_PROMPT`, User Message = 위 문제 원문 — Case A Step 1과 동일한 프롬프트 구조)

```
- Instructional goal alignment: This solution demonstrates the instructional goal by applying
  arithmetic operations, logical reasoning, and mathematical concepts to solve a real-world
  problem and ensuring the final answer is accurate and consistent with all problem conditions.

- Step-by-step reasoning:
  1. **Understand the Problem Context**:
     - James writes a 3-page letter to 2 friends twice a week.
     - We need to calculate how many pages he writes in a year.
     - Key information: 3 pages per letter, 2 friends, 2 times per week, 52 weeks in a year.

  2. **Apply Arithmetic Operations**:
     - First, calculate the number of pages James writes to one friend in one week:
       $ 3 \text{ pages} $ (per letter) × $ 1 \text{ friend} $ = $ 3 \text{ pages} $ per week
       to one friend.
     - Next, calculate the number of pages James writes to both friends in one week:
       $ 3 \text{ pages} $ (per friend) × $ 2 \text{ friends} $ = $ 6 \text{ pages} $ per week.
     - Since he writes twice a week, calculate the weekly total:
       $ 6 \text{ pages/week} $ × $ 2 \text{ times/week} $ = $ 12 \text{ pages/week} $.
     - Finally, calculate the yearly total:
       $ 12 \text{ pages/week} $ × $ 52 \text{ weeks} $ = $ 624 \text{ pages/year} $.

  3. **Apply Logical Reasoning**:
     - We verified that the weekly total is calculated correctly by multiplying the pages per
       friend, the number of friends, and the number of times he writes per week.
     - We also ensured that the yearly total is calculated by multiplying the weekly total by
       the number of weeks in a year.

  4. **Verify Final Answer**:
     - Substitute the final answer back into the problem:
       If James writes 12 pages per week and there are 52 weeks in a year, then $ 12 \times 52 = 624 $.
     - This matches the ground truth, and all problem conditions are satisfied.

- Final answer: "The answer is $\boxed{624}$"
```

**추출된 답**: 624 (정답과 일치)

> **핵심 관찰**: Student는 실제로 정답(624)을 도출했습니다. 그러나 로그에서 Teacher가 Student의 답을 312로 인식하여 3/6 PO 미충족으로 평가했습니다. 이는 Teacher 모델(Qwen3-4B)이 Student 응답의 계산 과정을 잘못 해석한 사례입니다. "twice a week" 조건이 답에 반영되었는지를 Teacher가 오판한 것입니다.

**Step 2 — Teacher PO Evaluation (Iteration 1):**

```json
{
  "performance_evaluation": [
    {
      "objective_content": "Given a complex mathematical word problem, the model will solve it by applying arithmetic operations, logical reasoning, and mathematical concepts to determine the correct numerical answer in real-world contexts with 100% accuracy.",
      "is_satisfied": false,
      "reasoning": "WHY: The student's final answer of 312 does not match the ground truth of 624. This indicates that the model did not arrive at the correct numerical answer..."
    },
    {
      "objective_content": "Given a mathematical word problem requiring multi-step arithmetic calculations, the model will perform the calculations using appropriate operations (Add, Subtract, Multiply, Divide) with 100% accuracy.",
      "is_satisfied": false,
      "reasoning": "WHY: The student's calculation of 3 pages × 2 friends = 6 pages per week is correct, but the final multiplication of 6 × 52 = 312 is incorrect..."
    },
    {
      "objective_content": "Given a mathematical word problem, the model will substitute the computed final answer back into the original problem constraints, check for consistency with all problem conditions, and ensure the answer is correct and consistent with every condition stated in the problem with 100% accuracy.",
      "is_satisfied": false,
      "reasoning": "WHY: The student's final answer of 312 does not match the ground truth of 624, indicating that the answer is not consistent with the problem's constraints..."
    }
  ]
}
```

**미충족 PO (3/6):**

| PO # | Target | 미충족 이유 |
|------|--------|-----------|
| 1 | Instructional Goal | Teacher가 최종 답을 312로 오인식 |
| 4 | Apply Arithmetic Operations | "twice a week" 계산 누락으로 판단 |
| 6 | Verify Final Answer | 답 불일치로 판단 |

#### Case B Step 3: Scaffolding Artifact Generation

Teacher가 미충족 PO별로 차별화된 Scaffolding과 **서술형 피드백**을 생성합니다.

##### 실제 프롬프트

**System Message** (`SCAFFOLDED_CORRECTIVE_FEEDBACK_SYSTEM_PROMPT`):
```
You are an instructional design expert (Dick & Carey model) creating a Scaffolding Artifact to help a student improve.

Your role is to design pedagogical scaffolding for Performance Objectives that the student failed to meet. This scaffolding will be stored as a "Scaffolding Artifact" that the student can reference in their next attempt.
```

**User Message** (`SCAFFOLDED_CORRECTIVE_FEEDBACK_USER_PROMPT`):

> Placeholder: `{problem_text}` → 문제 원문, `{student_response}` → Student의 응답, `{po_evaluation}` → Teacher PO 평가 JSON, `{previous_iteration_summaries}` → 이전 반복 요약 목록, `{instructional_goal}` → Instructional Goal, `{task_analysis}` → Task Analysis Tree

> Scaffolding 프롬프트는 Case A의 PO 평가 프롬프트와 동일한 구조이며, 미충족 PO에 대해 HOT/LOT 구분에 따라 차별화된 Scaffolding을 설계합니다. 상세 형식은 [부록 A](#부록-a-주요-개념-정리)를 참고하세요.

**Scaffolding 결과:** 3개 HOT (High-Order Thinking) Scaffolding 생성

#### Iteration 2: Student 재응답 → PO 전체 충족

**Step 4 — Student Reattempt:**

Student가 Teacher의 Scaffolding Artifact를 참조하여 개선된 응답을 생성합니다.

**System Message** (`TEACHER_SUPPORTED_REATTEMPT_SYSTEM_PROMPT`):

> Placeholder: `{scaffolding_system_prompt}` → `LEARNING_TASK_SYSTEM_PROMPT` (채워진 상태)

```
[LEARNING_TASK_SYSTEM_PROMPT — Task Analysis 포함]

[Instructions]
1. Carefully study the scaffolding artifact provided in the user message, including the strategies and examples
2. For High Order Skills: follow the suggested strategies and reasoning approaches
3. For Low Order Skills: review the missed concepts and explanations
4. Pay special attention to the Key Attention Points and Feedback sections
5. Address each unsatisfied performance objective systematically
6. Show your improved thinking step by step
7. Provide your final answer clearly
```

**User Message** (`TEACHER_SUPPORTED_REATTEMPT_USER_PROMPT`):

> Placeholder: `{scaffolding_artifact}` → Scaffolding Artifact 전체 텍스트 (`_raw_text`), `{problem_text}` → 문제 원문

```
[Scaffolding Artifact]
Your teacher has evaluated your previous response and designed the following scaffolding to
guide your improvement:

[... 전체 Scaffolding Artifact ...]

[Problem]
James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
```

**Step 2 — Teacher PO Evaluation (Iteration 2):**

**모든 PO 충족 (6/6)** → 반복 종료 → **Case B: Scaffolded & Coached Mastery 확정** (2회차 성공)

#### Case B Step 5: Feedback-Driven Elaboration

Case B에서도 Case A와 동일하게, PO가 모두 충족된 후 **Teacher Positive Reinforcement → Student Feedback-Driven Elaboration** 과정을 수행합니다.

```python
# nodes.py
# Case A / Case B: PO 충족 후 Feedback-Driven Elaboration
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
sft_case = SFTCase.SCAFFOLDED_COACHED_MASTERY.value
```

##### Teacher Positive Reinforcement 결과

```
[Strengths]
- PO 1: The student demonstrated strong instructional goal alignment by applying arithmetic
  operations and logical reasoning to solve a real-world problem, resulting in an accurate
  final answer of 624 pages per year.
- PO 2: The student effectively identified key information from the problem, including the
  number of pages per letter, number of friends, frequency of writing, and weeks in a year,
  which are essential for solving the problem.

[Enhancement Suggestions]
1. Clarify the reasoning structure by explicitly labeling each step with a clear heading or
   number, making it easier to follow the logical progression of calculations.
2. Provide more explicit connections between the steps, such as explaining why multiplying
   by 2 friends and then by 2 times per week is appropriate in this context.
...
```

##### Feedback-Driven Elaboration 응답 결과

```
- Instructional goal alignment: This solution demonstrates the instructional goal by applying
  arithmetic operations and logical reasoning to solve a real-world problem. It clearly identifies
  key information, applies mathematical concepts like multiplication and unit conversion, and
  verifies the final answer through logical steps.

- Step-by-step reasoning:
  **Step 1: Calculate Pages per Friend per Week**
  James writes 3 pages per letter. Since he writes to 2 friends, we multiply the number of
  pages per letter by the number of friends:
  $ 3 \text{ pages/letter} \times 2 \text{ friends} = 6 \text{ pages/week} $
  This uses multiplication to scale the number of pages to both friends.

  **Step 2: Calculate Total Pages per Week**
  James writes twice a week, so we multiply the weekly total by 2:
  $ 6 \text{ pages/week} \times 2 \text{ times/week} = 12 \text{ pages/week} $

  **Step 3: Calculate Total Pages per Year**
  $ 12 \text{ pages/week} \times 52 \text{ weeks/year} = 624 \text{ pages/year} $

  **Step 4: Verify Final Answer**
  $ 3 \times 2 \times 2 \times 52 = 624 $. All conditions satisfied.

- Final answer: "The answer is $\boxed{624}$"
```

#### 최종 로그 요약

```json
{
  "id": "gsm8k_train_4",
  "sft_case": "case_b_scaffolded_coached_mastery",
  "iterative_scaffolding": {
    "success": true,
    "iterations_needed": 2
  },
  "hot_count": 3,
  "lot_count": null
}
```

**Case B: Scaffolded & Coached Mastery 흐름 요약:**
```
Iteration 1: Student(624✓) → Teacher(3/6 PO — Teacher 오판) → Scaffolding(3 HOT)
Iteration 2: Student(624✓) → Teacher(6/6 PO ✓) → Case B 확정
     ↓
Step 5a-1: Teacher Positive Reinforcement → Step 5a-2: Student Feedback-Driven Elaboration → Refined Response
     ↓
Step 6: Refined Response를 SFT 데이터로 사용
```

> **핵심 관찰**: Student는 1회차에서 이미 정답(624)을 도출했지만, Teacher(Qwen3-4B)가 Student 응답의 계산 과정을 잘못 해석하여 "twice a week" 조건 미반영으로 판단했습니다. 이는 작은 모델(4B)이 Teacher 역할을 할 때 발생할 수 있는 평가 오류 사례입니다. Scaffolding을 거친 2회차에서 Teacher가 정확히 평가하여 모든 PO를 충족시켰습니다.

---

### Case C: Teacher Modeling Distillation: 교사 모델링 증류

> **문제 ID**: `gsm8k_train_129`
> **문제**: "Dan owns an ice cream shop and every sixth customer gets a free ice cream cone. Cones cost $2 each. If he sold $100 worth of cones, how many free ones did he give away?"
> **정답**: 10

5회 반복 후에도 모든 PO를 충족하지 못한 경우 → **Case C: Teacher Modeling Distillation 확정**

#### Iteration 1~5 요약

| Iteration | Student 답 | 정답 일치 | 충족 PO | 미충족 PO | 핵심 오류 |
|-----------|-----------|----------|---------|----------|----------|
| 1 | 8 | ✗ | 2/6 | 4/6 | floor(50/6)=8로 계산, "every sixth customer" 해석 오류 |
| 2 | 9 | ✗ | —/1 | 1/1 | 답 검증 PO 미충족 |
| 3 | 9 | ✗ | —/1 | 1/1 | 답 검증 PO 미충족 |
| 4 | 9 | ✗ | —/1 | 1/1 | 답 검증 PO 미충족 |
| 5 | 8 | ✗ | —/1 | 1/1 | 답 검증 PO 미충족 |

> **핵심 관찰**: Student(Qwen3-4B)는 "$100 ÷ $2 = 50 cones sold"까지는 정확히 계산하지만, "every sixth customer gets a free cone"의 해석에서 floor(50/6)=8로 일관되게 계산합니다. 정답은 **유료 고객 5명마다 무료 1명이므로 총 고객 수에 free cone을 포함**하여 50÷5=10입니다. 5회 Scaffolding에도 이 해석 오류가 교정되지 않았습니다.

#### Iteration 1 상세

**Student 응답:**

```
- Instructional goal alignment: This solution demonstrates the instructional goal by applying
  arithmetic operations, logical reasoning, and mathematical concepts to solve a real-world
  problem.

- Step-by-step reasoning:
  1. **Identify key information in the problem**:
     - Every sixth customer gets a free ice cream cone.
     - Cones cost $2 each.
     - Dan sold $100 worth of cones.
     - We need to find how many free cones were given away.

  2. **Determine relevant mathematical concepts**:
     - Division to find the total number of cones sold.
     - Understanding of how free cones are distributed (every 6th cone).
     - Logical reasoning to determine how many of those are free.

  3. **Apply arithmetic operations**:
     - Total cones sold: $100 / $2 = 50
     - Number of free cones: floor(50 / 6) = 8

  4. **Verify final answer**: [반복적으로 floor(50/6)=8을 확인하며 답이 10이어야 한다고 인식하나 교정 실패]

- Final answer: "The answer is \boxed{8}"
```

**Teacher PO 평가 (Iteration 1):** 2/6 충족, 4/6 미충족

미충족 PO:
1. **Instructional Goal PO**: 최종 답 8 부정확
2. **Apply Arithmetic Operations**: 산술 계산 오류
3. **Apply Logical Reasoning**: 변수 관계 분석 오류
4. **Verify Final Answer**: 답 검증 실패

**Scaffolding Artifact (Iteration 1):** 미충족 PO별 HOT Scaffolding 제공

#### Iterations 2~5

각 iteration에서 Student는 8 또는 9를 답하며, 핵심 해석 오류(every sixth customer의 의미)를 교정하지 못했습니다. Teacher는 매 iteration마다 답 검증 PO 미충족을 지적하는 Scaffolding을 제공했지만, Student는 5회 반복 후에도 해석을 교정하지 못했습니다.

→ **5회 반복 후 PO 미충족** → **Case C: Teacher Modeling Distillation 확정**

#### Case C Step 5b: Teacher Final Solution

Teacher가 전체 iteration history를 참고하여 정답(10)을 기반으로 Step 1과 동일한 형식의 풀이를 **평문 텍스트**로 생성합니다.

##### 실제 프롬프트

**System Message** (`TEACHER_MODELING_SYSTEM_PROMPT`):

> Placeholder: `{instructional_goal}` → Instructional Goal 텍스트, `{task_analysis}` → Task Analysis Tree (최대 1500자 제한)

```
The purpose of your response is to demonstrate the attainment of the Instructional Goal: [Instructional Goal 텍스트]

You must adhere to the specific performance procedures and required knowledge/skills outlined in the Instructional Analysis results below. Ensure that your solution describes the full reasoning process using all provided steps and resources before arriving at the final answer.

[Instructional Analysis]
[Task Analysis Tree (최대 1500자)]

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

**User Message** (`TEACHER_MODELING_USER_PROMPT`):

> Placeholder: `{problem_text}` → 문제 원문, `{ground_truth}` → "10", `{iteration_history}` → `_format_iteration_history()`로 포맷된 전체 iteration history

```
[Problem]
Dan owns an ice cream shop and every sixth customer gets a free ice cream cone. Cones cost $2 each. If he sold $100 worth of cones, how many free ones did he give away?

[Ground Truth]
10

[Iteration History]
The following is a summary of each iteration's student attempt and teacher scaffolding:
Iter 1: Student calculated 50 cones sold ($100/$2) but computed floor(50/6)=8 free cones.
        Scaffolding guided student to reconsider the relationship between paying and free customers.
Iter 2-5: Student alternated between 8 and 9, unable to correctly interpret that every 6th
          customer (including free ones) means 1 free per 5 paying customers → 50/5 = 10.
```

##### Teacher 최종 풀이 결과

```
- Instructional goal alignment: This solution demonstrates the instructional goal by applying
  arithmetic operations, logical reasoning, and mathematical concepts to solve a real-world
  problem. It identifies key information, applies the correct operations, analyzes the distribution
  of free cones, and verifies the final answer against the problem's conditions.

- Step-by-step reasoning:
  1. **Identify key information in the problem**:
     - Every sixth customer gets a free ice cream cone.
     - Cones cost $2 each.
     - Dan sold $100 worth of cones.
     - We need to find how many free cones were given away.

  2. **Determine relevant mathematical concepts**:
     - Division to find the total number of cones sold.
     - Understanding of how free cones are distributed (every 6th customer).
     - Logical reasoning to determine how many of those are free.

  3. **Apply arithmetic operations**:
     - Total cones sold: $100 / $2 = 50
     - Every 6th customer gets a free cone, meaning for every 5 paying customers,
       1 gets a free cone. So: 50 / 5 = 10 free cones.

  4. **Verify final answer**:
     - 50 paid cones + 10 free cones = 60 total customers
     - Every 6th customer: 60 / 6 = 10 free cones ✓
     - Revenue check: 50 × $2 = $100 ✓

- Final answer: "The answer is \boxed{10}"
```

> **핵심 차이**: Teacher의 풀이는 "every sixth customer"를 **총 고객 중 6번째마다** (= 5명 유료 + 1명 무료)로 정확히 해석합니다. Student가 5회 반복 동안 고수한 floor(50/6)=8 계산 오류를 교정하는 교육적 풀이입니다.

#### 최종 로그 요약

```json
{
  "id": "gsm8k_train_129",
  "sft_case": "case_c_teacher_modeling_distillation",
  "iterative_scaffolding": {
    "success": false,
    "iterations_needed": 5
  },
  "hot_count": 11,
  "lot_count": null,
  "sft_response": "[Teacher의 최종 풀이 — 위 텍스트]"
}
```

**Case C: Teacher Modeling Distillation 흐름 요약:**
```
Iteration 1: Student(8✗) → Teacher(2/6 PO) → Scaffolding → "every 6th customer" 교정 시도
Iteration 2: Student(9✗) → Teacher(PO 미충족) → Scaffolding → 교정 실패
Iteration 3: Student(9✗) → Teacher(PO 미충족) → Scaffolding → 교정 실패
Iteration 4: Student(9✗) → Teacher(PO 미충족) → Scaffolding → 교정 실패
Iteration 5: Student(8✗) → Teacher(PO 미충족) → 5회 초과 → Case C 확정
     ↓
Step 5b: Teacher Final Solution: 정답(10) 기반 교육적 풀이 생성 → SFT 데이터로 사용
```

> **핵심 관찰**: 이 사례는 Student 모델이 **문제 해석 오류**를 5회 Scaffolding에도 교정하지 못한 경우입니다. "Every sixth customer gets a free cone"에서 "6번째 고객마다 무료"라는 해석이 핵심 쟁점이며, Student는 일관되게 `floor(50/6)`로 계산한 반면, 정답은 유료 5명 + 무료 1명 = 6명 단위로 50/5=10입니다. 이런 경우 Teacher가 교육적 모범 풀이를 제공하여 SFT 데이터의 품질을 보장합니다.

---

## 4. Phase 3: Instructional Delivery

Phase 2에서 생성된 SFT 데이터로 Student 모델을 Fine-tuning하고, 평가 데이터셋에서 성능을 측정합니다.

### SFT 데이터 형식

```json
{
  "instruction": "[Original Instruction + LEARNING_TASK_SYSTEM_PROMPT (동적 결합)]",
  "input": "Question: [문제 텍스트]",
  "output": "[Case별 SFT 응답]",
  "metadata": {
    "id": "gsm8k_train_XXX",
    "sft_case": "case_a_independent_performance_mastery|case_b_scaffolded_coached_mastery|case_c_teacher_modeling_distillation",
    "ground_truth": "[정답]"
  }
}
```

> **instruction 동적 결합**: SFT 데이터 생성 시 `_create_sft_entry()`에서 `original_instruction + "\n\n" + LEARNING_TASK_SYSTEM_PROMPT`를 런타임에 결합합니다.

### Case별 SFT 엔트리 예시

#### Case A: Independent Performance Mastery 엔트리 (gsm8k_train_0)

```json
{
  "instruction": "You are a helpful math assistant.\nSolve this mathematical problem step by step...\n\nThe purpose of your response is to demonstrate the attainment of the Instructional Goal: The model will solve complex mathematical word problems by applying arithmetic operations, logical reasoning, and mathematical concepts to determine the correct numerical answer in real-world contexts.\n\nYou must adhere to the specific performance procedures...\n\n[Instructional Analysis]\n### Instructional Analysis Results\n...\n\n[Instructions]\n1. Identify which skills...\n...\n\n[Output Format]\n- Instructional goal alignment: ...\n- Step-by-step reasoning: ...\n- Final answer: \"The answer is \\boxed{your final answer}\"",
  "input": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "output": "- Instructional goal alignment: This solution demonstrates the instructional goal by applying arithmetic operations (addition and division) and logical reasoning to solve the problem, while also explaining the mathematical concepts and real-world implications...\n- Step-by-step reasoning:\n  1. **Identify key information**: ...\n  2. **Determine relevant mathematical concepts**: ...\n  3. **Apply arithmetic operations**: ... 48/2 = 24 ... 48 + 24 = 72\n  4. **Apply logical reasoning**: ...\n  5. **Verify final answer**: ...\n  6. **Real-world implications**: Natalia sold an average of 36 clips per month\n- Final answer: \"The answer is \\boxed{72}\"",
  "metadata": {
    "id": "gsm8k_train_0",
    "sft_case": "case_a_independent_performance_mastery",
    "ground_truth": "72"
  }
}
```

> **Case A**: PO 충족 후 Teacher Positive Reinforcement → Student Feedback-Driven Elaboration → Refined Response를 SFT output으로 사용.

#### Case B: Scaffolded & Coached Mastery 엔트리 (gsm8k_train_4)

```json
{
  "instruction": "Original Instruction + LEARNING_TASK_SYSTEM_PROMPT (동적 결합) — Case A와 동일한 구조",
  "input": "Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
  "output": "- Instructional goal alignment: This solution demonstrates the instructional goal by applying arithmetic operations and logical reasoning to solve a real-world problem...\n- Step-by-step reasoning:\n  **Step 1: Calculate Pages per Friend per Week**: 3 × 2 = 6\n  **Step 2: Calculate Total Pages per Week**: 6 × 2 = 12\n  **Step 3: Calculate Total Pages per Year**: 12 × 52 = 624\n  **Step 4: Verify Final Answer**: 3 × 2 × 2 × 52 = 624 ✓\n- Final answer: \"The answer is $\\boxed{624}$\"",
  "metadata": {
    "id": "gsm8k_train_4",
    "sft_case": "case_b_scaffolded_coached_mastery",
    "ground_truth": "624"
  }
}
```

> **Case B**: Scaffolding을 통해 PO를 충족한 후 Teacher Positive Reinforcement → Student Feedback-Driven Elaboration → Refined Response를 SFT output으로 사용.

#### Case C: Teacher Modeling Distillation 엔트리 (gsm8k_train_129)

```json
{
  "instruction": "Original Instruction + LEARNING_TASK_SYSTEM_PROMPT (동적 결합) — Case A와 동일한 구조",
  "input": "Question: Dan owns an ice cream shop and every sixth customer gets a free ice cream cone. Cones cost $2 each. If he sold $100 worth of cones, how many free ones did he give away?",
  "output": "- Instructional goal alignment: This solution demonstrates the instructional goal by applying arithmetic operations, logical reasoning, and mathematical concepts...\n- Step-by-step reasoning:\n  1. **Identify key information**: Every 6th customer free, cones $2, sold $100\n  2. **Apply arithmetic operations**: 50 cones sold, 50/5 = 10 free cones\n  3. **Verify final answer**: 50 paid + 10 free = 60, 60/6 = 10 ✓, 50 × $2 = $100 ✓\n- Final answer: \"The answer is \\boxed{10}\"",
  "metadata": {
    "id": "gsm8k_train_129",
    "sft_case": "case_c_teacher_modeling_distillation",
    "ground_truth": "10"
  }
}
```

> **Case C**: Teacher가 교육적 풀이를 평문 텍스트로 생성. 5회 Scaffolding 후에도 Student가 문제 해석 오류를 교정하지 못해 Teacher의 모범 풀이를 SFT 데이터로 사용.

### 분류별 SFT 응답 소스

| 분류 명칭 | SFT `output` 소스 | 특징 |
|------|-------------------|------|
| **Case A: Independent Performance Mastery** | Student Feedback-Driven Elaboration 응답 | PO 충족 후 Teacher Positive Reinforcement 기반 Feedback-Driven Elaboration |
| **Case B: Scaffolded & Coached Mastery** | Student Feedback-Driven Elaboration 응답 | Scaffolding + PO 충족 후 Teacher Positive Reinforcement 기반 Feedback-Driven Elaboration |
| **Case C: Teacher Modeling Distillation** | Teacher 최종 풀이 (평문 텍스트) | 학생 약점을 보완한 교육적 정답 풀이 |

### 평가 방법

| Method | 설명 |
|--------|------|
| Baseline | 파인튜닝 없는 기본 모델 |
| SFT | 일반 SFT 파인튜닝 |
| SFT_ID-MAS | ID-MAS 방식 SFT (Phase 2 데이터 활용) |

---

## 부록 A: 주요 개념 정리

### HOT vs LOT Scaffolding

| 유형 | 대상 인지 수준 | 제공 내용 |
|------|--------------|----------|
| **HOT** (High-Order Thinking) | 분석/평가/창조 | `strategy_suggestion`, `partial_example`, `key_attention_points` |
| **LOT** (Low-Order Thinking) | 기억/이해/적용 | `missed_concept`, `brief_explanation` |

### Scaffolding Artifact 및 피드백

- 각 iteration에서 생성된 Scaffolding Artifact가 **누적** 저장됩니다
- `SCAFFOLDED_CORRECTIVE_FEEDBACK_USER_PROMPT`는 **구조화된 마크다운**으로 출력합니다 (JSON 미사용)
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
| `INSTRUCTIONAL_GOAL_USER_PROMPT` | `prompts/design_prompts.py` | 데이터셋 분석 → Instructional Goal 도출 | Phase 1 / Step 0 | User | JSON |
| `INSTRUCTIONAL_ANALYSIS_SYSTEM_PROMPT` | `prompts/design_prompts.py` | 교수 분석 전문가 역할 설정 | Phase 1 / Step 2 | System | — |
| `INSTRUCTIONAL_ANALYSIS_USER_PROMPT` | `prompts/design_prompts.py` | Learning Objective → Task Analysis Tree 분해. **"Verify Final Answer" subskill 필수 포함 규칙** 적용 | Phase 1 / Step 2 | User | Text (Tree) |
| `PERFORMANCE_OBJECTIVES_SYSTEM_PROMPT` | `prompts/design_prompts.py` | PO 생성 전문가 역할 설정 | Phase 1 / Step 3 | System | — |
| `PERFORMANCE_OBJECTIVES_USER_PROMPT` | `prompts/design_prompts.py` | Task Analysis → Performance Objectives 생성. **"Verify Final Answer" PO 필수 규칙** 적용 | Phase 1 / Step 3 | User | JSON |
| `LEARNING_TASK_SYSTEM_PROMPT` | `prompts/learning_prompts.py` | Student 문제 해결 시스템 프롬프트 | Phase 2 / Step 1 | System | Text |
| `LEARNING_TASK_USER_PROMPT` | `prompts/learning_prompts.py` | Student 문제 전달 | Phase 2 / Step 1 | User | Text |
| `FORMATIVE_ASSESSMENT_SYSTEM_PROMPT` | `prompts/learning_prompts.py` | Teacher 평가 역할 설정 | Phase 2 / Step 2 | System | — |
| `FORMATIVE_ASSESSMENT_USER_PROMPT` | `prompts/learning_prompts.py` | Teacher PO 평가 (평가 전용). **CRITICAL Answer Verification Rule** 적용 | Phase 2 / Step 2 | User | JSON |
| `SCAFFOLDED_CORRECTIVE_FEEDBACK_SYSTEM_PROMPT` | `prompts/learning_prompts.py` | Scaffolding 생성 역할 설정 | Phase 2 / Step 3 | System | — |
| `SCAFFOLDED_CORRECTIVE_FEEDBACK_USER_PROMPT` | `prompts/learning_prompts.py` | 미충족 PO별 HOT/LOT Scaffolding + Feedback 생성 | Phase 2 / Step 3 | User | Structured Text |
| `TEACHER_SUPPORTED_REATTEMPT_SYSTEM_PROMPT` | `prompts/learning_prompts.py` | Student 재응답: `dataset_prompt` + `LEARNING_TASK_SYSTEM_PROMPT` 기반 지침 | Phase 2 / Step 4 | System | Text |
| `TEACHER_SUPPORTED_REATTEMPT_USER_PROMPT` | `prompts/learning_prompts.py` | Student 재응답: Scaffolding Artifact + 문제 전달 | Phase 2 / Step 4 | User | Text |
| `POSITIVE_REINFORCEMENT_SYSTEM_PROMPT` | `prompts/learning_prompts.py` | Positive Reinforcement 역할 설정 | Phase 2 / Step 5a-1 (Case A / Case B) | System | — |
| `POSITIVE_REINFORCEMENT_USER_PROMPT` | `prompts/learning_prompts.py` | PO 충족 후 강점 + 개선점 피드백 생성 | Phase 2 / Step 5a-1 (Case A / Case B) | User | Structured Text |
| `FEEDBACK_DRIVEN_ELABORATION_SYSTEM_PROMPT` | `prompts/learning_prompts.py` | Positive Reinforcement 기반 응답 정교화 지침 | Phase 2 / Step 5a-2 (Case A / Case B) | System | Text |
| `FEEDBACK_DRIVEN_ELABORATION_USER_PROMPT` | `prompts/learning_prompts.py` | Positive Reinforcement + 문제 전달 | Phase 2 / Step 5a-2 (Case A / Case B) | User | Text |
| `TEACHER_MODELING_SYSTEM_PROMPT` | `prompts/learning_prompts.py` | Final Solution 역할 설정 | Phase 2 / Step 5b (Case C) | System | — |
| `TEACHER_MODELING_USER_PROMPT` | `prompts/learning_prompts.py` | 최대 반복 실패 후 교육적 풀이 생성 | Phase 2 / Step 5b (Case C) | User | Text (평문) |
