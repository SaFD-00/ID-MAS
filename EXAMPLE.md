# ID-MAS 동작 예시: Phase 1~3 상세 설명

> GSM8K 데이터셋 + Qwen2.5-7B-Instruct 모델 기반 실제 실행 로그를 통해 ID-MAS의 3-Phase 파이프라인이 어떻게 작동하는지 Case별로 설명합니다.
> **각 Step에서 LLM에 입력되는 실제 프롬프트**를 포함하여 파이프라인 동작을 프롬프트 수준까지 설명합니다.

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

Phase 1은 데이터셋 단위로 **1회만** 실행됩니다. GSM8K 데이터셋에 대해 Teacher 모델(Qwen2.5-7B-Instruct)이 교수 설계를 수행합니다.

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
Solve this math problem.
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

### Sample 2
Solve this math problem.
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

#### 실제 프롬프트

**User Message** (`INSTRUCTIONAL_ANALYSIS_PROMPT`, system=None):

> Placeholder: `{learning_objective}` → 실제 GSM8K Instructional Goal

```
You are an instructional design expert. Perform the Instructional Analysis step of the Dick & Carey model for the learning objective provided below.

[Learning Goal]
The model will solve complex mathematical word problems by setting up and solving equations, interpreting relationships, and performing calculations.

[Instructions]
Perform the Instructional Analysis and construct a hierarchical structure in the form of:
Terminal Goal → Subskills → Subtasks

- Present the instructional analysis results as a text-based tree structure.
- If a required sub-component is unknown or not applicable, still keep the slot but write 'N/A' instead of modifying the structure.
- Write all skill statements concisely using an action verb + object format.
- For every function or sub-function, indicate the learning type based on Anderson and Krathwohl's Revised Taxonomy in the form:
  (Cognitive Process Dimension(Remember / Understand / Apply / Analyze / Evaluate / Create) – Knowledge Dimension(Factual, Conceptual, Procedural, Metacognitive knowledge))
  Examples: (Understand – Conceptual Knowledge), (Apply – Procedural Knowledge)
- Include only the minimum number of Subskills and Subtasks that are essential to achieving the Terminal Goal.

The final output must strictly follow the structure and labels in the Output Format below.
- Do not change the wording, ordering, line breaks, or section titles.
- The Output Format example is provided ONLY to specify formatting and structure.
- Determine all subskills and subtasks strictly based on the given Learning Goal.

[Output Format]
### Instructional Analysis Results
Terminal Goal: [Learning objective statement] (Cognitive Process Dimension – Knowledge Dimension)
 ├── [1] [Subskill statement] (Cognitive Process Dimension – Knowledge Dimension)
 │   ├── [1-1] [Subtask statement] (Cognitive Process Dimension – Knowledge Dimension)
 ├── [2] [Subskill statement] (Cognitive Process Dimension – Knowledge Dimension)
 │   ├── [2-1] [Subtask statement] (Cognitive Process Dimension – Knowledge Dimension)
 │   └── [2-2] [Subtask statement] (Cognitive Process Dimension – Knowledge Dimension)
 └── [3] [Subskill statement] (Cognitive Process Dimension – Knowledge Dimension)

[Requirements]
- Maintain the exact structure, titles, line breaks, and tree characters (├──, │, └──).
- Do not change section name ("Instructional Analysis Results").
- Output only the required instructional analysis products; do not include introductions, explanations, or references.
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

#### 실제 프롬프트

**User Message** (`PERFORMANCE_OBJECTIVES_PROMPT`, system=None):

> Placeholder: `{instructional_analysis}` → 실제 GSM8K Task Analysis Tree

```
You are an instructional designer specializing in the Dick and Carey instructional design model, and a researcher in LLM learning methodologies.
Based on the provided Terminal Goal and Instructional Analysis Result, generate a set of Performance Objectives that will serve as the criteria for evaluating the observable performance within the LLM's reasoning process.

Performance objectives should be written using the guidelines provided in Anderson & Krathwohl's Taxonomy for Learning.
Specifically, they should be created using information from the learning outcomes identified in the Instructional Analysis Results.

[Input Data]
Instructional Analysis Result: ### Instructional Analysis Results
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

[Instructions]
- For each Subskills and Subtask in the instructional analysis, you must create at least one Performance Objective.
- Every Performance Objective must include all three components—Behavior, Condition, and Criterion—and each component must be explicitly stated.
- Behavior: This is a description of LLM's intellectual skill including actions, content, and concepts.
- Condition: This is a description of the tools and resources that will be available to the learner when performing the skill. Write the conditions based solely on the data given in the problem or generated during the reasoning process. It should ALWAYS begin with 'Given ~'.
- Criterion: This is a description of acceptable performance of the skill. The Criterion component must be tailored to the nature of the task: for tasks with correct answers, it must include a clear and measurable standard such as accuracy requirements, acceptable error ranges, or the number of correct responses; whereas for tasks with no single correct answer, it must specify the information or features that must be present for an acceptable response.
- Furthermore, these criteria must be formulated to evaluate the observable reasoning process within a single problem-solving task.
- Each Performance Objective must correspond directly to a single Subskill and Subtask, and you must not add content that does not appear in the Instructional Analysis Result.

[Anderson & Krathwohl's Taxonomy Reference]

Verbs Used by Cognitive Process Dimension (Behavior):
  - Remember: Recognizing, Recalling
  - Understand: Interpreting, Exemplifying, Classifying, Summarizing, Inferring, Comparing, Explaining
  - Apply: Executing, Implementing
  - Analyze: Differentiating, Organizing, Attributing
  - Evaluate: Checking, Critiquing
  - Create: Generating, Planning, Producing

Description of Knowledge Dimensions:
  - Factual Knowledge: Basic elements that must be mastered to solve subjects or problems in a subject
  - Conceptual Knowledge: Interrelationships between basic elements within a superstructure
  - Procedural Knowledge: Methods of performing tasks, methods of inquiry, criteria, algorithms, techniques
  - Metacognitive Knowledge: Awareness of knowledge cognition and knowledge of knowledge and cognition in general

[Output Format]
Your output must be formatted as JSON, following this structure and no other form of explanation or commentary:

{
  "performance_objectives": [
    {
      "target": "Terminal Goal",
      "Behavior": "...",
      "Condition": "...",
      "Criterion": "..."
    },
    {
      "target": "Subskill X",
      "Behavior": "...",
      "Condition": "...",
      "Criterion": "..."
    },
    {
      "target": "Subtask X",
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
"instruction": "Solve this math problem."
```

**변환 후 (Enhanced):**
```
Solve this math problem.

## Learning Objective
Your response should demonstrate: The model will solve complex mathematical word problems by setting up and solving equations, interpreting relationships, and performing calculations.

## Problem-Solving Guidelines
Follow the structured approach below to ensure a complete and well-reasoned solution:

### Instructional Analysis Results
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

## Response Requirements
1. Explicitly connect each step to the relevant sub-skill or knowledge from the guidelines above
2. Verify your intermediate results before proceeding to the next step
3. Present your final answer clearly in the required format
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

#### Case A Step 1: Student 초기 응답

Student 모델이 Enhanced Instruction(Task Analysis 포함)을 참고하여 체계적 풀이를 생성합니다.

##### 실제 프롬프트

**System Message** (`SCAFFOLDING_SYSTEM_PROMPT`):

> Placeholder: `{instructional_goal}` → GSM8K Instructional Goal, `{task_analysis}` → Task Analysis Tree

```
The purpose of your response is to demonstrate the attainment of the Instructional Goal: The model will solve complex mathematical word problems by setting up and solving equations, interpreting relationships, and performing calculations.

You must adhere to the specific performance procedures and required knowledge/skills outlined in the Instructional Analysis results below. Ensure that your solution describes the full reasoning process using all provided steps and resources before arriving at the final answer.

[Instructional Analysis (Learning Structure)]
### Instructional Analysis Results
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

[Instructions]
1. Identify which skills and sub-skills from the instructional analysis are relevant to this problem
2. Plan your problem-solving strategy based on the terminal goal and subskills
3. Execute each step systematically, demonstrating the required performance behaviors
4. Ensure your solution describes the full reasoning process using all provided steps and resources
5. Provide your final answer clearly

[Output Format]
Problem-solving strategy and flow:
- Terminal goal alignment: [how this solution demonstrates the terminal goal]
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

#### Case A Step 2: Teacher PO 평가

Teacher가 Performance Objectives 기준으로 학생 응답을 평가합니다.

##### 실제 프롬프트

**User Message** (`TEACHER_INTERVENTION_PROMPT`):

> Placeholder: `{problem_text}` → 문제 원문, `{student_response}` → 위 Student 응답, `{performance_objectives}` → Phase 1에서 생성된 PO JSON, `{ground_truth}` → "72"

```
You are a teacher supporting the learning of a student.

Your role is NOT to provide correct answers, but to generate a reasoning state that guides the student's next response. You must monitor the student's reasoning steps to ensure they meet the established performance objectives.

In cases of non-compliance or error, you must generate tailored feedback to guide the student toward the desired outcome using Socratic questioning. Your feedback functions as an intermediate thought in a ReAct-style learning loop and must guide the student's next reasoning action.

[Input Data]
- Problem: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
- Student response: #### Step-by-Step Reasoning

1. **Identify Key Information and Relevant Mathematical Concepts**
   - In April, Natalia sold 48 clips.
   - In May, she sold half as many clips as in April.
   [... 이하 전체 응답 ...]

- Performance objectives: [
    {
      "target": "Terminal Goal",
      "Behavior": "Solve complex mathematical word problems by setting up and solving equations...",
      "Condition": "Given access to basic mathematical functions",
      "Criterion": "90% accuracy"
    },
    ...
  ]
- Ground truth (FOR REFERENCE ONLY - DO NOT REVEAL): 72

[Instructions]
1. Assess student performance according to each performance objective
2. Use the Criterion defined in each performance objective as the evaluation standard
3. DO NOT reveal correct answers or model solutions
4. Analyze the student response and determine which performance objectives are satisfied and which are not
5. All judgments must be grounded in observable reasoning behaviors in the student response
6. Avoid vague or abstract evaluations
7. For each unsatisfied performance objective, derive a reasoning action that the student should perform in the next iteration
8. Do not provide final conclusions, correct answers, or complete reasoning paths
9. Instead, specify what type of reasoning process, analytical step, or judgment perspective should be explicitly carried out next

[Output Format - JSON]
{
  "performance_evaluation": [
    {
      "objective_content": "The specific objective being evaluated",
      "is_satisfied": true or false,
      "reason_for_unmet_objective": "Detailed description if false; null if true",
      "socratic_question": "Socratic question if false; null if true"
    }
  ],
  "overall_assessment": {
    "objectives_met": "X of Y objectives satisfied",
    "all_satisfied": true or false,
    "primary_weakness": "Main area needing improvement; null if all satisfied",
    "recommended_focus": "What the student should focus on next; null if complete"
  }
}

CRITICAL INSTRUCTIONS FOR JSON OUTPUT:
1. Your response MUST be ONLY valid JSON - no additional text before or after
2. Do NOT include explanations, comments, markdown code blocks, or any text outside the JSON
3. Do NOT include LaTeX expressions, mathematical notation, or equations outside JSON string values
4. Ensure ALL brackets { }, [ ], and quotes are properly closed
5. If you need to include mathematical expressions, place them INSIDE JSON string values with proper escaping

Output ONLY the JSON object above. Do not include any additional text, explanation, or commentary outside the JSON structure.
```

##### Teacher 평가 결과

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

(System Message = `SCAFFOLDING_SYSTEM_PROMPT`, User Message = 위 문제 원문 — Case A Step 1과 동일한 프롬프트 구조)

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

(User Message = `TEACHER_INTERVENTION_PROMPT` — Case A Step 2와 동일한 프롬프트 구조, `{student_response}` 및 `{problem_text}` 치환)

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

#### Case B Step 3: Scaffolding Artifact 생성

Teacher가 미충족 PO별로 차별화된 Scaffolding을 생성합니다.

##### 실제 프롬프트

**User Message** (`SCAFFOLDING_ARTIFACT_PROMPT`):

> Placeholder: `{problem_text}` → 문제 원문, `{student_response}` → Student의 오답 응답, `{po_evaluation}` → Teacher PO 평가 JSON, `{failed_objectives}` → 미충족 PO 목록, `{task_analysis}` → Task Analysis Tree, `{iteration_number}` → 1, `{max_iterations}` → 5

```
You are an instructional design expert (Dick & Carey model) creating a Scaffolding Artifact to help a student improve.

Your role is to design pedagogical scaffolding for Performance Objectives that the student failed to meet. This scaffolding will be stored as a "Scaffolding DB" that the student can reference in their next attempt.

[Input Data]
- Problem: Valerie needs to put stamps on the envelopes she is about to mail. She has thank you cards for each of her grandmother, uncle and aunt for the birthday presents they sent. She also has to pay the water bill and the electric bill separately. She wants to send three more mail-in rebates than she does bills and she has twice as many job applications as rebates to mail. How many stamps does she need if everything needs 1 stamp except the electric bill, which needs 2?
- Student's Response: 1. Thank you cards: 3 (grandmother, uncle, aunt)
  2. Bills: 2 (water, electric)
  3. Mail-in rebates = bills + 3 = 5
  4. Job applications = 2 × rebates = 10
  Total items needing 1 stamp: 3 + 2 + 5 + 10 = 20
  Stamps for electric bill: 2
  Total stamps: 20 + 2 = 22
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
   - Socratic question: Guide thinking without revealing the answer
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
      "cognitive_level": "Analyze/Evaluate/Create" or "Remember/Understand/Apply",
      "failure_analysis": "Why the student failed this objective",
      "scaffolding_content": {
        "strategy_suggestion": "Suggested approach (for HOT) or null",
        "partial_example": "Partial worked example (for HOT) or null",
        "socratic_question": "Guiding question (for HOT) or null",
        "missed_concept": "Concept the student missed (for LOT) or null",
        "brief_explanation": "Concise explanation (for LOT) or null",
        "key_attention_points": "What to focus on in next attempt"
      }
    }
  ],
  "scaffolding_summary": "A 3-5 sentence summary synthesizing the key guidance..."
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

#### Case B Step 4: Student 재응답

Student가 Scaffolding DB를 참조하여 개선된 응답을 생성합니다.

##### 실제 프롬프트

**User Message** (`STUDENT_WITH_ARTIFACT_PROMPT`):

> Placeholder: `{problem_text}` → 문제 원문, `{previous_response}` → 이전 Student 응답, `{scaffolding_summary}` → Scaffolding 요약, `{scaffolding_artifacts}` → Scaffolding Artifact JSON, `{task_analysis}` → Task Analysis Tree

```
You are a student learning to solve problems with scaffolding support.

Your teacher has analyzed your previous attempt and prepared scaffolding guidance to help you improve. You must carefully use this scaffolding information to generate a better solution.

[Problem]
Valerie needs to put stamps on the envelopes she is about to mail. She has thank you cards for each of her grandmother, uncle and aunt for the birthday presents they sent. She also has to pay the water bill and the electric bill separately. She wants to send three more mail-in rebates than she does bills and she has twice as many job applications as rebates to mail. How many stamps does she need if everything needs 1 stamp except the electric bill, which needs 2?

[Your Previous Response]
1. Thank you cards: 3 (grandmother, uncle, aunt)
2. Bills: 2 (water, electric)
3. Mail-in rebates = bills + 3 = 5
4. Job applications = 2 × rebates = 10
Total items needing 1 stamp: 3 + 2 + 5 + 10 = 20
Stamps for electric bill: 2
Total stamps: 20 + 2 = 22

[Scaffolding Database]
The following scaffolding information has been prepared to help you:

Break down the problem into smaller parts and set up equations based on the relationships given. Pay special attention to the number of stamps required for each item type. The electric bill requires 2 stamps, not 1.

[Detailed Scaffolding Artifacts]
[HOT Artifact + LOT Artifact JSON 전체]

[Instructional Analysis (Learning Structure)]
[Task Analysis Tree 전체]

[Instructions]
1. Carefully review the Scaffolding Database and identify which guidance applies to your mistakes
2. For each piece of scaffolding guidance you use, explicitly acknowledge it
3. Apply the strategies, concepts, or explanations from the scaffolding to improve your solution
4. Show your improved reasoning step by step
5. Provide your final answer clearly

CRITICAL: You MUST explicitly state which information you retrieved from the Scaffolding DB. This helps track learning progress.

[Output Format]
Information Retrieved from Scaffolding DB:
- [List the specific concepts, strategies, or guidance you are using from the scaffolding]
- [Be specific about what you learned and will apply]

Improved Reasoning:
- Applying scaffolding guidance: [explain how you are using the scaffolding]
- Step-by-step solution: [your detailed improved solution]

Answer: [your final answer]
```

##### Iteration 2 Student 응답

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

#### Case B Step 5: Reconstruction

Teacher가 3회에 걸친 대화 히스토리를 분석하여, Scaffolding 과정에서의 학습 포인트를 통합한 정제된 응답을 생성합니다.

##### 실제 프롬프트 (1) — 대화 요약

**User Message** (`CONVERSATION_SUMMARIZATION_PROMPT`):

> Placeholder: `{problem_text}` → 문제 원문, `{ground_truth}` → "21", `{conversation_history}` → 3회 반복 전체 대화 히스토리

```
You are a teacher analyzing a tutoring session where a student struggled with a problem.

[Problem]
Valerie needs to put stamps on the envelopes she is about to mail...

[Correct Answer]
21

[Full Conversation History]
[Iteration 1~3의 Student 응답 + Teacher 평가 + Scaffolding Artifact 전체]

[Your Task]
Summarize this tutoring session concisely, focusing on what's important for understanding the student's learning gaps.

Extract and preserve:
1. The specific mathematical/logical errors in each attempt (not vague descriptions)
2. How the student's approach changed between iterations
3. Any recurring misconceptions or patterns
4. The final answer attempted in each iteration

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

> Placeholder: `{problem_text}` → 문제 원문, `{ground_truth}` → "21", `{task_analysis}` → Task Analysis Tree, `{iterations_needed}` → 3, `{conversation_summary}` → 위 대화 요약 결과, `{final_response}` → Student 3회차 최종 응답

```
You are an expert teacher reconstructing a successful learning outcome into clean SFT training data.

[Problem]
Valerie needs to put stamps on the envelopes she is about to mail...

[Correct Answer]
21

[Task Analysis]
[Task Analysis Tree 전체]

[Scaffolding Process Summary]
The student succeeded after 3 iterations.
[대화 요약 결과]

[Final Successful Response]
[Student 3회차 응답]

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

CRITICAL INSTRUCTIONS FOR JSON OUTPUT:
1. Your response MUST be ONLY valid JSON - no additional text before or after
2. Do NOT include explanations, comments, markdown code blocks, or any text outside the JSON
3. Do NOT include LaTeX expressions, mathematical notation, or equations outside JSON string values
4. Ensure ALL brackets { }, [ ], and quotes are properly closed
5. If you need to include mathematical expressions in the reconstructed_response, place them INSIDE the JSON string value with proper escaping (use double backslashes: \\)

Output ONLY the JSON object above.
```

##### Reconstruction 결과

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

#### Case C Step 5: Final Solution

Teacher가 Student의 약점을 분석한 뒤, 정답(70)을 기반으로 교육적 풀이를 생성합니다.

##### 실제 프롬프트

**User Message** (`TEACHER_FINAL_SOLUTION_PROMPT`):

> Placeholder: `{max_iterations}` → 5, `{problem_text}` → 문제 원문, `{ground_truth}` → "70", `{task_analysis}` → Task Analysis Tree, `{iterations_count}` → 5, `{scaffolding_history}` → 5회 반복 Scaffolding 히스토리, `{student_weaknesses}` → 반복 실패 분석 결과

```
You are a teacher providing a complete, correct solution after the student failed to solve the problem after 5 attempts.

[Problem]
Madeline has 5 boxes with 24 crayons in each. She noticed that 5/8 of the crayons in the 2 boxes were not yet used. In the 2 other boxes, only 2/3 of the crayons were used while the last box was not entirely used. How many unused crayons did Madeline have?

[Correct Answer]
70

[Instructional Analysis]
[Task Analysis Tree 전체]

[Scaffolding History]
The following scaffolding was provided across 5 iterations:
[5회 반복 Scaffolding Artifact + Teacher 평가 히스토리]

[Student's Persistent Weaknesses]
Based on the failed attempts, the student consistently struggled with:
- Interpreting "not entirely used" as meaning all 24 crayons are unused
- Making arbitrary assumptions (x=8, 1/4) instead of using given information
- Failing to recognize that "not entirely used" is a distinct condition from partial usage

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
  "solution_explanation": "Complete step-by-step solution with clear reasoning...",
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

### Case별 SFT 엔트리 예시

#### Case A 엔트리 (gsm8k_train_0)

```json
{
  "instruction": "Solve this math problem.\n\n## Learning Objective\nYour response should demonstrate: The model will solve complex mathematical word problems by setting up and solving equations, interpreting relationships, and performing calculations.\n\n## Problem-Solving Guidelines\nFollow the structured approach below to ensure a complete and well-reasoned solution:\n\n### Instructional Analysis Results\nTerminal Goal: Solve complex mathematical word problems... (Apply – Procedural Knowledge)\n ├── [1] Setting up equations from word problems...\n ...\n\n## Response Requirements\n1. Explicitly connect each step to the relevant sub-skill...\n2. Verify your intermediate results...\n3. Present your final answer clearly in the required format",
  "input": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "output": "#### Step-by-Step Reasoning\n\n1. **Identify Key Information and Relevant Mathematical Concepts**\n   - In April, Natalia sold 48 clips.\n   - In May, she sold half as many clips as in April.\n\n2. **Formulate a Plan to Solve the Problem**\n   - Calculate the number of clips sold in May.\n   - Add the number of clips sold in April and May to get the total.\n\n3. **Execute the Solution Plan Accurately**\n   - Clips sold in May = 48/2 = 24\n   - Total clips sold = 48 + 24 = 72\n\n4. **Communicate the Solution Clearly**\n   - The total number of clips Natalia sold is 72.\n\n### Final Answer\n\\boxed{72}",
  "metadata": {
    "id": "gsm8k_train_0",
    "sft_case": "A",
    "ground_truth": "72"
  }
}
```

> **Case A**: Student 원본 응답을 그대로 사용. 1회차에 모든 PO를 충족했으므로 추가 가공 없음.

#### Case B 엔트리 (gsm8k_train_116)

```json
{
  "instruction": "[Enhanced Instruction with Task Analysis — Case A와 동일한 구조]",
  "input": "Question: Valerie needs to put stamps on the envelopes she is about to mail...",
  "output": "[Teacher가 재구성한 정제된 풀이]\n\n[Understanding the Problem]\nValerie needs stamps for: 3 thank you cards, 2 bills, mail-in rebates, and job applications.\n\n[Setting Up Relationships]\n- Bills = 2 (water + electric)\n- Mail-in rebates = bills + 3 = 2 + 3 = 5\n- Job applications = 2 × rebates = 2 × 5 = 10\n\n[Calculating Stamps]\n- Thank you cards: 3 × 1 = 3 stamps\n- Water bill: 1 × 1 = 1 stamp\n- Electric bill: 1 × 2 = 2 stamps\n- Rebates: 5 × 1 = 5 stamps\n- Job applications: 10 × 1 = 10 stamps\n\nTotal = 3 + 1 + 2 + 5 + 10 = 21\n\nThe answer is \\boxed{21}",
  "metadata": {
    "id": "gsm8k_train_116",
    "sft_case": "B",
    "ground_truth": "21"
  }
}
```

> **Case B**: Teacher가 3회 반복의 학습 과정을 통합하여 재구성한 응답. Scaffolding 언급 없이 이상적인 학생 응답 형태로 변환됨.

#### Case C 엔트리 (gsm8k_train_187)

```json
{
  "instruction": "[Enhanced Instruction with Task Analysis — Case A와 동일한 구조]",
  "input": "Question: Madeline has 5 boxes with 24 crayons in each...",
  "output": "[Teacher가 생성한 교육적 풀이]\n\n[Understanding the Problem]\nMadeline has 5 boxes with 24 crayons in each.\n- 2 boxes: 5/8 of the crayons were not yet used\n- 2 other boxes: 2/3 of the crayons were used\n- Last box: was not entirely used (= all unused)\n\n[Step-by-Step Solution]\n1. 2 boxes (5/8 unused): 24 × 2 = 48 crayons, unused = 48 × 5/8 = 30\n2. 2 boxes (2/3 used → 1/3 unused): 24 × 2 = 48 crayons, unused = 48 × 1/3 = 16\n3. Last box (not entirely used = all 24 unused): 24\n\nTotal unused = 30 + 16 + 24 = 70\n\n[Common Pitfalls Addressed]\n- \"not entirely used\" means the box was not used at all, so all 24 crayons remain unused\n- Do not make assumptions about partial usage without explicit information\n\nAnswer: \\boxed{70}",
  "metadata": {
    "id": "gsm8k_train_187",
    "sft_case": "C",
    "ground_truth": "70"
  }
}
```

> **Case C**: Teacher가 Student의 반복적 약점(임의 가정, "not entirely used" 해석 오류)을 직접 보완한 교육적 풀이. `[Common Pitfalls Addressed]` 섹션이 포함됨.

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

### GSM8K + Qwen2.5-7B-Instruct 실행 결과

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

> **해석**: Qwen2.5-7B-Instruct 모델은 GSM8K 문제의 92.8%를 Task Analysis 기반 Enhanced Instruction만으로 1회에 해결했습니다. 5.88%는 Teacher의 Scaffolding을 통해 개선되었고, 1.0%만이 최대 반복 후에도 해결하지 못했습니다.

---

## 부록 A: 주요 개념 정리

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
| `TEACHER_INTERVENTION_PROMPT` | `prompts/learning_prompts.py` | Teacher PO 평가 + Socratic 질문 | Phase 2 / Step 2 | User | JSON |
| `SCAFFOLDING_ARTIFACT_PROMPT` | `prompts/learning_prompts.py` | 미충족 PO별 HOT/LOT Scaffolding 생성 | Phase 2 / Step 3 | User | JSON |
| `STUDENT_WITH_ARTIFACT_PROMPT` | `prompts/learning_prompts.py` | Student Scaffolding DB 참조 재응답 | Phase 2 / Step 4 | User | Text |
| `CONVERSATION_SUMMARIZATION_PROMPT` | `prompts/learning_prompts.py` | 튜터링 세션 대화 요약 | Phase 2 / Step 5 (Case B) | User | Text |
| `SUCCESSFUL_SCAFFOLDING_RECONSTRUCTION_PROMPT` | `prompts/learning_prompts.py` | 성공 학습 과정 → SFT 응답 재구성 | Phase 2 / Step 5 (Case B) | User | JSON |
| `TEACHER_FINAL_SOLUTION_PROMPT` | `prompts/learning_prompts.py` | 최대 반복 실패 후 교육적 풀이 생성 | Phase 2 / Step 5 (Case C) | User | JSON |
