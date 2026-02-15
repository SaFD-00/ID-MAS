# ID-MAS 동작 예시: Phase 1~3 상세 설명

> GSM8K 데이터셋 + Qwen3-8B 모델 기반 실제 실행 로그를 통해 ID-MAS의 3-Phase 파이프라인이 어떻게 작동하는지 Case별로 설명합니다.
> **Student와 Teacher 모두 Qwen3-8B 모델**을 사용하며, 각 Step에서 LLM에 입력되는 실제 프롬프트를 포함하여 파이프라인 동작을 프롬프트 수준까지 설명합니다.

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
  ├── Step 1: Student 초기 응답
  ├── Step 2: Teacher PO 평가 (평가 전용) → 성공이면 Case A: Independent Performance Mastery/Case B: Scaffolded & Coached Mastery
  ├── Step 3: Scaffolding Artifact + 서술형 피드백 생성
  ├── Step 4: Student 재응답 (Teacher 피드백 참조)
  ├── (Step 2~4 반복, 최대 5회)
  ├── Step 5a-1: Teacher Positive Reinforcement (Case A: Independent Performance Mastery/Case B: Scaffolded & Coached Mastery) — 강점 + 개선점
  ├── Step 5a-2: Student Feedback-Driven Elaboration (Case A: Independent Performance Mastery/Case B: Scaffolded & Coached Mastery) — 응답 개선
  ├── Step 5b: Final Solution (Case C: Teacher Modeling Distillation만) — 교육적 풀이 평문 텍스트 출력
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
  "instructional_goal": "The model will solve multi-step mathematical problems by applying arithmetic operations, proportional reasoning, and real-world context understanding to arrive at accurate numerical solutions.",
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
Instructional Goal: The model will solve multi-step mathematical problems by applying
              arithmetic operations, proportional reasoning, and real-world context
              understanding to arrive at accurate numerical solutions. (Apply)

 ├── 1. Apply arithmetic operations to solve mathematical problems (Apply)
 │   ├── 1-1. Perform addition, subtraction, multiplication, and division accurately (Apply)
 │   ├── 1-2. Use order of operations to solve complex expressions (Apply)
 │   └── 1-3. Apply arithmetic operations in multi-step problems (Apply)

 ├── 2. Use proportional reasoning to solve mathematical problems (Apply)
 │   ├── 2-1. Interpret and solve ratios and proportions (Apply)
 │   ├── 2-2. Apply proportional reasoning to scale or convert measurements (Apply)
 │   └── 2-3. Use proportional reasoning in real-world scenarios (Apply)

 ├── 3. Understand and apply real-world context to mathematical problems (Apply)
 │   ├── 3-1. Identify relevant real-world information in a problem (Understand)
 │   ├── 3-2. Translate real-world scenarios into mathematical expressions (Apply)
 │   └── 3-3. Verify the relevance and accuracy of solutions in real-world contexts (Evaluate)
```

#### 실제 프롬프트

**System Message** (`INSTRUCTIONAL_ANALYSIS_SYSTEM_PROMPT`):
```
You are an instructional design expert. Perform the Instructional Analysis step of the Dick & Carey model for the learning objective provided below.
```

**User Message** (`INSTRUCTIONAL_ANALYSIS_USER_PROMPT`):

> Placeholder: `{learning_objective}` → 실제 GSM8K Instructional Goal

```
[Learning objective]: The model will solve multi-step mathematical problems by applying arithmetic operations, proportional reasoning, and real-world context understanding to arrive at accurate numerical solutions.

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

**생성 결과 (13개 PO):**

| # | Target | Performance Objective |
|---|--------|----------------------|
| 1 | Instructional Goal | Given a multi-step mathematical problem, the model will solve it by applying arithmetic operations, proportional reasoning, and real-world context understanding to arrive at an accurate numerical solution with 100% accuracy. |
| 2 | Subskill 1 | Given a mathematical problem requiring arithmetic operations, the model will perform addition, subtraction, multiplication, and division accurately with 100% accuracy. |
| 3 | Subtask 1-1 | Given a mathematical expression involving addition, subtraction, multiplication, and division, the model will perform these operations accurately with 100% accuracy. |
| 4 | Subtask 1-2 | Given a complex mathematical expression, the model will use the order of operations to solve it accurately with 100% accuracy. |
| 5 | Subtask 1-3 | Given a multi-step mathematical problem, the model will apply arithmetic operations in sequence to arrive at an accurate solution with 100% accuracy. |
| 6 | Subskill 2 | Given a mathematical problem requiring proportional reasoning, the model will interpret and solve ratios and proportions accurately with 100% accuracy. |
| 7 | Subtask 2-1 | Given a ratio or proportion problem, the model will interpret and solve it accurately with 100% accuracy. |
| 8 | Subtask 2-2 | Given a problem requiring scaling or converting measurements, the model will apply proportional reasoning to solve it accurately with 100% accuracy. |
| 9 | Subtask 2-3 | Given a real-world scenario involving proportional reasoning, the model will apply proportional reasoning to solve the problem accurately with 100% accuracy. |
| 10 | Subskill 3 | Given a mathematical problem with real-world context, the model will understand and apply real-world context to arrive at an accurate solution with 100% accuracy. |
| 11 | Subtask 3-1 | Given a mathematical problem with real-world context, the model will identify relevant real-world information accurately. |
| 12 | Subtask 3-2 | Given a real-world scenario, the model will translate it into a mathematical expression accurately. |
| 13 | Subtask 3-3 | Given a mathematical solution, the model will verify its relevance and accuracy in the real-world context with 100% accuracy. |

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
Instructional Goal: The model will solve multi-step mathematical problems by applying
              arithmetic operations, proportional reasoning, and real-world context
              understanding to arrive at accurate numerical solutions. (Apply)

 ├── 1. Apply arithmetic operations to solve mathematical problems (Apply)
 │   ├── 1-1. Perform addition, subtraction, multiplication, and division accurately (Apply)
 │   ├── 1-2. Use order of operations to solve complex expressions (Apply)
 │   └── 1-3. Apply arithmetic operations in multi-step problems (Apply)

 ├── 2. Use proportional reasoning to solve mathematical problems (Apply)
 │   ├── 2-1. Interpret and solve ratios and proportions (Apply)
 │   ├── 2-2. Apply proportional reasoning to scale or convert measurements (Apply)
 │   └── 2-3. Use proportional reasoning in real-world scenarios (Apply)

 ├── 3. Understand and apply real-world context to mathematical problems (Apply)
 │   ├── 3-1. Identify relevant real-world information in a problem (Understand)
 │   ├── 3-2. Translate real-world scenarios into mathematical expressions (Apply)
 │   └── 3-3. Verify the relevance and accuracy of solutions in real-world contexts (Evaluate)

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
    "instructional_goal": "The model will solve multi-step mathematical problems by applying arithmetic operations, proportional reasoning, and real-world context understanding to arrive at accurate numerical solutions.",
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

#### Case A: Independent Performance Mastery Step 1: Student 초기 응답

Student 모델이 Enhanced Instruction(Task Analysis 포함)을 참고하여 체계적 풀이를 생성합니다.

##### 실제 프롬프트

**System Message** (`LEARNING_TASK_SYSTEM_PROMPT`):

> Placeholder: `{instructional_goal}` → GSM8K Instructional Goal, `{task_analysis}` → Task Analysis Tree

```
The purpose of your response is to demonstrate the attainment of the Instructional Goal: The model will solve multi-step mathematical problems by applying arithmetic operations, proportional reasoning, and real-world context understanding to arrive at accurate numerical solutions.

You must adhere to the specific performance procedures and required knowledge/skills outlined in the Instructional Analysis results below. Ensure that your solution describes the full reasoning process using all provided steps and resources before arriving at the final answer.

[Instructional Analysis]
### Instructional Analysis Results
Instructional Goal: The model will solve multi-step mathematical problems by applying
              arithmetic operations, proportional reasoning, and real-world context
              understanding to arrive at accurate numerical solutions. (Apply)

 ├── 1. Apply arithmetic operations to solve mathematical problems (Apply)
 │   ├── 1-1. Perform addition, subtraction, multiplication, and division accurately (Apply)
 │   ├── 1-2. Use order of operations to solve complex expressions (Apply)
 │   └── 1-3. Apply arithmetic operations in multi-step problems (Apply)

 ├── 2. Use proportional reasoning to solve mathematical problems (Apply)
 │   ├── 2-1. Interpret and solve ratios and proportions (Apply)
 │   ├── 2-2. Apply proportional reasoning to scale or convert measurements (Apply)
 │   └── 2-3. Use proportional reasoning in real-world scenarios (Apply)

 ├── 3. Understand and apply real-world context to mathematical problems (Apply)
 │   ├── 3-1. Identify relevant real-world information in a problem (Understand)
 │   ├── 3-2. Translate real-world scenarios into mathematical expressions (Apply)
 │   └── 3-3. Verify the relevance and accuracy of solutions in real-world contexts (Evaluate)

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
- **Instructional goal alignment**: This solution demonstrates the instructional goal by applying
  arithmetic operations (addition and division) and real-world context understanding to solve a
  multi-step mathematical problem.

- **Step-by-step reasoning**:

  1. **Identify the given information**:
     - Natalia sold clips to **48 friends in April**.
     - In **May**, she sold **half as many clips** as she did in April.

  2. **Apply arithmetic operations**:
     - To find the number of clips sold in May, we calculate **half of 48**:
       $$
       \frac{48}{2} = 24
       $$
     - So, Natalia sold **24 clips in May**.

  3. **Add the number of clips sold in April and May**:
     $$
     48 + 24 = 72
     $$

  4. **Verify the real-world context**:
     - The problem involves a real-world scenario (selling clips), and the operations used
       (addition and division) are appropriate for calculating the total number of clips sold
       over two months.

- **Final answer**: The answer is $\boxed{72}$.
```

**추출된 답**: 72 (정답과 일치)

#### Case A: Independent Performance Mastery Step 2: Teacher PO 평가

Teacher가 Performance Objectives 기준으로 학생 응답을 **평가만** 수행합니다. 피드백은 생성하지 않습니다.

##### 실제 프롬프트

**System Message** (`FORMATIVE_ASSESSMENT_SYSTEM_PROMPT`):
```
You are a teacher supporting the learning of a student.

Your role is to evaluate the student's response against the established performance objectives. You must monitor the student's reasoning steps to ensure they meet the performance objectives.
```

**User Message** (`FORMATIVE_ASSESSMENT_USER_PROMPT`):

> Placeholder: `{problem_text}` → 문제 원문, `{student_response}` → 위 Student 응답, `{performance_objectives}` → Phase 1에서 생성된 13개 PO JSON, `{ground_truth}` → "72"

```
[Input Data]
- Problem: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
- Student response:
  - Instructional goal alignment: This solution demonstrates the instructional goal by applying
    arithmetic operations (addition and division) and real-world context understanding...
  [... 전체 응답 ...]
- Performance objectives: [13개 PO JSON 전체]
- Ground truth (FOR REFERENCE ONLY - DO NOT REVEAL): The answer is \boxed{72}

[Instructions]
Evaluate the student model's response according to the following rules.
1. Assess student performance according to the performance objectives. Use the criterion embedded in each performance objective as the evaluation standard. Do not reveal correct answers or model solutions.
2. Analyze the student response and determine which performance objectives are satisfied and which are not. All judgments must be grounded in observable reasoning behaviors in the student response, such as how claims are justified, how relationships are analyzed, or how judgments are formed. Avoid vague or abstract evaluations.
3. For each PO, write a "reasoning" that explains your evaluation:
   - State WHY you determined the objective is satisfied or not, citing specific evidence from the student's response (e.g., which reasoning steps, strategies, expressions, or omissions led to your judgment).
   - Then describe HOW the student could improve or elaborate: if satisfied, suggest ways to strengthen or deepen the demonstrated reasoning; if NOT satisfied, explain what specific changes or additions would help meet the objective.

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

##### Teacher 평가 결과

```json
{
  "performance_evaluation": [
    {
      "objective_content": "Given a multi-step mathematical problem, the model will solve it by applying arithmetic operations, proportional reasoning, and real-world context understanding to arrive at an accurate numerical solution with 100% accuracy.",
      "is_satisfied": true,
      "reasoning": "WHY: The student correctly solved the multi-step problem by applying arithmetic operations (division and addition) and proportional reasoning (interpreting 'half as many'). The real-world context of selling clips was also appropriately considered, and the final answer of 72 is accurate. HOW: The student could further strengthen the response by explicitly stating the relationship between the two months' sales before computing, making the proportional reasoning more transparent."
    },
    {
      "objective_content": "Given a mathematical problem requiring proportional reasoning, the model will interpret and solve ratios and proportions accurately with 100% accuracy.",
      "is_satisfied": true,
      "reasoning": "WHY: The student correctly interpreted the phrase 'half as many' as a proportional relationship and used division to compute the proportional value, demonstrating accurate proportional reasoning. HOW: The student could elaborate by explicitly defining the proportion (e.g., 'second month = first month / 2') to make the reasoning structure more visible."
    },
    {
      "objective_content": "Given a mathematical solution, the model will verify its relevance and accuracy in the real-world context with 100% accuracy.",
      "is_satisfied": true,
      "reasoning": "WHY: The student verified the solution by checking that the operations (division and addition) were appropriate for the real-world context of selling clips over two months. HOW: The student could improve by adding a final sanity check (e.g., verifying that the total is greater than either individual month's sales)."
    }
  ]
}
```

> **참고**: 위는 13개 PO 중 대표 3개만 발췌한 것입니다. 실제 평가에서는 13개 PO 전체에 대해 개별 평가가 수행됩니다.
>
> **평가 전용**: 현재 `FORMATIVE_ASSESSMENT_USER_PROMPT`는 `objective_content`, `is_satisfied`, `feedback`만 출력합니다. `overall_assessment`는 포함되지 않습니다.

**모든 PO 충족 (13/13)** → 반복 종료 → **Case A: Independent Performance Mastery 확정**

#### Case A: Independent Performance Mastery Step 3: Teacher Positive Reinforcement (Teacher 피드백)

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
[13개 PO 평가 JSON — 모두 is_satisfied: true]

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
- PO 1: The student effectively applied arithmetic operations (division and addition) and
  proportional reasoning to solve the multi-step problem. They accurately interpreted "half as
  many" as a proportional relationship and calculated the correct number of clips sold in May.
- PO 2: The student performed division and addition with 100% accuracy, correctly calculating
  48 ÷ 2 = 24 and 48 + 24 = 72.
- PO 6: The student demonstrated accurate proportional reasoning by correctly interpreting
  "half as many" and using division to find the proportional value.
- PO 13: The student verified the solution by checking that the operations (division and addition)
  were appropriate for the real-world context of selling clips over two months.

[Enhancement Suggestions]
1. Clarify the connection between "half as many" and the mathematical operation used: explicitly
   stating that "half as many" means dividing by 2 would strengthen the conceptual understanding.
2. Provide a brief explanation of why the operations are appropriate for the real-world context:
   a more explicit explanation of how the operations relate to the scenario would enhance reasoning.
3. Use a more structured format for the step-by-step reasoning: organizing the steps with more
   consistent labels would make the thought process even more transparent.

[Integration Guidance]
To incorporate these improvements naturally, the student can add a short sentence that explicitly
links the phrase "half as many" to the mathematical operation (e.g., "Since Natalia sold half as
many clips in May, we divide April's sales by 2 to find May's sales"). Additionally, they can
briefly explain how the operations used are relevant to the real-world scenario. Finally, they can
use consistent labels for each step to make the structure more organized and visually clear without
altering the core logic or final answer.
```

#### Case A: Independent Performance Mastery Step 4: Student Feedback-Driven Elaboration (Student 정교화)

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
- PO 1: The student effectively applied arithmetic operations...
[... 전체 Positive Feedback ...]

[Problem]
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
```

> **변경**: 기존에는 system에 지침 + 피드백이 모두 포함되고, user에는 `question["input"]`만 전달되었습니다. 이제 system (`FEEDBACK_DRIVEN_ELABORATION_SYSTEM_PROMPT`)에는 지침만, user (`FEEDBACK_DRIVEN_ELABORATION_USER_PROMPT`)에는 피드백 + 문제가 포함됩니다.

##### Feedback-Driven Elaboration 응답 결과

```
- **Instructional goal alignment**: This solution demonstrates the instructional goal by applying
  arithmetic operations (division and addition), proportional reasoning (interpreting "half as many"
  as a proportional relationship), and real-world context understanding (translating the scenario
  of selling clips into mathematical expressions). The solution also verifies the relevance and
  accuracy of the operations in the given context.

- **Step-by-step reasoning**:
  **Step 1: Identify the given information**
  Natalia sold 48 clips in April. In May, she sold half as many clips as she did in April.

  **Step 2: Calculate the number of clips sold in May**
  Since Natalia sold half as many clips in May, we divide the number of clips sold in April by 2.
  $$
  \text{Clips sold in May} = \frac{48}{2} = 24
  $$

  **Step 3: Calculate the total number of clips sold in April and May**
  To find the total number of clips sold over the two months, we add the number of clips sold
  in April and May.
  $$
  \text{Total clips sold} = 48 + 24 = 72
  $$

  **Step 4: Verify the relevance and accuracy of the solution**
  The operations used (division and addition) are appropriate for the real-world context of selling
  clips over two months. Division is used to scale down the number of clips sold in April by half
  to find the number sold in May, and addition is used to combine the totals for both months.
  The result is logically consistent with the problem description.

- **Final answer**: The answer is $\boxed{72}$.
```

#### Step 5 & 6: SFT 데이터 생성

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

---

### Case B: Scaffolded & Coached Mastery: 스캐폴딩 기반 숙달

> **문제 ID**: `gsm8k_train_6`
> **문제**: "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?"
> **정답**: 48

#### Iteration 1: Student 초기 응답 → PO 평가 부분 실패

**Step 1 — Student 초기 응답:**

(System Message = `LEARNING_TASK_SYSTEM_PROMPT`, User Message = 위 문제 원문 — Case A: Independent Performance Mastery Step 1과 동일한 프롬프트 구조)

```
- Instructional goal alignment: This solution demonstrates the instructional goal by applying
  arithmetic operations (multiplication and addition) to solve a multi-step problem, and by
  using real-world context (pizza slices) to translate the scenario into a mathematical expression.

- Step-by-step reasoning:
  1. **Identify the number of slices per pizza**:
     - A large pizza has 16 slices.
     - A small pizza has 8 slices.

  2. **Calculate the total number of slices from large pizzas**:
     Albert buys 2 large pizzas.
     $$
     2 \times 16 = 32 \text{ slices}
     $$

  3. **Calculate the total number of slices from small pizzas**:
     Albert buys 2 small pizzas.
     $$
     2 \times 8 = 16 \text{ slices}
     $$

  4. **Add the slices from large and small pizzas to find the total**:
     $$
     32 + 16 = 48 \text{ slices}
     $$

- Final answer: The answer is \boxed{48}
```

**추출된 답**: 48 (정답과 일치)

**Step 2 — Teacher PO 평가:**

```json
{
  "performance_evaluation": [
    {
      "objective_content": "Given a mathematical problem requiring proportional reasoning, the model will interpret and solve ratios and proportions accurately with 100% accuracy.",
      "is_satisfied": false,
      "reasoning": "WHY: The problem does not involve ratios or proportions, so this objective is not applicable. HOW: No improvement needed for this objective as it is outside the problem scope."
    },
    {
      "objective_content": "Given a ratio or proportion problem, the model will interpret and solve it accurately with 100% accuracy.",
      "is_satisfied": false,
      "reasoning": "WHY: The problem does not involve ratios or proportions, so this objective is not applicable. HOW: No improvement needed for this objective as it is outside the problem scope."
    },
    {
      "objective_content": "Given a problem requiring scaling or converting measurements, the model will apply proportional reasoning to solve it accurately with 100% accuracy.",
      "is_satisfied": false,
      "reasoning": "WHY: The problem does not involve scaling or converting measurements, so this objective is not applicable. HOW: No improvement needed for this objective as it is outside the problem scope."
    },
    {
      "objective_content": "Given a real-world scenario involving proportional reasoning, the model will apply proportional reasoning to solve the problem accurately with 100% accuracy.",
      "is_satisfied": false,
      "reasoning": "WHY: The problem does not involve proportional reasoning, so this objective is not applicable. HOW: No improvement needed for this objective as it is outside the problem scope."
    }
  ]
}
```

> **참고**: 위는 미충족 PO 4개만 발췌한 것입니다. 나머지 9개 PO(산술 연산, 실세계 맥락 관련)는 모두 `is_satisfied: true`입니다.

**미충족 PO (4/13):**

| PO # | Target | 미충족 이유 |
|------|--------|-----------|
| 6 | Subskill 2 | 비례 추론 미시연 |
| 7 | Subtask 2-1 | 비율/비례 개념 미활용 |
| 8 | Subtask 2-2 | 스케일링/변환 미적용 |
| 9 | Subtask 2-3 | 실세계 시나리오에서 비례 추론 미시연 |

> **핵심 관찰**: Student는 정답(48)을 도출했지만, 풀이 과정에서 **비례 추론(proportional reasoning)**을 명시적으로 시연하지 않았습니다. Teacher의 PO 평가는 정답 여부와 무관하게, Instructional Analysis에 정의된 모든 역량의 시연을 요구합니다. 이는 ID-MAS의 핵심 설계: **정답만으로는 충분하지 않으며, 추론 과정의 완전성이 평가 기준**임을 보여줍니다.

#### Case B: Scaffolded & Coached Mastery Step 3: Scaffolding Artifact 생성

Teacher가 미충족 PO별로 차별화된 Scaffolding과 **서술형 피드백**을 생성합니다.

##### 실제 프롬프트

**System Message** (`SCAFFOLDED_CORRECTIVE_FEEDBACK_SYSTEM_PROMPT`):
```
You are an instructional design expert (Dick & Carey model) creating a Scaffolding Artifact to help a student improve.

Your role is to design pedagogical scaffolding for Performance Objectives that the student failed to meet. This scaffolding will be stored as a "Scaffolding Artifact" that the student can reference in their next attempt.
```

**User Message** (`SCAFFOLDED_CORRECTIVE_FEEDBACK_USER_PROMPT`):

> Placeholder: `{problem_text}` → 문제 원문, `{student_response}` → Student의 응답, `{po_evaluation}` → Teacher PO 평가 JSON, `{previous_iteration_summaries}` → 이전 반복 요약 목록, `{instructional_goal}` → Instructional Goal, `{task_analysis}` → Task Analysis Tree

##### Scaffolding 결과 (발췌)

```
[Instructional Goal]
The model will solve multi-step mathematical problems by applying arithmetic operations,
proportional reasoning, and real-world context understanding to arrive at accurate
numerical solutions.

[Instructional Analysis]
### Instructional Analysis Results
[... Task Analysis Tree ...]

[Scaffolding for Task [1] (High Order Skill)]
- Target Objective: Given a mathematical problem requiring proportional reasoning, the model
  will interpret and solve ratios and proportions accurately with 100% accuracy.
- Cognitive Level: Analyze
- Failure Analysis: The student did not demonstrate any use of proportional reasoning, which
  is necessary for solving problems that involve comparing parts to a whole or scaling quantities.

  If High Order Skill:
  - Suggested Strategy:
    (a) Strategy 1: Consider how the number of pizza slices relates to the number of pizzas,
        and think about how the total number of slices can be expressed as a ratio of large
        to small pizzas.
      - Partial worked example (stop before the final answer):
        - Large pizzas to small pizzas ratio: 2:2
        - Total slices from large: 2 × 16 = 32
        - Total slices from small: 2 × 8 = 16
        - Total slices: 32 + 16 = ?
    (b) Strategy 2: Think about how the size of each pizza (large vs. small) affects the total
        number of slices.
      - Teacher's reasoning clarification:
        - A large pizza has twice as many slices as a small pizza, so the total number of
          slices will be influenced by the relative number of large and small pizzas.
  - Key Attention Points: Focus on how the size of the pizza (number of slices) relates to
    the quantity purchased, and how this can be expressed as a ratio or proportion.

[Scaffolding for Task [2] (High Order Skill)]
- Target Objective: Given a ratio or proportion problem, the model will interpret and solve
  it accurately with 100% accuracy.
- Cognitive Level: Analyze
- Failure Analysis: The student did not engage with ratio or proportion concepts.
  [... 전략 및 부분 예시 ...]

[Scaffolding for Task [3] (High Order Skill)]
- Target Objective: Given a problem requiring scaling or converting measurements, the model
  will apply proportional reasoning to solve it accurately with 100% accuracy.
- Cognitive Level: Analyze
- Failure Analysis: The student did not use proportional reasoning to scale the number of
  slices or convert between pizza sizes.
  [... 전략 및 부분 예시 ...]

[Feedback]
The student correctly applied arithmetic operations and translated the real-world scenario
into a mathematical expression, but did not engage with proportional reasoning, which is
required for problems involving ratios, scaling, or comparisons between quantities. To improve,
focus on how the number of slices per pizza relates to the total number of pizzas purchased,
and consider how this can be expressed as a ratio or proportion. Start by identifying the
ratio of large to small pizzas and how this affects the total number of slices. Then, verify
your calculations by checking each step.

[Iteration Summary]
The student correctly solved the problem using arithmetic operations and real-world context,
but did not demonstrate proportional reasoning, which is required for several performance
objectives. Scaffolding was provided to guide the student in understanding ratios, scaling,
and how to apply proportional reasoning in similar problems.
```

> **핵심 변경**: 출력 형식이 JSON에서 구조화된 마크다운으로 변경되었습니다. `[Feedback]` 섹션은 서술형 단락으로, (1) 오류 분석, (2) 개선 방향, (3) 검증 단계를 통합합니다. 전체 Scaffolding Artifact 텍스트(`_raw_text`)가 학생에게 직접 전달됩니다.

#### Case B: Scaffolded & Coached Mastery Step 4: Student 재응답

Student가 Teacher의 **Scaffolding Artifact**를 참조하여 개선된 응답을 생성합니다.

##### 실제 프롬프트

**System Message** (`TEACHER_SUPPORTED_REATTEMPT_SYSTEM_PROMPT`):

> Placeholder: `{scaffolding_system_prompt}` → `LEARNING_TASK_SYSTEM_PROMPT` (채워진 상태)

```
[dataset_prompt — 원본 instruction]

[LEARNING_TASK_SYSTEM_PROMPT — Task Analysis 포함]

[Instructions]
1. Carefully study the scaffolding artifact above, including the strategies and examples provided
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

[Instructional Goal]
The model will solve multi-step mathematical problems...

[Scaffolding for Task [1] (High Order Skill)]
- Target Objective: Given a mathematical problem requiring proportional reasoning...
[... 전체 Scaffolding Artifact ...]

[Problem]
Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?
```

##### Iteration 2 Student 응답

```
- Instructional goal alignment: This solution demonstrates the instructional goal by applying
  arithmetic operations to calculate the total number of pizza slices, using proportional
  reasoning to understand the relationship between the number of pizzas and their slice counts,
  and applying real-world context to interpret the scenario accurately.

- Step-by-step reasoning:
  1. **Identify the number of slices per pizza**:
     - A large pizza has 16 slices.
     - A small pizza has 8 slices.

  2. **Calculate the total number of slices from large pizzas**:
     - Albert buys 2 large pizzas.
     - Total slices from large pizzas = $ 2 \times 16 = 32 $ slices.

  3. **Calculate the total number of slices from small pizzas**:
     - Albert buys 2 small pizzas.
     - Total slices from small pizzas = $ 2 \times 8 = 16 $ slices.

  4. **Add the total slices from large and small pizzas**:
     - Total slices = $ 32 + 16 = 48 $ slices.

  5. **Verify the real-world context**:
     - Albert eats all the pizza, so the total number of slices he eats is indeed the sum of
       all slices from both large and small pizzas.

- Final answer: The answer is \boxed{48}
```

**추출된 답**: 48 (정답과 일치)

**Step 2 — Teacher PO 평가 (Iteration 2):**

**모든 PO 충족 (13/13)** → 반복 종료 → **Case B: Scaffolded & Coached Mastery 확정** (2회차 성공)

> Iteration 2에서는 Student가 비례 추론을 풀이에 반영하여, 이전에 미충족이었던 PO 6~9가 모두 충족되었습니다. Teacher는 "The student interpreted the proportional relationship between the number of pizzas and the number of slices"로 평가했습니다.

#### Case B: Scaffolded & Coached Mastery Step 5: Feedback-Driven Elaboration

Case B: Scaffolded & Coached Mastery에서도 Case A: Independent Performance Mastery와 동일하게, PO가 모두 충족된 후 **Teacher Positive Reinforcement → Student Feedback-Driven Elaboration** 과정을 수행합니다.

```python
# nodes.py
# Case A: Independent Performance Mastery / Case B: Scaffolded & Coached Mastery: PO 충족 후 Feedback-Driven Elaboration
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

**SFT output = Iteration 2의 Student Feedback-Driven Elaboration 응답**

##### Feedback-Driven Elaboration 응답 결과

```
- Instructional goal alignment: This solution demonstrates the instructional goal by applying
  arithmetic operations to calculate the total number of pizza slices, using proportional reasoning
  to understand the relationship between the number of pizzas and slices, and applying real-world
  context understanding by interpreting the scenario of Albert eating all the pizza.

- Step-by-step reasoning:
  1. **Identify real-world information**: Albert buys 2 large pizzas and 2 small pizzas. A large
     pizza has 16 slices, and a small pizza has 8 slices. He eats all the pizza.
  2. **Apply arithmetic operations**:
     - Calculate the total number of slices from large pizzas:
       $ 2 \text{ large pizzas} \times 16 \text{ slices per large pizza} = 32 \text{ slices} $
     - Calculate the total number of slices from small pizzas:
       $ 2 \text{ small pizzas} \times 8 \text{ slices per small pizza} = 16 \text{ slices} $
     - Add the slices from both types of pizzas:
       $ 32 \text{ slices (large)} + 16 \text{ slices (small)} = 48 \text{ slices} $
  3. **Apply proportional reasoning**: The number of slices is directly proportional to the number
     of pizzas. Since each large pizza has twice as many slices as a small pizza, the total number
     of slices increases proportionally with the number of pizzas. This proportional relationship
     allows us to scale the number of slices based on the quantity of pizzas.
  4. **Verify real-world context**: Since Albert eats all the pizza, the total number of slices he
     consumes is equal to the total number of slices available, which is 48.

- Final answer: The answer is $\boxed{48}$
```

> **설계 의도**: Scaffolding 과정을 거쳐 Student가 자력으로 PO를 충족시킨 후, Teacher의 Positive Reinforcement를 통해 응답의 reasoning을 더욱 강화합니다. PO 평가의 강점/개선점을 기존 풀이에 녹여 SFT 데이터 품질을 향상시킵니다.

#### 최종 로그 요약

```json
{
  "id": "gsm8k_train_6",
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
Iteration 1: Student(48✓) → Teacher(9/13 PO) → Scaffolding(3 HOT) + 서술형 피드백
Iteration 2: Student(48✓) → Teacher(13/13 PO ✓) → Case B: Scaffolded & Coached Mastery 확정
     ↓
Feedback-Driven Elaboration: Teacher Positive Reinforcement → Student Feedback-Driven Elaboration → Refined Response
     ↓
Case B: Scaffolded & Coached Mastery: Refined Response를 SFT 데이터로 사용
```

> **핵심 관찰**: Student는 1회차에서 정답(48)을 도출했지만, 풀이에서 비례 추론(proportional reasoning)을 명시적으로 시연하지 않아 4개 PO가 미충족되었습니다. Teacher의 HOT Scaffolding에서 "large pizza has twice as many slices as a small pizza"와 같은 비례 관계를 분석하도록 안내한 후, 2회차에서 비례 추론을 풀이에 통합하여 모든 PO를 충족했습니다. 이는 **정답만으로는 충분하지 않고, 추론 과정의 완전성이 평가 기준**인 ID-MAS의 특성을 보여주는 사례입니다.

---

### Case C: Teacher Modeling Distillation: 교사 모델링 증류

> **문제 ID**: `gsm8k_train_59`
> **문제**: "Every hour Joanne has to collect the coins out of the fountain inside the mall. During the first hour, she collected 15 coins. For the next two hours, she collected 35 coins from the fountain. In the fourth hour, she collected 50 coins from the fountain but she gave 15 of them to her coworker so she could buy a soda. How many coins did she have after the fourth hour?"
> **정답**: 120

5회 반복 후에도 모든 PO를 충족하지 못한 경우 → **Case C: Teacher Modeling Distillation 확정**

#### Iteration 1~5 요약

| Iteration | Student 답 | 정답 일치 | 충족 PO | 미충족 PO | 핵심 오류 |
|-----------|-----------|----------|---------|----------|----------|
| 1 | 85 | ✗ | 3/13 | 10/13 | "35 coins for two hours"를 총 35로 해석 |
| 2 | 85 | ✗ | 4/13 | 9/13 | 동일 해석 오류 지속 |
| 3 | 85 | ✗ | 0/13 | 13/13 | 동일 해석 오류 지속 |
| 4 | 85 | ✗ | 3/13 | 10/13 | 동일 해석 오류 지속 |
| 5 | 85 | ✗ | 9/13 | 4/13 | 동일 해석 오류 지속 |

> **핵심 관찰**: Student(Qwen3-8B)는 5회 반복 모두에서 "For the next two hours, she collected 35 coins"를 **2시간 동안 총 35개**로 해석하여 일관되게 85를 답했습니다. 정답은 **매 시간 35개 × 2시간 = 70개**이므로 15 + 70 + 35 = 120입니다. Teacher의 Scaffolding에도 불구하고 문제 해석 오류가 교정되지 않은 사례입니다.

#### Iteration 1 상세

**Student 응답:**

```
- Instructional goal alignment: This solution demonstrates the instructional goal by applying
  arithmetic operations (addition and subtraction) and real-world context understanding to solve
  a multi-step mathematical problem involving coin collection.

- Step-by-step reasoning:
  1. **First hour**: Joanne collected 15 coins.
  2. **Next two hours**: She collected 35 coins (total for two hours).
  3. **Fourth hour**: She collected 50 coins but gave 15 to her coworker.
     Net coins from fourth hour = 50 - 15 = 35 coins.
  4. **Total**: 15 + 35 + 35 = 85 coins.

- Final answer: The answer is \boxed{85}
```

**Teacher PO 평가 (Iteration 1):** 3/13 충족, 10/13 미충족

**Scaffolding Artifact (Iteration 1, 발췌):**

```
[Scaffolding for Task [1] (Low Order Skill)]
- Target Objective: Given a multi-step mathematical problem, the model will solve it by applying
  arithmetic operations, proportional reasoning, and real-world context understanding to arrive
  at an accurate numerical solution with 100% accuracy.
- Cognitive Level: Apply
- Failure Analysis: The student arrived at 85 coins, which is incorrect. The error lies in
  interpreting "for the next two hours, she collected 35 coins" as 35 total, rather than
  35 per hour.
- Missed Concept/Information: The problem states "For the next two hours, she collected 35 coins
  from the fountain." This means she collected 35 coins EACH hour for the next two hours,
  not 35 coins total.
- Brief Explanation: When a problem says "for the next two hours, she collected 35 coins,"
  it means 35 coins per hour for 2 hours, totaling 70 coins.

[Feedback]
The student made a critical error in interpreting the problem statement. The phrase "For the
next two hours, she collected 35 coins" means 35 coins PER HOUR for 2 hours, not 35 coins
total. This misinterpretation led to an incorrect total. Re-read the problem carefully and
recalculate: first hour (15) + second hour (35) + third hour (35) + fourth hour (50 - 15).

[Iteration Summary]
The student attempted to solve the coin collection problem by adding up the coins from each
time period and subtracting the coins given away. However, the student misinterpreted the
phrase "for the next two hours, she collected 35 coins" as 35 total rather than 35 per hour.
Scaffolding was provided to clarify the correct interpretation.
```

#### Iterations 2~5

각 iteration에서 Student는 동일한 해석 오류를 반복하여 85를 답했습니다. Teacher는 매 iteration마다 해석 오류를 지적하는 Scaffolding을 제공했지만, Student는 5회 반복 후에도 해석을 교정하지 못했습니다.

**Iteration 5 Teacher 평가:** 9/13 PO 충족, 4/13 미충족

미충족 PO:
1. **Instructional Goal PO**: 최종 답 85 부정확
2. **Subtask 1-3**: 산술 연산 순서적 적용 오류
3. **Subskill 3**: 실세계 맥락 잘못 적용
4. **Subtask 3-3**: 해답 검증 실패

→ **5회 반복 후 PO 미충족** → **Case C: Teacher Modeling Distillation 확정**

#### Case C: Teacher Modeling Distillation Step 5b: Final Solution

Teacher가 전체 iteration history를 참고하여 정답(120)을 기반으로 Step 1과 동일한 형식의 풀이를 **평문 텍스트**로 생성합니다.

##### 실제 프롬프트

**System Message** (`TEACHER_MODELING_SYSTEM_PROMPT`):

> Placeholder: `{instructional_goal}` → Instructional Goal 텍스트, `{task_analysis}` → Task Analysis Tree (최대 1500자 제한)

```
The purpose of your response is to demonstrate the attainment of the Instructional Goal: [Instructional Goal 텍스트]

You must adhere to the specific performance procedures and required knowledge/skills outlined in the Instructional Analysis results below. Ensure that your solution describes the full reasoning process using all provided steps and resources before arriving at the final answer.

[Instructional Analysis]
[Task Analysis Tree (최대 1500자)]
```

**User Message** (`TEACHER_MODELING_USER_PROMPT`):

> Placeholder: `{problem_text}` → 문제 원문, `{ground_truth}` → "120", `{iteration_history}` → `_format_iteration_history()`로 포맷된 전체 iteration history

```
[Problem]
Every hour Joanne has to collect the coins out of the fountain inside the mall. During the first hour, she collected 15 coins. For the next two hours, she collected 35 coins from the fountain. In the fourth hour, she collected 50 coins from the fountain but she gave 15 of them to her coworker so she could buy a soda. How many coins did she have after the fourth hour?

[Ground Truth]
120

[Iteration History]
The following is a summary of each iteration's student attempt and teacher scaffolding:
Iter 1: The student attempted to solve the problem but misinterpreted "for the next two hours, she collected 35 coins" as 35 total instead of 35 per hour. The student's answer was 85. Scaffolding provided focused on clarifying the per-hour interpretation.
Iter 2: The student corrected the per-hour interpretation but made an arithmetic error in the subtraction step. The student's answer was 125. Scaffolding provided guided the student to verify each calculation step.
...

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

##### Teacher 최종 풀이 결과

```
- Instructional goal alignment: This solution demonstrates the instructional goal by systematically applying multi-step arithmetic operations with careful attention to per-unit interpretation and conditional deductions.

- Step-by-step reasoning:
Step 1: Calculate the coins collected in the first hour
Joanne collected 15 coins in the first hour.
Total so far = 15 coins

Step 2: Calculate the coins collected in the next two hours
For the next two hours, she collected 35 coins each hour.
Total for the second and third hours = 35 + 35 = 70 coins

Step 3: Calculate the coins collected in the fourth hour
In the fourth hour, Joanne collected 50 coins, but she gave 15 of them to her coworker.
Therefore, the number of coins she kept from the fourth hour = 50 – 15 = 35 coins

Step 4: Add up all the coins collected over the four hours
Total coins = (First hour) + (Second and third hours) + (Fourth hour)
Total coins = 15 + 70 + 35 = 120 coins

- Final answer: "The answer is \boxed{120}"
```

> **핵심 차이**: Teacher의 풀이는 "For the next two hours, she collected 35 coins"를 **매 시간 35개**로 해석하여 35 + 35 = 70으로 계산합니다. Student가 5회 반복 동안 고수한 "2시간 총 35개" 해석 오류를 명시적으로 교정하는 교육적 풀이입니다.

#### 최종 로그 요약

```json
{
  "id": "gsm8k_train_59",
  "sft_case": "case_c_teacher_modeling_distillation",
  "iterative_scaffolding": {
    "success": false,
    "iterations_needed": 5
  },
  "hot_count": null,
  "lot_count": null,
  "sft_response": "[Teacher의 최종 풀이 — 위 텍스트]"
}
```

**Case C: Teacher Modeling Distillation 흐름 요약:**
```
Iteration 1: Student(85✗) → Teacher(3/13 PO) → Scaffolding → "35 per hour" 교정 시도
Iteration 2: Student(85✗) → Teacher(4/13 PO) → Scaffolding → 교정 실패
Iteration 3: Student(85✗) → Teacher(0/13 PO) → Scaffolding → 교정 실패
Iteration 4: Student(85✗) → Teacher(3/13 PO) → Scaffolding → 교정 실패
Iteration 5: Student(85✗) → Teacher(9/13 PO) → 5회 초과 → Case C: Teacher Modeling Distillation 확정
     ↓
Teacher Final Solution: 정답(120) 기반 교육적 풀이 생성 → SFT 데이터로 사용
```

> **핵심 관찰**: 이 사례는 Student 모델이 **문제 해석 오류**를 5회 Scaffolding에도 교정하지 못한 경우입니다. "For the next two hours, she collected 35 coins"의 해석이 핵심 쟁점이며, Student는 일관되게 "총 35"로 해석한 반면 ground truth는 "매 시간 35"입니다. 이런 경우 Teacher가 교육적 모범 풀이를 제공하여 SFT 데이터의 품질을 보장합니다.

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
  "instruction": "You are a helpful math assistant.\nSolve this mathematical problem step by step...\n\nThe purpose of your response is to demonstrate the attainment of the Instructional Goal: The model will solve multi-step mathematical problems by applying arithmetic operations, proportional reasoning, and real-world context understanding to arrive at accurate numerical solutions.\n\nYou must adhere to the specific performance procedures...\n\n[Instructional Analysis]\n### Instructional Analysis Results\n...\n\n[Instructions]\n1. Identify which skills...\n...\n\n[Output Format]\n- Instructional goal alignment: ...\n- Step-by-step reasoning: ...\n- Final answer: \"The answer is \\boxed{your final answer}\"",
  "input": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "output": "- **Instructional goal alignment**: This solution demonstrates the instructional goal by applying arithmetic operations (division and addition), proportional reasoning (interpreting \"half as many\" as a proportional relationship), and real-world context understanding...\n- **Step-by-step reasoning**:\n  **Step 1: Identify the given information** ...\n  **Step 2: Calculate the number of clips sold in May** ... 48/2 = 24\n  **Step 3: Calculate the total** ... 48 + 24 = 72\n  **Step 4: Verify the relevance and accuracy** ...\n- **Final answer**: The answer is $\\boxed{72}$.",
  "metadata": {
    "id": "gsm8k_train_0",
    "sft_case": "case_a_independent_performance_mastery",
    "ground_truth": "72"
  }
}
```

> **Case A: Independent Performance Mastery**: PO 충족 후 Teacher Positive Reinforcement → Student Feedback-Driven Elaboration → Refined Response를 SFT output으로 사용.

#### Case B: Scaffolded & Coached Mastery 엔트리 (gsm8k_train_6)

```json
{
  "instruction": "Original Instruction + LEARNING_TASK_SYSTEM_PROMPT (동적 결합) — Case A: Independent Performance Mastery와 동일한 구조",
  "input": "Question: Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
  "output": "- Instructional goal alignment: This solution demonstrates the instructional goal by applying arithmetic operations to calculate the total number of pizza slices, using proportional reasoning to understand the relationship between the number of pizzas and slices...\n- Step-by-step reasoning:\n  1. **Identify real-world information**: Albert buys 2 large pizzas and 2 small pizzas...\n  2. **Apply arithmetic operations**: 2×16=32, 2×8=16, 32+16=48\n  3. **Apply proportional reasoning**: The number of slices is directly proportional to the number of pizzas...\n  4. **Verify real-world context**: Since Albert eats all the pizza, the total is 48.\n- Final answer: The answer is $\\boxed{48}$",
  "metadata": {
    "id": "gsm8k_train_6",
    "sft_case": "case_b_scaffolded_coached_mastery",
    "ground_truth": "48"
  }
}
```

> **Case B: Scaffolded & Coached Mastery**: Scaffolding을 통해 PO를 충족한 후 Teacher Positive Reinforcement → Student Feedback-Driven Elaboration → Refined Response를 SFT output으로 사용.

#### Case C: Teacher Modeling Distillation 엔트리 (gsm8k_train_59)

```json
{
  "instruction": "Original Instruction + LEARNING_TASK_SYSTEM_PROMPT (동적 결합) — Case A: Independent Performance Mastery와 동일한 구조",
  "input": "Question: Every hour Joanne has to collect the coins out of the fountain inside the mall. During the first hour, she collected 15 coins. For the next two hours, she collected 35 coins from the fountain. In the fourth hour, she collected 50 coins from the fountain but she gave 15 of them to her coworker so she could buy a soda. How many coins did she have after the fourth hour?",
  "output": "[Understanding the Problem]\nLet me analyze this problem step by step...\n\n[Key Concepts Applied]\nThe key concepts here include: Accurate addition and subtraction, Order of operations, Real-world context understanding...\n\n[Step-by-Step Solution]\nStep 1: First hour: 15 coins\nStep 2: Next two hours: 35 + 35 = 70 coins\nStep 3: Fourth hour: 50 – 15 = 35 coins\nStep 4: Total: 15 + 70 + 35 = 120 coins\n\nThe answer is \\boxed{120}",
  "metadata": {
    "id": "gsm8k_train_59",
    "sft_case": "case_c_teacher_modeling_distillation",
    "ground_truth": "120"
  }
}
```

> **Case C: Teacher Modeling Distillation**: Teacher가 교육적 풀이를 평문 텍스트로 생성. 5회 Scaffolding 후에도 Student가 문제 해석 오류를 교정하지 못해 Teacher의 모범 풀이를 SFT 데이터로 사용.

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
| Step 5 (재구성) | 재구성 실패 | Case B: Scaffolded & Coached Mastery: 학생 최종 응답 / Case C: Teacher Modeling Distillation: ground_truth 기반 |

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
| `INSTRUCTIONAL_ANALYSIS_USER_PROMPT` | `prompts/design_prompts.py` | Learning Objective → Task Analysis Tree 분해 | Phase 1 / Step 2 | User | Text (Tree) |
| `PERFORMANCE_OBJECTIVES_SYSTEM_PROMPT` | `prompts/design_prompts.py` | PO 생성 전문가 역할 설정 | Phase 1 / Step 3 | System | — |
| `PERFORMANCE_OBJECTIVES_USER_PROMPT` | `prompts/design_prompts.py` | Task Analysis → Performance Objectives 생성 | Phase 1 / Step 3 | User | JSON |
| `LEARNING_TASK_SYSTEM_PROMPT` | `prompts/learning_prompts.py` | Student 문제 해결 시스템 프롬프트 | Phase 2 / Step 1 | System | Text |
| `FORMATIVE_ASSESSMENT_SYSTEM_PROMPT` | `prompts/learning_prompts.py` | Teacher 평가 역할 설정 | Phase 2 / Step 2 | System | — |
| `FORMATIVE_ASSESSMENT_USER_PROMPT` | `prompts/learning_prompts.py` | Teacher PO 평가 (평가 전용) | Phase 2 / Step 2 | User | JSON |
| `SCAFFOLDED_CORRECTIVE_FEEDBACK_SYSTEM_PROMPT` | `prompts/learning_prompts.py` | Scaffolding 생성 역할 설정 | Phase 2 / Step 3 | System | — |
| `SCAFFOLDED_CORRECTIVE_FEEDBACK_USER_PROMPT` | `prompts/learning_prompts.py` | 미충족 PO별 HOT/LOT Scaffolding + Feedback 생성 | Phase 2 / Step 3 | User | Structured Text |
| `TEACHER_SUPPORTED_REATTEMPT_SYSTEM_PROMPT` | `prompts/learning_prompts.py` | Student 재응답: `dataset_prompt` + `LEARNING_TASK_SYSTEM_PROMPT` 기반 지침 | Phase 2 / Step 4 | System | Text |
| `TEACHER_SUPPORTED_REATTEMPT_USER_PROMPT` | `prompts/learning_prompts.py` | Student 재응답: Scaffolding Artifact + 문제 전달 | Phase 2 / Step 4 | User | Text |
| `POSITIVE_REINFORCEMENT_SYSTEM_PROMPT` | `prompts/learning_prompts.py` | Positive Reinforcement 역할 설정 | Phase 2 / Step 5a-1 (Case A: Independent Performance Mastery / Case B: Scaffolded & Coached Mastery) | System | — |
| `POSITIVE_REINFORCEMENT_USER_PROMPT` | `prompts/learning_prompts.py` | PO 충족 후 강점 + 개선점 피드백 생성 | Phase 2 / Step 5a-1 (Case A: Independent Performance Mastery / Case B: Scaffolded & Coached Mastery) | User | Structured Text |
| `FEEDBACK_DRIVEN_ELABORATION_SYSTEM_PROMPT` | `prompts/learning_prompts.py` | Positive Reinforcement 기반 응답 정교화 지침 | Phase 2 / Step 5a-2 (Case A: Independent Performance Mastery / Case B: Scaffolded & Coached Mastery) | System | Text |
| `FEEDBACK_DRIVEN_ELABORATION_USER_PROMPT` | `prompts/learning_prompts.py` | Positive Reinforcement + 문제 전달 | Phase 2 / Step 5a-2 (Case A: Independent Performance Mastery / Case B: Scaffolded & Coached Mastery) | User | Text |
| `TEACHER_MODELING_SYSTEM_PROMPT` | `prompts/learning_prompts.py` | Final Solution 역할 설정 | Phase 2 / Step 5b (Case C: Teacher Modeling Distillation) | System | — |
| `TEACHER_MODELING_USER_PROMPT` | `prompts/learning_prompts.py` | 최대 반복 실패 후 교육적 풀이 생성 | Phase 2 / Step 5b (Case C: Teacher Modeling Distillation) | User | Text (평문) |
