"""교수설계 단계별 프롬프트 템플릿 모듈.

이 모듈은 Dick & Carey 교수설계 모델의 각 단계에서 사용하는 프롬프트 템플릿을 정의합니다.
LLM을 통해 교수설계 산출물(학습목표, 교수분석, 수행목표)을 자동 생성하는 데 활용됩니다.

프롬프트 상수 (system/user 분리):
    학습목표 (Step 0):
        - INSTRUCTIONAL_GOAL_SYSTEM_MESSAGE: 시스템 메시지 (역할 정의)
        - INSTRUCTIONAL_GOAL_USER_PROMPT: 사용자 프롬프트 (입력 데이터 + 지시)

    교수분석 (2단계):
        - INSTRUCTIONAL_ANALYSIS_SYSTEM_PROMPT: 시스템 메시지 (역할 정의)
        - INSTRUCTIONAL_ANALYSIS_USER_PROMPT: 사용자 프롬프트 (입력 데이터 + 지시)

    수행목표 (4단계):
        - PERFORMANCE_OBJECTIVES_SYSTEM_PROMPT: 시스템 메시지 (역할 정의)
        - PERFORMANCE_OBJECTIVES_USER_PROMPT: 사용자 프롬프트 (입력 데이터 + 지시)

Note:
    프롬프트 내용은 LLM이 이해하기 쉽도록 영어로 작성되어 있습니다.
    프롬프트 수정 시 출력 형식(Output Format)을 변경하지 마세요.
"""


# ==============================================================================
# Step 0: 학습목표 (Instructional Goal)
# ------------------------------------------------------------------------------
# 데이터셋 샘플을 분석하여 Instructional Goal(학습목표)을 도출합니다.
# Bloom 분류체계를 기반으로 인지적 수준을 결정합니다.
# ==============================================================================

INSTRUCTIONAL_GOAL_SYSTEM_MESSAGE = """You are an expert in instructional design and educational assessment.
Your role is to analyze learning materials and derive clear, measurable performance objectives.

Principles:
- Objectives must be Specific, Measurable, Achievable, and Relevant
- Focus on observable behaviors that can be assessed
- Consider the cognitive complexity required by the tasks

Respond with valid JSON only."""


INSTRUCTIONAL_GOAL_USER_PROMPT = """You are given a sample of items representing a specific task domain. These items are used to evaluate the student you are teaching. Your mission is to analyze the entire test set and determine a core instructional requirement that defines the instructional goal.

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
Below are {sample_count} representative samples from the dataset:

{train_data}


## Output (JSON)
{{
  "pattern_analysis": "Brief summary of common patterns found in samples",
  "cognitive_demands": ["list", "of", "required", "cognitive", "processes"],
  "instructional_goal": "The model will ...",
  "cognitive_level": "Remember|Understand|Apply|Analyze|Evaluate|Create",
  "primary_verb": "the main action verb used",
  "rationale": "Why this goal was chosen based on the analysis"
}}

"""


# ==============================================================================
# 2단계: 교수분석 (Instructional Analysis) — system/user 분리
# ------------------------------------------------------------------------------
# Dick & Carey 모델의 2단계로, 학습목표를 달성하기 위한
# 하위 기능(Subskills)과 과제(Subtasks)의 위계적 구조를 분석합니다.
# Anderson & Krathwohl의 개정된 Bloom 분류체계를 기반으로 합니다.
# system: 역할 정의 / user: 입력 데이터 + 지시 + 출력 형식
# ==============================================================================

INSTRUCTIONAL_ANALYSIS_SYSTEM_PROMPT = """You are an instructional design expert. Perform the Instructional Analysis step of the Dick & Carey model for the learning objective provided below."""

INSTRUCTIONAL_ANALYSIS_USER_PROMPT = """[Learning objective]: {learning_objective}

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
"""


# ==============================================================================
# 4단계: 수행목표 진술 (Performance Objectives) — system/user 분리
# ------------------------------------------------------------------------------
# Dick & Carey 모델의 4단계로, 교수분석 결과를 바탕으로
# 학습자가 달성해야 할 구체적인 수행목표를 진술합니다.
# 각 수행목표는 행동(Behavior), 조건(Condition), 기준(Criterion)을 하나의 문장으로 통합합니다.
# system: 역할 정의 / user: 입력 데이터 + 지시 + 출력 형식
# ==============================================================================

PERFORMANCE_OBJECTIVES_SYSTEM_PROMPT = """You are an instructional designer specializing in the Dick and Carey instructional design model, and a researcher in LLM learning methodologies.
Based on the provided Instructional Goal and Instructional Analysis Result, generate a set of Performance Objectives that will serve as the criteria for evaluating the observable performance within the LLM's reasoning process.
Specifically, they should be created using information from the learning outcomes identified in the Instructional Analysis Results."""

PERFORMANCE_OBJECTIVES_USER_PROMPT = """[Input Data]
Instructional Analysis Result: {instructional_analysis}

[Instructions]
For each Subskills and Subtasks in the instructional analysis, you must create at least one Performance Objective. You can create multiple performance objectives for subskills or subtasks that have more than one requirement.
Every Performance Objective must include all three components—Behavior, Condition, and Criterion—and each component must be explicitly stated in one sentence.
- Behavior: This is a description of LLM's intellectual skill including actions, content, and concepts.
- Condition: This is a description of the tools and resources that will be available to the learner when performing the skill. Write the conditions based solely on the data given in the problem or generated during the reasoning process. And it should always begin with 'given ~'.
- Criterion: This is a description of acceptable performance of the skill. The Criterion component must be tailored to the nature of the task: for tasks with correct answers, it must include a clear and measurable standard such as accuracy requirements, acceptable error ranges, or the number of correct responses; whereas for tasks with no single correct answer, it must specify the information or features that must be present for an acceptable response. Furthermore, these criteria must be formulated to evaluate the observable reasoning process within a single problem-solving task.
Each Performance Objective must correspond directly to a single Subskill and Subtask, and you must not add content that does not appear in the Instructional Analysis Result. Each performance objective must start with an action verb and must not include an explicit subject.

[Output Format]
Your output must be formatted as JSON, following this structure and no other form of explanation or commentary:

{{
  "performance_objectives": [
    {{
      "target": "Instructional Goal",
      "performance_objective": "A single sentence integrating behavior, condition, and criteria"
    }},
    {{
      "target": "Subskill X",
      "performance_objective": "A single sentence integrating behavior, condition, and criteria"
    }},
    {{
      "target": "Subtask X",
      "performance_objective": "A single sentence integrating behavior, condition, and criteria"
    }}
  ]
}}

Output ONLY the JSON object above. Do not include any additional text, explanation, or commentary outside the JSON structure.
"""
