"""교수설계 단계별 프롬프트 템플릿 모듈.

이 모듈은 Dick & Carey 교수설계 모델의 각 단계에서 사용하는 프롬프트 템플릿을 정의합니다.
LLM을 통해 교수설계 산출물(교수분석, 수행목표)을 자동 생성하는 데 활용됩니다.

프롬프트 상수:
    INSTRUCTIONAL_ANALYSIS_PROMPT: 교수분석 생성 프롬프트 (2단계)
    PERFORMANCE_OBJECTIVES_PROMPT: 수행목표 진술 생성 프롬프트 (4단계)

Note:
    프롬프트 내용은 LLM이 이해하기 쉽도록 영어로 작성되어 있습니다.
    프롬프트 수정 시 출력 형식(Output Format)을 변경하지 마세요.
"""


# ==============================================================================
# 2단계: 교수분석 (Instructional Analysis)
# ------------------------------------------------------------------------------
# Dick & Carey 모델의 2단계로, 학습목표를 달성하기 위한
# 하위 기능(Subskills)과 과제(Subtasks)의 위계적 구조를 분석합니다.
# Anderson & Krathwohl의 개정된 Bloom 분류체계를 기반으로 합니다.
# ==============================================================================

INSTRUCTIONAL_ANALYSIS_PROMPT = """
You are an instructional design expert. Perform the Instructional Analysis step of the Dick & Carey model for the learning objective provided below.

[Learning objective]: {learning_objective}

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
# 4단계: 수행목표 진술 (Performance Objectives)
# ------------------------------------------------------------------------------
# Dick & Carey 모델의 4단계로, 교수분석 결과를 바탕으로
# 학습자가 달성해야 할 구체적인 수행목표를 진술합니다.
# 각 수행목표는 행동(Behavior), 조건(Condition), 기준(Criterion)을 포함합니다.
# ==============================================================================

PERFORMANCE_OBJECTIVES_PROMPT = """
You are an instructional designer specializing in the Dick and Carey instructional design model, and a researcher in LLM learning methodologies.
Based on the provided Instructional Goal and Instructional Analysis Result, generate a set of Performance Objectives that will serve as the criteria for evaluating the observable performance within the LLM's reasoning process.

Performance objectives should be written using the guidelines provided in Anderson & Krathwohl's Taxonomy for Learning.
Specifically, they should be created using information from the learning outcomes identified in the Instructional Analysis Results.

[Input Data]
Instructional Analysis Result: {instructional_analysis}

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

Verbs Used by Cognitive Process Dimension (Behavior):
  - Remember: Recognizing, Recalling
  - Understand: Interpreting, Exemplifying, Classifying, Summarizing, Inferring, Comparing, Explaining
  - Apply: Executing, Implementing
  - Analyze: Differentiating, Organizing, Attributing
  - Evaluate: Checking, Critiquing
  - Create: Generating, Planning, Producing

Description of Knowledge Dimensions:
  - Factual Knowledge: Basic elements that must be mastered to solve subjects or problems in a subject
    · Terminology: Technical terms, musical symbols, etc.
    · Specific facts and elements: Key resources, reliable sources of information, etc.
  - Conceptual Knowledge: Interrelationships between basic elements within a superstructure that allows elements to function in an integrated manner
    · Classification and categories: Geological timescales, corporate ownership patterns, etc.
    · Principles and generalizations: The Pythagorean theorem, the law of supply and demand, etc.
    · Theories, models, and structures: Evolution, parliamentary organizations, etc.
  - Procedural Knowledge: Methods of performing tasks, methods of inquiry, criteria, algorithms, techniques, and methods for utilizing skills
    · Subject-specific functions and algorithms: Watercolor painting skills, integer division algorithms, etc.
    · Subject-specific techniques and methods: Interview techniques, scientific methods, etc.
    · Criteria for determining when to use appropriate procedures: Criteria for determining when to apply procedures involving Newton's second law, etc.
  - Metacognitive Knowledge: Awareness of knowledge cognition and knowledge of knowledge and cognition in general
    · Strategic knowledge: Knowledge of outlining as a means of understanding the structure of textbook units, knowledge of using heuristics, etc.
    · Cognitive tasks: Knowledge of the types of tests administered by specific teachers, knowledge of the cognitive demands of the task, etc.
    · Self-knowledge: The knowledge that critiquing papers is a personal strength, but writing papers is a personal weakness, and awareness of one's own level of knowledge

[Output Format]
Your output must be formatted as JSON, following this structure and no other form of explanation or commentary:

{{
  "performance_objectives": [
    {{
      "target": "Instructional Goal",
      "Behavior": "...",
      "Condition": "...",
      "Criterion": "..."
    }},
    {{
      "target": "Subskill [X]",
      "Behavior": "...",
      "Condition": "...",
      "Criterion": "..."
    }},
    {{
      "target": "Subtask [X-Y]",
      "Behavior": "...",
      "Condition": "...",
      "Criterion": "..."
    }}
  ]
}}

Output ONLY the JSON object above. Do not include any additional text, explanation, or commentary outside the JSON structure.
"""
