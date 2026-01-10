"""
교수 설계 단계별 프롬프트 템플릿
"""


# ==============================================================================
# 2단계: 교수 분석 (Instructional Analysis)
# ==============================================================================

INSTRUCTIONAL_ANALYSIS_PROMPT = """
You are an instructional design expert. Perform the Instructional Analysis step of the Dick & Carey model for the learning objective provided below.

[Learning Goal]
{learning_objective}

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

Present prerequisite knowledge separately in a section titled "Prerequisite Knowledge."
Each item must follow this formatting:
"Function Number / Learning Type (Cognitive Process – Knowledge Dimension) / Prerequisite Knowledge Description"

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

### Prerequisite Knowledge
Function Number / (Cognitive Process – Knowledge Dimension) / Prerequisite Knowledge Description
Function Number / (Cognitive Process – Knowledge Dimension) / Prerequisite Knowledge Description
Function Number / (Cognitive Process – Knowledge Dimension) / Prerequisite Knowledge Description

[Requirements]
- Maintain the exact structure, titles, line breaks, and tree characters (├──, │, └──).
- Do not change section names ("Instructional Analysis Results", "Prerequisite Knowledge").
- Output only the required instructional analysis products; do not include introductions, explanations, or references.
"""


# ==============================================================================
# 4단계: 수행목표 진술 (Performance Objectives)
# ==============================================================================

PERFORMANCE_OBJECTIVES_PROMPT = """
You are an instructional designer specializing in the Dick and Carey instructional design model, and a researcher in LLM learning methodologies.
Based on the provided Terminal Goal and Instructional Analysis Result, generate a set of Performance Objectives that will serve as the criteria for evaluating the observable performance within the LLM's reasoning process.

Performance objectives should be written using the guidelines provided in Anderson & Krathwohl's Taxonomy for Learning.
Specifically, they should be created using information from the learning outcomes identified in the Instructional Analysis Results.

[Input Data]
Instructional Analysis Result: {instructional_analysis}

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
      "target": "Terminal Goal",
      "Behavior": "...",
      "Condition": "...",
      "Criterion": "..."
    }},
    {{
      "target": "Subskill X",
      "Behavior": "...",
      "Condition": "...",
      "Criterion": "..."
    }},
    {{
      "target": "Subtask X",
      "Behavior": "...",
      "Condition": "...",
      "Criterion": "..."
    }}
  ]
}}

Output ONLY the JSON object above. Do not include any additional text, explanation, or commentary outside the JSON structure.
"""


# ==============================================================================
# 5단계: Test Item 개발
# ==============================================================================

TEST_ITEM_DEVELOPMENT_PROMPT = """
You are an instructional designer specializing in criterion-referenced assessment development with Anderson & Krathwohl's Taxonomy for Learning.
Your task is to generate essay-type assessment items based strictly on the given Performance Objective.
You will be given the Terminal Goal and a set of Performance Objectives in JSON format.

[Input Data]
Performance Objective: {performance_objective}

[Instructions]
Using the provided data, generate one essay-type assessment item for each Performance Objective.
- Every assessment item must directly measure the Behavior stated in that Performance Objective, incorporating the Condition or Criterion.
- Each item must be unambiguous, aligned to the Performance Objective, and appropriate for evaluating an LLM.
- Using these rules, generate high-quality, reliable, and valid assessment items that accurately measure whether the learner can achieve the Terminal Goal and each Performance Objective.

[Output Format]
The output must be in the following JSON format:

{{
  "assessment_items": [
    {{
      "target": "Terminal Goal or Subskill/Subtask Name",
      "item": "Write the essay-type assessment question here.",
      "problem context": "Write a problem situation that supports the question.",
      "condition": "Write down the elements that must be met or considered in order to address the context of the problem."
    }}
  ]
}}

Output ONLY the JSON object above. Do not include any additional text, explanation, or commentary outside the JSON structure.
"""
