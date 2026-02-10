"""교수설계 단계별 프롬프트 템플릿 모듈.

이 모듈은 Dick & Carey 교수설계 모델의 각 단계에서 사용하는 프롬프트 템플릿을 정의합니다.
LLM을 통해 교수설계 산출물(학습목표, 교수분석, 수행목표)을 자동 생성하는 데 활용됩니다.

프롬프트 상수:
    INSTRUCTIONAL_GOAL_SYSTEM_MESSAGE: LLM 시스템 메시지 (Step 0)
    INSTRUCTIONAL_GOAL_PROMPT: 학습목표 생성 프롬프트 (Step 0)
    INSTRUCTIONAL_ANALYSIS_PROMPT: 교수분석 생성 프롬프트 (2단계)
    PERFORMANCE_OBJECTIVES_PROMPT: 수행목표 진술 생성 프롬프트 (4단계)

함수:
    format_samples_for_prompt: 샘플 데이터를 프롬프트용 문자열로 변환
    get_instructional_goal_prompt: 완성된 학습목표 생성 프롬프트 구성

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


INSTRUCTIONAL_GOAL_PROMPT = """You are given a sample of items representing a specific task domain. These items are used to evaluate the student you are teaching. Your mission is to analyze the entire test set and determine a core instructional requirement that defines the instructional goal.

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


def format_samples_for_prompt(samples: list, max_samples: int = 20) -> str:
    """샘플 데이터를 프롬프트용 문자열로 변환합니다.

    각 샘플의 instruction과 input 필드를 추출하여 번호가 매겨진
    형식의 문자열로 변환합니다. output 필드는 학습목표 도출에
    편향을 줄 수 있으므로 의도적으로 제외합니다.

    Args:
        samples: 샘플 데이터 리스트. 각 샘플은 instruction, input 키를 가진 딕셔너리
        max_samples: 프롬프트에 포함할 최대 샘플 수. 기본값: 20

    Returns:
        "### Sample N\\n{instruction}\\n{input}" 형식으로 구성된 문자열.
        instruction은 최대 200자, input은 최대 500자로 절단됩니다.
    """
    formatted_samples = []

    for i, sample in enumerate(samples[:max_samples]):
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")

        # instruction과 input만 포함 (output 제외)
        instruction_truncated = instruction[:200] if instruction else "N/A"
        input_truncated = input_text[:500] if input_text else "N/A"

        sample_text = f"""### Sample {i + 1}
{instruction_truncated}
{input_truncated}
"""
        formatted_samples.append(sample_text)

    return "\n".join(formatted_samples)


def get_instructional_goal_prompt(
    domain: str,
    dataset: str,
    samples: list,
    custom_template: str = None
) -> str:
    """학습목표 생성용 완성된 프롬프트를 구성합니다.

    샘플 데이터를 포맷하고 템플릿에 삽입하여 LLM에 전달할
    최종 프롬프트 문자열을 생성합니다.

    Args:
        domain: 도메인 이름 (math, logical, commonsense 등)
        dataset: 데이터셋 이름 (gsm8k, reclor, arc_c 등)
        samples: 학습목표 도출에 사용할 샘플 데이터 리스트
        custom_template: 커스텀 프롬프트 템플릿. None이면 기본 템플릿 사용

    Returns:
        {sample_count}, {train_data} 등이 채워진 완성된 프롬프트 문자열
    """
    template = custom_template or INSTRUCTIONAL_GOAL_PROMPT

    train_data = format_samples_for_prompt(samples)

    return template.format(
        domain=domain,
        dataset=dataset,
        sample_count=len(samples),
        train_data=train_data
    )


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
