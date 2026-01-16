"""
Instructional Goal Generation Prompts

Instructional Goal 자동 생성을 위한 프롬프트 템플릿.
Design Phase Step 0에서 사용.
"""


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
    """
    샘플 데이터를 프롬프트용 문자열로 포맷

    Args:
        samples: 샘플 데이터 리스트
        max_samples: 프롬프트에 포함할 최대 샘플 수

    Returns:
        포맷된 문자열
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
    """
    Instructional Goal 생성 프롬프트 구성

    Args:
        domain: 도메인 이름
        dataset: 데이터셋 이름
        samples: 샘플 데이터 리스트
        custom_template: 커스텀 프롬프트 템플릿 (None이면 기본 사용)

    Returns:
        완성된 프롬프트 문자열
    """
    template = custom_template or INSTRUCTIONAL_GOAL_PROMPT

    train_data = format_samples_for_prompt(samples)

    return template.format(
        domain=domain,
        dataset=dataset,
        sample_count=len(samples),
        train_data=train_data
    )
