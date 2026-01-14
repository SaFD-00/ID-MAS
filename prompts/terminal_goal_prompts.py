"""
Terminal Goal Generation Prompts

Terminal Goal 자동 생성을 위한 프롬프트 템플릿.
Design Phase Step 0에서 사용.
"""

TERMINAL_GOAL_PROMPT = """## Context
- Domain: {domain}
- Dataset: {dataset}
- Sample Size: {sample_count} items

## Task
Analyze the provided samples and derive a single Terminal Goal that captures the core competency required to solve these problems.

## Input Data
Below are {sample_count} representative samples from the {dataset} dataset:

{train_data}

## Analysis Steps (Think step by step)
1. **Pattern Recognition**: Identify common problem types and structures across samples
2. **Cognitive Demand Analysis**: Determine what mental processes are required (recall, comprehension, application, analysis, evaluation, creation)
3. **Action Verb Selection**: Choose a single, observable, measurable verb that best represents the required competency
4. **Goal Formulation**: Synthesize findings into a clear Terminal Goal

## Guidelines
- Start with: "The model should be able to..."
- Use action verbs like: generate, solve, analyze, apply, evaluate, construct
- Make it domain-specific and achievable through the demonstrated problem types
- Do NOT mention learning theories by name

## Reference Examples
- Math (GSM8K): "The model should be able to generate coherent, step-by-step mathematical reasoning in natural language that leads to a correct numerical answer for grade-school level math problems."
- Logical (ReClor): "The model should be able to analyze logical reasoning problems by comprehending complex passages, identifying logical relationships, and selecting the most appropriate conclusion based on formal reasoning principles."

## Output (JSON)
{{
  "pattern_analysis": "Brief summary of common patterns found in samples",
  "cognitive_demands": ["list", "of", "required", "cognitive", "processes"],
  "terminal_goal": "The model should be able to ...",
  "cognitive_level": "Remember|Understand|Apply|Analyze|Evaluate|Create",
  "primary_verb": "the main action verb used",
  "rationale": "Why this goal was chosen based on the analysis"
}}
"""


TERMINAL_GOAL_SYSTEM_MESSAGE = """You are an expert in instructional design and educational assessment.
Your role is to analyze learning materials and derive clear, measurable learning objectives.

Principles:
- Objectives must be Specific, Measurable, Achievable, and Relevant
- Focus on observable behaviors that can be assessed
- Consider the cognitive complexity required by the tasks

Respond with valid JSON only."""


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


def get_terminal_goal_prompt(
    domain: str,
    dataset: str,
    samples: list,
    custom_template: str = None
) -> str:
    """
    Terminal Goal 생성 프롬프트 구성

    Args:
        domain: 도메인 이름
        dataset: 데이터셋 이름
        samples: 샘플 데이터 리스트
        custom_template: 커스텀 프롬프트 템플릿 (None이면 기본 사용)

    Returns:
        완성된 프롬프트 문자열
    """
    template = custom_template or TERMINAL_GOAL_PROMPT

    train_data = format_samples_for_prompt(samples)

    return template.format(
        domain=domain,
        dataset=dataset,
        sample_count=len(samples),
        train_data=train_data
    )
