"""
Terminal Goal Generation Prompts

Terminal Goal 자동 생성을 위한 프롬프트 템플릿.
Design Phase Step 0에서 사용.
"""

TERMINAL_GOAL_PROMPT = """You are an expert instructional designer and AI learning researcher.

## Context
- Domain: {domain}
- Dataset: {dataset}
- Sample Size: {sample_count} items

## Task
Analyze the provided test items and derive a single Terminal Goal that captures the core competency required to solve these problems.

## Guidelines
1. Focus on the highest cognitive level demonstrated in the samples
2. Use a single, observable, measurable action verb (e.g., generate, solve, analyze, apply)
3. Start with: "The model should be able to..."
4. Do NOT mention learning theories by name (e.g., avoid "Bloom's Taxonomy")
5. Make it actionable and domain-specific
6. The goal should be achievable through the demonstrated problem types

## Reference Examples
- Math (GSM8K): "The model should be able to generate coherent, step-by-step mathematical reasoning in natural language that leads to a correct numerical answer for grade-school level math problems."
- Math (MATH): "The model should be able to solve advanced mathematical problems by selecting appropriate mathematical concepts and constructing logically valid, multi-step reasoning that leads to a correct solution."
- Logical (ReClor): "The model should be able to analyze logical reasoning problems by comprehending complex passages, identifying logical relationships, and selecting the most appropriate conclusion based on formal reasoning principles."
- Commonsense (ARC-C): "The model should be able to apply commonsense scientific knowledge to solve elementary science problems by understanding fundamental concepts and selecting the correct answer from multiple choices."

## Input Data
Below are {sample_count} representative samples from the {dataset} dataset:

{train_data}

## Output (JSON)
Provide your response in the following JSON format:
{{
  "terminal_goal": "The model should be able to ...",
  "cognitive_level": "Remember|Understand|Apply|Analyze|Evaluate|Create",
  "primary_verb": "the main action verb used",
  "rationale": "Brief explanation of why this goal was chosen based on the sample analysis"
}}
"""


TERMINAL_GOAL_SYSTEM_MESSAGE = """You are an expert in instructional design and educational assessment.
Your task is to analyze learning materials and derive clear, measurable learning objectives.

Key principles:
- Learning objectives should be SMART: Specific, Measurable, Achievable, Relevant, Time-bound
- Use action verbs from Bloom's Taxonomy (but don't name the taxonomy)
- Focus on observable behaviors that can be assessed
- Consider the cognitive complexity required by the tasks

Output must be valid JSON only, with no additional text."""


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
