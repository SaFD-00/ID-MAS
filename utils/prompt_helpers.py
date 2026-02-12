"""프롬프트 구성 헬퍼 함수 모듈.

프롬프트 템플릿에 데이터를 삽입하는 유틸리티 함수를 제공합니다.

함수:
    format_samples_for_prompt: 샘플 데이터를 프롬프트용 문자열로 변환
    get_instructional_goal_prompt: 완성된 학습목표 생성 프롬프트 구성
"""
from prompts.design_prompts import INSTRUCTIONAL_GOAL_USER_PROMPT


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
    template = custom_template or INSTRUCTIONAL_GOAL_USER_PROMPT

    train_data = format_samples_for_prompt(samples)

    return template.format(
        domain=domain,
        dataset=dataset,
        sample_count=len(samples),
        train_data=train_data
    )
