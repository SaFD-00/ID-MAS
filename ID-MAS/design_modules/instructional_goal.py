"""교수설계 Step 0: Instructional Goal 생성 모듈.

이 모듈은 샘플 데이터를 분석하여 Instructional Goal(학습 목표)을
동적으로 생성합니다. Teacher 모델을 사용하여 각 데이터셋에 맞는
학습 목표를 도출합니다.

주요 클래스:
    InstructionalGoalGenerator: 학습 목표 생성 에이전트

Bloom's Taxonomy 인지 수준:
    - Remember: 기억
    - Understand: 이해
    - Apply: 적용
    - Analyze: 분석
    - Evaluate: 평가
    - Create: 창작

사용 예시:
    >>> from design_modules.instructional_goal import InstructionalGoalGenerator
    >>> generator = InstructionalGoalGenerator(teacher_config)
    >>> result = generator.generate(samples, domain="math", dataset="gsm8k")
    >>> print(result["instructional_goal"])
"""
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from models.teacher_wrapper import TeacherModelWrapper
from prompts.instructional_goal_prompts import (
    get_instructional_goal_prompt,
    INSTRUCTIONAL_GOAL_SYSTEM_MESSAGE
)


class InstructionalGoalGenerator:
    """Instructional Goal(학습 목표) 생성 에이전트.

    샘플 데이터를 분석하여 데이터셋에 적합한 학습 목표를 생성합니다.
    Teacher 모델을 사용하여 Bloom's Taxonomy 기반의 인지 수준과
    함께 학습 목표를 도출합니다.

    Attributes:
        llm: Teacher 모델 래퍼
        teacher_config: Teacher 모델 설정

    Example:
        >>> generator = InstructionalGoalGenerator()
        >>> result = generator.generate(
        ...     train_samples=samples,
        ...     domain="math",
        ...     dataset="gsm8k"
        ... )
        >>> print(result["instructional_goal"])
        "The model should be able to generate coherent..."
    """

    def __init__(self, teacher_config: dict = None):
        """InstructionalGoalGenerator를 초기화합니다.

        Args:
            teacher_config: Teacher 모델 설정. None이면 기본 설정 사용.
        """
        self.llm = TeacherModelWrapper(teacher_config)
        self.teacher_config = teacher_config

    def generate(
        self,
        train_samples: List[Dict],
        domain: str,
        dataset: str,
        prompt_template: str = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Instructional Goal을 생성합니다.

        샘플 데이터를 분석하여 데이터셋에 적합한 학습 목표를 생성합니다.
        실패 시 최대 max_retries 횟수만큼 재시도합니다.

        Args:
            train_samples: 분석할 샘플 데이터 리스트 (권장: 10-30개)
            domain: 도메인명 (예: "math", "logical", "commonsense")
            dataset: 데이터셋명 (예: "gsm8k", "math", "reclor")
            prompt_template: 커스텀 프롬프트 템플릿. None이면 기본 템플릿 사용.
            max_retries: 최대 재시도 횟수 (기본: 3)

        Returns:
            학습 목표 결과 딕셔너리:
                - instructional_goal (str): 생성된 학습 목표
                - cognitive_level (str): Bloom's Taxonomy 인지 수준
                - primary_verb (str): 주요 동사
                - rationale (str): 근거 설명
                - raw_output (str): 원본 JSON 출력
                - metadata (dict): 생성 메타데이터
                    - generated_at: 생성 시각
                    - sample_count: 사용된 샘플 수
                    - model: 사용된 모델명
                    - prompt_version: 프롬프트 버전

        Raises:
            RuntimeError: max_retries 횟수 초과 시
        """
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                # 프롬프트 구성
                prompt = get_instructional_goal_prompt(
                    domain=domain,
                    dataset=dataset,
                    samples=train_samples,
                    custom_template=prompt_template
                )

                # LLM으로 Instructional Goal 생성 (JSON 형식)
                try:
                    result_json = self.llm.generate_json(
                        prompt=prompt,
                        system_message=INSTRUCTIONAL_GOAL_SYSTEM_MESSAGE
                    )
                except Exception as e:
                    # JSON 파싱 실패 시 텍스트로 재시도
                    raw_text = self.llm.generate(
                        prompt=prompt,
                        system_message=INSTRUCTIONAL_GOAL_SYSTEM_MESSAGE
                    )
                    result_json = self._parse_fallback(raw_text)

                # 결과 검증
                instructional_goal = result_json.get("instructional_goal", "")
                if not instructional_goal:
                    raise ValueError("Empty instructional_goal in response")

                # 메타데이터 추가
                model_name = self.teacher_config.get("model", "unknown") if self.teacher_config else "default"

                return {
                    "instructional_goal": instructional_goal,
                    "cognitive_level": result_json.get("cognitive_level", "Apply"),
                    "primary_verb": result_json.get("primary_verb", ""),
                    "rationale": result_json.get("rationale", ""),
                    "raw_output": json.dumps(result_json, ensure_ascii=False),
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "sample_count": len(train_samples),
                        "model": model_name,
                        "prompt_version": "v1"
                    }
                }

            except Exception as e:
                last_error = e
                print(f"  [Attempt {attempt}/{max_retries}] Instructional Goal generation failed: {e}")

                if attempt < max_retries:
                    print(f"  Retrying...")
                else:
                    print(f"  [FATAL] All {max_retries} attempts failed.")

        # 모든 재시도 실패 → RuntimeError 발생
        raise RuntimeError(
            f"Instructional Goal generation failed after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _parse_fallback(self, raw_text: str) -> Dict[str, Any]:
        """텍스트 응답에서 Instructional Goal을 추출합니다.

        JSON 파싱 실패 시 텍스트에서 학습 목표를 추출하는 fallback 메서드입니다.
        다양한 패턴을 시도하여 "The model should/will..." 형식의 목표를 찾습니다.

        Args:
            raw_text: LLM의 텍스트 응답

        Returns:
            파싱된 결과 딕셔너리:
                - instructional_goal: 추출된 학습 목표
                - cognitive_level: 인지 수준 (기본: "Apply")
                - primary_verb: 주요 동사
                - rationale: 근거 설명
        """
        result = {
            "instructional_goal": "",
            "cognitive_level": "Apply",
            "primary_verb": "",
            "rationale": ""
        }

        lines = raw_text.strip().split('\n')

        for line in lines:
            line_lower = line.lower().strip()

            # Instructional Goal 패턴 찾기
            if "instructional_goal" in line_lower or "the model should" in line_lower or "the model will" in line_lower:
                # "instructional_goal": "..." 또는 Instructional Goal: ... 패턴
                if ":" in line:
                    value = line.split(":", 1)[1].strip().strip('"').strip(',')
                    if value:
                        result["instructional_goal"] = value

            # Cognitive Level 패턴
            elif "cognitive_level" in line_lower:
                if ":" in line:
                    value = line.split(":", 1)[1].strip().strip('"').strip(',')
                    if value:
                        result["cognitive_level"] = value

        # Instructional Goal이 여전히 없으면 전체 텍스트에서 "The model will/should..." 패턴 찾기
        if not result["instructional_goal"]:
            import re
            pattern = r'["\']?(The model (?:should be able to|will)[^"\']+)["\']?'
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                result["instructional_goal"] = match.group(1).strip()

        return result


if __name__ == "__main__":
    # 테스트
    import sys
    sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

    from pathlib import Path
    from config import create_teacher_config

    # 샘플 데이터 로드 테스트
    project_root = Path(__file__).parent.parent
    samples_path = project_root / "data" / "math" / "train" / "data" / "gsm8k_samples.json"

    if samples_path.exists():
        with open(samples_path, 'r') as f:
            samples = json.load(f)

        print("=" * 60)
        print("Instructional Goal Generator Test")
        print("=" * 60)
        print(f"Loaded {len(samples)} samples")

        # Generator 초기화 (기본 설정 사용)
        generator = InstructionalGoalGenerator()

        result = generator.generate(
            train_samples=samples,
            domain="math",
            dataset="gsm8k"
        )

        print("\n=== Generated Instructional Goal ===")
        print(f"Goal: {result['instructional_goal']}")
        print(f"Cognitive Level: {result['cognitive_level']}")
        print(f"Primary Verb: {result['primary_verb']}")
        print(f"Rationale: {result['rationale']}")

    else:
        print(f"Samples file not found: {samples_path}")
        print("Run 'python -m utils.sample_extractor' first to generate samples.")
