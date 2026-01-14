"""
Design Phase Step 0: Terminal Goal Generation

샘플 데이터를 분석하여 Terminal Goal을 동적으로 생성.
Teacher Model을 사용하여 각 데이터셋에 맞는 학습 목표 도출.
"""
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from models.teacher_wrapper import TeacherModelWrapper
from prompts.terminal_goal_prompts import (
    get_terminal_goal_prompt,
    TERMINAL_GOAL_SYSTEM_MESSAGE
)


class TerminalGoalGenerator:
    """
    Design Phase Step 0: Terminal Goal 생성 에이전트

    샘플 데이터를 분석하여 데이터셋에 적합한 Terminal Goal을 생성.
    Teacher Model (teacher_config로 설정된 모델)을 사용.
    """

    def __init__(self, teacher_config: dict = None):
        """
        Args:
            teacher_config: Teacher model 설정 (None이면 기본 설정 사용)
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
        """
        Terminal Goal 생성 (최대 3번 재시도)

        Args:
            train_samples: 샘플 데이터 (10-30개)
            domain: 도메인 이름 (e.g., "math", "logical")
            dataset: 데이터셋 이름 (e.g., "gsm8k", "math")
            prompt_template: 커스텀 프롬프트 (None이면 기본 사용)
            max_retries: 최대 재시도 횟수 (기본 3)

        Returns:
            {
                "terminal_goal": "The model should be able to...",
                "cognitive_level": "Apply",
                "primary_verb": "generate",
                "rationale": "...",
                "raw_output": "...",
                "metadata": {
                    "generated_at": "2026-01-11T...",
                    "sample_count": 25,
                    "model": "Qwen/Qwen2.5-72B-Instruct",
                    "prompt_version": "v1"
                }
            }

        Raises:
            RuntimeError: max_retries 초과 시 프로그램 종료 필요
        """
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                # 프롬프트 구성
                prompt = get_terminal_goal_prompt(
                    domain=domain,
                    dataset=dataset,
                    samples=train_samples,
                    custom_template=prompt_template
                )

                # LLM으로 Terminal Goal 생성 (JSON 형식)
                try:
                    result_json = self.llm.generate_json(
                        prompt=prompt,
                        system_message=TERMINAL_GOAL_SYSTEM_MESSAGE
                    )
                except Exception as e:
                    # JSON 파싱 실패 시 텍스트로 재시도
                    raw_text = self.llm.generate(
                        prompt=prompt,
                        system_message=TERMINAL_GOAL_SYSTEM_MESSAGE
                    )
                    result_json = self._parse_fallback(raw_text)

                # 결과 검증
                terminal_goal = result_json.get("terminal_goal", "")
                if not terminal_goal:
                    raise ValueError("Empty terminal_goal in response")

                # 메타데이터 추가
                model_name = self.teacher_config.get("model", "unknown") if self.teacher_config else "default"

                return {
                    "terminal_goal": terminal_goal,
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
                print(f"  [Attempt {attempt}/{max_retries}] Terminal Goal generation failed: {e}")

                if attempt < max_retries:
                    print(f"  Retrying...")
                else:
                    print(f"  [FATAL] All {max_retries} attempts failed.")

        # 모든 재시도 실패 → RuntimeError 발생
        raise RuntimeError(
            f"Terminal Goal generation failed after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _parse_fallback(self, raw_text: str) -> Dict[str, Any]:
        """
        텍스트 응답에서 Terminal Goal 추출 (fallback)

        Args:
            raw_text: LLM의 텍스트 응답

        Returns:
            파싱된 결과 딕셔너리
        """
        result = {
            "terminal_goal": "",
            "cognitive_level": "Apply",
            "primary_verb": "",
            "rationale": ""
        }

        lines = raw_text.strip().split('\n')

        for line in lines:
            line_lower = line.lower().strip()

            # Terminal Goal 패턴 찾기
            if "terminal_goal" in line_lower or "the model should" in line_lower:
                # "terminal_goal": "..." 또는 Terminal Goal: ... 패턴
                if ":" in line:
                    value = line.split(":", 1)[1].strip().strip('"').strip(',')
                    if value:
                        result["terminal_goal"] = value

            # Cognitive Level 패턴
            elif "cognitive_level" in line_lower:
                if ":" in line:
                    value = line.split(":", 1)[1].strip().strip('"').strip(',')
                    if value:
                        result["cognitive_level"] = value

        # Terminal Goal이 여전히 없으면 전체 텍스트에서 "The model should..." 패턴 찾기
        if not result["terminal_goal"]:
            import re
            pattern = r'["\']?(The model should be able to[^"\']+)["\']?'
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                result["terminal_goal"] = match.group(1).strip()

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
        print("Terminal Goal Generator Test")
        print("=" * 60)
        print(f"Loaded {len(samples)} samples")

        # Generator 초기화 (기본 설정 사용)
        generator = TerminalGoalGenerator()

        result = generator.generate(
            train_samples=samples,
            domain="math",
            dataset="gsm8k"
        )

        print("\n=== Generated Terminal Goal ===")
        print(f"Goal: {result['terminal_goal']}")
        print(f"Cognitive Level: {result['cognitive_level']}")
        print(f"Primary Verb: {result['primary_verb']}")
        print(f"Rationale: {result['rationale']}")

    else:
        print(f"Samples file not found: {samples_path}")
        print("Run 'python -m utils.sample_extractor' first to generate samples.")
