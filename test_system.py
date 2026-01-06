"""
ID-MAS 시스템 테스트 스크립트
각 모듈의 기본 동작을 검증합니다.
"""
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

import json


def test_mmlu_loader():
    """MMLU 데이터 로더 테스트"""
    print("\n" + "=" * 60)
    print("TEST 1: MMLU Loader")
    print("=" * 60)

    try:
        from utils.mmlu_loader import MMLULoader

        loader = MMLULoader()

        # Subject 목록 확인
        subjects = loader.get_available_subjects()
        print(f"✓ Available subjects: {len(subjects)}")
        print(f"  First 5: {subjects[:5]}")

        # 샘플 데이터 로드
        questions = loader.load_subject("abstract_algebra", split="validation")
        print(f"✓ Loaded {len(questions)} questions")

        # 프롬프트 생성
        prompt = loader.format_question_as_prompt(questions[0])
        print(f"✓ Generated prompt:\n{prompt[:100]}...")

        return True

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def test_gpt_wrapper():
    """GPT 래퍼 테스트 (API 키 필요)"""
    print("\n" + "=" * 60)
    print("TEST 2: GPT Wrapper")
    print("=" * 60)

    try:
        from models.gpt_wrapper import GPTWrapper
        from config.config import OPENAI_API_KEY

        if not OPENAI_API_KEY:
            print("⚠ Skipped: OPENAI_API_KEY not set in .env")
            return None

        gpt = GPTWrapper()

        # 간단한 생성 테스트
        response = gpt.generate("Say 'Hello, ID-MAS!'")
        print(f"✓ GPT response: {response}")

        # JSON 생성 테스트
        json_response = gpt.generate_json(
            "Generate a JSON object with a 'test' field containing 'success'.",
            system_message="You are a JSON generator."
        )
        print(f"✓ GPT JSON response: {json.dumps(json_response, indent=2)}")

        return True

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def test_design_modules():
    """교수 설계 모듈 테스트"""
    print("\n" + "=" * 60)
    print("TEST 3: Design Modules")
    print("=" * 60)

    try:
        from design_modules.step2_analysis import InstructionalAnalysis

        analyzer = InstructionalAnalysis()

        # 간단한 목표로 테스트 (실제 GPT 호출 없이)
        print("✓ InstructionalAnalysis module loaded")

        from design_modules.step4_objectives import PerformanceObjectives
        print("✓ PerformanceObjectives module loaded")

        from design_modules.step5_test import TestItemDevelopment
        print("✓ TestItemDevelopment module loaded")

        from design_modules.step5_rubric import RubricDevelopment
        print("✓ RubricDevelopment module loaded")

        return True

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_learning_modules():
    """학습 루프 모듈 테스트"""
    print("\n" + "=" * 60)
    print("TEST 4: Learning Loop Modules")
    print("=" * 60)

    try:
        from learning_loop.student_model import StudentModel
        print("✓ StudentModel module loaded")

        from learning_loop.teacher_model import TeacherModel
        print("✓ TeacherModel module loaded")

        from learning_loop.reflection import ReflectionModule
        print("✓ ReflectionModule module loaded")

        from learning_loop.loop_controller import LearningLoopController
        print("✓ LearningLoopController module loaded")

        return True

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """설정 파일 테스트"""
    print("\n" + "=" * 60)
    print("TEST 5: Configuration")
    print("=" * 60)

    try:
        from config.config import (
            OPENAI_API_KEY,
            DESIGN_MODEL_CONFIG,
            STUDENT_MODEL_CONFIG,
            LEARNING_LOOP_CONFIG,
            DATA_DIR,
            get_domain_data_dirs
        )

        print(f"✓ API Key set: {'Yes' if OPENAI_API_KEY else 'No'}")
        print(f"✓ Design model: {DESIGN_MODEL_CONFIG['model']}")
        print(f"✓ Student model: {STUDENT_MODEL_CONFIG['model_name']}")
        print(f"✓ Max iterations: {LEARNING_LOOP_CONFIG['max_iterations']}")
        print(f"✓ Data directory: {DATA_DIR}")

        # 디렉토리 존재 확인
        assert DATA_DIR.exists(), "DATA_DIR does not exist"

        # Test domain-based directory structure
        for domain in ["math", "knowledge"]:
            dirs = get_domain_data_dirs(domain, train_dataset="gsm8k")
            assert dirs["learning_logs_dir"].exists(), f"{domain} learning logs directory does not exist"
            print(f"✓ {domain.capitalize()} domain directories: {dirs['learning_logs_dir']}")

        return True

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """전체 테스트 실행"""
    print("\n" + "=" * 60)
    print("ID-MAS SYSTEM TEST")
    print("=" * 60)

    results = {
        "MMLU Loader": test_mmlu_loader(),
        "GPT Wrapper": test_gpt_wrapper(),
        "Design Modules": test_design_modules(),
        "Learning Modules": test_learning_modules(),
        "Configuration": test_config()
    }

    # 결과 요약
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⚠ SKIP"

        print(f"{test_name:.<40} {status}")

    passed = sum(1 for r in results.values() if r is True)
    total = len([r for r in results.values() if r is not None])

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠ Some tests failed or were skipped.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
