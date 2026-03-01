"""SFT 데이터 생성 파이프라인 테스트.

Normal 실행 경로에서 save_results → generate_sft_data → SFT 재저장
순서가 올바르게 동작하는지 검증합니다.

의존성: conftest.py에서 GPU/vllm mock 처리
"""
import json
from pathlib import Path

import pytest

from learning_loop.graph.nodes import generate_sft_data, save_results
from learning_loop.graph.state import SFTCase


# ==================== Fixtures ====================

def _make_result(case: str, question_id: str = "q1") -> dict:
    """테스트용 QuestionResult를 생성합니다."""
    return {
        "id": question_id,
        "question_id": question_id,
        "sft_case": case,
        "sft_response": f"SFT response for {question_id}",
        "initial_response": f"Initial response for {question_id}",
        "instruction": f"Solve {question_id}",
        "input": f"Question text for {question_id}",
        "output": f"Answer for {question_id}",
        "is_correct": case != SFTCase.TEACHER_MODELING_DISTILLATION.value,
    }


def _make_logs_json(results: list, path: Path):
    """테스트용 logs JSON 파일을 생성합니다."""
    data = {
        "scaffolding_results": results,
        "statistics": {},
        "is_complete": True,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _make_logs_jsonl(results: list, path: Path):
    """테스트용 logs JSONL 파일을 생성합니다."""
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ==================== Unit Tests: generate_sft_data ====================

class TestGenerateSftData:
    """generate_sft_data() 단위 테스트."""

    def test_from_json_file(self, tmp_path):
        """JSON 파일에서 SFT 데이터를 정상 생성한다."""
        results = [
            _make_result(SFTCase.INDEPENDENT_PERFORMANCE_MASTERY.value, "q1"),
            _make_result(SFTCase.SCAFFOLDED_COACHED_MASTERY.value, "q2"),
            _make_result(SFTCase.TEACHER_MODELING_DISTILLATION.value, "q3"),
        ]
        logs_path = tmp_path / "test_logs.json"
        _make_logs_json(results, logs_path)

        state = {"logs_file_path": str(logs_path)}
        result = generate_sft_data(state)

        assert len(result["sft_data"]) == 3
        assert result["is_complete"] is True
        assert result["current_phase"] == "complete"

    def test_fallback_to_state(self):
        """파일이 없으면 state의 scaffolding_results를 사용한다."""
        results = [
            _make_result(SFTCase.INDEPENDENT_PERFORMANCE_MASTERY.value, "q1"),
        ]
        state = {
            "logs_file_path": None,
            "scaffolding_results": results,
        }
        result = generate_sft_data(state)

        assert len(result["sft_data"]) == 1

    def test_empty_when_no_source(self):
        """파일도 없고 state에도 결과가 없으면 빈 리스트를 반환한다."""
        state = {
            "logs_file_path": "/nonexistent/path.json",
            "scaffolding_results": [],
        }
        result = generate_sft_data(state)

        assert result["sft_data"] == []
        assert result["is_complete"] is True

    def test_all_cases_included(self, tmp_path):
        """Case A, B, C 모두 SFT 데이터에 포함된다."""
        results = [
            _make_result(SFTCase.INDEPENDENT_PERFORMANCE_MASTERY.value, "q1"),
            _make_result(SFTCase.SCAFFOLDED_COACHED_MASTERY.value, "q2"),
            _make_result(SFTCase.TEACHER_MODELING_DISTILLATION.value, "q3"),
            _make_result(SFTCase.SKIPPED.value, "q4"),  # 제외되어야 함
        ]
        logs_path = tmp_path / "test_logs.json"
        _make_logs_json(results, logs_path)

        state = {"logs_file_path": str(logs_path)}
        result = generate_sft_data(state)

        assert len(result["sft_data"]) == 3  # SKIPPED 제외

    def test_nonexistent_file_falls_back(self):
        """존재하지 않는 파일 경로이면 state fallback을 사용한다."""
        results = [
            _make_result(SFTCase.INDEPENDENT_PERFORMANCE_MASTERY.value, "q1"),
            _make_result(SFTCase.SCAFFOLDED_COACHED_MASTERY.value, "q2"),
        ]
        state = {
            "logs_file_path": "/tmp/nonexistent_file_12345.json",
            "scaffolding_results": results,
        }
        result = generate_sft_data(state)

        assert len(result["sft_data"]) == 2


# ==================== Unit Tests: save_results ====================

class TestSaveResults:
    """save_results() 단위 테스트."""

    def test_jsonl_to_json_conversion(self, tmp_path):
        """JSONL 파일이 JSON으로 변환된다."""
        results = [
            _make_result(SFTCase.INDEPENDENT_PERFORMANCE_MASTERY.value, "q1"),
            _make_result(SFTCase.SCAFFOLDED_COACHED_MASTERY.value, "q2"),
        ]
        jsonl_path = tmp_path / "test_logs.jsonl"
        _make_logs_jsonl(results, jsonl_path)

        state = {
            "scaffolding_results": [],
            "sft_data": [],
            "scaffolding_processed": 2,
            "total_questions": 2,
            "case_a_independent_performance_mastery_count": 1,
            "case_b_scaffolded_coached_mastery_count": 1,
            "case_c_teacher_modeling_distillation_count": 0,
            "skipped_count": 0,
            "hot_scaffolding_count": 0,
            "lot_scaffolding_count": 0,
        }

        results_path, sft_path = save_results(
            state=state,
            output_dir=tmp_path,
            sft_filename="test_sft.json",
            logs_filename="test_logs.json",
        )

        # JSON 파일이 생성되어야 함
        assert results_path.exists()
        with open(results_path, "r") as f:
            data = json.load(f)
        assert len(data["scaffolding_results"]) == 2

        # JSONL은 삭제되어야 함
        assert not jsonl_path.exists()


# ==================== Integration Tests ====================

class TestSftPipelineIntegration:
    """SFT 파이프라인 통합 테스트.

    Normal 경로: save_results → generate_sft_data → SFT 재저장
    """

    def test_normal_path_sft_generation_order(self, tmp_path):
        """Normal 경로에서 올바른 순서로 SFT가 생성된다.

        이 테스트는 버그 수정의 핵심을 검증합니다:
        1. JSONL 파일에서 save_results로 JSON 변환
        2. generate_sft_data로 JSON에서 SFT 생성
        3. SFT 파일 재저장
        """
        # Setup: JSONL에 결과 저장 (실제 파이프라인 실행 시뮬레이션)
        results = [
            _make_result(SFTCase.INDEPENDENT_PERFORMANCE_MASTERY.value, f"q{i}")
            for i in range(5)
        ] + [
            _make_result(SFTCase.SCAFFOLDED_COACHED_MASTERY.value, f"q{i}")
            for i in range(5, 8)
        ] + [
            _make_result(SFTCase.TEACHER_MODELING_DISTILLATION.value, f"q{i}")
            for i in range(8, 10)
        ]

        jsonl_path = tmp_path / "test_logs.jsonl"
        _make_logs_jsonl(results, jsonl_path)

        state = {
            "scaffolding_results": [],  # 의도적으로 비어있음 (메모리 최적화)
            "sft_data": [],
            "logs_file_path": str(tmp_path / "test_logs.json"),
            "scaffolding_processed": 10,
            "total_questions": 10,
            "case_a_independent_performance_mastery_count": 5,
            "case_b_scaffolded_coached_mastery_count": 3,
            "case_c_teacher_modeling_distillation_count": 2,
            "skipped_count": 0,
            "hot_scaffolding_count": 0,
            "lot_scaffolding_count": 0,
        }

        # Step 1: save_results — JSONL → JSON 변환
        results_path, sft_path = save_results(
            state=state,
            output_dir=tmp_path,
            sft_filename="test_sft.json",
            logs_filename="test_logs.json",
        )

        # 이 시점에서 JSON 파일이 존재해야 함
        assert results_path.exists()

        # Step 2: generate_sft_data — JSON에서 SFT 생성
        sft_update = generate_sft_data(state)
        state.update(sft_update)

        # Step 3: SFT 파일 재저장
        with open(sft_path, "w", encoding="utf-8") as f:
            json.dump(state.get("sft_data", []), f, ensure_ascii=False, indent=2)

        # 검증: SFT 데이터가 10건 생성되어야 함
        assert len(state["sft_data"]) == 10

        # 검증: SFT 파일에 10건 저장되어야 함
        with open(sft_path, "r") as f:
            saved_sft = json.load(f)
        assert len(saved_sft) == 10

    def test_bug_reproduction_without_fix(self, tmp_path):
        """수정 전 버그를 재현: JSON 파일 없이 generate_sft_data를 호출하면 0건.

        이 테스트는 버그의 동작을 문서화합니다.
        """
        # JSON 파일이 없는 상태 (JSONL만 존재)
        results = [
            _make_result(SFTCase.INDEPENDENT_PERFORMANCE_MASTERY.value, "q1"),
        ]
        jsonl_path = tmp_path / "test_logs.jsonl"
        _make_logs_jsonl(results, jsonl_path)

        state = {
            "scaffolding_results": [],  # state에 결과 없음 (graph.py:354-355)
            "logs_file_path": str(tmp_path / "test_logs.json"),  # JSON 미존재
        }

        # JSON 파일이 없으므로 fallback → 빈 리스트
        result = generate_sft_data(state)
        assert result["sft_data"] == []  # 버그 동작: 0건
