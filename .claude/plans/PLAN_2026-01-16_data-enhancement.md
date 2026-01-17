# ID-MAS 데이터 Enhancement 계획

## 1. 문제 분석

### 현재 상황
- Student Model의 `generate_initial_response_with_scaffolding()` 메서드는 학습목표(instructional_goal)와 과제분석(task_analysis)을 **system prompt**로 전달
- 데이터 파일(예: `gsm8k_train.json`)의 instruction 필드는 기본 지시만 포함:
  ```
  "You are a helpful math assistant.\nSolve this mathematical problem step by step..."
  ```

### 요구사항
- 데이터의 **instruction 필드 자체**에 학습목표와 과제분석을 포함시켜 SFT 학습에 직접 사용할 수 있도록 함
- 새로운 instruction 형식:
  ```
  {기존 instruction}

  당신이 질문에 대한 답변을 하는 것은 {학습목표}라는 목표를 성취하는 것을 확인하기 위함입니다.
  제시된 수행의 단계와 필요한 정보자원에 대한 정보를 전부 사용하여 풀이과정을 서술하세요.

  {과제분석}
  ```

### Root Cause
- 현재 시스템은 런타임에 학습목표/과제분석을 생성하여 system prompt로 전달
- SFT 학습 시에는 데이터 자체에 이 정보가 포함되어야 instruction tuning이 효과적

---

## 2. 요구사항 명세

### Confirmed Decisions (사용자 확인)
| 항목 | 결정 |
|------|------|
| 데이터 출처 | 기존 시스템 (InstructionalGoalGenerator, InstructionalAnalysis) 사용 |
| 처리 방식 | 새 데이터 파일 생성 (원본 보존) |
| 파일명 형식 | `{dataset}_train_ID-MAS_{teacher-model}.json` |
| 적용 범위 | 전체 도메인 (math, logical, commonsense) |
| 적용 방식 | 데이터셋별 동일 학습목표/과제분석 적용 |

### 대상 데이터셋
| Domain | Dataset | Source File |
|--------|---------|-------------|
| math | gsm8k | `data/math/train/data/gsm8k_train.json` |
| math | math | `data/math/train/data/math_train.json` |
| logical | reclor | `data/logical/train/data/reclor_train.json` |
| commonsense | arc_c | `data/commonsense/train/data/arc_c_train.json` |

### 출력 파일 예시
Teacher model이 `Qwen/Qwen2.5-72B-Instruct`인 경우:
- `data/math/train/data/gsm8k_train_ID-MAS_Qwen2.5-72B-Instruct.json`
- `data/math/train/data/math_train_ID-MAS_Qwen2.5-72B-Instruct.json`
- `data/logical/train/data/reclor_train_ID-MAS_Qwen2.5-72B-Instruct.json`
- `data/commonsense/train/data/arc_c_train_ID-MAS_Qwen2.5-72B-Instruct.json`

### 추가 요구사항: 파일명 규칙 통일

**현재 문제**: 기존 reasoning 파일들의 파일명 규칙이 일관성 없음
- 현재: `{dataset}_reasoning_train.json` (예: `gsm8k_reasoning_train.json`)
- 변경: `{dataset}_train_reasoning.json` (예: `gsm8k_train_reasoning.json`)

**대상 파일**:
| 현재 파일명 | 변경 후 파일명 |
|------------|---------------|
| `gsm8k_reasoning_train.json` | `gsm8k_train_reasoning.json` |
| `math_reasoning_train.json` | `math_train_reasoning.json` |

**코드 수정 대상**:
- [utils/dataset_preparer.py:269](utils/dataset_preparer.py#L269): `gsm8k_reasoning_{split}.json` → `gsm8k_{split}_reasoning.json`
- [utils/dataset_preparer.py:341](utils/dataset_preparer.py#L341): `math_reasoning_{split}.json` → `math_{split}_reasoning.json`

### Acceptance Criteria
- [ ] AC1: 각 데이터셋에 대해 학습목표(instructional_goal)가 생성됨
- [ ] AC2: 각 데이터셋에 대해 과제분석(task_analysis)이 생성됨
- [ ] AC3: 새 데이터 파일의 instruction 필드에 학습목표와 과제분석이 포함됨
- [ ] AC4: 원본 데이터 파일은 변경되지 않음
- [ ] AC5: 새 데이터를 사용하여 StudentModel이 initial response를 생성할 수 있음
- [ ] AC6: CLI에서 `--enhanced-data` 또는 유사한 플래그로 enhanced 데이터 사용 가능
- [ ] AC7: 기존 `{dataset}_reasoning_train.json` 파일이 `{dataset}_train_reasoning.json`으로 이름 변경됨
- [ ] AC8: `dataset_preparer.py`가 새 파일명 규칙으로 파일을 생성함

---

## 3. 아키텍처 설계

### Approach 1: 데이터 전처리 스크립트 (선택)
- **장점**: 일회성 실행, 데이터 재사용 용이, SFT 학습에 직접 사용 가능
- **단점**: 학습목표/과제분석 변경 시 재생성 필요

### Approach 2: 런타임 동적 처리
- **장점**: 항상 최신 학습목표/과제분석 사용
- **단점**: 매번 API 호출 필요, 비용/시간 증가

### Approach 3: 캐싱 기반 하이브리드
- **장점**: 초기 생성 후 캐시 재사용
- **단점**: 구현 복잡성 증가

**선택**: Approach 1 (데이터 전처리 스크립트)
- SFT 학습 목적에 가장 적합
- 일관된 학습목표/과제분석으로 데이터 품질 보장
- 사용자 요청과 일치

### 시스템 구조

```
scripts/enhance_data.py (NEW)
├── InstructionalGoalGenerator (기존)
├── InstructionalAnalysis (기존)
└── DataEnhancer (NEW)
    ├── load_source_data()
    ├── generate_instructional_goal()
    ├── generate_task_analysis()
    ├── enhance_instruction()
    └── save_enhanced_data()

config/config.py (MODIFY)
└── Add ENHANCED_DATA_SUFFIX config

utils/domain_loader.py (MODIFY)
└── Add load_enhanced_training_data() method

main.py (MODIFY)
└── Add --use-enhanced-data flag
```

---

## 4. Task Decomposition

### Task List

| ID | Task | Dependencies | Files | Priority |
|----|------|--------------|-------|----------|
| T0a | 기존 reasoning 파일 이름 변경 | - | `data/math/train/data/` | P0 |
| T0b | dataset_preparer.py 파일명 규칙 수정 | - | `utils/dataset_preparer.py` | P0 |
| T1 | DataEnhancer 클래스 구현 | - | `scripts/enhance_data.py` | P0 |
| T2 | 학습목표 생성 통합 | T1 | `scripts/enhance_data.py` | P0 |
| T3 | 과제분석 생성 통합 | T1, T2 | `scripts/enhance_data.py` | P0 |
| T4 | instruction 필드 확장 로직 | T2, T3 | `scripts/enhance_data.py` | P0 |
| T5 | CLI 인터페이스 구현 | T4 | `scripts/enhance_data.py` | P1 |
| T6 | DomainLoader 수정 | T5 | `utils/domain_loader.py` | P1 |
| T7 | main.py 플래그 추가 | T6 | `main.py` | P2 |
| T8 | 전체 데이터셋 처리 실행 | T5 | - | P2 |

### Dependency Graph
```
T0a (파일 이름 변경) ──► (완료)
T0b (dataset_preparer 수정) ──► (완료)

T1 (DataEnhancer) ──┬──► T2 (학습목표) ──┬──► T4 (instruction 확장) ──► T5 (CLI)
                    │                      │
                    └──► T3 (과제분석) ───┘

T5 ──► T6 (DomainLoader) ──► T7 (main.py)

T5 ──► T8 (데이터 처리 실행)
```

### T0a, T0b 구현 가이드

#### T0a: 기존 reasoning 파일 이름 변경
```bash
# 실행 명령
mv data/math/train/data/gsm8k_reasoning_train.json data/math/train/data/gsm8k_train_reasoning.json
mv data/math/train/data/math_reasoning_train.json data/math/train/data/math_train_reasoning.json
```

#### T0b: dataset_preparer.py 수정
```python
# utils/dataset_preparer.py

# 변경 전 (line 269)
save_json(records_full, output_base / f"gsm8k_reasoning_{split}.json")

# 변경 후
save_json(records_full, output_base / f"gsm8k_{split}_reasoning.json")

# 변경 전 (line 341)
save_json(records_full, output_base / f"math_reasoning_{split}.json")

# 변경 후
save_json(records_full, output_base / f"math_{split}_reasoning.json")
```

---

## 5. Implementation Strategy

### 5.1 Code Writing Method (TDD)

각 Task별 TDD 접근:

#### T1: DataEnhancer 클래스
```
RED: test_data_enhancer_initialization()
     test_load_source_data()
GREEN: DataEnhancer 클래스 기본 구현
REFACTOR: simplifier로 정리
```

#### T4: instruction 필드 확장
```
RED: test_enhance_instruction_format()
     test_instruction_contains_goal_and_analysis()
GREEN: enhance_instruction() 메서드 구현
REFACTOR: 문자열 템플릿 최적화
```

### 5.2 Incremental Verification Steps

| Stage | Verify | Tool | Pass Criteria |
|-------|--------|------|---------------|
| T1 완료 | DataEnhancer 초기화 | pytest | 테스트 통과 |
| T4 완료 | instruction 형식 | pytest | 학습목표/과제분석 포함 확인 |
| T5 완료 | CLI 동작 | manual test | 스크립트 실행 성공 |
| T8 완료 | 전체 데이터 처리 | manual test | 4개 데이터셋 처리 완료 |

### 5.3 Per-Task Implementation Guide

#### T1: DataEnhancer 클래스 구현

```python
# scripts/enhance_data.py

class DataEnhancer:
    """데이터 Enhancement를 위한 클래스"""

    def __init__(self, teacher_model, config):
        self.teacher = teacher_model
        self.config = config
        self.goal_generator = InstructionalGoalGenerator(teacher_model)
        self.analysis_generator = InstructionalAnalysis(teacher_model)

    def enhance_dataset(self, domain: str, dataset: str, model_name: str) -> str:
        """데이터셋 전체 enhancement 수행"""
        pass
```

#### T4: instruction 확장 로직

새로운 instruction 템플릿:
```python
ENHANCED_INSTRUCTION_TEMPLATE = """{original_instruction}

당신이 질문에 대한 답변을 하는 것은 {instructional_goal}라는 목표를 성취하는 것을 확인하기 위함입니다.
제시된 수행의 단계와 필요한 정보자원에 대한 정보를 전부 사용하여 풀이과정을 서술하세요.

[Instructional Analysis (수행 단계 및 정보자원)]
{task_analysis}
"""
```

---

## 6. Parallel Agent Execution Plan

### 6.1 Execution Mode
- [x] Sequential (의존성 있음, T1→T2→T3→T4→T5)
- [ ] Parallel
- [ ] Competitive

### 6.2 Agent Assignment

| Agent | Task | Model | Expected Output |
|-------|------|-------|-----------------|
| Main | T1-T5 | sonnet | 스크립트 및 로직 구현 |
| Verifier | T8 | haiku | 데이터 검증 |

### 6.3 Integration Strategy
1. T1-T5 순차 구현
2. T5 완료 후 T6, T7 병렬 가능
3. T8은 T5 완료 후 별도 실행
4. verifier로 최종 데이터 품질 검증

---

## 7. Quality Gates

### Phase 1: Context & Problem
- [x] 코드베이스 컨텍스트 수집 완료
- [x] 문제/목표 명확화 완료
- [x] Root cause 분석 완료

### Phase 2: Requirements Clarification
- [x] 사용자와 목표 정의 확인
- [x] 범위 경계 명확화 (전체 도메인)
- [x] 제약사항 식별 (파일명 형식)
- [x] 모호한 요구사항 명확화 완료

### Phase 3: Spec & Architecture
- [x] 테스트 가능한 AC 존재
- [x] 아키텍처 접근법 선택 완료 (전처리 스크립트)
- [x] 데이터 모델 정의 완료

### Phase 4: Task & Execution
- [x] Task 분해 완료 (Least-to-Most)
- [x] 의존성 매핑 완료
- [x] TDD 전략 정의
- [x] 검증 단계 정의

### Implementation Readiness
- [x] 모든 Task에 명확한 입출력 정의
- [x] Task별 구현 가이드 작성
- [x] 통합 전략 정의

---

## 8. 구현 예시 코드

### scripts/enhance_data.py (핵심 로직)

```python
#!/usr/bin/env python3
"""
ID-MAS 데이터 Enhancement 스크립트

Usage:
    python scripts/enhance_data.py --domain math --dataset gsm8k
    python scripts/enhance_data.py --all  # 전체 데이터셋 처리
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

from config import get_config
from design_modules.instructional_goal import InstructionalGoalGenerator
from design_modules.analysis import InstructionalAnalysis
from models.teacher_wrapper import TeacherModelWrapper


ENHANCED_INSTRUCTION_TEMPLATE = """{original_instruction}

당신이 질문에 대한 답변을 하는 것은 {instructional_goal}라는 목표를 성취하는 것을 확인하기 위함입니다.
제시된 수행의 단계와 필요한 정보자원에 대한 정보를 전부 사용하여 풀이과정을 서술하세요.

[Instructional Analysis (수행 단계 및 정보자원)]
{task_analysis}
"""


class DataEnhancer:
    def __init__(self, config):
        self.config = config
        self.teacher = TeacherModelWrapper()
        self.goal_generator = InstructionalGoalGenerator(self.teacher)
        self.analysis_generator = InstructionalAnalysis(self.teacher)

    def enhance_dataset(
        self,
        domain: str,
        dataset: str,
        model_suffix: str = None
    ) -> Path:
        """데이터셋 enhancement 수행"""

        # 1. 소스 데이터 로드
        source_path = self._get_source_path(domain, dataset)
        data = self._load_json(source_path)

        # 2. 학습목표 생성 (샘플 기반)
        samples = data[:min(25, len(data))]
        goal_result = self.goal_generator.generate(samples)
        instructional_goal = goal_result.get("instructional_goal", "")

        # 3. 과제분석 생성
        analysis_result = self.analysis_generator.analyze(instructional_goal)
        task_analysis = analysis_result.get("raw_output", "")

        # 4. instruction 확장
        enhanced_data = self._enhance_instructions(
            data, instructional_goal, task_analysis
        )

        # 5. 저장
        output_path = self._get_output_path(domain, dataset, model_suffix)
        self._save_json(enhanced_data, output_path)

        return output_path

    def _enhance_instructions(
        self,
        data: List[Dict],
        instructional_goal: str,
        task_analysis: str
    ) -> List[Dict]:
        """모든 데이터 항목의 instruction 확장"""
        enhanced = []
        for item in data:
            new_item = item.copy()
            new_item["instruction"] = ENHANCED_INSTRUCTION_TEMPLATE.format(
                original_instruction=item.get("instruction", ""),
                instructional_goal=instructional_goal,
                task_analysis=task_analysis
            )
            # 메타데이터 추가
            new_item["_enhanced"] = True
            new_item["_instructional_goal"] = instructional_goal
            enhanced.append(new_item)
        return enhanced

    def _get_source_path(self, domain: str, dataset: str) -> Path:
        return Path(f"data/{domain}/train/data/{dataset}_train.json")

    def _get_output_path(self, domain: str, dataset: str, model_suffix: str) -> Path:
        suffix = model_suffix or self.teacher.model_name.split("/")[-1]
        return Path(f"data/{domain}/train/data/{dataset}_train_ID-MAS_{suffix}.json")

    def _load_json(self, path: Path) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_json(self, data: List[Dict], path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="ID-MAS Data Enhancement")
    parser.add_argument("--domain", type=str, help="Domain to process")
    parser.add_argument("--dataset", type=str, help="Dataset to process")
    parser.add_argument("--all", action="store_true", help="Process all datasets")
    parser.add_argument("--model-suffix", type=str, help="Model suffix for output file")

    args = parser.parse_args()

    config = get_config()
    enhancer = DataEnhancer(config)

    if args.all:
        datasets = [
            ("math", "gsm8k"),
            ("math", "math"),
            ("logical", "reclor"),
            ("commonsense", "arc_c"),
        ]
        for domain, dataset in datasets:
            output = enhancer.enhance_dataset(domain, dataset, args.model_suffix)
            print(f"Enhanced: {output}")
    else:
        output = enhancer.enhance_dataset(args.domain, args.dataset, args.model_suffix)
        print(f"Enhanced: {output}")


if __name__ == "__main__":
    main()
```

---

## 9. 실행 순서

### Step 0: 파일명 규칙 통일 (T0a, T0b)
1. **T0a**: 기존 reasoning 파일 이름 변경
   ```bash
   mv data/math/train/data/gsm8k_reasoning_train.json data/math/train/data/gsm8k_train_reasoning.json
   mv data/math/train/data/math_reasoning_train.json data/math/train/data/math_train_reasoning.json
   ```
2. **T0b**: `utils/dataset_preparer.py` 파일명 생성 규칙 수정

### Step 1: 데이터 Enhancement (T1-T5)
3. **T1-T5**: `scripts/enhance_data.py` 구현

### Step 2: 데이터 생성 (T8)
4. **T8**: 전체 데이터셋 처리
   ```bash
   python scripts/enhance_data.py --all
   ```

### Step 3: 통합 (T6-T7, 선택적)
5. **T6**: DomainLoader에 enhanced 데이터 로드 메서드 추가
6. **T7**: main.py에 `--use-enhanced-data` 플래그 추가

---

## 10. 예상 출력

### Enhanced Instruction 예시 (GSM8K)

```json
{
  "instruction": "You are a helpful math assistant.\nSolve this mathematical problem step by step. Show your reasoning clearly and use proper mathematical notation.\nYour final answer MUST be within \\boxed{}.\nExample: \\boxed{42}\n\n당신이 질문에 대한 답변을 하는 것은 Generate coherent, step-by-step mathematical reasoning in natural language that leads to a correct numerical answer for grade-school level math problems.라는 목표를 성취하는 것을 확인하기 위함입니다. \n제시된 수행의 단계와 필요한 정보자원에 대한 정보를 전부 사용하여 풀이과정을 서술하세요.\n\n[Instructional Analysis (수행 단계 및 정보자원)]\nTerminal Goal: Generate coherent, step-by-step mathematical reasoning...\n├── [1] Problem Comprehension\n│   ├── [1-1] Identify given information\n│   └── [1-2] Identify what is asked\n├── [2] Solution Planning\n│   └── [2-1] Determine appropriate operations\n└── [3] Execution & Verification\n    ├── [3-1] Perform calculations\n    └── [3-2] Verify reasonableness",
  "input": "Natalia sold clips to 48 of her friends in April...",
  "output": "The answer is \\boxed{72}"
}
```
