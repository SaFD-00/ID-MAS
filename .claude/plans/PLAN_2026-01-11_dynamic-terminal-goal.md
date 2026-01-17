# Dynamic Terminal Goal Generation Plan

## Overview

현재 하드코딩된 Terminal Goal을 **Design Phase의 첫 번째 에이전트**로 동적 생성하도록 변경.
**Teacher Model** (teacher_config로 설정된 모델) 사용.

## Current State Analysis

### 현재 Design Phase 구조 ([main.py:145-218](main.py#L145-L218))
```
[Step 1] Learning Objective (하드코딩된 terminal_goal 사용)
[Step 2] Instructional Analysis
[Step 3] Performance Objectives
[Step 4] Rubric Development
```

### 변경 후 Design Phase 구조
```
[Step 0] Terminal Goal Generation ← 새로운 첫 번째 에이전트
[Step 1] Learning Objective (생성된 terminal_goal 사용)
[Step 2] Instructional Analysis
[Step 3] Performance Objectives
[Step 4] Rubric Development
```

### 현재 Terminal Goal 하드코딩 위치
- [config/config.py:310-313](config/config.py#L310-L313) - `TERMINAL_GOALS` dict
- [utils/domain_loader.py:40-53](utils/domain_loader.py#L40-L53) - `TERMINAL_GOALS` dict

### 현재 Instructional Design 구조
- 저장 위치: `data/{domain}/train/{teacher_model}/instructional-design/{domain}_{dataset}_design.json`
- 이미 `terminal_goal` 필드가 존재 (Line 5)

### 지원 데이터셋
| Domain | Dataset | Train File |
|--------|---------|------------|
| math | gsm8k | `data/math/train/data/gsm8k_train.json` |
| math | math | `data/math/train/data/math_train.json` |
| logical | reclor | `data/logical/train/data/reclor_train.json` |
| commonsense | arc_c | `data/commonsense/train/data/arc_c_train.json` |

---

## Implementation Plan

### Step 1: Sample Data Extractor Module

**목적**: 각 데이터셋별 대표 샘플 추출 및 저장 (Design Phase 실행 전 미리 준비)

**파일**: `utils/sample_extractor.py` (신규)

**기능**:
1. 데이터셋별 10-30개 샘플 추출
2. 다양성 확보 전략:
   - Random sampling (기본)
   - 문제 길이 기반 stratified sampling
   - MATH: 문제 유형 추론 후 균등 분배
3. 샘플 저장: `data/{domain}/train/data/{dataset}_samples.json`

**샘플 추출 로직**:
```python
def extract_samples(
    dataset_path: Path,
    num_samples: int = 20,
    strategy: str = "diverse"  # "random", "diverse", "stratified"
) -> List[Dict]
```

**CLI로 미리 실행**:
```bash
python -m utils.sample_extractor --domain math --dataset math --num-samples 25
```

---

### Step 2: Terminal Goal Generator (Design Phase Agent)

**목적**: Design Phase의 **첫 번째 에이전트**로 Terminal Goal 자동 생성

**파일**: `design_modules/terminal_goal.py` (신규)

**핵심 변경**:
- ~~GPT~~ → **Teacher Model** (teacher_config) 사용
- 기존 design_modules와 동일한 패턴 (InstructionalAnalysis, PerformanceObjectives 등)

**클래스 구조**:
```python
from models.teacher_wrapper import TeacherModelWrapper

class TerminalGoalGenerator:
    """Design Phase Step 0: Terminal Goal 생성 에이전트"""

    def __init__(self, teacher_config: dict = None):
        """Teacher Model 사용 (GPT가 아닌 teacher_config 모델)"""
        self.llm = TeacherModelWrapper(teacher_config)

    def generate(
        self,
        train_samples: List[Dict],
        domain: str,
        dataset: str,
        prompt_template: str = None
    ) -> Dict[str, Any]:
        """
        Terminal Goal 생성

        Args:
            train_samples: 샘플 데이터 (10-30개)
            domain: 도메인 이름
            dataset: 데이터셋 이름
            prompt_template: 커스텀 프롬프트 (None이면 기본 사용)

        Returns:
            {
                "terminal_goal": "The model should be able to...",
                "cognitive_level": "Apply",
                "rationale": "...",
                "raw_output": "..."
            }
        """
```

---

### Step 3: Prompt Improvement

**현재 프롬프트 분석**:
```
You are an expert instructional designer and AI learning researcher...
### Output Format
Terminal Goal: { Terminal Goal }
```

**개선 제안**:

1. **도메인 컨텍스트 추가**:
   ```
   ### Domain Context
   Domain: {domain}
   Dataset: {dataset}
   Sample Count: {sample_count}
   ```

2. **출력 형식 구체화** (JSON):
   ```
   ### Output Format (JSON)
   {
     "terminal_goal": "The model should be able to...",
     "cognitive_level": "Apply|Analyze|Evaluate|Create",
     "key_verb": "solve|analyze|evaluate|..."
   }
   ```

3. **Few-shot Examples 추가**:
   ```
   ### Examples
   [GSM8K] Terminal Goal: The model should be able to generate coherent...
   [RECLOR] Terminal Goal: The model should be able to analyze logical...
   ```

4. **Constraint 명시**:
   ```
   ### Constraints
   - Single verb only
   - No theory names (Bloom's Taxonomy)
   - Action-oriented, measurable
   ```

---

### Step 4: Integration with Existing Code

**수정 파일**:

1. **[main.py](main.py)** - `IDMASPipeline.__init__()` 및 `run_design_phase()` 수정
   - Step 0: Terminal Goal 생성 에이전트 추가
   - 샘플 데이터 로드 → Teacher Model로 생성

2. **[config/domains.py](config/domains.py)** - `get_terminal_goal()` 수정
   - 하드코딩 → design JSON 파일에서 로드
   - Fallback: 기존 하드코딩 값

3. **[utils/domain_loader.py](utils/domain_loader.py)** - `get_learning_objective()` 수정
   - design JSON 파일에서 terminal_goal 로드

**main.py `run_design_phase()` 변경**:
```python
def run_design_phase(self, learning_objective: Optional[str] = None) -> Dict:
    print("\n" + "=" * 60)
    print("INSTRUCTIONAL DESIGN PHASE")
    print("=" * 60)

    # [Step 0] Terminal Goal Generation (NEW - 첫 번째 에이전트)
    print(f"\n[Step 0] Terminal Goal Generation")
    samples_path = self.raw_data_dir / f"{self.train_dataset}_samples.json"

    if samples_path.exists():
        with open(samples_path, 'r') as f:
            train_samples = json.load(f)
        print(f"  Loaded {len(train_samples)} samples from {samples_path.name}")

        # Teacher Model로 Terminal Goal 생성
        terminal_goal_result = self.terminal_goal_gen.generate(
            train_samples=train_samples,
            domain=self.domain,
            dataset=self.train_dataset
        )
        self.terminal_goal = terminal_goal_result["terminal_goal"]
        print(f"  Generated: {self.terminal_goal[:80]}...")
    else:
        print(f"  [Warning] Samples not found. Using fallback terminal goal.")
        # Fallback: 기존 하드코딩 값 사용

    # [Step 1] Learning Objective
    if learning_objective is None:
        learning_objective = self.terminal_goal
    ...
```

**데이터 흐름**:
```
0. Sample Extraction (미리 실행)
   └─> data/{domain}/train/data/{dataset}_samples.json

1. Design Phase 시작
   └─> [Step 0] Terminal Goal Generation
       └─> samples + prompt → Teacher Model → terminal_goal

2. Save to Design File
   └─> data/{domain}/train/{teacher}/instructional-design/{domain}_{dataset}_design.json
       └─> "terminal_goal": "The model should be able to..."
       └─> "terminal_goal_metadata": {
             "generated_at": "2026-01-11T...",
             "sample_count": 25,
             "model": "Qwen/Qwen2.5-72B-Instruct",  ← Teacher Model
             "prompt_version": "v1"
           }

3. Learning Phase에서 로드
   └─> get_terminal_goal() reads from design JSON
```

---

### Step 5: CLI Enhancement

**main.py 인자 추가**:
```bash
# 기존: 고정 terminal goal 사용
python main.py --domain math --dataset math

# 신규: terminal goal 재생성
python main.py --domain math --dataset math --regenerate-terminal-goal

# 신규: 커스텀 프롬프트 사용
python main.py --domain math --dataset math --terminal-goal-prompt "custom_prompt.txt"
```

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `utils/sample_extractor.py` | CREATE | 샘플 데이터 추출 모듈 (미리 실행) |
| `design_modules/terminal_goal.py` | CREATE | **Design Phase Step 0 에이전트** (Teacher Model 사용) |
| `prompts/terminal_goal_prompts.py` | CREATE | 개선된 프롬프트 템플릿 |
| `config/domains.py` | MODIFY | `get_terminal_goal()` - JSON에서 로드 |
| `utils/domain_loader.py` | MODIFY | JSON 기반 로드 |
| `main.py` | MODIFY | **`run_design_phase()` Step 0 추가**, CLI 인자 |

---

## Sample Data Extraction Strategy

### MATH 데이터셋 (EleutherAI/hendrycks_math)

**원본 데이터 사용**: HuggingFace `EleutherAI/hendrycks_math`에서 직접 추출

**샘플링 전략**:
- **5 Types**: Algebra, Geometry, Number Theory, Counting & Probability, Precalculus (또는 전체 7 types)
- **Levels**: 1, 3, 5만 선택
- **샘플 수**: 각 (type, level) 조합에서 1개씩 랜덤 추출
- **총 샘플**: 5 types × 3 levels = **15개** (또는 7 types × 3 levels = 21개)

```python
# MATH 샘플 추출 로직
from datasets import load_dataset

def extract_math_samples():
    ds = load_dataset('EleutherAI/hendrycks_math', 'all')['train']

    target_levels = ['Level 1', 'Level 3', 'Level 5']
    samples = []

    for math_type in ds.unique('type'):
        for level in target_levels:
            # 해당 type + level 필터링 후 1개 랜덤 추출
            filtered = [x for x in ds if x['type'] == math_type and x['level'] == level]
            if filtered:
                samples.append(random.choice(filtered))

    return samples  # 15~21개
```

**저장 위치**: `data/math/train/data/math_samples.json`

---

### 나머지 데이터셋 (GSM8K, ReClor, ARC-C)

**샘플링 전략**:
1. **1차**: 60개 랜덤 추출
2. **2차**: 다양성 기반 15개 선별
   - 문제 길이 분포 (short, medium, long)
   - 키워드 다양성 (TF-IDF 또는 단순 키워드)
   - 정답 유형 다양성 (숫자, 텍스트, MCQ 등)

```python
def extract_diverse_samples(data: List[Dict], num_final: int = 15):
    # Step 1: 60개 랜덤 추출
    random_60 = random.sample(data, min(60, len(data)))

    # Step 2: 다양성 기반 15개 선별
    # - 문제 길이로 3그룹 분류 (short/medium/long)
    # - 각 그룹에서 5개씩 선별
    by_length = categorize_by_length(random_60)

    diverse_15 = []
    for group in ['short', 'medium', 'long']:
        diverse_15.extend(select_diverse(by_length[group], 5))

    return diverse_15
```

**저장 위치**:
- `data/math/train/data/gsm8k_samples.json`
- `data/logical/train/data/reclor_samples.json`
- `data/commonsense/train/data/arc_c_samples.json`

---

## Improved Prompt Template

```python
TERMINAL_GOAL_PROMPT = """
You are an expert instructional designer and AI learning researcher.

## Context
- Domain: {domain}
- Dataset: {dataset}
- Sample Size: {sample_count} items

## Task
Analyze the test items and derive a single Terminal Goal that captures the core competency required.

## Guidelines
1. Focus on the highest cognitive level demonstrated in the samples
2. Use a single, observable, measurable verb
3. Start with: "The model should be able to..."
4. Do NOT mention learning theories by name
5. Make it actionable and domain-specific

## Reference Examples
- Math (GSM8K): "The model should be able to generate step-by-step mathematical reasoning that produces correct numerical answers."
- Logical (ReClor): "The model should be able to analyze logical arguments and identify valid conclusions."

## Input Data
{train_data}

## Output (JSON)
{{
  "terminal_goal": "The model should be able to ...",
  "cognitive_level": "Remember|Understand|Apply|Analyze|Evaluate|Create",
  "primary_verb": "...",
  "rationale": "Brief explanation of why this goal was chosen"
}}
"""
```

---

## Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Teacher Model 생성 결과 품질 불안정 | Medium | Validation + fallback to hardcoded |
| 샘플 대표성 부족 | Low | Diverse sampling strategy |
| Teacher Model 리소스 사용 | Low | 캐싱 (design JSON에 저장, 한 번만 생성) |

---

## Execution Order

1. `utils/sample_extractor.py` 생성 및 테스트
2. `prompts/terminal_goal_prompts.py` 생성
3. `design_modules/terminal_goal.py` 생성 및 테스트
4. `config/domains.py` 수정 (JSON 로드)
5. `main.py` 수정 (CLI + 통합)
6. 전체 테스트

---

## Validation Criteria

- [ ] 각 데이터셋별 샘플 파일 생성됨
- [ ] Terminal Goal이 design JSON에 저장됨
- [ ] 기존 파이프라인이 새 terminal goal로 정상 동작
- [ ] CLI에서 재생성 옵션 동작
- [ ] Fallback 로직 동작 확인
