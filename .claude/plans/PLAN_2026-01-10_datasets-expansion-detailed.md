# 논리/상식 데이터셋 추가 구체화 계획

## 개요

논리(Logical)와 상식(Commonsense) 도메인을 신규로 추가하고, 지정된 학습 데이터셋 + In-Domain 테스트셋 + OOD 평가셋을 현재 파이프라인에 통합합니다.

**기존 계획**: [PLAN_2026-01-10_datasets-expansion.md](.claude/plans/PLAN_2026-01-10_datasets-expansion.md)

## 확정된 사항

### 도메인 키 네이밍

- **논리 도메인**: `logical`
- **상식 도메인**: `commonsense`
- CLI 옵션: `--domain logical` 또는 `--domain commonsense`

### 데이터셋 범위

**학습 데이터셋 (Training)**
- **논리**: ReClor
- **상식**: ARC-c, StrategyQA

**평가 데이터셋 (Evaluation)**
- **In-Domain**: 위 학습 데이터셋의 테스트 셋
- **OOD (논리)**: BBH (논리 관련 서브태스크만), ANLI-R2, ANLI-R3
- **OOD (상식)**: OpenBookQA

### BBH 서브태스크 범위

**논리적 추론 관련 서브태스크만 포함** (총 9개):
1. `boolean_expressions` - Boolean 논리식 평가
2. `formal_fallacies` - 형식논리 오류 탐지
3. `logical_deduction_three_objects` - 3개 객체 논리 추론
4. `logical_deduction_five_objects` - 5개 객체 논리 추론
5. `logical_deduction_seven_objects` - 7개 객체 논리 추론
6. `tracking_shuffled_objects_three_objects` - 3개 객체 추적
7. `tracking_shuffled_objects_five_objects` - 5개 객체 추적
8. `tracking_shuffled_objects_seven_objects` - 7개 객체 추적
9. `web_of_lies` - 거짓말 추론

### HuggingFace 데이터셋 정보

| 데이터셋 | HF 경로 | Config | Train Split | Test Split | Answer Type |
|---------|---------|--------|-------------|------------|-------------|
| ReClor | `community-datasets/reclor` | None | `train` | `test` | MCQ (A/B/C/D) |
| ARC-c | `allenai/ai2_arc` | `ARC-Challenge` | `train` | `test` | MCQ (A/B/C/D) |
| StrategyQA | `wics/strategy-qa` | None | None | `test` | BOOLEAN (Yes/No) |
| OpenBookQA | `allenai/openbookqa` | `main` | `train` | `test` | MCQ (A/B/C/D) |
| ANLI | `facebook/anli` | None | `train_r2`, `train_r3` | `test_r2`, `test_r3` | MCQ (A/B/C) |
| BBH | `lukaemon/bbh` | `{subtask}` | None | `test` | MCQ/BOOLEAN/TEXT |

**주의사항**:
- StrategyQA는 train split이 없으므로 평가용으로만 사용
- ANLI는 R2, R3 두 라운드를 별도로 처리
- BBH는 서브태스크별로 config가 다름

### ANLI 라벨 표현 방식

**MCQ 형식으로 통일** (A/B/C):
- `A. entailment` (0)
- `B. neutral` (1)
- `C. contradiction` (2)

입력 포맷:
```
Premise: {premise}
Hypothesis: {hypothesis}

A. entailment
B. neutral
C. contradiction
```

출력 포맷: `\boxed{A}`, `\boxed{B}`, 또는 `\boxed{C}`

## 디렉토리 구조

```
data/
├── logical/
│   ├── train/
│   │   └── data/
│   │       └── reclor_train.json
│   └── eval/
│       └── data/
│           ├── reclor_test.json
│           ├── bbh_boolean_expressions_test.json
│           ├── bbh_formal_fallacies_test.json
│           ├── bbh_logical_deduction_three_objects_test.json
│           ├── bbh_logical_deduction_five_objects_test.json
│           ├── bbh_logical_deduction_seven_objects_test.json
│           ├── bbh_tracking_shuffled_objects_three_objects_test.json
│           ├── bbh_tracking_shuffled_objects_five_objects_test.json
│           ├── bbh_tracking_shuffled_objects_seven_objects_test.json
│           ├── bbh_web_of_lies_test.json
│           ├── anli_r2_test.json
│           └── anli_r3_test.json
└── commonsense/
    ├── train/
    │   └── data/
    │       └── arc_c_train.json
    └── eval/
        └── data/
            ├── arc_c_test.json
            ├── strategyqa_test.json
            └── openbookqa_test.json
```

## 구현 계획

### Step 1: 데이터 준비 스크립트 확장

**파일**: `utils/dataset_preparer.py`

#### 1.1 DATASET_PROMPTS 확장

```python
DATASET_PROMPTS = {
    # ... 기존 프롬프트 ...

    # 논리 도메인
    "reclor": """You are a logical reasoning assistant. Read the passage and question, then select the correct option (A, B, C, or D). Your final answer MUST be a single letter within \\boxed{}.
Example: \\boxed{A}""",

    "anli": """You are a natural language inference assistant. Determine the relationship between the premise and hypothesis. Choose from: A. entailment, B. neutral, C. contradiction. Your final answer MUST be a single letter within \\boxed{}.
Example: \\boxed{A}""",

    # 상식 도메인
    "arc_c": """You are a helpful commonsense science assistant. Solve the problem and select the correct option (A, B, C, or D). Your final answer MUST be a single letter within \\boxed{}.
Example: \\boxed{A}""",

    "strategyqa": """You are a helpful commonsense reasoning assistant. Answer the question with Yes or No based on reliable commonsense knowledge. Your final answer MUST be \\boxed{Yes} or \\boxed{No}.
Example: \\boxed{Yes}""",

    "openbookqa": """You are a helpful science question-answering assistant. Use the given options and choose the best answer (A, B, C, or D). Your final answer MUST be a single letter within \\boxed{}.
Example: \\boxed{A}""",
}

# BBH 서브태스크별 템플릿
BBH_PROMPTS = {
    "boolean_expressions": """You are a helpful reasoning assistant. Evaluate the boolean expression and answer with True or False. Your final answer MUST be \\boxed{True} or \\boxed{False}.
Example: \\boxed{True}""",

    "formal_fallacies": """You are a helpful reasoning assistant. Determine if the argument is valid or invalid. Your final answer MUST be \\boxed{valid} or \\boxed{invalid}.
Example: \\boxed{valid}""",

    "logical_deduction_three_objects": """You are a helpful reasoning assistant. Solve the logical deduction problem and select the correct option (A, B, or C). Your final answer MUST be a single letter within \\boxed{}.
Example: \\boxed{A}""",

    # ... 나머지 서브태스크별 템플릿

    "default_mcq": """You are a helpful reasoning assistant. Solve the problem and select the correct option. Your final answer MUST be a single letter within \\boxed{}.
Example: \\boxed{A}""",
}

# BBH 논리 추론 관련 서브태스크
BBH_LOGICAL_SUBTASKS = [
    "boolean_expressions",
    "formal_fallacies",
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "web_of_lies",
]
```

#### 1.2 데이터셋 처리 함수 추가

```python
def process_reclor(train_dir: Path, eval_dir: Path):
    """
    Process ReClor dataset.

    ReClor format:
    - context: passage text
    - question: question text
    - answers: list of 4 choices
    - label: answer index (0-3)
    """
    print("\n[ReClor] Processing...")
    dataset_id = "community-datasets/reclor"

    for split in ["train", "test"]:
        print(f"  Loading {split} split...")
        data = load_dataset(dataset_id, split=split)

        records = []
        for item in data:
            context = item.get("context", "")
            question = item.get("question", "")
            answers = item.get("answers", [])
            label = item.get("label", 0)

            # 선택지 포맷팅
            question_with_choices = format_mcq_input(
                f"{context}\n\n{question}",
                answers
            )

            # 정답 라벨 (0-3 -> A-D)
            answer_letter = chr(65 + label)

            records.append({
                "instruction": DATASET_PROMPTS["reclor"],
                "input": question_with_choices,
                "output": format_output(None, answer_letter)
            })

        output_base = train_dir if split == "train" else eval_dir
        save_json(records, output_base / f"reclor_{split}.json")


def process_arc_c(train_dir: Path, eval_dir: Path):
    """
    Process ARC-Challenge dataset.

    ARC format:
    - question: question text
    - choices: {"text": [...], "label": [...]}
    - answerKey: answer label (e.g., "A", "1")
    """
    print("\n[ARC-Challenge] Processing...")
    dataset_id = "allenai/ai2_arc"
    config = "ARC-Challenge"

    for split in ["train", "test"]:
        print(f"  Loading {split} split...")
        data = load_dataset(dataset_id, config, split=split)

        records = []
        for item in data:
            question = item["question"]
            choices = item["choices"]
            answer_key = item["answerKey"]

            # 선택지 텍스트 추출
            choice_texts = choices["text"]
            choice_labels = choices["label"]

            # 선택지를 A, B, C, D로 재매핑
            question_with_choices = format_mcq_input(question, choice_texts)

            # answerKey를 A-D로 변환 (1->A, 2->B 또는 A->A)
            if answer_key.isdigit():
                answer_letter = chr(65 + int(answer_key) - 1)
            else:
                answer_letter = answer_key

            records.append({
                "instruction": DATASET_PROMPTS["arc_c"],
                "input": question_with_choices,
                "output": format_output(None, answer_letter)
            })

        output_base = train_dir if split == "train" else eval_dir
        save_json(records, output_base / f"arc_c_{split}.json")


def process_strategyqa(eval_dir: Path):
    """
    Process StrategyQA dataset (test only).

    StrategyQA format:
    - question: yes/no question
    - answer: boolean (true/false)
    """
    print("\n[StrategyQA] Processing...")
    dataset_id = "wics/strategy-qa"

    print("  Loading test split...")
    data = load_dataset(dataset_id, split="test")

    records = []
    for item in data:
        question = item["question"]
        answer_bool = item["answer"]

        # boolean을 Yes/No로 변환
        answer_text = "Yes" if answer_bool else "No"

        records.append({
            "instruction": DATASET_PROMPTS["strategyqa"],
            "input": question,
            "output": format_output(None, answer_text)
        })

    save_json(records, eval_dir / "strategyqa_test.json")


def process_openbookqa(eval_dir: Path):
    """
    Process OpenBookQA dataset.

    OpenBookQA format:
    - question_stem: question text
    - choices: {"text": [...], "label": [...]}
    - answerKey: answer label (e.g., "A")
    """
    print("\n[OpenBookQA] Processing...")
    dataset_id = "allenai/openbookqa"
    config = "main"

    print("  Loading test split...")
    data = load_dataset(dataset_id, config, split="test")

    records = []
    for item in data:
        question = item["question_stem"]
        choices = item["choices"]
        answer_key = item["answerKey"]

        # 선택지 텍스트 추출
        choice_texts = choices["text"]

        # MCQ 포맷
        question_with_choices = format_mcq_input(question, choice_texts)

        records.append({
            "instruction": DATASET_PROMPTS["openbookqa"],
            "input": question_with_choices,
            "output": format_output(None, answer_key)
        })

    save_json(records, eval_dir / "openbookqa_test.json")


def process_anli(eval_dir: Path, round_name: str):
    """
    Process ANLI dataset for a specific round.

    ANLI format:
    - premise: premise text
    - hypothesis: hypothesis text
    - label: 0 (entailment), 1 (neutral), 2 (contradiction)

    Args:
        eval_dir: Output directory
        round_name: "r2" or "r3"
    """
    print(f"\n[ANLI-{round_name.upper()}] Processing...")
    dataset_id = "facebook/anli"

    split_name = f"test_{round_name}"
    print(f"  Loading {split_name} split...")
    data = load_dataset(dataset_id, split=split_name)

    records = []
    label_map = {0: "A", 1: "B", 2: "C"}

    for item in data:
        premise = item["premise"]
        hypothesis = item["hypothesis"]
        label = item["label"]

        # 입력 포맷: Premise + Hypothesis + 선택지
        input_text = f"""Premise: {premise}
Hypothesis: {hypothesis}

A. entailment
B. neutral
C. contradiction"""

        answer_letter = label_map[label]

        records.append({
            "instruction": DATASET_PROMPTS["anli"],
            "input": input_text,
            "output": format_output(None, answer_letter)
        })

    save_json(records, eval_dir / f"anli_{round_name}_test.json")


def process_bbh(eval_dir: Path, subtasks: List[str]):
    """
    Process BBH dataset for specific subtasks.

    BBH format varies by subtask:
    - input: question text
    - target: answer text

    Args:
        eval_dir: Output directory
        subtasks: List of subtask names
    """
    print("\n[BBH] Processing...")
    dataset_id = "lukaemon/bbh"

    for subtask in subtasks:
        print(f"  Loading subtask: {subtask}...")
        try:
            data = load_dataset(dataset_id, subtask, split="test")

            # 서브태스크별 프롬프트 선택
            prompt = BBH_PROMPTS.get(subtask, BBH_PROMPTS["default_mcq"])

            records = []
            for item in data:
                input_text = item["input"]
                target = item["target"]

                records.append({
                    "instruction": prompt,
                    "input": input_text,
                    "output": format_output(None, target)
                })

            save_json(records, eval_dir / f"bbh_{subtask}_test.json")

        except Exception as e:
            print(f"    Error loading {subtask}: {e}")
```

#### 1.3 main() 함수 확장

```python
def main():
    """Main entry point."""
    set_random_seed()
    print("=" * 60)
    print("Dataset Preparation Script")
    print("=" * 60)

    # ==========================================
    # Process Math Domain (기존 코드)
    # ==========================================
    math_dir = DATA_DIR / "math"
    # ... 기존 math 도메인 처리 ...

    # ==========================================
    # Process Logical Domain
    # ==========================================
    print("\n" + "=" * 60)
    print("LOGICAL DOMAIN")
    print("=" * 60)

    logical_dir = DATA_DIR / "logical"
    logical_train_dir = logical_dir / "train" / "data"
    logical_eval_dir = logical_dir / "eval" / "data"
    logical_train_dir.mkdir(parents=True, exist_ok=True)
    logical_eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. ReClor (train + test)
    process_reclor(logical_train_dir, logical_eval_dir)

    # 2. ANLI R2, R3 (test only)
    process_anli(logical_eval_dir, "r2")
    process_anli(logical_eval_dir, "r3")

    # 3. BBH 논리 서브태스크 (test only)
    process_bbh(logical_eval_dir, BBH_LOGICAL_SUBTASKS)

    # ==========================================
    # Process Commonsense Domain
    # ==========================================
    print("\n" + "=" * 60)
    print("COMMONSENSE DOMAIN")
    print("=" * 60)

    commonsense_dir = DATA_DIR / "commonsense"
    commonsense_train_dir = commonsense_dir / "train" / "data"
    commonsense_eval_dir = commonsense_dir / "eval" / "data"
    commonsense_train_dir.mkdir(parents=True, exist_ok=True)
    commonsense_eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. ARC-Challenge (train + test)
    process_arc_c(commonsense_train_dir, commonsense_eval_dir)

    # 2. StrategyQA (test only)
    process_strategyqa(commonsense_eval_dir)

    # 3. OpenBookQA (test only)
    process_openbookqa(commonsense_eval_dir)

    # ==========================================
    # Print Summary
    # ==========================================
    print("\n" + "=" * 60)
    print("Dataset preparation completed!")
    print("=" * 60)

    print("\n[Summary]")
    for domain_dir in [math_dir, logical_dir, commonsense_dir]:
        if not domain_dir.exists():
            continue
        print(f"\n{domain_dir.name}/")
        for section_name, section_dir in [("train/data", domain_dir / "train" / "data"),
                                           ("eval/data", domain_dir / "eval" / "data")]:
            if not section_dir.exists():
                continue
            print(f"  {section_name}/")
            for json_file in sorted(section_dir.glob("*.json")):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"    {json_file.name}: {len(data)} records")
```

### Step 2: 도메인/데이터셋 설정 동기화

#### 2.1 `utils/domain_loader.py` 확장

**위치**: Line 39-43 (TERMINAL_GOALS), Line 45-63 (DOMAIN_CONFIG)

```python
# Terminal Goals for each training dataset
TERMINAL_GOALS = {
    # Math domain
    "gsm8k": "Generate coherent, step-by-step mathematical reasoning in natural language that leads to a correct numerical answer for grade-school level math problems.",
    "math": "Solve advanced mathematical problems by selecting appropriate mathematical concepts and constructing logically valid, multi-step reasoning that leads to a correct solution.",

    # Logical domain
    "reclor": "Analyze logical reasoning problems by comprehending complex passages, identifying logical relationships, and selecting the most appropriate conclusion based on formal reasoning principles.",

    # Commonsense domain
    "arc_c": "Apply commonsense scientific knowledge to solve elementary science problems by understanding fundamental concepts and selecting the correct answer from multiple choices.",
}

DOMAIN_CONFIG = {
    "math": {
        # ... 기존 설정 ...
    },
    "logical": {
        "training_datasets": {
            "reclor": {"filename": "reclor_train.json", "answer_type": AnswerType.MCQ},
        },
        "eval_datasets": {
            "reclor": {"filename": "reclor_test.json", "answer_type": AnswerType.MCQ},
            "anli_r2": {"filename": "anli_r2_test.json", "answer_type": AnswerType.MCQ},
            "anli_r3": {"filename": "anli_r3_test.json", "answer_type": AnswerType.MCQ},
            "bbh_boolean_expressions": {"filename": "bbh_boolean_expressions_test.json", "answer_type": AnswerType.BOOLEAN},
            "bbh_formal_fallacies": {"filename": "bbh_formal_fallacies_test.json", "answer_type": AnswerType.TEXT},
            "bbh_logical_deduction_three_objects": {"filename": "bbh_logical_deduction_three_objects_test.json", "answer_type": AnswerType.MCQ},
            "bbh_logical_deduction_five_objects": {"filename": "bbh_logical_deduction_five_objects_test.json", "answer_type": AnswerType.MCQ},
            "bbh_logical_deduction_seven_objects": {"filename": "bbh_logical_deduction_seven_objects_test.json", "answer_type": AnswerType.MCQ},
            "bbh_tracking_shuffled_objects_three_objects": {"filename": "bbh_tracking_shuffled_objects_three_objects_test.json", "answer_type": AnswerType.MCQ},
            "bbh_tracking_shuffled_objects_five_objects": {"filename": "bbh_tracking_shuffled_objects_five_objects_test.json", "answer_type": AnswerType.MCQ},
            "bbh_tracking_shuffled_objects_seven_objects": {"filename": "bbh_tracking_shuffled_objects_seven_objects_test.json", "answer_type": AnswerType.MCQ},
            "bbh_web_of_lies": {"filename": "bbh_web_of_lies_test.json", "answer_type": AnswerType.BOOLEAN},
        },
        "default_answer_type": AnswerType.MCQ,
        "domain_category": "logical_reasoning",
        "data_dir": "data/logical"
    },
    "commonsense": {
        "training_datasets": {
            "arc_c": {"filename": "arc_c_train.json", "answer_type": AnswerType.MCQ},
        },
        "eval_datasets": {
            "arc_c": {"filename": "arc_c_test.json", "answer_type": AnswerType.MCQ},
            "strategyqa": {"filename": "strategyqa_test.json", "answer_type": AnswerType.BOOLEAN},
            "openbookqa": {"filename": "openbookqa_test.json", "answer_type": AnswerType.MCQ},
        },
        "default_answer_type": AnswerType.MCQ,
        "domain_category": "commonsense_reasoning",
        "data_dir": "data/commonsense"
    }
}
```

#### 2.2 `utils/dataset_registry.py` 확장

**위치**: Line 23-31 (DOMAIN_CONFIG)

```python
DOMAIN_CONFIG = {
    "math": {
        # ... 기존 설정 ...
    },
    "logical": {
        "training_datasets": ["reclor"],
        "eval_datasets": [
            "reclor", "anli_r2", "anli_r3",
            "bbh_boolean_expressions", "bbh_formal_fallacies",
            "bbh_logical_deduction_three_objects", "bbh_logical_deduction_five_objects",
            "bbh_logical_deduction_seven_objects",
            "bbh_tracking_shuffled_objects_three_objects",
            "bbh_tracking_shuffled_objects_five_objects",
            "bbh_tracking_shuffled_objects_seven_objects",
            "bbh_web_of_lies"
        ],
        "default_eval": "reclor",
        "default_answer_type": AnswerType.MCQ,
        "category": "logical_reasoning"
    },
    "commonsense": {
        "training_datasets": ["arc_c"],
        "eval_datasets": ["arc_c", "strategyqa", "openbookqa"],
        "default_eval": "arc_c",
        "default_answer_type": AnswerType.MCQ,
        "category": "commonsense_reasoning"
    }
}
```

#### 2.3 `config/domains.py` 확장

**위치**: Line 11-14 (TERMINAL_GOALS), Line 17-20 (DATASET_TO_DOMAIN), Line 23-25 (TRAINING_DATASETS), Line 31-38 (DOMAIN_CONFIG)

```python
# Terminal Goals for each training dataset
TERMINAL_GOALS = {
    # Math
    "gsm8k": "Generate coherent, step-by-step mathematical reasoning in natural language that leads to a correct numerical answer for grade-school level math problems.",
    "math": "Solve advanced mathematical problems by selecting appropriate mathematical concepts and constructing logically valid, multi-step reasoning that leads to a correct solution.",

    # Logical
    "reclor": "Analyze logical reasoning problems by comprehending complex passages, identifying logical relationships, and selecting the most appropriate conclusion based on formal reasoning principles.",

    # Commonsense
    "arc_c": "Apply commonsense scientific knowledge to solve elementary science problems by understanding fundamental concepts and selecting the correct answer from multiple choices.",
}

# Dataset to domain mapping
DATASET_TO_DOMAIN = {
    "gsm8k": "math",
    "math": "math",
    "reclor": "logical",
    "arc_c": "commonsense",
}

# Available training datasets per domain
TRAINING_DATASETS = {
    "math": ["gsm8k", "math"],
    "logical": ["reclor"],
    "commonsense": ["arc_c"],
}

# Domain configurations
DOMAIN_CONFIG = {
    "math": {
        # ... 기존 설정 ...
    },
    "logical": {
        "data_dir": DATA_DIR / "logical",
        "training_datasets": ["reclor"],
        "eval_datasets": [
            "reclor", "anli_r2", "anli_r3",
            "bbh_boolean_expressions", "bbh_formal_fallacies",
            "bbh_logical_deduction_three_objects", "bbh_logical_deduction_five_objects",
            "bbh_logical_deduction_seven_objects",
            "bbh_tracking_shuffled_objects_three_objects",
            "bbh_tracking_shuffled_objects_five_objects",
            "bbh_tracking_shuffled_objects_seven_objects",
            "bbh_web_of_lies"
        ],
        "default_eval": "reclor"
    },
    "commonsense": {
        "data_dir": DATA_DIR / "commonsense",
        "training_datasets": ["arc_c"],
        "eval_datasets": ["arc_c", "strategyqa", "openbookqa"],
        "default_eval": "arc_c"
    }
}
```

#### 2.4 `config/config.py` 확장

**위치**: 적절한 위치에 도메인 설정 추가 (기존 파일 구조에 따라 조정)

기존 `config.py`는 주로 모델 설정에 집중되어 있으므로, 도메인 관련 설정은 `config/domains.py`를 import하여 사용하도록 유지합니다.

#### 2.5 `config/dataset_config.py` 확장

**위치**: Line 10-77 (DATASET_CONFIGS)

```python
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ... 기존 math 도메인 데이터셋 ...

    # Logical domain
    "reclor": {
        "hf_name": "community-datasets/reclor",
        "hf_config": None,
        "train_split": "train",
        "test_split": "test",
        "answer_type": "mcq",
        "domain": "logical",
        "description": "ReClor - Logical reasoning from standardized tests",
    },
    "anli_r2": {
        "hf_name": "facebook/anli",
        "hf_config": None,
        "train_split": "train_r2",
        "test_split": "test_r2",
        "answer_type": "mcq",
        "domain": "logical",
        "description": "ANLI Round 2 - Adversarial Natural Language Inference",
    },
    "anli_r3": {
        "hf_name": "facebook/anli",
        "hf_config": None,
        "train_split": "train_r3",
        "test_split": "test_r3",
        "answer_type": "mcq",
        "domain": "logical",
        "description": "ANLI Round 3 - Adversarial Natural Language Inference",
    },
    # BBH subtasks - logical reasoning related
    "bbh_boolean_expressions": {
        "hf_name": "lukaemon/bbh",
        "hf_config": "boolean_expressions",
        "train_split": None,
        "test_split": "test",
        "answer_type": "boolean",
        "domain": "logical",
        "description": "BBH - Boolean expression evaluation",
    },
    "bbh_formal_fallacies": {
        "hf_name": "lukaemon/bbh",
        "hf_config": "formal_fallacies",
        "train_split": None,
        "test_split": "test",
        "answer_type": "text",
        "domain": "logical",
        "description": "BBH - Formal logic fallacy detection",
    },
    "bbh_logical_deduction_three_objects": {
        "hf_name": "lukaemon/bbh",
        "hf_config": "logical_deduction_three_objects",
        "train_split": None,
        "test_split": "test",
        "answer_type": "mcq",
        "domain": "logical",
        "description": "BBH - Logical deduction with 3 objects",
    },
    "bbh_logical_deduction_five_objects": {
        "hf_name": "lukaemon/bbh",
        "hf_config": "logical_deduction_five_objects",
        "train_split": None,
        "test_split": "test",
        "answer_type": "mcq",
        "domain": "logical",
        "description": "BBH - Logical deduction with 5 objects",
    },
    "bbh_logical_deduction_seven_objects": {
        "hf_name": "lukaemon/bbh",
        "hf_config": "logical_deduction_seven_objects",
        "train_split": None,
        "test_split": "test",
        "answer_type": "mcq",
        "domain": "logical",
        "description": "BBH - Logical deduction with 7 objects",
    },
    "bbh_tracking_shuffled_objects_three_objects": {
        "hf_name": "lukaemon/bbh",
        "hf_config": "tracking_shuffled_objects_three_objects",
        "train_split": None,
        "test_split": "test",
        "answer_type": "mcq",
        "domain": "logical",
        "description": "BBH - Object tracking with 3 objects",
    },
    "bbh_tracking_shuffled_objects_five_objects": {
        "hf_name": "lukaemon/bbh",
        "hf_config": "tracking_shuffled_objects_five_objects",
        "train_split": None,
        "test_split": "test",
        "answer_type": "mcq",
        "domain": "logical",
        "description": "BBH - Object tracking with 5 objects",
    },
    "bbh_tracking_shuffled_objects_seven_objects": {
        "hf_name": "lukaemon/bbh",
        "hf_config": "tracking_shuffled_objects_seven_objects",
        "train_split": None,
        "test_split": "test",
        "answer_type": "mcq",
        "domain": "logical",
        "description": "BBH - Object tracking with 7 objects",
    },
    "bbh_web_of_lies": {
        "hf_name": "lukaemon/bbh",
        "hf_config": "web_of_lies",
        "train_split": None,
        "test_split": "test",
        "answer_type": "boolean",
        "domain": "logical",
        "description": "BBH - Truth/lie reasoning",
    },

    # Commonsense domain
    "arc_c": {
        "hf_name": "allenai/ai2_arc",
        "hf_config": "ARC-Challenge",
        "train_split": "train",
        "test_split": "test",
        "answer_type": "mcq",
        "domain": "commonsense",
        "description": "AI2 ARC Challenge - Elementary science questions",
    },
    "strategyqa": {
        "hf_name": "wics/strategy-qa",
        "hf_config": None,
        "train_split": None,  # StrategyQA는 train split 없음
        "test_split": "test",
        "answer_type": "boolean",
        "domain": "commonsense",
        "description": "StrategyQA - Multi-hop commonsense yes/no questions",
    },
    "openbookqa": {
        "hf_name": "allenai/openbookqa",
        "hf_config": "main",
        "train_split": "train",
        "test_split": "test",
        "answer_type": "mcq",
        "domain": "commonsense",
        "description": "OpenBookQA - Elementary science with open book",
    },
}
```

### Step 3: 문서 업데이트

#### 3.1 README.md 업데이트

**파일**: [README.md](README.md)

##### 수정 1: "지원 도메인 및 Terminal Goal" 섹션 확장 (Line 13-25)

**기존 내용을 다음으로 교체**:

```markdown
## 지원 도메인 및 Terminal Goal

| 도메인 | 학습 데이터셋 | Terminal Goal |
|--------|--------------|---------------|
| **Math** | GSM8K | Generate coherent, step-by-step mathematical reasoning in natural language that leads to a correct numerical answer for grade-school level math problems. |
| **Math** | MATH | Solve advanced mathematical problems by selecting appropriate mathematical concepts and constructing logically valid, multi-step reasoning that leads to a correct solution. |
| **Logical** | ReClor | Analyze logical reasoning problems by comprehending complex passages, identifying logical relationships, and selecting the most appropriate conclusion based on formal reasoning principles. |
| **Commonsense** | ARC-Challenge | Apply commonsense scientific knowledge to solve elementary science problems by understanding fundamental concepts and selecting the correct answer from multiple choices. |

### 평가 데이터셋

| 도메인 | In-Domain 평가 | OOD 평가 |
|--------|---------------|----------|
| **Math** | GSM8K, MATH | SVAMP, ASDiv, MAWPS, MMLU |
| **Logical** | ReClor | ANLI-R2, ANLI-R3, BBH (논리 추론 9개 태스크) |
| **Commonsense** | ARC-Challenge | StrategyQA, OpenBookQA |
```

##### 수정 2: "데이터 구조" 섹션 확장 (Line 310-344)

**`data/` 디렉토리 구조**를 다음으로 교체:

```markdown
## 데이터 구조

```
data/
├── math/                                           # Math 도메인
│   ├── train/                                      # 학습 데이터
│   │   ├── data/                                   # 원본 학습 데이터
│   │   │   ├── gsm8k_train.json                   # GSM8K 학습 데이터
│   │   │   └── math_train.json                    # MATH 학습 데이터
│   │   │
│   │   └── {Teacher-Model}/                        # Teacher 모델별
│   │       ├── instructional-design/               # 설계 결과
│   │       └── {Student-Model}/                    # Student 모델별 SFT 데이터
│   │
│   └── eval/                                       # 평가 데이터
│       ├── data/                                   # 원본 평가 데이터
│       │   ├── gsm8k_test.json
│       │   ├── math_test.json
│       │   ├── svamp_test.json
│       │   ├── asdiv_test.json
│       │   ├── mawps_test.json
│       │   └── mmlu_test.json
│       └── {Model}/                                # 모델별 평가 결과
│
├── logical/                                        # Logical 도메인 (NEW)
│   ├── train/
│   │   ├── data/
│   │   │   └── reclor_train.json
│   │   └── {Teacher-Model}/
│   │       ├── instructional-design/
│   │       └── {Student-Model}/
│   └── eval/
│       ├── data/
│       │   ├── reclor_test.json                   # In-Domain
│       │   ├── anli_r2_test.json                  # OOD
│       │   ├── anli_r3_test.json                  # OOD
│       │   ├── bbh_boolean_expressions_test.json  # OOD
│       │   ├── bbh_formal_fallacies_test.json     # OOD
│       │   ├── bbh_logical_deduction_three_objects_test.json
│       │   ├── bbh_logical_deduction_five_objects_test.json
│       │   ├── bbh_logical_deduction_seven_objects_test.json
│       │   ├── bbh_tracking_shuffled_objects_three_objects_test.json
│       │   ├── bbh_tracking_shuffled_objects_five_objects_test.json
│       │   ├── bbh_tracking_shuffled_objects_seven_objects_test.json
│       │   └── bbh_web_of_lies_test.json
│       └── {Model}/
│
└── commonsense/                                    # Commonsense 도메인 (NEW)
    ├── train/
    │   ├── data/
    │   │   └── arc_c_train.json
    │   └── {Teacher-Model}/
    │       ├── instructional-design/
    │       └── {Student-Model}/
    └── eval/
        ├── data/
        │   ├── arc_c_test.json                    # In-Domain
        │   ├── strategyqa_test.json               # OOD
        │   └── openbookqa_test.json               # OOD
        └── {Model}/
```
```

##### 수정 3: "데이터 준비" 섹션 업데이트 (Line 413-423)

**기존 내용**:
```markdown
이 스크립트는 다음 데이터셋을 다운로드합니다:
- **Math 도메인**: GSM8K, MATH, SVAMP, ASDiv, MAWPS, MMLU (수학 과목)
```

**수정 후**:
```markdown
이 스크립트는 다음 데이터셋을 다운로드합니다:
- **Math 도메인**: GSM8K, MATH, SVAMP, ASDiv, MAWPS, MMLU (수학 과목)
- **Logical 도메인**: ReClor, ANLI-R2, ANLI-R3, BBH (논리 추론 9개 태스크)
- **Commonsense 도메인**: ARC-Challenge, StrategyQA, OpenBookQA
```

##### 수정 4: "실행 예제" 섹션에 신규 도메인 예제 추가 (Line 159 이후)

**"학습 모드 (--mode train)" 섹션에 추가**:

```markdown
# Logical 도메인 - ReClor로 학습
python main.py --mode train --domain logical --train-dataset reclor

# Commonsense 도메인 - ARC-Challenge로 학습
python main.py --mode train --domain commonsense --train-dataset arc_c
```

**"평가 모드 (--mode eval)" → "Baseline 평가" 섹션에 추가**:

```markdown
# Logical 도메인 - ReClor Baseline 평가
python main.py --mode eval --method baseline \
    --domain logical --eval-dataset reclor

# Logical 도메인 - ANLI-R2 OOD 평가
python main.py --mode eval --method baseline \
    --domain logical --eval-dataset anli_r2

# Commonsense 도메인 - StrategyQA OOD 평가
python main.py --mode eval --method baseline \
    --domain commonsense --eval-dataset strategyqa
```

##### 수정 5: CLI 옵션 테이블 업데이트 (Line 278)

**"학습 모드 전용 옵션" 테이블의 `--domain` 행 수정**:

**기존**:
```
| `--domain` | 도메인: `math` | (필수) |
```

**수정 후**:
```
| `--domain` | 도메인: `math`, `logical`, `commonsense` | (필수) |
```

**`--train-dataset` 행 수정**:

**기존**:
```
| `--train-dataset` | 학습 데이터셋: `gsm8k`, `math` | (필수) |
```

**수정 후**:
```
| `--train-dataset` | 학습 데이터셋: `gsm8k`, `math`, `reclor`, `arc_c` | (필수) |
```

### Step 4: Git Commit 및 Push

구현이 완료되고 모든 검증이 통과한 후, 변경사항을 커밋하고 원격 저장소에 푸시합니다.

#### 4.1 변경사항 확인

```bash
# 변경된 파일 확인
git status

# 예상 변경 파일:
# - utils/dataset_preparer.py
# - utils/domain_loader.py
# - utils/dataset_registry.py
# - config/domains.py
# - config/dataset_config.py
# - README.md
# - data/logical/... (신규 생성된 데이터 파일들)
# - data/commonsense/... (신규 생성된 데이터 파일들)
```

#### 4.2 데이터 파일 제외 확인

`.gitignore`에 데이터 파일이 제외되어 있는지 확인:

```bash
# .gitignore 확인
cat .gitignore | grep data

# 일반적으로 다음과 같이 설정되어 있어야 함:
# data/*/train/data/*.json
# data/*/eval/data/*.json
```

**중요**: 데이터 파일은 용량이 크므로 git에 커밋하지 않습니다. 사용자는 `python -m utils.dataset_preparer`로 직접 생성합니다.

#### 4.3 Skill 사용: `/git:git-push`

`/git:git-push` 스킬을 사용하여 변경사항을 커밋하고 푸시:

```bash
/git:git-push
```

**또는 수동으로**:

```bash
# 변경된 파일 스테이징 (데이터 파일 제외)
git add utils/dataset_preparer.py
git add utils/domain_loader.py
git add utils/dataset_registry.py
git add config/domains.py
git add config/dataset_config.py
git add README.md
git add .claude/plans/PLAN_2026-01-10_datasets-expansion-detailed.md

# 커밋 메시지 작성
git commit -m "feat: Add logical and commonsense domains with new datasets

- Add logical domain with ReClor (train), ANLI-R2/R3, BBH logical tasks (eval)
- Add commonsense domain with ARC-Challenge (train), StrategyQA, OpenBookQA (eval)
- Extend dataset_preparer.py with 6 new processing functions
- Update domain configurations across all config files
- Update README.md with new domain information and usage examples

Datasets added:
- Logical: ReClor, ANLI-R2, ANLI-R3, BBH (9 logical reasoning subtasks)
- Commonsense: ARC-Challenge, StrategyQA, OpenBookQA

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# 원격 저장소에 푸시
git push origin main
```

#### 4.4 푸시 확인

```bash
# 원격 저장소 상태 확인
git log --oneline -3

# GitHub/GitLab에서 커밋 확인
# - 변경된 파일 목록 확인
# - 커밋 메시지 확인
# - 코드 diff 확인
```

## 검증 방법

### Step 1: 데이터셋 준비 실행

```bash
# 전체 데이터셋 준비
python utils/dataset_preparer.py

# 예상 출력:
# - data/logical/train/data/reclor_train.json
# - data/logical/eval/data/reclor_test.json
# - data/logical/eval/data/anli_r2_test.json
# - data/logical/eval/data/anli_r3_test.json
# - data/logical/eval/data/bbh_*.json (9개 파일)
# - data/commonsense/train/data/arc_c_train.json
# - data/commonsense/eval/data/arc_c_test.json
# - data/commonsense/eval/data/strategyqa_test.json
# - data/commonsense/eval/data/openbookqa_test.json
```

**검증 포인트**:
- [ ] 모든 JSON 파일이 생성되었는가?
- [ ] 각 파일의 레코드 수가 0보다 큰가?
- [ ] JSON 파일이 올바른 스키마 (`instruction`, `input`, `output`)를 따르는가?

### Step 2: DomainLoader 테스트

```bash
# DomainLoader 직접 테스트
python utils/domain_loader.py

# 또는 Python REPL에서:
python
>>> from utils.domain_loader import DomainLoader
>>> loader = DomainLoader("logical")
>>> loader.get_available_training_datasets()
['reclor']
>>> loader.get_available_eval_datasets()
['reclor', 'anli_r2', 'anli_r3', 'bbh_boolean_expressions', ...]
>>> train_data = loader.load_training_data("reclor", limit=5)
>>> print(len(train_data))
5
>>> print(train_data[0].question[:100])
>>> print(train_data[0].ground_truth)
```

**검증 포인트**:
- [ ] 도메인 로더가 정상적으로 초기화되는가?
- [ ] 학습 데이터셋 목록이 올바른가?
- [ ] 평가 데이터셋 목록이 올바른가?
- [ ] 샘플 데이터가 올바른 형식으로 로딩되는가?
- [ ] `ground_truth`가 올바른 형식(A/B/C/D, Yes/No 등)인가?
- [ ] `answer_type`이 올바르게 설정되어 있는가?

### Step 3: 학습 및 평가 파이프라인 테스트

```bash
# 1. Logical 도메인 학습 (dry-run)
python main.py --mode train --domain logical --dataset reclor --limit 10 --student_model Qwen/Qwen2.5-3B-Instruct

# 2. Logical 도메인 평가 (In-Domain)
python main.py --mode eval --domain logical --eval_dataset reclor --limit 10

# 3. Logical 도메인 평가 (OOD)
python main.py --mode eval --domain logical --eval_dataset anli_r2 --limit 10
python main.py --mode eval --domain logical --eval_dataset bbh_boolean_expressions --limit 10

# 4. Commonsense 도메인 학습
python main.py --mode train --domain commonsense --dataset arc_c --limit 10 --student_model Qwen/Qwen2.5-3B-Instruct

# 5. Commonsense 도메인 평가 (OOD)
python main.py --mode eval --domain commonsense --eval_dataset strategyqa --limit 10
python main.py --mode eval --domain commonsense --eval_dataset openbookqa --limit 10
```

**검증 포인트**:
- [ ] 학습 모드가 정상적으로 실행되는가?
- [ ] 평가 모드가 정상적으로 실행되는가?
- [ ] Terminal Goal이 올바르게 표시되는가?
- [ ] 답안 추출이 올바르게 동작하는가? (MCQ: A-D, BOOLEAN: Yes/No)
- [ ] 평가 결과가 합리적인가?

### Step 4: 통합 테스트 케이스

**테스트 시나리오**:

1. **ReClor MCQ 답안 추출 테스트**
   ```python
   from utils.answer_extractor import extract_boxed_answer

   test_output = "The passage suggests that... Therefore, the answer is \\boxed{B}"
   answer = extract_boxed_answer(test_output)
   assert answer == "B", f"Expected 'B', got '{answer}'"
   ```

2. **StrategyQA BOOLEAN 답안 추출 테스트**
   ```python
   test_output = "Based on common knowledge, the answer is \\boxed{Yes}"
   answer = extract_boxed_answer(test_output)
   assert answer == "Yes", f"Expected 'Yes', got '{answer}'"
   ```

3. **ANLI 3-way 분류 답안 추출 테스트**
   ```python
   test_output = "The hypothesis contradicts the premise, so \\boxed{C}"
   answer = extract_boxed_answer(test_output)
   assert answer == "C", f"Expected 'C', got '{answer}'"
   ```

4. **도메인 간 전환 테스트**
   ```bash
   # Math -> Logical -> Commonsense -> Math 순서로 전환하여 설정이 올바르게 로드되는지 확인
   python main.py --mode eval --domain math --eval_dataset gsm8k --limit 5
   python main.py --mode eval --domain logical --eval_dataset reclor --limit 5
   python main.py --mode eval --domain commonsense --eval_dataset arc_c --limit 5
   python main.py --mode eval --domain math --eval_dataset math --limit 5
   ```

### Step 5: 데이터 품질 검증

**스크립트**: `scripts/validate_datasets.py` (신규 생성)

```python
#!/usr/bin/env python3
"""
Validate prepared datasets for quality and consistency.
"""
import json
import re
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path("data")

def validate_json_schema(data: List[Dict]) -> List[str]:
    """Validate JSON schema."""
    errors = []
    required_fields = ["instruction", "input", "output"]

    for i, item in enumerate(data):
        for field in required_fields:
            if field not in item:
                errors.append(f"Record {i}: Missing field '{field}'")

        # Validate output format
        output = item.get("output", "")
        if not re.search(r'\\boxed\{.+?\}', output):
            errors.append(f"Record {i}: Output missing \\boxed{{}} format")

    return errors

def validate_domain(domain_name: str):
    """Validate all datasets in a domain."""
    print(f"\n{'='*60}")
    print(f"Validating {domain_name.upper()} Domain")
    print(f"{'='*60}")

    domain_dir = DATA_DIR / domain_name
    if not domain_dir.exists():
        print(f"  ❌ Domain directory not found: {domain_dir}")
        return

    for split_dir in ["train/data", "eval/data"]:
        split_path = domain_dir / split_dir
        if not split_path.exists():
            continue

        print(f"\n[{split_dir}]")
        for json_file in sorted(split_path.glob("*.json")):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            errors = validate_json_schema(data)

            if errors:
                print(f"  ❌ {json_file.name}: {len(data)} records, {len(errors)} errors")
                for error in errors[:3]:  # Show first 3 errors
                    print(f"      - {error}")
            else:
                print(f"  ✓ {json_file.name}: {len(data)} records, OK")

if __name__ == "__main__":
    validate_domain("math")
    validate_domain("logical")
    validate_domain("commonsense")

    print(f"\n{'='*60}")
    print("Validation completed!")
    print(f"{'='*60}")
```

실행:
```bash
python scripts/validate_datasets.py
```

## 참고 자료

### HuggingFace 데이터셋 링크

- [ReClor Dataset](https://huggingface.co/datasets/community-datasets/reclor)
- [ARC Dataset](https://huggingface.co/datasets/allenai/ai2_arc)
- [StrategyQA Dataset](https://huggingface.co/datasets/wics/strategy-qa)
- [OpenBookQA Dataset](https://huggingface.co/datasets/allenai/openbookqa)
- [ANLI Dataset](https://huggingface.co/datasets/facebook/anli)
- [BBH Dataset](https://huggingface.co/datasets/lukaemon/bbh)

### 구현 순서 요약

1. **Step 1**: `utils/dataset_preparer.py` 확장 (프롬프트, 처리 함수, main)
2. **Step 2.1**: `utils/domain_loader.py` 확장 (TERMINAL_GOALS, DOMAIN_CONFIG)
3. **Step 2.2**: `utils/dataset_registry.py` 확장 (DOMAIN_CONFIG)
4. **Step 2.3**: `config/domains.py` 확장 (전체 도메인 설정)
5. **Step 2.5**: `config/dataset_config.py` 확장 (HF 데이터셋 정보)
6. **Step 3**: 문서 업데이트 (README.md, USAGE.md)
7. **Step 4**: Git Commit 및 Push (`/git:git-push` 스킬 사용)
8. **검증**: 데이터셋 준비 → 로더 테스트 → 파이프라인 테스트 → 품질 검증

## 마일스톤

- [ ] Step 1: 데이터 준비 스크립트 확장 완료
- [ ] Step 2: 설정 파일 동기화 완료
- [ ] Step 3: 문서 업데이트 완료
- [ ] 검증 1: 데이터셋 준비 성공
- [ ] 검증 2: DomainLoader 테스트 통과
- [ ] 검증 3: 학습/평가 파이프라인 테스트 통과
- [ ] 검증 4: 통합 테스트 통과
- [ ] 검증 5: 데이터 품질 검증 통과
- [ ] Step 4: Git Commit 및 Push 완료
