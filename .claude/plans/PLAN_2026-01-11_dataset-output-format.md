# Dataset Output Format 수정 계획

**작성일**: 2026-01-11
**Domain**: coding
**대상 파일**: `utils/dataset_preparer.py`

---

## 1. 요구사항 분석

### 1.1 전체 데이터셋 output 형식 통일
- **현재**: 다양한 형식으로 저장 (일부는 reasoning 포함, 일부는 answer만)
- **목표**: 모든 데이터셋을 "The answer is \\boxed{answer}." 형식으로 통일

### 1.2 Math Domain 특별 처리 (gsm8k, math)
- **현재 형식**: `"{reasoning}\n\n\\boxed{answer}"`
- **목표**: 두 가지 버전 저장
  1. **짧은 버전** (`{dataset_name}_{train/test}.json`):
     `"The answer is \\boxed{answer}."`
  2. **추론 포함 버전** (`{dataset_name}_reasoning_{train/test}.json`):
     `"{reasoning}\n\nThe answer is \\boxed{answer}"`

### 1.3 영향 받는 데이터셋

#### Math Domain (두 버전 저장)
- `gsm8k` (train + test)
- `math` (train + test)

#### Math Domain (짧은 버전만)
- `svamp` (test only) - 이미 answer만 저장
- `asdiv` (test only) - 이미 answer만 저장
- `mawps` (test only) - 이미 answer만 저장

#### Logical Domain (짧은 버전만)
- `reclor` (train + val + test)
- `anli` (r2, r3 test)
- `bbh` (logical subtasks test)

#### Commonsense Domain (짧은 버전만)
- `arc_c` (train + test)
- `strategyqa` (test)
- `openbookqa` (test)

---

## 2. 현재 코드 분석

### 2.1 핵심 함수

#### `format_output(reasoning: Optional[str], answer: str) -> str` (line 150-166)
```python
def format_output(reasoning: Optional[str], answer: str) -> str:
    boxed_answer = f"\\boxed{{{answer}}}"

    if reasoning and reasoning.strip():
        return f"{reasoning.strip()}\n\n{boxed_answer}"
    return boxed_answer
```

**문제점**:
- "The answer is" 접두사 없음
- 두 가지 버전 생성 불가

#### `save_json(data: List[Dict], output_path: Path)` (line 186-191)
```python
def save_json(data: List[Dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
```

**문제점**:
- 단일 파일만 저장
- 두 가지 버전 저장 로직 없음

### 2.2 데이터셋 처리 함수들

#### Reasoning 포함 데이터셋
- `process_gsm8k()` (line 223-259): reasoning 추출 (`#### ` 구분자)
- `process_math()` (line 262-321): full solution 포함

#### Answer만 있는 데이터셋
- `process_svamp()` (line 324-354): `format_output(None, answer)`
- `process_asdiv()` (line 357-401): `format_output(None, answer_clean)`
- `process_mawps()` (line 404-440): `format_output(None, str(answer))`
- `process_reclor()` (line 443-513): `format_output(None, answer_letter)`
- `process_anli()` (line 629-673): `format_output(None, answer_letter)`
- `process_bbh()` (line 676-716): `format_output(None, target)`
- `process_arc_c()` (line 516-558): `format_output(None, answer_letter)`
- `process_strategyqa()` (line 561-589): `format_output(None, answer_text)`
- `process_openbookqa()` (line 592-626): `format_output(None, answer_key)`

---

## 3. 구현 계획

### 3.1 Phase 1: format_output() 함수 수정

#### 새로운 함수 시그니처
```python
def format_output(
    answer: str,
    reasoning: Optional[str] = None,
    include_reasoning: bool = False
) -> str:
    """
    Format output with "The answer is \\boxed{answer}." format.

    Args:
        answer: Final answer
        reasoning: Step-by-step reasoning (optional)
        include_reasoning: If True, include reasoning before answer

    Returns:
        Formatted output string
    """
```

#### 구현 로직
1. `boxed_answer = f"\\boxed{{{answer}}}"`
2. `final_answer = f"The answer is {boxed_answer}"`
3. `include_reasoning=True`이고 reasoning 존재 시:
   → `f"{reasoning.strip()}\n\nThe answer is {boxed_answer}"`
4. 그 외: `f"The answer is {boxed_answer}"`

### 3.2 Phase 2: save_json() 함수 확장

#### 옵션 1: 기존 함수 유지 + 새 함수 추가
```python
def save_dual_versions(
    records_with_reasoning: List[Dict],
    records_without_reasoning: List[Dict],
    base_output_path: Path,
    dataset_name: str,
    split: str
):
    """
    Save both reasoning and non-reasoning versions.

    Files:
        - {dataset_name}_{split}.json (without reasoning)
        - {dataset_name}_reasoning_{split}.json (with reasoning)
    """
```

#### 옵션 2: save_json()에 파라미터 추가 (더 간단)
```python
def save_json(
    data: List[Dict],
    output_path: Path,
    reasoning_data: Optional[List[Dict]] = None
):
    """
    Save data to JSON file.
    If reasoning_data is provided, save it as {name}_reasoning.json
    """
```

**선택**: 옵션 2 (코드 변경 최소화)

### 3.3 Phase 3: 데이터셋 처리 함수 수정

#### 3.3.1 gsm8k 수정 (`process_gsm8k()`)
```python
def process_gsm8k(train_dir: Path, eval_dir: Path):
    # ...
    for split in ["train", "test"]:
        # ...
        records_short = []  # without reasoning
        records_full = []   # with reasoning

        for item in data:
            question = item["question"]
            answer_text = item["answer"]

            if "####" in answer_text:
                reasoning = parts[0].strip()
                final_answer = parts[1].strip()
            else:
                reasoning = ""
                final_answer = answer_text

            # Short version
            records_short.append({
                "instruction": DATASET_PROMPTS["gsm8k"],
                "input": question,
                "output": format_output(final_answer, reasoning=None)
            })

            # Full version
            records_full.append({
                "instruction": DATASET_PROMPTS["gsm8k"],
                "input": question,
                "output": format_output(
                    final_answer,
                    reasoning=reasoning,
                    include_reasoning=True
                )
            })

        output_base = train_dir if split == "train" else eval_dir
        save_json(records_short, output_base / f"gsm8k_{split}.json")
        save_json(records_full, output_base / f"gsm8k_reasoning_{split}.json")
```

#### 3.3.2 math 수정 (`process_math()`)
- gsm8k와 동일한 패턴
- `reasoning = item["solution"]`
- `boxed_answer = extract_boxed_answer(solution)`

#### 3.3.3 다른 데이터셋들 수정
모든 `format_output()` 호출을:
```python
# Before
format_output(None, answer)

# After
format_output(answer)  # reasoning=None이 기본값
```

### 3.4 Phase 4: 함수 호출 순서 변경

#### 현재 format_output() 호출
```python
format_output(reasoning, answer)
```

#### 새로운 시그니처
```python
format_output(answer, reasoning=None, include_reasoning=False)
```

**호환성 이슈**: 모든 호출을 수정해야 함

---

## 4. 단계별 실행 계획

### Step 1: format_output() 함수 재작성
- **파일**: [utils/dataset_preparer.py:150-166](utils/dataset_preparer.py#L150-L166)
- **작업**:
  1. 함수 시그니처 변경: `format_output(answer, reasoning=None, include_reasoning=False)`
  2. "The answer is" 접두사 추가
  3. `include_reasoning` 파라미터에 따른 분기 처리
- **검증**: 함수 독립 테스트

### Step 2: 모든 format_output() 호출 수정
- **대상 함수들**:
  - `process_gsm8k()` (line 255)
  - `process_math()` (line 314)
  - `process_svamp()` (line 351)
  - `process_asdiv()` (line 395)
  - `process_mawps()` (line 434)
  - `process_reclor()` (line 501)
  - `process_arc_c()` (line 554)
  - `process_strategyqa()` (line 586)
  - `process_openbookqa()` (line 623)
  - `process_anli()` (line 670)
  - `process_bbh()` (line 708)
- **작업**: `format_output(reasoning, answer)` → `format_output(answer, reasoning=...)`

### Step 3: gsm8k와 math 처리 함수 수정
- **파일**: [utils/dataset_preparer.py:223-321](utils/dataset_preparer.py#L223-L321)
- **작업**:
  1. 두 개의 records 리스트 생성 (short, full)
  2. 각 데이터 포인트마다 두 가지 버전 생성
  3. 두 파일로 저장 (base name, base name + `_reasoning`)
- **검증**:
  - 파일 개수 확인 (기존 2개 → 4개)
  - 내용 확인 (short vs full)

### Step 4: 전체 스크립트 실행 및 검증
- **실행**: `python utils/dataset_preparer.py`
- **검증 항목**:
  1. 모든 데이터셋 파일이 정상 생성되는지
  2. Math domain: `{name}_{split}.json`과 `{name}_reasoning_{split}.json` 모두 생성
  3. 다른 domain: `{name}_{split}.json`만 생성
  4. Output 형식: `"The answer is \\boxed{...}"`로 통일
  5. Reasoning 버전: `"{reasoning}\n\nThe answer is \\boxed{...}"`

---

## 5. 위험 요소 및 대응

### 5.1 호환성 이슈
- **위험**: format_output() 시그니처 변경으로 인한 기존 코드 오류
- **대응**:
  - 모든 호출부를 한 번에 수정
  - 또는 기존 함수 유지하고 새 함수 추가 (`format_output_v2()`)

### 5.2 데이터 손실
- **위험**: 스크립트 실행 중 기존 파일 덮어쓰기
- **대응**:
  - 실행 전 `data/` 디렉토리 백업
  - 또는 다른 출력 경로 사용 후 확인

### 5.3 Reasoning 추출 실패
- **위험**: gsm8k, math에서 reasoning이 비어있는 경우
- **대응**:
  - Reasoning이 없으면 빈 문자열로 처리
  - 로그 출력으로 확인

---

## 6. 테스트 계획

### 6.1 Unit Test
- `format_output()` 함수:
  - `format_output("42")` → `"The answer is \\boxed{42}"`
  - `format_output("42", reasoning="Step 1...")` → `"Step 1...\n\nThe answer is \\boxed{42}"`
  - `format_output("42", include_reasoning=True)` → `"The answer is \\boxed{42}"` (reasoning 없으면 무시)

### 6.2 Integration Test
- 전체 스크립트 실행 후:
  1. 파일 개수 확인
  2. 샘플 데이터 검증 (첫 5개 레코드)
  3. JSON 파싱 가능 여부
  4. Output 형식 일치 여부

---

## 7. 완료 기준

### 7.1 코드 수정
- [ ] `format_output()` 함수 수정 완료
- [ ] 모든 `format_output()` 호출 수정 완료
- [ ] `process_gsm8k()` 두 버전 저장 구현
- [ ] `process_math()` 두 버전 저장 구현

### 7.2 검증
- [ ] 스크립트 실행 시 에러 없음
- [ ] Math domain: 4개 파일 생성 (gsm8k × 2, math × 2 per split)
- [ ] 모든 output이 "The answer is \\boxed{...}" 형식
- [ ] Reasoning 버전 파일들이 추론 포함

### 7.3 문서화
- [ ] 변경 사항 CHANGELOG 기록 (필요시)
- [ ] 코드 주석 업데이트

---

## 8. 예상 결과

### 8.1 파일 구조

#### Math Domain
```
data/math/train/data/
├── gsm8k_train.json              # "The answer is \\boxed{...}"
├── gsm8k_reasoning_train.json    # "{reasoning}\n\nThe answer is \\boxed{...}"
├── math_train.json               # "The answer is \\boxed{...}"
└── math_reasoning_train.json     # "{reasoning}\n\nThe answer is \\boxed{...}"

data/math/eval/data/
├── gsm8k_test.json
├── gsm8k_reasoning_test.json
├── math_test.json
├── math_reasoning_test.json
├── svamp_test.json               # "The answer is \\boxed{...}"
├── asdiv_test.json               # "The answer is \\boxed{...}"
└── mawps_test.json               # "The answer is \\boxed{...}"
```

#### Other Domains
```
data/logical/eval/data/
├── reclor_*.json                 # "The answer is \\boxed{A}"
├── anli_*.json                   # "The answer is \\boxed{A}"
└── bbh_test.json                 # "The answer is \\boxed{...}"

data/commonsense/eval/data/
├── arc_c_*.json                  # "The answer is \\boxed{A}"
├── strategyqa_test.json          # "The answer is \\boxed{Yes}"
└── openbookqa_test.json          # "The answer is \\boxed{A}"
```

### 8.2 샘플 데이터

#### gsm8k_test.json
```json
[
  {
    "instruction": "You are a helpful math assistant...",
    "input": "Janet's ducks lay 16 eggs per day...",
    "output": "The answer is \\boxed{18}"
  }
]
```

#### gsm8k_reasoning_test.json
```json
[
  {
    "instruction": "You are a helpful math assistant...",
    "input": "Janet's ducks lay 16 eggs per day...",
    "output": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n\nThe answer is \\boxed{18}"
  }
]
```

---

## 9. 다음 단계

이 계획이 승인되면:
1. Step 1부터 순차적으로 구현
2. 각 Step마다 검증
3. 전체 테스트 실행
4. 문서 업데이트
