# 데이터셋 수정 계획

**작성일**: 2026-01-10
**목적**: 세 가지 데이터셋 준비 문제 해결 (BBH 통합, StrategyQA 수정, ReClor 로컬 데이터 활용)

## 문제 요약

### 1. BBH 파일 통합 필요
- **현재 상태**: BBH 데이터셋이 9개의 개별 파일로 분산되어 있음
- **목표**: 단일 `bbh_test.json` 파일로 통합

### 2. StrategyQA 로딩 오류
- **현재 상태**: `wics/strategy-qa`에서 "Dataset scripts are no longer supported" 오류 발생
- **원인**: 해당 레포지토리가 deprecated dataset script 형식 사용
- **목표**: 작동하는 대체 레포지토리로 변경

### 3. ReClor 로컬 데이터 활용
- **현재 상태**: `reclor_test.json`의 모든 output이 `\boxed{@}`로 저장되어 있음
- **원인**: HuggingFace 데이터셋의 test split에 레이블이 없음
- **목표**: `.claude/references/data/reclor_data/`의 로컬 JSON 파일 활용하여 재생성

---

## 해결 방안

## Issue 1: BBH 파일 통합

### 현재 상황 분석
[utils/dataset_preparer.py:646-684](utils/dataset_preparer.py#L646-L684)의 `process_bbh()` 함수는 각 subtask를 개별 파일로 저장:

```python
# Line 680
save_json(records, eval_dir / f"bbh_{subtask}_test.json")
```

현재 9개 파일:
- `bbh_boolean_expressions_test.json` (73KB)
- `bbh_formal_fallacies_test.json` (195KB)
- `bbh_logical_deduction_three_objects_test.json` (170KB)
- `bbh_logical_deduction_five_objects_test.json` (209KB)
- `bbh_logical_deduction_seven_objects_test.json` (251KB)
- `bbh_tracking_shuffled_objects_three_objects_test.json` (185KB)
- `bbh_tracking_shuffled_objects_five_objects_test.json` (225KB)
- `bbh_tracking_shuffled_objects_seven_objects_test.json` (269KB)
- `bbh_web_of_lies_test.json` (104KB)

### 수정 계획

#### Step 1: `process_bbh()` 함수 수정
[utils/dataset_preparer.py:646-684](utils/dataset_preparer.py#L646-L684)

**Before:**
```python
def process_bbh(eval_dir: Path, subtasks: List[str]):
    for subtask in subtasks:
        # ... 데이터 로드 및 처리 ...
        save_json(records, eval_dir / f"bbh_{subtask}_test.json")
```

**After:**
```python
def process_bbh(eval_dir: Path, subtasks: List[str]):
    all_records = []  # 모든 subtask의 레코드를 저장할 리스트

    for subtask in subtasks:
        print(f"  Loading subtask: {subtask}...")
        try:
            data = load_dataset(dataset_id, subtask, split="test")
            prompt = BBH_PROMPTS.get(subtask, BBH_PROMPTS["default_mcq"])

            for item in data:
                input_text = item["input"]
                target = item["target"]

                all_records.append({
                    "instruction": prompt,
                    "input": input_text,
                    "output": format_output(None, target)
                })
        except Exception as e:
            print(f"    Error loading {subtask}: {e}")

    # 단일 파일로 저장
    save_json(all_records, eval_dir / "bbh_test.json")
```

#### Step 2: 기존 9개 파일 삭제
수정된 스크립트 실행 후:
```bash
rm data/logical/eval/data/bbh_*_test.json
```

단, `bbh_test.json`은 유지.

#### Step 3: domain_loader.py 업데이트 확인
[utils/domain_loader.py](utils/domain_loader.py)에서 BBH 데이터 로딩 시 개별 파일이 아닌 통합 파일을 참조하도록 확인 필요.

---

## Issue 2: StrategyQA 로딩 오류 해결

### 문제 분석
[utils/dataset_preparer.py:531-559](utils/dataset_preparer.py#L531-L559)

현재 사용 중인 `wics/strategy-qa`는 deprecated dataset script를 사용하여 오류 발생:
```
RuntimeError: Dataset scripts are no longer supported, but found strategy-qa.py
```

### 수정 계획

#### Step 1: 대체 레포지토리로 변경

웹 검색 결과([HuggingFace StrategyQA repositories](https://huggingface.co/datasets/ChilleD/StrategyQA)) 기반 대안:
- **ChilleD/StrategyQA** - train.json과 test.json 파일 보유 (권장)
- amydeng2000/strategy-qa
- njf/StrategyQA

#### Step 2: 코드 수정
[utils/dataset_preparer.py:540](utils/dataset_preparer.py#L540)

**Before:**
```python
dataset_id = "wics/strategy-qa"
```

**After:**
```python
dataset_id = "ChilleD/StrategyQA"
```

#### Step 3: 데이터 형식 검증
대체 레포지토리의 데이터 형식이 동일한지 확인 필요:
- `question` 필드
- `answer` 필드 (boolean)

만약 형식이 다르면 추가 변환 로직 필요.

#### Step 4: config 파일 업데이트
[config/dataset_config.py:190-197](config/dataset_config.py#L190-L197) 수정:

**Before:**
```python
"strategyqa": {
    "hf_name": "wics/strategy-qa",
    # ...
}
```

**After:**
```python
"strategyqa": {
    "hf_name": "ChilleD/StrategyQA",
    # ...
}
```

---

## Issue 3: ReClor 로컬 데이터 활용

### 현재 상황 분석

[utils/dataset_preparer.py:443-483](utils/dataset_preparer.py#L443-L483)의 `process_reclor()` 함수는 HuggingFace에서 데이터를 로드하지만, test split에 레이블이 없어 모든 output이 `\boxed{@}`로 저장됨.

현재 문제:
```python
# Line 454: HuggingFace에서 로드
dataset_id = "sxiong/ReClor"
data = load_dataset(dataset_id, split=split)

# test split의 label이 없어서 output이 @로 저장됨
```

### 로컬 데이터 구조

**경로**: `.claude/references/data/reclor_data/`

**파일 목록**:
- `train.json` (4.7MB)
- `val.json` (533KB)
- `test.json` (1MB)

**JSON 구조** (각 항목):
```json
{
  "context": "지문 텍스트",
  "question": "질문 텍스트",
  "answers": ["선택지 A", "선택지 B", "선택지 C", "선택지 D"],
  "label": 1,  // 정답 인덱스 (0-3)
  "id_string": "train_0"
}
```

### 수정 계획

#### Step 1: 새로운 `process_reclor_local()` 함수 생성

**위치**: [utils/dataset_preparer.py](utils/dataset_preparer.py) 내 `process_reclor()` 함수 교체

**요구 형식** (사용자 지정):
```
Context:
[context text]
Question: [question text]
Options:
A. [선택지 A]
B. [선택지 B]
C. [선택지 C]
D. [선택지 D]
```

**구현 코드**:
```python
def process_reclor_local(train_dir: Path, eval_dir: Path):
    """
    Process ReClor dataset from local JSON files.

    Local files location: .claude/references/data/reclor_data/
    - train.json
    - val.json
    - test.json
    """
    print("\n[ReClor - Local] Processing...")

    # 로컬 데이터 경로
    local_data_dir = Path(__file__).parent.parent / ".claude" / "references" / "data" / "reclor_data"

    # 파일 매핑: split -> (파일명, 출력 디렉토리)
    split_mapping = {
        "train": ("train.json", train_dir),
        "val": ("val.json", eval_dir),
        "test": ("test.json", eval_dir)
    }

    for split_name, (filename, output_dir) in split_mapping.items():
        json_path = local_data_dir / filename

        if not json_path.exists():
            print(f"  Warning: {json_path} not found, skipping {split_name}")
            continue

        print(f"  Loading {split_name} from {filename}...")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        records = []
        for item in data:
            context = item.get("context", "")
            question = item.get("question", "")
            answers = item.get("answers", [])
            label = item.get("label", 0)

            # 사용자 지정 형식으로 input 구성
            input_text = f"Context:\n{context}\nQuestion: {question}\nOptions:\n"
            for i, answer in enumerate(answers):
                input_text += f"{chr(65 + i)}. {answer}\n"

            # 정답 레터 (0-3 -> A-D)
            answer_letter = chr(65 + label)

            records.append({
                "instruction": DATASET_PROMPTS["reclor"],
                "input": input_text.strip(),
                "output": format_output(None, answer_letter)
            })

        # 파일명 결정
        if split_name == "train":
            output_file = "reclor_train.json"
        elif split_name == "val":
            output_file = "reclor_val.json"
        else:
            output_file = "reclor_test.json"

        save_json(records, output_dir / output_file)
        print(f"  Saved {len(records)} records to {output_file}")
```

#### Step 2: 메인 실행 부분 수정

[utils/dataset_preparer.py](utils/dataset_preparer.py)의 메인 실행 부분에서 함수 호출 교체:

**Before:**
```python
process_reclor(train_dir, eval_dir)
```

**After:**
```python
process_reclor_local(train_dir, eval_dir)
```

또는 기존 `process_reclor()` 함수를 완전히 교체.

---

## 구현 순서

### Phase 1: 코드 수정
1. **StrategyQA 수정**
   - [utils/dataset_preparer.py:540](utils/dataset_preparer.py#L540) 수정
   - [config/dataset_config.py:191](config/dataset_config.py#L191) 수정

2. **BBH 통합**
   - [utils/dataset_preparer.py:646-684](utils/dataset_preparer.py#L646-L684) 수정
   - 모든 subtask를 단일 리스트로 수집

3. **ReClor 로컬 데이터 활용**
   - [utils/dataset_preparer.py:443-483](utils/dataset_preparer.py#L443-L483) `process_reclor()` 함수를 `process_reclor_local()`로 교체
   - 로컬 JSON 파일 읽기 및 사용자 지정 형식으로 변환

### Phase 2: 데이터 재생성
```bash
cd /home/seungwoo.baek/project/ID-MAS
python utils/dataset_preparer.py
```

### Phase 3: 검증
1. **StrategyQA 검증**
   ```bash
   # strategyqa_test.json이 생성되었는지 확인
   ls -lh data/commonsense/eval/data/strategyqa_test.json
   # 샘플 데이터 확인
   head -30 data/commonsense/eval/data/strategyqa_test.json
   ```

2. **BBH 검증**
   ```bash
   # 통합 파일 생성 확인
   ls -lh data/logical/eval/data/bbh_test.json
   # 레코드 수 확인 (기존 9개 파일의 합과 동일해야 함)
   python -c "import json; data=json.load(open('data/logical/eval/data/bbh_test.json')); print(f'Total records: {len(data)}')"
   ```

3. **ReClor 검증**
   ```bash
   # reclor_test.json이 재생성되었는지 확인
   ls -lh data/logical/eval/data/reclor_test.json
   # output이 \boxed{A}, \boxed{B} 등으로 올바르게 저장되었는지 확인
   python -c "import json; data=json.load(open('data/logical/eval/data/reclor_test.json')); print('First 3 outputs:', [d['output'] for d in data[:3]])"
   # input 형식 확인 (Context:, Question:, Options: 포함 여부)
   python -c "import json; data=json.load(open('data/logical/eval/data/reclor_test.json')); print(data[0]['input'][:300])"
   ```

### Phase 4: 기존 파일 정리
```bash
# BBH 개별 파일 삭제
rm data/logical/eval/data/bbh_boolean_expressions_test.json
rm data/logical/eval/data/bbh_formal_fallacies_test.json
rm data/logical/eval/data/bbh_logical_deduction_three_objects_test.json
rm data/logical/eval/data/bbh_logical_deduction_five_objects_test.json
rm data/logical/eval/data/bbh_logical_deduction_seven_objects_test.json
rm data/logical/eval/data/bbh_tracking_shuffled_objects_three_objects_test.json
rm data/logical/eval/data/bbh_tracking_shuffled_objects_five_objects_test.json
rm data/logical/eval/data/bbh_tracking_shuffled_objects_seven_objects_test.json
rm data/logical/eval/data/bbh_web_of_lies_test.json
```

---

## 예상 결과

### 파일 구조 변경

**Before:**
```
data/logical/eval/data/
├── reclor_test.json (모든 output이 \boxed{@})
├── bbh_boolean_expressions_test.json
├── bbh_formal_fallacies_test.json
├── ... (7개 더)
└── bbh_web_of_lies_test.json

data/commonsense/eval/data/
└── (strategyqa_test.json 생성 실패)
```

**After:**
```
data/logical/train/data/
└── reclor_train.json (로컬 데이터에서 생성)

data/logical/eval/data/
├── reclor_val.json (로컬 데이터에서 생성)
├── reclor_test.json (로컬 데이터에서 재생성, 올바른 정답 레이블 포함)
└── bbh_test.json (모든 BBH subtask 통합)

data/commonsense/eval/data/
└── strategyqa_test.json (정상 생성)
```

### 데이터 통계

| 데이터셋 | Before | After |
|---------|--------|-------|
| BBH | 9개 파일 | 1개 파일 (동일한 레코드 수) |
| StrategyQA | 생성 실패 | 정상 생성 |
| ReClor | output: `\boxed{@}` (레이블 없음) | 로컬 데이터 활용, 올바른 정답 레이블 포함 (A-D) |

---

## 리스크 및 대응

### Risk 1: ChilleD/StrategyQA 형식 불일치
- **확률**: 중간
- **영향**: StrategyQA 데이터 파싱 실패
- **대응**: 데이터 로드 후 샘플 확인, 필요시 다른 대안 시도 (amydeng2000/strategy-qa)

### Risk 2: BBH 통합 파일 크기
- **확률**: 낮음
- **영향**: 메모리 부족 또는 로딩 느림
- **대응**: 현재 총 1.7MB 정도로 문제 없을 것으로 예상

### Risk 3: ReClor 로컬 파일 경로 오류
- **확률**: 중간
- **영향**: 파일을 찾지 못해 ReClor 데이터 생성 실패
- **대응**: 경로 확인 및 파일 존재 여부 검증, 상대 경로 사용

---

## 참고 자료

### HuggingFace Datasets
- [ChilleD/StrategyQA](https://huggingface.co/datasets/ChilleD/StrategyQA)
- [lukaemon/bbh](https://huggingface.co/datasets/lukaemon/bbh)

### 관련 커밋
- b0755a5: domain_loader에서 \boxed{} 패턴으로 정답 추출
- 16e3a8b: ReClor 경로 수정 및 MMLU 제거

---

## 체크리스트

구현 전:
- [ ] 코드 백업 (git stash 또는 branch 생성)
- [ ] Python 환경 확인 (datasets 라이브러리 설치)
- [ ] 기존 데이터 백업 (선택적)

구현 중:
- [ ] utils/dataset_preparer.py 수정 (StrategyQA, BBH, ReClor)
- [ ] config/dataset_config.py 수정 (StrategyQA)
- [ ] 데이터 재생성 실행

구현 후:
- [ ] StrategyQA 생성 확인
- [ ] BBH 통합 파일 생성 확인
- [ ] ReClor 재생성 확인 (정답 레이블 및 형식)
- [ ] 기존 BBH 개별 파일 삭제
- [ ] Git commit 및 push

---

**계획 작성 완료**
다음 단계: 수정 사항 구현 시작
