# Answer Extractor 업데이트 계획

## 개요

**목표**: `data/math/eval/data`의 모든 데이터셋(`gsm8k`, `math`, `svamp`, `asdiv`, `mawps`, `mmlu`)의 `output`에서 정답을 올바르게 추출하도록 코드 수정

## 현재 문제점 분석

### 1. ground_truth 추출 실패

**현재 상태**:
```python
# domain_loader.py:_extract_answer()
ground_truth: \boxed{18}  # 전체가 저장됨
```

**원인**: `_extract_answer`가 `#### <answer>` 패턴만 찾고, `\boxed{...}` 패턴은 마지막 줄 전체를 반환

### 2. answer_type 불일치

| 데이터셋 | 현재 answer_type | 올바른 answer_type | output 예시 |
|----------|------------------|-------------------|-------------|
| gsm8k | NUMERIC | NUMERIC | `\boxed{18}` |
| svamp | NUMERIC | NUMERIC | `\boxed{27}` |
| asdiv | NUMERIC | NUMERIC | `\boxed{9}` |
| mawps | NUMERIC | **LATEX** | `\boxed{56/9}`, `\boxed{320/19}` |
| mmlu | NUMERIC | **MCQ** | `\boxed{B}` |
| math | LATEX | LATEX | `\boxed{\frac{1}{8}}` |

### 3. 데이터셋별 output 형식

```
asdiv:  \boxed{정수}
gsm8k:  풀이과정... \boxed{정수}
math:   풀이과정... \boxed{LaTeX 수식}
mawps:  \boxed{정수 또는 분수}
mmlu:   \boxed{A/B/C/D}
svamp:  \boxed{정수}
```

## 수정 계획

### Step 1: `domain_loader.py` - 데이터셋별 answer_type 명시

**파일**: [utils/domain_loader.py](utils/domain_loader.py)

```python
DOMAIN_CONFIG = {
    "math": {
        "training_datasets": {...},
        "eval_datasets": {
            "gsm8k": {"filename": "gsm8k_test.json", "answer_type": AnswerType.NUMERIC},
            "math": {"filename": "math_test.json", "answer_type": AnswerType.LATEX},
            "svamp": {"filename": "svamp_test.json", "answer_type": AnswerType.NUMERIC},
            "asdiv": {"filename": "asdiv_test.json", "answer_type": AnswerType.NUMERIC},
            "mawps": {"filename": "mawps_test.json", "answer_type": AnswerType.LATEX},  # 분수 포함
            "mmlu": {"filename": "mmlu_test.json", "answer_type": AnswerType.MCQ},
        },
        ...
    }
}
```

### Step 2: `domain_loader.py` - `_extract_answer` 메서드 수정

**현재 로직**:
```python
def _extract_answer(self, output: str, dataset_name: str) -> tuple:
    match = re.search(r'####\s*(.+?)$', output.strip(), re.MULTILINE)
    if match:
        answer = match.group(1).strip()
    else:
        answer = output.strip().split('\n')[-1].strip()  # 마지막 줄
    ...
```

**수정 로직**:
```python
def _extract_answer(self, output: str, dataset_name: str) -> tuple:
    # 1. \boxed{...} 패턴 우선 추출 (중첩 괄호 처리)
    boxed_answer = extract_boxed_answer(output)
    if boxed_answer:
        answer = boxed_answer
    # 2. #### 패턴 시도
    elif match := re.search(r'####\s*(.+?)$', output.strip(), re.MULTILINE):
        answer = match.group(1).strip()
    # 3. 폴백: 마지막 줄
    else:
        answer = output.strip().split('\n')[-1].strip()

    # answer_type은 데이터셋 설정에서 가져옴 (추론 대신)
    answer_type = self._get_dataset_answer_type(dataset_name)
    return answer, answer_type
```

### Step 3: `domain_loader.py` - `_get_dataset_answer_type` 메서드 추가

```python
def _get_dataset_answer_type(self, dataset_name: str) -> AnswerType:
    """데이터셋별로 명시된 answer_type 반환"""
    dataset_name = dataset_name.lower()

    # training_datasets 확인
    if dataset_name in self.config.get("training_datasets", {}):
        ds_config = self.config["training_datasets"][dataset_name]
        if isinstance(ds_config, dict) and "answer_type" in ds_config:
            return ds_config["answer_type"]

    # eval_datasets 확인
    if dataset_name in self.config.get("eval_datasets", {}):
        ds_config = self.config["eval_datasets"][dataset_name]
        if isinstance(ds_config, dict) and "answer_type" in ds_config:
            return ds_config["answer_type"]

    # 폴백: 기존 추론 로직
    return self.config.get("default_answer_type", AnswerType.TEXT)
```

### Step 4: `_load_json_file` 및 관련 메서드 호환성 유지

기존 문자열 기반 설정(`"gsm8k_test.json"`)과 딕셔너리 기반 설정 모두 지원:

```python
def _get_filename(self, ds_config) -> str:
    """데이터셋 설정에서 파일명 추출"""
    if isinstance(ds_config, str):
        return ds_config
    return ds_config.get("filename", "")
```

### Step 5: Training datasets도 동일 구조로 업데이트

```python
"training_datasets": {
    "gsm8k": {"filename": "gsm8k_train.json", "answer_type": AnswerType.NUMERIC},
    "math": {"filename": "math_train.json", "answer_type": AnswerType.LATEX},
}
```

## 변경 파일 목록

| 파일 | 변경 내용 |
|------|----------|
| [utils/domain_loader.py](utils/domain_loader.py) | DOMAIN_CONFIG 구조 변경, `_extract_answer` 수정, 헬퍼 메서드 추가 |

## 검증 방법

```python
# 수정 후 테스트
from utils.domain_loader import DomainLoader

loader = DomainLoader('math')

for dataset in ['gsm8k', 'svamp', 'asdiv', 'mawps', 'mmlu', 'math']:
    data = loader.load_eval_data(dataset, limit=3)
    for q in data:
        print(f'{dataset}: answer_type={q.answer_type}, ground_truth={q.ground_truth}')

# 예상 결과:
# gsm8k: answer_type=NUMERIC, ground_truth=18
# mmlu: answer_type=MCQ, ground_truth=B
# math: answer_type=LATEX, ground_truth=\frac{1}{8}
```

## 영향 범위

### 변경 사항
- `ground_truth`가 `\boxed{}`를 제거한 순수 값으로 저장됨
- 각 데이터셋에 맞는 `answer_type` 적용

### 하위 호환성
- `answer_extractor.py`의 `extract_boxed_answer` 함수 재사용
- 기존 `####` 패턴도 계속 지원
- 기존 문자열 기반 설정도 하위 호환

## 실행 순서

1. `domain_loader.py` DOMAIN_CONFIG 구조 변경
2. `_get_filename` 헬퍼 메서드 추가
3. `_get_dataset_answer_type` 메서드 추가
4. `_extract_answer` 메서드 수정
5. `load_training_data`, `load_eval_data` 메서드 업데이트
6. 테스트 실행
7. 문서 업데이트 (README.md, ARCHITECTURE.md)
