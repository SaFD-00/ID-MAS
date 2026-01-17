# Plan: 다중 도메인 지원 오류 수정

## 개요

`--domain logical` 실행 시 "Unknown domain: logical. Available: ['math']" 오류를 수정하여 logical, commonsense 등 다른 도메인도 학습 및 평가가 가능하도록 합니다.

## 문제 분석

### 현재 상태
```
config/config.py     → DOMAIN_CONFIG: ['math'] 만 정의 ❌
config/domains.py    → DOMAIN_CONFIG: ['math', 'logical', 'commonsense'] 정의 ✅
config/__init__.py   → get_domain_data_dirs를 config.config에서 import ❌
```

### 오류 발생 경로
```
main.py (Line 124)
  ↓ get_domain_data_dirs(domain)
config/__init__.py (Line 53)
  ↓ from config.config import get_domain_data_dirs
config/config.py (Line 358-359)
  ↓ if domain not in DOMAIN_CONFIG  ← math만 포함
ValueError: Unknown domain: logical
```

## 구현 계획

### Step 1: config/config.py의 DOMAIN_CONFIG 제거 및 import 수정

**파일**: `config/config.py`

**변경 사항**:
1. `DOMAIN_CONFIG` 딕셔너리 제거 (Line 327-334)
2. `get_domain_data_dirs` 함수가 `config.domains`의 `DOMAIN_CONFIG`를 사용하도록 수정

**수정 전**:
```python
# config/config.py
DOMAIN_CONFIG = {
    "math": {
        "data_dir": DATA_DIR / "math",
        ...
    }
}

def get_domain_data_dirs(domain: str, ...):
    if domain not in DOMAIN_CONFIG:
        raise ValueError(...)
```

**수정 후**:
```python
# config/config.py
from config.domains import DOMAIN_CONFIG

def get_domain_data_dirs(domain: str, ...):
    if domain not in DOMAIN_CONFIG:
        raise ValueError(...)
```

### Step 2: Circular Import 방지

`config.domains`에서 `config.config`를 import하면 순환 import 발생 가능. 이를 방지하기 위해:

**옵션 A (권장)**: `get_domain_data_dirs` 함수를 `config/domains.py`로 이동
- 장점: 도메인 관련 코드가 한 곳에 집중
- 단점: 기존 import 경로 변경 필요

**옵션 B**: `config/config.py`에서 local import 사용
- 장점: 최소 변경
- 단점: 코드 구조가 덜 깔끔

### Step 3: config/__init__.py 수정

`get_domain_data_dirs`의 import 경로를 domains.py로 변경 (옵션 A 선택 시)

**수정 전**:
```python
from config.config import get_domain_data_dirs
```

**수정 후**:
```python
from config.domains import get_domain_data_dirs
```

### Step 4: 테스트 실행

```bash
# logical 도메인 테스트
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain logical --train-dataset reclor \
    --student-model Qwen/Qwen2.5-3B-Instruct \
    --teacher-model Qwen/Qwen2.5-3B-Instruct

# commonsense 도메인 테스트
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain commonsense --train-dataset arc_c \
    --student-model Qwen/Qwen2.5-3B-Instruct \
    --teacher-model Qwen/Qwen2.5-3B-Instruct

# math 도메인 회귀 테스트
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain math --train-dataset gsm8k \
    --student-model Qwen/Qwen2.5-3B-Instruct \
    --teacher-model Qwen/Qwen2.5-3B-Instruct
```

## 상세 구현 단계

### 1단계: config/domains.py에 get_domain_data_dirs 함수 이동

1. `config/config.py`에서 `get_domain_data_dirs` 함수 복사
2. `config/domains.py`에 붙여넣기
3. 필요한 import 추가 (Path 등)
4. `config/config.py`에서 해당 함수 및 DOMAIN_CONFIG 제거

### 2단계: config/__init__.py 수정

1. `get_domain_data_dirs` import 경로 변경
   - `from config.config import get_domain_data_dirs` → `from config.domains import get_domain_data_dirs`

### 3단계: config/config.py 정리

1. 사용되지 않는 DOMAIN_CONFIG 제거
2. 관련 import 정리

### 4단계: 검증

1. 모든 도메인에서 train/eval 명령 테스트
2. 기존 math 도메인 회귀 테스트

## 파일 변경 목록

| 파일 | 변경 내용 |
|------|----------|
| `config/domains.py` | `get_domain_data_dirs` 함수 추가 |
| `config/config.py` | `DOMAIN_CONFIG`, `get_domain_data_dirs` 제거 |
| `config/__init__.py` | import 경로 수정 |

## 리스크 및 대응

### 리스크 1: Circular Import
- **대응**: domains.py가 config.py를 import하지 않도록 함수 의존성 분리

### 리스크 2: 기존 코드 호환성
- **대응**: config/__init__.py를 통해 동일한 API 유지

### 리스크 3: 데이터 파일 누락
- **대응**: 데이터 폴더 구조 확인 (이미 존재함 확인됨)

## 예상 결과

수정 후:
```bash
$ python main.py --mode train --domain logical --train-dataset reclor ...
============================================================
ID-MAS: TRAIN MODE (Iterative Scaffolding Pipeline)
============================================================
Domain: logical
Train Dataset: reclor
...
# 정상 실행
```

## 작성자
- 날짜: 2026-01-14
- 워크플로우: Planning Workflow (coding domain)
