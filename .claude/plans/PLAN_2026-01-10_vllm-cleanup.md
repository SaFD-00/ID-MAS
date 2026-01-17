# Plan: DEFAULT_VLLM_TEACHER_MODEL 변수 제거

## 개요

`DEFAULT_VLLM_TEACHER_MODEL` 변수 및 관련 참조를 코드베이스에서 제거하여 ImportError를 해결합니다.

## 문제 분석

```
ImportError: cannot import name 'DEFAULT_VLLM_TEACHER_MODEL' from 'config.config'
```

**원인**: `main.py`가 `config/config.py`에서 `DEFAULT_VLLM_TEACHER_MODEL`을 import하려 하지만, 해당 변수는 `config/models.py`에만 정의되어 있음.

## 수정 계획

### Step 1: main.py 수정

**파일**: `main.py`

| 라인 | 현재 | 수정 |
|------|------|------|
| 49 | `DEFAULT_VLLM_TEACHER_MODEL` import | import 제거 |
| 772 | vLLM 관련 help 텍스트 | help 텍스트 단순화 |

**변경 내용**:
```python
# Line 49: import 제거
- AVAILABLE_TEACHER_MODELS, DEFAULT_TEACHER_MODEL, DEFAULT_VLLM_TEACHER_MODEL,
+ AVAILABLE_TEACHER_MODELS, DEFAULT_TEACHER_MODEL,

# Line 770-772: help 텍스트 수정
- help=f"Teacher model for instructional design and evaluation. "
-      f"Default: {DEFAULT_TEACHER_MODEL} (OpenAI). "
-      f"For vLLM (GPU server): {DEFAULT_VLLM_TEACHER_MODEL}"
+ help=f"Teacher model for instructional design and evaluation. "
+      f"Default: {DEFAULT_TEACHER_MODEL}. "
+      f"Available: {AVAILABLE_TEACHER_MODELS}"
```

### Step 2: config/__init__.py 수정

**파일**: `config/__init__.py`

| 라인 | 현재 | 수정 |
|------|------|------|
| 18 | `DEFAULT_VLLM_TEACHER_MODEL` import | import 제거 |
| 69 | `__all__`에 포함 | 항목 제거 |

### Step 3: config/models.py 수정

**파일**: `config/models.py`

| 라인 | 현재 | 수정 |
|------|------|------|
| 27 | `DEFAULT_VLLM_TEACHER_MODEL` 정의 | 라인 삭제 |

### Step 4: tests/test_config.py 수정

**파일**: `tests/test_config.py`

| 라인 | 현재 | 수정 |
|------|------|------|
| 15 | `DEFAULT_VLLM_TEACHER_MODEL` import | import 제거 |
| 56-70 | `test_default_vllm_teacher_model` 메서드 | 메서드 삭제 |

### Step 5: ARCHITECTURE.md 수정

**파일**: `ARCHITECTURE.md`

| 라인 | 현재 | 수정 |
|------|------|------|
| 738 | `DEFAULT_VLLM_TEACHER_MODEL` 변수 예시 | 라인 삭제 |
| 775-779 | Teacher Model Defaults 섹션 | 섹션 단순화 |

## 검증 계획

### 검증 1: 오류 해결 확인
```bash
python main.py --help
```
- ImportError 없이 help 출력 확인

### 검증 2: 테스트 실행
```bash
python -m pytest tests/test_config.py -v
```
- 모든 테스트 통과 확인

### 검증 3: 실제 실행 테스트
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --domain math --train-dataset math \
    --student-model Qwen/Qwen3-4B-Instruct-2507 \
    --teacher-model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
```
- 정상 실행 확인

## 영향 범위

- **코드 파일**: 4개 (main.py, config/__init__.py, config/models.py, tests/test_config.py)
- **문서 파일**: 1개 (ARCHITECTURE.md)
- **기능 영향**: 없음 (변수만 제거, vLLM 기능 자체는 그대로 유지)

## 리스크

- **낮음**: 단순 변수 제거로 기능에 영향 없음
- vLLM 모델은 여전히 `--teacher-model` 옵션으로 지정 가능
