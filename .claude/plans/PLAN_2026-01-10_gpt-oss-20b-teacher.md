# openai/gpt-oss-20b Teacher 모델 기본 설정 계획

## 현재 상태

- `openai/gpt-oss-20b`는 이미 `AVAILABLE_TEACHER_MODELS`에 포함됨
- LLaMA-Factory 서버가 `http://0.0.0.0:2000`에서 실행 중
- CLI에서 `--teacher-model openai/gpt-oss-20b` 사용 가능

## 목표

1. `openai/gpt-oss-20b`를 기본 Teacher 모델로 변경 (선택적)
2. 문서 업데이트 (README, ARCHITECTURE, USAGE)

## 구현 계획

### Step 1: 기본 Teacher 모델 변경 (선택적)

**수정 파일**:
- config/config.py
- config/models.py

**변경 내용**:
```python
# Before
DEFAULT_TEACHER_MODEL = "gpt-5-2025-08-07"

# After (선택적 - 사용자 확인 필요)
DEFAULT_TEACHER_MODEL = "openai/gpt-oss-20b"
```

> 참고: 기본값을 변경하지 않아도 `--teacher-model openai/gpt-oss-20b`로 사용 가능

### Step 2: README.md 업데이트

**수정 내용**:
- LLaMA-Factory Teacher 모델 사용 방법 추가
- `openai/gpt-oss-20b` 사용 예시 추가

### Step 3: ARCHITECTURE.md 업데이트

**수정 내용**:
- Teacher Model 설정 섹션에 LLaMA-Factory 설명 추가
- 환경 변수 `LLAMA_FACTORY_BASE_URL` 설명 추가

### Step 4: USAGE.md 업데이트

**수정 내용**:
- LLaMA-Factory Teacher 모델 사용 예시 추가
- 환경 변수 설정 방법 추가

## 사용 예시 (현재도 가능)

```bash
# LLaMA-Factory 서버 실행 (이미 실행 중)
# INFO: Uvicorn running on http://0.0.0.0:2000

# openai/gpt-oss-20b Teacher 모델로 학습
python main.py --mode train --domain math --train-dataset math \
    --student-model Qwen/Qwen3-4B-Instruct-2507 \
    --teacher-model openai/gpt-oss-20b
```

## 예상 작업 시간

- Step 1: 2분 (선택적)
- Step 2-4: 10분

---

**생성일**: 2026-01-10
**상태**: 승인 대기
