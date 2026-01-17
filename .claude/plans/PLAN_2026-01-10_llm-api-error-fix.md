# LLM API 호출 오류 해결 및 명칭 리팩토링 계획

## 문제 요약

**증상**: `--teacher-model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` 사용 시 "GPT API 호출 오류: Internal Server Error" 반복 발생

**근본 원인**:
1. LLaMA-Factory 서버 내부에서 500 오류 발생
2. 코드 전체에서 "GPT", "vLLM" 용어가 혼용되어 혼란 야기
3. 재시도 로직 없이 즉시 실패 처리

## 해결 계획

---

### Step 1: 파일명 변경

**목적**: 실제 용도에 맞는 파일명으로 변경

| 기존 파일 | 새 파일 | 설명 |
|-----------|---------|------|
| `models/gpt_wrapper.py` | `models/llm_wrapper.py` | LLM API 래퍼 |

---

### Step 2: 클래스명 및 변수명 변경

**목적**: 코드의 의미를 명확히 전달

#### 2.1 클래스명 변경

| 기존 | 새 이름 | 파일 |
|------|---------|------|
| `GPTWrapper` | `LLMWrapper` | models/llm_wrapper.py |

#### 2.2 변수명 변경 (모든 파일에서)

| 기존 | 새 이름 | 사용 파일 |
|------|---------|-----------|
| `self.gpt` | `self.llm` | design_modules/step2_analysis.py |
| `self.gpt` | `self.llm` | design_modules/step4_objectives.py |
| `self.gpt` | `self.llm` | design_modules/step5_test.py |
| `self.gpt` | `self.llm` | design_modules/step5_rubric.py |
| `self.gpt` | `self.llm` | learning_loop/teacher_model.py |

#### 2.3 Import 문 변경

```python
# Before
from models.gpt_wrapper import GPTWrapper

# After
from models.llm_wrapper import LLMWrapper
```

---

### Step 3: 주석 및 Docstring 수정

**목적**: 실제 사용하는 서비스에 맞게 문서화

#### 3.1 models/llm_wrapper.py (기존 gpt_wrapper.py)

```python
# Before
"""
GPT/LLM 모델 래퍼 (OpenAI API 및 vLLM 호환)
"""
class GPTWrapper:
    """GPT-5 또는 vLLM 호환 모델을 위한 래퍼 클래스"""

# After
"""
LLM 모델 래퍼 (OpenAI API 및 LLaMA-Factory 호환)
"""
class LLMWrapper:
    """Teacher Model API 래퍼 클래스 (OpenAI, LLaMA-Factory 호환)"""
```

#### 3.2 config/config.py

```python
# Before
# LLaMA-Factory vLLM
# LLaMA-Factory vLLM 모델 (기본값 고정: localhost:2000/v1)

# After
# LLaMA-Factory API (OpenAI-compatible)
# LLaMA-Factory 모델 (기본값: localhost:2000/v1)
```

#### 3.3 config/models.py

동일하게 "vLLM" → "LLaMA-Factory API" 수정

---

### Step 4: 오류 메시지 개선

**목적**: 실제 사용 중인 엔드포인트를 표시하여 디버깅 용이

```python
# Before (models/llm_wrapper.py)
raise Exception(f"GPT API 호출 오류: {str(e)}")

# After
endpoint_info = self.model_config.get("base_url", "OpenAI API")
raise Exception(f"LLM API 호출 오류 ({endpoint_info}): {str(e)}")
```

---

### Step 5: 재시도 로직 추가

**목적**: 일시적인 서버 오류 시 자동 복구

```python
import time

def generate(self, prompt: str, ...):
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            error_str = str(e)
            # 5xx 서버 오류만 재시도
            if "500" in error_str or "502" in error_str or "503" in error_str or "Internal Server Error" in error_str:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1, 2, 4초
                    print(f"  서버 오류 발생, {wait_time}초 후 재시도 ({attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
            break

    endpoint_info = self.model_config.get("base_url", "OpenAI API")
    raise Exception(f"LLM API 호출 오류 ({endpoint_info}): {str(last_error)}")
```

---

### Step 6: 서버 URL 환경 변수화

**목적**: 하드코딩된 URL을 환경 변수로 유연하게 설정

#### config/config.py

```python
# Before
"base_url": "http://localhost:2000/v1"

# After
LLAMA_FACTORY_BASE_URL = os.getenv("LLAMA_FACTORY_BASE_URL", "http://localhost:2000/v1")
# ...
"base_url": LLAMA_FACTORY_BASE_URL
```

#### .env.example 추가

```
# LLaMA-Factory API Server
LLAMA_FACTORY_BASE_URL=http://localhost:2000/v1
```

---

### Step 7: 테스트 코드 업데이트

**수정 파일**:
- tests/test_model_wrappers.py
- tests/test_config.py
- test_system.py

**변경 사항**:
- Import 경로 변경: `gpt_wrapper` → `llm_wrapper`
- 클래스명 변경: `GPTWrapper` → `LLMWrapper`
- 함수명 변경: `test_gpt_wrapper` → `test_llm_wrapper`
- 테스트명/주석 수정: "GPT" → "LLM"

---

## 수정 파일 전체 목록

| 파일 | 수정 유형 |
|------|----------|
| models/gpt_wrapper.py → models/llm_wrapper.py | 파일명 변경, 클래스명, 주석, 재시도 로직 |
| models/__init__.py | Export 변경 (필요시) |
| config/config.py | 주석 수정, 환경 변수화 |
| config/models.py | 주석 수정 |
| design_modules/step2_analysis.py | Import, 변수명 |
| design_modules/step4_objectives.py | Import, 변수명, 주석 |
| design_modules/step5_test.py | Import, 변수명 |
| design_modules/step5_rubric.py | Import, 변수명 |
| learning_loop/teacher_model.py | Import, 변수명 |
| main.py | 주석 수정 |
| tests/test_model_wrappers.py | Import, 클래스명, 테스트명 |
| tests/test_config.py | 주석 수정 |
| test_system.py | 함수명, Import, 주석 |
| .env.example | 환경 변수 추가 |

---

## 구현 순서

| 순서 | 작업 | 우선순위 |
|------|------|----------|
| 1 | models/gpt_wrapper.py → llm_wrapper.py 변경 (핵심 수정) | 높음 |
| 2 | 모든 Import 및 변수명 일괄 변경 | 높음 |
| 3 | config/config.py, config/models.py 주석 수정 | 중간 |
| 4 | 테스트 코드 업데이트 | 중간 |
| 5 | .env.example 업데이트 | 낮음 |
| 6 | 테스트 실행하여 검증 | 높음 |

---

## 예상 결과

1. 일시적인 서버 오류 시 자동 재시도로 안정성 향상
2. 오류 메시지가 실제 사용 중인 API 엔드포인트를 정확히 표시
3. 코드 전체에서 일관된 용어 사용 (GPT → LLM, vLLM → LLaMA-Factory)
4. 서버 URL을 환경 변수로 유연하게 설정 가능

---

## 추가 확인 필요

LLaMA-Factory 서버 로그에서 500 오류의 실제 원인 확인:
```bash
curl http://localhost:2000/v1/models
```

---

**생성일**: 2026-01-10
**도메인**: coding
**상태**: 승인 대기
