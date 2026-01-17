# Plan: Teacher Model 로컬 로드 및 모델 공유 최적화

## 목표

1. **Teacher Model을 로컬에서 로드**: LLaMA-Factory API 대신 HuggingFace 모델을 직접 로드
2. **모델 공유 최적화**: Teacher와 Student 모델이 동일할 경우 하나만 로드

## 현재 구조 분석

### TeacherModel (API 기반)
```
TeacherModel (learning_loop/teacher_model.py)
└── LLMWrapper (models/llm_wrapper.py)
    └── OpenAI Client (openai.OpenAI)
        ├── OpenAI API (gpt-* 모델)
        └── LLaMA-Factory API (custom endpoint)
```

### StudentModel (로컬 로드)
```
StudentModel
└── StudentModelWrapper (models/student_wrapper.py)
    └── HuggingFace (AutoModelForCausalLM, AutoTokenizer)
        └── _model_cache (클래스 레벨 캐시)
```

## 영향 범위

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `models/teacher_wrapper.py` | **신규 생성** | 로컬 Teacher 모델 래퍼 |
| `models/model_cache.py` | **신규 생성** | 공유 모델 캐시 매니저 |
| `models/student_wrapper.py` | 수정 | 공유 캐시(`ModelCache`) 사용 |
| `models/llm_wrapper.py` | **변경 없음** | 기존 API 래퍼 유지 |
| `learning_loop/teacher_model.py` | 수정 | `TeacherModelWrapper` 사용 |
| `main.py` | 수정 | 모델 공유 로깅 |
| `config/config.py` | 수정 | Teacher 모델 설정 업데이트 |

## 설계

### 1. 공유 모델 캐시 매니저 (`models/model_cache.py`)

```python
class ModelCache:
    """글로벌 모델 캐시 (Teacher/Student 공유)"""
    _cache: Dict[Tuple[str, str], Dict] = {}  # {(model_name, device): {model, tokenizer}}

    @classmethod
    def get_or_load(cls, model_name: str, device: str = "cuda") -> Dict:
        """캐시에서 모델 반환, 없으면 로드"""
        ...

    @classmethod
    def is_loaded(cls, model_name: str, device: str = "cuda") -> bool:
        """모델이 로드되어 있는지 확인"""
        ...

    @classmethod
    def clear(cls):
        """캐시 초기화"""
        ...
```

### 2. TeacherModelWrapper (`models/teacher_wrapper.py` 신규 생성)

```python
class TeacherModelWrapper:
    """Teacher 모델 래퍼 - API와 로컬 모델 모두 지원"""

    def __init__(self, config: dict = None):
        model_name = config.get("model", "")

        # 1. OpenAI API 모델 (gpt-*)
        if model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3"):
            self._use_api = True
            self._api_wrapper = LLMWrapper(config)  # 기존 API 래퍼 사용
        # 2. 로컬 모델 (Qwen, Llama 등)
        else:
            self._use_api = False
            cached = ModelCache.get_or_load(model_name, device)
            self.model = cached["model"]
            self.tokenizer = cached["tokenizer"]

    def generate(self, prompt: str, system_message: str = None) -> str:
        if self._use_api:
            return self._api_wrapper.generate(prompt, system_message)
        else:
            # 로컬 모델 추론 로직
            ...

    def generate_json(self, prompt: str, system_message: str = None) -> Dict:
        if self._use_api:
            return self._api_wrapper.generate_json(prompt, system_message)
        else:
            # 로컬 모델 JSON 생성 + 파싱
            ...
```

### 3. StudentModelWrapper 수정

```python
class StudentModelWrapper:
    def __init__(self, ...):
        # 기존 _model_cache 대신 공유 ModelCache 사용
        cached = ModelCache.get_or_load(actual_model_name, device)
        self.model = cached["model"]
        self.tokenizer = cached["tokenizer"]
```

### 4. TeacherModel 수정 (`learning_loop/teacher_model.py`)

```python
class TeacherModel:
    def __init__(self, config: dict = None):
        # LLMWrapper → TeacherModelWrapper로 변경
        self.llm = TeacherModelWrapper(config)
```

### 5. main.py 수정

```python
# Teacher와 Student가 동일 모델인지 확인
if teacher_model_name == student_model_name:
    print(f"Teacher and Student use same model: {teacher_model_name}")
    print("Model will be loaded once and shared.")
```

## 구현 단계

### Step 1: 공유 모델 캐시 매니저 생성
- [ ] `models/model_cache.py` 생성
- [ ] `ModelCache` 클래스 구현
  - `get_or_load()`: 캐시에서 모델 반환 또는 로드
  - `is_loaded()`: 로드 여부 확인
  - `clear()`: 캐시 초기화

### Step 2: TeacherModelWrapper 생성 (신규)
- [ ] `models/teacher_wrapper.py` 생성
- [ ] API 모델과 로컬 모델 분기 처리
  - `gpt-*`, `o1-*`, `o3-*` → `LLMWrapper` 위임 (기존 API 로직)
  - 그 외 → 로컬 로드 (`ModelCache` 사용)
- [ ] `generate()` 메소드 구현 (API/로컬 분기)
- [ ] `generate_json()` 메소드 구현 (JSON 추출 로직 강화)

### Step 3: StudentModelWrapper 수정
- [ ] `models/student_wrapper.py` 수정
- [ ] 기존 `_model_cache`를 `ModelCache`로 교체
- [ ] 공유 캐시 사용하도록 변경

### Step 4: TeacherModel 수정
- [ ] `learning_loop/teacher_model.py` 수정
- [ ] `LLMWrapper` → `TeacherModelWrapper`로 변경

### Step 5: config 업데이트
- [ ] `config/config.py` 수정
- [ ] `create_teacher_config()` 함수 업데이트
  - 로컬 모델용 설정 추가 (max_new_tokens, temperature 등)
- [ ] `DEFAULT_VLLM_TEACHER_MODEL` 설정 검토

### Step 6: main.py 및 통합 테스트
- [ ] main.py에 모델 공유 로깅 추가
- [ ] 동일 모델 사용 시 메모리 절약 확인
- [ ] 테스트 실행

## 위험 요소 및 대응

### 1. JSON 응답 파싱
- **위험**: 로컬 모델은 JSON 형식을 정확히 따르지 않을 수 있음
- **대응**: `generate_json()`에서 JSON 추출 로직 강화
  - \`\`\`json ... \`\`\` 블록 파싱
  - `{` ... `}` 패턴 추출
  - 기존 `_fix_json_escapes()` 재사용

### 2. 메모리 부족
- **위험**: 대형 모델(14B+) 로드 시 OOM 가능
- **대응**: `device_map="auto"` 사용, 필요시 quantization 지원

### 3. 생성 품질 차이
- **위험**: API 모델과 로컬 모델의 품질 차이
- **대응**: 충분한 max_new_tokens 설정, 온도 조절

### 4. 하위 호환성
- **위험**: 기존 코드 손상
- **대응**:
  - `LLMWrapper`는 변경 없이 유지
  - `TeacherModel`만 `TeacherModelWrapper` 사용
  - API 모델은 내부적으로 `LLMWrapper` 위임

## 예상 결과

### Before (현재)
```
Teacher: LLaMA-Factory API → 외부 서버 호출
Student: HuggingFace → 로컬 GPU 로드 (자체 캐시)
→ 동일 모델도 각각 로드 (비효율)
```

### After (변경 후)
```
Teacher == Student (로컬 모델):
  → ModelCache에서 1회만 로드
  → Teacher/Student 모두 동일 모델 참조

Teacher != Student (로컬 모델):
  → 각각 ModelCache에 로드 (자동 캐싱)

Teacher가 gpt-*/o1-*/o3-* 모델:
  → TeacherModelWrapper가 LLMWrapper에 위임
  → 기존 OpenAI API 사용 (변경 없음)
```

## 파일 구조 (변경 후)

```
models/
├── model_cache.py         # 신규: 공유 모델 캐시
├── teacher_wrapper.py     # 신규: Teacher 래퍼 (API + 로컬)
├── student_wrapper.py     # 수정: ModelCache 사용
├── llm_wrapper.py         # 유지: API 전용 (변경 없음)
└── base_wrapper.py        # 유지
```

## 완료 기준

- [ ] `models/model_cache.py` 생성 및 테스트
- [ ] `models/teacher_wrapper.py` 생성 및 테스트
- [ ] `StudentModelWrapper`가 공유 캐시 사용
- [ ] `TeacherModel`이 `TeacherModelWrapper` 사용
- [ ] Teacher == Student 시 모델 1회만 로드 확인
- [ ] 기존 OpenAI API 모델 정상 동작 확인
- [ ] 전체 파이프라인 테스트 통과
