# ID-MAS 3가지 버그 수정 계획

## 개요

ID-MAS 실행 시 발생하는 3가지 이슈를 수정합니다.

## 이슈 분석

### 이슈 1: Terminal Goal 잘림

- **증상**: `Terminal Goal: Solve advanced mathematical problems by selecting appropriate mathematical conce...`
- **파일**: [main.py:574](main.py#L574)
- **문제 코드**:
```python
print(f"Terminal Goal: {pipeline.terminal_goal[:80]}...")
```
- **원인**: 80자로 자르고 `...`을 붙여서 출력
- **수정 방향**: 전체 Terminal Goal 출력

---

### 이슈 2: torch_dtype Deprecation Warning

- **증상**: `` `torch_dtype` is deprecated! Use `dtype` instead! ``
- **파일**: [models/model_cache.py:56-58](models/model_cache.py#L56-L58)
- **문제 코드**:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,  # deprecated
    ...
)
```
- **원인**: transformers 라이브러리에서 `torch_dtype` 파라미터가 deprecated됨
- **수정 방향**: `torch_dtype`를 `dtype`으로 변경

---

### 이슈 3: Teacher Model 불일치 (INSTRUCTIONAL DESIGN PHASE)

- **증상**:
  - CLI: `--teacher-model meta-llama/Llama-3.1-8B-Instruct`
  - 출력: `[TeacherModelWrapper] Using API model: gpt-5-2025-08-07`
  - Rubric Development 단계에서만 API 모델 사용

- **파일**: [main.py:183](main.py#L183)
- **문제 코드**:
```python
# run_design_phase 메서드 내부 (183-184줄)
rubric_dev = RubricDevelopment()  # teacher_config 없이 호출!
```

- **원인 분석**:
  1. `IDMASPipeline.__init__`에서 모든 design module이 올바르게 초기화됨 (109-113줄):
     ```python
     self.analysis = InstructionalAnalysis(teacher_config)
     self.objectives = PerformanceObjectives(teacher_config)
     self.test_dev = TestItemDevelopment(teacher_config)
     self.rubric_dev = RubricDevelopment(teacher_config)
     ```
  2. 그러나 `run_design_phase`에서 `rubric_dev = RubricDevelopment()`로 **새로운 인스턴스** 생성 (183줄)
  3. config 없이 생성 → `TeacherModelWrapper(None)` → `DESIGN_MODEL_CONFIG` 사용
  4. `DESIGN_MODEL_CONFIG`는 `gpt-5-2025-08-07`로 설정됨 ([config/config.py:96](config/config.py#L96))

- **Design Module 구조** (모두 teacher_config 지원):
  | 모듈 | 파일 | 초기화 |
  |------|------|--------|
  | InstructionalAnalysis | [design_modules/analysis.py:18](design_modules/analysis.py#L18) | `TeacherModelWrapper(teacher_config)` |
  | PerformanceObjectives | [design_modules/objectives.py:18](design_modules/objectives.py#L18) | `TeacherModelWrapper(teacher_config)` |
  | TestItemDevelopment | [design_modules/test.py:18](design_modules/test.py#L18) | `TeacherModelWrapper(teacher_config)` |
  | RubricDevelopment | [design_modules/rubric.py:21](design_modules/rubric.py#L21) | `TeacherModelWrapper(teacher_config)` |

- **수정 방향**: `self.rubric_dev` 사용 (이미 올바르게 초기화된 인스턴스)

---

## 수정 계획

### Step 1: Terminal Goal 전체 출력

**파일**: `main.py`
**위치**: 574줄

```python
# Before
print(f"Terminal Goal: {pipeline.terminal_goal[:80]}...")

# After
print(f"Terminal Goal: {pipeline.terminal_goal}")
```

---

### Step 2: torch_dtype → dtype 변경

**파일**: `models/model_cache.py`
**위치**: 56-61줄

```python
# Before
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
    token=HF_TOKEN
)

# After
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=dtype,
    device_map="auto" if device == "cuda" else None,
    token=HF_TOKEN
)
```

---

### Step 3: RubricDevelopment에서 self.rubric_dev 사용

**파일**: `main.py`
**위치**: 183-192줄

```python
# Before
rubric_dev = RubricDevelopment()  # config 없이 새 인스턴스

rubric = rubric_dev.generate_rubric(
    task_description=self.terminal_goal,
    output_type=output_type,
    performance_objectives=objectives_result
)

# After (self.rubric_dev 사용)
rubric = self.rubric_dev.generate_rubric(
    task_description=self.terminal_goal,
    output_type=output_type,
    performance_objectives=objectives_result
)
```

---

## 검증 방법

수정 후 동일한 명령어로 실행하여 확인:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --domain math --train-dataset math \
    --student-model meta-llama/Llama-3.1-8B-Instruct \
    --teacher-model meta-llama/Llama-3.1-8B-Instruct
```

### 예상 결과

1. **Terminal Goal**: 전체 텍스트 출력
   ```
   Terminal Goal: Solve advanced mathematical problems by selecting appropriate mathematical concepts and constructing logically valid, multi-step reasoning that leads to a correct solution.
   ```

2. **Deprecation Warning**: 경고 메시지 없음

3. **Teacher Model**: Rubric Development에서도 로컬 모델 사용
   ```
   [TeacherModelWrapper] Using local model: meta-llama/Llama-3.1-8B-Instruct
   ```

---

## 영향 범위

- `main.py`: 2곳 수정
- `models/model_cache.py`: 1곳 수정
- 기존 동작에 영향 없음 (버그 수정만)
