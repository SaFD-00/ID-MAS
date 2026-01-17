# Terminal Goal Fallback 제거 계획

## 요약

Terminal Goal의 하드코딩된 초기값(fallback)을 제거하고, 생성된 것이 없으면 "Not generated yet"으로 표시하도록 수정.

## 현재 상태

### 문제점

1. **초기화 시 항상 fallback 사용** ([main.py:102](main.py#L102))
   ```python
   self.terminal_goal = get_terminal_goal(self.train_dataset)  # teacher_model 미전달 → fallback
   ```

2. **fallback이 두 곳에 중복 정의**
   - [config/domains.py:14-24](config/domains.py#L14-L24): `TERMINAL_GOALS` 딕셔너리
   - [design_modules/terminal_goal.py:158-163](design_modules/terminal_goal.py#L158-L163): `FALLBACK_TERMINAL_GOALS` 딕셔너리

3. **Design JSON에서 로드 로직이 있지만 활용 안 됨**
   - `_load_terminal_goal_from_design()` 함수 존재하나, 초기화 시 teacher_model 미전달

### Terminal Goal 흐름

```
Pipeline 초기화 (Line 102)
    ↓
get_terminal_goal(dataset)  ← teacher_model 없음 → fallback 반환
    ↓
Design Phase 실행 시 (Line 170-190)
    ├── 샘플 파일 있음 → Teacher Model로 생성 → self.terminal_goal 업데이트
    └── 샘플 파일 없음 → fallback 유지
    ↓
Design JSON 저장 (Line 238)
    └── terminal_goal 필드로 저장
```

## 수정 계획

### Step 1: `config/domains.py` 수정

**파일**: [config/domains.py](config/domains.py)

**수정 내용**:

1. `TERMINAL_GOALS` 딕셔너리 제거 (Line 14-24)

2. `get_terminal_goal()` 함수 수정 (Line 95-126):
   - Design JSON에서만 로드 시도
   - fallback 제거, 없으면 `None` 반환

```python
def get_terminal_goal(
    dataset: str,
    teacher_model: Optional[str] = None,
    design_dir: Optional[Path] = None
) -> Optional[str]:
    """
    Get Terminal Goal for a training dataset from design JSON.

    Args:
        dataset: 데이터셋 이름 (e.g., "gsm8k", "math")
        teacher_model: Teacher 모델 이름
        design_dir: Design JSON 디렉토리 (None이면 기본 경로)

    Returns:
        Terminal Goal 문자열 또는 None (생성된 것이 없는 경우)
    """
    if teacher_model:
        domain = DATASET_TO_DOMAIN.get(dataset)
        if domain:
            design_goal = _load_terminal_goal_from_design(domain, dataset, teacher_model, design_dir)
            if design_goal:
                return design_goal

    # Design JSON에 없으면 None 반환 (fallback 제거)
    return None
```

### Step 2: `design_modules/terminal_goal.py` 수정

**파일**: [design_modules/terminal_goal.py](design_modules/terminal_goal.py)

**수정 내용**:

1. `FALLBACK_TERMINAL_GOALS` 딕셔너리 제거 (Line 157-163)

2. `get_fallback_terminal_goal()` 함수 제거 또는 deprecation 경고 추가 (Line 166-179)

### Step 3: `main.py` 수정

**파일**: [main.py](main.py)

**수정 내용**:

1. **초기화 로직 수정** (Line 102):
   ```python
   # Design JSON에서 Terminal Goal 로드 시도
   self.terminal_goal = get_terminal_goal(
       self.train_dataset,
       teacher_model=self.teacher_model_name
   )
   ```

2. **Terminal Goal 출력 수정** (Line 613):
   ```python
   if pipeline.terminal_goal:
       print(f"Terminal Goal: {pipeline.terminal_goal}")
   else:
       print(f"Terminal Goal: Not generated yet")
       print(f"  Run design phase first or use --run-design flag")
   ```

3. **Design Phase 실패 시 처리 수정** (Line 187-190):
   ```python
   except Exception as e:
       print(f"  [Error] Terminal Goal generation failed: {e}")
       # fallback 사용하지 않고 None 유지
       self.terminal_goal = None
   ```

4. **Design Phase 스킵 시 메시지 수정** (Line 191-197):
   ```python
   else:
       if not samples_path.exists():
           print(f"\n[Step 0] Terminal Goal Generation (SKIPPED)")
           print(f"  Samples file not found: {samples_path.name}")
           print(f"  Run 'python -m utils.sample_extractor' to generate samples.")
       # Terminal Goal이 없으면 Design Phase 실행 필요
       if not self.terminal_goal:
           print(f"  Terminal Goal: Not generated yet")
   ```

5. **Learning Objective 설정 시 검증 추가** (Line 199-201):
   ```python
   if learning_objective is None:
       if not self.terminal_goal:
           raise ValueError(
               f"Terminal Goal not found for {self.train_dataset}. "
               f"Run design phase with --run-design flag first."
           )
       learning_objective = self.terminal_goal
   ```

### Step 4: Import 정리

**파일**: [main.py](main.py) Line 39

**수정 내용**:
- `get_fallback_terminal_goal` import 제거

```python
from design_modules.terminal_goal import TerminalGoalGenerator  # get_fallback_terminal_goal 제거
```

## 수정 후 흐름

```
Pipeline 초기화
    ↓
get_terminal_goal(dataset, teacher_model)
    ├── Design JSON 있음 → terminal_goal 로드
    └── Design JSON 없음 → None 반환
    ↓
Terminal Goal 출력 (Line 613)
    ├── 있음 → "Terminal Goal: {value}"
    └── 없음 → "Terminal Goal: Not generated yet"
    ↓
Design Phase 실행 (--run-design)
    ├── 성공 → self.terminal_goal 설정 → Design JSON 저장
    └── 실패 → 에러 발생 (fallback 없음)
    ↓
Learning Phase
    └── terminal_goal 필수 → 없으면 에러
```

## 영향 범위

| 파일 | 변경 내용 |
|------|----------|
| `config/domains.py` | `TERMINAL_GOALS` 제거, `get_terminal_goal()` 수정 |
| `design_modules/terminal_goal.py` | `FALLBACK_TERMINAL_GOALS` 제거, `get_fallback_terminal_goal()` 제거 |
| `main.py` | 초기화/출력 로직 수정, import 정리 |

### Step 5: 재시도 로직 추가

**파일**: [design_modules/terminal_goal.py](design_modules/terminal_goal.py)

**수정 내용**:

`generate()` 메서드에 3번 재시도 로직 추가:

```python
def generate(
    self,
    train_samples: List[Dict],
    domain: str,
    dataset: str,
    prompt_template: str = None,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Terminal Goal 생성 (최대 3번 재시도)

    Args:
        ...
        max_retries: 최대 재시도 횟수 (기본 3)

    Raises:
        RuntimeError: max_retries 초과 시 프로그램 종료
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            # 프롬프트 구성
            prompt = get_terminal_goal_prompt(...)

            # LLM으로 Terminal Goal 생성
            result_json = self.llm.generate_json(
                prompt=prompt,
                system_message=TERMINAL_GOAL_SYSTEM_MESSAGE
            )

            # 결과 검증
            terminal_goal = result_json.get("terminal_goal", "")
            if not terminal_goal:
                raise ValueError("Empty terminal_goal in response")

            # 성공 시 반환
            return {
                "terminal_goal": terminal_goal,
                ...
            }

        except Exception as e:
            last_error = e
            print(f"  [Attempt {attempt}/{max_retries}] Terminal Goal generation failed: {e}")

            if attempt < max_retries:
                print(f"  Retrying...")
            else:
                print(f"  [FATAL] All {max_retries} attempts failed.")

    # 모든 재시도 실패 → 프로그램 종료
    raise RuntimeError(
        f"Terminal Goal generation failed after {max_retries} attempts. "
        f"Last error: {last_error}"
    )
```

### Step 6: main.py에서 RuntimeError 처리

**파일**: [main.py](main.py) Line 178-190

**수정 내용**:

```python
# Teacher Model로 Terminal Goal 생성 (3번 재시도, 실패 시 종료)
try:
    terminal_goal_result = self.terminal_goal_gen.generate(
        train_samples=train_samples,
        domain=self.domain,
        dataset=self.train_dataset,
        max_retries=3
    )
    self.terminal_goal = terminal_goal_result["terminal_goal"]
    terminal_goal_metadata = terminal_goal_result.get("metadata", {})
    print(f"  Generated: {self.terminal_goal[:80]}...")

except RuntimeError as e:
    # 3번 재시도 후 실패 → 프로그램 종료
    print(f"\n[FATAL] {e}")
    print(f"Please check:")
    print(f"  1. Teacher model availability")
    print(f"  2. Sample data quality ({samples_path.name})")
    print(f"  3. Network connection (if using API)")
    sys.exit(1)

except Exception as e:
    # 기타 예외 → 프로그램 종료 (fallback 없음)
    print(f"\n[FATAL] Unexpected error during Terminal Goal generation: {e}")
    sys.exit(1)
```

### Step 7: Design 모듈 전체에 재시도 로직 추가

**적용 대상**: 모든 Design 모듈에 동일한 3번 재시도 패턴 적용

| 모듈 | 메서드 | 변경 내용 |
|------|--------|----------|
| `terminal_goal.py` | `generate()` | 3번 재시도 + RuntimeError |
| `analysis.py` | `analyze()` | 3번 재시도 + RuntimeError |
| `objectives.py` | `generate_objectives()` | 3번 재시도 + RuntimeError |
| `rubric.py` | `generate_rubric()` | 3번 재시도 + RuntimeError |

**공통 패턴 (각 모듈에 적용)**:

```python
def _execute_with_retry(self, operation_name: str, func, *args, max_retries: int = 3, **kwargs):
    """
    재시도 로직이 포함된 실행 래퍼

    Args:
        operation_name: 작업 이름 (로그용)
        func: 실행할 함수
        max_retries: 최대 재시도 횟수

    Raises:
        RuntimeError: max_retries 초과 시
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            print(f"  [Attempt {attempt}/{max_retries}] {operation_name} failed: {e}")
            if attempt < max_retries:
                print(f"  Retrying...")
            else:
                print(f"  [FATAL] All {max_retries} attempts failed.")

    raise RuntimeError(
        f"{operation_name} failed after {max_retries} attempts. "
        f"Last error: {last_error}"
    )
```

#### 7.1 analysis.py 수정

**파일**: [design_modules/analysis.py](design_modules/analysis.py)

```python
def analyze(self, learning_objective: str, max_retries: int = 3) -> Dict[str, Any]:
    """교수 분석 수행 (최대 3번 재시도)"""
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            prompt = INSTRUCTIONAL_ANALYSIS_PROMPT.format(
                learning_objective=learning_objective
            )
            result_text = self.llm.generate(prompt)

            if not result_text or not result_text.strip():
                raise ValueError("Empty response from LLM")

            parsed_result = self._parse_analysis_result(result_text)

            return {
                "learning_objective": learning_objective,
                "raw_output": result_text,
                "parsed": parsed_result
            }

        except Exception as e:
            last_error = e
            print(f"  [Attempt {attempt}/{max_retries}] Instructional Analysis failed: {e}")
            if attempt < max_retries:
                print(f"  Retrying...")

    raise RuntimeError(
        f"Instructional Analysis failed after {max_retries} attempts. "
        f"Last error: {last_error}"
    )
```

#### 7.2 objectives.py 수정

**파일**: [design_modules/objectives.py](design_modules/objectives.py)

동일한 패턴으로 `generate_objectives()` 메서드에 재시도 로직 추가

#### 7.3 rubric.py 수정

**파일**: [design_modules/rubric.py](design_modules/rubric.py)

동일한 패턴으로 `generate_rubric()` 메서드에 재시도 로직 추가

### Step 8: main.py Design Phase 에러 처리 통합

**파일**: [main.py](main.py)

모든 Design 단계에서 RuntimeError 처리:

```python
def run_design_phase(self, ...):
    try:
        # Step 0: Terminal Goal Generation
        ...

        # Step 1: Instructional Analysis
        analysis_result = self.analysis.analyze(learning_objective, max_retries=3)

        # Step 2: Performance Objectives
        objectives_result = self.objectives.generate_objectives(
            analysis_result["raw_output"], max_retries=3
        )

        # Step 3: Rubric Development
        rubric = self.rubric_dev.generate_rubric(..., max_retries=3)

    except RuntimeError as e:
        print(f"\n[FATAL] Design Phase failed: {e}")
        sys.exit(1)
```

## 영향 범위 (최종)

| 파일 | 변경 내용 |
|------|----------|
| `config/domains.py` | `TERMINAL_GOALS` 제거, `get_terminal_goal()` → None 반환 |
| `design_modules/terminal_goal.py` | fallback 제거, **3번 재시도 + RuntimeError** |
| `design_modules/analysis.py` | **3번 재시도 + RuntimeError** |
| `design_modules/objectives.py` | **3번 재시도 + RuntimeError** |
| `design_modules/rubric.py` | **3번 재시도 + RuntimeError** |
| `main.py` | 초기화/출력 로직 수정, import 정리, **RuntimeError → sys.exit(1)** |

## 테스트 시나리오

1. **Design JSON 없는 상태에서 실행**
   - 예상: "Terminal Goal: Not generated yet" 출력
   - Learning phase 실행 시 에러

2. **--run-design으로 Design Phase 실행**
   - 예상: Terminal Goal 생성 → Design JSON 저장

3. **Design JSON 있는 상태에서 실행**
   - 예상: Design JSON에서 Terminal Goal 로드 → 정상 출력

4. **Resume 시나리오**
   - 예상: 기존 Design JSON에서 Terminal Goal 로드

5. **Terminal Goal 생성 실패 시나리오 (NEW)**
   - 예상: 3번 재시도 후 실패 → `sys.exit(1)`로 프로그램 종료
   - 로그 출력:
     ```
     [Attempt 1/3] Terminal Goal generation failed: ...
     Retrying...
     [Attempt 2/3] Terminal Goal generation failed: ...
     Retrying...
     [Attempt 3/3] Terminal Goal generation failed: ...
     [FATAL] All 3 attempts failed.
     ```
