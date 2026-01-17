# Test Item 코드 삭제 계획

**생성일**: 2026-01-15
**목표**: 학습 과정에서 사용하지 않는 Test Item 관련 코드 전체 삭제

---

## 1. Problem Analysis

### 현재 상황
- `TestItemDevelopment` 클래스가 존재하지만 **실제 파이프라인에서 사용되지 않음**
- `main.py`에서 인스턴스는 생성되나 (`self.test_dev`) 어디서도 호출되지 않음
- 불필요한 코드 약 140줄이 유지되고 있음

### Root Cause (Five Whys)
1. Why 미사용? → 설계 파이프라인에서 Test Item 단계가 생략됨
2. Why 생략? → Rubric Development가 직접 Performance Objectives에서 생성
3. Why 제거 안함? → 초기 설계 시 포함되었으나 정리되지 않음

### 리스크 평가: **LOW RISK**
- 현재 미사용 코드이므로 삭제해도 런타임 영향 없음

---

## 2. Requirements Specification

### Acceptance Criteria
- [ ] AC1: `design_modules/test.py` 파일 삭제됨
- [ ] AC2: `TEST_ITEM_DEVELOPMENT_PROMPT` 프롬프트 삭제됨
- [ ] AC3: 모든 import 및 참조 제거됨
- [ ] AC4: `__init__.py` docstring에서 Step 3 설명 제거됨
- [ ] AC5: 삭제 후 기존 기능 정상 동작 (import 에러 없음)

---

## 3. Task Decomposition

| ID | Task | 파일 | 삭제 내용 | Priority |
|----|------|------|----------|----------|
| T1 | test.py 파일 삭제 | `design_modules/test.py` | 전체 파일 (107줄) | P0 |
| T2 | 프롬프트 삭제 | `prompts/design_prompts.py` | 라인 130-163 (34줄) | P0 |
| T3 | __init__.py 정리 | `design_modules/__init__.py` | 라인 8, 14, 21 | P0 |
| T4 | main.py 정리 | `main.py` | 라인 37, 120 | P0 |
| T5 | 검증 | - | import 테스트 | P1 |

### Dependency Graph
```
T1 ──┐
T2 ──┼──► T5 (검증)
T3 ──┤
T4 ──┘
```

---

## 4. 상세 변경 사항

### T1: design_modules/test.py 삭제
```bash
rm design_modules/test.py
```

### T2: prompts/design_prompts.py 수정
**삭제 대상** (라인 130-163):
```python
# ==============================================================================
# 5단계: Test Item 개발
# ==============================================================================

TEST_ITEM_DEVELOPMENT_PROMPT = """..."""
```

### T3: design_modules/__init__.py 수정

**Before:**
```python
"""
Design Modules for ID-MAS Instructional Design Phase

Design Phase 단계:
- Step 0: Terminal Goal Generation (TerminalGoalGenerator)
- Step 1: Instructional Analysis (InstructionalAnalysis)
- Step 2: Performance Objectives (PerformanceObjectives)
- Step 3: Test Item Development (TestItemDevelopment)  # 삭제
- Step 4: Rubric Development (RubricDevelopment)
"""
from design_modules.test import TestItemDevelopment  # 삭제

__all__ = [
    ...
    "TestItemDevelopment",  # 삭제
    ...
]
```

**After:**
```python
"""
Design Modules for ID-MAS Instructional Design Phase

Design Phase 단계:
- Step 0: Terminal Goal Generation (TerminalGoalGenerator)
- Step 1: Instructional Analysis (InstructionalAnalysis)
- Step 2: Performance Objectives (PerformanceObjectives)
- Step 3: Rubric Development (RubricDevelopment)
"""

__all__ = [
    "TerminalGoalGenerator",
    "InstructionalAnalysis",
    "PerformanceObjectives",
    "RubricDevelopment",
]
```

### T4: main.py 수정

**삭제할 라인:**
- 라인 37: `from design_modules.test import TestItemDevelopment`
- 라인 120: `self.test_dev = TestItemDevelopment(teacher_config)`

---

## 5. Implementation Strategy

### 실행 순서
1. T1~T4는 **병렬 실행 가능** (독립적)
2. T5는 T1~T4 완료 후 순차 실행

### 검증 방법 (T5)
```bash
# Python import 테스트
python -c "from design_modules import *; from main import IDMASPipeline; print('OK')"

# 또는 main.py 직접 실행 (--help)
python main.py --help
```

---

## 6. Quality Gates

### Pre-implementation
- [x] 삭제 대상 파일/라인 확인 완료
- [x] 현재 미사용 상태 확인 완료
- [x] 의존성 분석 완료

### Post-implementation
- [ ] test.py 파일 삭제됨
- [ ] 프롬프트 삭제됨
- [ ] 모든 import 제거됨
- [ ] Python import 테스트 통과
- [ ] main.py --help 정상 실행

---

## 7. 영향 받는 파일 요약

| 파일 | 변경 유형 | 삭제 라인 수 |
|------|----------|-------------|
| `design_modules/test.py` | 파일 삭제 | 107줄 |
| `prompts/design_prompts.py` | 부분 삭제 | 34줄 |
| `design_modules/__init__.py` | 수정 | 3줄 |
| `main.py` | 수정 | 2줄 |
| **Total** | | **~146줄** |

---

## 8. Rollback Plan

문제 발생 시 git으로 복구:
```bash
git checkout -- design_modules/test.py design_modules/__init__.py main.py prompts/design_prompts.py
```
