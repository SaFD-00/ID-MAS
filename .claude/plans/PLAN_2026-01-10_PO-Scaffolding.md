# PO 미충족 시 Scaffolding 계속 진행 계획

## 개요

**목표**: 학생 모델이 정답은 맞췄지만 수행목표(PO)를 모두 충족하지 못한 경우, scaffolding을 계속 진행하도록 로직 변경

**도메인**: coding
**영향 범위**: `learning_loop/graph/nodes.py`
**예상 변경량**: ~20 lines

---

## 요구사항 정의

### 현재 동작
```
정답 맞춤 OR 모든 PO 충족 → 성공 (Case A)
```

### 변경 후 동작
```
정답 맞춤 AND 모든 PO 충족 → 성공 (Case A)
정답 맞춤 BUT PO 미충족 → Scaffolding 계속
max_iterations 도달 시 PO 미충족 → 실패 (Case A-Failed, reconstruction)
```

### 결정 사항
| 항목 | 결정 |
|------|------|
| 마감 처리 | 실패로 처리 (Case A-Failed) + reconstruction |
| 피드백 방식 | 기존 방식 유지 (전체 PO 평가) |

---

## 구현 계획

### Step 1: 성공 조건 변경

**파일**: `learning_loop/graph/nodes.py`
**위치**: Line 224-228

**Before**:
```python
# Success condition: correct answer OR all POs satisfied
if is_correct or all_satisfied:
    print(f"    -> Correct on iteration {iteration}! (PO satisfied: {all_satisfied})")
    is_correct = True
    break
```

**After**:
```python
# Success condition: correct answer AND all POs satisfied
if is_correct and all_satisfied:
    print(f"    -> Correct on iteration {iteration}! (PO satisfied: True)")
    break
elif is_correct and not all_satisfied:
    # 정답은 맞았지만 PO 미충족 - scaffolding 계속
    print(f"    -> Answer correct but PO not satisfied on iteration {iteration}. Continuing scaffolding...")
```

**완료 기준**:
- [x] `is_correct and all_satisfied` 조건으로 변경
- [x] PO 미충족 시 scaffolding 계속 진행

---

### Step 2: max_iterations 도달 시 처리 로직 수정

**파일**: `learning_loop/graph/nodes.py`
**위치**: Line 234-289 (루프 종료 후 결과 처리)

**현재 로직**: `is_correct`가 True면 무조건 성공 처리

**변경 로직**:
- `is_correct and all_satisfied` → 성공 (Case A)
- 그 외 (정답 미맞춤 또는 PO 미충족) → 실패 (Case A-Failed) + reconstruction

**완료 기준**:
- [x] max_iterations 도달 시 PO 충족 여부 확인
- [x] PO 미충족 시 reconstruction 수행

---

### Step 3: 로그 메시지 개선

**목적**: 정답 맞춤 vs PO 충족 상태를 명확히 구분

**변경 내용**:
```python
# 성공 시
print(f"    -> Success on iteration {iteration}! (Correct: True, PO satisfied: True)")

# 정답 맞았지만 PO 미충족 시
print(f"    -> Answer correct but PO not satisfied on iteration {iteration}. Continuing...")

# 최종 실패 시
print(f"    -> Failed after {max_iterations} iterations. (Last answer correct: {is_correct}, PO satisfied: {all_satisfied})")
```

**완료 기준**:
- [x] 상태별 명확한 로그 메시지

---

### Step 4: 마지막 정답 iteration 추적 (선택)

**목적**: 정답을 맞췄던 마지막 iteration 정보 보존

**변경 내용**:
- `last_correct_iteration`, `last_correct_response` 변수 추가
- reconstruction 시 참조 가능하도록 정보 전달

**완료 기준**:
- [x] 정답 맞춘 iteration 정보 추적
- [x] reconstruction 시 활용 가능

---

## 테스트 계획

### Unit Test
1. `is_correct=True, all_satisfied=True` → 성공 (Case A)
2. `is_correct=True, all_satisfied=False` → Scaffolding 계속
3. `is_correct=False, all_satisfied=True` → 성공 (Case A)
4. `is_correct=False, all_satisfied=False` → Scaffolding 계속
5. max_iterations 도달 + PO 미충족 → 실패 (Case A-Failed)

### Integration Test
- 실제 파이프라인 실행하여 로그 확인
- `PO satisfied: False` 케이스에서 scaffolding 진행 확인

---

## 리스크 및 고려사항

### 리스크
1. **학습 시간 증가**: 정답을 맞춰도 scaffolding이 계속되어 처리 시간 증가
2. **무한 루프 가능성**: 특정 PO가 절대 충족되지 않는 경우 max_iterations까지 진행

### 대응 방안
1. `max_iterations` 설정으로 시간 제한
2. 로그 모니터링을 통해 문제 케이스 식별

---

## 체크리스트

- [ ] Step 1: 성공 조건 변경 (`OR` → `AND`)
- [ ] Step 2: max_iterations 도달 시 처리 로직 수정
- [ ] Step 3: 로그 메시지 개선
- [ ] Step 4: 마지막 정답 iteration 추적 (선택)
- [ ] 테스트 실행 및 검증
