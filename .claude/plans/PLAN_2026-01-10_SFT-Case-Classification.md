# SFT 케이스 분류 로직 변경 계획

## 개요

**목표**: SFT 케이스(A/A-Failed) 분류 시 **정답 여부(`is_correct`)를 무시**하고 **PO 충족 여부(`all_satisfied`)만** 기준으로 판정하도록 수정

**현재 동작**:
- 성공(Case A): `is_correct AND all_satisfied` (정답 + PO 충족)
- 실패(Case A-Failed): 위 조건 미충족 시

**변경 후 동작**:
- 성공(Case A): `all_satisfied` (PO 충족만)
- 실패(Case A-Failed): PO 미충족 시

## 변경 대상 파일

### 1. learning_loop/graph/nodes.py (핵심 로직)

| 라인 | 현재 코드 | 변경 코드 |
|------|----------|----------|
| 232 | `if is_correct and all_satisfied:` | `if all_satisfied:` |
| 233-240 | 성공/조건부 메시지 출력 | PO 충족 여부만 기준으로 메시지 수정 |
| 248 | `if is_correct and all_satisfied:` | `if all_satisfied:` |
| 271 | `failure_reason = "answer_incorrect" if not is_correct else "po_not_satisfied"` | `failure_reason = "po_not_satisfied"` (단일화) |

**상세 변경 사항**:

```python
# L232-241: 성공 조건 및 콘솔 출력 변경
# Before:
if is_correct and all_satisfied:
    print(f"    -> Success on iteration {iteration}! (Correct: True, PO satisfied: True)")
    break
elif is_correct and not all_satisfied:
    print(f"    -> Answer correct but PO not satisfied on iteration {iteration}. Continuing scaffolding...")
elif not is_correct and all_satisfied:
    print(f"    -> PO satisfied but answer incorrect on iteration {iteration}. Continuing scaffolding...")

# After:
if all_satisfied:
    print(f"    -> Success on iteration {iteration}! (PO satisfied: True)")
    break
else:
    print(f"    -> PO not satisfied on iteration {iteration}. Continuing scaffolding...")
```

```python
# L248: 결과 빌드 조건 변경
# Before:
if is_correct and all_satisfied:

# After:
if all_satisfied:
```

```python
# L270-272: 실패 이유 단순화
# Before:
failure_reason = "answer_incorrect" if not is_correct else "po_not_satisfied"
print(f"    -> Failed after {max_iterations} iterations. (Reason: {failure_reason}, Last correct iteration: {last_correct_iteration}) Reconstructing...")

# After:
failure_reason = "po_not_satisfied"
print(f"    -> Failed after {max_iterations} iterations. (Reason: {failure_reason}) Reconstructing...")
```

### 2. README.md (문서 업데이트)

**L402-411** SFT Case 분류 섹션 수정:

```markdown
# Before:
| `A` | Iterative Scaffolding 성공 (정답 AND PO 충족) | 학생 모델 응답 (1회 또는 다중 시도) |
| `A-Failed` | 5회 실패 후 재구성 (오답 또는 PO 미충족) | AI 기반 대화 분석 후 재구성된 응답 |

**성공 조건**: 정답을 맞추고(`is_correct=True`) **동시에** 모든 수행목표(PO)가 충족되어야(`all_satisfied=True`) Case A로 처리됩니다.

# After:
| `A` | Iterative Scaffolding 성공 (PO 충족) | 학생 모델 응답 (1회 또는 다중 시도) |
| `A-Failed` | 5회 실패 후 재구성 (PO 미충족) | AI 기반 대화 분석 후 재구성된 응답 |

**성공 조건**: 모든 수행목표(PO)가 충족되면(`all_satisfied=True`) Case A로 처리됩니다. 정답 여부(`is_correct`)는 SFT 케이스 분류에 영향을 주지 않습니다.
```

### 3. ARCHITECTURE.md (문서 업데이트)

**L275, L330-345, L353-356** 관련 섹션 수정:

```markdown
# L275: 플로우차트 노드 레이블 변경
S_EVAL{정답 AND PO 충족?} → S_EVAL{PO 충족?}

# L330-343: 프로세스 설명 변경
[Teacher 평가] → 정답 여부 + 수행목표(PO) 충족 여부 판정
    ↓
[정답 AND PO 충족?]
→
[Teacher 평가] → 수행목표(PO) 충족 여부 판정
    ↓
[PO 충족?]

# L345-347: 성공 조건 설명 변경
**성공 조건**: 정답을 맞추고(`is_correct=True`) **동시에** 모든 수행목표(PO)가 충족되어야(`all_satisfied=True`) 성공(Case A)으로 처리됩니다.
→
**성공 조건**: 모든 수행목표(PO)가 충족되면(`all_satisfied=True`) 성공(Case A)으로 처리됩니다.

# L353-356: SFT 데이터 생성 조건 테이블 변경
| 정답 AND PO 충족 | Case A | 학생 응답 사용 |
| max_iterations 후 실패 (오답 또는 PO 미충족) | Case A-Failed | Reconstruction 응답 사용 |
→
| PO 충족 | Case A | 학생 응답 사용 |
| max_iterations 후 PO 미충족 | Case A-Failed | Reconstruction 응답 사용 |
```

## 영향 범위

### 변경되는 동작

1. **정답이지만 PO 미충족**: 기존 → 실패(A-Failed) / 변경 후 → 실패(A-Failed) (동일)
2. **오답이지만 PO 충족**: 기존 → 실패(A-Failed) / 변경 후 → **성공(A)** (변경)
3. **정답 + PO 충족**: 기존 → 성공(A) / 변경 후 → 성공(A) (동일)
4. **오답 + PO 미충족**: 기존 → 실패(A-Failed) / 변경 후 → 실패(A-Failed) (동일)

### 변경되지 않는 부분

- `iterative_scaffolding.iterations[].is_correct`: 여전히 기록됨 (로그용)
- `iterative_scaffolding.iterations[].predicted_answer`: 여전히 기록됨 (로그용)
- `last_correct_iteration`, `last_correct_response`: 제거 가능 (더 이상 사용되지 않음)

## 실행 순서

1. [learning_loop/graph/nodes.py](learning_loop/graph/nodes.py) 수정
2. [README.md](README.md) 업데이트
3. [ARCHITECTURE.md](ARCHITECTURE.md) 업데이트
4. Git commit 및 push

## 검증 방법

변경 후 기존 로그와 동일한 데이터로 테스트하여:
- PO 충족 시 정답 여부와 관계없이 Case A로 분류되는지 확인
- PO 미충족 시 Case A-Failed로 분류되는지 확인
