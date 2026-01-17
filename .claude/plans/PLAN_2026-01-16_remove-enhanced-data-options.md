# Plan: Enhanced Data 옵션 제거 및 자동화

## 1. Problem Analysis

### 현재 상태
- `--use-enhanced-data`: Enhanced data 사용 여부를 CLI에서 선택
- `--enhanced-model-suffix`: 어떤 Teacher model의 enhanced data를 사용할지 지정
- 불필요한 복잡성: Enhanced data는 항상 사용하고, suffix는 항상 teacher model

### Root Cause (Five Whys)
1. **Why** CLI 옵션이 필요한가? → 다른 teacher model의 enhanced data를 사용할 수 있게 하려고
2. **Why** 다른 teacher model의 data가 필요한가? → 실제로는 필요 없음, 항상 현재 teacher model 사용
3. **Why** 옵션을 분리했나? → 유연성을 위해, 하지만 실제 사용 패턴에서 불필요
4. **Why** 항상 enhanced data인가? → ID-MAS의 핵심 기능이며 품질 향상에 필수
5. **Why** suffix가 teacher model인가? → Enhanced data 생성 시 teacher model이 Instructional Goal과 Task Analysis를 생성하기 때문

### 결론
- `--use-enhanced-data`, `--enhanced-model-suffix` 옵션 삭제
- Enhanced data는 항상 사용
- Enhanced model suffix는 항상 teacher model에서 자동 파생

---

## 2. Requirements Specification

### Acceptance Criteria

| ID | Criteria | Test Method |
|----|----------|-------------|
| AC1 | `--use-enhanced-data` CLI 옵션이 제거됨 | `python main.py --help`에서 옵션 없음 확인 |
| AC2 | `--enhanced-model-suffix` CLI 옵션이 제거됨 | `python main.py --help`에서 옵션 없음 확인 |
| AC3 | Enhanced data가 없으면 자동 생성됨 | 새 teacher model로 실행 시 enhanced data 생성 확인 |
| AC4 | 기존 enhanced data 있으면 재사용 | 동일 teacher model로 재실행 시 기존 파일 사용 확인 |
| AC5 | README.md에서 관련 문서 제거됨 | 문서 검토 |
| AC6 | 기존 학습 워크플로우가 정상 동작 | `python main.py --mode train --domain math --train-dataset gsm8k` 성공 |

---

## 3. Architecture Design

### 변경 전 흐름
```
CLI Args
  ├── --use-enhanced-data (optional)
  ├── --enhanced-model-suffix (required if use-enhanced-data)
  └── --teacher-model
         │
         ▼
IDMASPipeline.__init__()
  ├── use_enhanced_data: bool (from args)
  └── enhanced_model_suffix: str (from args)
         │
         ▼
run_learning_phase()
  ├── if use_enhanced_data and enhanced_model_suffix:
  │     → load_enhanced_training_data()
  └── else:
        → load_training_data()
```

### 변경 후 흐름
```
CLI Args
  └── --teacher-model
         │
         ▼
IDMASPipeline.__init__()
  └── teacher_model_name: str
         │
         ▼
run() / run_learning_phase()
  │
  ├── model_suffix = get_model_short_name(teacher_model_name)
  ├── enhanced_path = {dataset}_train_ID-MAS_{model_suffix}.json
  │
  ├── if enhanced_path.exists():
  │     → load_enhanced_training_data(model_suffix)
  └── else:
        → generate_enhanced_data() → load_enhanced_training_data()
```

### 선택 근거
- **단순화**: CLI 옵션 2개 제거로 사용자 경험 개선
- **일관성**: Enhanced data는 항상 teacher model과 연동
- **자동화**: 사용자가 수동으로 옵션을 지정할 필요 없음

---

## 4. Task Decomposition

### Task List

| ID | Task | Dependencies | Files | Priority |
|----|------|--------------|-------|----------|
| T1 | CLI 옵션 정의 삭제 | - | main.py | P0 |
| T2 | CLI 검증 로직 삭제 | T1 | main.py | P0 |
| T3 | IDMASPipeline.__init__() 파라미터 정리 | T1 | main.py | P0 |
| T4 | IDMASPipeline 인스턴스 생성 코드 정리 | T3 | main.py | P0 |
| T5 | run() 메서드에서 enhanced data 로직 단순화 | T4 | main.py | P0 |
| T6 | run_learning_phase() 로직 단순화 | T5 | main.py | P1 |
| T7 | README.md 옵션 설명 제거 | - | README.md | P1 |
| T8 | 통합 테스트 (--help 확인) | T1-T6 | - | P2 |

### Dependency Graph
```
T1 ──► T2 ──► T3 ──► T4 ──► T5 ──► T6 ──► T8
                                           ▲
T7 ────────────────────────────────────────┘
```

---

## 5. Implementation Strategy

### 5.1 Code Writing Method
순차적 수정 (TDD 불필요 - 단순 삭제/수정 작업)

### 5.2 Per-Task Implementation Guide

#### T1: CLI 옵션 정의 삭제
**파일**: main.py (라인 962-973)
**삭제할 코드**:
```python
parser.add_argument(
    "--use-enhanced-data",
    action="store_true",
    dest="use_enhanced_data",
    help="Use enhanced training data..."
)
parser.add_argument(
    "--enhanced-model-suffix",
    type=str,
    default=None,
    dest="enhanced_model_suffix",
    help="Model suffix for enhanced data file..."
)
```

#### T2: CLI 검증 로직 삭제
**파일**: main.py (라인 1048-1050)
**삭제할 코드**:
```python
if args.use_enhanced_data and not args.enhanced_model_suffix:
    parser.error("--enhanced-model-suffix is required when using --use-enhanced-data")
```

#### T3: IDMASPipeline.__init__() 파라미터 정리
**파일**: main.py (라인 73-102)
**삭제할 파라미터**:
- `use_enhanced_data: bool = False`
- `enhanced_model_suffix: Optional[str] = None`

**삭제할 인스턴스 변수**:
- `self.use_enhanced_data`
- `self.enhanced_model_suffix`

#### T4: IDMASPipeline 인스턴스 생성 코드 정리
**파일**: main.py (라인 734-759)
**삭제할 코드**:
```python
print(f"Enhanced Data: Yes (suffix: {args.enhanced_model_suffix})")
...
use_enhanced_data=getattr(args, 'use_enhanced_data', False),
enhanced_model_suffix=getattr(args, 'enhanced_model_suffix', None)
```

#### T5: run() 메서드에서 enhanced data 로직 단순화
**파일**: main.py (라인 796-809)
**변경 전**:
```python
if not getattr(args, 'use_enhanced_data', False):
    model_suffix = get_model_short_name(pipeline.teacher_model_name)
    enhanced_path = pipeline.raw_data_dir / f"{pipeline.train_dataset}_train_ID-MAS_{model_suffix}.json"
    if enhanced_path.exists() and args.resume:
        print(f"\n[Enhanced Data] Using existing enhanced data...")
        pipeline.use_enhanced_data = True
        pipeline.enhanced_model_suffix = model_suffix
    else:
        pipeline.generate_enhanced_data(design_result)
```

**변경 후**:
- 조건문 제거, 항상 enhanced data 체크 및 생성 로직 실행
- `self.use_enhanced_data`, `self.enhanced_model_suffix` 인스턴스 변수 대신 로컬 변수 사용

#### T6: run_learning_phase() 로직 단순화
**파일**: main.py (라인 410-426)
**변경 전**:
```python
if self.use_enhanced_data and self.enhanced_model_suffix:
    questions = self.loader.load_enhanced_training_data(...)
else:
    questions = self.loader.load_training_data(...)
```

**변경 후**:
```python
model_suffix = get_model_short_name(self.teacher_model_name)
questions = self.loader.load_enhanced_training_data(
    dataset=self.train_dataset,
    model_suffix=model_suffix,
    ...
)
```

#### T7: README.md 옵션 설명 제거
**파일**: README.md
**삭제할 내용**:
1. 학습 모드 옵션 테이블에서 `--use-enhanced-data`, `--enhanced-model-suffix` 행
2. Enhanced data 사용 예시 코드 블록
3. dataset_enhancer.py 섹션 (선택적 - 유지해도 됨)

---

## 6. Quality Gates

### Phase 완료 체크리스트

- [ ] T1: `--use-enhanced-data` 옵션 삭제됨
- [ ] T2: `--enhanced-model-suffix` 옵션 삭제됨
- [ ] T3: 검증 로직 삭제됨
- [ ] T4: IDMASPipeline 파라미터 정리됨
- [ ] T5: 인스턴스 생성 코드 정리됨
- [ ] T6: run() 메서드 단순화됨
- [ ] T7: run_learning_phase() 단순화됨
- [ ] T8: README.md 업데이트됨
- [ ] 통합 테스트: `python main.py --help` 정상
- [ ] 통합 테스트: 기존 워크플로우 정상 동작

---

## 7. Estimated Changes Summary

| 파일 | 삭제 | 수정 | 추가 |
|------|------|------|------|
| main.py | ~50줄 | ~20줄 | ~5줄 |
| README.md | ~15줄 | - | - |

**총 변경**: 약 90줄 (주로 삭제)
