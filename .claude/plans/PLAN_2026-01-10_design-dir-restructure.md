# Plan: instructional-design 경로 구조 변경

## Overview

**목표**: `data/{domain}/train/instructional-design/` → `data/{domain}/train/{teacher_model}/instructional-design/`

**배경**: teacher 모델별로 생성되는 설계 결과물을 분리하여 관리

## 영향 범위 분석

### 발견된 이슈: 코드 중복

**문제**: `get_design_output_dir()` 함수가 두 곳에 중복 정의됨
- `config/config.py:105-121` ← `main.py`에서 import
- `config/paths.py:12-28` ← `config/__init__.py`에서 re-export

**해결**: `config/config.py`에서 함수 정의 제거, `config/paths.py`만 유지
- `main.py`의 import를 `config.config` → `config`로 변경하여 `config/__init__.py` 경유

### 수정 필요 파일

| 파일 | 변경 유형 | 우선순위 |
|------|----------|---------|
| `config/paths.py` | 함수 수정 (primary) | 높음 |
| `config/config.py` | 중복 함수 제거 + get_domain_data_dirs 수정 | 높음 |
| `main.py` | 함수 호출 수정 | 높음 |
| `README.md` | 문서 업데이트 | 낮음 |
| `ARCHITECTURE.md` | 문서 업데이트 | 낮음 |

### 변경 사항 상세

#### 1. config/config.py (라인 105-121, 370)

**현재 코드:**
```python
def get_design_output_dir(domain: str) -> Path:
    design_dir = DATA_DIR / domain.lower() / "train" / "instructional-design"
    ...
```

**변경 후:**
```python
def get_design_output_dir(domain: str, teacher_model_name: str = None) -> Path:
    from config.config import DEFAULT_TEACHER_MODEL
    teacher_short = get_model_short_name(teacher_model_name) if teacher_model_name else get_model_short_name(DEFAULT_TEACHER_MODEL)
    design_dir = DATA_DIR / domain.lower() / "train" / teacher_short / "instructional-design"
    ...
```

**get_domain_data_dirs() 함수 내 design_dir (라인 370):**
```python
# 현재
"design_dir": DATA_DIR / domain / "train" / "instructional-design",

# 변경 후
"design_dir": DATA_DIR / domain / "train" / teacher_short / "instructional-design",
```

#### 2. config/paths.py (라인 12-28)

**현재 코드:**
```python
def get_design_output_dir(domain: str) -> Path:
    design_dir = DATA_DIR / domain.lower() / "train" / "instructional-design"
    ...
```

**변경 후:**
```python
def get_design_output_dir(domain: str, teacher_model_name: str = None) -> Path:
    from config.models import DEFAULT_TEACHER_MODEL
    teacher_short = get_model_short_name(teacher_model_name) if teacher_model_name else get_model_short_name(DEFAULT_TEACHER_MODEL)
    design_dir = DATA_DIR / domain.lower() / "train" / teacher_short / "instructional-design"
    ...
```

#### 3. main.py (라인 565)

**현재 코드:**
```python
design_dir = get_design_output_dir(pipeline.domain)
```

**변경 후:**
```python
design_dir = get_design_output_dir(pipeline.domain, pipeline.teacher_model_name)
```

#### 4. 문서 업데이트

**README.md (라인 332 주변):**
- `instructional-design/` 경로 설명을 teacher 모델 하위로 변경

**ARCHITECTURE.md (라인 436 주변):**
- 동일하게 경로 설명 업데이트

## 구현 단계

### Step 1: config/paths.py 수정 (Primary)
1. `get_design_output_dir()` 함수에 `teacher_model_name` 파라미터 추가
2. teacher 모델 short name 계산 로직 추가
3. 경로 구성 변경: `{domain}/train/{teacher_short}/instructional-design/`
4. `get_domain_data_dirs()` 함수의 `design_dir` 경로도 동일하게 수정

### Step 2: config/config.py 정리
1. 중복된 `get_design_output_dir()` 함수 제거 (config/paths.py에서 import)
2. `get_domain_data_dirs()` 함수의 `design_dir` 경로 수정
   - 이미 `teacher_short` 변수가 존재하므로 경로만 수정

### Step 3: main.py 수정
1. `get_design_output_dir()` 호출 시 `teacher_model_name` 전달
2. 레거시 경로 마이그레이션 로직 업데이트 (기존 flat 경로 → 새 구조)

### Step 4: 문서 업데이트
1. README.md 경로 설명 업데이트
2. ARCHITECTURE.md 경로 설명 업데이트

## 위험 요소 및 대응

### 1. 기존 데이터 마이그레이션
- **위험**: 기존 `data/{domain}/train/instructional-design/` 디렉토리에 있는 파일들
- **대응**: 레거시 경로 확인 및 자동 마이그레이션 로직 추가 (main.py에 이미 유사 패턴 존재)

### 2. 하위 호환성
- **위험**: 다른 곳에서 `get_design_output_dir()`를 teacher_model 없이 호출
- **대응**: `teacher_model_name` 파라미터에 기본값 설정 (`None` → DEFAULT_TEACHER_MODEL 사용)

## 검증 방법

1. **단위 테스트**: 함수별 경로 반환값 확인
2. **통합 테스트**: 전체 파이프라인 실행하여 파일 저장 경로 확인
3. **레거시 마이그레이션 테스트**: 기존 경로 파일 자동 이동 확인

## 예상 결과 디렉토리 구조

```
data/
└── math/
    └── train/
        ├── data/                           # 원본 데이터
        ├── gpt-5-2025-08-07/              # GPT-5 teacher 모델
        │   ├── instructional-design/      # 설계 결과
        │   │   ├── math_gsm8k_design.json
        │   │   └── math_math_design.json
        │   └── Qwen2.5-3B-Instruct/        # Student 모델별 학습 결과
        └── Qwen2.5-14B-Instruct/           # 다른 teacher 모델
            ├── instructional-design/
            └── ...
```

## 완료 기준

- [ ] `get_design_output_dir()` 함수가 teacher 모델명을 받아 올바른 경로 반환
- [ ] `get_domain_data_dirs()` 함수의 `design_dir`이 teacher 모델 하위 경로 반환
- [ ] main.py에서 올바른 경로로 설계 파일 저장/로드
- [ ] 레거시 경로 마이그레이션 로직 동작
- [ ] 문서 업데이트 완료
