# 논리/상식 데이터셋 추가 계획

## 개요

논리(Logical)와 상식(Commonsense) 도메인을 신규로 추가하고,
지정된 학습 데이터셋 + In-Domain 테스트셋 + OOD 평가셋을 현재 파이프라인에 통합한다.

## 대상 데이터셋 범위

### 학습 데이터셋 (Training)

**논리**
- ReClor (Logical)

**상식**
- ARC-c (AI2 ARC Challenge)
- StrategyQA

### 평가 데이터셋 (Evaluation)

**In-Domain**
- 위 학습 데이터셋의 테스트 셋

**OOD**
- 논리: BBH, ANLI-A2, ANLI-A3
- 상식: OpenBookQA

## 데이터 포맷 및 프롬프트 정책

- JSON 스키마는 기존과 동일:
  - `instruction`, `input`, `output`
- Train/Test 모두 동일한 `instruction` 사용
- 답안 포맷: `\\boxed{...}` 고정

**데이터셋별 instruction (Train/Test 동일)**
- ReClor:
  - "You are a logical reasoning assistant. Read the passage and question, then select the correct option (A, B, C, or D). Your final answer MUST be a single letter within \\boxed{}."
- ARC-c:
  - "You are a helpful commonsense science assistant. Solve the problem and select the correct option (A, B, C, or D). Your final answer MUST be a single letter within \\boxed{}."
- StrategyQA:
  - "You are a helpful commonsense reasoning assistant. Answer the question with Yes or No based on reliable commonsense knowledge. Your final answer MUST be \\boxed{Yes} or \\boxed{No}."
- OpenBookQA:
  - "You are a helpful science question-answering assistant. Use the given options and choose the best answer (A, B, C, or D). Your final answer MUST be a single letter within \\boxed{}."
- ANLI-A2 / ANLI-A3:
  - "You are a natural language inference assistant. Determine the relationship between the premise and hypothesis. Choose from: A. entailment, B. neutral, C. contradiction. Your final answer MUST be a single letter within \\boxed{}."
- BBH (서브태스크별 템플릿 적용):
  - MCQ 템플릿: "You are a helpful reasoning assistant. Solve the problem and select the correct option (A, B, C, or D). Your final answer MUST be a single letter within \\boxed{}."
  - Boolean 템플릿: "You are a helpful reasoning assistant. Answer the question with Yes or No. Your final answer MUST be \\boxed{Yes} or \\boxed{No}."
  - Text 템플릿: "You are a helpful reasoning assistant. Provide a concise final answer. Your final answer MUST be within \\boxed{}."

**프롬프트/답안 타입 제안**
- ReClor, ARC-c, OpenBookQA: MCQ 형식 (A/B/C/D)
  - `input`: 문제 + 선택지 `A. ...` `B. ...`
  - `output`: `\\boxed{A}` 등
- StrategyQA: Yes/No (BOOLEAN)
  - `output`: `\\boxed{Yes}` 또는 `\\boxed{No}`
- ANLI-A2/A3: 3-way NLI 분류
  - MCQ로 변환: `A. entailment`, `B. neutral`, `C. contradiction`
  - `output`: `\\boxed{A|B|C}`
- BBH: 태스크별 상이 (MCQ/BOOL/TEXT 혼재)
  - 태스크별 answer_type 매핑 테이블 필요

## 구현 계획

### Step 1: 도메인/데이터셋 정의 확정

- 신규 도메인 키 확정 (예: `logical`, `commonsense`)
- 도메인별 디렉토리 구조 확정:
  - `data/<domain>/train/data/*.json`
  - `data/<domain>/eval/data/*.json`
- 데이터셋별 AnswerType 확정 (MCQ/BOOLEAN/TEXT)

### Step 2: 데이터 준비 스크립트 확장

**파일**: `utils/dataset_preparer.py`

- `DATASET_PROMPTS`에 신규 프롬프트 추가
- 신규 처리 함수 추가:
  - `process_reclor()`
  - `process_arc_c()`
  - `process_strategyqa()`
  - `process_openbookqa()`
  - `process_anli(round=R2/R3)`
  - `process_bbh(subtasks=[...])`
- 도메인별 `train_dir`, `eval_dir` 분리 저장
- OOD 데이터는 해당 도메인의 `eval/data`에 저장

### Step 3: 로더/설정 동기화

**도메인/데이터셋 등록**
- `utils/domain_loader.py`
  - `DOMAIN_CONFIG`, `TERMINAL_GOALS` 확장
- `utils/dataset_registry.py`
  - `DOMAIN_CONFIG` 확장
- `config/domains.py`
  - `DOMAIN_CONFIG`, `TRAINING_DATASETS`, `DATASET_TO_DOMAIN`, `TERMINAL_GOALS`
- `config/config.py`
  - 동일한 DOMAIN_CONFIG/TERMINAL_GOALS 반영
- `config/dataset_config.py`
  - 신규 데이터셋 HF 경로/스플릿/answer_type 추가

**AnswerType 매핑 예시**
- ReClor: MCQ
- ARC-c: MCQ
- StrategyQA: BOOLEAN
- ANLI-A2/A3: MCQ (3-way)
- OpenBookQA: MCQ
- BBH: 태스크별 MCQ/BOOLEAN/TEXT

### Step 4: 문서 및 사용법 업데이트

- `README.md`, `USAGE.md`
  - 신규 도메인/데이터셋 실행 예시 추가
  - 데이터 디렉토리 구조 및 평가 셋 설명 업데이트

## 검증 방법

1. `dataset_preparer.py` 실행 후 파일 생성 확인
   - `data/<domain>/train/data/*.json`
   - `data/<domain>/eval/data/*.json`
2. `DomainLoader`로 샘플 로딩 확인
3. `main.py --mode train/eval`로 신규 도메인 실행 확인

## 확인 필요 사항

1. 도메인 키 네이밍 확정 (CLI 옵션에 영향)
2. BBH 포함 범위: 전체 태스크 vs 논리적 추론 관련 서브태스크만
3. ANLI 라벨 표현 방식 (A/B/C vs 텍스트 라벨)
4. HF 데이터셋 ID/스플릿 확인
   - ReClor / StrategyQA / ANLI / OpenBookQA 정확한 HF 경로
