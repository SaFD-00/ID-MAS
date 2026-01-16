# ID-MAS: Instructional Design Multi-Agent System

LLM 학습을 위한 Dick & Carey 모델 기반 교수 설계 시스템

## 개요

ID-MAS는 교수 설계(Instructional Design) 이론을 LLM 학습에 적용한 Multi-Agent 시스템입니다.

**핵심 특징:**
- **데이터셋별 분리 학습**: GSM8K, MATH 각각 고유한 Instructional Goal로 독립 학습
- **Iterative Scaffolding Pipeline**: Performance Objectives 기반 평가 + Socratic 질문을 통한 반복 학습 (최대 5회)
- **LangGraph 기반 워크플로우**: StateGraph 상태 관리, 체크포인트 기반 Resume 지원

## 지원 도메인 및 데이터셋

| 도메인 | 학습 데이터셋 | 평가 데이터셋 |
|--------|--------------|---------------|
| **Math** | gsm8k, math | gsm8k, math, svamp, asdiv, mawps |
| **Logical** | reclor | reclor, anli_r2, anli_r3, bbh |
| **Commonsense** | arc_c | arc_c, strategyqa, openbookqa |

## 지원 모델

### Teacher 모델 (설계 및 평가)

| 유형 | 모델 | 비고 |
|------|------|------|
| OpenAI | gpt-5-2025-08-07 | 기본값 (API) |
| 로컬 | meta-llama/Llama-3.1-8B-Instruct | HuggingFace |
| 로컬 | meta-llama/Llama-3.1-70B-Instruct | HuggingFace |
| 로컬 | meta-llama/Llama-3.2-3B-Instruct | HuggingFace |
| 로컬 | meta-llama/Llama-3.3-70B-Instruct | HuggingFace |
| 로컬 | Qwen/Qwen2.5-3B-Instruct | HuggingFace |
| 로컬 | Qwen/Qwen2.5-7B-Instruct | HuggingFace |
| 로컬 | Qwen/Qwen2.5-14B-Instruct | HuggingFace |
| 로컬 | Qwen/Qwen2.5-72B-Instruct | HuggingFace |
| 로컬 | Qwen/Qwen3-4B-Instruct-2507 | HuggingFace |

### Student 모델 (학습 대상)

| 모델 | 비고 |
|------|------|
| Qwen/Qwen2.5-3B-Instruct | 기본값 |
| Qwen/Qwen2.5-7B-Instruct | |
| Qwen/Qwen2.5-14B-Instruct | |
| Qwen/Qwen2.5-72B-Instruct | |
| Qwen/Qwen3-4B-Instruct-2507 | |
| meta-llama/Llama-3.1-8B-Instruct | |
| meta-llama/Llama-3.1-70B-Instruct | |
| meta-llama/Llama-3.2-3B-Instruct | |
| meta-llama/Llama-3.3-70B-Instruct | |

## 빠른 시작

```bash
# 1. 환경 설정
conda create -n ID-MAS python=3.11 -y
conda activate ID-MAS
pip install -r requirements.txt

# 2. 환경변수 설정
cp .env.example .env
# .env 파일을 열어 OPENAI_API_KEY, HF_TOKEN 설정

# 3. 데이터 준비
python -m utils.dataset_preparer

# 4. 샘플 추출 (Instructional Goal 생성용)
python -m utils.sample_extractor

# 5. 학습 실행
python main.py --mode train --domain math --train-dataset gsm8k

# 6. 평가 실행
python main.py --mode eval --method baseline --domain math --eval-dataset gsm8k
```

## 데이터 준비 유틸리티

### dataset_preparer.py

HuggingFace에서 데이터셋을 다운로드하고 통일된 JSON 형식으로 변환합니다.

```bash
# 모든 데이터셋 다운로드 및 전처리
python -m utils.dataset_preparer
```

**지원 데이터셋:**
- **Math**: gsm8k, math, svamp, asdiv, mawps
- **Logical**: reclor (로컬 파일), anli_r2, anli_r3, bbh (9개 서브태스크 통합)
- **Commonsense**: arc_c, strategyqa, openbookqa

**출력 형식:**
```json
{
  "instruction": "You are a helpful math assistant...",
  "input": "문제 텍스트",
  "output": "The answer is \\boxed{42}",
  "metadata": {}
}
```

**특이사항:**
- ReClor: `.claude/references/data/reclor_data/` 로컬 파일 사용
- BBH: 9개 서브태스크가 `bbh_test.json`으로 통합, `metadata.subtask`로 구분
- GSM8K/MATH: reasoning 버전도 함께 생성 (`*_reasoning.json`)

### sample_extractor.py

Instructional Goal 동적 생성을 위한 대표 샘플을 추출합니다.

```bash
# 모든 데이터셋에서 샘플 추출
python -m utils.sample_extractor

# 특정 데이터셋만 추출
python -m utils.sample_extractor --domain math --dataset gsm8k --num-samples 20

# 샘플링 전략 지정
python -m utils.sample_extractor --strategy stratified
```

**샘플링 전략:**

| 전략 | 설명 | 사용 예시 |
|------|------|----------|
| `diverse` (기본) | 문제 길이 기반 균등 분배 (short/medium/long) | GSM8K, ReClor |
| `stratified` | 메타데이터(type/level) 기반 층화 샘플링 | MATH (algebra, geometry 등) |
| `random` | 단순 랜덤 샘플링 | 빠른 테스트용 |

**출력 경로:**
```
data/{domain}/train/data/{dataset}_samples.json
```

## CLI 사용법

### 학습 모드 (--mode train)

```bash
# 기본 학습
python main.py --mode train --domain math --train-dataset gsm8k

# 다른 학생 모델로 학습
python main.py --mode train --domain math --train-dataset gsm8k \
    --student-model Qwen/Qwen2.5-7B-Instruct

# 로컬 Teacher 모델 사용
python main.py --mode train --domain math --train-dataset gsm8k \
    --teacher-model Qwen/Qwen2.5-72B-Instruct

# 처음부터 새로 학습 (체크포인트 무시 + Instructional Goal 재생성)
python main.py --mode train --domain math --train-dataset gsm8k --resume False

# 설계 단계 강제 재실행
python main.py --mode train --domain math --train-dataset gsm8k --run-design
```

**학습 모드 옵션:**

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--domain` | 도메인 (math, logical, commonsense) | 필수 |
| `--train-dataset` | 학습 데이터셋 | 필수 |
| `--student-model` | 학생 모델 | Qwen/Qwen2.5-3B-Instruct |
| `--teacher-model` | 교사 모델 | gpt-5-2025-08-07 |
| `--resume` | 체크포인트에서 재개 (True/False) | True |
| `--run-design` | 설계 단계 강제 재실행 | False |

### 평가 모드 (--mode eval)

```bash
# Baseline 평가
python main.py --mode eval --method baseline \
    --domain math --eval-dataset gsm8k

# SFT 평가
python main.py --mode eval --method sft \
    --domain math --eval-dataset gsm8k \
    --student-model Qwen/Qwen2.5-3B-Instruct

# SFT_ID-MAS 평가
python main.py --mode eval --method sft_id-mas \
    --domain math --eval-dataset gsm8k

# Cross-dataset 평가 (SVAMP)
python main.py --mode eval --method sft_id-mas \
    --domain math --eval-dataset svamp

# 처음부터 새로 평가
python main.py --mode eval --method baseline \
    --domain math --eval-dataset gsm8k --eval-resume False
```

**평가 모드 옵션:**

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--domain` | 도메인 | 필수 |
| `--eval-dataset` | 평가 데이터셋 | 필수 |
| `--method` | 평가 방법 (baseline, sft, sft_id-mas) | 필수 |
| `--student-model` | 학생 모델 | Qwen/Qwen2.5-3B-Instruct |
| `--eval-resume` | 기존 결과에서 재개 (True/False) | True |

**평가 방법:**

| Method | 설명 |
|--------|------|
| `baseline` | 베이스 모델로 평가 (파인튜닝 없음) |
| `sft` | HuggingFace Hub의 SFT 모델 (`SaFD-00/{model}-{domain}`) |
| `sft_id-mas` | ID-MAS SFT 모델 (`SaFD-00/{model}-{domain}_id-mas`) |

## 환경변수

| 변수 | 설명 | 필수 여부 |
|------|------|----------|
| `OPENAI_API_KEY` | OpenAI API 키 | gpt-* 모델 사용 시 |
| `HF_TOKEN` | HuggingFace 토큰 | 로컬 모델 사용 시 |
| `IDMAS_DEBUG_API` | API 디버그 로그 출력 | 선택 (기본값: 0) |

```bash
# 디버그 모드로 실행
IDMAS_DEBUG_API=1 python main.py --mode train --domain math --train-dataset gsm8k
```

## 디렉토리 구조

```
ID-MAS/
├── main.py                      # 메인 실행 파일
├── config/                      # 설정 모듈
│   ├── __init__.py             # 통합 인터페이스
│   ├── api.py                  # API 키 관리
│   ├── models.py               # Teacher/Student 모델 설정
│   ├── domains.py              # 도메인/데이터셋 설정
│   ├── sft.py                  # SFT 모델 매핑
│   └── paths.py                # 경로 헬퍼
├── design_modules/              # 교수 설계 단계
│   ├── instructional_goal.py   # Step 0: Instructional Goal 생성
│   ├── analysis.py             # Step 1: 교수 분석
│   ├── objectives.py           # Step 2: 수행목표 진술
│   └── rubric.py               # Step 3: 루브릭 개발
├── learning_loop/               # Iterative Scaffolding Pipeline
│   ├── graph/                  # LangGraph 구현
│   │   ├── state.py           # 상태 스키마
│   │   ├── nodes.py           # 노드 함수
│   │   └── graph.py           # StateGraph 구성
│   ├── student_model.py        # 학생 모델
│   └── teacher_model.py        # 교사 모델
├── models/                      # 모델 래퍼
│   ├── base_wrapper.py         # 베이스 클래스
│   ├── teacher_wrapper.py      # Teacher 래퍼 (API + 로컬)
│   ├── student_wrapper.py      # Student 래퍼 (로컬)
│   └── model_cache.py          # 모델 캐시
├── utils/                       # 유틸리티
│   ├── dataset_preparer.py     # 데이터셋 다운로드/전처리
│   ├── sample_extractor.py     # 샘플 추출
│   ├── domain_loader.py        # 도메인 로더
│   └── answer_extractor.py     # 답변 추출기
└── data/                        # 데이터 디렉토리
    ├── math/
    │   ├── train/data/         # 학습 데이터
    │   └── eval/data/          # 평가 데이터
    ├── logical/
    └── commonsense/
```

## 데이터 구조

### 학습/평가 데이터 형식

```json
{
  "instruction": "You are a helpful math assistant...",
  "input": "문제 텍스트",
  "output": "풀이 과정...\n\nThe answer is \\boxed{42}",
  "metadata": {}
}
```

### 출력 디렉토리 구조

```
data/{domain}/
├── train/
│   ├── data/                           # 원본 데이터
│   │   ├── {dataset}_train.json
│   │   └── {dataset}_samples.json      # Instructional Goal용 샘플
│   └── {Teacher-Model}/
│       ├── instructional-design/       # 설계 결과
│       │   └── {domain}_{dataset}_design.json
│       └── {Student-Model}/            # 학습 결과
│           ├── {dataset}_sft_{model}.json
│           └── {dataset}_train_summary_{model}.json
└── eval/
    ├── data/                           # 평가 데이터
    │   └── {dataset}_test.json
    └── {Student-Model}/                # 평가 결과
        └── {dataset}_eval_results-{method}.json
```

## 참고 문헌

1. Dick, W., Carey, L., & Carey, J. O. (2015). The systematic design of instruction (8th ed.). Pearson.
2. Anderson, L. W., & Krathwohl, D. R. (2001). A taxonomy for learning, teaching, and assessing. Longman.
