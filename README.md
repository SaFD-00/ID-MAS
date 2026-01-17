# ID-MAS: 교수설계 기반 다중 에이전트 추론 학습 시스템

대규모 언어 모델(LLM)의 추론 능력 향상을 위한 교수설계 기반 학습 프레임워크입니다.

## 프로젝트 구조

이 저장소는 세 가지 주요 프로젝트를 포함합니다:

```
ID-MAS/
├── ID-MAS/          # 메인 프로젝트: 교수설계 기반 다중 에이전트 시스템
├── STaR/            # Self-Taught Reasoner 구현
├── V-STaR/          # Verifier 학습을 포함한 STaR 확장
└── ReGenesis/       # 다중 모델 추론 생성 파이프라인
```

## 프로젝트 개요

### ID-MAS (Instructional Design Multi-Agent System)

Dick & Carey 교수설계 모델에 기반한 LangGraph 파이프라인으로, 교사-학생 모델 구조를 통해 추론 능력을 향상시킵니다.

**핵심 기능:**
- Instructional Goal 자동 생성 (데이터셋 샘플 기반)
- Performance Objectives 기반 평가
- Iterative Scaffolding (최대 5회 반복)
- 소크라테스식 질문을 통한 학생 모델 가이드
- SFT 데이터 자동 생성 (Case A/B/C)

**지원 도메인:**
- Math (GSM8K, MATH)
- Logical (ReClor, LogiQA)
- Commonsense (ARC-C, OBQA)

### STaR (Self-Taught Reasoner)

[STaR 논문](https://arxiv.org/abs/2203.14465) 구현체로, 반복적 자기개선을 통해 추론 rationale을 학습합니다.

**핵심 알고리즘:**
1. Few-shot 프롬프팅으로 rationale 생성
2. 정답으로 이어지는 rationale 필터링
3. 오답에 대해 정답 힌트를 주고 rationalization
4. 정답 + rationalized 예제로 파인튜닝
5. N회 반복

자세한 내용은 [STaR/README.md](STaR/README.md)를 참조하세요.

### V-STaR (Verifier for Self-Taught Reasoner)

[V-STaR 논문](https://arxiv.org/abs/2402.06457) 구현체로, 생성 모델과 검증 모델을 함께 학습합니다.

**핵심 기능:**
- Generator: 다양한 풀이 생성 (k개 샘플링)
- Verifier: DPO 기반 선호도 학습
- Best-of-N 선택으로 정확도 향상

자세한 내용은 [V-STaR/README.md](V-STaR/README.md)를 참조하세요.

## 설치

```bash
# 저장소 클론
git clone https://github.com/your-repo/ID-MAS.git
cd ID-MAS

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는 venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

## 빠른 시작

### ID-MAS 학습

```bash
# GSM8K로 학습
python -m ID-MAS.main --mode train --domain math --train-dataset gsm8k

# 이어서 학습 (체크포인트에서 재개)
python -m ID-MAS.main --mode train --domain math --train-dataset gsm8k --resume True

# 다른 모델로 학습
python -m ID-MAS.main --mode train --domain math --train-dataset gsm8k \
    --student-model Qwen/Qwen3-4B-Instruct-2507
```

### ID-MAS 평가

```bash
# Baseline 평가 (파인튜닝 없는 기본 모델)
python -m ID-MAS.main --mode eval --method baseline --domain math --eval-dataset gsm8k

# SFT 평가 (일반 SFT 모델)
python -m ID-MAS.main --mode eval --method sft --domain math --eval-dataset gsm8k

# SFT_ID-MAS 평가 (ID-MAS 방식 SFT 모델)
python -m ID-MAS.main --mode eval --method sft_id-mas --domain math --eval-dataset gsm8k
```

### STaR 학습

```bash
# GSM8K로 STaR 학습
python -m STaR.cli train --model Qwen/Qwen2.5-3B-Instruct --dataset gsm8k

# 커스텀 설정
python -m STaR.cli train --model Qwen/Qwen2.5-7B-Instruct --dataset gsm8k \
    --iterations 5 --output-dir ./my_output
```

### V-STaR 학습

```bash
# V-STaR 학습
python V-STaR/main.py train --model Qwen/Qwen2.5-3B-Instruct \
    --domains math --iterations 3 --k 16
```

## 지원 모델

### 학생 모델
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen3-4B-Instruct-2507`
- `meta-llama/Llama-3.2-3B-Instruct`

### 교사 모델
- OpenAI: `gpt-4o-mini`, `gpt-4o`, `o1-mini`
- 로컬: 학생 모델과 동일 (모델 공유 가능)

## 프로젝트 아키텍처

### ID-MAS 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│                    INSTRUCTIONAL DESIGN                      │
│  Step 0: Instructional Goal 생성 (샘플 기반)                │
│  Step 1-4: 교수분석 → 수행목표 → 루브릭 개발                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   ITERATIVE SCAFFOLDING                      │
│  1. 학생 모델 초기 응답                                     │
│  2. 교사 모델 PO 평가 + 소크라테스 질문                     │
│  3. 스캐폴딩 아티팩트 생성                                   │
│  4. 학생 재응답 (최대 5회)                                  │
│  5. 응답 재구성 (Case A/B/C)                                │
│  6. SFT 데이터 생성                                         │
└─────────────────────────────────────────────────────────────┘
```

### 데이터 흐름

```
원본 데이터 → Enhanced Instruction → Iterative Learning → SFT Data
     ↓              ↓                      ↓                ↓
 GSM8K 등    + Instructional Goal   Teacher-Student    Case A/B/C
             + Task Analysis          Interaction       데이터셋
```

## 디렉토리 구조

```
ID-MAS/
├── ID-MAS/
│   ├── main.py                 # 메인 진입점
│   ├── config/                 # 설정 파일
│   ├── design_modules/         # 교수설계 모듈
│   ├── learning_loop/          # LangGraph 파이프라인
│   ├── models/                 # 모델 래퍼
│   ├── prompts/                # 프롬프트 템플릿
│   └── utils/                  # 유틸리티
├── STaR/
│   ├── cli.py                  # CLI 진입점
│   ├── star/                   # STaR 핵심 모듈
│   └── ...
├── V-STaR/
│   ├── main.py                 # CLI 진입점
│   ├── training/               # SFT/DPO 학습
│   └── ...
└── data/                       # 데이터셋 (gitignore)
```

## 환경 변수

```bash
# OpenAI API (교사 모델 사용 시)
export OPENAI_API_KEY="your-api-key"

# HuggingFace (모델 다운로드)
export HF_TOKEN="your-token"
```

## 기여

이슈 및 PR을 환영합니다. 기여 전 다음을 확인해주세요:
1. 코드 스타일 가이드 준수
2. 테스트 추가 또는 업데이트
3. 문서화 업데이트

## 참고 문헌

- [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)
- [V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457)
- [Dick & Carey Instructional Design Model](https://en.wikipedia.org/wiki/Dick_and_Carey_model)

## 라이선스

MIT License
