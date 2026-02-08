# ID-MAS: Instructional Design Multi-Agent System

LLM 학습을 위한 Dick & Carey 모델 기반 교수 설계 시스템

## 개요

ID-MAS는 교수 설계(Instructional Design) 이론을 LLM 학습에 적용한 Multi-Agent 시스템입니다.

**핵심 특징:**
- **데이터셋별 분리 학습**: GSM8K, MATH 각각 고유한 Instructional Goal로 독립 학습
- **Iterative Scaffolding Pipeline**: Performance Objectives 기반 평가 + HOT/LOT Scaffolding을 통한 6-Step 반복 학습 (최대 5회 반복)
- **LangGraph 기반 워크플로우**: StateGraph 상태 관리, 체크포인트 기반 Resume 지원

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
python -m utils.sample_extractor

# 4. 학습 실행
python main.py --mode train --domain math --train-dataset gsm8k

# 5. 평가 실행
python main.py --mode eval --method baseline --domain math --eval-dataset gsm8k
```

### 환경변수

| 변수 | 설명 | 필수 여부 |
|------|------|----------|
| `OPENAI_API_KEY` | OpenAI API 키 | gpt-* 모델 사용 시 |
| `HF_TOKEN` | HuggingFace 토큰 | 로컬 모델 사용 시 |
| `IDMAS_DEBUG_API` | API 디버그 로그 출력 | 선택 (기본값: 0) |

## 디렉토리 구조

```
ID-MAS/
├── main.py                      # 메인 실행 파일
├── config/                      # 설정 모듈
├── design_modules/              # 교수 설계 단계
├── learning_loop/               # Iterative Scaffolding Pipeline
├── models/                      # 모델 래퍼
├── prompts/                     # 프롬프트 템플릿
├── utils/                       # 유틸리티
├── data/                        # 원본 데이터 (train/data/, eval/data/)
└── outputs/                     # 학습 결과물
    └── {domain}/train/{teacher}/{student}/
```

## 문서

| 문서 | 설명 |
|------|------|
| [시스템 아키텍처](ARCHITECTURE.md) | 3-Phase 파이프라인 구조, 모듈 설계, 데이터 흐름 |
| [사용 가이드](USAGE.md) | 지원 모델, 데이터셋, CLI 상세 사용법, 실행 예제 |

## 참고 문헌

1. Dick, W., Carey, L., & Carey, J. O. (2015). The systematic design of instruction (8th ed.). Pearson.
2. Anderson, L. W., & Krathwohl, D. R. (2001). A taxonomy for learning, teaching, and assessing. Longman.
