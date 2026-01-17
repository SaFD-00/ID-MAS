# ReGenesis: 자기 개선을 통한 LLM 추론 일반화

ICLR-2025 Oral Presentation 채택 논문

논문: [Arxiv Link](https://arxiv.org/abs/2410.02108)

## 개요

ReGenesis는 외부 감독 없이 LLM이 스스로 추론 경로를 생성하여 추론 능력을 향상시키는 방법론입니다. 기존 자기 합성 방법(STaR 등)의 OOD(Out-of-Domain) 일반화 문제를 해결하기 위해, **추상에서 구체로(Abstract to Concrete)** 진행하는 추론 경로 합성 방식을 제안합니다.

### 핵심 특징

- **3단계 추론 경로 생성**: 일반 가이드라인 → 태스크 특화 구조 → 추론 경로
- **OOD 성능 향상**: 기존 방법 대비 약 10.7% 성능 개선
- **다중 모델 지원**: Llama 3.x, Qwen2.5 시리즈 (8개 모델)

---

## 지원 모델

| 모델 | 크기 | 학습 전략 |
|------|------|-----------|
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | Full Fine-tuning |
| `meta-llama/Llama-3.1-70B-Instruct` | 70B | LoRA + 4-bit |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | Full Fine-tuning |
| `meta-llama/Llama-3.3-70B-Instruct` | 70B | LoRA + 4-bit |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | Full Fine-tuning |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | Full Fine-tuning |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | Full Fine-tuning |
| `Qwen/Qwen2.5-72B-Instruct` | 72B | LoRA + 4-bit |

---

## 환경 설정

```bash
pip install -r requirements_updated.txt
```

### 주요 의존성

- `torch>=2.3.0`
- `transformers>=4.41.1`
- `vllm>=0.4.2` - 추론 경로 생성
- `peft>=0.11.1` - LoRA 지원
- `bitsandbytes>=0.43.0` - 4-bit 양자화

---

## 빠른 시작

### 단일 모델 전체 파이프라인 실행

```bash
./scripts/run_full_pipeline.sh meta-llama/Llama-3.1-8B-Instruct
```

### 8개 모델 일괄 실행

```bash
./scripts/run_all_models.sh all
```

---

## 파이프라인

### 전체 흐름

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1단계          │     │  2단계          │     │  3단계          │
│  데이터 생성    │────▶│  필터링         │────▶│  모델 학습      │
│  (vLLM)         │     │  (Exact Match)  │     │  (HF Trainer)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 1단계: 추론 경로 생성

ReGenesis의 핵심 단계로, 25개 시드 모듈에서 태스크별 추론 경로를 생성합니다.

```bash
./scripts/generate_reasoning_paths.sh <모델명> <데이터셋> [시작인덱스] [종료인덱스]

# 예시
./scripts/generate_reasoning_paths.sh meta-llama/Llama-3.1-8B-Instruct gsm8k
./scripts/generate_reasoning_paths.sh Qwen/Qwen2.5-7B-Instruct math 0 1000
```

Python 직접 실행:

```bash
python -m src.reasoning.multi_model_reasoning_gen \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset gsm8k \
    --output_dir data/generated \
    --num_samples 20
```

### 2단계: 데이터 필터링

Ground Truth와 Exact Match로 정답인 추론 경로만 선별합니다.

```bash
./scripts/filter_data.sh <모델명> <데이터셋> <정답유형>

# 예시
./scripts/filter_data.sh meta-llama/Llama-3.1-8B-Instruct gsm8k numeric
./scripts/filter_data.sh meta-llama/Llama-3.1-8B-Instruct reclor option
```

정답 유형:
- `numeric` - 수학 데이터셋 (GSM8K, MATH)
- `option` - 객관식 (ReClor, ARC-c)

### 3단계: 모델 학습

필터링된 데이터로 SFT(Supervised Fine-tuning)를 수행합니다.

```bash
./scripts/train_model.sh <모델명> <학습데이터> [출력경로] [LoRA사용] [GPU수]

# Full Fine-tuning (소형 모델)
./scripts/train_model.sh meta-llama/Llama-3.1-8B-Instruct data/filtered/train.json

# LoRA 학습 (70B+ 대형 모델)
./scripts/train_model.sh meta-llama/Llama-3.1-70B-Instruct data/filtered/train.json "" true 4
```

---

## 프로젝트 구조

```
ReGenesis/
├── src/
│   ├── config/
│   │   ├── model_config.py          # 모델별 설정 (템플릿, 토큰 등)
│   │   └── training_config.py       # 학습 하이퍼파라미터
│   ├── reasoning/
│   │   ├── multi_model_reasoning_gen.py  # 다중 모델 추론 경로 생성
│   │   ├── template_utils.py        # 채팅 템플릿 유틸리티
│   │   ├── read_datasets.py         # 데이터셋 로더
│   │   └── process_reason.py        # 레거시 필터링
│   ├── pipeline/
│   │   ├── filtering.py             # 데이터 필터링 모듈
│   │   └── training_pipeline.py     # 통합 파이프라인
│   └── finetune_code/
│       ├── multi_model_finetune.py  # LoRA 지원 다중 모델 학습
│       └── finetune_code.py         # 레거시 학습 코드
├── scripts/
│   ├── generate_reasoning_paths.sh  # 추론 경로 생성
│   ├── filter_data.sh               # 데이터 필터링
│   ├── train_model.sh               # 모델 학습
│   ├── run_full_pipeline.sh         # 전체 파이프라인
│   └── run_all_models.sh            # 8개 모델 일괄 실행
├── configs/
│   ├── models/                      # 모델별 YAML 설정
│   └── training/                    # 학습 설정
├── data/
│   ├── math/                        # GSM8K, MATH
│   ├── logical/                     # ReClor
│   └── commonsense/                 # ARC-c
└── requirements_updated.txt
```

---

## 설정

### 모델 설정

각 모델은 `configs/models/`에 YAML 파일로 관리됩니다:

```yaml
# configs/models/llama-3.1-8b.yaml
model:
  name: "meta-llama/Llama-3.1-8B-Instruct"
  template: "llama-3"
  tensor_parallel: 1

generation:
  temperature: 1.2
  top_p: 0.9
  max_tokens: 2048
  num_samples: 20

training:
  strategy: "full"
  learning_rate: 1e-6
  epochs: 3
  batch_size: 16
```

### 학습 하이퍼파라미터 (논문 기준)

| 파라미터 | 값 |
|----------|-----|
| Learning Rate | 1e-6 |
| Epochs | 3 |
| Batch Size | 16 (effective) |
| Warmup Ratio | 0.03 |
| LR Scheduler | Cosine |

---

## 데이터셋

| 데이터셋 | 도메인 | 정답 유형 |
|----------|--------|-----------|
| GSM8K | 수학 | 숫자 |
| MATH | 수학 | 숫자 |
| ReClor | 논리 | 객관식 (A/B/C/D) |
| ARC-c | 상식 | 객관식 (A/B/C/D) |

---

## 메모리 요구사항

| 모델 크기 | 학습 전략 | GPU 메모리 | 권장 GPU 수 |
|-----------|-----------|------------|-------------|
| 3B | Full FT | ~16GB | 1 |
| 7B-8B | Full FT | ~32GB | 1-2 |
| 14B | Full FT | ~48GB | 2 |
| 70B+ | LoRA + 4-bit | ~80GB | 4 |

---

## 인용

```bibtex
@article{peng2024regenesis,
  title={ReGenesis: LLMs can Grow into Reasoning Generalists via Self-Improvement},
  author={Peng, Xiangyu and Xia, Congying and Yang, Xinyi and Xiong, Caiming and Wu, Chien-Sheng and Xing, Chen},
  journal={arXiv preprint arXiv:2410.02108},
  year={2024}
}
```

---

## 평가

평가 코드는 [Eurus](https://github.com/OpenBMB/Eurus)를 사용합니다.
