# V-STaR: Training Verifiers for Self-Taught Reasoners

[V-STaR 논문](https://arxiv.org/abs/2402.06457)의 PyTorch/HuggingFace 구현체입니다.

> **V-STaR: Training Verifiers for Self-Taught Reasoners**
> Arian Hosseini, Xingdi Yuan, Nikolay Malkin, Aaron Courville, Alessandro Sordoni, Rishabh Agarwal
> ICML 2024

## 개요

V-STaR는 STaR 알고리즘을 확장하여, 생성 모델(Generator)과 검증 모델(Verifier)을 함께 학습합니다. 검증 모델은 여러 풀이 중 올바른 것을 선택하는 데 사용되어, Best-of-N 방식으로 추론 정확도를 크게 향상시킵니다.

### 핵심 아이디어

1. **다양한 풀이 생성**: 각 문제에 대해 k개의 풀이를 샘플링
2. **정답/오답 분류**: 생성된 풀이를 정답 여부로 분류
3. **Generator 학습 (SFT)**: 정답 풀이로 SFT 학습
4. **Verifier 학습 (DPO)**: 정답/오답 쌍으로 선호도 학습
5. **추론 시 Best-of-N**: Verifier로 가장 좋은 풀이 선택

## 알고리즘 (논문 Algorithm 1)

```
Algorithm 1: V-STaR Training

Input: 질문 집합 D_query, SFT 데이터 D_SFT, 반복 횟수 N, 샘플 수 k
Output: Generator π, Verifier v

# 초기화: D_SFT로 Generator 학습
π ← SFT(D_SFT)

for iteration i = 1 to N:
    D_correct = {}  # 정답 풀이 집합
    D_pairs = {}    # 선호도 쌍 집합

    for each question q in D_query:
        # Step 1: k개 풀이 생성
        solutions = [π.generate(q) for _ in range(k)]

        for solution in solutions:
            answer = extract_answer(solution)
            if answer == ground_truth[q]:
                D_correct.add((q, solution))
            else:
                D_pairs.add((q, random_correct, solution))  # (정답, 오답) 쌍

    # Step 2: Generator SFT
    π ← SFT(π, D_correct)

    # Step 3: Verifier DPO (최종 반복에서만)
    if i == N:
        v ← DPO(π, D_pairs)

return π, v
```

---

## 학습 파이프라인 상세

### 상세 파이프라인 흐름도

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        V-STaR 학습 파이프라인                                │
│                                                                             │
│    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐               │
│    │  초기화     │─────▶│  반복 학습   │─────▶│  마무리     │               │
│    │  (G_SFT)    │      │  (N회)      │      │  (Verifier) │               │
│    └─────────────┘      └─────────────┘      └─────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                              초기화 단계
═══════════════════════════════════════════════════════════════════════════════

[입력]
  D_SFT: 초기 SFT 학습 데이터 (정답 풀이 포함)
  D_query: 학습에 사용할 질문들
  G_base: 사전학습된 베이스 모델
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  G_SFT 학습 (Reference Policy)                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  G_SFT ← SFT(G_base, D_SFT)                                                │
│                                                                             │
│  • D_SFT로 베이스 모델을 SFT 학습                                           │
│  • 이후 Verifier DPO 학습의 Reference Model로 사용                          │
│  • 학습 후 고정 (requires_grad = False)                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  데이터 집합 초기화                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  D_GEN ← D_SFT          # Generator 학습용 (정답 풀이만)                    │
│  D_VER ← D_SFT          # Verifier 학습용 (전체 풀이, is_correct 라벨)      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                           반복 단계 (N회)
═══════════════════════════════════════════════════════════════════════════════
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Iteration i = 1 to N                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Step 1] Generator SFT 학습                                               │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │ G_i ← SFT(G_base, D_GEN)  ⭐ 매번 G_base에서 시작                  │     │
│  │                                                                    │     │
│  │ • 누적된 D_GEN 데이터로 새로운 Generator 학습                      │     │
│  │ • LoRA 사용 (r=8, alpha=32)                                        │     │
│  │ • learning_rate: 2e-5, epochs: 2                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│       │                                                                     │
│       ▼                                                                     │
│  [Step 2] 풀이 샘플링                                                       │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │ for each question q in D_query:                                    │     │
│  │   solutions = G_i.generate(q, k=16, temperature=0.7)               │     │
│  │   # 각 질문에 대해 k개 다양한 풀이 생성                            │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│       │                                                                     │
│       ▼                                                                     │
│  [Step 3] 풀이 라벨링                                                       │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │ for each solution in solutions:                                    │     │
│  │   answer = extract_answer(solution)                                │     │
│  │   is_correct = (answer == ground_truth)                            │     │
│  │                                                                    │     │
│  │   ┌─────────────────┐    ┌─────────────────┐                      │     │
│  │   │ is_correct=True │    │ is_correct=False│                      │     │
│  │   │     (정답)       │    │     (오답)      │                      │     │
│  │   └────────┬────────┘    └────────┬────────┘                      │     │
│  │            │                      │                                │     │
│  │            ▼                      ▼                                │     │
│  │      D_GEN에 추가           D_VER에만 추가                         │     │
│  │      D_VER에도 추가         (선호도 쌍용)                          │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│       │                                                                     │
│       ▼                                                                     │
│  [Step 4] 데이터 누적                                                       │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │ D_GEN ← D_GEN ∪ {정답 풀이들}                                      │     │
│  │ D_VER ← D_VER ∪ {전체 풀이들}                                      │     │
│  │                                                                    │     │
│  │ 반복이 진행될수록 D_GEN, D_VER 크기 증가                           │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│       │                                                                     │
│       ▼                                                                     │
│  [Step 5] Generator 저장 및 메모리 해제                                     │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │ save(G_i, f"checkpoints/iteration_{i}/generator/")                 │     │
│  │ del G_i  # VRAM 절약                                               │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ─────────────────────────── 다음 반복으로 ───────────────────────────      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                              마무리 단계
═══════════════════════════════════════════════════════════════════════════════
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  선호도 쌍 (Preference Pairs) 생성                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  for each question q in D_VER:                                              │
│    correct_solutions = [s for s in D_VER[q] if s.is_correct]               │
│    incorrect_solutions = [s for s in D_VER[q] if not s.is_correct]         │
│                                                                             │
│    # Cartesian product: 정답 × 오답                                         │
│    for c in correct_solutions:                                              │
│      for i in incorrect_solutions:                                          │
│        preference_pairs.add(PreferencePair(                                 │
│          prompt=q,                                                          │
│          chosen=c,      # y+ (정답 풀이)                                    │
│          rejected=i     # y- (오답 풀이)                                    │
│        ))                                                                   │
│                                                                             │
│  예시: 질문 1에 정답 3개, 오답 2개 → 3×2 = 6 쌍                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Verifier DPO 학습                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  V ← DPO(G_SFT, preference_pairs)                                          │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │ DPO Loss:                                                          │     │
│  │                                                                    │     │
│  │ L = -log(σ(β × (log V(y+|x) - log V(y-|x))))                      │     │
│  │                                                                    │     │
│  │ • V(y|x): Verifier가 풀이 y에 부여하는 log-probability             │     │
│  │ • y+: 정답 풀이 (chosen)                                           │     │
│  │ • y-: 오답 풀이 (rejected)                                         │     │
│  │ • β: 0.1 (DPO temperature)                                         │     │
│  │ • Reference Model: G_SFT (고정)                                    │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  하이퍼파라미터:                                                            │
│  • learning_rate: 5e-7 (매우 낮음)                                          │
│  • epochs: 1                                                                │
│  • batch_size: 4                                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
[출력]
  Generator: checkpoints/generator_final/
  Verifier: checkpoints/verifier_final/
```

### 핵심 컴포넌트

| 컴포넌트 | 파일 | 역할 |
|----------|------|------|
| **VSTaRGenerator** | `models/generator.py` | 풀이 생성 (SFT 학습) |
| **VSTaRVerifier** | `models/verifier.py` | 풀이 점수 매김 (DPO 학습) |
| **SFTTrainer** | `training/sft_trainer.py` | Generator SFT 학습 |
| **DPOTrainer** | `training/dpo_trainer.py` | Verifier DPO 학습 |
| **IterationRunner** | `training/iteration_runner.py` | Algorithm 1 구현 |
| **SolutionSampler** | `data/sampler.py` | 풀이 샘플링 및 라벨링 |
| **PreferenceDataset** | `data/preference_dataset.py` | DPO 선호도 쌍 생성 |

### Generator vs Verifier 역할

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            학습 시                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Generator]                        [Verifier]                              │
│  ┌───────────────────┐              ┌───────────────────┐                  │
│  │ • SFT 학습        │              │ • DPO 학습        │                  │
│  │ • 정답 풀이만     │              │ • 정답/오답 쌍    │                  │
│  │ • 매 반복마다     │              │ • 최종 반복에서만 │                  │
│  │   재학습          │              │   한 번 학습      │                  │
│  └───────────────────┘              └───────────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            추론 시 (Best-of-N)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────┐  │
│  │  문제 입력  │────▶│ Generator가 │────▶│ Verifier가  │────▶│ 최고점  │  │
│  │             │     │ N개 풀이    │     │ 각 풀이에   │     │ 풀이    │  │
│  │             │     │ 생성        │     │ 점수 부여   │     │ 선택    │  │
│  └─────────────┘     └─────────────┘     └─────────────┘     └─────────┘  │
│                                                                             │
│  예시 (N=64):                                                               │
│  solutions = generator.generate(prompt, k=64, temperature=0.7)             │
│  scores = [verifier.score(prompt, sol) for sol in solutions]               │
│  best = solutions[argmax(scores)]  # 가장 높은 점수의 풀이                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 데이터 흐름

```
[초기 데이터]
     │
     ├── D_SFT (정답 풀이들)
     │      │
     │      ▼
     │   ┌──────────────────┐
     │   │ G_SFT 학습       │ ──────────────────────────────────┐
     │   │ (Reference)      │                                   │
     │   └──────────────────┘                                   │
     │                                                          │
     ▼                                                          │
D_GEN ← D_SFT ────────────────────────┐                        │
D_VER ← D_SFT ───────────────────┐    │                        │
                                 │    │                        │
[반복 i]                         │    │                        │
     │                           │    │                        │
     ▼                           │    │                        │
┌──────────────────┐             │    │                        │
│ Generator 학습   │◀────────────┼────┘                        │
│ G_i ← SFT(D_GEN) │             │                             │
└────────┬─────────┘             │                             │
         │                       │                             │
         ▼                       │                             │
┌──────────────────┐             │                             │
│ k개 풀이 샘플링  │             │                             │
└────────┬─────────┘             │                             │
         │                       │                             │
         ▼                       │                             │
┌──────────────────┐             │                             │
│ 라벨링           │             │                             │
│ (정답/오답)      │             │                             │
└────────┬─────────┘             │                             │
         │                       │                             │
         ├── 정답 ───▶ D_GEN 누적 ┘                             │
         │                                                      │
         └── 전체 ───▶ D_VER 누적 ──────────┐                   │
                                           │                   │
[마무리]                                   │                   │
         ┌─────────────────────────────────┘                   │
         │                                                     │
         ▼                                                     │
┌──────────────────┐                                           │
│ 선호도 쌍 생성   │                                           │
│ (정답, 오답)     │                                           │
└────────┬─────────┘                                           │
         │                                                     │
         ▼                                                     │
┌──────────────────┐                                           │
│ Verifier DPO     │◀──────────────────────────────────────────┘
│ V ← DPO(G_SFT,   │    (Reference Model로 사용)
│        pairs)    │
└──────────────────┘
```

### STaR vs V-STaR 학습 비교

| 측면 | STaR | V-STaR |
|------|------|--------|
| **모델 수** | Generator 1개 | Generator + Verifier 2개 |
| **Generator 학습** | 정답 + Rationalized | 정답만 (D_GEN) |
| **오답 활용** | Rationalization으로 재생성 | DPO 선호도 학습 데이터 |
| **Verifier** | 없음 | DPO로 학습 |
| **Reference Model** | 없음 | G_SFT (초기 SFT 모델) |
| **추론 방식** | 단일 생성 | Best-of-N (Verifier 선택) |

---

## 설치

```bash
cd V-STaR
pip install -r requirements.txt
```

### 의존성

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- TRL (DPO 학습)
- PEFT (LoRA 지원)

## 사용법

### 학습

```bash
# 기본 학습
python main.py train \
    --model Qwen/Qwen2.5-3B-Instruct \
    --domains math \
    --iterations 3 \
    --k 16

# 상세 설정
python main.py train \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data-dir ./data \
    --output-dir ./checkpoints \
    --domains math,logical,commonsense \
    --iterations 5 \
    --k 32 \
    --temperature 0.7 \
    --learning-rate 1e-5 \
    --batch-size 4

# D_SFT 데이터 직접 지정
python main.py train \
    --model Qwen/Qwen2.5-3B-Instruct \
    --domains math \
    --d-sft-path ./data/math/train/gsm8k.json

# 이어서 학습
python main.py train \
    --model Qwen/Qwen2.5-3B-Instruct \
    --domains math \
    --resume-from 2
```

### 평가

```bash
# Generator + Verifier 평가
python main.py evaluate \
    --generator-path ./checkpoints/generator_final \
    --verifier-path ./checkpoints/verifier_final \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --domains math \
    --num-samples 64

# Self-Consistency 비교 포함
python main.py evaluate \
    --generator-path ./checkpoints/generator_final \
    --verifier-path ./checkpoints/verifier_final \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --domains math \
    --self-consistency \
    --output results.json
```

### 풀이 생성

```bash
# 특정 데이터셋에 대해 풀이 생성
python main.py sample \
    --model-path ./checkpoints/generator_final \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --data-dir ./data \
    --domain math \
    --dataset gsm8k \
    --output solutions.json \
    --num-samples 16
```

### 사용 가능한 모델 확인

```bash
python main.py list-models
```

## 프로젝트 구조

```
V-STaR/
├── main.py                 # CLI 진입점
├── config/
│   ├── training.py         # 학습 설정
│   ├── models.py           # 모델 설정
│   ├── domains.py          # 도메인 설정
│   └── paths.py            # 경로 설정
├── models/
│   ├── generator.py        # Generator 모델
│   ├── verifier.py         # Verifier 모델
│   └── model_cache.py      # 모델 캐시
├── training/
│   ├── sft_trainer.py      # SFT 학습
│   ├── dpo_trainer.py      # DPO 학습
│   └── iteration_runner.py # 반복 실행
├── evaluation/
│   ├── evaluator.py        # 평가 모듈
│   ├── answer_checker.py   # 정답 체크
│   └── metrics.py          # 평가 메트릭
├── data/
│   ├── loader.py           # 데이터 로더
│   ├── sampler.py          # 풀이 샘플러
│   └── preference_dataset.py # DPO 데이터셋
├── prompts/
│   └── templates.py        # 프롬프트 템플릿
├── utils/
│   └── checkpoint.py       # 체크포인트 관리
└── tests/                  # 테스트
```

## 주요 설정 옵션

### TrainingConfig

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `num_iterations` | 3 | V-STaR 반복 횟수 |
| `samples_per_query` | 16 | 문제당 생성할 풀이 수 (k) |
| `temperature` | 0.7 | 샘플링 온도 |
| `max_pairs_per_question` | None | DPO 쌍 최대 수 (제한 없음) |

### SFT 설정

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `learning_rate` | 2e-5 | 학습률 |
| `num_train_epochs` | 1 | 에폭 수 |
| `per_device_train_batch_size` | 4 | 배치 크기 |
| `gradient_accumulation_steps` | 4 | Gradient 누적 |

### DPO 설정

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `learning_rate` | 5e-7 | 학습률 |
| `beta` | 0.1 | DPO beta 파라미터 |
| `num_train_epochs` | 1 | 에폭 수 |

## Verifier 역할

### 학습 시
- Generator가 생성한 풀이 중 정답/오답을 분류
- 정답-오답 쌍으로 DPO(Direct Preference Optimization) 학습
- 논문에 따라 **최종 반복에서 한 번만** Verifier 학습

### 추론 시 (Best-of-N)
- Generator가 N개의 풀이 생성
- Verifier가 각 풀이에 점수 부여
- 가장 높은 점수의 풀이 선택

```python
# Best-of-N 추론 예시
solutions = generator.generate(prompt, k=64, temperature=0.7)
scores = [verifier.score(prompt, sol) for sol in solutions]
best_solution = solutions[argmax(scores)]
```

## 출력 구조

```
checkpoints/
├── iteration_1/
│   ├── generator/          # Generator 체크포인트
│   ├── sft_data.json       # SFT 학습 데이터
│   └── metrics.json        # 반복 메트릭
├── iteration_2/
│   └── ...
├── iteration_3/
│   ├── generator/
│   ├── verifier/           # Verifier (최종 반복)
│   └── preference_pairs.json
├── generator_final/        # 최종 Generator
└── verifier_final/         # 최종 Verifier
```

## 성능

GSM8K 데이터셋 기준 (논문 Table 1 참조):

| 방법 | pass@1 | Best-of-64 |
|------|--------|------------|
| Baseline | ~45% | ~55% (Self-Consistency) |
| STaR | ~55% | ~65% (Self-Consistency) |
| V-STaR | ~55% | **~75%** (Verifier) |

*Verifier를 사용한 Best-of-N이 Self-Consistency보다 10%p 이상 높은 성능*

## STaR vs V-STaR

| 특성 | STaR | V-STaR |
|------|------|--------|
| 모델 | Generator만 | Generator + Verifier |
| 학습 방법 | SFT | SFT + DPO |
| 추론 | 단일 생성 | Best-of-N (Verifier 선택) |
| 오답 활용 | Rationalization | 선호도 학습 데이터 |

## 참고 문헌

```bibtex
@inproceedings{hosseini2024vstar,
  title={V-STaR: Training Verifiers for Self-Taught Reasoners},
  author={Hosseini, Arian and Yuan, Xingdi and Malkin, Nikolay and Courville, Aaron and Sordoni, Alessandro and Agarwal, Rishabh},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

## 라이선스

MIT License
