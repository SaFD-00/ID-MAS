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
