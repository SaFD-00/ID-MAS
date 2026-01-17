# Teacher & Student Model 목록 수정 계획

**작성일**: 2026-01-11
**도메인**: Coding
**우선순위**: Medium

---

## 1. 개요

### 목표
Teacher model과 Student model 목록을 업데이트하고, 모델 순서를 일관되게 정렬합니다.

### 범위
- Teacher model 목록: 1개 삭제, 3개 추가
- Student model 목록: 3개 추가
- SFT 모델 매핑 추가
- 모델 순서 정렬 (llama → qwen, 버전 오름차순)

### 영향 파일
- `config/models.py`
- `config/config.py`
- `config/sft.py`

---

## 2. 요구사항 분석

### 2.1 순서 정렬 규칙
1. **1차 정렬**: llama 모델을 qwen 모델보다 앞에 배치
2. **2차 정렬**: 같은 시리즈 내에서 버전 번호 오름차순

**정렬 예시**:
```
meta-llama/Llama-3.1-8B-Instruct
meta-llama/Llama-3.1-70B-Instruct
meta-llama/Llama-3.2-3B-Instruct
meta-llama/Llama-3.3-70B-Instruct
Qwen/Qwen2.5-3B-Instruct
Qwen/Qwen2.5-7B-Instruct
Qwen/Qwen2.5-14B-Instruct
Qwen/Qwen2.5-72B-Instruct
Qwen/Qwen3-4B-Instruct-2507
```

### 2.2 Teacher Model 변경사항

#### 삭제
- `openai/gpt-oss-20b`

#### 추가
- `meta-llama/Llama-3.1-70B-Instruct`
- `meta-llama/Llama-3.3-70B-Instruct`
- `Qwen/Qwen2.5-72B-Instruct`

#### 최종 목록 (정렬 후)
```python
AVAILABLE_TEACHER_MODELS = [
    # OpenAI
    "gpt-5-2025-08-07",
    # LLaMA-Factory API (OpenAI-compatible)
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
]
```

### 2.3 Student Model 변경사항

#### 추가
- `meta-llama/Llama-3.1-70B-Instruct`
- `meta-llama/Llama-3.3-70B-Instruct`
- `Qwen/Qwen2.5-72B-Instruct`

#### 최종 목록 (정렬 후)
```python
AVAILABLE_STUDENT_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
]
```

### 2.4 SFT 모델 매핑 추가

`config/sft.py`의 `MODEL_NAME_TO_SHORT`에 다음 매핑 추가:

```python
MODEL_NAME_TO_SHORT = {
    "meta-llama/Llama-3.1-8B-Instruct": "llama3.1-8b",
    "meta-llama/Llama-3.1-70B-Instruct": "llama3.1-70b",    # 추가
    "meta-llama/Llama-3.2-3B-Instruct": "llama3.2-3b",
    "meta-llama/Llama-3.3-70B-Instruct": "llama3.3-70b",    # 추가
    "Qwen/Qwen2.5-3B-Instruct": "qwen2.5-3b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen2.5-7b",
    "Qwen/Qwen2.5-14B-Instruct": "qwen2.5-14b",
    "Qwen/Qwen2.5-72B-Instruct": "qwen2.5-72b",             # 추가
    "Qwen/Qwen3-4B-Instruct-2507": "qwen3-4b",
}
```

---

## 3. 구현 계획

### Phase 1: 파일 백업 및 분석 ✓
**상태**: 완료 (이미 파일들을 읽었음)

**확인된 사항**:
- `config/models.py`와 `config/config.py`에 동일한 모델 목록 정의됨
- 두 파일 모두 수정 필요
- SFT 매핑도 업데이트 필요

### Phase 2: config/models.py 수정

**작업 내용**:
1. `AVAILABLE_TEACHER_MODELS` (11-22줄) 수정
   - `openai/gpt-oss-20b` 삭제
   - 3개 모델 추가
   - llama → qwen 순, 버전 오름차순으로 정렬

2. `AVAILABLE_STUDENT_MODELS` (68-75줄) 수정
   - 3개 모델 추가
   - llama → qwen 순, 버전 오름차순으로 정렬

**완료 기준**:
- 모든 모델이 올바른 순서로 정렬됨
- 추가/삭제가 정확히 반영됨

### Phase 3: config/config.py 수정

**작업 내용**:
1. `AVAILABLE_TEACHER_MODELS` (23-34줄) 수정
   - `config/models.py`와 동일하게 수정

2. `AVAILABLE_STUDENT_MODELS` (89-96줄) 수정
   - `config/models.py`와 동일하게 수정

3. `MODEL_NAME_TO_SHORT` (166-173줄) 수정
   - 3개 모델 매핑 추가
   - llama → qwen 순, 버전 오름차순으로 정렬

**완료 기준**:
- `config/models.py`와 일관성 유지
- SFT 매핑이 올바르게 추가됨

### Phase 4: config/sft.py 수정

**작업 내용**:
1. `MODEL_NAME_TO_SHORT` (7-14줄) 수정
   - 3개 모델 매핑 추가
   - llama → qwen 순, 버전 오름차순으로 정렬

**완료 기준**:
- 새 모델에 대한 short name 매핑 추가됨
- 정렬 순서가 일관됨

### Phase 5: 검증

**작업 내용**:
1. 파일 간 일관성 확인
   - `config/models.py`와 `config/config.py`의 모델 목록 동일 여부
   - `config/config.py`와 `config/sft.py`의 매핑 일치 여부

2. Python 구문 검증
   ```bash
   python -m py_compile config/models.py
   python -m py_compile config/config.py
   python -m py_compile config/sft.py
   ```

3. 모델 목록 출력 테스트
   ```python
   from config.models import AVAILABLE_TEACHER_MODELS, AVAILABLE_STUDENT_MODELS
   from config.sft import MODEL_NAME_TO_SHORT

   print("Teacher Models:", AVAILABLE_TEACHER_MODELS)
   print("Student Models:", AVAILABLE_STUDENT_MODELS)
   print("SFT Mappings:", MODEL_NAME_TO_SHORT)
   ```

**완료 기준**:
- 모든 파일이 구문 오류 없이 컴파일됨
- 모델 목록이 올바르게 출력됨
- 파일 간 일관성 확인 완료

### Phase 6: 문서 업데이트

**작업 내용**:
1. **README.md 업데이트** (89-99줄, 115-122줄)
   - Teacher 모델 테이블 업데이트 (새 모델 추가, 삭제 반영)
   - Student 모델 테이블 업데이트 (새 모델 추가)
   - 정렬 순서 반영 (llama → qwen)

2. **ARCHITECTURE.md 업데이트** (722-730줄, 769-775줄)
   - AVAILABLE_TEACHER_MODELS 목록 업데이트
   - AVAILABLE_STUDENT_MODELS 목록 업데이트
   - 정렬 순서 반영

**완료 기준**:
- 두 문서의 모델 목록이 코드와 일치
- 추가/삭제된 모델이 정확히 반영됨
- 정렬 순서가 일관됨

### Phase 7: .gitignore 확인

**작업 내용**:
1. `.gitignore` 파일 확인
   - `USAGE.md`가 이미 포함되어 있는지 확인 (49번째 줄)
   - 필요시 추가

**완료 기준**:
- `USAGE.md`가 `.gitignore`에 포함되어 GitHub 업로드 차단됨

**현재 상태**:
- ✅ 이미 `.gitignore` 49번째 줄에 `USAGE.md` 존재

### Phase 8: requirements.txt 검사

**작업 내용**:
1. `requirements.txt` 의존성 검사
   - 현재 패키지 버전 확인
   - 새로운 70B/72B 모델 사용에 필요한 추가 패키지 확인
   - 버전 호환성 검증

2. 검사 항목:
   - `transformers>=4.40.0`: 최신 모델 지원 여부
   - `torch>=2.2.0`: 대형 모델 지원
   - `accelerate>=0.28.0`: multi-GPU 지원

**완료 기준**:
- 모든 의존성이 최신 모델을 지원함
- 버전 충돌 없음
- 필요시 버전 업데이트 권장사항 문서화

---

## 4. 위험 요소 및 대응

### 위험 요소

| 위험 | 영향도 | 확률 | 대응 방안 |
|------|--------|------|-----------|
| 파일 간 불일치 발생 | High | Medium | Phase 5에서 철저히 검증 |
| SFT 매핑 누락 | Medium | Low | 체크리스트로 확인 |
| 정렬 순서 오류 | Low | Low | 명확한 정렬 규칙 적용 |

### 롤백 계획
- Git을 사용 중이므로 문제 발생 시 `git checkout` 또는 `git revert`로 복구

---

## 5. 체크리스트

### config/models.py
- [ ] AVAILABLE_TEACHER_MODELS에서 `openai/gpt-oss-20b` 삭제
- [ ] AVAILABLE_TEACHER_MODELS에 3개 모델 추가
- [ ] AVAILABLE_TEACHER_MODELS 정렬 (llama → qwen, 버전 오름차순)
- [ ] AVAILABLE_STUDENT_MODELS에 3개 모델 추가
- [ ] AVAILABLE_STUDENT_MODELS 정렬 (llama → qwen, 버전 오름차순)

### config/config.py
- [ ] AVAILABLE_TEACHER_MODELS에서 `openai/gpt-oss-20b` 삭제
- [ ] AVAILABLE_TEACHER_MODELS에 3개 모델 추가
- [ ] AVAILABLE_TEACHER_MODELS 정렬
- [ ] AVAILABLE_STUDENT_MODELS에 3개 모델 추가
- [ ] AVAILABLE_STUDENT_MODELS 정렬
- [ ] MODEL_NAME_TO_SHORT에 3개 매핑 추가
- [ ] MODEL_NAME_TO_SHORT 정렬

### config/sft.py
- [ ] MODEL_NAME_TO_SHORT에 3개 매핑 추가
- [ ] MODEL_NAME_TO_SHORT 정렬

### 검증
- [ ] Python 구문 검증 (3개 파일)
- [ ] 파일 간 일관성 확인
- [ ] 모델 목록 출력 테스트

### 문서 업데이트
- [ ] README.md Teacher 모델 테이블 업데이트
- [ ] README.md Student 모델 테이블 업데이트
- [ ] ARCHITECTURE.md Teacher 모델 목록 업데이트
- [ ] ARCHITECTURE.md Student 모델 목록 업데이트

### .gitignore 확인
- [x] USAGE.md가 .gitignore에 포함되어 있는지 확인 (이미 존재)

### requirements.txt 검사
- [ ] transformers 버전 확인
- [ ] torch 버전 확인
- [ ] accelerate 버전 확인
- [ ] 70B/72B 모델 지원 여부 확인

---

## 6. 예상 소요 시간

| Phase | 예상 시간 |
|-------|----------|
| Phase 2: config/models.py 수정 | 5분 |
| Phase 3: config/config.py 수정 | 5분 |
| Phase 4: config/sft.py 수정 | 3분 |
| Phase 5: 검증 | 5분 |
| Phase 6: 문서 업데이트 | 10분 |
| Phase 7: .gitignore 확인 | 1분 |
| Phase 8: requirements.txt 검사 | 3분 |
| **총합** | **약 32분** |

---

## 7. 성공 기준

1. ✅ 모든 Teacher model 목록이 올바르게 업데이트됨
2. ✅ 모든 Student model 목록이 올바르게 업데이트됨
3. ✅ SFT 모델 매핑이 추가됨
4. ✅ 모든 파일에서 정렬 규칙이 일관되게 적용됨
5. ✅ Python 구문 오류 없음
6. ✅ 파일 간 일관성 유지
7. ✅ README.md와 ARCHITECTURE.md가 업데이트됨
8. ✅ .gitignore에 USAGE.md 포함 확인
9. ✅ requirements.txt 의존성 검증 완료

---

## 8. 후속 작업

- 새로운 70B/72B 모델에 대한 학습/평가 파이프라인 테스트
- 필요시 메모리 최적화 설정 추가 (70B/72B 모델은 리소스를 많이 사용)
- HuggingFace Hub에 SFT fine-tuned 모델 업로드 여부 결정
- 70B/72B 모델용 추가 문서화 (GPU 요구사항, 최적화 팁 등)

---

## 9. 참고사항

### 모델 크기 비교
- 3B/4B/7B/8B: 일반적인 학습/테스트 환경에서 사용 가능
- 14B: 중형 GPU 필요 (A100 40GB 권장)
- **70B/72B**: 대형 GPU 필요 (A100 80GB 또는 multi-GPU 설정)

### 리소스 요구사항
새로 추가되는 70B/72B 모델들은 상당한 GPU 메모리가 필요하므로, 실행 환경을 고려하여 사용해야 합니다.

### .gitignore 현재 상태
- ✅ `USAGE.md`는 이미 [.gitignore:49](.gitignore#L49)에 포함되어 있음
- 추가 작업 불필요

### requirements.txt 현재 상태
현재 의존성 버전:
- `transformers>=4.40.0`: ✅ Llama 3.1/3.2/3.3 및 Qwen2.5/3 지원
- `torch>=2.2.0`: ✅ 대형 모델 지원
- `accelerate>=0.28.0`: ✅ multi-GPU 및 메모리 최적화 지원

**결론**: 현재 requirements.txt는 70B/72B 모델을 지원하기에 충분합니다.

---

**계획 업데이트 완료 - 검토 및 승인 대기 중**
