# Reflection: instructional-design 경로 구조 변경 계획

## 세션 요약

**요청**: `data/math/train/instructional-design` → `data/math/train/{teacher_model}/instructional-design` 경로 변경 계획

**결과**: 상세 구현 계획 문서 작성 완료

## 발견 사항

### 1. 코드 중복 발견
- `get_design_output_dir()` 함수가 `config/config.py`와 `config/paths.py`에 중복 정의
- 이 중복은 유지보수 어려움을 초래할 수 있음
- 계획에 중복 해소 방안 포함

### 2. 일관된 구조
- `get_domain_data_dirs()` 함수는 이미 `teacher_model_name` 파라미터를 지원
- `model_dir`은 이미 `{domain}/train/{teacher_model}/{student_model}` 구조
- `design_dir`만 teacher_model 하위로 이동하면 일관성 확보

### 3. 레거시 마이그레이션 패턴 존재
- `main.py`에 이미 레거시 경로 → 새 경로 마이그레이션 로직 존재 (라인 568-572)
- 이 패턴을 확장하여 새로운 마이그레이션에도 적용 가능

## 학습 포인트

1. **중복 코드 검사 중요성**: 변경 전 중복 정의 확인으로 잠재적 버그 예방
2. **기존 패턴 활용**: 레거시 마이그레이션 패턴이 이미 존재하므로 재사용 가능
3. **하위 호환성 고려**: 기본값 설정으로 기존 호출 코드 영향 최소화

## 다음 단계

계획 승인 후 구현 진행:
1. Step 1: config/paths.py 수정
2. Step 2: config/config.py 정리
3. Step 3: main.py 수정
4. Step 4: 문서 업데이트
