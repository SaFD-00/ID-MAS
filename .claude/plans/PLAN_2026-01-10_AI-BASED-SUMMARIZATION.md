# AI 기반 대화 히스토리 축약 구현 계획

## 목표
Rule-based 축약을 Teacher 모델 기반 AI 축약으로 대체하여 중요한 학습 포인트를 보존

## 구현 항목

### 1. 새 프롬프트 정의
**파일**: `prompts/learning_prompts.py`

```python
CONVERSATION_SUMMARIZATION_PROMPT = """You are a teacher summarizing a tutoring session.

[Full Conversation History]
{conversation_history}

[Instructions]
Summarize this conversation focusing on:
1. Key mistakes the student made in each attempt
2. The core misconceptions revealed
3. How the student's approach evolved
4. The final answer attempted

Keep the summary concise (under 500 words) while preserving:
- Specific errors (not general descriptions)
- The student's reasoning patterns
- Failed vs partially correct steps

[Output Format]
Return a structured summary:

ITERATION 1:
- Approach: [brief description]
- Key Error: [specific mistake]
- Answer: [their answer]

ITERATION 2:
...

OVERALL PATTERN:
- Main weakness: [identified pattern]
- Misconception: [core misunderstanding]
"""
```

### 2. TeacherModel에 새 메서드 추가
**파일**: `learning_loop/teacher_model.py`

```python
def summarize_conversation_for_reconstruction(
    self,
    conversation_history: List[Dict]
) -> str:
    """
    AI 기반 대화 히스토리 축약

    Teacher 모델이 중요한 학습 포인트를 파악하여 축약
    """
    # 1. 전체 히스토리를 포맷팅
    formatted = self._format_conversation_history(conversation_history)

    # 2. Teacher 모델로 축약 요청
    prompt = CONVERSATION_SUMMARIZATION_PROMPT.format(
        conversation_history=formatted
    )

    # 3. 일반 텍스트로 생성 (JSON 아님)
    summary = self.llm.generate(prompt)

    return summary
```

### 3. summarize_and_reconstruct 수정
**파일**: `learning_loop/teacher_model.py`

```python
def summarize_and_reconstruct(...):
    # AI 기반 축약 사용
    summarized_history = self.summarize_conversation_for_reconstruction(
        conversation_history
    )

    prompt = SUMMARY_RECONSTRUCTION_PROMPT.format(
        ...
        conversation_history=summarized_history  # 축약된 버전 사용
    )
```

## 장점

| 항목 | 효과 |
|------|------|
| 맥락 보존 | 중요한 실수와 패턴 유지 |
| 길이 최적화 | 불필요한 반복 제거 |
| 학습 품질 | 재구성 응답 품질 향상 |

## 주의사항

1. **API 호출 증가**: 축약을 위한 추가 호출 1회
2. **Fallback 필요**: 축약 실패 시 기존 rule-based 사용
3. **토큰 비용**: 큰 모델 사용 시 비용 증가

## 작업 순서

1. [ ] `CONVERSATION_SUMMARIZATION_PROMPT` 프롬프트 추가
2. [ ] `summarize_conversation_for_reconstruction()` 메서드 구현
3. [ ] `summarize_and_reconstruct()` 수정
4. [ ] Fallback 로직 추가 (실패 시 rule-based)
5. [ ] 테스트
