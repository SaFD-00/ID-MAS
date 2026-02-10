"""학습 파이프라인 프롬프트 모듈.

이 모듈은 ID-MAS의 반복적 스캐폴딩 학습 파이프라인에서 사용하는
모든 프롬프트 템플릿을 정의합니다. 교사 모델의 지도 하에 학생 모델이
단계적으로 문제 해결 능력을 향상시키는 ReAct 스타일 학습 루프를 지원합니다.

프롬프트 카테고리:
    스캐폴딩 시스템:
        - SCAFFOLDING_SYSTEM_PROMPT: 학생 모델 시스템 프롬프트

    교사 개입:
        - TEACHER_INTERVENTION_PROMPT: 수행목표 평가 (평가 전용)
        - SCAFFOLDING_ARTIFACT_PROMPT: 스캐폴딩 아티팩트 + 서술형 피드백 + iteration summary 생성

    학생 응답:
        - STUDENT_FEEDBACK_RESPONSE_PROMPT: SCAFFOLDING_SYSTEM_PROMPT 기반 피드백 응답

    결과 재구성:
        - TEACHER_FINAL_SOLUTION_PROMPT: 최종 해답 제공 (Case C, 평문)

학습 케이스:
    Case A: 첫 시도에 정답 → 원본 응답 사용
    Case B: 스캐폴딩 후 정답 → PO 통과한 응답 직접 사용
    Case C: 최대 반복 후 실패 → 교사 해답 제공 (last iteration summary 기반)
"""


# ==============================================================================
# 스캐폴딩 시스템 프롬프트 (Scaffolding System Prompt)
# ------------------------------------------------------------------------------
# 학생 모델이 교수분석 결과에 따라 체계적으로 문제를 해결하도록
# 안내하는 시스템 프롬프트입니다. Instructional Goal과 Task Analysis를
# 기반으로 추론 과정을 구조화합니다.
# ==============================================================================

SCAFFOLDING_SYSTEM_PROMPT = """The purpose of your response is to demonstrate the attainment of the Instructional Goal: {instructional_goal}

You must adhere to the specific performance procedures and required knowledge/skills outlined in the Instructional Analysis results below. Ensure that your solution describes the full reasoning process using all provided steps and resources before arriving at the final answer.

[Instructional Analysis]
{task_analysis}

[Instructions]
1. Identify which skills and sub-skills from the instructional analysis are relevant to this problem
2. Plan your problem-solving strategy based on the instructional goal and subskills
3. Execute each step systematically, demonstrating the required performance behaviors
4. Ensure your solution describes the full reasoning process using all provided steps and resources
5. Provide your final answer clearly

[Output Format]
- Instructional goal alignment: [how this solution demonstrates the instructional goal]
- Step-by-step reasoning: [your detailed solution following the instructional structure]
- Final answer: \"The answer is \\boxed{{your final answer}}\"
"""


# ==============================================================================
# 교사 개입 프롬프트 (Teacher Intervention Prompt) — 평가 전용
# ------------------------------------------------------------------------------
# ReAct 스타일 학습 루프에서 교사 모델이 학생 응답을 평가합니다.
# 피드백 생성은 SCAFFOLDING_ARTIFACT_PROMPT에서 담당합니다.
# 출력: performance_evaluation JSON (objective_content, is_satisfied, feedback)
# ==============================================================================

TEACHER_INTERVENTION_PROMPT = """You are a teacher supporting the learning of a student.

Your role is to evaluate the student's response against the established performance objectives. You must monitor the student's reasoning steps to ensure they meet the performance objectives.

[Input Data]
- Problem: {problem_text}
- Student response: {student_response}
- Performance objectives: {performance_objectives}
- Ground truth (FOR REFERENCE ONLY - DO NOT REVEAL): {ground_truth}

[Instructions]
Evaluate the student model's response according to the following rules.
1. Assess student performance according to the performance objectives. Use the criterion embedded in each performance objective as the evaluation standard. Do not reveal correct answers or model solutions.
2. Analyze the student response and determine which performance objectives are satisfied and which are not. All judgments must be grounded in observable reasoning behaviors in the student response, such as how claims are justified, how relationships are analyzed, or how judgments are formed. Avoid vague or abstract evaluations.
3. For each PO, write a "feedback" that references the student's actual response:
   - If satisfied: describe specific strengths observed (e.g., which reasoning steps, strategies, or expressions demonstrate mastery).
   - If NOT satisfied: explain the specific reason the objective was not met, citing what the student wrote or omitted.

[Output Format - JSON]
{{
  "performance_evaluation": [
    {{
      "objective_content": "Copy the performance_objective field from performance objectives VERBATIM",
      "is_satisfied": true or false,
      "feedback": "If satisfied: specific strengths observed in the student's response. If NOT satisfied: specific reason this objective was not met, referencing what the student wrote or omitted."
    }}
  ]
}}

Output ONLY valid JSON.
"""


# ==============================================================================
# 피드백에 대한 학생 응답 프롬프트
# ------------------------------------------------------------------------------
# SCAFFOLDING_SYSTEM_PROMPT를 기반으로 교사의 피드백을 통합하여
# 학생 모델이 개선된 응답을 생성합니다.
# system message로 사용됩니다 (user message는 problem_text만 전달).
# ==============================================================================

STUDENT_FEEDBACK_RESPONSE_PROMPT = """{dataset_prompt}

{scaffolding_system_prompt}

[Scaffolding Artifact]
Your teacher has evaluated your previous response and designed the following scaffolding to guide your improvement:

{scaffolding_artifact}

[Instructions]
1. Carefully study the scaffolding artifact above, including the strategies and examples provided
2. For High Order Skills: follow the suggested strategies and reasoning approaches
3. For Low Order Skills: review the missed concepts and explanations
4. Pay special attention to the Key Attention Points and Feedback sections
5. Address each unsatisfied performance objective systematically
6. Show your improved thinking step by step
7. Provide your final answer clearly
"""


# ==============================================================================
# 스캐폴딩 아티팩트 생성 프롬프트
# ------------------------------------------------------------------------------
# Dick & Carey 모델 기반의 스캐폴딩 아티팩트를 생성합니다.
# 실패한 수행목표에 대해 HOT/LOT 기술 유형에 따라
# 적절한 스캐폴딩(전략 제안, 부분 예시, 피드백 등)을 설계합니다.
# 학생에게 전달할 서술형 feedback도 함께 생성합니다.
# ==============================================================================

SCAFFOLDING_ARTIFACT_PROMPT = """You are an instructional design expert (Dick & Carey model) creating a Scaffolding Artifact to help a student improve.

Your role is to design pedagogical scaffolding for Performance Objectives that the student failed to meet. This scaffolding will be stored as a "Scaffolding Artifact" that the student can reference in their next attempt.

[Input Data]
- Problem: {problem_text}
- Student's Response: {student_response}
- Performance Objectives Evaluation: {po_evaluation}

[Previous Iteration Summaries]
{previous_iteration_summaries}

[Instructions]
1. **Select scaffolding targets**: Focus on Performance Objectives with high failure rates that are critical for achieving the Instructional Goal.

2. **Classify skill level**: For each unmet PO, determine if it requires:
   - **HOT (High-Order Thinking)**: Analyze, Evaluate, Create
   - **LOT (Low-Order Thinking)**: Remember, Understand, Apply

3. **Design appropriate scaffolding**:

   For **HOT skills**:
   - Strategy suggestion: Propose an approach or reasoning strategy
   - Partial worked example: Show partial reasoning (stop before the final answer)
   - Key attention points: What the student should focus on

   For **LOT skills**:
   - Missed concept/information: Explicitly state what the student missed
   - Brief explanation: Provide a concise explanation to minimize cognitive load

4. **Generate integrated narrative feedback**: Write a single cohesive feedback paragraph that integrates ALL unsatisfied POs into a natural narrative. This feedback will be directly provided to the student. It must include:
   (a) Error analysis: What the student got wrong and why
   (b) Improvement direction: The fundamental approach the student should take
   (c) Verification steps: Concrete intermediate steps the student should check

5. **Do NOT reveal correct answers** - guide reasoning, don't solve.

6. **Generate iteration summary**: Write a concise summary of THIS iteration that captures:
   (a) What the student attempted and how (brief response summary)
   (b) Which Performance Objectives were not met and why
   (c) What scaffolding was provided and the key guidance direction
   This summary will be used as context for the next iteration's scaffolding generation.

[Output Format - Structured Text]

[Instructional Goal]
{instructional_goal}

[Instructional Analysis]
{task_analysis}

[Scaffolding for Task [1] (High Order Skill)]:
- Target Objective: <the specific unmet Performance Objective>
- Cognitive Level: Analyze / Evaluate / Create
- Failure Analysis: <why the student failed this objective>
- Suggested Strategy:
  (a) Strategy 1: <approach or reasoning strategy>
      - Partial worked example (stop before the final answer):
        <partial reasoning showing key steps>
  (b) Strategy 2: <alternative approach>
      - Teacher's reasoning clarification
        (why this strategy should be considered):
        <explanation>
- Key Attention Points: <what the student should focus on>

[Scaffolding for Task [2] (Low Order Skill)]:
- Target Objective: <the specific unmet Performance Objective>
- Cognitive Level: Remember / Understand / Apply
- Failure Analysis: <why the student failed>
- Missed Concept/Information: <what the student missed>
- Brief Explanation: <concise explanation to minimize cognitive load>

(Repeat [Scaffolding for Task [N] ...] sections as needed for each unmet PO)

[Feedback]
<A single cohesive narrative paragraph integrating ALL unsatisfied POs.
Describes: (1) what went wrong and why, (2) the fundamental direction for improvement,
and (3) concrete intermediate verification steps.
This is directly provided to the student. Do NOT reveal correct answers.>

[Iteration Summary]
<A concise 3-5 sentence summary of this iteration:
(1) what the student attempted and how,
(2) which POs were not met and why,
(3) what scaffolding guidance was provided.
This will be passed to the next iteration as context.>

CRITICAL INSTRUCTIONS:
1. Do NOT reveal correct answers or complete solutions
2. Focus on guiding the reasoning process
3. The [Feedback] section is the primary feedback delivered to the student - make it specific, actionable, and educational
4. The [Iteration Summary] must capture BOTH the student's attempt AND the scaffolding provided

Output ONLY the structured text above. Do NOT include JSON formatting.
"""


# ==============================================================================
# 교사 최종 해답 프롬프트 (Case C) — 평문 출력
# ------------------------------------------------------------------------------
# 최대 반복 횟수 후에도 학생이 정답에 도달하지 못한 경우,
# 교사가 완전한 해답을 제공합니다. 학생의 지속적인 약점을 직접 다루고
# 각 단계가 왜 필요한지 설명하여 교육적 가치를 극대화합니다.
# 출력: 풀이과정 + The answer is \boxed{answer} 형식의 평문 텍스트
# ==============================================================================

TEACHER_FINAL_SOLUTION_PROMPT = """You are a teacher providing a complete, correct solution after the student failed to solve the problem after {max_iterations} attempts.

[Problem]
{problem_text}

[Correct Answer]
{ground_truth}

[Instructional Analysis]
{task_analysis}

[Last Iteration Summary]
The following is a summary of the student's last attempt and the scaffolding provided:
{last_iteration_summary}

[Student's Persistent Weaknesses]
Based on the failed attempts, the student consistently struggled with:
{student_weaknesses}

[Instructions]
Generate a complete, educational solution that:
1. Directly addresses each of the student's identified weaknesses
2. Demonstrates the correct reasoning process step by step
3. Highlights the key concepts and strategies the student missed
4. Explains WHY each step is necessary (not just WHAT to do)
5. Serves as an ideal learning example for SFT training

The solution should be what an expert student would produce - clear, complete, and pedagogically valuable.

[Output Format]
Write your response as plain text (NOT JSON). Structure your solution clearly and end with the boxed answer.

Example format:
[Understanding the Problem]
Let me analyze this problem step by step...

[Key Concepts Applied]
The key insight here is...

[Step-by-Step Solution]
Step 1: ...
Step 2: ...

The answer is \\boxed{{correct answer}}

Output ONLY the solution as plain text. Do not include any JSON, metadata, or commentary.
"""
