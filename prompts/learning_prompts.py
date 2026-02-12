"""학습 파이프라인 프롬프트 모듈.

이 모듈은 ID-MAS의 반복적 스캐폴딩 학습 파이프라인에서 사용하는
모든 프롬프트 템플릿을 정의합니다. 교사 모델의 지도 하에 학생 모델이
단계적으로 문제 해결 능력을 향상시키는 ReAct 스타일 학습 루프를 지원합니다.

프롬프트 (파이프라인 실행 순서):
    Step 1 — 학생 초기 응답:
        - SCAFFOLDING_SYSTEM_PROMPT: 학생 모델 시스템 프롬프트

    Step 2 — 교사 PO 평가 (system/user 분리):
        - TEACHER_INTERVENTION_SYSTEM_PROMPT / _USER_PROMPT: 수행목표 평가

    Step 3 — 스캐폴딩 아티팩트 생성 (system/user 분리, PO 미충족 시):
        - SCAFFOLDING_ARTIFACT_SYSTEM_PROMPT / _USER_PROMPT: 스캐폴딩 아티팩트 생성

    Step 4 — 학생 재응답 (Iteration 2~5):
        - STUDENT_FEEDBACK_RESPONSE_PROMPT: 피드백 기반 개선 응답 (system 전용)

    Step 5a — 교사 긍정 피드백 (Case A/B, PO 충족 시):
        - TEACHER_POSITIVE_FEEDBACK_SYSTEM_PROMPT / _USER_PROMPT: 긍정 피드백 생성

    Step 5b — 학생 Self-Refinement (Case A/B):
        - STUDENT_SELF_REFINEMENT_PROMPT: Self-Refinement 응답 (system 전용)

    Step 5c — 교사 최종 해답 (Case C, 최대 반복 후 실패):
        - TEACHER_FINAL_SOLUTION_SYSTEM_PROMPT / _USER_PROMPT: 최종 해답 제공

학습 케이스:
    Case A: 첫 시도에 정답 → Self-Refined 응답 사용
    Case B: 스캐폴딩 후 정답 → Self-Refined 응답 사용
    Case C: 최대 반복 후 실패 → 교사 해답 제공 (last iteration summary 기반)
"""


# ==============================================================================
# Step 1: 스캐폴딩 시스템 프롬프트 (Scaffolding System Prompt)
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
# Step 2: 교사 개입 프롬프트 (Teacher Intervention) — 평가 전용
# ------------------------------------------------------------------------------
# ReAct 스타일 학습 루프에서 교사 모델이 학생 응답을 평가합니다.
# 피드백 생성은 SCAFFOLDING_ARTIFACT 프롬프트에서 담당합니다.
# system: 역할 정의 / user: 입력 데이터 + 지시 + 출력 형식
# ==============================================================================

TEACHER_INTERVENTION_SYSTEM_PROMPT = """You are a teacher supporting the learning of a student.

Your role is to evaluate the student's response against the established performance objectives. You must monitor the student's reasoning steps to ensure they meet the performance objectives."""

TEACHER_INTERVENTION_USER_PROMPT = """[Input Data]
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

Output ONLY valid JSON."""


# ==============================================================================
# Step 3: 스캐폴딩 아티팩트 생성 프롬프트
# ------------------------------------------------------------------------------
# Dick & Carey 모델 기반의 스캐폴딩 아티팩트를 생성합니다.
# 실패한 수행목표에 대해 HOT/LOT 기술 유형에 따라
# 적절한 스캐폴딩(전략 제안, 부분 예시, 피드백 등)을 설계합니다.
# system: 역할 정의 / user: 입력 데이터 + 지시 + 출력 형식
# ==============================================================================

SCAFFOLDING_ARTIFACT_SYSTEM_PROMPT = """You are an instructional design expert (Dick & Carey model) creating a Scaffolding Artifact to help a student improve.

Your role is to design pedagogical scaffolding for Performance Objectives that the student failed to meet. This scaffolding will be stored as a "Scaffolding Artifact" that the student can reference in their next attempt."""

SCAFFOLDING_ARTIFACT_USER_PROMPT = """[Input Data]
- Problem: {problem_text}
- Student's Response: {student_response}
- Performance Objectives Evaluation: {po_evaluation}

[Previous Iteration Summaries]
{previous_iteration_summaries}

[Instructions]
1. Select scaffolding targets: Generate scaffolding ONLY for Performance Objectives where is_satisfied is false in the evaluation. Do NOT create scaffolding for satisfied POs. The number of [Scaffolding for Task] sections must exactly match the number of unsatisfied POs.

2. Classify skill level: For each unmet PO, determine if it requires:
   - HOT (High-Order Thinking): Analyze, Evaluate, Create
   - LOT (Low-Order Thinking): Remember, Understand, Apply

3. Design appropriate scaffolding:

   For HOT skills:
   - Strategy suggestion: Propose an approach or reasoning strategy
   - Partial worked example: Show partial reasoning but STOP before computing the final numerical result. Leave the last calculation step incomplete (e.g., "3 × 4 = ?"). Never show the complete answer.
   - Key attention points: What the student should focus on

   For LOT skills:
   - Missed concept/information: Explicitly state what the student missed
   - Brief explanation: Provide a concise explanation to minimize cognitive load

4. Generate integrated narrative feedback: Write a single cohesive feedback paragraph that integrates ALL unsatisfied POs into a natural narrative. This feedback will be directly provided to the student. It must include:
   (a) Error analysis: What the student got wrong and why
   (b) Improvement direction: The fundamental approach the student should take
   (c) Verification steps: Concrete intermediate steps the student should check

5. Do NOT reveal correct answers — guide reasoning, don't solve.

6. Generate iteration summary: Write a concise summary of THIS iteration that captures:
   (a) What the student attempted and how (brief response summary)
   (b) Which Performance Objectives were not met and why
   (c) What scaffolding was provided and the key guidance direction
   This summary will be used as context for the next iteration's scaffolding generation.

[Output Format - Structured Text]

[Instructional Goal]
{instructional_goal}

[Instructional Analysis]
{task_analysis}

[Scaffolding for Task [1] (High Order Skill / Low Order Skill)]:
- Target Objective: <Copy the FULL text of the unmet Performance Objective VERBATIM from the evaluation. Do NOT paraphrase, summarize, or use subskill names from the Instructional Analysis.>
- Cognitive Level: Analyze / Evaluate / Create / Remember / Understand / Apply
- Failure Analysis: <why the student failed this objective>

  If High Order Skill:
  - Suggested Strategy:
    (a) Strategy 1: <approach or reasoning strategy>
        - Partial worked example (stop before the final answer):
          <partial reasoning showing key steps — leave the last calculation incomplete>
    (b) Strategy 2: <alternative approach>
        - Teacher's reasoning clarification (why this strategy should be considered):
          <explanation of the APPROACH only — do NOT compute or reveal the final answer>
  - Key Attention Points: <what the student should focus on>

  If Low Order Skill:
  - Missed Concept/Information: <what the student missed>
  - Brief Explanation: <concise explanation to minimize cognitive load>

(If there are additional unsatisfied POs, repeat with [Scaffolding for Task [2]], [3], etc.
Generate ONLY as many sections as there are unsatisfied POs. If only 1 PO is unsatisfied, output only 1 section.)

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
1. Do NOT reveal correct answers or complete solutions.
2. Focus on guiding the reasoning process.
3. The [Feedback] section is the primary feedback delivered to the student — make it specific, actionable, and educational.
4. The [Iteration Summary] must capture BOTH the student's attempt AND the scaffolding provided.
5. In Strategy 2 (alternative approach), explain ONLY the reasoning approach. Do NOT compute the final answer or show complete numerical calculations that lead to the answer.
6. The partial worked example must stop at least ONE step before the final answer. Show the setup and method, not the result.

Output ONLY the structured text above. Do NOT include JSON formatting.
"""


# ==============================================================================
# Step 4: 피드백에 대한 학생 응답 프롬프트 (system 전용)
# ------------------------------------------------------------------------------
# SCAFFOLDING_SYSTEM_PROMPT를 기반으로 교사의 피드백을 통합하여
# 학생 모델이 개선된 응답을 생성합니다.
# system message로 사용됩니다 (user message는 input만 전달).
# dataset_prompt(original instruction)는 호출 측에서 prepend합니다.
# ==============================================================================

STUDENT_FEEDBACK_RESPONSE_PROMPT = """{scaffolding_system_prompt}

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
# Step 5a: 교사 긍정 피드백 프롬프트 (Case A/B Self-Refinement용)
# ------------------------------------------------------------------------------
# 학생이 모든 PO를 충족한 경우, 교사가 강점 요약 및 개선 제안을 제공합니다.
# system: 역할 정의 / user: 입력 데이터 + 지시 + 출력 형식
# ==============================================================================

TEACHER_POSITIVE_FEEDBACK_SYSTEM_PROMPT = """You are a teacher providing constructive feedback to strengthen a student's already satisfactory response."""

TEACHER_POSITIVE_FEEDBACK_USER_PROMPT = """[Problem]
{problem_text}

[Student's Response]
{student_response}

[Performance Objectives Evaluation]
{po_evaluation}

[Instructions]
The student's response has satisfied all Performance Objectives. Provide feedback that will help the student further strengthen their response:

1. Strengths Summary: For each satisfied PO, briefly describe what the student did well, citing specific parts of their response.

2. Enhancement Suggestions: Identify 2-3 concrete ways the student could improve their response quality:
   - Clearer reasoning structure or explanations
   - More explicit connections between steps
   - Better demonstration of the underlying concepts
   - More rigorous justification of key steps

3. Integration Guidance: Explain how to incorporate these improvements naturally into the existing solution without changing the core logic or final answer.

[Output Format]

[Strengths]
- PO 1: <what was done well>
- PO 2: <what was done well>
...

[Enhancement Suggestions]
1. <specific improvement suggestion>
2. <specific improvement suggestion>
3. <specific improvement suggestion>

[Integration Guidance]
<How to naturally incorporate these improvements into the response>

Output ONLY the structured text above.
"""


# ==============================================================================
# Step 5b: 학생 Self-Refinement 프롬프트 (Case A/B) — system 전용
# ------------------------------------------------------------------------------
# 모든 PO를 충족한 학생이 교사의 긍정 피드백을 반영하여
# 응답을 개선합니다. 최종 답은 유지하되 추론 과정의 질을 높입니다.
# system message로 사용됩니다 (user message는 input만 전달).
# ==============================================================================

STUDENT_SELF_REFINEMENT_PROMPT = """{scaffolding_system_prompt}

[Teacher's Feedback on Your Response]
Your teacher has evaluated your response and confirmed that it meets all performance objectives.
The following feedback highlights your strengths and suggests ways to further improve your response:

{positive_feedback}

[Instructions]
1. Keep your correct reasoning and final answer unchanged.
2. Integrate the enhancement suggestions naturally into your solution.
3. Strengthen the clarity, structure, and justification of your reasoning.
4. Demonstrate deeper understanding of the underlying concepts.
5. Your improved response should be a complete, standalone solution (not a diff or list of changes).

[Output Format]
Write your complete improved solution following the original output format:
- Instructional goal alignment: [how this solution demonstrates the instructional goal]
- Step-by-step reasoning: [your improved detailed solution]
- Final answer: "The answer is \\boxed{{your final answer}}"
"""


# ==============================================================================
# Step 5c: 교사 최종 해답 프롬프트 (Case C)
# ------------------------------------------------------------------------------
# 최대 반복 횟수 후에도 학생이 정답에 도달하지 못한 경우,
# 교사가 완전한 해답을 제공합니다.
# system: 역할 정의 / user: 입력 데이터 + 지시 + 출력 형식
# ==============================================================================

TEACHER_FINAL_SOLUTION_SYSTEM_PROMPT = """You are a teacher providing a complete, correct solution after the student failed to solve the problem after {max_iterations} attempts.

The solution should be what an expert student would produce - clear, complete, and pedagogically valuable."""

TEACHER_FINAL_SOLUTION_USER_PROMPT = """[Problem]
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
