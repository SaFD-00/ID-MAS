"""학습 파이프라인 프롬프트 모듈.

이 모듈은 ID-MAS의 반복적 스캐폴딩 학습 파이프라인에서 사용하는
모든 프롬프트 템플릿을 정의합니다. 교사 모델의 지도 하에 학생 모델이
단계적으로 문제 해결 능력을 향상시키는 ReAct 스타일 학습 루프를 지원합니다.

프롬프트 카테고리:
    스캐폴딩 시스템:
        - SCAFFOLDING_SYSTEM_PROMPT: 학생 모델 시스템 프롬프트
        - ITERATIVE_SCAFFOLDING_SYSTEM_PROMPT: 반복 학습용 시스템 프롬프트

    교사 개입:
        - TEACHER_INTERVENTION_PROMPT: 수행목표 평가 및 피드백
        - INITIAL_HINT_PROMPT: 첫 번째 힌트 제공
        - PROGRESSIVE_HINT_PROMPT: 점진적 힌트 제공
        - SCAFFOLDING_ARTIFACT_PROMPT: 스캐폴딩 아티팩트 생성

    학생 응답:
        - STUDENT_FEEDBACK_RESPONSE_PROMPT: 피드백에 대한 응답
        - STUDENT_WITH_HINT_PROMPT: 힌트 기반 응답
        - STUDENT_WITH_ARTIFACT_PROMPT: 스캐폴딩 아티팩트 기반 응답

    결과 재구성:
        - SUMMARY_RECONSTRUCTION_PROMPT: 실패 후 요약 및 재구성
        - SUCCESSFUL_SCAFFOLDING_RECONSTRUCTION_PROMPT: 성공 후 재구성
        - TEACHER_FINAL_SOLUTION_PROMPT: 최종 해답 제공 (Case C)

    유틸리티:
        - SFT_OUTPUT_TEMPLATE: SFT 데이터 출력 형식
        - CONVERSATION_SUMMARIZATION_PROMPT: 대화 요약

학습 케이스:
    Case A: 첫 시도에 정답 → 원본 응답 사용
    Case B: 스캐폴딩 후 정답 → 성공 재구성
    Case C: 최대 반복 후 실패 → 교사 해답 제공
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

[Instructional Analysis (Learning Structure)]
{task_analysis}

[Instructions]
1. Identify which skills and sub-skills from the instructional analysis are relevant to this problem
2. Plan your problem-solving strategy based on the instructional goal and subskills
3. Execute each step systematically, demonstrating the required performance behaviors
4. Ensure your solution describes the full reasoning process using all provided steps and resources
5. Provide your final answer clearly

[Output Format]
Problem-solving strategy and flow:
- Instructional goal alignment: [how this solution demonstrates the instructional goal]
- Relevant skills applied: [list the relevant skills from instructional analysis]
- Step-by-step reasoning: [your detailed solution following the instructional structure]

Answer: [your final answer]
"""


# ==============================================================================
# 교사 개입 프롬프트 (Teacher Intervention Prompt)
# ------------------------------------------------------------------------------
# ReAct 스타일 학습 루프에서 교사 모델이 학생 응답을 평가하고
# 피드백을 통해 학습을 안내합니다. 정답을 직접 알려주지 않고
# 추론 과정을 개선하도록 유도합니다.
# 출력: performance_evaluation, overall_assessment JSON
# ==============================================================================

TEACHER_INTERVENTION_PROMPT = """You are a teacher supporting the learning of a student.

Your role is NOT to provide correct answers, but to generate a reasoning state that guides the student's next response. You must monitor the student's reasoning steps to ensure they meet the established performance objectives.

In cases of non-compliance or error, you must generate tailored, specific feedback to guide the student toward the desired outcome. Your feedback functions as an intermediate thought in a ReAct-style learning loop and must guide the student's next reasoning action.

[Input Data]
- Problem: {problem_text}
- Student response: {student_response}
- Performance objectives: {performance_objectives}
- Ground truth (FOR REFERENCE ONLY - DO NOT REVEAL): {ground_truth}

[Instructions]
1. Assess student performance according to each performance objective.
2. Use the Criterion defined in each performance objective as the evaluation standard.
3. DO NOT reveal correct answers or model solutions.
4. Analyze the student response and determine which performance objectives are satisfied and which are not.
5. All judgments must be grounded in observable reasoning behaviors in the student response, such as how claims are justified, how relationships are analyzed, or how judgments are formed.
6. Avoid vague or abstract evaluations.

For UNSATISFIED performance objectives, provide structured feedback with ALL four components:
  (a) Error Analysis: Identify EXACTLY what area the student got wrong and WHY, referencing specific parts of their actual response.
  (b) Improvement Direction: Suggest a CONCRETE direction and strategy for how to correct and improve.
  (c) Response Comment: Provide a specific comment on the student's previous response process (e.g., "You correctly identified the variables but applied the wrong operation in step 3").
  (d) Metacognitive Prompt: Ask a question that prompts self-reflection (e.g., "Did you consider using X? Think about why it is needed in this context.").

7. Do not provide final conclusions, correct answers, or complete reasoning paths.
8. Instead, specify what type of reasoning process, analytical step, or judgment perspective should be explicitly carried out next.

HOT/LOT Differentiation for Feedback Depth:
- For HOT (High-Order Thinking: Analyze, Evaluate, Create) objectives: Provide MORE detailed feedback, including specific strategies, partial reasoning examples, and guiding frameworks. In early iterations, increase feedback volume with concrete direction.
- For LOT (Low-Order Thinking: Remember, Understand, Apply) objectives: Provide CONCISE feedback focusing on the key concept or fact that was missed. Keep it brief to minimize cognitive load.

For SATISFIED performance objectives:
- Provide a brief positive comment acknowledging what the student did well (e.g., "Correctly identified the key variables and applied the formula accurately").

IMPORTANT: When describing student errors or suggesting improvements, use SPECIFIC and CONCRETE vocabulary.
- BAD: "Your approach needs improvement" / "Think more carefully" / "Review your reasoning"
- GOOD: "You failed to isolate the variable x by dividing both sides by 3" / "Apply the distributive property to expand (a+b)^2" / "Your step 2 incorrectly assumes linearity when the relationship is quadratic"

CRITICAL: The "objective_content" field MUST contain the EXACT text from the input performance objectives.
Do NOT generate new descriptions. Copy the Behavior text from the provided Performance Objectives word-for-word.
Do NOT paraphrase, summarize, or rewrite the objective in any way.

[Output Format - JSON]
{{
  "performance_evaluation": [
    {{
      "objective_content": "MUST be the EXACT text from the performance objectives. Copy the Behavior field verbatim. Do NOT paraphrase, summarize, or rewrite.",
      "is_satisfied": true or false,
      "reason_for_unmet_objective": "Detailed description of the cause if false; null if true",
      "feedback": {{
        "error_analysis": "What specific area the student got wrong and why, referencing their actual response (if false; null if true)",
        "improvement_direction": "Concrete direction and strategy for how to correct and improve (if false; null if true)",
        "response_comment": "Specific comment on the student's previous response process (if false; positive comment if true)",
        "metacognitive_prompt": "Question to prompt self-reflection, e.g., 'Did you consider using X? Think about why it is needed.' (if false; null if true)"
      }}
    }}
  ],
  "overall_assessment": {{
    "objectives_met": "X of Y objectives satisfied",
    "all_satisfied": true or false,
    "primary_weakness": "Main area needing improvement if any; null if all satisfied",
    "recommended_focus": "What the student should focus on next if not all satisfied; null if complete"
  }}
}}

CRITICAL INSTRUCTIONS FOR JSON OUTPUT:
1. Your response MUST be ONLY valid JSON - no additional text before or after
2. Do NOT include explanations, comments, markdown code blocks, or any text outside the JSON
3. Do NOT include LaTeX expressions, mathematical notation, or equations outside JSON string values
4. Ensure ALL brackets {{ }}, [ ], and quotes are properly closed
5. If you need to include mathematical expressions, place them INSIDE JSON string values with proper escaping

Example of CORRECT response:
{{
  "performance_evaluation": [...],
  "overall_assessment": {{...}}
}}

Example of INCORRECT response (DO NOT DO THIS):
Here is my evaluation:
{{
  "performance_evaluation": [...],
  "overall_assessment": {{...}}
}}
Additional notes about the evaluation...

Output ONLY the JSON object above. Do not include any additional text, explanation, or commentary outside the JSON structure.
"""


# ==============================================================================
# 피드백에 대한 학생 응답 프롬프트
# ------------------------------------------------------------------------------
# 교사의 평가와 피드백을 받은 학생 모델이
# 피드백을 반영하여 개선된 응답을 생성합니다.
# ==============================================================================

STUDENT_FEEDBACK_RESPONSE_PROMPT = """You are a student learning to solve problems with teacher guidance.

Your teacher has evaluated your previous response and provided feedback to guide your thinking. You must carefully consider this feedback and improve your response.

[Problem]
{problem_text}

[Your Previous Response]
{previous_response}

[Teacher's Evaluation and Feedback]
{teacher_evaluation}

[Instructional Analysis (Learning Structure)]
{task_analysis}

[Instructions]
1. Carefully read and consider each piece of feedback from your teacher
2. Identify where your previous reasoning was incomplete or incorrect
3. Address each unsatisfied performance objective
4. Show your improved thinking step by step
5. Provide your final answer clearly

[Output Format]
Improved reasoning:
[Your detailed improved solution addressing the teacher's feedback]

Answer: [your final answer]
"""


# ==============================================================================
# SFT 데이터 출력 형식 템플릿
# ------------------------------------------------------------------------------
# 최종 SFT 학습 데이터의 output 필드 형식을 정의합니다.
# 문제 해결 전략과 최종 답변을 포함합니다.
# ==============================================================================

SFT_OUTPUT_TEMPLATE = """Problem-solving strategy and flow:
{strategy_and_flow}

Answer: {answer}
"""


# ==============================================================================
# 반복적 스캐폴딩 프롬프트 (Iterative Scaffolding Prompts)
# ------------------------------------------------------------------------------
# 학생이 문제를 풀지 못할 때 점진적으로 더 구체적인 힌트를 제공합니다.
# 반복 횟수에 따라 힌트의 수준이 달라집니다:
# - 1회차: 개념적 방향 제시
# - 2회차: 구체적 오류 지적
# - 3회차: 부분적 예시 제공
# - 4회차: 더 완전한 예시 제공
# - 5회차: 거의 완성된 가이드 제공
# ==============================================================================

INITIAL_HINT_PROMPT = """You are a teacher providing the first hint to help a student solve a problem.

[Problem]
{problem_text}

[Task Analysis (Learning Structure)]
{task_analysis}

[Correct Answer - FOR YOUR REFERENCE ONLY, DO NOT REVEAL]
{ground_truth}

[Instructions]
Generate a helpful initial hint that:
1. Points the student toward the correct approach without revealing the answer
2. Identifies which skills from the task analysis are relevant
3. Suggests a good starting point for reasoning
4. Encourages systematic problem-solving

The hint should be like a gentle nudge in the right direction - not too specific, but enough to guide initial thinking.

[Output Format]
Start with a brief encouragement, then provide:
- Key concept to consider: [what they should focus on]
- Suggested first step: [how to begin]
- Watch out for: [common pitfall to avoid]
"""


# 점진적 힌트 프롬프트 - 반복 횟수에 따라 힌트 수준 조절
PROGRESSIVE_HINT_PROMPT = """You are a teacher providing a follow-up hint to a student who hasn't solved the problem yet.

[Problem]
{problem_text}

[Task Analysis]
{task_analysis}

[Conversation So Far]
{conversation_history}

[Student's Last Attempt]
{last_response}

[Current Iteration]
{iteration_number} of {max_iterations}

[Correct Answer - FOR YOUR REFERENCE ONLY]
{ground_truth}

[Instructions]
Generate a more specific hint based on iteration number:
- Iteration 2: Point out the specific conceptual error in their approach
- Iteration 3: Provide a partial worked example showing the key step
- Iteration 4: Show a more complete worked example with explanation
- Iteration 5: Provide near-complete guidance, leaving only final calculation

Analyze what went wrong in their last attempt and address that specifically.

[Output Format]
- What went wrong: [specific error in their reasoning]
- Corrected approach: [how to fix it]
- Example/Guidance: [worked example at appropriate level for this iteration]
- Next step to try: [concrete action]
"""


# 요약 재구성 프롬프트 - 5회 시도 후 실패 시 학습 요약 및 해답 재구성
SUMMARY_RECONSTRUCTION_PROMPT = """You are a teacher summarizing a learning session where a student failed to solve a problem after 5 attempts.

[Problem]
{problem_text}

[Correct Answer]
{ground_truth}

[Task Analysis]
{task_analysis}

[Conversation History]
{conversation_history}

[Instructions]
Create a comprehensive learning summary that:
1. Analyzes why the student consistently failed
2. Identifies specific knowledge/skill gaps
3. Reconstructs a correct solution that addresses their weaknesses
4. Highlights what they should learn from this experience

The reconstructed solution should explicitly address and correct the student's errors.

[Output Format - JSON]
{{
    "summary": "Brief summary of the learning session (2-3 sentences)",
    "student_weaknesses": [
        "Specific weakness 1",
        "Specific weakness 2"
    ],
    "reconstructed_response": "Complete correct solution formatted as:\\n[Understanding the problem]\\n...\\n[Key insight they missed]\\n...\\n[Step-by-step solution]\\n...\\n[Common pitfall to avoid]\\n...\\nAnswer: [correct answer]",
    "learning_points": [
        "Key takeaway 1",
        "Key takeaway 2"
    ]
}}

CRITICAL INSTRUCTIONS FOR JSON OUTPUT:
1. Your response MUST be ONLY valid JSON - no additional text before or after
2. Do NOT include explanations, comments, markdown code blocks, or any text outside the JSON
3. Do NOT include LaTeX expressions, mathematical notation, or equations outside JSON string values
4. Ensure ALL brackets {{ }}, [ ], and quotes are properly closed
5. If you need to include mathematical expressions in the reconstructed_response, place them INSIDE the JSON string value with proper escaping (use double backslashes: \\\\)

Example of CORRECT response:
{{
  "summary": "...",
  "student_weaknesses": [...],
  "reconstructed_response": "...",
  "learning_points": [...]
}}

Example of INCORRECT response (DO NOT DO THIS):
Let me reconstruct the solution:
{{
  "summary": "...",
  ...
}}
The student should focus on...

Output ONLY the JSON object above. Do not include any additional text, explanation, or commentary outside the JSON structure.
"""


# 반복 스캐폴딩 시스템 프롬프트 - 교사 힌트 기반 학생 응답용
ITERATIVE_SCAFFOLDING_SYSTEM_PROMPT = """You are a student learning to solve problems with teacher guidance.

[Task Analysis (Your Learning Framework)]
{task_analysis}

[Instructions]
1. Carefully consider the teacher's hint before responding
2. Show your thinking step by step
3. Apply the relevant skills from the task analysis
4. If you made an error before, explicitly address how you're correcting it
5. Provide your final answer clearly

[Output Format]
My thinking:
- Understanding from hint: [what the hint tells you]
- Approach: [your strategy]
- Step-by-step reasoning: [your work]

Answer: [your final answer]
"""


# 힌트 기반 학생 응답 프롬프트 - 교사 힌트를 활용한 문제 풀이
STUDENT_WITH_HINT_PROMPT = """Based on the teacher's guidance, solve this problem.

[Problem]
{problem_text}

[Teacher's Hint]
{teacher_hint}

Now solve the problem, incorporating the teacher's guidance. Show your thinking step by step and provide your final answer clearly.
"""


# 대화 요약 프롬프트 - 튜터링 세션의 핵심 내용 추출
CONVERSATION_SUMMARIZATION_PROMPT = """You are a teacher analyzing a tutoring session where a student struggled with a problem.

[Problem]
{problem_text}

[Correct Answer]
{ground_truth}

[Full Conversation History]
{conversation_history}

[Your Task]
Summarize this tutoring session concisely, focusing on what's important for understanding the student's learning gaps.

Extract and preserve:
1. The specific mathematical/logical errors in each attempt (not vague descriptions)
2. How the student's approach changed between iterations
3. Any recurring misconceptions or patterns
4. The final answer attempted in each iteration

[Output Format]
Keep your summary under 1000 characters. Use this structure:

ATTEMPT SUMMARY:
- Iter 1: [approach] → [specific error] → Answer: [answer]
- Iter 2: [approach] → [specific error] → Answer: [answer]
...

KEY PATTERNS:
- Main weakness: [specific skill/concept gap]
- Recurring error: [pattern across attempts]

Do NOT include lengthy explanations. Be telegraphic and specific.
"""


# ==============================================================================
# 성공적 스캐폴딩 재구성 프롬프트 (Case B)
# ------------------------------------------------------------------------------
# 스캐폴딩을 통해 학생이 정답에 도달한 경우, 학습 과정을 통합하여
# 깔끔한 SFT 학습 데이터로 재구성합니다. 스캐폴딩 과정의 명시적 언급 없이
# 이상적인 학생의 응답 형태로 변환합니다.
# ==============================================================================

SUCCESSFUL_SCAFFOLDING_RECONSTRUCTION_PROMPT = """You are an expert teacher reconstructing a successful learning outcome into clean SFT training data.

[Problem]
{problem_text}

[Correct Answer]
{ground_truth}

[Task Analysis]
{task_analysis}

[Scaffolding Process Summary]
The student succeeded after {iterations_needed} iterations.
{conversation_summary}

[Final Successful Response]
{final_response}

[Your Task]
Reconstruct the student's learning journey into a single, clean response that:
1. Incorporates the key insights gained through scaffolding
2. Presents a clear, step-by-step solution
3. Naturally integrates the guidance that led to success
4. Is suitable for SFT training (no explicit mention of scaffolding process)

The reconstructed response should be what an ideal student would produce after having learned from this scaffolding experience.

[Output Format - JSON]
{{
    "reconstructed_response": "A clean, comprehensive solution that incorporates the learning from scaffolding...",
    "key_learning_points": ["Point 1", "Point 2", "Point 3"],
    "improvement_summary": "How the student improved through the scaffolding process..."
}}

CRITICAL INSTRUCTIONS FOR JSON OUTPUT:
1. Your response MUST be ONLY valid JSON - no additional text before or after
2. Do NOT include explanations, comments, markdown code blocks, or any text outside the JSON
3. Do NOT include LaTeX expressions, mathematical notation, or equations outside JSON string values
4. Ensure ALL brackets {{ }}, [ ], and quotes are properly closed
5. If you need to include mathematical expressions in the reconstructed_response, place them INSIDE the JSON string value with proper escaping (use double backslashes: \\\\)

Example of CORRECT response:
{{
  "reconstructed_response": "...",
  "key_learning_points": [...],
  "improvement_summary": "..."
}}

Example of INCORRECT response (DO NOT DO THIS):
Based on the scaffolding process:
{{
  "reconstructed_response": "...",
  ...
}}
$$x = 5$$

Output ONLY the JSON object above. Do not include any additional text, explanation, or commentary outside the JSON structure.
"""


# ==============================================================================
# 스캐폴딩 아티팩트 생성 프롬프트
# ------------------------------------------------------------------------------
# Dick & Carey 모델 기반의 스캐폴딩 아티팩트를 생성합니다.
# 실패한 수행목표에 대해 HOT/LOT 기술 유형에 따라
# 적절한 스캐폴딩(전략 제안, 부분 예시, 피드백 등)을 설계합니다.
# 학생이 다음 시도에서 참조할 수 있는 "Scaffolding Artifact"로 활용됩니다.
# ==============================================================================

SCAFFOLDING_ARTIFACT_PROMPT = """You are an instructional design expert (Dick & Carey model) creating a Scaffolding Artifact to help a student improve.

Your role is to design pedagogical scaffolding for Performance Objectives that the student failed to meet. This scaffolding will be stored as a "Scaffolding Artifact" that the student can reference in their next attempt.

[Input Data]
- Problem: {problem_text}
- Student's Response: {student_response}
- Performance Objectives Evaluation: {po_evaluation}
- Failed Performance Objectives: {failed_objectives}
- Instructional Analysis: {task_analysis}
- Iteration Number: {iteration_number} of {max_iterations}

[Instructions]
1. **Select scaffolding targets**: Focus on Performance Objectives with high failure rates that are critical for achieving the Instructional Goal.

2. **Classify skill level**: For each unmet PO, determine if it requires:
   - **HOT (High-Order Thinking)**: Analyze, Evaluate, Create
   - **LOT (Low-Order Thinking)**: Remember, Understand, Apply

3. **Design appropriate scaffolding**:

   For **HOT skills**:
   - Strategy suggestion: Propose an approach or reasoning strategy
   - Partial worked example: Show partial reasoning (stop before the final answer)
   - Feedback question: Guide thinking without revealing the answer
   - Key attention points: What the student should focus on

   For **LOT skills**:
   - Missed concept/information: Explicitly state what the student missed
   - Brief explanation: Provide a concise explanation to minimize cognitive load

4. **Do NOT reveal correct answers** - guide reasoning, don't solve.

[Output Format - JSON]
{{
  "scaffolding_artifacts": [
    {{
      "target_objective": "The specific unmet Performance Objective",
      "skill_type": "HOT" or "LOT",
      "cognitive_level": "Analyze/Evaluate/Create" or "Remember/Understand/Apply",
      "failure_analysis": "Why the student failed this objective",
      "scaffolding_content": {{
        "strategy_suggestion": "Suggested approach (for HOT) or null",
        "partial_example": "Partial worked example showing key reasoning (for HOT) or null",
        "feedback": "Guiding feedback (for HOT) or null",
        "missed_concept": "Concept the student missed (for LOT) or null",
        "brief_explanation": "Concise explanation (for LOT) or null",
        "key_attention_points": "What to focus on in next attempt"
      }}
    }}
  ],
  "scaffolding_summary": "A 3-5 sentence summary synthesizing the key guidance for the student's next attempt. This should be actionable and reference the specific strategies or concepts without revealing answers."
}}

CRITICAL INSTRUCTIONS:
1. Your response MUST be ONLY valid JSON - no additional text
2. Do NOT reveal correct answers or complete solutions
3. Focus on guiding the reasoning process
4. The scaffolding_summary should be directly usable by the student

Output ONLY the JSON object above.
"""


# ==============================================================================
# 스캐폴딩 아티팩트 기반 학생 응답 프롬프트
# ------------------------------------------------------------------------------
# 교사가 준비한 스캐폴딩 정보를 활용하여 개선된 응답을 생성합니다.
# 학생은 Scaffolding Artifact에서 어떤 정보를 활용했는지 명시해야 합니다.
# ==============================================================================

STUDENT_WITH_ARTIFACT_PROMPT = """You are a student learning to solve problems with scaffolding support.

Your teacher has evaluated your previous attempt and provided feedback and scaffolding guidance to help you improve. You must carefully use this information to generate a better solution.

[Problem]
{problem_text}

[Teacher's Feedback on Your Previous Response]
{teacher_feedback}

[Scaffolding Artifact]
The following scaffolding information has been prepared to help you:

{scaffolding_summary}

[Detailed Scaffolding Artifacts]
{scaffolding_artifacts}

[Instructional Analysis (Learning Structure)]
{task_analysis}

[Instructions]
1. Carefully read your teacher's feedback to understand what you got wrong and why
2. Review the Scaffolding Artifact and identify which guidance applies to your mistakes
3. Apply the strategies, concepts, or explanations from the scaffolding to improve your solution
4. Show your improved reasoning step by step
5. Provide your final answer clearly

[Output Format]
Improved Reasoning:
- Applying scaffolding guidance: [explain how you are using the scaffolding]
- Step-by-step solution: [your detailed improved solution]

Answer: [your final answer]
"""


# ==============================================================================
# 교사 최종 해답 프롬프트 (Case C)
# ------------------------------------------------------------------------------
# 최대 반복 횟수 후에도 학생이 정답에 도달하지 못한 경우,
# 교사가 완전한 해답을 제공합니다. 학생의 지속적인 약점을 직접 다루고
# 각 단계가 왜 필요한지 설명하여 교육적 가치를 극대화합니다.
# ==============================================================================

TEACHER_FINAL_SOLUTION_PROMPT = """You are a teacher providing a complete, correct solution after the student failed to solve the problem after {max_iterations} attempts.

[Problem]
{problem_text}

[Correct Answer]
{ground_truth}

[Instructional Analysis]
{task_analysis}

[Scaffolding History]
The following scaffolding was provided across {iterations_count} iterations:
{scaffolding_history}

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

[Output Format - JSON]
{{
  "solution_explanation": "Complete step-by-step solution with clear reasoning. Format:\\n[Understanding the Problem]\\n...\\n[Key Concepts Applied]\\n...\\n[Step-by-Step Solution]\\n...\\n[Common Pitfalls Addressed]\\n...\\nAnswer: [correct answer]",
  "addressed_weaknesses": [
    "How weakness 1 was addressed in the solution",
    "How weakness 2 was addressed in the solution"
  ],
  "key_learning_points": [
    "Key takeaway 1 from this problem",
    "Key takeaway 2 from this problem",
    "Key takeaway 3 from this problem"
  ],
  "final_answer": "The correct answer"
}}

CRITICAL INSTRUCTIONS:
1. Your response MUST be ONLY valid JSON - no additional text
2. The solution_explanation should be comprehensive and educational
3. Explicitly connect the solution to the student's weaknesses
4. Ensure all brackets and quotes are properly closed

Output ONLY the JSON object above.
"""
