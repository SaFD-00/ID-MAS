"""
Iterative Scaffolding Pipeline Prompts (PDF Proposal Based)

Scaffolding - Iterative Response Generation with Teacher Guidance
- Initial response generation
- Iterative scaffolding with Socratic questions (max 5 iterations)
- Case B: Reconstruction after max iterations
"""


# ==============================================================================
# Scaffolding System Prompt
# ==============================================================================

SCAFFOLDING_SYSTEM_PROMPT = """The purpose of your response is to demonstrate the attainment of the Terminal Goal: {terminal_goal}

You must adhere to the specific performance procedures and required knowledge/skills outlined in the Instructional Analysis results below. Ensure that your solution describes the full reasoning process using all provided steps and resources before arriving at the final answer.

[Instructional Analysis (Learning Structure)]
{task_analysis}

[Instructions]
1. Identify which skills and sub-skills from the instructional analysis are relevant to this problem
2. Plan your problem-solving strategy based on the terminal goal and subskills
3. Execute each step systematically, demonstrating the required performance behaviors
4. Ensure your solution describes the full reasoning process using all provided steps and resources
5. Provide your final answer clearly

[Output Format]
Problem-solving strategy and flow:
- Terminal goal alignment: [how this solution demonstrates the terminal goal]
- Relevant skills applied: [list the relevant skills from instructional analysis]
- Step-by-step reasoning: [your detailed solution following the instructional structure]

Answer: [your final answer]
"""


# ==============================================================================
# Teacher Intervention Prompt (ReAct-style with Socratic Questions)
# ==============================================================================

TEACHER_INTERVENTION_PROMPT = """You are a teacher model supporting the learning of a student model.

Your role is NOT to provide correct answers, but to generate a reasoning state that guides the student model's next response. You must monitor the student model's reasoning steps to ensure they meet the established performance objectives.

In cases of non-compliance or error, you must generate tailored feedback to guide the model toward the desired outcome using Socratic questioning. Your feedback functions as an intermediate thought in a ReAct-style learning loop and must guide the student model's next reasoning action.

[Input Data]
- Problem: {problem_text}
- Student model response: {student_response}
- Performance objectives: {performance_objectives}
- Ground truth (FOR REFERENCE ONLY - DO NOT REVEAL): {ground_truth}

[Instructions]
1. Assess student performance according to each performance objective
2. Use the Criterion defined in each performance objective as the evaluation standard
3. DO NOT reveal correct answers or model solutions
4. Analyze the student response and determine which performance objectives are satisfied and which are not
5. All judgments must be grounded in observable reasoning behaviors in the student response, such as how claims are justified, how relationships are analyzed, or how judgments are formed
6. Avoid vague or abstract evaluations
7. For each unsatisfied performance objective, derive a reasoning action that the student model should perform in the next iteration
8. Do not provide final conclusions, correct answers, or complete reasoning paths
9. Instead, specify what type of reasoning process, analytical step, or judgment perspective should be explicitly carried out next

[Output Format - JSON]
{{
  "performance_evaluation": [
    {{
      "objective_content": "The specific objective being evaluated (copy from performance objectives)",
      "is_satisfied": true or false,
      "reason_for_unmet_objective": "Detailed description of the cause if false; null if true",
      "socratic_question": "Socratic question to bridge the gap if false; null if true"
    }}
  ],
  "overall_assessment": {{
    "objectives_met": "X of Y objectives satisfied",
    "all_satisfied": true or false,
    "primary_weakness": "Main area needing improvement if any; null if all satisfied",
    "recommended_focus": "What the student should focus on next if not all satisfied; null if complete"
  }}
}}

Output ONLY the JSON object above. Do not include any additional text, explanation, or commentary outside the JSON structure.
"""


# ==============================================================================
# Student Response to Socratic Questions
# ==============================================================================

STUDENT_SOCRATIC_RESPONSE_PROMPT = """You are a student model learning to solve problems with teacher guidance.

Your teacher has evaluated your previous response and provided Socratic questions to guide your thinking. You must carefully consider this feedback and improve your response.

[Problem]
{problem_text}

[Your Previous Response]
{previous_response}

[Teacher's Evaluation and Socratic Questions]
{teacher_evaluation}

[Instructional Analysis (Learning Structure)]
{task_analysis}

[Instructions]
1. Carefully read and consider each Socratic question from your teacher
2. Identify where your previous reasoning was incomplete or incorrect
3. Address each unsatisfied performance objective
4. Show your improved thinking step by step
5. Provide your final answer clearly

[Output Format]
Reflection on teacher feedback:
- Questions to address: [summarize the Socratic questions]
- Improvements to make: [what you will change in your approach]

Improved reasoning:
[Your detailed improved solution addressing the teacher's questions]

Answer: [your final answer]
"""


# ==============================================================================
# SFT Data Output Format Template
# ==============================================================================

SFT_OUTPUT_TEMPLATE = """Problem-solving strategy and flow:
{strategy_and_flow}

Answer: {answer}
"""


# ==============================================================================
# Iterative Scaffolding Prompts
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

Output ONLY the JSON object above. Do not include any additional text, explanation, or commentary outside the JSON structure.
"""


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


STUDENT_WITH_HINT_PROMPT = """Based on the teacher's guidance, solve this problem.

[Problem]
{problem_text}

[Teacher's Hint]
{teacher_hint}

Now solve the problem, incorporating the teacher's guidance. Show your thinking step by step and provide your final answer clearly.
"""


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
Keep your summary under 800 characters. Use this structure:

ATTEMPT SUMMARY:
- Iter 1: [approach] → [specific error] → Answer: [answer]
- Iter 2: [approach] → [specific error] → Answer: [answer]
...

KEY PATTERNS:
- Main weakness: [specific skill/concept gap]
- Recurring error: [pattern across attempts]

Do NOT include lengthy explanations. Be telegraphic and specific.
"""
