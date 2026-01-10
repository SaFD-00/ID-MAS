"""
3-Phase Learning Pipeline Prompts (PDF Proposal Based)

Phase 1: Scaffolding - Initial Response Generation
Phase 2: Coaching - Performance Objective Scoring, Weak Analysis, Fixed Response
Phase 3: Modeling - Teacher's Articulate Reasoning
"""


# ==============================================================================
# Phase 1: Scaffolding System Prompt
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
# Phase 2: Performance Objective Scoring Prompt
# ==============================================================================

PERFORMANCE_SCORING_PROMPT = """You are evaluating a student's response against specific Performance Objectives.

[Student Response]
{student_response}

[Performance Objectives]
{performance_objectives}

[Ground Truth]
{ground_truth}

[Instructions]
For each Performance Objective:
1. Evaluate whether the student demonstrated the required behavior
2. Score from 0.0 (completely incorrect) to 1.0 (fully correct)
3. Identify specific weaknesses if score < 1.0

[Output Format - JSON]
{{
  "overall_correct": true or false,
  "objective_scores": [
    {{
      "objective_target": "name of the objective",
      "score": 0.0 to 1.0,
      "demonstrated_behavior": "what the student did correctly",
      "weaknesses": ["list of specific weaknesses"]
    }}
  ],
  "weak_objectives": ["list of objective targets with score < 0.6"]
}}

Output ONLY the JSON object above. Do not include any additional text, explanation, or commentary outside the JSON structure.
"""


# ==============================================================================
# Phase 2: Teacher Intervention Prompt (ReAct-style with Socratic Questions)
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
# Phase 2: Student Response to Socratic Questions
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
# Phase 2: Weak Objective Analysis Prompt (Batch)
# ==============================================================================

WEAK_OBJECTIVE_ANALYSIS_PROMPT = """You are analyzing systematic weaknesses across multiple student responses.

[Weak Objectives Identified]
{weak_objectives}

[Sample Student Responses with Errors]
{student_responses}

[Task Analysis]
{task_analysis}

[Instructions]
1. Identify common error patterns related to each weak objective
2. Determine the underlying conceptual or procedural gaps
3. Suggest specific remediation strategies with examples

[Output Format - JSON]
{{
  "weak_performance_areas": [
    {{
      "objective": "objective name",
      "error_rate": 0.0 to 1.0,
      "common_errors": ["list of common error patterns"],
      "conceptual_gaps": ["underlying knowledge/skill gaps"],
      "remediation_hints": ["suggested strategies to overcome"]
    }}
  ],
  "recommended_strategies": ["general strategies for improvement"],
  "examples_needed": ["types of examples that would help"]
}}

Output ONLY the JSON object above. Do not include any additional text, explanation, or commentary outside the JSON structure.
"""


# ==============================================================================
# Phase 2: Coaching DB Generation Prompt
# ==============================================================================

COACHING_DB_GENERATION_PROMPT = """You are a Teacher Model with expertise in instructional design (Dick & Carey).

Your role is to design pedagogical scaffolding artifacts for the Student Model's next attempt (fixed response). This scaffolding:
- Does NOT provide correct answers directly
- Provides strategies and cognitive scaffolds for restructuring performance
- Will be used as a reference DB during the rollout process

[Input Data]
- Learning Objective (Terminal Goal): {learning_objective}
- Instructional Analysis (Task Hierarchy): {task_analysis}
- Performance Evaluation Summary: {performance_evaluation}
- Initial Response Error Summary: {initial_response_error_summary}

[Design Principles]
1. Scaffolding Target Selection:
   - Select only performance objectives with high error rates (e.g., ≥40%)
   - Focus on objectives critical to achieving the Terminal Goal

2. Skill Level Classification (based on Bloom's Taxonomy):
   - High Order Skills (HOT): Analyze / Evaluate / Create
   - Low Order Skills (LOT): Remember / Understand / Apply

3. Pedagogical Intervention by Skill Level:
   - For HOT:
     · Suggest performance strategies
     · Provide examples showing student's partial performance or errors (up to the error point)
     · Include Socratic questions
     · Do NOT include final answer derivation steps
   - For LOT:
     · Specify concepts or information the learner missed
     · Provide explanations that minimize cognitive load

[Output Format - JSON]
{{
  "learning_objective": "{learning_objective}",
  "task_analysis_summary": "Key performance steps and required knowledge/information for achieving the terminal goal",
  "high_order_skill_scaffolding": [
    {{
      "skill_reference": "[Subskill/Subtask number from instructional analysis]",
      "skill_type": "HOT",
      "bloom_level": "Analyze / Evaluate / Create",
      "repeated_failure_point": "Where the learner repeatedly failed",
      "strategies_for_next_attempt": [
        {{
          "strategy_name": "Strategy 1",
          "partial_performance_example": "Example showing student's work up to error point",
          "teacher_reasoning_clarification": "Why this strategy should be considered"
        }}
      ],
      "key_considerations": "Critical points to note during performance",
      "socratic_question": "Guiding question to prompt reflection"
    }}
  ],
  "low_order_skill_scaffolding": [
    {{
      "skill_reference": "[Subskill/Subtask number from instructional analysis]",
      "skill_type": "LOT",
      "bloom_level": "Remember / Understand / Apply",
      "missed_information": "Concepts or information the learner missed",
      "concise_explanation": "Brief, clear explanation to minimize cognitive load"
    }}
  ],
  "scaffolding_summary": "3-5 sentence summary of key strategies and concepts the Student Model must reference when using this scaffolding"
}}

Output ONLY the JSON object above. Do not include any additional text, explanation, or commentary outside the JSON structure.
"""


# ==============================================================================
# Phase 2: Fixed Response Generation Prompt (with Coaching DB)
# ==============================================================================

COACHING_RESPONSE_PROMPT = """Solve this problem by referencing the Coaching DB below.

[Learning Objective]
{learning_objective}

[Instructional Analysis]
{task_analysis}

[Coaching DB]
{coaching_db}

[Problem]
{problem_text}

[Instructions]
1. Review the scaffolding strategies and explanations in the Coaching DB
2. Apply the relevant strategies (HOT scaffolding for analysis/evaluation tasks, LOT scaffolding for foundational knowledge)
3. You MUST explicitly mention which information from the Coaching DB you are using in your solution
4. Show your reasoning step by step, demonstrating how the scaffolding guided your approach
5. Provide a clear final answer

[Output Format]
Problem-solving strategy and flow:
- Information retrieved from Coaching DB: [explicitly state what strategies/concepts you used from the DB]
- Strategy application: [how you applied the scaffolding strategies]
- Step-by-step reasoning: [your detailed solution incorporating the DB guidance]

Answer: [your final answer]
"""


# ==============================================================================
# Phase 3: Modeling Prompt (Teacher's Articulate Reasoning)
# ==============================================================================

MODELING_PROMPT = """You are a teacher providing a model solution with clear, articulated reasoning.

[Problem]
{problem_text}

[Correct Answer]
{ground_truth}

[Task Analysis for Reference]
{task_analysis}

[Instructions]
Provide a complete, well-structured solution that:
1. Explicitly states the problem-solving strategy
2. Shows each reasoning step clearly
3. Explains why each step is taken (articulate reasoning)
4. Arrives at the correct answer with justification

Your solution should serve as an exemplary model that students can learn from.

[Output Format]
Problem-solving strategy and flow:
- Strategy selection: [explain which approach is best and why]
- Step-by-step reasoning:
  * Step 1: [action] - [justification for this step]
  * Step 2: [action] - [justification for this step]
  * ...
- Key insights: [important concepts applied]

Answer: {ground_truth}
"""


# ==============================================================================
# SFT Data Output Format Template
# ==============================================================================

SFT_OUTPUT_TEMPLATE = """Problem-solving strategy and flow:
{strategy_and_flow}

Answer: {answer}
"""


# ==============================================================================
# Phase 1: Iterative Scaffolding Prompts (NEW)
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

[Full Conversation History]
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
