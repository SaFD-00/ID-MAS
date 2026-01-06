"""
3-Phase Learning Pipeline Prompts (PDF Proposal Based)

Phase 1: Scaffolding - Initial Response Generation
Phase 2: Coaching - Performance Objective Scoring, Weak Analysis, Fixed Response
Phase 3: Modeling - Teacher's Articulate Reasoning
"""


# ==============================================================================
# Phase 1: Scaffolding System Prompt
# ==============================================================================

SCAFFOLDING_SYSTEM_PROMPT = """You are solving problems using a structured approach based on the task analysis below.

[Task Analysis (Scaffolding)]
{task_analysis}

[Instructions]
1. Read the problem carefully
2. Identify which skills and sub-skills from the task analysis are relevant
3. Apply the appropriate reasoning strategy for each step
4. Show your work step by step
5. Provide your final answer clearly

[Output Format]
Problem-solving strategy and flow:
- Relevant skills: [list the relevant skills from task analysis]
- Reasoning steps: [describe your step-by-step approach]

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
"""


# ==============================================================================
# Phase 2: Coaching DB Generation Prompt
# ==============================================================================

COACHING_DB_GENERATION_PROMPT = """Generate a Coaching Database to help students overcome identified weaknesses.

[Learning Objective]
{learning_objective}

[Task Analysis]
{task_analysis}

[Weakness Analysis]
{weak_analysis}

[Instructions]
Create a structured coaching database that:
1. Summarizes the learning objective and task analysis
2. Provides targeted interventions for each weak area
3. Includes worked examples and correct strategies

[Output Format - JSON]
{{
  "learning_objective": "summary of the learning goal",
  "task_analysis_summary": "key steps and skills needed",
  "performance_areas": [
    {{
      "area_name": "name of the performance area",
      "what_went_wrong": "common mistakes students make",
      "correct_strategy": "how to approach this correctly",
      "key_considerations": ["important points to remember"],
      "worked_example": {{
        "problem": "example problem",
        "correct_approach": "step-by-step solution"
      }}
    }}
  ],
  "general_tips": ["overall tips for success"]
}}
"""


# ==============================================================================
# Phase 2: Fixed Response Generation Prompt (with Coaching DB)
# ==============================================================================

COACHING_RESPONSE_PROMPT = """Based on the coaching information below, solve this problem correctly.

[Learning Objective]
{learning_objective}

[Task Analysis]
{task_analysis}

[Coaching Information - Areas for Improvement]
{coaching_areas}

[General Tips]
{general_tips}

[Problem]
{problem_text}

[Instructions]
1. Apply the strategies from the coaching information
2. Explicitly mention which strategies you are using
3. Show your reasoning step by step
4. Provide a clear final answer

[Output Format]
Problem-solving strategy and flow:
- Strategy selection: [which strategies from coaching you're applying]
- Step-by-step reasoning: [your detailed solution]

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
