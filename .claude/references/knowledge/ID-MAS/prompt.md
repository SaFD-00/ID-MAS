**1\. 교수 목표 설정**: ( \- )

**2\. 과제분석 진행**:

| \# 2단계: 교수 분석 (Instructional Analysis) INSTRUCTIONAL\_ANALYSIS\_PROMPT \= """ You are an instructional design expert. Perform the Instructional Analysis step of the Dick & Carey model for the learning objective provided below. \[Learning objective\]: {learning\_objective} \[Instructions\] Perform the Instructional Analysis and construct a hierarchical structure in the form of: Terminal Goal → Subskills → Subtasks. Present the instructional analysis results as a text-based tree structure. Write all skill statements concisely using an action verb \+ object format. Include only the minimum number of Subskills and Subtasks that are essential to achieving the Terminal Goal. For every function or sub-function, indicate the learning outcome based on Bloom’s revised Taxonomy: Remember / Understand / Apply / Analyze / Evaluate / Create. The final output must follow the structure and labels in the Output Format below. Do not change the wording, ordering, line breaks, or section titles. The Output Format example is provided ONLY to specify formatting and structure. Determine all subskills and subtasks strictly based on the given Learning Goal. \[Output Format\] \#\#\# Instructional Analysis Results Terminal Goal: \[Learning objective statement\] (learning outcome)  ├── \[Subskill statements\] (learning outcome)  │   ├── \[Subtask statements, if needed\] (learning outcome) \[Output Format Description\] \- Use consistent numbering (e.g., \[1\], \[1-1\]) \- Use tree characters (├──, │, └──) where applicable """ |
| :---- |

**3\. 학습해야하는 내용을 실제로 학습 했는지를 평가하는 기준 개발**

| \# 4단계: 수행목표 진술 (Performance Objectives) PERFORMANCE\_OBJECTIVES\_PROMPT \= """ You are an instructional designer specializing in the Dick and Carey instructional design model, and a researcher in LLM learning methodologies. Based on the provided Terminal Goal and Instructional Analysis Result, generate a set of Performance Objectives that will serve as the criteria for evaluating the observable performance within the LLM's reasoning process. Specifically, they should be created using information from the learning outcomes identified in the Instructional Analysis Results \[Input Data\] Instructional Analysis Result: {instructional\_analysis} \[Instructions\] For each Subskills and Subtask in the instructional analysis, you must create at least one Performance Objective. Every Performance Objective must include all three components—Behavior, Condition, and Criterion—and each component must be explicitly stated. \- Behavior: This is a description of LLM’s intellectual skill including actions, content, and concepts.  \- Condition: This is a description of the tools and resources that will be available to the learner when performing the skill. Write the conditions based solely on the data given in the problem or generated during the reasoning process.  And it should always begin with 'Given \~'. \- Criterion: This is a description of acceptable performance of the skill. The Criterion component must be tailored to the nature of the task: for tasks with correct answers, it must include a clear and measurable standard such as accuracy requirements, acceptable error ranges, or the number of correct responses; whereas for tasks with no single correct answer, it must specify the information or features that must be present for an acceptable response. Furthermore, these criteria must be formulated to evaluate the observable reasoning process within a single problem-solving task. Each Performance Objective must correspond directly to a single Subskill and Subtask, and you must not add content that does not appear in the Instructional Analysis Result.   \[Output Format\] Your output must be formatted as JSON, following this structure and no other form of explanation or commentary: {{   "performance\_objectives": \[     {{       "target": "Terminal Goal",       "Behavior": "...",       "Condition": "...",       "Criterion": "..."     }},     {{       "target": "Subskill X",       "Behavior": "...",       "Condition": "...",       "Criterion": "..."     }},     {{       "target": "Subtask X",       "Behavior": "...",       "Condition": "...",       "Criterion": "..."     }}   \] }} """ |
| :---- |

**4\. 학습 매체 제작하기 (Mt와 Ms의 상호작용 결과를 바탕으로한 학습 매체 제작)**

1) **initial response 뽑기**

| system prompt “question": {question} “instruction”: The purpose of your response is to demonstrate the attainment of the Terminal Goal: *{Insert Terminal Goal}*. You must adhere to the specific performance procedures and required knowledge/skills outlined in the Instructional Analysis results: *{Insert Analysis Results}*. Ensure that your solution describes the full reasoning process using all provided steps and resources before arriving at the final answer. 당신이 질문에 대한 답변을 하는 것은 { terminal goal \- 교수 분석결과에서 terminal goal만 가져옴 }라는 목표를 성취하는 것을 확인하기 위함입니다. 또한 고려해야하는 수행의 절차와 필요한 지식/능력은 다음과 같습니다. {교수분석 결과} 제시된 수행의 단계와 필요한 정보자원에 대한 정보를 전부 사용하여 풀이과정을 서술하고, 문제에 대한 답을 생성하세요.  |
| :---- |

   

   

2) **initial response가 나온 후에 생성하는 Mt의 교수적 처치 \+ Rollout 생성**

| You are a teacher model supporting the learning of a student model, a small LLM. Your role is not to provide correct answers, but to generate a reasoning state that guides the student model’s next response. Your role is to monitor the student model's reasoning steps to ensure they meet the established performance objectives. In cases of non-compliance or error, you must generate tailored feedback to guide the model toward the desired outcome and give a student model some questions for scaffolding such as socratic questions. Your feedback functions as an intermediate thought in a ReAct-style learning loop and must guide the student model’s next reasoning action. \[Input data\]\- Problem description: {problem\_text}\- Student model original response: {student\_response}\- Performance objective list: {performance\_objectives} \[Instructions\]Evaluate the student model’s response according to the following rules. 1\. Assess student performance according to the performance objectives. Use the Criterion defined in the performance objectives as the evaluation standard. Do not reveal correct answers or model solutions. 2\. Analyze the student response and determine which performance objectives are satisfied and which are not. All judgments must be grounded in observable reasoning behaviors in the student response, such as how claims are justified, how relationships are analyzed, or how judgments are formed. Avoid vague or abstract evaluations. 3\. For each unsatisfied performance, derive a reasoning action that the student model should perform in the next iteration. Do not provide final conclusions, correct answers, or complete reasoning paths. Instead, specify what type of reasoning process, analytical step, or judgment perspective should be explicitly carried out next. \[Output format\] Your output must be written only in valid JSON. For each performance objective, you must evaluate the student's performance and provide the following three elements in a structured array:" Your output must be written only in valid JSON and must follow exactly the structure below. For each performance objective: If the objective is satisfied, set `is_satisfied` to true and fill the remaining fields as null. If the objective is not satisfied, set `is_satisfied` to false and provide a detailed description and a Socratic question." {   "performance\_evaluation": \[     {       "objective\_content": "Specific objective from the instructional analysis",       "is\_satisfied": "boolean",       "reason\_for\_unmet\_objective": "Description of the cause if false; otherwise null",       "socratic\_question": "Socratic question to bridge the gap if false; otherwise null"     }   \] } This output will be used as the reasoning state that conditions the student model’s next response in a ReAct-style learning loop. |
| :---- |

 


+ **) 분석한 내용 기반 scaffolding artifact 만들기 (수정이 많이 필요할 것 같아서 일단 한글로 만들었습니다.)**

| 당신은 교수설계(Dick & Carey)에 전문성을 가진 Teacher Model이다. 당신의 역할은 initial response 이후, Student Model이 반복적으로 실패한 수행 요소에 대해 다음 시도(fixed response)에서 활용할 수 있는 교수적 스캐폴딩(scaffolding artifact)을 설계하는 것이다. 이 스캐폴딩은 \- 정답을 직접 제시하지 않으며, \- 수행을 재구조화할 수 있는 전략과 사고의 발판을 제공하고, \- 이후 rollout 과정에서 Student Model이 참고하는 DB로 사용된다. ──────────────── \[입력 데이터\] \- 학습 목표 (Terminal Goal): {learning\_objective} \- 과제분석 결과 (Instructional Analysis: Task Hierarchy): {instructional\_analysis} \- 수행 평가 요약 (Performance Objective 기준): {performance\_evaluation} \- Initial Response 오류 요약: {initial\_response\_error\_summary} (오류가 집중된 performance objective, 오류 유형, 발생 빈도 포함) ──────────────── \[설계 원칙\] 1\. 스캐폴딩 대상 선정 \- 오류 빈도가 높고(예: 40% 이상), \- Terminal Goal 달성에 핵심적인 performance objective만 선택하라. 2\. 과제 수준 분류 \- High Order Skill (HOT): Analyze / Evaluate / Create \- Low Order Skill (LOT): Remember / Understand / Apply 3\. 교수적 처치 방식 \- HOT:   · 수행 전략 제안   · 학생의 부분 수행 또는 오류가 드러나는 예시 제시   · socratic\_question   ※ 최종 정답 도출 단계는 포함하지 말 것 \- LOT:   · 학습자가 놓친 개념 또는 정보 명시   · 인지 부하를 최소화한 설명 제공 ──────────────── \[출력 형식 — 반드시 유지\] “학습 목표”: {learning\_objective} “과제분석 결과”: {learning\_objective}를 달성하기 위해 필요한 주요 수행 단계와 보조적 지식/정보는 다음과 같다. {instructional\_analysis} “과제분석 \[1\] (High Order Skill)에 대한 스캐폴딩”: \- 학습자가 반복적으로 실패한 지점:   \~\~\~ \- 다음 시도에서 활용할 수 있는 수행 전략:   ⓐ 전략 1: \~\~\~      · 학생의 부분 수행 예시(오류 또는 중단 지점까지):        \~\~\~   ⓑ 전략 2: \~\~\~      · 교사의 추론 명료화 모델링        (왜 이 전략을 고려해야 하는지에 대한 설명):        \~\~\~ \- 수행 시 주의해야 할 핵심 포인트:   \~\~\~ “과제분석 \[2\] (Low Order Skill)에 대한 보완 스캐폴딩”: \- 학습자가 놓친 정보/개념:   \~\~\~ \- 간결한 설명:   \~\~\~ \[Scaffolding Summary for Rollout\] \- 위 스캐폴딩을 사용할 때 Student Model이   반드시 참조해야 할 핵심 전략 또는 개념을   3\~5문장으로 요약하라.  |
| :---- |


3) **Ms는 해당 코칭 DB를 보고 틀렸던 문항에 대한 fixed response를 제공한다**

| “question": {question} “instruction”: 해당 DB를 참고하여 문제 풀이를 진행하세요. {DB} 단, 풀이과정에 DB에서 꺼내 쓴 정보에 대한 내용을 반드시 언급하여 풀이하세요. |
| :---- |

 


4) Mt는 fixed response에 나온 결과를 채점한다. ( \- )  
5) **Mt는 틀린 결과에 대해서 풀이과정을 설명한다.** 

| system prompt “question": {question} “instruction”: The purpose of your response is to demonstrate the attainment of the Terminal Goal: *{Insert Terminal Goal}*. You must adhere to the specific performance procedures and required knowledge/skills outlined in the Instructional Analysis results: *{Insert Analysis Results}*. Ensure that your solution describes the full reasoning process using all provided steps and resources before arriving at the final answer. 당신이 질문에 대한 답변을 하는 것은 { terminal goal \- 교수 분석결과에서 terminal goal만 가져옴 }라는 목표를 성취하는 것을 확인하기 위함입니다. 또한 고려해야하는 수행의 절차와 필요한 지식/능력은 다음과 같습니다. {교수분석 결과} 제시된 수행의 단계와 필요한 정보자원에 대한 정보를 전부 사용하여 풀이과정을 서술하고, 문제에 대한 답을 생성하세요.  |
| :---- |

   

**이후 SFT용 데이터 생성**

- fixed response에 나온 결과를 채점 한 것을 모은다.  
  - ⓐ initial response가 정답이었던 문제&풀이과정: 기존에 자신이 풀었던 풀이과정을 그대로 SFT  
    - 기존에 풀 수 있는 문제였으니, 그냥 그대로 학습  
    - ⓑ DB를 바탕으로 구한 fixed response가 정답이었던 문제&풀이과정: 

    풀었던 풀이과정을 그대로 SFT

- 스케폴딩의 역할을 하는 DB를 통해 성공한 performance이므로, 그 performance을 보이는 과정에 DB내용이 잘 적용되게 넣음  
  - ⓒ Fixed response도 틀렸을 경우: Mt의 답변 생성 결과를 SFT  
    - 스케폴딩을 해도 수행을 못한 경우. 이는 Mt의 P	erformance 을 그대로 따라하도록 함 ⇒ 모델링

- **SFT용 데이터셋 구조**

| Input:  “system prompt” : 다음과 같은 단계로 차근차근 진행하세요. {과제분석 결과} 에 따라 문제풀이 전략 및 흐름을 미리 계획하고, 그에 맞추어 계획에서 생략하는 것 없이 풀이과정을 서술하세요. “Question” : \~\~\~\~\~ Output :  “문제풀이 전략 및 흐름”: { 과제분석 결과를 바탕으로 한 문제 풀이 계획 } Answer: (풀이 절차대로 작성) |
| :---- |


