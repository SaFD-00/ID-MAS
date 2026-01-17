"""루브릭 평가 기준 템플릿 모듈.

이 모듈은 에세이 기반 과제 평가를 위한 분석적 루브릭(Analytic Rubric) 템플릿을 정의합니다.
Dick & Carey 교수설계 모델에 기반하여 추론 행동을 진단하고 피드백을 제공하는 데 사용됩니다.

프롬프트 상수:
    RUBRIC_CRITERION_TEMPLATES: 출력 유형별 루브릭 기준 템플릿 (JSON 구조)
    RUBRIC_GENERATION_PROMPT: 루브릭 자동 생성 프롬프트

지원하는 출력 유형:
    - explanatory_text: 설명적 텍스트 (개념/원리 정의, 인과/논리 연결)
    - analytical_essay: 분석적 에세이 (구성요소 식별, 관계 분석, 분석 기준)
    - evaluative_essay: 평가적 에세이 (평가 기준 수립, 기준 기반 판단)
    - argumentative_essay: 논증적 에세이 (중심 주장, 증거 기반 논증)
    - comparative_analysis: 비교 분석 (비교 기준 수립, 기준 기반 비교)
    - design_proposal: 설계 제안 (문제 정의, 설계 결정 정당화)

Note:
    각 기준은 4점 척도(4-1)로 추론 행동의 질적 차이를 구분합니다.
"""


# ==============================================================================
# 루브릭 평가 기준 템플릿 (Rubric Criterion Templates)
# ------------------------------------------------------------------------------
# 출력 유형별로 평가 기준과 4점 척도 수행 수준을 정의합니다.
# base_schema는 기준 구조를, output_types는 유형별 구체 기준을 포함합니다.
# ==============================================================================

RUBRIC_CRITERION_TEMPLATES = {
    "rubric_criterion_templates": {
        "base_schema": {
            "criterion_id": "",
            "criterion_name": "",
            "description": "",
            "aligned_objective_criterion": "",
            "levels": {
                "4": "",
                "3": "",
                "2": "",
                "1": ""
            }
        },
        "output_types": {
            "explanatory_text": {
                "criteria_template": [
                    {
                        "criterion_id": "ET1",
                        "criterion_name": "Definition of Concepts or Principles",
                        "description": "Reasoning behavior involving explaining key concepts or principles in one's own words",
                        "aligned_objective_criterion": "",
                        "levels": {
                            "4": "Accurately defines key concepts or principles and explicitly explains their meaning",
                            "3": "Generally explains the concept or principle, but some meanings remain implicit",
                            "2": "Mentions the concept or principle only partially or provides fragmented explanations",
                            "1": "Provides little to no definition or explanation of the concept or principle, or explains it inaccurately"
                        }
                    },
                    {
                        "criterion_id": "ET2",
                        "criterion_name": "Causal or Logical Connections",
                        "description": "Reasoning behavior involving logical or causal connections between concepts",
                        "aligned_objective_criterion": "",
                        "levels": {
                            "4": "Clearly articulates causal or logical relationships between concepts",
                            "3": "Mentions relationships, but the connection process is not fully explicit",
                            "2": "Relationships are only implicitly suggested or weakly connected",
                            "1": "No meaningful connection between concepts is provided"
                        }
                    }
                ]
            },
            "analytical_essay": {
                "criteria_template": [
                    {
                        "criterion_id": "AE1",
                        "criterion_name": "Identification of Components",
                        "description": "Reasoning behavior involving decomposing the analysis target into meaningful components",
                        "aligned_objective_criterion": "",
                        "levels": {
                            "4": "Systematically decomposes the target into core components",
                            "3": "Identifies components, but some are unclear or insufficiently specified",
                            "2": "Mentions components in a limited or fragmented manner",
                            "1": "Does not decompose the target into components"
                        }
                    },
                    {
                        "criterion_id": "AE2",
                        "criterion_name": "Analysis of Relationships Among Components",
                        "description": "Reasoning behavior involving analysis of relationships or structures among components",
                        "aligned_objective_criterion": "",
                        "levels": {
                            "4": "Clearly analyzes and structurally explains relationships among components",
                            "3": "Mentions relationships, but analytical depth is limited",
                            "2": "Relationships are only implicitly suggested",
                            "1": "No analysis of relationships among components is present"
                        }
                    },
                    {
                        "criterion_id": "AE3",
                        "criterion_name": "Explicit Analytical Criteria",
                        "description": "Reasoning behavior involving explicit articulation of analytical criteria or perspectives",
                        "aligned_objective_criterion": "",
                        "levels": {
                            "4": "Clearly establishes and consistently applies analytical criteria or perspectives",
                            "3": "Analytical criteria are partially evident",
                            "2": "Criteria are inconsistent or vague",
                            "1": "No analytical criteria are articulated"
                        }
                    }
                ]
            },
            "evaluative_essay": {
                "criteria_template": [
                    {
                        "criterion_id": "EE1",
                        "criterion_name": "Establishment of Evaluation Criteria",
                        "description": "Reasoning behavior involving the establishment of criteria for evaluation",
                        "aligned_objective_criterion": "",
                        "levels": {
                            "4": "Establishes explicit and valid evaluation criteria",
                            "3": "Evaluation criteria are present but partially implicit",
                            "2": "Evaluation criteria are unclear or limited",
                            "1": "No evaluation criteria are established"
                        }
                    },
                    {
                        "criterion_id": "EE2",
                        "criterion_name": "Criteria-Based Judgment",
                        "description": "Reasoning behavior involving judgments based on established criteria",
                        "aligned_objective_criterion": "",
                        "levels": {
                            "4": "Performs consistent judgments grounded in the established criteria",
                            "3": "Judgments are made, but application of criteria is partial",
                            "2": "Weak alignment between judgments and criteria",
                            "1": "No judgment based on criteria is evident"
                        }
                    }
                ]
            },
            "argumentative_essay": {
                "criteria_template": [
                    {
                        "criterion_id": "AR1",
                        "criterion_name": "Clear Statement of Central Claim",
                        "description": "Reasoning behavior involving explicit articulation of a central claim or position",
                        "aligned_objective_criterion": "",
                        "levels": {
                            "4": "Presents a clear, consistent, and explicit central claim",
                            "3": "A claim is present, but expression is partially unclear",
                            "2": "The claim is implicit or ambiguous",
                            "1": "No central claim is articulated"
                        }
                    },
                    {
                        "criterion_id": "AR2",
                        "criterion_name": "Evidence-Based Argumentation",
                        "description": "Reasoning behavior involving the construction of arguments supported by evidence",
                        "aligned_objective_criterion": "",
                        "levels": {
                            "4": "Supports claims with sufficient and highly relevant evidence",
                            "3": "Provides evidence, but argumentative structure is limited",
                            "2": "Evidence is weak, fragmented, or insufficient",
                            "1": "No evidence is provided to support claims"
                        }
                    }
                ]
            },
            "comparative_analysis": {
                "criteria_template": [
                    {
                        "criterion_id": "CA1",
                        "criterion_name": "Establishment of Comparison Criteria",
                        "description": "Reasoning behavior involving the establishment of clear criteria for comparison",
                        "aligned_objective_criterion": "",
                        "levels": {
                            "4": "Clearly establishes and consistently applies comparison criteria",
                            "3": "Comparison criteria are present but partially unclear",
                            "2": "Comparison criteria are limited or poorly specified",
                            "1": "No comparison criteria are established"
                        }
                    },
                    {
                        "criterion_id": "CA2",
                        "criterion_name": "Criterion-Based Comparison",
                        "description": "Reasoning behavior involving systematic comparison across defined criteria",
                        "aligned_objective_criterion": "",
                        "levels": {
                            "4": "Systematically compares targets across all defined criteria",
                            "3": "Comparison is conducted, but some criteria are omitted",
                            "2": "Comparison is fragmented or superficial",
                            "1": "No meaningful comparative analysis is present"
                        }
                    }
                ]
            },
            "design_proposal": {
                "criteria_template": [
                    {
                        "criterion_id": "DP1",
                        "criterion_name": "Problem Definition",
                        "description": "Reasoning behavior involving clear articulation of the problem to be addressed",
                        "aligned_objective_criterion": "",
                        "levels": {
                            "4": "Clearly and specifically defines the problem",
                            "3": "Defines the problem, but some aspects remain unclear",
                            "2": "Problem definition is limited or vague",
                            "1": "No problem definition is provided"
                        }
                    },
                    {
                        "criterion_id": "DP2",
                        "criterion_name": "Justification of Design Decisions",
                        "description": "Reasoning behavior involving justification of design choices based on explicit reasoning",
                        "aligned_objective_criterion": "",
                        "levels": {
                            "4": "Clearly and logically justifies design decisions",
                            "3": "Provides justification, but explanations are partial",
                            "2": "Justification is weak or fragmented",
                            "1": "No justification for design decisions is provided"
                        }
                    }
                ]
            }
        }
    }
}


# ==============================================================================
# 루브릭 생성 프롬프트 (Rubric Generation Prompt)
# ------------------------------------------------------------------------------
# 수행목표와 기준 템플릿을 기반으로 분석적 루브릭을 자동 생성합니다.
# 에세이 기반 과제에서만 사용되며, ReAct 스타일 학습 루프에서
# 추론 행동 진단과 반복 학습을 지원합니다.
# ==============================================================================

RUBRIC_GENERATION_PROMPT = """
You are an expert in educational assessment and instructional design, specializing in analytic rubrics aligned with the Dick and Carey instructional design model. Your task is to generate an analytic evaluation rubric that functions as a diagnostic tool for reasoning behaviors and supports iterative learning in a ReAct-style instructional loop.

This rubric will be used only when the evaluation format is an essay-based task.

[Input]
- Task description: {task_description}
- Expected output type (choose one): explanatory_text | analytical_essay | evaluative_essay | argumentative_essay | comparative_analysis | design_proposal
- Performance objectives (including Criterion): {performance_objectives}
- Rubric criterion templates (JSON): {rubric_criterion_templates}

[Instructions]
Design an analytic rubric strictly based on the provided inputs.

First, from the provided rubric_criterion_templates JSON, select the criterion template that corresponds exactly to the given expected output type.
Use the selected template as the fixed structural backbone of the rubric.

Second, map each criterion in the selected template to the provided performance objectives.
Each final evaluation criterion must correspond directly to at least one Criterion from the performance objectives.
Remove template criteria that cannot be mapped to any performance objective.
If necessary, refine or specialize the wording of a template criterion to better reflect the performance objectives, but do not change the underlying reasoning behavior defined by the template.

Third, define four performance levels for each criterion.
Performance levels must represent qualitative differences in reasoning behavior, not point scores or vague quality judgments.
Levels must distinguish whether a specific reasoning step or judgment process was explicitly performed, partially performed, implicitly assumed, or missing.
Avoid vague adjectives such as "clear," "appropriate," or "sufficient" unless they are explicitly tied to observable reasoning actions.
Each level description must enable a teacher to diagnose satisfied and unsatisfied reasoning criteria and to generate actionable feedback.

[Output]
Your output must be written only in valid JSON and must follow exactly the structure below.

{{
  "rubric": {{
    "criteria": [
      {{
        "name": "Criterion name aligned with both the selected template and a performance objective",
        "levels": {{
          "4": "Description of explicitly and fully demonstrated reasoning behavior",
          "3": "Description of partially demonstrated or implicit reasoning behavior",
          "2": "Description of minimally demonstrated or fragmented reasoning behavior",
          "1": "Description of missing or incorrect reasoning behavior"
        }}
      }}
    ]
  }}
}}

Do not include any text outside of the JSON output.
"""
