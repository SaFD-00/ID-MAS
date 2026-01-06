"""
ID-MAS Learning Loop Module.

Contains the 3-Phase Learning Pipeline with State Machine support.
"""

from learning_loop.state_machine import (
    LearningState,
    LearningContext,
    LearningStateMachine,
    QuestionProgress,
    StateTransition,
    IterationRecord,
    VALID_TRANSITIONS,
)
from learning_loop.pipeline_controller import IDMASPipelineController
from learning_loop.student_model import StudentModel
from learning_loop.teacher_model import TeacherModel

__all__ = [
    # State Machine
    "LearningState",
    "LearningContext",
    "LearningStateMachine",
    "QuestionProgress",
    "StateTransition",
    "IterationRecord",
    "VALID_TRANSITIONS",
    # Pipeline
    "IDMASPipelineController",
    "StudentModel",
    "TeacherModel",
]
