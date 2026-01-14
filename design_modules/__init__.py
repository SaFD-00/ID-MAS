"""
Design Modules for ID-MAS Instructional Design Phase

Design Phase 단계:
- Step 0: Terminal Goal Generation (TerminalGoalGenerator)
- Step 1: Instructional Analysis (InstructionalAnalysis)
- Step 2: Performance Objectives (PerformanceObjectives)
- Step 3: Test Item Development (TestItemDevelopment)
- Step 4: Rubric Development (RubricDevelopment)
"""
from design_modules.terminal_goal import TerminalGoalGenerator
from design_modules.analysis import InstructionalAnalysis
from design_modules.objectives import PerformanceObjectives
from design_modules.test import TestItemDevelopment
from design_modules.rubric import RubricDevelopment

__all__ = [
    "TerminalGoalGenerator",
    "InstructionalAnalysis",
    "PerformanceObjectives",
    "TestItemDevelopment",
    "RubricDevelopment",
]
