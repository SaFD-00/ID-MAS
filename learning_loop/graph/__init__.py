"""
LangGraph-based ID-MAS Iterative Scaffolding Pipeline.

This module implements the Iterative Scaffolding Pipeline using LangGraph:
- Task Analysis + Initial Response generation
- Performance Objectives based evaluation with Socratic questions
- Case A (PO satisfied) / Case B (reconstructed) SFT data generation

Based on the Dick & Carey Instructional Design Model.
"""
from learning_loop.graph.state import IDMASState, QuestionResult
from learning_loop.graph.graph import create_idmas_graph, IDMASGraphRunner

__all__ = [
    "IDMASState",
    "QuestionResult",
    "create_idmas_graph",
    "IDMASGraphRunner",
]
