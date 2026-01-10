"""
LangGraph-based ID-MAS 3-Phase Pipeline.

This module implements the 3-Phase Learning Pipeline using LangGraph:
- Phase 1: Initial Response with Scaffolding
- Phase 2: Teacher Coaching + Fixed Response
- Phase 3: Teacher Modeling

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
