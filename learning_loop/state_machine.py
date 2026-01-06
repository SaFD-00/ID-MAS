"""
State Machine for ID-MAS Learning Pipeline.

Manages the learning workflow states and transitions:
INIT -> DESIGN_ANALYSIS -> DESIGN_OBJECTIVES -> DESIGN_RUBRIC -> DESIGN_COMPLETE
     -> PHASE1_RUNNING -> PHASE1_COMPLETE
     -> PHASE2_SCORING -> PHASE2_COACHING -> PHASE2_FIXING -> PHASE2_COMPLETE
     -> PHASE3_MODELING -> PHASE3_COMPLETE
     -> SAVING -> COMPLETE

Supports checkpoint-based resume functionality.
"""
from __future__ import annotations

import json
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable


class LearningState(Enum):
    """
    ID-MAS Learning Pipeline States.

    Each state has: (state_number, color_code, color_name)
    - state_number: Sequential order in the pipeline
    - color_code: ANSI color code for terminal display
    - color_name: Human-readable color name

    Color Legend:
    - Gray (90): Initial/Idle states
    - Cyan (96): Design Phase
    - Blue (94): Phase 1 (Scaffolding)
    - Yellow (93): Phase 2 (Coaching)
    - Magenta (95): Phase 3 (Modeling)
    - Green (92): Success/Complete states
    - Red (91): Error state
    - White (97): Paused state
    """

    # Initial state
    INIT = (0, 90, "gray")                          # Initial state

    # Design Phase states (Cyan)
    DESIGN_ANALYSIS = (1, 96, "cyan")               # Instructional analysis
    DESIGN_OBJECTIVES = (2, 96, "cyan")             # Performance objectives
    DESIGN_RUBRIC = (3, 96, "cyan")                 # Rubric development
    DESIGN_COMPLETE = (4, 96, "cyan")               # Design complete

    # Phase 1: Scaffolding (Blue)
    PHASE1_RUNNING = (5, 94, "blue")                # Phase 1 in progress
    PHASE1_COMPLETE = (6, 94, "blue")               # Phase 1 complete

    # Phase 2: Coaching (Yellow)
    PHASE2_SCORING = (7, 93, "yellow")              # PO-based scoring
    PHASE2_COACHING = (8, 93, "yellow")             # Coaching DB generation
    PHASE2_FIXING = (9, 93, "yellow")               # Fixed response generation
    PHASE2_COMPLETE = (10, 93, "yellow")            # Phase 2 complete

    # Phase 3: Modeling (Magenta)
    PHASE3_MODELING = (11, 95, "magenta")           # Modeling response generation
    PHASE3_COMPLETE = (12, 95, "magenta")           # Phase 3 complete

    # Final states
    SAVING = (13, 92, "green")                      # Saving results
    COMPLETE = (14, 92, "green")                    # All done
    ERROR = (99, 91, "red")                         # Error state
    PAUSED = (98, 97, "white")                      # Paused (resume pending)

    def __init__(self, state_num: int, color_code: int, color_name: str):
        self.state_num = state_num
        self.color_code = color_code
        self.color_name = color_name

    def __str__(self) -> str:
        return self.name

    @property
    def colored_name(self) -> str:
        """Return ANSI-colored state name for terminal display."""
        return f"\033[{self.color_code}m[{self.state_num:02d}] {self.name}\033[0m"

    @property
    def display_name(self) -> str:
        """Return formatted display name with number."""
        return f"[{self.state_num:02d}] {self.name}"

    def get_phase(self) -> str:
        """Return the phase name for this state."""
        if self.name.startswith("DESIGN"):
            return "Design"
        elif self.name.startswith("PHASE1"):
            return "Phase 1: Scaffolding"
        elif self.name.startswith("PHASE2"):
            return "Phase 2: Coaching"
        elif self.name.startswith("PHASE3"):
            return "Phase 3: Modeling"
        elif self.name in ("SAVING", "COMPLETE"):
            return "Finalization"
        elif self.name == "ERROR":
            return "Error"
        elif self.name == "PAUSED":
            return "Paused"
        else:
            return "Initial"


# Valid state transitions
VALID_TRANSITIONS: Dict[LearningState, List[LearningState]] = {
    # Initial state
    LearningState.INIT: [
        LearningState.DESIGN_ANALYSIS,  # Generate new design (--run-design)
        LearningState.PHASE1_RUNNING,   # Load existing design (default)
        LearningState.ERROR,
    ],

    # Design Phase
    LearningState.DESIGN_ANALYSIS: [
        LearningState.DESIGN_OBJECTIVES,
        LearningState.ERROR,
        LearningState.PAUSED,
    ],
    LearningState.DESIGN_OBJECTIVES: [
        LearningState.DESIGN_RUBRIC,
        LearningState.ERROR,
        LearningState.PAUSED,
    ],
    LearningState.DESIGN_RUBRIC: [
        LearningState.DESIGN_COMPLETE,
        LearningState.ERROR,
        LearningState.PAUSED,
    ],
    LearningState.DESIGN_COMPLETE: [
        LearningState.PHASE1_RUNNING,
        LearningState.SAVING,  # Save design only
        LearningState.PAUSED,
    ],

    # Phase 1: Scaffolding
    LearningState.PHASE1_RUNNING: [
        LearningState.PHASE1_COMPLETE,
        LearningState.ERROR,
        LearningState.PAUSED,
    ],
    LearningState.PHASE1_COMPLETE: [
        LearningState.PHASE2_SCORING,   # Has incorrect answers -> Phase 2
        LearningState.SAVING,            # All correct -> Save
        LearningState.PAUSED,
    ],

    # Phase 2: Coaching
    LearningState.PHASE2_SCORING: [
        LearningState.PHASE2_COACHING,
        LearningState.ERROR,
        LearningState.PAUSED,
    ],
    LearningState.PHASE2_COACHING: [
        LearningState.PHASE2_FIXING,
        LearningState.ERROR,
        LearningState.PAUSED,
    ],
    LearningState.PHASE2_FIXING: [
        LearningState.PHASE2_COMPLETE,
        LearningState.ERROR,
        LearningState.PAUSED,
    ],
    LearningState.PHASE2_COMPLETE: [
        LearningState.PHASE3_MODELING,   # Still has incorrect -> Phase 3
        LearningState.SAVING,             # All fixed -> Save
        LearningState.PAUSED,
    ],

    # Phase 3: Modeling
    LearningState.PHASE3_MODELING: [
        LearningState.PHASE3_COMPLETE,
        LearningState.ERROR,
        LearningState.PAUSED,
    ],
    LearningState.PHASE3_COMPLETE: [
        LearningState.SAVING,
        LearningState.PAUSED,
    ],

    # Final states
    LearningState.SAVING: [
        LearningState.COMPLETE,
        LearningState.ERROR,
    ],
    LearningState.COMPLETE: [],  # Terminal state

    # Special states
    LearningState.ERROR: [
        LearningState.INIT,       # Restart
        LearningState.PAUSED,     # Pause
    ],
    LearningState.PAUSED: [
        # Can resume to any running state
        LearningState.DESIGN_ANALYSIS,
        LearningState.DESIGN_OBJECTIVES,
        LearningState.DESIGN_RUBRIC,
        LearningState.PHASE1_RUNNING,
        LearningState.PHASE2_SCORING,
        LearningState.PHASE2_COACHING,
        LearningState.PHASE2_FIXING,
        LearningState.PHASE3_MODELING,
        LearningState.SAVING,
        LearningState.INIT,
    ],
}


@dataclass
class StateTransition:
    """Record of a state transition."""
    from_state: LearningState
    to_state: LearningState
    timestamp: datetime
    reason: str = ""
    context_snapshot: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_state": self.from_state.name,
            "to_state": self.to_state.name,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "context_snapshot": self.context_snapshot,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateTransition":
        return cls(
            from_state=LearningState[data["from_state"]],
            to_state=LearningState[data["to_state"]],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            reason=data.get("reason", ""),
            context_snapshot=data.get("context_snapshot"),
        )


@dataclass
class IterationRecord:
    """Record of a single iteration in the iterative scaffolding loop."""
    iteration_number: int
    teacher_hint: str
    student_response: str
    predicted_answer: Optional[str] = None
    is_correct: bool = False
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration_number": self.iteration_number,
            "teacher_hint": self.teacher_hint,
            "student_response": self.student_response,
            "predicted_answer": self.predicted_answer,
            "is_correct": self.is_correct,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationRecord":
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])
        return cls(
            iteration_number=data["iteration_number"],
            teacher_hint=data.get("teacher_hint", ""),
            student_response=data.get("student_response", ""),
            predicted_answer=data.get("predicted_answer"),
            is_correct=data.get("is_correct", False),
            timestamp=timestamp,
        )


@dataclass
class QuestionProgress:
    """Progress tracking for individual questions."""
    question_id: str
    phase1_complete: bool = False
    phase1_correct: Optional[bool] = None
    phase2_complete: bool = False
    phase2_correct: Optional[bool] = None
    phase3_complete: bool = False
    sft_case: Optional[str] = None  # 'A', 'A-Failed', 'B', 'C', or None

    # Iterative scaffolding tracking (NEW)
    phase1_iteration_count: int = 0
    phase1_iterations: List[Dict] = field(default_factory=list)
    phase1_conversation_history: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "phase1_complete": self.phase1_complete,
            "phase1_correct": self.phase1_correct,
            "phase2_complete": self.phase2_complete,
            "phase2_correct": self.phase2_correct,
            "phase3_complete": self.phase3_complete,
            "sft_case": self.sft_case,
            "phase1_iteration_count": self.phase1_iteration_count,
            "phase1_iterations": self.phase1_iterations,
            "phase1_conversation_history": self.phase1_conversation_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuestionProgress":
        return cls(
            question_id=data["question_id"],
            phase1_complete=data.get("phase1_complete", False),
            phase1_correct=data.get("phase1_correct"),
            phase2_complete=data.get("phase2_complete", False),
            phase2_correct=data.get("phase2_correct"),
            phase3_complete=data.get("phase3_complete", False),
            sft_case=data.get("sft_case"),
            phase1_iteration_count=data.get("phase1_iteration_count", 0),
            phase1_iterations=data.get("phase1_iterations", []),
            phase1_conversation_history=data.get("phase1_conversation_history", []),
        )


@dataclass
class LearningContext:
    """Context data for the learning pipeline."""

    # Basic info
    domain: str = ""
    train_dataset: str = ""
    terminal_goal: str = ""
    student_model: str = ""
    teacher_model: str = ""

    # Design results
    design_result: Optional[Dict[str, Any]] = None
    task_analysis: str = ""
    performance_objectives: List[Dict] = field(default_factory=list)
    rubric: Optional[Dict] = None

    # Question data
    questions: List[Dict] = field(default_factory=list)
    total_questions: int = 0
    current_question_index: int = 0

    # Phase results
    phase1_results: List[Dict] = field(default_factory=list)
    phase2_results: List[Dict] = field(default_factory=list)
    phase3_results: List[Dict] = field(default_factory=list)
    coaching_db: Dict = field(default_factory=dict)

    # Intermediate results (for Phase 2/3)
    incorrect_after_phase1: List[Dict] = field(default_factory=list)
    still_incorrect_after_phase2: List[Dict] = field(default_factory=list)

    # Per-question progress (for resume)
    question_progress: Dict[str, QuestionProgress] = field(default_factory=dict)

    # Progress tracking
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    phase1_processed: int = 0
    phase2_processed: int = 0
    phase3_processed: int = 0

    # Iterative scaffolding statistics (NEW)
    phase1_first_attempt_correct: int = 0    # Solved on iteration 1
    phase1_multi_attempt_correct: int = 0    # Solved on iterations 2-5
    phase1_failed_reconstructed: int = 0     # Failed after 5 attempts, reconstructed

    # Error tracking
    last_error: Optional[str] = None
    error_count: int = 0
    retry_count: int = 0
    max_retries: int = 3

    # Checkpoint path
    checkpoint_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "domain": self.domain,
            "train_dataset": self.train_dataset,
            "terminal_goal": self.terminal_goal,
            "student_model": self.student_model,
            "teacher_model": self.teacher_model,
            "design_result": self.design_result,
            "task_analysis": self.task_analysis,
            "performance_objectives": self.performance_objectives,
            "rubric": self.rubric,
            "questions": self.questions,
            "total_questions": self.total_questions,
            "current_question_index": self.current_question_index,
            "phase1_results": self.phase1_results,
            "phase2_results": self.phase2_results,
            "phase3_results": self.phase3_results,
            "coaching_db": self.coaching_db,
            "incorrect_after_phase1": self.incorrect_after_phase1,
            "still_incorrect_after_phase2": self.still_incorrect_after_phase2,
            "question_progress": {
                qid: qp.to_dict()
                for qid, qp in self.question_progress.items()
            },
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "phase1_processed": self.phase1_processed,
            "phase2_processed": self.phase2_processed,
            "phase3_processed": self.phase3_processed,
            "phase1_first_attempt_correct": self.phase1_first_attempt_correct,
            "phase1_multi_attempt_correct": self.phase1_multi_attempt_correct,
            "phase1_failed_reconstructed": self.phase1_failed_reconstructed,
            "last_error": self.last_error,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningContext":
        """Deserialize from dictionary."""
        ctx = cls(
            domain=data.get("domain", ""),
            train_dataset=data.get("train_dataset", ""),
            terminal_goal=data.get("terminal_goal", ""),
            student_model=data.get("student_model", ""),
            teacher_model=data.get("teacher_model", ""),
            design_result=data.get("design_result"),
            task_analysis=data.get("task_analysis", ""),
            performance_objectives=data.get("performance_objectives", []),
            rubric=data.get("rubric"),
            questions=data.get("questions", []),
            total_questions=data.get("total_questions", 0),
            current_question_index=data.get("current_question_index", 0),
            phase1_results=data.get("phase1_results", []),
            phase2_results=data.get("phase2_results", []),
            phase3_results=data.get("phase3_results", []),
            coaching_db=data.get("coaching_db", {}),
            incorrect_after_phase1=data.get("incorrect_after_phase1", []),
            still_incorrect_after_phase2=data.get("still_incorrect_after_phase2", []),
            phase1_processed=data.get("phase1_processed", 0),
            phase2_processed=data.get("phase2_processed", 0),
            phase3_processed=data.get("phase3_processed", 0),
            phase1_first_attempt_correct=data.get("phase1_first_attempt_correct", 0),
            phase1_multi_attempt_correct=data.get("phase1_multi_attempt_correct", 0),
            phase1_failed_reconstructed=data.get("phase1_failed_reconstructed", 0),
            last_error=data.get("last_error"),
            error_count=data.get("error_count", 0),
            retry_count=data.get("retry_count", 0),
        )

        # Restore datetime fields
        if data.get("started_at"):
            ctx.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("updated_at"):
            ctx.updated_at = datetime.fromisoformat(data["updated_at"])
        if data.get("checkpoint_path"):
            ctx.checkpoint_path = Path(data["checkpoint_path"])

        # Restore question_progress
        for qid, qp_data in data.get("question_progress", {}).items():
            ctx.question_progress[qid] = QuestionProgress.from_dict(qp_data)

        return ctx

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get progress summary."""
        phase1_correct = sum(1 for r in self.phase1_results if r.get("phase1_correct"))
        return {
            "total_questions": self.total_questions,
            "phase1_processed": self.phase1_processed,
            "phase1_correct": phase1_correct,
            "phase2_processed": self.phase2_processed,
            "phase2_fixed": len([r for r in self.phase2_results if r.get("phase2_correct")]),
            "phase3_processed": self.phase3_processed,
            "phase3_modeled": len(self.phase3_results),
            "error_count": self.error_count,
        }


# Type alias for callbacks
StateCallback = Callable[["LearningStateMachine", LearningContext], None]
TransitionCallback = Callable[[StateTransition], None]


class LearningStateMachine:
    """
    ID-MAS Learning Pipeline State Machine.

    Manages state transitions, context, and event callbacks.
    Supports checkpoint-based resume functionality.

    Example:
        sm = LearningStateMachine()
        sm.on_enter(LearningState.PHASE1_RUNNING, start_phase1)
        sm.transition_to(LearningState.PHASE1_RUNNING)
    """

    def __init__(self, context: Optional[LearningContext] = None):
        """Initialize state machine."""
        self._state = LearningState.INIT
        self._context = context or LearningContext()
        self._history: List[StateTransition] = []
        self._enter_callbacks: Dict[LearningState, List[StateCallback]] = {}
        self._exit_callbacks: Dict[LearningState, List[StateCallback]] = {}
        self._transition_callbacks: List[TransitionCallback] = []

    @property
    def state(self) -> LearningState:
        """Get current state."""
        return self._state

    @property
    def context(self) -> LearningContext:
        """Get context data."""
        return self._context

    @property
    def history(self) -> List[StateTransition]:
        """Get state transition history."""
        return self._history.copy()

    def can_transition_to(self, target: LearningState) -> bool:
        """Check if transition to target state is valid."""
        valid_targets = VALID_TRANSITIONS.get(self._state, [])
        return target in valid_targets

    def transition_to(
        self,
        target: LearningState,
        reason: str = "",
        save_snapshot: bool = True,
        force: bool = False,
    ) -> bool:
        """
        Transition to new state.

        Args:
            target: Target state
            reason: Transition reason
            save_snapshot: Save context snapshot to transition record
            force: Force transition even if not valid

        Returns:
            True if transition succeeded

        Raises:
            ValueError: If transition is not valid and force=False
        """
        if not force and not self.can_transition_to(target):
            valid = VALID_TRANSITIONS.get(self._state, [])
            raise ValueError(
                f"Invalid transition: {self._state} -> {target}. "
                f"Valid transitions: {[s.name for s in valid]}"
            )

        old_state = self._state
        now = datetime.now()

        # Execute exit callbacks
        for callback in self._exit_callbacks.get(old_state, []):
            try:
                callback(self, self._context)
            except Exception as e:
                print(f"Warning: on_exit callback error: {e}")

        # Update state
        self._state = target
        self._context.updated_at = now

        # Record transition
        transition = StateTransition(
            from_state=old_state,
            to_state=target,
            timestamp=now,
            reason=reason,
            context_snapshot=self._context.get_progress_summary() if save_snapshot else None,
        )
        self._history.append(transition)

        # Execute transition callbacks
        for callback in self._transition_callbacks:
            try:
                callback(transition)
            except Exception as e:
                print(f"Warning: on_transition callback error: {e}")

        # Execute enter callbacks
        for callback in self._enter_callbacks.get(target, []):
            try:
                callback(self, self._context)
            except Exception as e:
                print(f"Warning: on_enter callback error: {e}")

        return True

    def on_enter(self, state: LearningState, callback: StateCallback) -> None:
        """Register callback for entering a state."""
        if state not in self._enter_callbacks:
            self._enter_callbacks[state] = []
        self._enter_callbacks[state].append(callback)

    def on_exit(self, state: LearningState, callback: StateCallback) -> None:
        """Register callback for exiting a state."""
        if state not in self._exit_callbacks:
            self._exit_callbacks[state] = []
        self._exit_callbacks[state].append(callback)

    def on_transition(self, callback: TransitionCallback) -> None:
        """Register callback for any state transition."""
        self._transition_callbacks.append(callback)

    def reset(self) -> None:
        """Reset state machine to initial state."""
        self._state = LearningState.INIT
        self._context = LearningContext()
        self._history.clear()

    def set_error(self, error_message: str) -> None:
        """Set error and transition to ERROR state."""
        self._context.last_error = error_message
        self._context.error_count += 1
        if self.can_transition_to(LearningState.ERROR):
            self.transition_to(LearningState.ERROR, reason=f"Error: {error_message}")

    def pause(self, reason: str = "User requested pause") -> None:
        """Transition to PAUSED state."""
        if self.can_transition_to(LearningState.PAUSED):
            self.transition_to(LearningState.PAUSED, reason=reason)

    # ==================== Checkpoint Support ====================

    def save_checkpoint(self, path: Optional[Path] = None) -> Path:
        """
        Save current state to checkpoint file.

        New path structure: data/{domain}/train/{Model}/{dataset}_checkpoint_{timestamp}.json

        Args:
            path: Save path (uses default if None)

        Returns:
            Path to saved checkpoint file
        """
        if path is None:
            if self._context.checkpoint_path:
                path = self._context.checkpoint_path
            else:
                # Create default path with new structure
                from config.config import DATA_DIR, get_model_short_name
                model_short = get_model_short_name(self._context.student_model)
                checkpoint_dir = DATA_DIR / self._context.domain / "train" / model_short
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = checkpoint_dir / f"{self._context.train_dataset}_checkpoint_{timestamp}.json"

        checkpoint_data = {
            "state": self._state.name,
            "context": self._context.to_dict(),
            "history": [t.to_dict() for t in self._history],
            "saved_at": datetime.now().isoformat(),
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

        self._context.checkpoint_path = path
        return path

    @classmethod
    def load_checkpoint(cls, path: Path) -> "LearningStateMachine":
        """
        Load state machine from checkpoint file.

        Args:
            path: Checkpoint file path

        Returns:
            Restored LearningStateMachine
        """
        with open(path, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)

        context = LearningContext.from_dict(checkpoint_data["context"])
        context.checkpoint_path = path

        machine = cls(context=context)
        machine._state = LearningState[checkpoint_data["state"]]

        # Restore history
        for t_data in checkpoint_data.get("history", []):
            transition = StateTransition.from_dict(t_data)
            machine._history.append(transition)

        return machine

    @classmethod
    def find_latest_checkpoint(
        cls,
        domain: str,
        train_dataset: str,
        model_name: Optional[str] = None
    ) -> Optional[Path]:
        """
        Find the most recent checkpoint file.

        New path structure: data/{domain}/train/{Model}/{dataset}_checkpoint_{timestamp}.json

        Args:
            domain: Domain name
            train_dataset: Training dataset name
            model_name: Model name (required for new structure)

        Returns:
            Path to checkpoint file (None if not found)
        """
        from config.config import DATA_DIR, get_model_short_name

        if model_name:
            model_short = get_model_short_name(model_name)
            checkpoint_dir = DATA_DIR / domain / "train" / model_short
        else:
            # Fallback: search all model directories
            train_dir = DATA_DIR / domain / "train"
            if not train_dir.exists():
                return None

            all_checkpoints = []
            for model_dir in train_dir.iterdir():
                if model_dir.is_dir():
                    pattern = f"{train_dataset}_checkpoint_*.json"
                    all_checkpoints.extend(model_dir.glob(pattern))

            if not all_checkpoints:
                return None

            return sorted(all_checkpoints, reverse=True)[0]

        if not checkpoint_dir.exists():
            return None

        pattern = f"{train_dataset}_checkpoint_*.json"
        checkpoints = sorted(checkpoint_dir.glob(pattern), reverse=True)

        return checkpoints[0] if checkpoints else None

    # ==================== Utility Methods ====================

    def get_summary(self) -> Dict[str, Any]:
        """Get state machine summary."""
        return {
            "current_state": self._state.name,
            "context_summary": self._context.get_progress_summary(),
            "transition_count": len(self._history),
            "last_transition": self._history[-1].to_dict() if self._history else None,
            "started_at": self._context.started_at.isoformat() if self._context.started_at else None,
            "updated_at": self._context.updated_at.isoformat() if self._context.updated_at else None,
        }

    def is_complete(self) -> bool:
        """Check if pipeline is complete."""
        return self._state == LearningState.COMPLETE

    def is_error(self) -> bool:
        """Check if in error state."""
        return self._state == LearningState.ERROR

    def is_paused(self) -> bool:
        """Check if paused."""
        return self._state == LearningState.PAUSED

    def get_resumable_state(self) -> Optional[LearningState]:
        """
        Determine which state to resume from based on progress.

        Returns:
            State to resume from
        """
        ctx = self._context

        # Design not complete
        if not ctx.design_result:
            if not ctx.task_analysis:
                return LearningState.DESIGN_ANALYSIS
            if not ctx.performance_objectives:
                return LearningState.DESIGN_OBJECTIVES
            return LearningState.DESIGN_RUBRIC

        # Phase 1 not complete
        if ctx.phase1_processed < ctx.total_questions:
            return LearningState.PHASE1_RUNNING

        # Phase 2 needed and not complete
        if ctx.incorrect_after_phase1:
            if not ctx.coaching_db:
                return LearningState.PHASE2_SCORING
            if ctx.phase2_processed < len(ctx.incorrect_after_phase1):
                return LearningState.PHASE2_FIXING

        # Phase 3 needed and not complete
        if ctx.still_incorrect_after_phase2:
            if ctx.phase3_processed < len(ctx.still_incorrect_after_phase2):
                return LearningState.PHASE3_MODELING

        # Only saving left
        return LearningState.SAVING

    def get_state_duration(self, state: LearningState) -> float:
        """Calculate total time spent in a state (seconds)."""
        total_seconds = 0.0
        in_state = False
        enter_time = None

        for transition in self._history:
            if transition.to_state == state:
                in_state = True
                enter_time = transition.timestamp
            elif in_state and transition.from_state == state:
                in_state = False
                if enter_time:
                    total_seconds += (transition.timestamp - enter_time).total_seconds()

        # If still in this state
        if in_state and enter_time:
            total_seconds += (datetime.now() - enter_time).total_seconds()

        return total_seconds
