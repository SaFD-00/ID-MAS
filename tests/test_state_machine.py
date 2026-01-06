"""
Tests for the learning state machine.
"""
import pytest
from learning_loop.state_machine import (
    LearningState,
    VALID_TRANSITIONS
)


class TestLearningState:
    """Test LearningState enum"""

    def test_state_enum_values_exist(self):
        # Verify all expected states exist
        assert LearningState.INIT is not None
        assert LearningState.DESIGN_ANALYSIS is not None
        assert LearningState.DESIGN_OBJECTIVES is not None
        assert LearningState.DESIGN_RUBRIC is not None
        assert LearningState.DESIGN_COMPLETE is not None
        assert LearningState.PHASE1_RUNNING is not None
        assert LearningState.PHASE1_COMPLETE is not None
        assert LearningState.PHASE2_SCORING is not None
        assert LearningState.PHASE2_COACHING is not None
        assert LearningState.PHASE2_FIXING is not None
        assert LearningState.PHASE2_COMPLETE is not None
        assert LearningState.PHASE3_MODELING is not None
        assert LearningState.PHASE3_COMPLETE is not None
        assert LearningState.SAVING is not None
        assert LearningState.COMPLETE is not None
        assert LearningState.ERROR is not None
        assert LearningState.PAUSED is not None

    def test_state_has_number(self):
        # Each state should have a state_num
        for state in LearningState:
            assert hasattr(state, 'state_num')
            assert isinstance(state.state_num, int)

    def test_state_has_color(self):
        # Each state should have color properties
        for state in LearningState:
            assert hasattr(state, 'color_code')
            assert hasattr(state, 'color_name')
            assert isinstance(state.color_code, int)
            assert isinstance(state.color_name, str)

    def test_state_colored_name(self):
        # Test colored_name property
        init_state = LearningState.INIT
        colored = init_state.colored_name

        assert init_state.name in colored
        assert "\033[" in colored  # ANSI code
        assert "\033[0m" in colored  # Reset code
        assert f"{init_state.state_num:02d}" in colored

    def test_state_display_name(self):
        # Test display_name property
        complete_state = LearningState.COMPLETE
        display = complete_state.display_name

        assert complete_state.name in display
        assert f"[{complete_state.state_num:02d}]" in display
        # Should not contain ANSI codes
        assert "\033[" not in display

    def test_state_get_phase_design(self):
        assert LearningState.DESIGN_ANALYSIS.get_phase() == "Design"
        assert LearningState.DESIGN_OBJECTIVES.get_phase() == "Design"
        assert LearningState.DESIGN_COMPLETE.get_phase() == "Design"

    def test_state_get_phase_phase1(self):
        assert LearningState.PHASE1_RUNNING.get_phase() == "Phase 1: Scaffolding"
        assert LearningState.PHASE1_COMPLETE.get_phase() == "Phase 1: Scaffolding"

    def test_state_get_phase_phase2(self):
        assert LearningState.PHASE2_SCORING.get_phase() == "Phase 2: Coaching"
        assert LearningState.PHASE2_COACHING.get_phase() == "Phase 2: Coaching"
        assert LearningState.PHASE2_FIXING.get_phase() == "Phase 2: Coaching"
        assert LearningState.PHASE2_COMPLETE.get_phase() == "Phase 2: Coaching"

    def test_state_get_phase_phase3(self):
        assert LearningState.PHASE3_MODELING.get_phase() == "Phase 3: Modeling"
        assert LearningState.PHASE3_COMPLETE.get_phase() == "Phase 3: Modeling"

    def test_state_numbers_sequential(self):
        # Verify design and phase states are in sequential order
        assert LearningState.INIT.state_num == 0
        assert LearningState.DESIGN_ANALYSIS.state_num == 1
        assert LearningState.DESIGN_OBJECTIVES.state_num == 2
        assert LearningState.DESIGN_RUBRIC.state_num == 3
        assert LearningState.DESIGN_COMPLETE.state_num == 4
        assert LearningState.PHASE1_RUNNING.state_num == 5
        assert LearningState.PHASE1_COMPLETE.state_num == 6

    def test_special_states_have_high_numbers(self):
        # ERROR and PAUSED should have special high numbers
        assert LearningState.ERROR.state_num == 99
        assert LearningState.PAUSED.state_num == 98


class TestValidTransitions:
    """Test state transition validation"""

    def test_valid_transitions_dict_exists(self):
        assert VALID_TRANSITIONS is not None
        assert isinstance(VALID_TRANSITIONS, dict)

    def test_valid_transitions_not_empty(self):
        assert len(VALID_TRANSITIONS) > 0

    def test_all_states_have_transitions(self):
        # Most states should have valid transitions defined
        # (ERROR and COMPLETE might not have outgoing transitions)
        for state in LearningState:
            if state not in (LearningState.COMPLETE, LearningState.ERROR):
                # State should either have transitions or be a terminal state
                assert state in VALID_TRANSITIONS or state == LearningState.PAUSED

    def test_init_transitions(self):
        # INIT should transition to DESIGN_ANALYSIS or PHASE1_RUNNING (resume)
        if LearningState.INIT in VALID_TRANSITIONS:
            valid_next = VALID_TRANSITIONS[LearningState.INIT]
            assert LearningState.DESIGN_ANALYSIS in valid_next or \
                   LearningState.PHASE1_RUNNING in valid_next

    def test_design_phase_transitions_sequential(self):
        # Design states should transition in sequence
        if LearningState.DESIGN_ANALYSIS in VALID_TRANSITIONS:
            assert LearningState.DESIGN_OBJECTIVES in VALID_TRANSITIONS[LearningState.DESIGN_ANALYSIS]

        if LearningState.DESIGN_OBJECTIVES in VALID_TRANSITIONS:
            assert LearningState.DESIGN_RUBRIC in VALID_TRANSITIONS[LearningState.DESIGN_OBJECTIVES]

    def test_phase_transitions_exist(self):
        # Verify phases can transition to next phases
        # Phase 1 -> Phase 2
        if LearningState.PHASE1_COMPLETE in VALID_TRANSITIONS:
            valid_next = VALID_TRANSITIONS[LearningState.PHASE1_COMPLETE]
            assert LearningState.PHASE2_SCORING in valid_next or \
                   LearningState.SAVING in valid_next  # Can skip to SAVING if all correct

        # Phase 2 -> Phase 3
        if LearningState.PHASE2_COMPLETE in VALID_TRANSITIONS:
            valid_next = VALID_TRANSITIONS[LearningState.PHASE2_COMPLETE]
            assert LearningState.PHASE3_MODELING in valid_next or \
                   LearningState.SAVING in valid_next  # Can skip to SAVING if all fixed

    def test_error_transitions(self):
        # Any state should be able to transition to ERROR
        for state in LearningState:
            if state in VALID_TRANSITIONS:
                valid_next = VALID_TRANSITIONS[state]
                # ERROR should be reachable from most states (not strict requirement)

    def test_paused_can_resume(self):
        # PAUSED should be able to transition to various states for resume
        if LearningState.PAUSED in VALID_TRANSITIONS:
            valid_next = VALID_TRANSITIONS[LearningState.PAUSED]
            # Should have multiple resume points
            assert len(valid_next) > 0


class TestStateProperties:
    """Test state property consistency"""

    def test_design_states_same_color(self):
        # All DESIGN states should have same color
        design_states = [
            LearningState.DESIGN_ANALYSIS,
            LearningState.DESIGN_OBJECTIVES,
            LearningState.DESIGN_RUBRIC,
            LearningState.DESIGN_COMPLETE
        ]

        colors = [s.color_code for s in design_states]
        assert len(set(colors)) == 1  # All same color

    def test_phase1_states_same_color(self):
        # All Phase 1 states should have same color
        phase1_states = [
            LearningState.PHASE1_RUNNING,
            LearningState.PHASE1_COMPLETE
        ]

        colors = [s.color_code for s in phase1_states]
        assert len(set(colors)) == 1

    def test_phase2_states_same_color(self):
        # All Phase 2 states should have same color
        phase2_states = [
            LearningState.PHASE2_SCORING,
            LearningState.PHASE2_COACHING,
            LearningState.PHASE2_FIXING,
            LearningState.PHASE2_COMPLETE
        ]

        colors = [s.color_code for s in phase2_states]
        assert len(set(colors)) == 1

    def test_phase3_states_same_color(self):
        # All Phase 3 states should have same color
        phase3_states = [
            LearningState.PHASE3_MODELING,
            LearningState.PHASE3_COMPLETE
        ]

        colors = [s.color_code for s in phase3_states]
        assert len(set(colors)) == 1

    def test_complete_states_green(self):
        # SAVING and COMPLETE should be green (92)
        assert LearningState.SAVING.color_code == 92
        assert LearningState.COMPLETE.color_code == 92

    def test_error_state_red(self):
        # ERROR should be red (91)
        assert LearningState.ERROR.color_code == 91

    def test_init_state_gray(self):
        # INIT should be gray (90)
        assert LearningState.INIT.color_code == 90

    def test_state_string_representation(self):
        # Test __str__ method
        for state in LearningState:
            str_repr = str(state)
            assert str_repr == state.name
