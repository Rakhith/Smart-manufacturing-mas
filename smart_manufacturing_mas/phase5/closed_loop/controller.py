"""Closed-Loop Prescriptive Maintenance Controller."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd

from phase5.closed_loop.executor import ActionExecutor
from phase5.closed_loop.feedback_store import FeedbackStore
from phase5.closed_loop.transition_models import (
    BaseStateTransitionModel,
    HistoricalReplayTransition,
    SurrogateStateTransition,
)
from phase5.closed_loop.types import ExecutionMode, MaintenanceOutcome, TransitionRecord
from phase5.frozen_model import FrozenPhase4Recommender


class ClosedLoopController:
    """Orchestrates prescriptive recommendation, action selection, state transition, and feedback."""

    def __init__(
        self,
        recommender: FrozenPhase4Recommender,
        executor: Optional[ActionExecutor] = None,
        feedback_store: Optional[FeedbackStore] = None,
        mode: ExecutionMode = ExecutionMode.HISTORICAL_REPLAY,
        transition_model: Optional[BaseStateTransitionModel] = None,
    ):
        self.recommender = recommender
        self.executor = executor or ActionExecutor()
        self.feedback_store = feedback_store or FeedbackStore()
        self.mode = mode

        if transition_model is not None:
            self.transition_model = transition_model
        elif mode == ExecutionMode.HISTORICAL_REPLAY:
            self.transition_model = HistoricalReplayTransition()
        else:
            self.transition_model = SurrogateStateTransition()

        self.step_counter = 0

    def step(
        self,
        decision_state: Dict[str, Any],
        timestamp: str,
        asset_id: str,
        next_telemetry_state: Optional[Dict[str, Any]] = None,
        ground_truth_context: Optional[Dict[str, Any]] = None,
        enforce_preconditions: bool = True,
    ) -> TransitionRecord:
        """Executes a single closed-loop step for an asset."""
        self.step_counter += 1
        current_sev = decision_state.get("decision_severity", "WATCH")
        arch = decision_state.get("asset_context", {}).get("machine_archetype", "industrial_machine")
        pre_hi = float(decision_state.get("input_machine_state", {}).get("health_degradation", {}).get("health_index", 75.0))

        # 1. Generate prescriptive rankings via frozen recommender
        ranked_candidates = self.recommender.rank_candidates(decision_state)

        # 2. Select executable action via action executor
        selected_action = self.executor.select_executable_action(
            ranked_recommendations=ranked_candidates,
            current_severity=current_sev,
            machine_archetype=arch,
            enforce_preconditions=enforce_preconditions,
        )

        # 3. Execute transition model
        post_state, outcome = self.transition_model.transition(
            current_state=decision_state,
            executed_action=selected_action,
            next_observed_telemetry=next_telemetry_state,
            ground_truth_context=ground_truth_context,
        )

        post_sev = post_state.get("decision_severity", current_sev)
        post_hi = pre_hi + outcome.health_index_delta

        # 4. Check ground truth alignment
        gt = ground_truth_context or {}
        matched_wo = gt.get("matched_work_order")
        matched_fail = gt.get("matched_failure_event")
        gt_matched = False
        if matched_wo:
            canonical_target = matched_wo.get("canonical_action_id")
            if canonical_target and selected_action["action_id"] == canonical_target:
                gt_matched = True

        # 5. Create transition record
        record = TransitionRecord(
            step=self.step_counter,
            timestamp=timestamp,
            asset_id=asset_id,
            pre_severity=current_sev,
            pre_health_index=pre_hi,
            recommended_action_id=ranked_candidates[0]["action_id"],
            recommended_action_name=ranked_candidates[0]["action_name"],
            executed_action_id=selected_action["action_id"],
            outcome=outcome,
            post_severity=post_sev,
            post_health_index=post_hi,
            execution_mode=self.mode,
            ground_truth_matched=gt_matched,
            ground_truth_work_order=matched_wo,
            ground_truth_failure=matched_fail,
        )

        # 6. Store in episodic feedback store
        self.feedback_store.add_record(record)
        return record
