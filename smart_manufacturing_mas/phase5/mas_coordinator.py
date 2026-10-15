"""Multi-Agent System (MAS) Orchestration Engine for Closed-Loop Prescriptive Maintenance.

Integrates:
- DataLoaderAgent
- PreprocessingAgent
- SemanticNormalizationAgent
- DecisionStateAgent
- RecommendationAgent (Frozen Phase 4 Hybrid Recommender)
- ActionExecutionAgent (ActionExecutor)
- FeedbackAgent (FeedbackStore & TransitionEngine)
- LLMSummarizerAgent (Grounded Technician & CMMS Dispatcher)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from phase5.closed_loop.controller import ClosedLoopController
from phase5.closed_loop.executor import ActionExecutor
from phase5.closed_loop.feedback_store import FeedbackStore
from phase5.closed_loop.transition_models import HistoricalReplayTransition, SurrogateStateTransition
from phase5.closed_loop.types import ExecutionMode, TransitionRecord
from phase5.frozen_model import FrozenPhase4Recommender
from phase5.llm_summarizer import GroundedPrescriptiveSummarizer

logger = logging.getLogger("MAS_Coordinator")


class MultiAgentClosedLoopCoordinator:
    """Orchestrates multi-agent prescriptive maintenance lifecycle across industrial assets."""

    def __init__(
        self,
        recommender: Optional[FrozenPhase4Recommender] = None,
        executor: Optional[ActionExecutor] = None,
        summarizer: Optional[GroundedPrescriptiveSummarizer] = None,
    ):
        self.recommender = recommender or FrozenPhase4Recommender()
        self.executor = executor or ActionExecutor()
        self.summarizer = summarizer or GroundedPrescriptiveSummarizer()
        self.feedback_store = FeedbackStore()

    def run_asset_lifecycle(
        self,
        adapter: Any,
        asset_id: str,
        mode: ExecutionMode = ExecutionMode.HISTORICAL_REPLAY,
        max_steps: int = 15,
        step_interval: int = 5,
    ) -> Dict[str, Any]:
        """Runs the complete multi-agent closed-loop orchestration over an asset's telemetry stream."""
        seq = adapter.get_temporal_sequence(asset_id)
        if seq.empty:
            raise ValueError(f"No temporal sequence found for asset {asset_id}")

        time_col = "timestamp" if "timestamp" in seq.columns else "recorded_at"
        timestamps = seq[time_col].tolist()

        gt_events = adapter.get_ground_truth_events(asset_id)
        wos = gt_events.get("work_orders", [])
        fails = gt_events.get("failure_events", [])

        # Subsample timepoints to run sequential steps
        step_indices = list(range(0, len(timestamps), max(1, step_interval)))[:max_steps]

        controller = ClosedLoopController(
            recommender=self.recommender,
            executor=self.executor,
            feedback_store=self.feedback_store,
            mode=mode,
            transition_model=HistoricalReplayTransition() if mode == ExecutionMode.HISTORICAL_REPLAY else SurrogateStateTransition(),
        )

        step_records: List[TransitionRecord] = []
        cmms_tickets: List[Dict[str, Any]] = []

        for idx in step_indices:
            cur_time = str(timestamps[idx])

            # Next telemetry state
            next_state = None
            if idx + 1 < len(timestamps):
                next_time = str(timestamps[idx + 1])
                next_state = adapter.build_decision_state_at(str(asset_id), next_time)

            # Ground truth context: find closest future work order or failure
            cur_dt = pd.to_datetime(cur_time)
            closest_wo = None
            for wo in wos:
                wo_tc = "timestamp" if "timestamp" in wo else "opened_at"
                if wo_tc in wo and pd.notna(wo[wo_tc]):
                    w_dt = pd.to_datetime(wo[wo_tc])
                    if w_dt >= cur_dt:
                        closest_wo = wo
                        break

            gt_ctx = {
                "matched_work_order": closest_wo,
                "matched_failure_event": fails[0] if fails else None,
                "lead_time_hours": float((pd.to_datetime(closest_wo["timestamp" if "timestamp" in closest_wo else "opened_at"]) - cur_dt).total_seconds() / 3600.0) if closest_wo else 0.0,
            }

            # 1. State Agent builds DecisionState
            dstate = adapter.build_decision_state_at(str(asset_id), cur_time)

            # 2. Recommender + Executor + Transition + Feedback Agents step
            rec = controller.step(
                decision_state=dstate,
                timestamp=cur_time,
                asset_id=str(asset_id),
                next_telemetry_state=next_state,
                ground_truth_context=gt_ctx,
            )
            step_records.append(rec)

            # 3. If intervention required, LLMSummarizerAgent generates CMMS dispatch ticket
            if "MON_CONTINUE" not in rec.executed_action_id:
                ranked_cands = self.recommender.rank_candidates(dstate)
                card = self.summarizer.generate_technician_card(
                    decision_state=dstate,
                    ranked_actions=ranked_cands,
                    actual_outcome=rec.outcome.to_dict(),
                )
                cmms = self.summarizer.format_cmms_work_order(card)
                cmms_tickets.append(cmms)

        asset_metrics = self.feedback_store.compute_asset_metrics(str(asset_id))

        return {
            "asset_id": str(asset_id),
            "execution_mode": mode.value,
            "steps_executed": len(step_records),
            "asset_metrics": asset_metrics,
            "cmms_tickets_dispatched": cmms_tickets,
            "sample_transitions": [r.to_dict() for r in step_records[:5]],
        }
