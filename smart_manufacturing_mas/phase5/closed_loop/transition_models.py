"""State Transition Models for Closed-Loop Evaluation.

Implements two distinct, rigorously isolated evaluation modes:
1. Mode A: Historical Replay (Empirical Plant Ground Truth)
2. Mode B: Simulated Closed Loop (Surrogate Transition Model - Explicitly Labeled)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from phase5.closed_loop.types import ExecutionMode, MaintenanceOutcome


class BaseStateTransitionModel(ABC):
    """Abstract base class for state transition engines."""

    @abstractmethod
    def transition(
        self,
        current_state: Dict[str, Any],
        executed_action: Dict[str, Any],
        next_observed_telemetry: Optional[Dict[str, Any]] = None,
        ground_truth_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], MaintenanceOutcome]:
        pass


class HistoricalReplayTransition(BaseStateTransitionModel):
    """Mode A: Historical Replay against actual recorded maintenance events.

    In this mode, future physical plant events serve strictly as evaluation ground truth.
    The physical telemetry sequence evolves exactly as observed in the industrial logs.
    """

    def __init__(self, action_matching_fn: Optional[Any] = None):
        self.action_matching_fn = action_matching_fn

    def transition(
        self,
        current_state: Dict[str, Any],
        executed_action: Dict[str, Any],
        next_observed_telemetry: Optional[Dict[str, Any]] = None,
        ground_truth_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], MaintenanceOutcome]:
        gt = ground_truth_context or {}
        actual_work_order = gt.get("matched_work_order")
        actual_failure = gt.get("matched_failure_event")

        rec_aid = executed_action.get("action_id", "ACT_MON_CONTINUE")
        matched = False
        lead_time = float(gt.get("lead_time_hours", 0.0))
        downtime = 0.0
        parts_cost = 0.0
        labor_hours = 0.0

        if actual_work_order:
            downtime = float(actual_work_order.get("downtime_hours", 0.0))
            parts_cost = float(actual_work_order.get("parts_cost", actual_work_order.get("parts_cost_usd", 0.0)))
            labor_hours = float(actual_work_order.get("labor_hours", 0.0))
            canonical_target = actual_work_order.get("canonical_action_id")
            if canonical_target and rec_aid == canonical_target:
                matched = True

        prevented = False
        missed = False
        if actual_failure:
            if "REPL" in rec_aid or "CORR" in rec_aid or "OP_CONTROLLED_SHUTDOWN" in rec_aid:
                prevented = True
            else:
                missed = True

        outcome = MaintenanceOutcome(
            action_id=rec_aid,
            canonical_action_id=rec_aid,
            success=True,
            downtime_hours=downtime,
            parts_cost_usd=parts_cost,
            labor_hours=labor_hours,
            prevented_failure=prevented,
            missed_failure=missed,
            unnecessary_intervention=(matched is False and "MON" not in rec_aid and not actual_failure),
            lead_time_hours=lead_time,
            health_index_delta=0.0,
            notes="Evaluated via empirical historical plant replay.",
            is_simulated=False,
        )

        # Post state is determined directly by next observed telemetry
        post_state = next_observed_telemetry if next_observed_telemetry is not None else current_state
        return post_state, outcome


class SurrogateStateTransition(BaseStateTransitionModel):
    """Mode B: Simulated Closed Loop with physics-inspired surrogate dynamics.

    IMPORTANT: ALL transitions in this model are explicitly marked is_simulated=True.
    No simulated physical state transition is ever claimed as real empirical plant outcome.
    """

    def __init__(
        self,
        inspection_cost: float = 150.0,
        lubrication_cost: float = 300.0,
        alignment_cost: float = 650.0,
        bearing_replacement_cost: float = 2500.0,
        seal_replacement_cost: float = 1800.0,
        overhaul_cost: float = 8500.0,
        hourly_downtime_cost: float = 500.0,
    ):
        self.inspection_cost = inspection_cost
        self.lubrication_cost = lubrication_cost
        self.alignment_cost = alignment_cost
        self.bearing_replacement_cost = bearing_replacement_cost
        self.seal_replacement_cost = seal_replacement_cost
        self.overhaul_cost = overhaul_cost
        self.hourly_downtime_cost = hourly_downtime_cost

    def transition(
        self,
        current_state: Dict[str, Any],
        executed_action: Dict[str, Any],
        next_observed_telemetry: Optional[Dict[str, Any]] = None,
        ground_truth_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], MaintenanceOutcome]:
        rec_aid = executed_action.get("action_id", "ACT_MON_CONTINUE")
        pre_sev = current_state.get("decision_severity", "HEALTHY").upper()

        post_state = dict(current_state)
        # Deepcopy input machine state dict if present
        if "input_machine_state" in current_state:
            post_state["input_machine_state"] = {k: dict(v) for k, v in current_state["input_machine_state"].items()}

        downtime = 0.0
        parts_cost = 0.0
        labor_hours = 0.0
        hi_delta = 0.0
        prevented = False
        missed = False

        # --- Transition Rules by Canonical Action Class ---
        if rec_aid == "ACT_MON_CONTINUE":
            # Natural degradation
            if next_observed_telemetry:
                post_state = next_observed_telemetry
            else:
                hi_delta = -2.0

        elif rec_aid in {"ACT_MON_ENHANCED", "ACT_MON_PARAMETER_LOG"}:
            # Supervisory monitoring: state unchanged
            hi_delta = 0.0
            labor_hours = 0.5

        elif rec_aid.startswith("ACT_INSP_"):
            # Diagnostic inspection: information increases, physical state unchanged
            labor_hours = 2.0
            downtime = 0.5
            parts_cost = self.inspection_cost
            hi_delta = 0.0

        elif rec_aid == "ACT_CORR_LUBRICATE":
            # Lubrication: reduces friction and heat, partially restores health
            labor_hours = 1.5
            downtime = 1.0
            parts_cost = self.lubrication_cost
            hi_delta = +15.0
            post_state["decision_severity"] = "WATCH" if pre_sev in {"DEGRADING", "CRITICAL"} else "HEALTHY"

        elif rec_aid == "ACT_CORR_FASTEN_ALIGN":
            # Alignment & Fastening: mitigates vibration imbalance
            labor_hours = 3.5
            downtime = 2.5
            parts_cost = self.alignment_cost
            hi_delta = +20.0
            post_state["decision_severity"] = "WATCH" if pre_sev in {"DEGRADING", "CRITICAL"} else "HEALTHY"

        elif rec_aid == "ACT_CORR_CALIBRATE":
            # Recalibrate sensors
            labor_hours = 1.0
            downtime = 0.5
            parts_cost = 200.0
            hi_delta = +5.0

        elif rec_aid == "ACT_CORR_CLEAN_PURGE":
            # Clean and purge lines
            labor_hours = 2.5
            downtime = 1.5
            parts_cost = 350.0
            hi_delta = +10.0

        elif rec_aid == "ACT_REPL_BEARING":
            # Bearing replacement: complete restoration of mechanical assembly
            labor_hours = 8.0
            downtime = 6.0
            parts_cost = self.bearing_replacement_cost
            hi_delta = +45.0
            post_state["decision_severity"] = "HEALTHY"
            prevented = (pre_sev in {"DEGRADING", "CRITICAL"})

        elif rec_aid == "ACT_REPL_SEAL_VALVE":
            # Seal / valve replacement: restores fluid pressure integrity
            labor_hours = 5.0
            downtime = 4.0
            parts_cost = self.seal_replacement_cost
            hi_delta = +40.0
            post_state["decision_severity"] = "HEALTHY"
            prevented = (pre_sev in {"DEGRADING", "CRITICAL"})

        elif rec_aid == "ACT_REPL_OVERHAUL":
            # Full overhaul
            labor_hours = 24.0
            downtime = 16.0
            parts_cost = self.overhaul_cost
            hi_delta = +60.0
            post_state["decision_severity"] = "HEALTHY"
            prevented = (pre_sev in {"DEGRADING", "CRITICAL"})

        elif rec_aid in {"ACT_OP_CONTROLLED_SHUTDOWN", "ACT_OP_EMERGENCY_STOP"}:
            # Shutdown: prevents catastrophic failure
            downtime = 8.0
            labor_hours = 2.0
            post_state["decision_severity"] = "WATCH"
            prevented = (pre_sev == "CRITICAL")
            hi_delta = 0.0

        else:
            # Default mitigation
            labor_hours = 1.0
            hi_delta = +5.0

        outcome = MaintenanceOutcome(
            action_id=rec_aid,
            canonical_action_id=rec_aid,
            success=True,
            downtime_hours=downtime,
            parts_cost_usd=parts_cost,
            labor_hours=labor_hours,
            prevented_failure=prevented,
            missed_failure=missed,
            unnecessary_intervention=(pre_sev == "HEALTHY" and "MON" not in rec_aid),
            lead_time_hours=0.0,
            health_index_delta=hi_delta,
            notes="[SIMULATED SURROGATE] State delta generated via physics-informed surrogate transition model.",
            is_simulated=True,
        )

        return post_state, outcome
