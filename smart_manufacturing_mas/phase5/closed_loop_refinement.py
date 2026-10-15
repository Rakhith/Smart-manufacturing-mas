"""Closed-Loop Refinement & Sensitivity Analysis Engine (Parts 16, 17, 18).

Implements rigorous closed-loop evaluation separating empirical Historical Replay
from Physics-Informed Surrogate Simulation, with sensitivity analysis across
operational cost regimes and event-level cost-benefit accounting.

All simulated quantities are strictly marked is_simulated=True with
[SIMULATED SURROGATE] disclaimers to prevent any claim of empirical plant ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from phase5.closed_loop.controller import ClosedLoopController
from phase5.closed_loop.executor import ActionExecutor
from phase5.closed_loop.feedback_store import FeedbackStore
from phase5.closed_loop.transition_models import HistoricalReplayTransition, SurrogateStateTransition
from phase5.closed_loop.types import ExecutionMode, MaintenanceOutcome, TransitionRecord
from phase5.frozen_model import FrozenPhase4Recommender


class CostRegime(str, Enum):
    BASELINE = "baseline"
    CONSERVATIVE = "conservative"
    OPTIMISTIC = "optimistic"


@dataclass
class SensitivityParameters:
    regime: CostRegime = CostRegime.BASELINE
    hourly_downtime_cost: float = 500.0  # USD/hr
    labor_hourly_rate: float = 75.0      # USD/hr
    inspection_cost: float = 150.0       # USD
    lubrication_cost: float = 300.0      # USD
    alignment_cost: float = 650.0        # USD
    bearing_replacement_cost: float = 2500.0  # USD
    seal_replacement_cost: float = 1800.0     # USD
    overhaul_cost: float = 8500.0             # USD
    catastrophic_failure_penalty: float = 15000.0  # USD per unmitigated breakdown
    unnecessary_intervention_penalty: float = 250.0  # USD false alarm cost
    health_recovery_multiplier: float = 1.0  # Multiplier on HI restoration delta
    is_simulated: bool = True

    @classmethod
    def get_regime_params(cls, regime: CostRegime) -> SensitivityParameters:
        if regime == CostRegime.CONSERVATIVE:
            # Conservative: higher downtime cost, higher breakdown penalty, reduced restoration effect
            return cls(
                regime=CostRegime.CONSERVATIVE,
                hourly_downtime_cost=1000.0,
                labor_hourly_rate=95.0,
                bearing_replacement_cost=3200.0,
                seal_replacement_cost=2300.0,
                overhaul_cost=11000.0,
                catastrophic_failure_penalty=25000.0,
                unnecessary_intervention_penalty=500.0,
                health_recovery_multiplier=0.8,
            )
        elif regime == CostRegime.OPTIMISTIC:
            # Optimistic: lower downtime cost, efficient repairs, high restoration effect
            return cls(
                regime=CostRegime.OPTIMISTIC,
                hourly_downtime_cost=250.0,
                labor_hourly_rate=60.0,
                bearing_replacement_cost=2000.0,
                seal_replacement_cost=1400.0,
                overhaul_cost=6500.0,
                catastrophic_failure_penalty=10000.0,
                unnecessary_intervention_penalty=100.0,
                health_recovery_multiplier=1.2,
            )
        else:
            return cls(regime=CostRegime.BASELINE)


@dataclass
class PolicyEvaluationResult:
    policy_name: str
    execution_mode: str
    regime: str
    total_steps: int
    total_downtime_hours: float
    total_parts_cost_usd: float
    total_labor_hours: float
    total_labor_cost_usd: float
    downtime_cost_usd: float
    failure_penalty_usd: float
    unnecessary_penalty_usd: float
    total_cost_usd: float
    avoided_failures: int
    missed_failures: int
    unnecessary_interventions: int
    net_savings_vs_rtf_usd: float = 0.0
    savings_pct_vs_rtf: float = 0.0
    net_savings_vs_reactive_usd: float = 0.0
    savings_pct_vs_reactive: float = 0.0
    is_simulated: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ParametricSurrogateTransition(SurrogateStateTransition):
    """Configurable surrogate transition model honoring sensitivity parameters."""

    def __init__(self, params: SensitivityParameters):
        super().__init__(
            inspection_cost=params.inspection_cost,
            lubrication_cost=params.lubrication_cost,
            alignment_cost=params.alignment_cost,
            bearing_replacement_cost=params.bearing_replacement_cost,
            seal_replacement_cost=params.seal_replacement_cost,
            overhaul_cost=params.overhaul_cost,
            hourly_downtime_cost=params.hourly_downtime_cost,
        )
        self.params = params

    def transition(
        self,
        current_state: Dict[str, Any],
        executed_action: Dict[str, Any],
        next_observed_telemetry: Optional[Dict[str, Any]] = None,
        ground_truth_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], MaintenanceOutcome]:
        post_state, outcome = super().transition(
            current_state, executed_action, next_observed_telemetry, ground_truth_context
        )
        # Apply regime health multiplier
        adjusted_hi = outcome.health_index_delta * self.params.health_recovery_multiplier
        outcome.health_index_delta = adjusted_hi
        outcome.notes = f"[SIMULATED SURROGATE] Regime: {self.params.regime.value}. Recovery mult: {self.params.health_recovery_multiplier}"
        return post_state, outcome


class ClosedLoopRefinementEvaluator:
    """Evaluates policies under Historical Replay and Parametric Surrogate Simulation."""

    def __init__(self, recommender: FrozenPhase4Recommender):
        self.recommender = recommender

    def evaluate_historical_replay(
        self,
        decision_states_with_context: List[Dict[str, Any]],
        params: Optional[SensitivityParameters] = None,
    ) -> PolicyEvaluationResult:
        """Mode A: Replay against empirical recorded work orders and failures.

        Evaluates what actually happened historically vs what the AI prescribed.
        """
        p = params or SensitivityParameters.get_regime_params(CostRegime.BASELINE)
        transition_model = HistoricalReplayTransition()
        controller = ClosedLoopController(
            recommender=self.recommender,
            mode=ExecutionMode.HISTORICAL_REPLAY,
            transition_model=transition_model,
        )

        total_downtime = 0.0
        total_parts = 0.0
        total_labor = 0.0
        avoided_count = 0
        missed_count = 0
        unnecessary_count = 0

        for item in decision_states_with_context:
            dstate = item["decision_state"]
            ts = item.get("timestamp", "2026-01-01T00:00:00")
            asset_id = item.get("asset_id", "asset_1")
            gt_ctx = item.get("ground_truth_context", {})

            rec = controller.step(
                decision_state=dstate,
                timestamp=ts,
                asset_id=asset_id,
                ground_truth_context=gt_ctx,
            )
            out = rec.outcome
            total_downtime += out.downtime_hours
            total_parts += out.parts_cost_usd
            total_labor += out.labor_hours
            if out.prevented_failure:
                avoided_count += 1
            if out.missed_failure:
                missed_count += 1
            if out.unnecessary_intervention:
                unnecessary_count += 1

        labor_cost = total_labor * p.labor_hourly_rate
        downtime_cost = total_downtime * p.hourly_downtime_cost
        failure_penalty = missed_count * p.catastrophic_failure_penalty
        unnecessary_penalty = unnecessary_count * p.unnecessary_intervention_penalty
        total_cost = total_parts + labor_cost + downtime_cost + failure_penalty + unnecessary_penalty

        return PolicyEvaluationResult(
            policy_name="Phase4_Frozen_Recommender",
            execution_mode="Historical_Replay",
            regime="empirical_recorded",
            total_steps=len(decision_states_with_context),
            total_downtime_hours=round(total_downtime, 2),
            total_parts_cost_usd=round(total_parts, 2),
            total_labor_hours=round(total_labor, 2),
            total_labor_cost_usd=round(labor_cost, 2),
            downtime_cost_usd=round(downtime_cost, 2),
            failure_penalty_usd=round(failure_penalty, 2),
            unnecessary_penalty_usd=round(unnecessary_penalty, 2),
            total_cost_usd=round(total_cost, 2),
            avoided_failures=avoided_count,
            missed_failures=missed_count,
            unnecessary_interventions=unnecessary_count,
            is_simulated=False,
            notes="Evaluated strictly on historical plant records. Physical events unperturbed.",
        )

    def evaluate_surrogate_regime(
        self,
        decision_states_with_context: List[Dict[str, Any]],
        regime: CostRegime = CostRegime.BASELINE,
        custom_params: Optional[SensitivityParameters] = None,
        policy_override: Optional[str] = None,  # "RTF", "REACTIVE", or None for AI recommender
    ) -> PolicyEvaluationResult:
        """Mode B: Evaluates counterfactual outcomes using Parametric Surrogate Transition.

        Strictly marked [SIMULATED SURROGATE] is_simulated=True.
        """
        p = custom_params or SensitivityParameters.get_regime_params(regime)
        transition_model = ParametricSurrogateTransition(p)
        controller = ClosedLoopController(
            recommender=self.recommender,
            mode=ExecutionMode.SURROGATE_SIMULATION,
            transition_model=transition_model,
        )

        total_downtime = 0.0
        total_parts = 0.0
        total_labor = 0.0
        avoided_count = 0
        missed_count = 0
        unnecessary_count = 0

        current_states = [dict(item["decision_state"]) for item in decision_states_with_context]

        for i, item in enumerate(decision_states_with_context):
            dstate = current_states[i]
            ts = item.get("timestamp", "2026-01-01T00:00:00")
            asset_id = item.get("asset_id", "asset_1")
            gt_ctx = item.get("ground_truth_context", {})
            sev = dstate.get("decision_severity", "WATCH").upper()

            if policy_override == "RTF":
                # Run-To-Failure: Do nothing unless catastrophic failure occurs
                if sev == "CRITICAL":
                    # Breakdown replacement
                    exec_action = {"action_id": "ACT_REPL_OVERHAUL", "action_name": "Emergency Breakdown Overhaul"}
                    dstate["decision_severity"] = "CRITICAL"
                else:
                    exec_action = {"action_id": "ACT_MON_CONTINUE", "action_name": "Continue Operation"}

                post_st, out = transition_model.transition(dstate, exec_action, ground_truth_context=gt_ctx)
                if sev == "CRITICAL":
                    out.downtime_hours = 24.0  # Extended unplanned breakdown downtime
                    out.parts_cost_usd = p.overhaul_cost * 1.3
                    out.labor_hours = 32.0
                    missed_count += 1
                current_states[i] = post_st

            elif policy_override == "REACTIVE":
                # Reactive: Fix only when degrading or critical
                if sev in {"DEGRADING", "CRITICAL"}:
                    exec_action = {"action_id": "ACT_REPL_BEARING", "action_name": "Reactive Bearing Replacement"}
                else:
                    exec_action = {"action_id": "ACT_MON_CONTINUE", "action_name": "Continue Monitoring"}

                post_st, out = transition_model.transition(dstate, exec_action, ground_truth_context=gt_ctx)
                if sev in {"DEGRADING", "CRITICAL"}:
                    out.downtime_hours += 4.0  # Unplanned scheduling penalty
                    avoided_count += 1
                current_states[i] = post_st

            else:
                # Prescriptive AI Policy
                rec = controller.step(
                    decision_state=dstate,
                    timestamp=ts,
                    asset_id=asset_id,
                    ground_truth_context=gt_ctx,
                )
                out = rec.outcome
                current_states[i] = rec.outcome.to_dict()  # update tracked state

            total_downtime += out.downtime_hours
            total_parts += out.parts_cost_usd
            total_labor += out.labor_hours
            if out.prevented_failure:
                avoided_count += 1
            if out.missed_failure:
                missed_count += 1
            if out.unnecessary_intervention:
                unnecessary_count += 1

        labor_cost = total_labor * p.labor_hourly_rate
        downtime_cost = total_downtime * p.hourly_downtime_cost
        failure_penalty = missed_count * p.catastrophic_failure_penalty
        unnecessary_penalty = unnecessary_count * p.unnecessary_intervention_penalty
        total_cost = total_parts + labor_cost + downtime_cost + failure_penalty + unnecessary_penalty

        p_name = policy_override if policy_override else "Phase4_Frozen_Recommender"

        return PolicyEvaluationResult(
            policy_name=p_name,
            execution_mode="Surrogate_Simulation",
            regime=p.regime.value,
            total_steps=len(decision_states_with_context),
            total_downtime_hours=round(total_downtime, 2),
            total_parts_cost_usd=round(total_parts, 2),
            total_labor_hours=round(total_labor, 2),
            total_labor_cost_usd=round(labor_cost, 2),
            downtime_cost_usd=round(downtime_cost, 2),
            failure_penalty_usd=round(failure_penalty, 2),
            unnecessary_penalty_usd=round(unnecessary_penalty, 2),
            total_cost_usd=round(total_cost, 2),
            avoided_failures=avoided_count,
            missed_failures=missed_count,
            unnecessary_interventions=unnecessary_count,
            is_simulated=True,
            notes=f"[SIMULATED SURROGATE] Counterfactual physics surrogate under {p.regime.value} regime.",
        )

    def run_full_sensitivity_suite(
        self,
        decision_states_with_context: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Runs the complete multi-scenario sensitivity analysis across all regimes and policies."""
        results: Dict[str, Any] = {}

        # 1. Historical Replay
        replay_res = self.evaluate_historical_replay(decision_states_with_context)
        results["historical_replay"] = replay_res.to_dict()

        # 2. Surrogate across Regimes: Baseline, Conservative, Optimistic
        surrogate_regimes = {}
        for regime in [CostRegime.BASELINE, CostRegime.CONSERVATIVE, CostRegime.OPTIMISTIC]:
            ai_res = self.evaluate_surrogate_regime(decision_states_with_context, regime=regime)
            rtf_res = self.evaluate_surrogate_regime(decision_states_with_context, regime=regime, policy_override="RTF")
            rec_res = self.evaluate_surrogate_regime(decision_states_with_context, regime=regime, policy_override="REACTIVE")

            # Compute relative savings
            ai_cost = ai_res.total_cost_usd
            rtf_cost = rtf_res.total_cost_usd
            rec_cost = rec_res.total_cost_usd

            ai_res.net_savings_vs_rtf_usd = round(rtf_cost - ai_cost, 2)
            ai_res.savings_pct_vs_rtf = round((rtf_cost - ai_cost) / max(rtf_cost, 1.0) * 100.0, 2)
            ai_res.net_savings_vs_reactive_usd = round(rec_cost - ai_cost, 2)
            ai_res.savings_pct_vs_reactive = round((rec_cost - ai_cost) / max(rec_cost, 1.0) * 100.0, 2)

            surrogate_regimes[regime.value] = {
                "AI_Recommender": ai_res.to_dict(),
                "Run_To_Failure": rtf_res.to_dict(),
                "Reactive": rec_res.to_dict(),
            }

        results["surrogate_regimes"] = surrogate_regimes

        # 3. Downtime Cost Parametric Sweep ($200 to $1,000/hr in $200 increments)
        downtime_sweep = []
        for dt_cost in [200.0, 400.0, 600.0, 800.0, 1000.0]:
            p = SensitivityParameters(hourly_downtime_cost=dt_cost)
            res = self.evaluate_surrogate_regime(decision_states_with_context, custom_params=p)
            rtf = self.evaluate_surrogate_regime(decision_states_with_context, custom_params=p, policy_override="RTF")
            savings = rtf.total_cost_usd - res.total_cost_usd
            pct = (savings / max(rtf.total_cost_usd, 1.0)) * 100.0
            downtime_sweep.append({
                "hourly_downtime_cost_usd": dt_cost,
                "ai_total_cost_usd": res.total_cost_usd,
                "rtf_total_cost_usd": rtf.total_cost_usd,
                "net_savings_usd": round(savings, 2),
                "savings_pct": round(pct, 2),
            })
        results["downtime_cost_sweep"] = downtime_sweep

        return results
