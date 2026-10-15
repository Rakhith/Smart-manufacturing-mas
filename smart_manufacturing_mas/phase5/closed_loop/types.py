"""Core data structures and types for Closed-Loop Prescriptive Maintenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExecutionMode(str, Enum):
    HISTORICAL_REPLAY = "historical_replay"
    SURROGATE_SIMULATION = "surrogate_simulation"


class ActionPreconditionStatus(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED_SEVERITY_TOO_LOW = "violated_severity_too_low"
    VIOLATED_SEVERITY_TOO_HIGH = "violated_severity_too_high"
    INCOMPATIBLE_ARCHETYPE = "incompatible_archetype"


@dataclass
class RankedRecommendation:
    action_id: str
    action_name: str
    action_category: str
    rank: int
    hybrid_score: float
    lgbm_score: float
    cbr_score: float
    description: str = ""
    precondition_status: ActionPreconditionStatus = ActionPreconditionStatus.SATISFIED

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MaintenanceOutcome:
    action_id: str
    canonical_action_id: str
    success: bool
    downtime_hours: float = 0.0
    parts_cost_usd: float = 0.0
    labor_hours: float = 0.0
    prevented_failure: bool = False
    missed_failure: bool = False
    unnecessary_intervention: bool = False
    lead_time_hours: float = 0.0
    health_index_delta: float = 0.0
    notes: str = ""
    is_simulated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TransitionRecord:
    step: int
    timestamp: str
    asset_id: str
    pre_severity: str
    pre_health_index: float
    recommended_action_id: str
    recommended_action_name: str
    executed_action_id: str
    outcome: MaintenanceOutcome
    post_severity: str
    post_health_index: float
    execution_mode: ExecutionMode
    ground_truth_matched: bool = False
    ground_truth_work_order: Optional[Dict[str, Any]] = None
    ground_truth_failure: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["execution_mode"] = self.execution_mode.value
        return res
