"""Action Execution Layer for Closed-Loop Prescriptive Maintenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from phase5.closed_loop.types import ActionPreconditionStatus, RankedRecommendation

SEVERITY_LEVELS = {
    "HEALTHY": 0,
    "WATCH": 1,
    "DEGRADING": 2,
    "CRITICAL": 3,
}


class ActionExecutor:
    """Dispatches and validates maintenance actions against canonical constraints."""

    def __init__(self, ontology_path: Path = Path("outputs/phase3b/action_ontology/action_ontology.json")):
        with open(ontology_path, "r", encoding="utf-8") as f:
            self.ontology: Dict[str, Dict[str, Any]] = json.load(f)

    def validate_preconditions(
        self,
        action_id: str,
        current_severity: str,
        machine_archetype: str = "industrial_machine",
    ) -> Tuple[ActionPreconditionStatus, str]:
        """Verifies whether an action's operational preconditions are satisfied."""
        if action_id not in self.ontology:
            return ActionPreconditionStatus.INCOMPATIBLE_ARCHETYPE, f"Unknown action: {action_id}"

        spec = self.ontology[action_id]
        preconds = spec.get("preconditions", {})
        min_sev_str = preconds.get("min_severity", "HEALTHY").upper()
        max_sev_str = preconds.get("max_severity", "CRITICAL").upper()

        current_lvl = SEVERITY_LEVELS.get(current_severity.upper(), 1)
        min_lvl = SEVERITY_LEVELS.get(min_sev_str, 0)
        max_lvl = SEVERITY_LEVELS.get(max_sev_str, 3)

        if current_lvl < min_lvl:
            return (
                ActionPreconditionStatus.VIOLATED_SEVERITY_TOO_LOW,
                f"Severity {current_severity} is below required minimum {min_sev_str} for {action_id}",
            )
        if current_lvl > max_lvl:
            return (
                ActionPreconditionStatus.VIOLATED_SEVERITY_TOO_HIGH,
                f"Severity {current_severity} exceeds allowable maximum {max_sev_str} for {action_id}",
            )

        app_archs = spec.get("applicable_archetypes", ["all"])
        if "all" not in app_archs and machine_archetype not in app_archs:
            return (
                ActionPreconditionStatus.INCOMPATIBLE_ARCHETYPE,
                f"Machine archetype {machine_archetype} not in applicable list {app_archs}",
            )

        return ActionPreconditionStatus.SATISFIED, "Preconditions satisfied"

    def select_executable_action(
        self,
        ranked_recommendations: List[Dict[str, Any]],
        current_severity: str,
        machine_archetype: str,
        enforce_preconditions: bool = True,
    ) -> Dict[str, Any]:
        """Selects the top-ranked recommendation that satisfies operational preconditions."""
        if not ranked_recommendations:
            raise ValueError("No ranked recommendations provided.")

        if not enforce_preconditions:
            top = dict(ranked_recommendations[0])
            top["precondition_status"] = ActionPreconditionStatus.SATISFIED.value
            return top

        # Iterate in order of rank
        for rec in ranked_recommendations:
            aid = rec["action_id"]
            status, reason = self.validate_preconditions(aid, current_severity, machine_archetype)
            if status == ActionPreconditionStatus.SATISFIED:
                selected = dict(rec)
                selected["precondition_status"] = status.value
                selected["precondition_reason"] = reason
                return selected

        # Fallback to rank 1 if all strictly violated
        fallback = dict(ranked_recommendations[0])
        status, reason = self.validate_preconditions(fallback["action_id"], current_severity, machine_archetype)
        fallback["precondition_status"] = status.value
        fallback["precondition_reason"] = f"Fallback with status: {reason}"
        return fallback

    def build_execution_payload(
        self,
        selected_action: Dict[str, Any],
        asset_id: str,
        timestamp: str,
        severity: str,
    ) -> Dict[str, Any]:
        """Builds dispatch payload with provenance tracking."""
        aid = selected_action["action_id"]
        spec = self.ontology.get(aid, {})
        return {
            "dispatch_id": f"DISPATCH-{asset_id}-{timestamp}",
            "asset_id": asset_id,
            "timestamp": timestamp,
            "action_id": aid,
            "action_name": spec.get("action_name", aid),
            "action_category": spec.get("category", "monitoring_observation"),
            "intervention_risk": spec.get("intervention_risk", "low"),
            "downtime_cost_category": spec.get("operational_downtime_cost", "low"),
            "machine_severity": severity,
            "rank_assigned": selected_action.get("rank", 1),
            "hybrid_confidence": selected_action.get("hybrid_score", 0.0),
            "provenance": {
                "source_ontology": "outputs/phase3b/action_ontology/action_ontology.json",
                "recommender_model": "Phase 4 Frozen Hybrid (LGBM + CBR)",
                "execution_layer": "Phase 5 ActionExecutor",
            },
        }
