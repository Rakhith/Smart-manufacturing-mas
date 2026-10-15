"""LLM-Assisted Prescriptive Explanation & CMMS Work Order Summarizer.

Strictly follows Phase 5 epistemics:
1. LLM does NOT make maintenance decisions or rank actions.
2. Recommendations are derived exclusively from the frozen LGBM + CBR model.
3. Explanations are strictly grounded in supplied telemetry evidence without hallucinated facts.
4. LLM outputs do not alter or contaminate numeric benchmark metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class GroundedPrescriptiveSummarizer:
    """Generates structured, evidence-grounded technician explanation cards and CMMS tickets."""

    def __init__(self, model_name: str = "deterministic_grounded_template"):
        self.model_name = model_name

    def generate_technician_card(
        self,
        decision_state: Dict[str, Any],
        ranked_actions: List[Dict[str, Any]],
        actual_outcome: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synthesizes a structured, audit-ready technician card grounded in physical evidence."""
        top_action = ranked_actions[0]
        runner_up = ranked_actions[1] if len(ranked_actions) > 1 else None

        sid = decision_state.get("decision_state_id", "UNKNOWN")
        asset_id = decision_state.get("asset_context", {}).get("asset_id", "UNKNOWN")
        arch = decision_state.get("asset_context", {}).get("machine_archetype", "industrial_machine")
        sev = decision_state.get("decision_severity", "WATCH")
        t_stamp = decision_state.get("temporal_context", {}).get("timestamp", "N/A")

        # Collect physical evidence
        evidence_items = []
        inp_state = decision_state.get("input_machine_state", {})
        mech = inp_state.get("mechanical", {})
        if "vibration_rms_mm_s" in mech:
            evidence_items.append(f"Vibration RMS: {mech['vibration_rms_mm_s']:.2f} mm/s")

        therm = inp_state.get("thermal", {})
        if "canonical_temp_degC" in therm:
            evidence_items.append(f"Temperature: {therm['canonical_temp_degC']:.1f} °C")

        fluid = inp_state.get("fluid", {})
        if "canonical_pressure_bar" in fluid:
            evidence_items.append(f"Pressure: {fluid['canonical_pressure_bar']:.2f} bar")

        elec = inp_state.get("electrical", {})
        if "electric_current_A" in elec:
            evidence_items.append(f"Motor Current: {elec['electric_current_A']:.1f} A")

        health = inp_state.get("health_degradation", {})
        if "health_index" in health:
            evidence_items.append(f"Health Index: {health['health_index']:.1f}/100")

        # Trends
        trends = decision_state.get("trend_summary", {})
        trend_strs = [f"{k}: {v:+.3f}" for k, v in trends.items()]

        # Urgency assignment based on severity
        urgency_map = {
            "HEALTHY": "Routine / Planned",
            "WATCH": "Elevated Supervisory Monitoring",
            "DEGRADING": "High Priority (Intervene within 24-48h)",
            "CRITICAL": "Emergency / Immediate Action Required",
        }
        urgency = urgency_map.get(sev.upper(), "Standard")

        # Build grounded rationale
        rationale_parts = [
            f"Asset {asset_id} ({arch}) is operating in {sev} condition at timestamp {t_stamp}.",
            f"Observed physical telemetry indicates: {', '.join(evidence_items)}." if evidence_items else "",
            f"Trailing trend dynamics indicate: {', '.join(trend_strs)}." if trend_strs else "",
            f"Frozen Prescriptive Engine ranked '{top_action['action_name']}' as the optimal intervention "
            f"with hybrid confidence score {top_action['hybrid_score']:.4f} "
            f"(LGBM gain: {top_action['lgbm_score']:.3f}, CBR neighborhood agreement: {top_action['cbr_score']:.3f}).",
        ]
        if runner_up:
            rationale_parts.append(
                f"Alternative secondary candidate: '{runner_up['action_name']}' (confidence: {runner_up['hybrid_score']:.4f})."
            )

        card = {
            "card_id": f"TC-{asset_id}-{sid}",
            "decision_state_id": sid,
            "asset_id": asset_id,
            "timestamp": t_stamp,
            "machine_archetype": arch,
            "detected_severity": sev,
            "operational_urgency": urgency,
            "prescribed_action": {
                "action_id": top_action["action_id"],
                "action_name": top_action["action_name"],
                "action_category": top_action.get("action_category", "monitoring_observation"),
                "hybrid_confidence": round(top_action["hybrid_score"], 4),
                "lgbm_score": round(top_action["lgbm_score"], 4),
                "cbr_score": round(top_action["cbr_score"], 4),
            },
            "alternative_actions": [
                {
                    "rank": a.get("rank", idx + 2),
                    "action_id": a["action_id"],
                    "action_name": a["action_name"],
                    "confidence": round(a["hybrid_score"], 4),
                }
                for idx, a in enumerate(ranked_actions[1:4])
            ],
            "grounded_evidence": evidence_items,
            "trend_dynamics": trends,
            "rationale": " ".join([p for p in rationale_parts if p]),
            "uncertainty_assessment": (
                "High confidence: strong concord between LGBM ranking and historical case retrieval."
                if abs(top_action["norm_lgbm_score"] - top_action["norm_cbr_score"]) < 0.3
                else "Moderate confidence: partial divergence between discriminative tree ranking and historical case base."
            ) if "norm_lgbm_score" in top_action else "Standard confidence bounds.",
            "actual_plant_outcome": actual_outcome,
            "epistemic_provenance": {
                "generator": "GroundedPrescriptiveSummarizer",
                "model": self.model_name,
                "recommender_source": "Phase 4 Frozen LGBMRanker + CBR Hybrid",
            },
        }

        return card

    def format_cmms_work_order(self, card: Dict[str, Any]) -> Dict[str, Any]:
        """Formats the prescribed recommendation as an enterprise SAP/Maximo work order."""
        act = card["prescribed_action"]
        return {
            "WorkOrder": f"WO-{card['asset_id']}-{card['decision_state_id'][-6:]}",
            "Equipment_ID": card["asset_id"],
            "OrderType": "PM02" if "CORR" in act["action_id"] or "REPL" in act["action_id"] else "PM01",
            "Priority": "1-Emergency" if card["detected_severity"] == "CRITICAL" else ("2-High" if card["detected_severity"] == "DEGRADING" else "3-Normal"),
            "WorkOrderDescription": f"PRESCRIBED: {act['action_name']} on {card['machine_archetype']}",
            "OperationDescription": f"Condition: {card['detected_severity']}. Evidence: {'; '.join(card['grounded_evidence'])}. {card['rationale']}",
            "MaintenanceWorkCenter": "MECH_MAINT" if "BEARING" in act["action_id"] or "ALIGN" in act["action_id"] else "ELEC_MAINT",
            "OrderStatus": "CRTD",
            "DispatchedBy": "Smart_Manufacturing_MAS_Phase5",
        }
