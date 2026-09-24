"""Explainability Engine for Prescriptive Maintenance Recommender.

Computes global SHAP / feature importances and generates compact, telemetry-grounded
explanation cards for individual recommendations without reusing LLM reasoning text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class PrescriptiveExplainer:
    """Provides model interpretability and recommendation rationales."""

    def __init__(self, ranker_model: Any, feature_names: List[str]):
        self.ranker = ranker_model
        self.feature_names = feature_names

    def compute_global_shap(
        self,
        X_sample: np.ndarray,
        max_samples: int = 100,
    ) -> Dict[str, Any]:
        """Computes global SHAP values using TreeExplainer, with fallback to gain importance."""
        sample_matrix = X_sample[:max_samples]
        try:
            import shap
            # Use booster directly
            booster = getattr(self.ranker.model, "booster_", self.ranker.model)
            explainer = shap.TreeExplainer(booster)
            shap_values = explainer.shap_values(sample_matrix)

            if isinstance(shap_values, list):
                # Multiclass or ranking list
                shap_matrix = shap_values[0]
            else:
                shap_matrix = shap_values

            mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
            sorted_idx = np.argsort(mean_abs_shap)[::-1]

            top_shap = {}
            for idx in sorted_idx:
                fname = self.feature_names[idx]
                top_shap[fname] = float(mean_abs_shap[idx])

            return {
                "method": "shap_tree_explainer",
                "status": "success",
                "top_features_by_mean_abs_shap": dict(list(top_shap.items())[:25]),
            }
        except Exception as e:
            # Documented native feature importance fallback
            native_gain = self.ranker.get_feature_importance()
            return {
                "method": "native_lgbm_gain_importance_fallback",
                "status": f"fallback_due_to: {str(e)[:150]}",
                "top_features_by_gain": dict(list(native_gain.items())[:25]),
            }

    def generate_recommendation_card(
        self,
        state_id: str,
        state_metadata: Dict[str, Any],
        candidate_rows: pd.DataFrame,
        predicted_scores: np.ndarray,
        cbr_neighbors: Optional[List[Dict[str, Any]]] = None,
        top_k_features: int = 4,
    ) -> Dict[str, Any]:
        """Generates an interpretable card explaining why the top action was chosen."""
        # Order candidates by predicted score
        order = np.argsort(predicted_scores)[::-1]
        top_idx = order[0]
        top_row = candidate_rows.iloc[top_idx]
        top_action = top_row["action_id"]
        top_action_name = top_row.get("action_name", top_action)
        top_score = float(predicted_scores[top_idx])

        # State Telemetry Highlights
        sev = state_metadata.get("decision_severity", "UNKNOWN")
        arch = state_metadata.get("machine_archetype", "UNKNOWN")
        ds = state_metadata.get("dataset_id", "UNKNOWN")

        # Key state telemetry triggers
        telemetry_evidence = []
        telemetry_evidence.append(f"Calibrated severity tier: {sev}")
        telemetry_evidence.append(f"Asset archetype: {arch} ({ds})")
        if state_metadata.get("cycle"):
            telemetry_evidence.append(f"Cycle count: {state_metadata['cycle']}")

        # Top feature importance cues from action interaction
        action_cues = []
        action_cues.append(f"Action category: {top_row.get('action_category')}")
        action_cues.append(f"Intervention risk tier: {top_row.get('intervention_risk')}")
        action_cues.append(f"Operational downtime cost: {top_row.get('operational_downtime_cost')}")

        card = {
            "decision_state_id": state_id,
            "recommended_action_id": top_action,
            "recommended_action_name": top_action_name,
            "recommendation_score": top_score,
            "state_context": {
                "dataset_id": ds,
                "machine_archetype": arch,
                "decision_severity": sev,
            },
            "telemetry_evidence": telemetry_evidence,
            "action_rationale_cues": action_cues,
            "candidate_ranking": [
                {
                    "rank": rank + 1,
                    "action_id": candidate_rows.iloc[idx]["action_id"],
                    "action_name": candidate_rows.iloc[idx].get("action_name"),
                    "score": float(predicted_scores[idx]),
                }
                for rank, idx in enumerate(order)
            ],
            "cbr_similar_historical_cases": cbr_neighbors[:3] if cbr_neighbors else [],
        }
        return card

    def save_sample_cards(
        self,
        cards: List[Dict[str, Any]],
        output_dir: Path,
    ) -> None:
        """Saves JSON and Markdown versions of sample explanation cards."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        with open(output_dir / "sample_explanation_cards.json", "w", encoding="utf-8") as f:
            json.dump(cards, f, indent=2)

        # Markdown
        md_lines = ["# Prescriptive Maintenance Recommendation Explanation Cards\n"]
        for c in cards:
            md_lines.append(f"## State `{c['decision_state_id']}`")
            md_lines.append(f"- **Recommended Action**: **`{c['recommended_action_id']}`** ({c['recommended_action_name']})")
            md_lines.append(f"- **Learned Recommender Score**: `{c['recommendation_score']:.4f}`")
            md_lines.append(f"- **Asset & Severity**: `{c['state_context']['machine_archetype']}` | Severity: **{c['state_context']['decision_severity']}**")
            md_lines.append("### Candidate Ranked Order:")
            for item in c["candidate_ranking"]:
                md_lines.append(f"  {item['rank']}. `{item['action_id']}` (Score: {item['score']:.4f})")

            if c.get("cbr_similar_historical_cases"):
                md_lines.append("### Case-Based Reasoning Precedents:")
                for neighbor in c["cbr_similar_historical_cases"]:
                    md_lines.append(
                        f"  - `{neighbor.get('neighbor_state_id')}` ({neighbor.get('dataset_id')}, "
                        f"Sim: {neighbor.get('similarity_score', 0.0):.3f}) -> Historical Best: `{neighbor.get('historical_best_action')}`"
                    )
            md_lines.append("\n---\n")

        with open(output_dir / "sample_explanation_cards.md", "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
