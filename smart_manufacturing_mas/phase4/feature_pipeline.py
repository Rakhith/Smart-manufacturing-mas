"""Feature Engineering Pipeline for Phase 4 Recommender.

Constructs compact, cross-domain, semantic structured feature vectors for
DecisionState-CandidateAction pairs. Excludes all target outcomes and LLM reasoning.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd


MODALITIES = [
    "thermal",
    "mechanical",
    "kinematic",
    "fluid",
    "electrical",
    "acoustic",
    "process_operating",
    "health_degradation",
    "unmapped",
]

SEVERITY_MAP = {
    "HEALTHY": 0,
    "WATCH": 1,
    "DEGRADING": 2,
    "CRITICAL": 3,
    "UNCLASSIFIED": 1,
}

RISK_MAP = {
    "negligible": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

COST_MAP = {
    "negligible": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}

ACTION_CATEGORIES = [
    "monitoring_observation",
    "inspection_diagnosis",
    "corrective_maintenance",
    "replacement_overhaul",
    "operational_mitigation",
]

ARCHETYPES = [
    "cnc_mill",
    "turbofan_engine",
    "industrial_machine",
    "iiot_machine",
    "etch_tool",
    "air_compressor",
    "hydraulic_test_rig",
    "chemical_plant",
    "bearing_test_rig",
]


class FeaturePipeline:
    """Extracts structured feature representations from joined state-action records."""

    def __init__(self, corpus_path: Optional[Path] = None):
        self.state_details: Dict[str, Dict[str, Any]] = {}
        if corpus_path and Path(corpus_path).exists():
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            self.state_details = {s["decision_state_id"]: s for s in corpus}

        self.feature_names: List[str] = []
        self.feature_groups: Dict[str, List[str]] = {}

    def extract_state_features(self, state_dict: Dict[str, Any]) -> Dict[str, float]:
        """Extracts semantic features for a single DecisionState."""
        feat: Dict[str, float] = {}

        # 1. Severity & Context
        sev_str = str(state_dict.get("decision_severity", "WATCH")).upper()
        sev_num = SEVERITY_MAP.get(sev_str, 1)
        feat["state__severity_numeric"] = float(sev_num)

        temp_ctx = state_dict.get("temporal_context", {})
        feat["state__cycle"] = float(temp_ctx.get("cycle") or 0.0)
        feat["state__sequence_step"] = float(temp_ctx.get("sequence_step") or 0.0)
        feat["state__is_time_series"] = 1.0 if state_dict.get("asset_context", {}).get("temporal_type") == "time_series" else 0.0

        # Archetype one-hot
        arch = state_dict.get("asset_context", {}).get("machine_archetype", "unknown")
        for a in ARCHETYPES:
            feat[f"state__arch_{a}"] = 1.0 if arch == a else 0.0

        def _clean_numeric(vals):
            cleaned = []
            for v in vals:
                if isinstance(v, (int, float)):
                    try:
                        f = float(v)
                        if np.isfinite(f):
                            cleaned.append(float(np.clip(f, -1e6, 1e6)))
                    except (ValueError, OverflowError):
                        pass
            return cleaned

        # 2. Modality aggregations
        input_state = state_dict.get("input_machine_state", {})
        for mod in MODALITIES:
            m_dict = input_state.get(mod, {})
            numeric_vals = _clean_numeric(m_dict.values())
            feat[f"state__has_{mod}"] = 1.0 if len(numeric_vals) > 0 else 0.0
            feat[f"state__count_{mod}"] = float(len(numeric_vals))
            feat[f"state__mean_{mod}"] = float(np.mean(numeric_vals)) if numeric_vals else 0.0
            feat[f"state__std_{mod}"] = float(np.std(numeric_vals)) if len(numeric_vals) > 1 else 0.0
            feat[f"state__max_{mod}"] = float(np.max(numeric_vals)) if numeric_vals else 0.0
            feat[f"state__min_{mod}"] = float(np.min(numeric_vals)) if numeric_vals else 0.0

        # 3. Canonical physical features
        thermal_dict = input_state.get("thermal", {})
        canon_temp = _clean_numeric([v for k, v in thermal_dict.items() if "canonical_degC" in k])
        feat["state__canonical_temp_degC"] = float(np.mean(canon_temp)) if canon_temp else 0.0

        mech_dict = input_state.get("mechanical", {})
        torque_vals = _clean_numeric([v for k, v in mech_dict.items() if "Torque" in k])
        feat["state__canonical_torque_Nm"] = float(np.mean(torque_vals)) if torque_vals else 0.0

        kin_dict = input_state.get("kinematic", {})
        speed_vals = _clean_numeric([v for k, v in kin_dict.items() if "speed" in k.lower()])
        feat["state__canonical_speed_rpm"] = float(np.mean(speed_vals)) if speed_vals else 0.0

        fluid_dict = input_state.get("fluid", {})
        pressure_vals = _clean_numeric([v for k, v in fluid_dict.items() if "pressure" in k.lower()])
        feat["state__canonical_pressure_bar"] = float(np.mean(pressure_vals)) if pressure_vals else 0.0

        health_dict = input_state.get("health_degradation", {})
        wear_vals = _clean_numeric([v for k, v in health_dict.items() if "wear" in k.lower()])
        feat["state__canonical_tool_wear_min"] = float(np.mean(wear_vals)) if wear_vals else 0.0

        # 4. Trend summary
        trends = state_dict.get("trend_summary", {})
        t_vals = _clean_numeric(trends.values())
        feat["state__has_trends"] = 1.0 if len(t_vals) > 0 else 0.0
        feat["state__trend_count"] = float(len(t_vals))
        feat["state__trend_mean"] = float(np.mean(t_vals)) if t_vals else 0.0
        feat["state__trend_std"] = float(np.std(t_vals)) if len(t_vals) > 1 else 0.0
        feat["state__trend_max"] = float(np.max(t_vals)) if t_vals else 0.0
        feat["state__trend_min"] = float(np.min(t_vals)) if t_vals else 0.0

        return feat

    def extract_action_features(self, row: pd.Series) -> Dict[str, float]:
        """Extracts features for the candidate action."""
        feat: Dict[str, float] = {}

        # Action Category One-Hot
        cat = str(row.get("action_category", "unknown"))
        for c in ACTION_CATEGORIES:
            feat[f"act__cat_{c}"] = 1.0 if cat == c else 0.0

        # Action Risk & Downtime Cost
        risk_str = str(row.get("intervention_risk", "low")).lower()
        cost_str = str(row.get("operational_downtime_cost", "low")).lower()
        feat["act__risk_numeric"] = float(RISK_MAP.get(risk_str, 1))
        feat["act__cost_numeric"] = float(COST_MAP.get(cost_str, 1))

        # Precondition bounds
        min_sev = SEVERITY_MAP.get(str(row.get("precondition_min_severity", "HEALTHY")).upper(), 0)
        max_sev = SEVERITY_MAP.get(str(row.get("precondition_max_severity", "CRITICAL")).upper(), 3)
        feat["act__precond_min_sev"] = float(min_sev)
        feat["act__precond_max_sev"] = float(max_sev)

        return feat

    def extract_interaction_features(
        self, state_feats: Dict[str, float], act_feats: Dict[str, float]
    ) -> Dict[str, float]:
        """Extracts interaction features between state and candidate action."""
        feat: Dict[str, float] = {}

        sev = state_feats.get("state__severity_numeric", 1.0)
        risk = act_feats.get("act__risk_numeric", 1.0)
        cost = act_feats.get("act__cost_numeric", 1.0)
        min_sev = act_feats.get("act__precond_min_sev", 0.0)
        max_sev = act_feats.get("act__precond_max_sev", 3.0)

        # Severity-Risk and Severity-Cost alignment
        feat["inter__sev_x_risk"] = sev * risk
        feat["inter__sev_x_cost"] = sev * cost
        feat["inter__admissible"] = 1.0 if (min_sev <= sev <= max_sev) else 0.0

        # Critical severity interactions
        is_crit = 1.0 if sev >= 3.0 else 0.0
        is_healthy = 1.0 if sev <= 0.0 else 0.0
        is_mitigation = act_feats.get("act__cat_operational_mitigation", 0.0)
        is_replacement = act_feats.get("act__cat_replacement_overhaul", 0.0)
        is_monitoring = act_feats.get("act__cat_monitoring_observation", 0.0)

        feat["inter__crit_x_mitigation"] = is_crit * is_mitigation
        feat["inter__crit_x_replacement"] = is_crit * is_replacement
        feat["inter__healthy_x_monitoring"] = is_healthy * is_monitoring

        # Modality alignment
        has_mech = state_feats.get("state__has_mechanical", 0.0)
        has_fluid = state_feats.get("state__has_fluid", 0.0)
        has_thermal = state_feats.get("state__has_thermal", 0.0)
        is_inspection = act_feats.get("act__cat_inspection_diagnosis", 0.0)

        feat["inter__mech_x_inspection"] = has_mech * is_inspection
        feat["inter__fluid_x_inspection"] = has_fluid * is_inspection
        feat["inter__thermal_x_inspection"] = has_thermal * is_inspection

        return feat

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        feature_subset: str = "full_semantic_trend",
    ) -> Tuple[np.ndarray, List[str]]:
        """Transforms a DataFrame of state-action rows into a feature matrix X."""
        rows = []
        for _, row in df.iterrows():
            sid = row["decision_state_id"]
            state_dict = self.state_details.get(sid, {
                "decision_severity": row.get("decision_severity", "WATCH"),
                "asset_context": {
                    "machine_archetype": row.get("machine_archetype", "unknown"),
                    "temporal_type": row.get("temporal_type", "unknown"),
                },
                "temporal_context": {
                    "cycle": row.get("cycle"),
                    "sequence_step": row.get("sequence_step"),
                },
                "input_machine_state": {},
                "trend_summary": {},
            })

            s_feats = self.extract_state_features(state_dict)
            a_feats = self.extract_action_features(row)
            i_feats = self.extract_interaction_features(s_feats, a_feats)

            combined = {}
            if feature_subset == "raw_structured":
                # Subset A: Basic context & action properties only
                for k, v in s_feats.items():
                    if "count_" in k or "severity" in k or "arch_" in k:
                        combined[k] = v
                combined.update(a_feats)
                combined["inter__sev_x_risk"] = i_feats["inter__sev_x_risk"]
                combined["inter__admissible"] = i_feats["inter__admissible"]

            elif feature_subset == "canonical":
                # Subset B: Semantic modalities + canonical physical measures (no trend dynamics)
                for k, v in s_feats.items():
                    if "trend" not in k and "sequence" not in k and "cycle" not in k:
                        combined[k] = v
                combined.update(a_feats)
                combined.update(i_feats)

            elif feature_subset == "full_semantic_trend":
                # Subset C: Full semantic modalities + trends + interactions
                combined.update(s_feats)
                combined.update(a_feats)
                combined.update(i_feats)
            else:
                raise ValueError(f"Unknown feature subset: {feature_subset}")

            rows.append(combined)

        feat_df = pd.DataFrame(rows).fillna(0.0).clip(-1e6, 1e6)
        feature_names = feat_df.columns.tolist()
        self.feature_names = feature_names

        # Group documentation
        self.feature_groups = {
            "state_severity_context": [c for c in feature_names if c.startswith("state__") and ("sev" in c or "arch" in c or "cycle" in c or "step" in c)],
            "state_modalities": [c for c in feature_names if c.startswith("state__") and any(f"_{m}" in c for m in MODALITIES)],
            "state_canonical": [c for c in feature_names if "canonical_" in c],
            "state_trends": [c for c in feature_names if "trend" in c],
            "action_properties": [c for c in feature_names if c.startswith("act__")],
            "state_action_interactions": [c for c in feature_names if c.startswith("inter__")],
        }

        X = feat_df.to_numpy(dtype=np.float32)
        return X, feature_names

    def get_state_vector(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """Extracts a normalized state-only representation for CBR similarity retrieval."""
        s_feats = self.extract_state_features(state_dict)
        # Sort keys deterministically
        sorted_keys = sorted(s_feats.keys())
        raw_vals = [float(np.clip(np.nan_to_num(s_feats[k], nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)) for k in sorted_keys]
        return np.array(raw_vals, dtype=np.float32)

    def save_feature_schema(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = {
            "total_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "feature_groups": self.feature_groups,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
