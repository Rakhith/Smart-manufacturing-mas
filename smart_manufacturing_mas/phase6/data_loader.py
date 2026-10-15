"""Data loading and unified instance representation for Phase 6 Adaptation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from phase4.feature_pipeline import FeaturePipeline
from phase5.action_mapper import CANONICAL_ACTIONS
from phase5.adapters import FactorySensorsAdapter, Oil040Adapter
from phase5.component_failure_evaluator import ComponentFailureEvaluator
from phase5.hierarchical_evaluation import ACTION_TO_FAMILY


ACTION_TO_CATEGORY = {
    "ACT_MON_CONTINUE": "monitoring_observation",
    "ACT_MON_ENHANCED": "monitoring_observation",
    "ACT_MON_PARAMETER_LOG": "monitoring_observation",
    "ACT_INSP_VIBRATION": "inspection_diagnosis",
    "ACT_INSP_LUBRICATION": "inspection_diagnosis",
    "ACT_INSP_THERMAL_ELECTRICAL": "inspection_diagnosis",
    "ACT_INSP_TOOL_WEAR": "inspection_diagnosis",
    "ACT_INSP_PRESSURE_HYDRAULIC": "inspection_diagnosis",
    "ACT_INSP_SUBSYSTEM": "inspection_diagnosis",
    "ACT_CORR_LUBRICATE": "corrective_maintenance",
    "ACT_CORR_CLEAN_PURGE": "corrective_maintenance",
    "ACT_CORR_CALIBRATE": "corrective_maintenance",
    "ACT_CORR_FASTEN_ALIGN": "corrective_maintenance",
    "ACT_REPL_TOOL_INSERT": "replacement_overhaul",
    "ACT_REPL_BEARING": "replacement_overhaul",
    "ACT_REPL_SEAL_VALVE": "replacement_overhaul",
    "ACT_REPL_OVERHAUL": "replacement_overhaul",
    "ACT_OP_ADJUST_PARAMS": "operational_mitigation",
    "ACT_OP_DERATE_LOAD": "operational_mitigation",
    "ACT_OP_CONTROLLED_SHUTDOWN": "operational_mitigation",
    "ACT_OP_EMERGENCY_STOP": "operational_mitigation",
}


@dataclass
class AdaptationInstance:
    """Standardized evaluation and adaptation instance."""
    instance_id: str
    domain_id: str
    asset_id: str
    event_timestamp: str
    prediction_timestamp: str
    target_action: str
    target_component: str
    target_category: str
    decision_state: Dict[str, Any]
    candidate_actions: List[Dict[str, Any]]
    X_features: np.ndarray  # Shape: (num_candidates, feature_dim)
    relevance_grades: np.ndarray  # Shape: (num_candidates,) in [0, 4]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "domain_id": self.domain_id,
            "asset_id": self.asset_id,
            "event_timestamp": self.event_timestamp,
            "prediction_timestamp": self.prediction_timestamp,
            "target_action": self.target_action,
            "target_component": self.target_component,
            "target_category": self.target_category,
            "num_candidates": len(self.candidate_actions),
            "metadata": self.metadata,
        }


class Phase6DataLoader:
    """Unified loader providing preprocessed and cached instances across domains."""

    def __init__(
        self,
        corpus_path: Path = Path("outputs/phase3a/selected_decision_states/decision_states_corpus.json"),
        ontology_path: Path = Path("outputs/phase3b/action_ontology/action_ontology.json"),
        training_table_path: Path = Path("outputs/phase4/training/phase4_training_table.parquet"),
    ):
        self.corpus_path = Path(corpus_path)
        self.ontology_path = Path(ontology_path)
        self.training_table_path = Path(training_table_path)

        with open(self.ontology_path, "r", encoding="utf-8") as f:
            self.action_ontology: Dict[str, Dict[str, Any]] = json.load(f)

        self.pipeline = FeaturePipeline(corpus_path=self.corpus_path)
        self.canonical_candidates = self._build_canonical_candidates()

    def _build_canonical_candidates(self) -> List[Dict[str, Any]]:
        candidates = []
        for aid, spec in self.action_ontology.items():
            preconds = spec.get("preconditions", {})
            candidates.append({
                "action_id": aid,
                "action_name": spec.get("action_name", aid),
                "action_category": spec.get("category", "monitoring_observation"),
                "intervention_risk": spec.get("intervention_risk", "low"),
                "operational_downtime_cost": spec.get("operational_downtime_cost", "low"),
                "precondition_min_severity": preconds.get("min_severity", "HEALTHY"),
                "precondition_max_severity": preconds.get("max_severity", "CRITICAL"),
                "description": spec.get("description", ""),
            })
        return candidates

    def _compute_relevance_grades(
        self,
        target_action: str,
        target_component: str,
        candidate_actions: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Assigns graded relevance in [0, 4] for candidate actions."""
        target_cat = ACTION_TO_CATEGORY.get(target_action, "unknown")
        target_fam = ACTION_TO_FAMILY.get(target_action, "unknown")

        grades = []
        for cand in candidate_actions:
            aid = cand["action_id"]
            if aid == target_action:
                grades.append(4)  # Exact action match
            else:
                c_comp = ComponentFailureEvaluator.get_action_component(aid)
                c_cat = ACTION_TO_CATEGORY.get(aid, "unknown")
                c_fam = ACTION_TO_FAMILY.get(aid, "unknown")

                if c_comp == target_component and target_component != "general":
                    grades.append(2)  # Correct physical component subsystem
                elif c_cat == target_cat:
                    grades.append(1)  # Same action category
                elif c_fam == target_fam:
                    grades.append(1)  # Same action family
                else:
                    grades.append(0)

        return np.array(grades, dtype=np.int32)

    def _extract_candidate_features(
        self,
        decision_state: Dict[str, Any],
        candidate_actions: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Transforms candidate actions for a decision state into 96-dim feature vectors."""
        sid = decision_state.get("decision_state_id", "QUERY_STATE")
        rows = []
        for cand in candidate_actions:
            row_dict = {
                "decision_state_id": sid,
                "decision_severity": decision_state.get("decision_severity", "WATCH"),
                "machine_archetype": decision_state.get("asset_context", {}).get("machine_archetype", "industrial_machine"),
                "temporal_type": decision_state.get("asset_context", {}).get("temporal_type", "time_series"),
                "cycle": decision_state.get("temporal_context", {}).get("cycle", 0),
                "sequence_step": decision_state.get("temporal_context", {}).get("sequence_step", 0),
                "action_id": cand["action_id"],
                "action_name": cand.get("action_name", cand["action_id"]),
                "action_category": cand.get("action_category", "monitoring_observation"),
                "intervention_risk": cand.get("intervention_risk", "low"),
                "operational_downtime_cost": cand.get("operational_downtime_cost", "low"),
                "precondition_min_severity": cand.get("precondition_min_severity", "HEALTHY"),
                "precondition_max_severity": cand.get("precondition_max_severity", "CRITICAL"),
            }
            rows.append(row_dict)

        df_query = pd.DataFrame(rows)
        self.pipeline.state_details[sid] = decision_state
        X_query, _ = self.pipeline.transform_dataframe(df_query, feature_subset="full_semantic_trend")
        self.pipeline.state_details.pop(sid, None)
        return X_query

    def load_factory_sensors(self) -> List[AdaptationInstance]:
        """Loads Factory Sensors centrifugal pump benchmark (63 work orders)."""
        adapter = FactorySensorsAdapter()
        adapter.load()
        asset_ids = adapter.get_asset_ids()

        raw_records = []
        for aid in asset_ids:
            gt = adapter.get_ground_truth_events(aid)
            tel = adapter.get_temporal_sequence(aid)
            for wo in gt.get("work_orders", []):
                wo_dt = pd.to_datetime(wo["opened_at"])
                tgt_act = wo.get("canonical_action_id")
                if not tgt_act:
                    continue

                avail = tel[tel["recorded_at"] < wo_dt]
                if avail.empty:
                    continue
                snapshot = avail.iloc[-1]
                pred_dt = snapshot["recorded_at"]

                raw_records.append({
                    "asset_id": str(aid),
                    "event_timestamp": str(wo_dt),
                    "dt": wo_dt,
                    "prediction_timestamp": str(pred_dt),
                    "target_action": tgt_act,
                    "wo": wo,
                })

        raw_records.sort(key=lambda x: x["dt"])

        instances: List[AdaptationInstance] = []
        for i, rec in enumerate(raw_records):
            aid = rec["asset_id"]
            dstate = adapter.build_decision_state_at(aid, rec["prediction_timestamp"])
            tgt_act = rec["target_action"]
            tgt_comp = ComponentFailureEvaluator.get_action_component(tgt_act)
            tgt_cat = ACTION_TO_CATEGORY.get(tgt_act, "unknown")

            X_feat = self._extract_candidate_features(dstate, self.canonical_candidates)
            y_grades = self._compute_relevance_grades(tgt_act, tgt_comp, self.canonical_candidates)

            instances.append(
                AdaptationInstance(
                    instance_id=f"FS_INST_{i+1:03d}_{aid}",
                    domain_id="factory_sensors",
                    asset_id=aid,
                    event_timestamp=rec["event_timestamp"],
                    prediction_timestamp=rec["prediction_timestamp"],
                    target_action=tgt_act,
                    target_component=tgt_comp,
                    target_category=tgt_cat,
                    decision_state=dstate,
                    candidate_actions=self.canonical_candidates,
                    X_features=X_feat,
                    relevance_grades=y_grades,
                    metadata={
                        "work_order_id": rec["wo"].get("work_order_id"),
                        "parts_cost": float(rec["wo"].get("parts_cost", 0.0)),
                        "labor_hours": float(rec["wo"].get("labor_hours", 0.0)),
                        "order_type": str(rec["wo"].get("order_type", "")),
                        "decision_severity": dstate.get("decision_severity", "WATCH"),
                    },
                )
            )

        return instances

    def load_oil040(self) -> List[AdaptationInstance]:
        """Loads OIL-040 rotating machinery benchmark (80 work orders)."""
        adapter = Oil040Adapter()
        adapter.load()
        asset_ids = adapter.get_asset_ids()

        raw_records = []
        for aid in asset_ids:
            gt = adapter.get_ground_truth_events(aid)
            tel = adapter.get_temporal_sequence(aid)
            for wo in gt.get("work_orders", []):
                wo_dt = pd.to_datetime(wo["timestamp"])
                tgt_act = wo.get("canonical_action_id")
                if not tgt_act:
                    continue

                avail = tel[tel["timestamp"] < wo_dt]
                if avail.empty:
                    continue
                snapshot = avail.iloc[-1]
                pred_dt = snapshot["timestamp"]

                raw_records.append({
                    "asset_id": str(aid),
                    "event_timestamp": str(wo_dt),
                    "dt": wo_dt,
                    "prediction_timestamp": str(pred_dt),
                    "target_action": tgt_act,
                    "wo": wo,
                })

        raw_records.sort(key=lambda x: x["dt"])

        instances: List[AdaptationInstance] = []
        for i, rec in enumerate(raw_records):
            aid = rec["asset_id"]
            dstate = adapter.build_decision_state_at(aid, rec["prediction_timestamp"])
            tgt_act = rec["target_action"]
            tgt_comp = ComponentFailureEvaluator.get_action_component(tgt_act)
            tgt_cat = ACTION_TO_CATEGORY.get(tgt_act, "unknown")

            X_feat = self._extract_candidate_features(dstate, self.canonical_candidates)
            y_grades = self._compute_relevance_grades(tgt_act, tgt_comp, self.canonical_candidates)

            instances.append(
                AdaptationInstance(
                    instance_id=f"OIL_INST_{i+1:03d}_{aid}",
                    domain_id="oil040",
                    asset_id=aid,
                    event_timestamp=rec["event_timestamp"],
                    prediction_timestamp=rec["prediction_timestamp"],
                    target_action=tgt_act,
                    target_component=tgt_comp,
                    target_category=tgt_cat,
                    decision_state=dstate,
                    candidate_actions=self.canonical_candidates,
                    X_features=X_feat,
                    relevance_grades=y_grades,
                    metadata={
                        "workorder_id": rec["wo"].get("workorder_id"),
                        "downtime_hours": float(rec["wo"].get("downtime_hours", 0.0)),
                        "priority": str(rec["wo"].get("priority", "")),
                        "decision_severity": dstate.get("decision_severity", "WATCH"),
                    },
                )
            )

        return instances

    def load_cmapss_lodo(self) -> Tuple[List[AdaptationInstance], pd.DataFrame]:
        """Loads C-MAPSS turbofan engine instances for strict Leave-One-Domain-Out benchmark.
        
        Returns:
            cmapss_instances: List of AdaptationInstance for C-MAPSS.
            source_lodo_df: Training table excluding C-MAPSS (for training LODO source model).
        """
        df = pd.read_parquet(self.training_table_path)
        source_lodo_df = df[df["dataset_id"] != "cmapss"].copy()
        cmapss_df = df[df["dataset_id"] == "cmapss"].copy()

        # Load decision state details from corpus
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        corpus_map = {s["decision_state_id"]: s for s in corpus if s.get("dataset_id") == "cmapss"}

        # Group by decision_state_id
        instances: List[AdaptationInstance] = []
        state_groups = list(cmapss_df.groupby("decision_state_id", sort=False))

        # Sort states by cycle/sequence_step to maintain monotonic chronological progression
        def get_sort_key(grp_tuple):
            sid, g = grp_tuple
            c = g["cycle"].iloc[0] if "cycle" in g.columns and not pd.isna(g["cycle"].iloc[0]) else 0
            s = g["sequence_step"].iloc[0] if "sequence_step" in g.columns and not pd.isna(g["sequence_step"].iloc[0]) else 0
            aid = str(g["asset_id"].iloc[0]) if "asset_id" in g.columns else ""
            return (int(c), int(s), aid)

        state_groups.sort(key=get_sort_key)

        for i, (sid, grp) in enumerate(state_groups):
            dstate = corpus_map.get(sid, {})
            aid_asset = str(grp["asset_id"].iloc[0])
            cycle = int(grp["cycle"].iloc[0]) if "cycle" in grp.columns else 0

            # Target action is the highest relevance silver action
            sorted_grp = grp.sort_values(by="ranking_relevance", ascending=False)
            tgt_act = str(sorted_grp.iloc[0]["action_id"])
            tgt_comp = ComponentFailureEvaluator.get_action_component(tgt_act)
            tgt_cat = ACTION_TO_CATEGORY.get(tgt_act, "unknown")

            # Candidate actions present in the evaluation
            cand_actions = []
            for _, r in grp.iterrows():
                cand_actions.append({
                    "action_id": r["action_id"],
                    "action_name": r.get("action_name", r["action_id"]),
                    "action_category": r.get("action_category", "unknown"),
                    "intervention_risk": r.get("intervention_risk", "low"),
                    "operational_downtime_cost": r.get("operational_downtime_cost", "low"),
                    "precondition_min_severity": r.get("precondition_min_severity", "HEALTHY"),
                    "precondition_max_severity": r.get("precondition_max_severity", "CRITICAL"),
                })

            # Feature extraction for candidates in this state
            self.pipeline.state_details[sid] = dstate
            X_feat, _ = self.pipeline.transform_dataframe(grp, feature_subset="full_semantic_trend")
            self.pipeline.state_details.pop(sid, None)

            y_grades = grp["ranking_relevance"].values.astype(np.int32)

            instances.append(
                AdaptationInstance(
                    instance_id=f"CMAPSS_INST_{i+1:03d}_{aid_asset}",
                    domain_id="cmapss_lodo",
                    asset_id=aid_asset,
                    event_timestamp=f"cycle_{cycle:04d}",
                    prediction_timestamp=f"cycle_{cycle:04d}",
                    target_action=tgt_act,
                    target_component=tgt_comp,
                    target_category=tgt_cat,
                    decision_state=dstate,
                    candidate_actions=cand_actions,
                    X_features=X_feat,
                    relevance_grades=y_grades,
                    metadata={
                        "decision_state_id": sid,
                        "cycle": cycle,
                        "decision_severity": dstate.get("decision_severity", "WATCH"),
                        "silver_rank_top1": tgt_act,
                    },
                )
            )

        return instances, source_lodo_df
