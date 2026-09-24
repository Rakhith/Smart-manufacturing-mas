"""Dataset Builder for Phase 4 Prescriptive Maintenance Recommender.

Constructs a leakage-safe tabular dataset linking Phase 3A DecisionStates with
Phase 3B LLM silver-standard action preference evaluations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


class Phase4DatasetBuilder:
    """Extracts, joins, and validates Phase 3A states and Phase 3B evaluations into training tables."""

    def __init__(
        self,
        phase3a_corpus_path: Path,
        phase3b_judgments_path: Path,
        phase3b_ontology_path: Path,
        phase3b_pref_path: Optional[Path] = None,
    ):
        self.phase3a_corpus_path = Path(phase3a_corpus_path)
        self.phase3b_judgments_path = Path(phase3b_judgments_path)
        self.phase3b_ontology_path = Path(phase3b_ontology_path)
        self.phase3b_pref_path = Path(phase3b_pref_path) if phase3b_pref_path else None

    def build_training_table(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Constructs the master training DataFrame (one row per candidate action)."""
        # 1. Load Phase 3A corpus
        with open(self.phase3a_corpus_path, "r", encoding="utf-8") as f:
            corpus: List[Dict[str, Any]] = json.load(f)

        states_by_id = {s["decision_state_id"]: s for s in corpus}

        # 2. Load Phase 3B parsed judgments
        with open(self.phase3b_judgments_path, "r", encoding="utf-8") as f:
            judgments: Dict[str, Any] = json.load(f)

        # 3. Load Action Ontology
        with open(self.phase3b_ontology_path, "r", encoding="utf-8") as f:
            ontology: Dict[str, Any] = json.load(f)

        # 4. Optional: Load preference dataset for provenance cross-check
        prov_map = {}
        if self.phase3b_pref_path and self.phase3b_pref_path.exists():
            df_pref = pd.read_parquet(self.phase3b_pref_path)
            for _, r in df_pref.iterrows():
                prov_map[r["decision_state_id"]] = {
                    "llm_provider": r.get("llm_provider", "unknown"),
                    "llm_model": r.get("llm_model", "unknown"),
                    "eval_latency_ms": r.get("eval_latency_ms", 0),
                    "evaluated_at": str(r.get("evaluated_at", "")),
                }

        # 5. Build joined records
        rows = []
        for state_id, judgment in judgments.items():
            state = states_by_id.get(state_id)
            if not state:
                continue

            asset_ctx = state.get("asset_context", {})
            temp_ctx = state.get("temporal_context", {})
            dataset_id = state.get("dataset_id", "unknown")
            severity = state.get("decision_severity", "WATCH")
            machine_archetype = asset_ctx.get("machine_archetype", "unknown")
            operational_domain = asset_ctx.get("operational_domain", "unknown")
            temporal_type = asset_ctx.get("temporal_type", "unknown")
            asset_id = asset_ctx.get("asset_id", "unknown")

            evaluations = judgment.get("evaluations", [])
            candidate_count = len(evaluations)

            prov = prov_map.get(state_id, {
                "llm_provider": "unknown",
                "llm_model": "unknown",
                "eval_latency_ms": 0,
                "evaluated_at": "",
            })

            for ev in evaluations:
                action_id = ev.get("action_id", "UNKNOWN")
                action_info = ontology.get(action_id, {})
                precond = action_info.get("preconditions", {})

                row = {
                    # Identifiers
                    "decision_state_id": state_id,
                    "action_id": action_id,
                    "dataset_id": dataset_id,
                    "asset_id": asset_id,
                    "machine_archetype": machine_archetype,
                    "operational_domain": operational_domain,
                    "temporal_type": temporal_type,
                    "decision_severity": severity,
                    "candidate_count": candidate_count,

                    # Temporal context
                    "cycle": temp_ctx.get("cycle"),
                    "sequence_step": temp_ctx.get("sequence_step"),
                    "trajectory_phase": temp_ctx.get("trajectory_phase"),

                    # Action ontology features
                    "action_name": action_info.get("action_name", action_id),
                    "action_category": action_info.get("category", "unknown"),
                    "intervention_risk": action_info.get("intervention_risk", "low"),
                    "operational_downtime_cost": action_info.get("operational_downtime_cost", "negligible"),
                    "precondition_min_severity": precond.get("min_severity", "HEALTHY"),
                    "precondition_max_severity": precond.get("max_severity", "CRITICAL"),

                    # Silver preference labels (target candidates)
                    "silver_rank": ev.get("rank"),
                    "silver_suitability_score": ev.get("suitability_score"),
                    "silver_urgency_score": ev.get("urgency_score"),
                    "silver_effectiveness_score": ev.get("expected_effectiveness_score"),
                    "silver_operational_risk": ev.get("operational_risk_score"),
                    "silver_confidence": ev.get("confidence"),
                    "silver_final_verdict": ev.get("final_verdict"),

                    # Provenance metadata (NOT to be used as predictive features)
                    "provenance_llm_provider": prov.get("llm_provider", "unknown"),
                    "provenance_llm_model": prov.get("llm_model", "unknown"),
                    "provenance_eval_latency_ms": prov.get("eval_latency_ms", 0),
                    "provenance_evaluated_at": prov.get("evaluated_at", ""),
                }
                rows.append(row)

        df = pd.DataFrame(rows)

        meta = {
            "total_decision_states": len(states_by_id),
            "total_candidate_pairs": len(df),
            "datasets_represented": sorted(df["dataset_id"].unique().tolist()),
            "actions_represented": sorted(df["action_id"].unique().tolist()),
            "provider_distribution": df["provenance_llm_provider"].value_counts().to_dict(),
        }
        return df, meta
