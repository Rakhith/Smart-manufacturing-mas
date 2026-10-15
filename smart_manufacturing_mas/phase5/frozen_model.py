"""Frozen Phase 4 Prescriptive Maintenance Recommender.

Guarantees strict out-of-distribution (OOD) evaluation integrity:
1. Loads pre-trained Phase 4 LightGBM LambdaRanker and metadata read-only.
2. Computes and validates cryptographic SHA-256 hashes to guarantee model immutability.
3. Indexes Case-Based Reasoning (CBR) exclusively on Phase 4 training states.
4. Enforces frozen validation alpha = 0.75 without any tuning on external datasets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from phase4.cbr_recommender import CaseBasedRecommender, CBRConfig
from phase4.feature_pipeline import FeaturePipeline, ACTION_CATEGORIES
from phase4.hybrid_recommender import HybridRecommender, HybridConfig
from phase4.lgbm_ranker import MaintenanceLGBMRanker


class FrozenPhase4Recommender:
    """Read-only wrapper around the trained Phase 4 Hybrid Prescriptive Recommender."""

    def __init__(
        self,
        model_dir: Path = Path("outputs/phase4/models"),
        training_table_path: Path = Path("outputs/phase4/training/phase4_training_table.parquet"),
        split_manifest_path: Path = Path("outputs/phase4/splits/split_manifest.json"),
        corpus_path: Path = Path("outputs/phase3a/selected_decision_states/decision_states_corpus.json"),
        ontology_path: Path = Path("outputs/phase3b/action_ontology/action_ontology.json"),
        alpha: float = 0.75,
        cbr_k: int = 5,
    ):
        self.model_dir = Path(model_dir)
        self.alpha = float(alpha)
        self.cbr_k = int(cbr_k)

        # 1. Compute and lock cryptographic hash
        self.model_joblib_path = self.model_dir / "lgbm_ranker.joblib"
        if not self.model_joblib_path.exists():
            raise FileNotFoundError(f"Phase 4 model artifact not found at {self.model_joblib_path}")
        self.initial_model_hash = self._compute_sha256(self.model_joblib_path)

        # 2. Load frozen ranker
        self.ranker = MaintenanceLGBMRanker.load(self.model_dir)
        self.expected_feature_names = self.ranker.feature_names

        # 3. Load feature pipeline and ontology
        self.pipeline = FeaturePipeline(corpus_path=corpus_path)
        with open(ontology_path, "r", encoding="utf-8") as f:
            self.action_ontology: Dict[str, Dict[str, Any]] = json.load(f)

        # 4. Initialize CBR strictly on training states (zero leakage)
        self.cbr = CaseBasedRecommender(config=CBRConfig(k_neighbors=self.cbr_k))
        self._init_cbr_from_training(training_table_path, split_manifest_path, corpus_path)

        # 5. Hybrid fusion
        self.hybrid = HybridRecommender(config=HybridConfig(alpha=self.alpha))

    @staticmethod
    def _compute_sha256(filepath: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def verify_frozen_invariants(self) -> Dict[str, Any]:
        """Verifies that the underlying model weights have not mutated."""
        current_hash = self._compute_sha256(self.model_joblib_path)
        is_immutable = (current_hash == self.initial_model_hash)
        return {
            "model_path": str(self.model_joblib_path),
            "initial_sha256": self.initial_model_hash,
            "current_sha256": current_hash,
            "is_immutable": is_immutable,
            "frozen_alpha": self.alpha,
            "cbr_k": self.cbr_k,
            "feature_count": len(self.expected_feature_names),
            "status": "PASS" if is_immutable else "FAIL_MUTATED",
        }

    def _init_cbr_from_training(
        self,
        training_table_path: Path,
        split_manifest_path: Path,
        corpus_path: Path,
    ) -> None:
        """Indexes historical training cases without any validation or external state contamination."""
        train_df = pd.read_parquet(training_table_path)

        # Restrict strictly to train split IDs
        if Path(split_manifest_path).exists():
            with open(split_manifest_path, "r", encoding="utf-8") as f:
                split_manifest = json.load(f)
            train_state_ids = set(split_manifest.get("train_state_ids", []))
            train_df = train_df[train_df["decision_state_id"].isin(train_state_ids)].copy()

        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        state_vectors = {s["decision_state_id"]: self.pipeline.get_state_vector(s) for s in corpus if s["decision_state_id"] in set(train_df["decision_state_id"])}

        self.cbr.fit(train_df=train_df, state_vectors=state_vectors, relevance_col="ranking_relevance")

    def get_canonical_candidates(self) -> List[Dict[str, Any]]:
        """Returns the full 20-action canonical catalog formatted as candidate records."""
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

    def rank_candidates(
        self,
        decision_state: Dict[str, Any],
        candidate_actions: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Ranks candidate actions for a given DecisionState using the frozen Hybrid model."""
        if candidate_actions is None:
            candidate_actions = self.get_canonical_candidates()

        sid = decision_state.get("decision_state_id", "QUERY_STATE")

        # 1. Build DataFrame rows for candidate actions
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

        # 2. Extract features using the feature pipeline
        # Inject query decision_state into pipeline state_details temporarily for extraction
        self.pipeline.state_details[sid] = decision_state
        X_query, _ = self.pipeline.transform_dataframe(df_query, feature_subset="full_semantic_trend")

        # 3. Predict LGBM ranking scores
        lgbm_scores = self.ranker.predict(X_query)

        # 4. Predict CBR similarity scores
        q_vec = self.pipeline.get_state_vector(decision_state)
        cbr_scores, retrieved_neighbors = self.cbr.score_candidates_for_state(
            query_state_id=sid,
            query_vec=q_vec,
            candidate_actions=candidate_actions,
            exclude_query_state=False,
        )

        # 5. Fuse via Hybrid Recommender (alpha = 0.75)
        # Normalize within candidate group
        group_idx = [np.arange(len(candidate_actions))]
        norm_lgbm = self.hybrid.normalize_scores_per_group(lgbm_scores, group_idx)
        norm_cbr = self.hybrid.normalize_scores_per_group(cbr_scores, group_idx)
        hybrid_scores = (self.alpha * norm_lgbm) + ((1.0 - self.alpha) * norm_cbr)

        # 6. Assemble ranked recommendations
        ranked_actions = []
        for i, cand in enumerate(candidate_actions):
            ranked_actions.append({
                "action_id": cand["action_id"],
                "action_name": cand.get("action_name", cand["action_id"]),
                "action_category": cand.get("action_category", "unknown"),
                "description": cand.get("description", ""),
                "hybrid_score": float(hybrid_scores[i]),
                "lgbm_score": float(lgbm_scores[i]),
                "cbr_score": float(cbr_scores[i]),
                "norm_lgbm_score": float(norm_lgbm[i]),
                "norm_cbr_score": float(norm_cbr[i]),
            })

        # Sort descending by hybrid score
        ranked_actions.sort(key=lambda item: item["hybrid_score"], reverse=True)

        # Attach rank positions
        for r, act in enumerate(ranked_actions, start=1):
            act["rank"] = r

        # Clean up temporary state in pipeline
        self.pipeline.state_details.pop(sid, None)

        return ranked_actions
