"""Case-Based Reasoning (CBR) Recommender for Phase 4.

Retrieves k-nearest historical DecisionStates from the training corpus using normalized
semantic state vectors and aggregates their action preferences for query candidates.
Guarantees strict zero-leakage: index contains ONLY training states.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class CBRConfig:
    k_neighbors: int = 5
    metric: str = "cosine"  # 'cosine' | 'euclidean'
    similarity_temperature: float = 1.0
    category_transfer_weight: float = 0.4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CaseBasedRecommender:
    """Similarity-based Case-Based Reasoning prescriptive maintenance recommender."""

    def __init__(self, config: Optional[CBRConfig] = None):
        self.config = config or CBRConfig()
        self.scaler = StandardScaler()
        self.indexed_state_ids: List[str] = []
        self.indexed_state_vectors: Optional[np.ndarray] = None
        self.state_action_preferences: Dict[str, Dict[str, float]] = {}
        self.state_category_preferences: Dict[str, Dict[str, float]] = {}
        self.state_metadata: Dict[str, Dict[str, Any]] = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        state_vectors: Dict[str, np.ndarray],
        relevance_col: str = "ranking_relevance",
    ) -> CaseBasedRecommender:
        """Indexes historical training cases and their silver action preferences."""
        unique_states = sorted(train_df["decision_state_id"].unique().tolist())
        self.indexed_state_ids = unique_states

        # 1. Collect vectors for training states
        raw_vectors = []
        for sid in unique_states:
            if sid not in state_vectors:
                raise ValueError(f"State vector missing for training state: {sid}")
            raw_vectors.append(state_vectors[sid])

        X_raw = np.array(raw_vectors, dtype=np.float32)
        self.indexed_state_vectors = self.scaler.fit_transform(X_raw)

        # 2. Index action and category preferences per state
        self.state_action_preferences = {}
        self.state_category_preferences = {}
        self.state_metadata = {}

        for sid, group in train_df.groupby("decision_state_id"):
            act_prefs: Dict[str, float] = {}
            cat_prefs: Dict[str, float] = {}

            first_row = group.iloc[0]
            self.state_metadata[sid] = {
                "decision_state_id": sid,
                "dataset_id": first_row.get("dataset_id", "unknown"),
                "machine_archetype": first_row.get("machine_archetype", "unknown"),
                "decision_severity": first_row.get("decision_severity", "WATCH"),
            }

            for _, row in group.iterrows():
                aid = row["action_id"]
                rel = float(row.get(relevance_col, 0.0))
                act_prefs[aid] = rel

                cat = row.get("action_category", "unknown")
                if cat not in cat_prefs or rel > cat_prefs[cat]:
                    cat_prefs[cat] = rel

            self.state_action_preferences[sid] = act_prefs
            self.state_category_preferences[sid] = cat_prefs

        return self

    def _compute_similarity(self, query_vec_scaled: np.ndarray) -> np.ndarray:
        """Computes similarity vector against all indexed training states."""
        if self.indexed_state_vectors is None:
            raise ValueError("CBR index is empty.")

        if self.config.metric == "cosine":
            # Cosine similarity
            q_norm = np.linalg.norm(query_vec_scaled)
            idx_norms = np.linalg.norm(self.indexed_state_vectors, axis=1)
            denominator = (idx_norms * q_norm) + 1e-8
            cos_sim = np.dot(self.indexed_state_vectors, query_vec_scaled) / denominator
            # Clip to [0, 1]
            sims = np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0)
            return sims

        elif self.config.metric == "euclidean":
            dists = np.linalg.norm(self.indexed_state_vectors - query_vec_scaled, axis=1)
            sims = np.exp(-dists / (self.config.similarity_temperature * np.median(dists + 1e-5)))
            return sims
        else:
            raise ValueError(f"Unknown metric: {self.config.metric}")

    def score_candidates_for_state(
        self,
        query_state_id: str,
        query_vec: np.ndarray,
        candidate_actions: List[Dict[str, Any]],
        exclude_query_state: bool = True,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Scores candidate actions for a query state using k-NN case retrieval."""
        # Safety assertion: check leakage
        if exclude_query_state and query_state_id in self.indexed_state_ids:
            # Mask out query state if it happens to be in the index (e.g. during training self-eval)
            pass

        q_scaled = self.scaler.transform(query_vec.reshape(1, -1))[0]
        sims = self._compute_similarity(q_scaled)

        # Create mask if query state is indexed to prevent self-retrieval
        if exclude_query_state and query_state_id in self.indexed_state_ids:
            idx = self.indexed_state_ids.index(query_state_id)
            sims[idx] = -1.0

        # Retrieve top-k
        k = min(self.config.k_neighbors, len(self.indexed_state_ids))
        top_k_indices = np.argsort(sims)[::-1][:k]

        retrieved_cases = []
        for idx in top_k_indices:
            sid = self.indexed_state_ids[idx]
            sim_score = float(sims[idx])
            meta = self.state_metadata.get(sid, {})
            # Find best action in neighbor
            best_act = "NONE"
            if sid in self.state_action_preferences:
                prefs = self.state_action_preferences[sid]
                if prefs:
                    best_act = max(prefs.items(), key=lambda item: item[1])[0]

            retrieved_cases.append({
                "neighbor_state_id": sid,
                "similarity_score": sim_score,
                "dataset_id": meta.get("dataset_id"),
                "machine_archetype": meta.get("machine_archetype"),
                "decision_severity": meta.get("decision_severity"),
                "historical_best_action": best_act,
            })

        # Aggregate preference score per candidate
        candidate_scores = []
        total_weight = sum([sims[idx] for idx in top_k_indices]) + 1e-8

        for cand in candidate_actions:
            aid = cand["action_id"]
            cat = cand.get("action_category", "unknown")

            cand_score = 0.0
            for idx in top_k_indices:
                sid = self.indexed_state_ids[idx]
                w = sims[idx]
                act_prefs = self.state_action_preferences.get(sid, {})
                cat_prefs = self.state_category_preferences.get(sid, {})

                if aid in act_prefs:
                    cand_score += w * act_prefs[aid]
                elif cat in cat_prefs:
                    cand_score += w * (self.config.category_transfer_weight * cat_prefs[cat])
                else:
                    cand_score += 0.0

            cand_score /= total_weight
            candidate_scores.append(cand_score)

        return np.array(candidate_scores, dtype=np.float32), retrieved_cases

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        state_vectors: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Predicts CBR scores for all rows in a DataFrame."""
        scores = np.zeros(len(df), dtype=np.float32)

        for sid, group in df.groupby("decision_state_id", sort=False):
            q_vec = state_vectors[sid]
            cands = group[["action_id", "action_category"]].to_dict(orient="records")
            c_scores, _ = self.score_candidates_for_state(sid, q_vec, cands, exclude_query_state=True)
            scores[group.index.values] = c_scores

        return scores
