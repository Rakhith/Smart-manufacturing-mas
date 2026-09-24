"""Comprehensive Evaluation Suite for Prescriptive Maintenance Recommender.

Computes ranking metrics (NDCG@k, Recall@k, MRR), diversity, action coverage,
and stratified slices (dataset, severity, provider, confidence).
Strictly designated as evaluation against silver-standard LLM preferences.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class PrescriptiveEvaluator:
    """Evaluates predicted ranking scores against silver-standard preference labels."""

    def __init__(self, total_ontology_actions: int = 21):
        self.total_ontology_actions = total_ontology_actions

    @staticmethod
    def dcg_at_k(r: List[float], k: int) -> float:
        r = list(r)[:k]
        if not r:
            return 0.0
        return sum((2.0 ** rel - 1.0) / np.log2(idx + 2.0) for idx, rel in enumerate(r))

    @classmethod
    def ndcg_at_k(cls, true_relevance: List[float], pred_scores: List[float], k: int) -> float:
        order = np.argsort(pred_scores)[::-1]
        ranked_true = [true_relevance[i] for i in order]

        actual_dcg = cls.dcg_at_k(ranked_true, k)
        ideal_dcg = cls.dcg_at_k(sorted(true_relevance, reverse=True), k)

        if ideal_dcg <= 0.0:
            return 1.0
        return float(actual_dcg / ideal_dcg)

    def evaluate_predictions(
        self,
        df: pd.DataFrame,
        pred_scores: np.ndarray,
        relevance_col: str = "ranking_relevance",
        rank_col: str = "silver_rank",
    ) -> Dict[str, float]:
        """Calculates NDCG@{1,3,5}, Recall@{1,3,5}, MRR, coverage, and diversity."""
        ndcg1_list = []
        ndcg3_list = []
        ndcg5_list = []
        recall1_list = []
        recall3_list = []
        recall5_list = []
        mrr_list = []

        top_recommended_actions = []

        # Iterate by state group
        for sid, group in df.groupby("decision_state_id", sort=False):
            grp_indices = group.index.values
            scores = pred_scores[grp_indices]
            true_rel = group[relevance_col].tolist()
            silver_ranks = group[rank_col].tolist()
            action_ids = group["action_id"].tolist()

            # Rank candidates by descending predicted score
            ranking_order = np.argsort(scores)[::-1]
            predicted_actions = [action_ids[i] for i in ranking_order]

            # Track top-1 predicted action
            top_recommended_actions.append(predicted_actions[0])

            # NDCG
            ndcg1_list.append(self.ndcg_at_k(true_rel, scores, k=1))
            ndcg3_list.append(self.ndcg_at_k(true_rel, scores, k=3))
            ndcg5_list.append(self.ndcg_at_k(true_rel, scores, k=5))

            # Identify index of silver rank 1 action
            try:
                top_silver_idx = silver_ranks.index(1)
                top_silver_action = action_ids[top_silver_idx]

                # Position in predicted ranking (1-based)
                pred_pos = predicted_actions.index(top_silver_action) + 1
                mrr_list.append(1.0 / pred_pos)

                recall1_list.append(1.0 if pred_pos <= 1 else 0.0)
                recall3_list.append(1.0 if pred_pos <= 3 else 0.0)
                recall5_list.append(1.0 if pred_pos <= 5 else 0.0)
            except ValueError:
                # Rank 1 was missing in this group
                pass

        # Action coverage & diversity
        unique_top_actions = set(top_recommended_actions)
        coverage = len(unique_top_actions) / float(self.total_ontology_actions)

        # Diversity: Normalized Shannon Entropy
        counts = Counter(top_recommended_actions)
        n_total = len(top_recommended_actions)
        probs = [c / n_total for c in counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs)
        max_entropy = np.log2(self.total_ontology_actions)
        norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

        return {
            "ndcg_at_1": float(np.mean(ndcg1_list)) if ndcg1_list else 0.0,
            "ndcg_at_3": float(np.mean(ndcg3_list)) if ndcg3_list else 0.0,
            "ndcg_at_5": float(np.mean(ndcg5_list)) if ndcg5_list else 0.0,
            "recall_at_1": float(np.mean(recall1_list)) if recall1_list else 0.0,
            "recall_at_3": float(np.mean(recall3_list)) if recall3_list else 0.0,
            "recall_at_5": float(np.mean(recall5_list)) if recall5_list else 0.0,
            "mrr": float(np.mean(mrr_list)) if mrr_list else 0.0,
            "action_coverage_pct": float(coverage * 100.0),
            "recommendation_diversity_entropy": float(entropy),
            "normalized_diversity": norm_entropy,
            "total_eval_states": len(ndcg1_list),
        }

    def evaluate_slices(
        self,
        df: pd.DataFrame,
        pred_scores: np.ndarray,
        slice_column: str,
        relevance_col: str = "ranking_relevance",
    ) -> Dict[str, Dict[str, float]]:
        """Evaluates metrics sliced by a specified column (e.g. dataset_id, severity, provider)."""
        results = {}
        for slice_val, group in df.groupby(slice_column):
            grp_indices = group.index.values
            slice_scores = pred_scores[grp_indices]
            slice_df = group.reset_index(drop=True)
            res = self.evaluate_predictions(slice_df, slice_scores, relevance_col=relevance_col)
            results[str(slice_val)] = res
        return results

    def run_baselines(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        relevance_col: str = "ranking_relevance",
        random_seed: int = 42,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluates Random, Majority/Popularity, and Severity-heuristic baselines."""
        rng = np.random.default_rng(random_seed)

        # 1. Random ranking
        random_scores = rng.standard_normal(len(test_df)).astype(np.float32)
        res_random = self.evaluate_predictions(test_df, random_scores, relevance_col=relevance_col)

        # 2. Majority / Popularity baseline: frequency of being silver rank 1 in training data
        rank1_train = train_df[train_df["silver_rank"] == 1]
        pop_counts = rank1_train["action_id"].value_counts().to_dict()
        pop_scores = np.array([float(pop_counts.get(aid, 0.0)) for aid in test_df["action_id"]], dtype=np.float32)
        # Add tiny jitter to break ties deterministically
        jitter = rng.uniform(0, 1e-4, size=len(test_df)).astype(np.float32)
        res_pop = self.evaluate_predictions(test_df, pop_scores + jitter, relevance_col=relevance_col)

        # 3. Severity-heuristic baseline:
        # HEALTHY -> ACT_MON_CONTINUE (high)
        # WATCH -> ACT_MON_ENHANCED / ACT_INSP_*
        # DEGRADING -> ACT_INSP_* / ACT_REPL_*
        # CRITICAL -> ACT_OP_CONTROLLED_SHUTDOWN / ACT_REPL_*
        heur_scores = []
        for _, row in test_df.iterrows():
            sev = str(row.get("decision_severity", "WATCH")).upper()
            aid = str(row.get("action_id", ""))
            cat = str(row.get("action_category", ""))

            s = 0.0
            if sev == "HEALTHY":
                if aid == "ACT_MON_CONTINUE":
                    s = 10.0
                elif cat == "monitoring_observation":
                    s = 5.0
            elif sev == "WATCH":
                if aid == "ACT_MON_ENHANCED":
                    s = 10.0
                elif cat == "inspection_diagnosis":
                    s = 8.0
                elif cat == "monitoring_observation":
                    s = 4.0
            elif sev == "DEGRADING":
                if cat == "inspection_diagnosis":
                    s = 10.0
                elif cat == "replacement_overhaul":
                    s = 8.0
                elif cat == "operational_mitigation":
                    s = 6.0
            elif sev == "CRITICAL":
                if aid == "ACT_OP_CONTROLLED_SHUTDOWN":
                    s = 10.0
                elif cat == "replacement_overhaul":
                    s = 8.0
                elif cat == "operational_mitigation":
                    s = 7.0
            heur_scores.append(s)

        heur_scores_arr = np.array(heur_scores, dtype=np.float32) + jitter
        res_heur = self.evaluate_predictions(test_df, heur_scores_arr, relevance_col=relevance_col)

        return {
            "random_ranking": res_random,
            "majority_popularity": res_pop,
            "severity_heuristic": res_heur,
        }
