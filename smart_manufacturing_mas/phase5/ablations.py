"""Ablation Studies for Phase 5 Prescriptive Maintenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from phase5.evaluator import ExternalPrescriptiveEvaluator
from phase5.frozen_model import FrozenPhase4Recommender


class Phase5AblationRunner:
    """Executes systematic ablations across model components and representations."""

    def __init__(self, recommender: FrozenPhase4Recommender):
        self.recommender = recommender

    def run_alpha_sweep_ablation(
        self,
        eval_records: List[Dict[str, Any]],
        adapter: Any,
    ) -> Dict[str, Any]:
        """Ablates the hybrid fusion parameter alpha: LGBM-only (1.0), CBR-only (0.0), and Hybrid (0.75)."""
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = {}

        canonical_cands = self.recommender.get_canonical_candidates()
        target_actions = [r["target_action"] for r in eval_records]

        for a in alphas:
            old_alpha = self.recommender.alpha
            self.recommender.alpha = a

            ndcg1_s = []
            ndcg3_s = []
            mrr_s = []
            top1_c = 0

            for r in eval_records:
                sid = r["asset_id"]
                t = r["prediction_timestamp"]
                dstate = adapter.build_decision_state_at(sid, t)
                ranked = self.recommender.rank_candidates(dstate, candidate_actions=canonical_cands)
                ranked_ids = [act["action_id"] for act in ranked]
                target = r["target_action"]

                n1 = ExternalPrescriptiveEvaluator.compute_ndcg_at_k(ranked_ids, target, k=1)
                n3 = ExternalPrescriptiveEvaluator.compute_ndcg_at_k(ranked_ids, target, k=3)
                rr = ExternalPrescriptiveEvaluator.compute_reciprocal_rank(ranked_ids, target)

                ndcg1_s.append(n1)
                ndcg3_s.append(n3)
                mrr_s.append(rr)
                if ranked_ids[0] == target:
                    top1_c += 1

            self.recommender.alpha = old_alpha
            n_tot = len(eval_records)
            results[f"alpha_{a:.2f}"] = {
                "alpha": a,
                "model_description": "CBR Only" if a == 0.0 else ("LGBM Only" if a == 1.0 else f"Hybrid (alpha={a:.2f})"),
                "top1_exact_agreement": round(top1_c / n_tot, 4) if n_tot > 0 else 0.0,
                "mean_ndcg_at_1": round(float(np.mean(ndcg1_s)), 4) if ndcg1_s else 0.0,
                "mean_ndcg_at_3": round(float(np.mean(ndcg3_s)), 4) if ndcg3_s else 0.0,
                "mean_reciprocal_rank": round(float(np.mean(mrr_s)), 4) if mrr_s else 0.0,
            }

        return results

    def run_feature_representation_ablation(
        self,
        eval_records: List[Dict[str, Any]],
        adapter: Any,
    ) -> Dict[str, Any]:
        """Ablates semantic feature representations: Raw, Canonical, and Full Semantic+Trend."""
        subsets = ["raw_structured", "canonical", "full_semantic_trend"]
        results = {}

        canonical_cands = self.recommender.get_canonical_candidates()

        for subset in subsets:
            ndcg1_s = []
            ndcg3_s = []
            mrr_s = []
            top1_c = 0

            for r in eval_records:
                sid = r["asset_id"]
                t = r["prediction_timestamp"]
                dstate = adapter.build_decision_state_at(sid, t)

                # Zero-mask feature ablation preserving 96-dim tensor layout
                orig_transform = self.recommender.pipeline.transform_dataframe

                def patched_transform(df, feature_subset="full_semantic_trend"):
                    X_full, fnames = orig_transform(df, feature_subset="full_semantic_trend")
                    X_ablated = X_full.copy()
                    for c_idx, fname in enumerate(fnames):
                        if subset == "raw_structured":
                            # Mask out modalities, canonical physical values, and trends
                            if "count_" not in fname and "severity" not in fname and "arch_" not in fname and not fname.startswith("act__") and "sev_x_risk" not in fname and "admissible" not in fname:
                                X_ablated[:, c_idx] = 0.0
                        elif subset == "canonical":
                            # Mask out trend and sequence dynamics only
                            if "trend" in fname or "sequence" in fname or "cycle" in fname:
                                X_ablated[:, c_idx] = 0.0
                    return X_ablated, fnames

                self.recommender.pipeline.transform_dataframe = patched_transform
                ranked = self.recommender.rank_candidates(dstate, candidate_actions=canonical_cands)
                self.recommender.pipeline.transform_dataframe = orig_transform

                ranked_ids = [act["action_id"] for act in ranked]
                target = r["target_action"]

                n1 = ExternalPrescriptiveEvaluator.compute_ndcg_at_k(ranked_ids, target, k=1)
                n3 = ExternalPrescriptiveEvaluator.compute_ndcg_at_k(ranked_ids, target, k=3)
                rr = ExternalPrescriptiveEvaluator.compute_reciprocal_rank(ranked_ids, target)

                ndcg1_s.append(n1)
                ndcg3_s.append(n3)
                mrr_s.append(rr)
                if ranked_ids[0] == target:
                    top1_c += 1

            n_tot = len(eval_records)
            desc_map = {
                "raw_structured": "Ablation A: Raw Structured Features Only",
                "canonical": "Ablation B: Canonical Semantic Representation",
                "full_semantic_trend": "Ablation C: Full Semantic + Temporal Trends (Proposed)",
            }
            results[subset] = {
                "subset_key": subset,
                "description": desc_map.get(subset, subset),
                "top1_exact_agreement": round(top1_c / n_tot, 4) if n_tot > 0 else 0.0,
                "mean_ndcg_at_1": round(float(np.mean(ndcg1_s)), 4) if ndcg1_s else 0.0,
                "mean_ndcg_at_3": round(float(np.mean(ndcg3_s)), 4) if ndcg3_s else 0.0,
                "mean_reciprocal_rank": round(float(np.mean(mrr_s)), 4) if mrr_s else 0.0,
            }

        return results

    def save_ablation_results(self, results: Dict[str, Any], output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
