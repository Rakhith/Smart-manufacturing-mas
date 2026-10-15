"""External Out-of-Distribution (OOD) Prescriptive Maintenance Evaluator.

Evaluates the frozen Phase 4 model and baselines against empirical dataset ground truth:
1. Action Agreement: Top-1 Match, Top-3 Recall, Category Agreement.
2. Ranking Quality: NDCG@1, NDCG@3, NDCG@5, MRR.
3. Failure Anticipation: Lead Time, Warning Lead-Time Distribution, Anticipation Rate.
4. Economic Outcomes: Downtime hours, Parts Cost ($), Labor Hours.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from phase5.action_mapper import CANONICAL_ACTIONS
from phase5.closed_loop.types import ExecutionMode
from phase5.frozen_model import FrozenPhase4Recommender
from phase5.leakage_guard import TemporalLeakageGuard


class ExternalPrescriptiveEvaluator:
    """Evaluates prescriptive maintenance systems against empirical dataset logs."""

    def __init__(self, recommender: FrozenPhase4Recommender):
        self.recommender = recommender
        self.leakage_guard = TemporalLeakageGuard()

    @staticmethod
    def compute_dcg_at_k(relevances: List[float], k: int) -> float:
        """Computes Discounted Cumulative Gain at rank k."""
        rel = np.asarray(relevances)[:k]
        if rel.size == 0:
            return 0.0
        discounts = np.log2(np.arange(2, rel.size + 2))
        return float(np.sum(rel / discounts))

    @staticmethod
    def compute_ndcg_at_k(ranked_action_ids: List[str], target_action_id: str, k: int) -> float:
        """Computes Normalized DCG@k for a binary target action."""
        relevances = [1.0 if aid == target_action_id else 0.0 for aid in ranked_action_ids[:k]]
        dcg = ExternalPrescriptiveEvaluator.compute_dcg_at_k(relevances, k)
        ideal_relevances = [1.0] + [0.0] * (k - 1)
        idcg = ExternalPrescriptiveEvaluator.compute_dcg_at_k(ideal_relevances, k)
        return float(dcg / idcg) if idcg > 0.0 else 0.0

    @staticmethod
    def compute_reciprocal_rank(ranked_action_ids: List[str], target_action_id: str) -> float:
        """Computes Reciprocal Rank (1/rank) for the target action."""
        try:
            rank = ranked_action_ids.index(target_action_id) + 1
            return 1.0 / float(rank)
        except ValueError:
            return 0.0

    def evaluate_dataset(
        self,
        adapter: Any,
        sample_lead_time_horizons_hours: List[float] = [24.0, 72.0, 168.0],
        max_evaluation_points: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Runs external OOD evaluation across all assets in an external dataset adapter."""
        asset_ids = adapter.get_asset_ids()
        canonical_candidates = self.recommender.get_canonical_candidates()

        eval_records = []
        ndcg1_list = []
        ndcg3_list = []
        ndcg5_list = []
        mrr_list = []
        top1_matches = 0
        top3_matches = 0
        cat_matches = 0
        total_eval_points = 0
        lead_times = []

        for asset_id in asset_ids:
            gt_events = adapter.get_ground_truth_events(asset_id)
            wos = gt_events.get("work_orders", [])
            fails = gt_events.get("failure_events", [])
            telemetry_seq = adapter.get_temporal_sequence(asset_id)
            if telemetry_seq.empty or not wos:
                continue

            # Evaluate each recorded work order
            for wo in wos:
                wo_time_col = "timestamp" if "timestamp" in wo else "opened_at"
                if wo_time_col not in wo or pd.isna(wo[wo_time_col]):
                    continue
                wo_dt = pd.to_datetime(wo[wo_time_col])
                target_action = wo.get("canonical_action_id")
                if not target_action:
                    continue

                # Query telemetry snapshot right before the work order (e.g. 6 to 12 hours before)
                time_col = "timestamp" if "timestamp" in telemetry_seq.columns else "recorded_at"
                pre_wo_data = telemetry_seq[telemetry_seq[time_col] < wo_dt]
                if pre_wo_data.empty:
                    continue

                eval_snapshot = pre_wo_data.iloc[-1]
                pred_timestamp = eval_snapshot[time_col]

                # Run temporal leakage audit
                self.leakage_guard.audit_prediction_point(
                    asset_id=str(asset_id),
                    prediction_timestamp=str(pred_timestamp),
                    latest_input_timestamp=str(pred_timestamp),
                    future_event_timestamp=str(wo_dt),
                    event_type="work_order",
                )

                # Build DecisionState at t
                dstate = adapter.build_decision_state_at(str(asset_id), str(pred_timestamp))

                # Generate recommendations using frozen model
                ranked = self.recommender.rank_candidates(dstate, candidate_actions=canonical_candidates)
                ranked_ids = [r["action_id"] for r in ranked]

                # Metric calculations
                total_eval_points += 1
                n1 = self.compute_ndcg_at_k(ranked_ids, target_action, k=1)
                n3 = self.compute_ndcg_at_k(ranked_ids, target_action, k=3)
                n5 = self.compute_ndcg_at_k(ranked_ids, target_action, k=5)
                rr = self.compute_reciprocal_rank(ranked_ids, target_action)

                ndcg1_list.append(n1)
                ndcg3_list.append(n3)
                ndcg5_list.append(n5)
                mrr_list.append(rr)

                is_top1 = (ranked_ids[0] == target_action)
                is_top3 = (target_action in ranked_ids[:3])

                # Check category match
                target_cat = next((c["action_category"] for c in canonical_candidates if c["action_id"] == target_action), "unknown")
                top_cat = ranked[0]["action_category"]
                is_cat_match = (target_cat == top_cat)

                if is_top1:
                    top1_matches += 1
                if is_top3:
                    top3_matches += 1
                if is_cat_match:
                    cat_matches += 1

                # Calculate warning lead time in hours
                lead_time_h = float((wo_dt - pd.to_datetime(pred_timestamp)).total_seconds() / 3600.0)
                lead_times.append(lead_time_h)

                eval_records.append({
                    "asset_id": str(asset_id),
                    "prediction_timestamp": str(pred_timestamp),
                    "event_timestamp": str(wo_dt),
                    "lead_time_hours": lead_time_h,
                    "target_action": target_action,
                    "predicted_action_top1": ranked_ids[0],
                    "predicted_action_top2": ranked_ids[1] if len(ranked_ids) > 1 else "",
                    "predicted_action_top3": ranked_ids[2] if len(ranked_ids) > 2 else "",
                    "ranked_action_ids": ranked_ids,
                    "top1_match": is_top1,
                    "top1_exact": 1 if is_top1 else 0,
                    "top3_match": is_top3,
                    "top3_recall": 1 if is_top3 else 0,
                    "category_match": is_cat_match,
                    "ndcg_at_1": n1,
                    "ndcg_at_3": n3,
                    "ndcg_at_5": n5,
                    "reciprocal_rank": rr,
                    "hybrid_confidence": ranked[0]["hybrid_score"],
                    "confidence": ranked[0]["hybrid_score"],
                    "downtime_hours": float(wo.get("downtime_hours", 0.0)),
                    "parts_cost": float(wo.get("parts_cost", wo.get("parts_cost_usd", 0.0))),
                    "decision_state": dstate,
                })

                if max_evaluation_points and total_eval_points >= max_evaluation_points:
                    break
            if max_evaluation_points and total_eval_points >= max_evaluation_points:
                break

        summary_metrics = {
            "dataset_id": adapter.dataset_id,
            "total_evaluation_points": total_eval_points,
            "top1_exact_agreement": round(top1_matches / total_eval_points, 4) if total_eval_points > 0 else 0.0,
            "top3_recall": round(top3_matches / total_eval_points, 4) if total_eval_points > 0 else 0.0,
            "category_agreement": round(cat_matches / total_eval_points, 4) if total_eval_points > 0 else 0.0,
            "mean_ndcg_at_1": round(float(np.mean(ndcg1_list)), 4) if ndcg1_list else 0.0,
            "mean_ndcg_at_3": round(float(np.mean(ndcg3_list)), 4) if ndcg3_list else 0.0,
            "mean_ndcg_at_5": round(float(np.mean(ndcg5_list)), 4) if ndcg5_list else 0.0,
            "mean_reciprocal_rank": round(float(np.mean(mrr_list)), 4) if mrr_list else 0.0,
            "lead_time_hours_mean": round(float(np.mean(lead_times)), 2) if lead_times else 0.0,
            "lead_time_hours_median": round(float(np.median(lead_times)), 2) if lead_times else 0.0,
            "leakage_audit": self.leakage_guard.get_summary(),
        }

        return {
            "summary_metrics": summary_metrics,
            "evaluation_records": eval_records,
        }
