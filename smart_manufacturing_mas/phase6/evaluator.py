"""Prescriptive ranking evaluation and grouped bootstrap statistical engine for Phase 6."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from phase5.component_failure_evaluator import ComponentFailureEvaluator
from phase6.adaptation_models import BaseRankingModel
from phase6.data_loader import ACTION_TO_CATEGORY, AdaptationInstance


@dataclass
class EvaluationMetrics:
    """Core ranking and agreement metrics for an evaluation split."""
    total_instances: int
    top1_exact_match: float
    top3_recall: float
    mean_reciprocal_rank: float
    mean_ndcg_at_1: float
    mean_ndcg_at_3: float
    mean_ndcg_at_5: float
    category_agreement_rate: float
    component_accuracy_rate: float
    mean_graded_relevance: float
    top1_count: int
    top3_count: int
    category_count: int
    component_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BootstrapMetricCI:
    """Bootstrap confidence interval for a metric."""
    metric_name: str
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Phase6Evaluator:
    """Evaluates prescriptive models on test instances with DCG, NDCG, MRR, and grouped bootstrapping."""

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
        dcg = Phase6Evaluator.compute_dcg_at_k(relevances, k)
        ideal_relevances = [1.0] + [0.0] * (k - 1)
        idcg = Phase6Evaluator.compute_dcg_at_k(ideal_relevances, k)
        return float(dcg / idcg) if idcg > 0.0 else 0.0

    @staticmethod
    def compute_reciprocal_rank(ranked_action_ids: List[str], target_action_id: str) -> float:
        """Computes Reciprocal Rank (1/rank) for the target action."""
        try:
            rank = ranked_action_ids.index(target_action_id) + 1
            return 1.0 / float(rank)
        except ValueError:
            return 0.0

    @staticmethod
    def compute_graded_relevance_score(
        predicted_action: str,
        target_action: str,
        target_component: str,
    ) -> float:
        """Assigns graded relevance in [0.0, 1.0] based on action and component agreement."""
        if predicted_action == target_action:
            return 1.0
        p_comp = ComponentFailureEvaluator.get_action_component(predicted_action)
        p_cat = ACTION_TO_CATEGORY.get(predicted_action, "unknown")
        t_cat = ACTION_TO_CATEGORY.get(target_action, "unknown")

        if p_comp == target_component and target_component != "general":
            return 0.70
        elif p_cat == t_cat:
            return 0.40
        else:
            return 0.0

    def evaluate_model(
        self,
        model: BaseRankingModel,
        test_instances: List[AdaptationInstance],
    ) -> Tuple[EvaluationMetrics, List[Dict[str, Any]]]:
        """Runs inference and computes ranking metrics on test instances."""
        if not test_instances:
            return (
                EvaluationMetrics(
                    total_instances=0,
                    top1_exact_match=0.0,
                    top3_recall=0.0,
                    mean_reciprocal_rank=0.0,
                    mean_ndcg_at_1=0.0,
                    mean_ndcg_at_3=0.0,
                    mean_ndcg_at_5=0.0,
                    category_agreement_rate=0.0,
                    component_accuracy_rate=0.0,
                    mean_graded_relevance=0.0,
                    top1_count=0,
                    top3_count=0,
                    category_count=0,
                    component_count=0,
                ),
                [],
            )

        records = []
        top1_c, top3_c, cat_c, comp_c = 0, 0, 0, 0
        mrrs, ndcg1s, ndcg3s, ndcg5s, gradeds = [], [], [], [], []

        for inst in test_instances:
            ranked = model.rank_candidates(inst)
            ranked_ids = [item["action_id"] for item in ranked]
            tgt_act = inst.target_action
            tgt_comp = inst.target_component
            tgt_cat = inst.target_category

            pred_top1 = ranked_ids[0] if ranked_ids else "NONE"
            pred_comp = ComponentFailureEvaluator.get_action_component(pred_top1)
            pred_cat = ACTION_TO_CATEGORY.get(pred_top1, "unknown")

            is_top1 = bool(pred_top1 == tgt_act)
            is_top3 = bool(tgt_act in ranked_ids[:3])
            is_cat = bool(pred_cat == tgt_cat)
            is_comp = bool(pred_comp == tgt_comp)

            rr = self.compute_reciprocal_rank(ranked_ids, tgt_act)
            n1 = self.compute_ndcg_at_k(ranked_ids, tgt_act, 1)
            n3 = self.compute_ndcg_at_k(ranked_ids, tgt_act, 3)
            n5 = self.compute_ndcg_at_k(ranked_ids, tgt_act, 5)
            gr = self.compute_graded_relevance_score(pred_top1, tgt_act, tgt_comp)

            if is_top1:
                top1_c += 1
            if is_top3:
                top3_c += 1
            if is_cat:
                cat_c += 1
            if is_comp:
                comp_c += 1

            mrrs.append(rr)
            ndcg1s.append(n1)
            ndcg3s.append(n3)
            ndcg5s.append(n5)
            gradeds.append(gr)

            records.append({
                "instance_id": inst.instance_id,
                "asset_id": inst.asset_id,
                "domain_id": inst.domain_id,
                "event_timestamp": inst.event_timestamp,
                "prediction_timestamp": inst.prediction_timestamp,
                "target_action": tgt_act,
                "target_component": tgt_comp,
                "target_category": tgt_cat,
                "predicted_action_top1": pred_top1,
                "predicted_action_top2": ranked_ids[1] if len(ranked_ids) > 1 else "",
                "predicted_action_top3": ranked_ids[2] if len(ranked_ids) > 2 else "",
                "top1_match": is_top1,
                "top3_recall": is_top3,
                "category_match": is_cat,
                "component_match": is_comp,
                "reciprocal_rank": rr,
                "ndcg_at_1": n1,
                "ndcg_at_3": n3,
                "ndcg_at_5": n5,
                "graded_relevance": gr,
            })

        n = len(test_instances)
        summary = EvaluationMetrics(
            total_instances=n,
            top1_exact_match=round(top1_c / n, 4),
            top3_recall=round(top3_c / n, 4),
            mean_reciprocal_rank=round(float(np.mean(mrrs)), 4),
            mean_ndcg_at_1=round(float(np.mean(ndcg1s)), 4),
            mean_ndcg_at_3=round(float(np.mean(ndcg3s)), 4),
            mean_ndcg_at_5=round(float(np.mean(ndcg5s)), 4),
            category_agreement_rate=round(cat_c / n, 4),
            component_accuracy_rate=round(comp_c / n, 4),
            mean_graded_relevance=round(float(np.mean(gradeds)), 4),
            top1_count=top1_c,
            top3_count=top3_c,
            category_count=cat_c,
            component_count=comp_c,
        )

        return summary, records

    @staticmethod
    def compute_grouped_bootstrap_ci(
        instance_records: List[Dict[str, Any]],
        metric_key: str,
        n_bootstraps: int = 1000,
        ci_level: float = 0.95,
        random_seed: int = 42,
    ) -> BootstrapMetricCI:
        """Computes grouped bootstrap confidence interval by resampling asset clusters."""
        if not instance_records:
            return BootstrapMetricCI(metric_key, 0.0, 0.0, 0.0, 0.0, ci_level)

        df = pd.DataFrame(instance_records)
        assets = list(df["asset_id"].unique())
        y_vals = df[metric_key].values
        asset_indices = [np.where(df["asset_id"] == aid)[0] for aid in assets]
        n_assets = len(assets)
        rng = np.random.RandomState(random_seed)

        boot_means = []
        for _ in range(n_bootstraps):
            sampled_a = rng.choice(n_assets, size=n_assets, replace=True)
            idxs = np.concatenate([asset_indices[a] for a in sampled_a])
            if len(idxs) > 0:
                boot_means.append(float(y_vals[idxs].mean()))

        boot_arr = np.array(boot_means)
        alpha = 1.0 - ci_level
        lower = float(np.percentile(boot_arr, 100 * (alpha / 2.0)))
        upper = float(np.percentile(boot_arr, 100 * (1.0 - alpha / 2.0)))

        return BootstrapMetricCI(
            metric_name=metric_key,
            mean=round(float(np.mean(boot_arr)), 4),
            std=round(float(np.std(boot_arr)), 4),
            ci_lower=round(lower, 4),
            ci_upper=round(upper, 4),
            confidence_level=ci_level,
        )

    @staticmethod
    def compute_paired_bootstrap_test(
        records_adapted: List[Dict[str, Any]],
        records_baseline: List[Dict[str, Any]],
        metric_key: str = "reciprocal_rank",
        n_bootstraps: int = 1000,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """Paired bootstrap significance test testing H0: Adapted <= Baseline."""
        if len(records_adapted) != len(records_baseline) or len(records_adapted) == 0:
            return {"delta_mean": 0.0, "p_value": 1.0, "ci_lower": 0.0, "ci_upper": 0.0}

        y_adapt = np.array([r[metric_key] for r in records_adapted])
        y_base = np.array([r[metric_key] for r in records_baseline])
        diff = y_adapt - y_base

        rng = np.random.RandomState(random_seed)
        boot_diffs = []
        n = len(diff)
        for _ in range(n_bootstraps):
            idx = rng.choice(n, size=n, replace=True)
            boot_diffs.append(float(np.mean(diff[idx])))

        boot_arr = np.array(boot_diffs)
        p_val = float(np.mean(boot_arr <= 0.0))  # One-sided p-value
        lower = float(np.percentile(boot_arr, 2.5))
        upper = float(np.percentile(boot_arr, 97.5))

        return {
            "metric_key": metric_key,
            "delta_mean": round(float(np.mean(diff)), 4),
            "p_value": round(p_val, 4),
            "ci_lower": round(lower, 4),
            "ci_upper": round(upper, 4),
            "is_significant": bool(p_val < 0.05),
        }
