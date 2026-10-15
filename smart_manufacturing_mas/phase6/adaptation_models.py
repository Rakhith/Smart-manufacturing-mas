"""Adaptation models and baseline ranking policies for Phase 6."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from phase4.cbr_recommender import CaseBasedRecommender, CBRConfig
from phase4.feature_pipeline import FeaturePipeline
from phase4.hybrid_recommender import HybridRecommender, HybridConfig
from phase4.lgbm_ranker import MaintenanceLGBMRanker, RankerHyperparameters
from phase5.action_mapper import CANONICAL_ACTIONS
from phase5.component_failure_evaluator import ComponentFailureEvaluator
from phase5.frozen_model import FrozenPhase4Recommender
from phase6.data_loader import AdaptationInstance


class BaseRankingModel:
    """Abstract base class for prescriptive ranking models."""

    def rank_candidates(self, instance: AdaptationInstance) -> List[Dict[str, Any]]:
        raise NotImplementedError


# ==============================================================================
# Method A: Zero-Shot Transfer (Frozen Phase 4 Hybrid Recommender)
# ==============================================================================

class ZeroShotHybridModel(BaseRankingModel):
    """Frozen reference Phase 4 Hybrid Recommender (alpha = 0.75, k = 5)."""

    def __init__(self, frozen_recommender: FrozenPhase4Recommender):
        self.frozen_recommender = frozen_recommender

    def rank_candidates(self, instance: AdaptationInstance) -> List[Dict[str, Any]]:
        ranked = self.frozen_recommender.rank_candidates(
            instance.decision_state,
            candidate_actions=instance.candidate_actions,
        )
        return ranked


# ==============================================================================
# Method C: Frozen Source LGBM Only (Without CBR)
# ==============================================================================

class FrozenSourceLGBMModel(BaseRankingModel):
    """Frozen Phase 4 LightGBM Ranker without CBR fusion."""

    def __init__(self, ranker: MaintenanceLGBMRanker):
        self.ranker = ranker

    def rank_candidates(self, instance: AdaptationInstance) -> List[Dict[str, Any]]:
        scores = self.ranker.predict(instance.X_features)
        ranked = []
        for i, cand in enumerate(instance.candidate_actions):
            ranked.append({
                "action_id": cand["action_id"],
                "action_name": cand.get("action_name", cand["action_id"]),
                "action_category": cand.get("action_category", "unknown"),
                "score": float(scores[i]),
                "lgbm_score": float(scores[i]),
            })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        for r, act in enumerate(ranked, start=1):
            act["rank"] = r
        return ranked


# ==============================================================================
# Method B: Target-Only Training from Scratch
# ==============================================================================

class TargetOnlyRankerModel(BaseRankingModel):
    """Trains a fresh LightGBM LambdaRank model strictly on the target adaptation subset."""

    def __init__(
        self,
        n_estimators: int = 25,
        learning_rate: float = 0.05,
        num_leaves: int = 7,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.model: Optional[lgb.LGBMRanker] = None
        self.fallback_priors: Dict[str, float] = {}

    def fit(self, adaptation_instances: List[AdaptationInstance]) -> TargetOnlyRankerModel:
        if not adaptation_instances:
            self.model = None
            return self

        # Empirical action frequency fallback
        counts: Dict[str, int] = {}
        for inst in adaptation_instances:
            counts[inst.target_action] = counts.get(inst.target_action, 0) + 1
        total = sum(counts.values()) or 1
        self.fallback_priors = {a: counts.get(a, 0) / total for a in CANONICAL_ACTIONS}

        # Need at least 2 groups with variance for LambdaRank
        if len(adaptation_instances) < 2:
            self.model = None
            return self

        X_list = [inst.X_features for inst in adaptation_instances]
        y_list = [inst.relevance_grades for inst in adaptation_instances]
        groups = [len(inst.candidate_actions) for inst in adaptation_instances]

        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)

        # Check if y has variance
        if np.std(y_train) < 1e-6:
            self.model = None
            return self

        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=1,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.model.fit(X_train, y_train, group=groups)
        return self

    def rank_candidates(self, instance: AdaptationInstance) -> List[Dict[str, Any]]:
        cands = instance.candidate_actions
        if self.model is not None:
            try:
                scores = self.model.predict(instance.X_features)
            except Exception:
                scores = np.array([self.fallback_priors.get(c["action_id"], 0.0) for c in cands])
        else:
            scores = np.array([self.fallback_priors.get(c["action_id"], 0.0) for c in cands])

        ranked = []
        for i, cand in enumerate(cands):
            ranked.append({
                "action_id": cand["action_id"],
                "action_name": cand.get("action_name", cand["action_id"]),
                "action_category": cand.get("action_category", "unknown"),
                "score": float(scores[i]),
            })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        for r, act in enumerate(ranked, start=1):
            act["rank"] = r
        return ranked


# ==============================================================================
# Method D1: Source Model Continuation Boosting Fine-Tuning
# ==============================================================================

class ContinuationBoostedRankerModel(BaseRankingModel):
    """LightGBM continuation boosting initialized from frozen source booster."""

    def __init__(
        self,
        source_ranker: MaintenanceLGBMRanker,
        additional_estimators: int = 15,
        adaptation_lr: float = 0.02,
        random_state: int = 42,
    ):
        self.source_ranker = source_ranker
        self.additional_estimators = additional_estimators
        self.adaptation_lr = adaptation_lr
        self.random_state = random_state
        self.adapted_booster: Optional[lgb.Booster] = None

    def fit(self, adaptation_instances: List[AdaptationInstance]) -> ContinuationBoostedRankerModel:
        if not adaptation_instances or len(adaptation_instances) < 1:
            self.adapted_booster = None
            return self

        X_list = [inst.X_features for inst in adaptation_instances]
        y_list = [inst.relevance_grades for inst in adaptation_instances]
        groups = [len(inst.candidate_actions) for inst in adaptation_instances]

        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)

        if np.std(y_train) < 1e-6:
            self.adapted_booster = None
            return self

        # Initialize new ranker that continues boosting from source model booster
        src_booster = self.source_ranker.model.booster_
        cont_ranker = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=self.additional_estimators,
            learning_rate=self.adaptation_lr,
            num_leaves=15,
            min_child_samples=1,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )

        try:
            cont_ranker.fit(
                X=X_train,
                y=y_train,
                group=groups,
                init_model=src_booster,
            )
            self.adapted_booster = cont_ranker.booster_
        except Exception:
            self.adapted_booster = None

        return self

    def rank_candidates(self, instance: AdaptationInstance) -> List[Dict[str, Any]]:
        cands = instance.candidate_actions
        if self.adapted_booster is not None:
            scores = self.adapted_booster.predict(instance.X_features)
        else:
            scores = self.source_ranker.predict(instance.X_features)

        ranked = []
        for i, cand in enumerate(cands):
            ranked.append({
                "action_id": cand["action_id"],
                "action_name": cand.get("action_name", cand["action_id"]),
                "action_category": cand.get("action_category", "unknown"),
                "score": float(scores[i]),
            })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        for r, act in enumerate(ranked, start=1):
            act["rank"] = r
        return ranked


# ==============================================================================
# Method D2: Residual / Correction Re-ranking Model
# ==============================================================================

class ResidualCorrectionRankerModel(BaseRankingModel):
    """Source model + target-trained lightweight residual correction layer."""

    def __init__(self, source_ranker: MaintenanceLGBMRanker, alpha_reg: float = 1.0):
        self.source_ranker = source_ranker
        self.alpha_reg = alpha_reg
        self.residual_regressor: Optional[Ridge] = None
        self.gamma: float = 0.5  # Residual blend factor

    def fit(self, adaptation_instances: List[AdaptationInstance], gamma: float = 0.5) -> ResidualCorrectionRankerModel:
        self.gamma = gamma
        if not adaptation_instances:
            self.residual_regressor = None
            return self

        X_list = [inst.X_features for inst in adaptation_instances]
        y_list = [inst.relevance_grades for inst in adaptation_instances]

        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)

        # Baseline source predictions
        src_preds = self.source_ranker.predict(X_train)
        # Target residual: normalize both to [0, 1] scale
        y_norm = y_train / 4.0
        s_norm = (src_preds - np.min(src_preds)) / (np.max(src_preds) - np.min(src_preds) + 1e-8)
        residual = y_norm - s_norm

        self.residual_regressor = Ridge(alpha=self.alpha_reg, random_state=42)
        self.residual_regressor.fit(X_train, residual)
        return self

    def rank_candidates(self, instance: AdaptationInstance) -> List[Dict[str, Any]]:
        cands = instance.candidate_actions
        src_scores = self.source_ranker.predict(instance.X_features)

        if self.residual_regressor is not None:
            res_pred = self.residual_regressor.predict(instance.X_features)
            # Combine source normalized score + gamma * residual
            s_min, s_max = np.min(src_scores), np.max(src_scores)
            s_norm = (src_scores - s_min) / (s_max - s_min + 1e-8)
            final_scores = s_norm + (self.gamma * res_pred)
        else:
            final_scores = src_scores

        ranked = []
        for i, cand in enumerate(cands):
            ranked.append({
                "action_id": cand["action_id"],
                "action_name": cand.get("action_name", cand["action_id"]),
                "action_category": cand.get("action_category", "unknown"),
                "score": float(final_scores[i]),
            })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        for r, act in enumerate(ranked, start=1):
            act["rank"] = r
        return ranked


# ==============================================================================
# Method E1: Target-Adapted Case-Based Reasoning (CBR)
# ==============================================================================

class TargetAdaptedCBRModel(BaseRankingModel):
    """Case-Based Reasoning index augmented with local target adaptation cases."""

    def __init__(self, base_cbr: CaseBasedRecommender, pipeline: FeaturePipeline):
        self.base_cbr = base_cbr
        self.pipeline = pipeline
        self.local_cases: List[Dict[str, Any]] = []

    def fit(self, adaptation_instances: List[AdaptationInstance]) -> TargetAdaptedCBRModel:
        self.local_cases = []
        for inst in adaptation_instances:
            q_vec = self.pipeline.get_state_vector(inst.decision_state)
            self.local_cases.append({
                "instance_id": inst.instance_id,
                "vector": q_vec,
                "target_action": inst.target_action,
                "target_component": inst.target_component,
                "target_category": inst.target_category,
            })
        return self

    def rank_candidates(self, instance: AdaptationInstance) -> List[Dict[str, Any]]:
        cands = instance.candidate_actions
        q_vec = self.pipeline.get_state_vector(instance.decision_state)

        # Baseline CBR scores from source case base
        base_scores, _ = self.base_cbr.score_candidates_for_state(
            query_state_id=instance.instance_id,
            query_vec=q_vec,
            candidate_actions=cands,
            exclude_query_state=False,
        )

        if not self.local_cases:
            final_scores = base_scores
        else:
            # Compute cosine similarities to local cases
            local_scores = np.zeros(len(cands), dtype=np.float32)
            q_norm = np.linalg.norm(q_vec) + 1e-8
            for lc in self.local_cases:
                v = lc["vector"]
                sim = float(np.dot(q_vec, v) / (q_norm * (np.linalg.norm(v) + 1e-8)))
                sim = max(0.0, sim)
                tgt_a = lc["target_action"]
                tgt_c = lc["target_component"]
                for i, c in enumerate(cands):
                    if c["action_id"] == tgt_a:
                        local_scores[i] += sim * 1.0
                    elif ComponentFailureEvaluator.get_action_component(c["action_id"]) == tgt_c:
                        local_scores[i] += sim * 0.5

            total_sim = sum(max(0.0, np.dot(q_vec, lc["vector"]) / (q_norm * (np.linalg.norm(lc["vector"]) + 1e-8))) for lc in self.local_cases) + 1e-8
            local_scores = local_scores / total_sim
            # Blend base CBR and local CBR (60% local, 40% base)
            final_scores = (0.4 * base_scores) + (0.6 * local_scores)

        ranked = []
        for i, cand in enumerate(cands):
            ranked.append({
                "action_id": cand["action_id"],
                "action_name": cand.get("action_name", cand["action_id"]),
                "action_category": cand.get("action_category", "unknown"),
                "score": float(final_scores[i]),
            })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        for r, act in enumerate(ranked, start=1):
            act["rank"] = r
        return ranked


# ==============================================================================
# Method E2: Prior-Adapted Hybrid Model (Validation-Tuned Beta)
# ==============================================================================

class PriorAdaptedHybridModel(BaseRankingModel):
    """Frozen Hybrid model with validation-selected empirical action log-prior adjustment."""

    def __init__(
        self,
        base_recommender: FrozenPhase4Recommender,
        action_priors: Dict[str, float],
        beta: float = 0.5,
    ):
        self.base_recommender = base_recommender
        self.action_priors = action_priors
        self.beta = float(beta)

    def rank_candidates(self, instance: AdaptationInstance) -> List[Dict[str, Any]]:
        frozen_ranked = self.base_recommender.rank_candidates(
            instance.decision_state,
            candidate_actions=instance.candidate_actions,
        )

        if self.beta <= 0.0 or not self.action_priors:
            return frozen_ranked

        adapted = []
        for i, item in enumerate(frozen_ranked):
            item_copy = dict(item)
            aid = item["action_id"]
            orig_score = float(item.get("hybrid_score", 1.0 - (i * 0.05)))
            prior = float(self.action_priors.get(aid, 1e-4))
            adjusted_score = orig_score + (self.beta * np.log(prior + 1e-4))
            item_copy["hybrid_score"] = float(adjusted_score)
            item_copy["score"] = float(adjusted_score)
            adapted.append(item_copy)

        adapted.sort(key=lambda x: x["score"], reverse=True)
        for r, act in enumerate(adapted, start=1):
            act["rank"] = r
        return adapted


# ==============================================================================
# Method E3: Full Combined Adapted Hybrid Model
# ==============================================================================

class FullAdaptedHybridModel(BaseRankingModel):
    """Combines Continuation-Boosted LGBM, Target-Adapted CBR, and Prior Shift."""

    def __init__(
        self,
        continuation_model: ContinuationBoostedRankerModel,
        adapted_cbr: TargetAdaptedCBRModel,
        action_priors: Dict[str, float],
        alpha: float = 0.75,
        beta: float = 0.4,
    ):
        self.continuation_model = continuation_model
        self.adapted_cbr = adapted_cbr
        self.action_priors = action_priors
        self.alpha = alpha
        self.beta = beta

    def rank_candidates(self, instance: AdaptationInstance) -> List[Dict[str, Any]]:
        cands = instance.candidate_actions
        lgbm_ranked = self.continuation_model.rank_candidates(instance)
        cbr_ranked = self.adapted_cbr.rank_candidates(instance)

        lgbm_score_map = {item["action_id"]: item["score"] for item in lgbm_ranked}
        cbr_score_map = {item["action_id"]: item["score"] for item in cbr_ranked}

        lgbm_arr = np.array([lgbm_score_map[c["action_id"]] for c in cands])
        cbr_arr = np.array([cbr_score_map[c["action_id"]] for c in cands])

        # Min-max normalization
        norm_lgbm = (lgbm_arr - np.min(lgbm_arr)) / (np.max(lgbm_arr) - np.min(lgbm_arr) + 1e-8)
        norm_cbr = (cbr_arr - np.min(cbr_arr)) / (np.max(cbr_arr) - np.min(cbr_arr) + 1e-8)

        fused = (self.alpha * norm_lgbm) + ((1.0 - self.alpha) * norm_cbr)

        # Prior shift
        if self.beta > 0.0 and self.action_priors:
            priors = np.array([self.action_priors.get(c["action_id"], 1e-4) for c in cands])
            fused = fused + (self.beta * np.log(priors + 1e-4))

        ranked = []
        for i, cand in enumerate(cands):
            ranked.append({
                "action_id": cand["action_id"],
                "action_name": cand.get("action_name", cand["action_id"]),
                "action_category": cand.get("action_category", "unknown"),
                "score": float(fused[i]),
            })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        for r, act in enumerate(ranked, start=1):
            act["rank"] = r
        return ranked


# ==============================================================================
# Method F: Simple Baselines
# ==============================================================================

class RandomRankingPolicy(BaseRankingModel):
    """Uniformly random candidate permutation."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

    def rank_candidates(self, instance: AdaptationInstance) -> List[Dict[str, Any]]:
        cands = list(instance.candidate_actions)
        self.rng.shuffle(cands)
        ranked = []
        for r, cand in enumerate(cands, start=1):
            ranked.append({
                "action_id": cand["action_id"],
                "action_name": cand.get("action_name", cand["action_id"]),
                "action_category": cand.get("action_category", "unknown"),
                "score": float(1.0 - (r * 0.04)),
                "rank": r,
            })
        return ranked


class GlobalMajorityPolicy(BaseRankingModel):
    """Ranks candidates by historical frequency in the adaptation subset."""

    def __init__(self, action_counts: Dict[str, int]):
        self.action_counts = action_counts

    def rank_candidates(self, instance: AdaptationInstance) -> List[Dict[str, Any]]:
        cands = list(instance.candidate_actions)
        ranked = sorted(cands, key=lambda c: self.action_counts.get(c["action_id"], 0), reverse=True)
        res = []
        for r, cand in enumerate(ranked, start=1):
            res.append({
                "action_id": cand["action_id"],
                "action_name": cand.get("action_name", cand["action_id"]),
                "action_category": cand.get("action_category", "unknown"),
                "score": float(self.action_counts.get(cand["action_id"], 0)),
                "rank": r,
            })
        return res


class ComponentMajorityPolicy(BaseRankingModel):
    """Ranks candidates conditioned on the diagnosed component from decision state."""

    def __init__(self, comp_counts: Dict[str, Dict[str, int]], global_counts: Dict[str, int]):
        self.comp_counts = comp_counts
        self.global_counts = global_counts

    def rank_candidates(self, instance: AdaptationInstance) -> List[Dict[str, Any]]:
        cands = list(instance.candidate_actions)
        # Determine component from input state
        comp = "bearing"
        input_ms = instance.decision_state.get("input_machine_state", {})
        if "fluid" in input_ms or "pressure" in input_ms:
            comp = "seal"
        elif "thermal" in input_ms and "mechanical" not in input_ms:
            comp = "cooling"

        action_freqs = self.comp_counts.get(comp, self.global_counts)
        ranked = sorted(cands, key=lambda c: action_freqs.get(c["action_id"], self.global_counts.get(c["action_id"], 0)), reverse=True)
        res = []
        for r, cand in enumerate(ranked, start=1):
            res.append({
                "action_id": cand["action_id"],
                "action_name": cand.get("action_name", cand["action_id"]),
                "action_category": cand.get("action_category", "unknown"),
                "score": float(action_freqs.get(cand["action_id"], self.global_counts.get(cand["action_id"], 0))),
                "rank": r,
            })
        return res


class SeverityHeuristicPolicy(BaseRankingModel):
    """Expert rule-based lookup based on health index and decision severity."""

    def rank_candidates(self, instance: AdaptationInstance) -> List[Dict[str, Any]]:
        sev = str(instance.decision_state.get("decision_severity", "WATCH")).upper()
        if sev == "CRITICAL":
            primary_act = "ACT_REPL_BEARING"
        elif sev == "DEGRADING":
            primary_act = "ACT_CORR_LUBRICATE"
        else:
            primary_act = "ACT_MON_ENHANCED"

        cands = list(instance.candidate_actions)
        ranked = [c for c in cands if c["action_id"] == primary_act] + [c for c in cands if c["action_id"] != primary_act]
        res = []
        for r, cand in enumerate(ranked, start=1):
            res.append({
                "action_id": cand["action_id"],
                "action_name": cand.get("action_name", cand["action_id"]),
                "action_category": cand.get("action_category", "unknown"),
                "score": float(1.0 if cand["action_id"] == primary_act else 0.5 - (r * 0.02)),
                "rank": r,
            })
        return res
