"""Baseline Policies for Phase 5 Prescriptive Maintenance Benchmark."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np


class NoMaintenancePolicy:
    """Baseline 1: Passive monitoring only, never intervenes proactively."""

    def rank_candidates(
        self,
        decision_state: Dict[str, Any],
        candidate_actions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ranked = []
        for cand in candidate_actions:
            aid = cand["action_id"]
            score = 1.0 if aid == "ACT_MON_CONTINUE" else 0.0
            ranked.append({
                "action_id": aid,
                "action_name": cand.get("action_name", aid),
                "action_category": cand.get("action_category", "monitoring_observation"),
                "hybrid_score": score,
                "lgbm_score": score,
                "cbr_score": score,
            })
        ranked.sort(key=lambda item: item["hybrid_score"], reverse=True)
        for r, act in enumerate(ranked, start=1):
            act["rank"] = r
        return ranked


class SeverityHeuristicPolicy:
    """Baseline 2: Hard-coded severity threshold heuristic."""

    def rank_candidates(
        self,
        decision_state: Dict[str, Any],
        candidate_actions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        sev = str(decision_state.get("decision_severity", "WATCH")).upper()
        preferred_aid = "ACT_MON_CONTINUE"
        if sev == "WATCH":
            preferred_aid = "ACT_MON_ENHANCED"
        elif sev == "DEGRADING":
            preferred_aid = "ACT_CORR_LUBRICATE"
        elif sev == "CRITICAL":
            preferred_aid = "ACT_REPL_BEARING"

        ranked = []
        for cand in candidate_actions:
            aid = cand["action_id"]
            if aid == preferred_aid:
                score = 1.0
            elif "MON" in aid and sev in {"HEALTHY", "WATCH"}:
                score = 0.5
            elif "CORR" in aid and sev == "DEGRADING":
                score = 0.5
            elif "REPL" in aid and sev == "CRITICAL":
                score = 0.5
            else:
                score = 0.1

            ranked.append({
                "action_id": aid,
                "action_name": cand.get("action_name", aid),
                "action_category": cand.get("action_category", "monitoring_observation"),
                "hybrid_score": score,
                "lgbm_score": score,
                "cbr_score": score,
            })

        ranked.sort(key=lambda item: item["hybrid_score"], reverse=True)
        for r, act in enumerate(ranked, start=1):
            act["rank"] = r
        return ranked


class RandomRankingPolicy:
    """Baseline 3: Uniformly random candidate permutation."""

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)

    def rank_candidates(
        self,
        decision_state: Dict[str, Any],
        candidate_actions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ranked = []
        scores = self.rng.uniform(0.0, 1.0, size=len(candidate_actions))
        for i, cand in enumerate(candidate_actions):
            ranked.append({
                "action_id": cand["action_id"],
                "action_name": cand.get("action_name", cand["action_id"]),
                "action_category": cand.get("action_category", "monitoring_observation"),
                "hybrid_score": float(scores[i]),
                "lgbm_score": float(scores[i]),
                "cbr_score": float(scores[i]),
            })
        ranked.sort(key=lambda item: item["hybrid_score"], reverse=True)
        for r, act in enumerate(ranked, start=1):
            act["rank"] = r
        return ranked
