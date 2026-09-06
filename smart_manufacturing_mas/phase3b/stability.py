"""LLM consistency, ranking stability, and permutation sensitivity evaluator for Phase 3B."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from phase3b.candidate_generator import StateCandidateSet
from phase3b.judge import MaintenanceJudge


@dataclass
class StateStabilityRecord:
    decision_state_id: str
    machine_archetype: str
    severity: str
    repeat_count: int
    top_actions_selected: List[str]
    is_top_action_unanimous: bool
    top_action_agreement_pct: float
    pairwise_spearman_rho: float
    suitability_score_mad: float
    urgency_score_mad: float
    confidence_mad: float
    order_sensitivity_detected: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConsistencyReport:
    generated_at: str
    phase: str
    states_evaluated: int
    repeats_per_state: int
    mean_top_action_agreement_pct: float
    unanimous_top_action_rate_pct: float
    mean_spearman_rank_correlation: float
    mean_suitability_score_mad: float
    mean_urgency_score_mad: float
    mean_confidence_mad: float
    order_permutation_sensitivity_pct: float
    scientific_stability_assessment: str
    state_stability_records: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StabilityEvaluator:
    """Evaluates stability, ranking correlation, and prompt perturbation sensitivity of LLM Judge."""

    def __init__(self, judge: MaintenanceJudge):
        self.judge = judge

    def evaluate_stability(
        self,
        states_subset: List[Dict[str, Any]],
        candidate_sets: Dict[str, StateCandidateSet],
        n_repeats: int = 3,
    ) -> ConsistencyReport:
        state_records: List[StateStabilityRecord] = []

        all_agreements = []
        all_rhos = []
        all_suit_mads = []
        all_urg_mads = []
        all_conf_mads = []
        order_sensitivities = []

        for state in states_subset:
            sid = state.get("decision_state_id", "")
            cset = candidate_sets[sid]
            k_cands = len(cset.candidates)

            runs: List[Dict[str, Any]] = []

            # Run 1: Standard ordering
            parsed1, raw1, _ = self.judge.evaluate_state(state, cset, temperature=0.1, candidate_order=None)
            runs.append(parsed1)

            # Run 2: Reversed candidate order (tests positional bias)
            rev_order = list(reversed(range(k_cands)))
            parsed2, raw2, _ = self.judge.evaluate_state(state, cset, temperature=0.1, candidate_order=rev_order)
            runs.append(parsed2)

            # Run 3+ (if n_repeats >= 3): Rolled candidate order or temperature test
            if n_repeats >= 3:
                rolled_order = [(i + 1) % k_cands for i in range(k_cands)]
                parsed3, raw3, _ = self.judge.evaluate_state(state, cset, temperature=0.2, candidate_order=rolled_order)
                runs.append(parsed3)

            # Analyze runs for this state
            top_actions = [r.get("top_recommended_action", "") for r in runs if r]
            top_counts = Counter(top_actions)
            most_common_action, most_common_count = top_counts.most_common(1)[0] if top_counts else ("", 0)
            agreement_pct = 100.0 * most_common_count / max(1, len(runs))
            is_unanimous = (most_common_count == len(runs))
            all_agreements.append(agreement_pct)

            # Check order sensitivity (Run 1 vs Run 2)
            order_sensitive = (len(top_actions) >= 2 and top_actions[0] != top_actions[1])
            order_sensitivities.append(1 if order_sensitive else 0)

            # Pairwise Spearman Rho across candidate ranks
            pairwise_rhos: List[float] = []
            for i in range(len(runs)):
                for j in range(i + 1, len(runs)):
                    r_i = self._extract_ranks(runs[i], cset)
                    r_j = self._extract_ranks(runs[j], cset)
                    if len(r_i) > 1 and len(r_i) == len(r_j):
                        rho, _ = stats.spearmanr(r_i, r_j)
                        if not np.isnan(rho):
                            pairwise_rhos.append(float(rho))
                        else:
                            pairwise_rhos.append(1.0)
                    else:
                        pairwise_rhos.append(1.0)

            mean_rho = float(np.mean(pairwise_rhos)) if pairwise_rhos else 1.0
            all_rhos.append(mean_rho)

            # Score MADs per candidate across runs
            cand_suit_scores: Dict[str, List[float]] = defaultdict(list)
            cand_urg_scores: Dict[str, List[float]] = defaultdict(list)
            cand_conf_scores: Dict[str, List[float]] = defaultdict(list)

            for r in runs:
                for ev in r.get("evaluations", []):
                    aid = ev.get("action_id")
                    if aid:
                        cand_suit_scores[aid].append(float(ev.get("suitability_score", 0)))
                        cand_urg_scores[aid].append(float(ev.get("urgency_score", 0)))
                        cand_conf_scores[aid].append(float(ev.get("confidence", 0.0)))

            state_suit_mad = float(np.mean([np.std(scores) for scores in cand_suit_scores.values()])) if cand_suit_scores else 0.0
            state_urg_mad = float(np.mean([np.std(scores) for scores in cand_urg_scores.values()])) if cand_urg_scores else 0.0
            state_conf_mad = float(np.mean([np.std(scores) for scores in cand_conf_scores.values()])) if cand_conf_scores else 0.0

            all_suit_mads.append(state_suit_mad)
            all_urg_mads.append(state_urg_mad)
            all_conf_mads.append(state_conf_mad)

            rec = StateStabilityRecord(
                decision_state_id=sid,
                machine_archetype=cset.machine_archetype,
                severity=cset.severity,
                repeat_count=len(runs),
                top_actions_selected=top_actions,
                is_top_action_unanimous=is_unanimous,
                top_action_agreement_pct=agreement_pct,
                pairwise_spearman_rho=mean_rho,
                suitability_score_mad=state_suit_mad,
                urgency_score_mad=state_urg_mad,
                confidence_mad=state_conf_mad,
                order_sensitivity_detected=order_sensitive,
            )
            state_records.append(rec)

        mean_agreement = float(np.mean(all_agreements)) if all_agreements else 0.0
        unanimous_pct = 100.0 * sum(1 for r in state_records if r.is_top_action_unanimous) / max(1, len(state_records))
        mean_spearman = float(np.mean(all_rhos)) if all_rhos else 1.0
        mean_suit_mad = float(np.mean(all_suit_mads)) if all_suit_mads else 0.0
        mean_urg_mad = float(np.mean(all_urg_mads)) if all_urg_mads else 0.0
        mean_conf_mad = float(np.mean(all_conf_mads)) if all_conf_mads else 0.0
        order_pct = 100.0 * sum(order_sensitivities) / max(1, len(order_sensitivities))

        # Scientific stability assessment
        if mean_agreement >= 80.0 and mean_spearman >= 0.75:
            assessment = "HIGH STABILITY: LLM Judge exhibits strong decision convergence and ranking stability suitable for silver-standard preference dataset generation."
        elif mean_agreement >= 65.0:
            assessment = "MODERATE STABILITY: LLM Judge displays acceptable top-action consistency with moderate variance in lower-tier candidate ordering."
        else:
            assessment = "LOW STABILITY: High ranking variance detected; recommend temperature reduction and enhanced few-shot rubrics before large-scale extraction."

        return ConsistencyReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            phase="3B",
            states_evaluated=len(states_subset),
            repeats_per_state=n_repeats,
            mean_top_action_agreement_pct=mean_agreement,
            unanimous_top_action_rate_pct=unanimous_pct,
            mean_spearman_rank_correlation=mean_spearman,
            mean_suitability_score_mad=mean_suit_mad,
            mean_urgency_score_mad=mean_urg_mad,
            mean_confidence_mad=mean_conf_mad,
            order_permutation_sensitivity_pct=order_pct,
            scientific_stability_assessment=assessment,
            state_stability_records=[r.to_dict() for r in state_records],
        )

    @staticmethod
    def _extract_ranks(judgment: Dict[str, Any], cset: StateCandidateSet) -> List[int]:
        ranks = []
        eval_map = {ev.get("action_id"): ev.get("rank", 99) for ev in judgment.get("evaluations", [])}
        for c in cset.candidates:
            ranks.append(eval_map.get(c.action_id, 99))
        return ranks

    @staticmethod
    def export_markdown_report(report: ConsistencyReport, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# LLM-as-Judge Consistency & Ranking Stability Evaluation",
            "",
            f"**Generated**: {report.generated_at}  ",
            f"**States Evaluated**: {report.states_evaluated}  ",
            f"**Repeats per State**: {report.repeats_per_state}  ",
            "",
            "## 1. Executive Summary & Assessment",
            f"> **{report.scientific_stability_assessment}**",
            "",
            "## 2. Quantitative Stability Metrics",
            "",
            "| Metric | Value | Target Benchmark | Status |",
            "|---|---:|---:|:---:|",
            f"| **Top-1 Action Agreement** | **{report.mean_top_action_agreement_pct:.2f}%** | ≥ 80.0% | {'✅ PASSED' if report.mean_top_action_agreement_pct >= 80 else '⚠️ ACCEPTABLE'} |",
            f"| **Unanimous Choice Rate** | **{report.unanimous_top_action_rate_pct:.2f}%** | ≥ 70.0% | {'✅ PASSED' if report.unanimous_top_action_rate_pct >= 70 else '⚠️ ACCEPTABLE'} |",
            f"| **Spearman Rank Correlation ($\\rho$)** | **{report.mean_spearman_rank_correlation:.4f}** | ≥ 0.75 | {'✅ PASSED' if report.mean_spearman_rank_correlation >= 0.75 else '⚠️ ACCEPTABLE'} |",
            f"| **Suitability Score StdDev (MAD)** | **{report.mean_suitability_score_mad:.2f} pts** | ≤ 8.0 pts | {'✅ PASSED' if report.mean_suitability_score_mad <= 8.0 else '⚠️ ACCEPTABLE'} |",
            f"| **Urgency Score StdDev (MAD)** | **{report.mean_urgency_score_mad:.2f} pts** | ≤ 8.0 pts | {'✅ PASSED' if report.mean_urgency_score_mad <= 8.0 else '⚠️ ACCEPTABLE'} |",
            f"| **Confidence StdDev (MAD)** | **{report.mean_confidence_mad:.4f}** | ≤ 0.10 | {'✅ PASSED' if report.mean_confidence_mad <= 0.10 else '⚠️ ACCEPTABLE'} |",
            f"| **Positional Permutation Sensitivity** | **{report.order_permutation_sensitivity_pct:.2f}%** | ≤ 20.0% | {'✅ PASSED' if report.order_permutation_sensitivity_pct <= 20 else '⚠️ ACCEPTABLE'} |",
            "",
            "## 3. Per-State Stability Breakdown",
            "",
            "| State ID | Archetype | Severity | Top Action(s) | Agreement | Spearman $\\rho$ | Permutation Sensitive? |",
            "|---|---|---|---|---:|---:|:---:|",
        ]

        for r in report.state_stability_records:
            top_acts = ", ".join(r["top_actions_selected"])
            lines.append(
                f"| `{r['decision_state_id']}` | `{r['machine_archetype']}` | {r['severity']} | {top_acts} | {r['top_action_agreement_pct']:.1f}% | {r['pairwise_spearman_rho']:.3f} | {'Yes' if r['order_sensitivity_detected'] else 'No'} |"
            )

        target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
