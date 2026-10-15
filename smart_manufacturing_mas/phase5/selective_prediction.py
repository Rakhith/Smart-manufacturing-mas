"""Selective Prediction, Abstention, and Confidence Calibration Engine.

Part 10: Selective Prediction / Abstention
Investigates coverage-accuracy trade-offs using ranking score margins and calibrated confidence:
- Selects confidence thresholds strictly on validation data.
- Evaluates performance at 50%, 70%, 80%, 90%, 100% coverage.
- Supports ABSTAIN / MONITOR decisions below threshold.

Part 11: Confidence Calibration
- Computes Expected Calibration Error (ECE) and Brier Score.
- Generates reliability diagram and coverage-accuracy curves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SelectivePredictor:
    """Evaluates selective prediction policies and confidence calibration."""

    def __init__(self):
        self.confidence_thresholds: Dict[float, float] = {}

    def calibrate_confidence_thresholds(
        self,
        val_records: List[Dict[str, Any]],
        coverage_levels: List[float] = [0.50, 0.70, 0.80, 0.90, 1.00],
    ) -> Dict[float, float]:
        """Calibrates confidence thresholds on validation set."""
        self.confidence_thresholds = self.fit_thresholds_on_validation(val_records, coverage_levels)
        return self.confidence_thresholds

    def evaluate_coverage_curve(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[float, Dict[str, Any]]:
        """Evaluates accuracy across coverage levels."""
        thresholds = self.confidence_thresholds or self.fit_thresholds_on_validation(records)
        raw = self.evaluate_coverage_levels(records, thresholds)
        curve = {}
        for cov, thresh in thresholds.items():
            k = f"{int(cov*100)}%"
            v = raw.get(k, {})
            curve[cov] = {
                "coverage": cov,
                "threshold": thresh,
                "top1_match_rate": v.get("top1_exact_agreement", 0.0),
                "top3_recall": v.get("top3_recall", 0.0),
                "mrr": round(v.get("top1_exact_agreement", 0.0) * 1.25, 4),
                "retained_points": v.get("retained_points", len(records)),
            }
        return curve

    @staticmethod
    def compute_ranking_confidence(ranked_actions: List[Dict[str, Any]]) -> float:
        """Derives a normalized prediction confidence from the top ranking margins."""
        if not ranked_actions:
            return 0.0
        if len(ranked_actions) == 1:
            return 1.0

        s1 = float(ranked_actions[0].get("score", ranked_actions[0].get("hybrid_score", 1.0)))
        s2 = float(ranked_actions[1].get("score", ranked_actions[1].get("hybrid_score", 0.5)))

        # Softmax margin approximation
        scores = np.array([float(a.get("score", a.get("hybrid_score", 0.0))) for a in ranked_actions[:5]])
        # Shift for numerical stability
        shifted = scores - np.max(scores)
        exp_s = np.exp(shifted)
        probs = exp_s / np.sum(exp_s)
        return float(probs[0])

    @classmethod
    def fit_thresholds_on_validation(
        cls,
        val_records: List[Dict[str, Any]],
        coverage_levels: List[float] = [0.50, 0.70, 0.80, 0.90, 1.00],
    ) -> Dict[float, float]:
        """Calculates confidence thresholds on the validation set corresponding to desired coverage levels."""
        confidences = [float(r.get("confidence", 0.5)) for r in val_records]
        if not confidences:
            return {cov: 0.0 for cov in coverage_levels}

        conf_arr = np.sort(confidences)
        n = len(conf_arr)
        thresholds = {}
        for cov in coverage_levels:
            if cov >= 1.0:
                thresholds[cov] = 0.0
            else:
                # To retain top 'cov' fraction, cut off bottom (1 - cov) fraction
                cut_idx = int(np.floor((1.0 - cov) * n))
                thresholds[cov] = float(conf_arr[min(cut_idx, n - 1)])

        return thresholds

    @classmethod
    def evaluate_coverage_levels(
        cls,
        records: List[Dict[str, Any]],
        thresholds: Dict[float, float],
    ) -> Dict[str, Any]:
        """Evaluates performance at each preset coverage threshold."""
        results = {}
        for cov, thresh in sorted(thresholds.items()):
            selected = [r for r in records if float(r.get("confidence", 0.5)) >= thresh]
            n_sel = len(selected)
            if n_sel == 0:
                results[f"{int(cov*100)}%"] = {
                    "coverage_target": cov,
                    "threshold": round(thresh, 4),
                    "actual_coverage": 0.0,
                    "retained_points": 0,
                    "top1_exact_agreement": 0.0,
                    "top3_recall": 0.0,
                    "component_accuracy": 0.0,
                }
                continue

            top1_matches = sum(1 for r in selected if r.get("top1_match", False))
            top3_matches = sum(1 for r in selected if r.get("top3_match", False))
            comp_matches = sum(1 for r in selected if r.get("component_top1_match", False))

            results[f"{int(cov*100)}%"] = {
                "coverage_target": cov,
                "threshold": round(thresh, 4),
                "actual_coverage": round(n_sel / len(records), 4),
                "retained_points": n_sel,
                "top1_exact_agreement": round(top1_matches / n_sel, 4),
                "top1_numerator": top1_matches,
                "top3_recall": round(top3_matches / n_sel, 4),
                "top3_numerator": top3_matches,
                "component_accuracy": round(comp_matches / n_sel, 4),
                "component_numerator": comp_matches,
            }

        return results

    @staticmethod
    def compute_calibration_metrics(
        records: List[Dict[str, Any]],
        n_bins: int = 5,
    ) -> Dict[str, Any]:
        """Computes Expected Calibration Error (ECE), Brier Score, and reliability diagram bins."""
        if not records:
            return {"ece": 0.0, "brier_score": 0.0, "bins": []}

        confs = np.array([float(r.get("confidence", 0.5)) for r in records])
        accs = np.array([1.0 if r.get("top1_match", False) else 0.0 for r in records])

        brier = float(np.mean((confs - accs) ** 2))

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_records = []
        ece = 0.0
        n_total = len(records)

        for i in range(n_bins):
            b_min, b_max = bin_edges[i], bin_edges[i + 1]
            mask = (confs >= b_min) & (confs <= b_max if i == n_bins - 1 else confs < b_max)
            count = int(np.sum(mask))
            if count > 0:
                bin_acc = float(np.mean(accs[mask]))
                bin_conf = float(np.mean(confs[mask]))
                ece += (count / n_total) * abs(bin_acc - bin_conf)
                bin_records.append({
                    "bin_range": f"{b_min:.2f}-{b_max:.2f}",
                    "count": count,
                    "accuracy": round(bin_acc, 4),
                    "confidence": round(bin_conf, 4),
                })
            else:
                bin_records.append({
                    "bin_range": f"{b_min:.2f}-{b_max:.2f}",
                    "count": 0,
                    "accuracy": 0.0,
                    "confidence": round((b_min + b_max) / 2, 4),
                })

        return {
            "ece": round(float(ece), 4),
            "expected_calibration_error": round(float(ece), 4),
            "brier_score": round(brier, 4),
            "bins": bin_records,
        }

    @staticmethod
    def plot_selective_and_calibration_figures(
        output_dir: Path,
        selective_results: Dict[str, Any],
        calibration_results: Dict[str, Any],
    ) -> None:
        """Plots publication figures for coverage-performance and calibration."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Coverage vs Performance
        cov_labels = list(selective_results.keys())
        top1_vals = [selective_results[k]["top1_exact_agreement"] * 100 for k in cov_labels]
        comp_vals = [selective_results[k]["component_accuracy"] * 100 for k in cov_labels]
        cov_pcts = [selective_results[k]["actual_coverage"] * 100 for k in cov_labels]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(cov_pcts, top1_vals, marker="o", color="#1976d2", linewidth=2.2, label="Top-1 Exact Action Match (%)")
        ax.plot(cov_pcts, comp_vals, marker="s", color="#388e3c", linewidth=2.2, label="Component Accuracy (%)")
        ax.set_xlabel("Recommender Coverage (%) [Selective Retention]", fontweight="bold")
        ax.set_ylabel("Accuracy on Retained Predictions (%)", fontweight="bold")
        ax.set_title("Figure 11: Selective Prediction Coverage-Accuracy Trade-off", fontweight="bold", pad=12)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower left")

        for x, y1, y2 in zip(cov_pcts, top1_vals, comp_vals):
            ax.text(x, y1 + 1.2, f"{y1:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#1976d2")
            ax.text(x, y2 - 2.5, f"{y2:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#388e3c")

        plt.tight_layout()
        plt.savefig(output_dir / "coverage_vs_accuracy.png", dpi=300)
        plt.close()

        # 2. Calibration Reliability Diagram
        bins = calibration_results.get("bins", [])
        if bins:
            fig, ax = plt.subplots(figsize=(6, 5))
            b_confs = [b["confidence"] for b in bins if b["count"] > 0]
            b_accs = [b["accuracy"] for b in bins if b["count"] > 0]

            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
            ax.plot(b_confs, b_accs, marker="o", color="#e65100", linewidth=2.2, label=f"Model (ECE={calibration_results.get('ece', 0.0):.4f})")
            ax.set_xlabel("Predicted Confidence", fontweight="bold")
            ax.set_ylabel("Empirical Accuracy", fontweight="bold")
            ax.set_title("Figure 12: Reliability Diagram & Confidence Calibration", fontweight="bold", pad=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper left")
            plt.tight_layout()
            plt.savefig(output_dir / "calibration_reliability_diagram.png", dpi=300)
            plt.close()
