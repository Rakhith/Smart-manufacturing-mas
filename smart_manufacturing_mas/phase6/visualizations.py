"""Publication-ready visualization generator for Phase 6 Low-Resource Cross-Domain Adaptation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Phase6Visualizer:
    """Generates high-resolution figures for data-efficiency curves, crossover points, and ablations."""

    def __init__(self, output_dir: Path = Path("outputs/phase6/figures")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Publication styling defaults
        plt.rcParams["font.sans-serif"] = "DejaVu Sans"
        plt.rcParams["axes.edgecolor"] = "#333333"
        plt.rcParams["axes.linewidth"] = 0.8

    def plot_domain_data_efficiency(
        self,
        df_results: pd.DataFrame,
        domain_id: str,
        title: str,
        filename: str,
    ) -> Path:
        """Plots data-efficiency learning curves (MRR, Top-3, Top-1) vs. adaptation fraction."""
        df_dom = df_results[df_results["domain_id"] == domain_id].copy()
        if df_dom.empty:
            return self.output_dir / filename

        # Methods to highlight
        methods_to_plot = {
            "Adapted_Hybrid": ("#1976d2", "-", "o", "Adapted Hybrid (Ours)"),
            "Target_Only": ("#d32f2f", "--", "s", "Target-Only from Scratch"),
            "Continuation_FT": ("#7b1fa2", "-.", "^", "Continuation Fine-Tuning"),
            "Residual_Correction": ("#00796b", ":", "v", "Residual Correction"),
            "Adapted_CBR": ("#e65100", "--", "d", "Target-Adapted CBR"),
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        # 1. Left subplot: MRR Curve
        zero_shot_mrr = df_dom[(df_dom["budget_fraction"] == 0.0) & (df_dom["method"] == "Zero_Shot")]["mrr"].values
        zs_val = zero_shot_mrr[0] if len(zero_shot_mrr) > 0 else 0.0

        ax1.axhline(zs_val, color="#555555", linestyle=":", linewidth=1.8, label=f"Zero-Shot Hybrid (MRR = {zs_val:.3f})")

        for m_name, (color, ls, marker, label) in methods_to_plot.items():
            sub = df_dom[df_dom["method"] == m_name].sort_values("budget_fraction")
            if not sub.empty:
                fracs = [f * 100 for f in sub["budget_fraction"]]
                ax1.plot(fracs, sub["mrr"], color=color, linestyle=ls, marker=marker, linewidth=2.0, label=label)

        ax1.set_xlabel("Local Adaptation Data Budget (%)", fontweight="bold", fontsize=11)
        ax1.set_ylabel("Mean Reciprocal Rank (MRR)", fontweight="bold", fontsize=11)
        ax1.set_title(f"A. Prescriptive Ranking Quality (MRR) vs. Data Budget", fontweight="bold", fontsize=12, pad=10)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="lower right", fontsize=9.5)

        # 2. Right subplot: Top-3 Recall Curve
        zero_shot_t3 = df_dom[(df_dom["budget_fraction"] == 0.0) & (df_dom["method"] == "Zero_Shot")]["top3_recall"].values
        zs_t3_val = zero_shot_t3[0] if len(zero_shot_t3) > 0 else 0.0

        ax2.axhline(zs_t3_val * 100, color="#555555", linestyle=":", linewidth=1.8, label=f"Zero-Shot Hybrid ({zs_t3_val*100:.1f}%)")

        for m_name, (color, ls, marker, label) in methods_to_plot.items():
            sub = df_dom[df_dom["method"] == m_name].sort_values("budget_fraction")
            if not sub.empty:
                fracs = [f * 100 for f in sub["budget_fraction"]]
                ax2.plot(fracs, sub["top3_recall"] * 100, color=color, linestyle=ls, marker=marker, linewidth=2.0, label=label)

        ax2.set_xlabel("Local Adaptation Data Budget (%)", fontweight="bold", fontsize=11)
        ax2.set_ylabel("Top-3 Action Recall (%)", fontweight="bold", fontsize=11)
        ax2.set_title(f"B. Top-3 Action Recall vs. Data Budget", fontweight="bold", fontsize=12, pad=10)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="lower right", fontsize=9.5)

        fig.suptitle(f"{title} (Data-Efficiency Learning Curve)", fontweight="bold", fontsize=14, y=1.02)
        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path

    def plot_crossover_analysis(self, df_results: pd.DataFrame, filename: str = "fig4_crossover_analysis.png") -> Path:
        """Plots Delta-MRR over Zero-Shot and Target-Only to illustrate crossover budgets."""
        domains = df_results["domain_id"].unique()
        fig, axes = plt.subplots(1, len(domains), figsize=(5.5 * len(domains), 4.8), dpi=300, sharey=True)
        if len(domains) == 1:
            axes = [axes]

        for ax, dom in zip(axes, domains):
            df_dom = df_results[df_results["domain_id"] == dom]
            zs_mrr = df_dom[(df_dom["budget_fraction"] == 0.0) & (df_dom["method"] == "Zero_Shot")]["mrr"].values[0]

            sub_adapt = df_dom[df_dom["method"] == "Adapted_Hybrid"].sort_values("budget_fraction")
            sub_tgt = df_dom[df_dom["method"] == "Target_Only"].sort_values("budget_fraction")

            fracs = [f * 100 for f in sub_adapt["budget_fraction"]]
            delta_zs = sub_adapt["mrr"].values - zs_mrr
            delta_tgt = sub_adapt["mrr"].values - sub_tgt["mrr"].values

            ax.axhline(0.0, color="#333333", linestyle="-", linewidth=1.0)
            ax.plot(fracs, delta_zs, color="#1976d2", marker="o", linewidth=2.2, label=r"$\Delta$ vs Zero-Shot")
            ax.plot(fracs, delta_tgt, color="#388e3c", marker="s", linewidth=2.2, label=r"$\Delta$ vs Target-Only")

            ax.set_xlabel("Adaptation Data Budget (%)", fontweight="bold")
            ax.set_title(f"{dom.replace('_', ' ').upper()}", fontweight="bold", fontsize=11)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="best", fontsize=9)

        axes[0].set_ylabel("Gain in MRR (Difference)", fontweight="bold")
        fig.suptitle("Observed Crossover Analysis: When Does Adaptation Outperform Baselines?", fontweight="bold", fontsize=13, y=1.02)
        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path

    def plot_method_comparison_bar(
        self,
        df_results: pd.DataFrame,
        filename: str = "fig5_method_comparison.png",
    ) -> Path:
        """Bar chart comparing all 11 methods at 100% budget across domains."""
        sub_100 = df_results[df_results["budget_fraction"] == 1.0].copy()
        domains = sub_100["domain_id"].unique()
        methods = [
            "Random", "Severity_Heuristic", "Global_Majority", "Component_Majority",
            "Frozen_Source", "Zero_Shot", "Target_Only", "Adapted_CBR",
            "Continuation_FT", "Residual_Correction", "Adapted_Hybrid", "Full_Adapted"
        ]

        # Filter methods present in results
        available_methods = [m for m in methods if m in sub_100["method"].unique()]

        fig, axes = plt.subplots(len(domains), 1, figsize=(12, 4.2 * len(domains)), dpi=300)
        if len(domains) == 1:
            axes = [axes]

        colors = [
            "#9e9e9e", "#ffb74d", "#ffd54f", "#aed581",
            "#90caf9", "#64b5f6", "#ef5350", "#ff8a65",
            "#ba68c8", "#4db6ac", "#1976d2", "#0d47a1"
        ][:len(available_methods)]

        for ax, dom in zip(axes, domains):
            dom_data = sub_100[sub_100["domain_id"] == dom]
            mrrs = []
            for m in available_methods:
                row = dom_data[dom_data["method"] == m]
                mrrs.append(row["mrr"].values[0] if not row.empty else 0.0)

            bars = ax.bar(available_methods, mrrs, color=colors, edgecolor="#333333", linewidth=0.8)
            ax.set_ylabel("MRR", fontweight="bold")
            ax.set_title(f"Method Comparison at 100% Adaptation Pool: {dom.replace('_', ' ').upper()}", fontweight="bold", fontsize=11)
            ax.set_ylim(0.0, max(mrrs) * 1.2 if mrrs else 1.0)
            ax.grid(True, axis="y", linestyle=":", alpha=0.6)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9.5)

            # Annotate bar values
            for bar, val in zip(bars, mrrs):
                ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.015, f"{val:.3f}", ha="center", fontsize=8.5, fontweight="bold")

        fig.suptitle("Comprehensive Policy Comparison Across Target Industrial Domains", fontweight="bold", fontsize=13, y=1.01)
        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path

    def plot_negative_transfer_analysis(
        self,
        df_results: pd.DataFrame,
        filename: str = "fig6_negative_transfer.png",
    ) -> Path:
        """Visualizes Top-1 Match vs. Top-3 Recall to analyze trade-offs and negative transfer phenomena."""
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

        domains = df_results["domain_id"].unique()
        markers = {"factory_sensors": "o", "oil040": "s", "cmapss_lodo": "^"}
        colors = {"Zero_Shot": "#757575", "Adapted_Hybrid": "#1976d2", "Target_Only": "#d32f2f", "Continuation_FT": "#7b1fa2"}

        for dom in domains:
            df_dom = df_results[(df_results["domain_id"] == dom) & (df_results["budget_fraction"] == 1.0)]
            m = markers.get(dom, "o")
            for meth, c in colors.items():
                row = df_dom[df_dom["method"] == meth]
                if not row.empty:
                    t1 = row["top1_exact_match"].values[0] * 100
                    t3 = row["top3_recall"].values[0] * 100
                    ax.scatter(t1, t3, color=c, marker=m, s=120, edgecolors="#222222", linewidth=1.2, label=f"{meth} ({dom})" if dom == domains[0] else "")
                    ax.text(t1 + 0.8, t3 - 0.5, f"{dom[:4]}-{meth[:4]}", fontsize=8)

        ax.set_xlabel("Top-1 Exact Action Agreement (%)", fontweight="bold", fontsize=11)
        ax.set_ylabel("Top-3 Action Recall (%)", fontweight="bold", fontsize=11)
        ax.set_title("Negative Transfer & Precision-Recall Trade-Off Landscape", fontweight="bold", fontsize=12, pad=12)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=9.5)

        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path
