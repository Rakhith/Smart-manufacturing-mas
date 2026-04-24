"""
Synthetic Data Quality Analysis Module

This module provides comprehensive analysis and comparison between synthetic and original datasets,
including statistical measures, distribution analysis, and data quality metrics.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class SyntheticQualityAnalyzer:
    """Analyzes and compares synthetic data quality against original dataset."""

    def __init__(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame):
        """
        Initialize analyzer with original and synthetic datasets.

        Args:
            original_df: Original dataset (source for statistics)
            synthetic_df: Synthetic dataset (to be evaluated)
        """
        self.original_df = original_df.copy()
        self.synthetic_df = synthetic_df.copy()
        self.numeric_cols = self._get_numeric_columns()
        self.categorical_cols = self._get_categorical_columns()

    def _get_numeric_columns(self) -> List[str]:
        """Get list of numeric columns common to both datasets."""
        orig_numeric = self.original_df.select_dtypes(include=[np.number]).columns.tolist()
        synth_numeric = self.synthetic_df.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in orig_numeric if col in synth_numeric]

    def _get_categorical_columns(self) -> List[str]:
        """Get list of categorical columns common to both datasets."""
        orig_cat = self.original_df.select_dtypes(include=["object", "category"]).columns.tolist()
        synth_cat = self.synthetic_df.select_dtypes(include=["object", "category"]).columns.tolist()
        return [col for col in orig_cat if col in synth_cat]

    def compare_numeric_distributions(self) -> Dict[str, Any]:
        """
        Compare numeric column distributions between original and synthetic data.

        Returns:
            Dictionary with distribution metrics for each numeric column
        """
        comparison = {}

        for col in self.numeric_cols:
            orig_series = self.original_df[col].dropna()
            synth_series = self.synthetic_df[col].dropna()

            if len(orig_series) == 0 or len(synth_series) == 0:
                continue

            # Basic statistics
            orig_stats = {
                "mean": float(orig_series.mean()),
                "median": float(orig_series.median()),
                "std": float(orig_series.std()),
                "min": float(orig_series.min()),
                "max": float(orig_series.max()),
                "q25": float(orig_series.quantile(0.25)),
                "q75": float(orig_series.quantile(0.75)),
            }

            synth_stats = {
                "mean": float(synth_series.mean()),
                "median": float(synth_series.median()),
                "std": float(synth_series.std()),
                "min": float(synth_series.min()),
                "max": float(synth_series.max()),
                "q25": float(synth_series.quantile(0.25)),
                "q75": float(synth_series.quantile(0.75)),
            }

            # Calculate differences
            diff = {
                "mean_diff": float(abs(orig_stats["mean"] - synth_stats["mean"])),
                "mean_diff_pct": float(
                    abs(orig_stats["mean"] - synth_stats["mean"]) / (abs(orig_stats["mean"]) + 1e-10) * 100
                ),
                "std_diff": float(abs(orig_stats["std"] - synth_stats["std"])),
                "std_diff_pct": float(
                    abs(orig_stats["std"] - synth_stats["std"]) / (abs(orig_stats["std"]) + 1e-10) * 100
                ),
                "range_coverage": float(
                    min(synth_stats["max"], orig_stats["max"]) / max(orig_stats["max"], 1e-10)
                ),
            }

            # Kolmogorov-Smirnov test (0 = identical, 1 = completely different)
            ks_statistic, ks_pvalue = stats.ks_2samp(orig_series, synth_series)

            # Wasserstein distance (optimal transport distance)
            wasserstein = float(stats.wasserstein_distance(orig_series, synth_series))
            mean_scale = abs(orig_stats["mean"]) + orig_stats["std"] + 1e-10
            std_scale = orig_stats["std"] + 1e-10
            mean_similarity = float(np.exp(-diff["mean_diff"] / mean_scale))
            std_similarity = float(np.exp(-diff["std_diff"] / std_scale))
            ks_similarity = float(max(0, 1 - ks_statistic))

            comparison[col] = {
                "original": orig_stats,
                "synthetic": synth_stats,
                "differences": diff,
                "ks_statistic": float(ks_statistic),  # 0 = same dist, 1 = different
                "ks_pvalue": float(ks_pvalue),
                "wasserstein_distance": wasserstein,
                "ks_similarity": ks_similarity,
                "mean_similarity": mean_similarity,
                "std_similarity": std_similarity,
                "similarity_score": float(0.5 * ks_similarity + 0.25 * mean_similarity + 0.25 * std_similarity),
            }

        return comparison

    def compare_categorical_distributions(self) -> Dict[str, Any]:
        """
        Compare categorical column distributions between original and synthetic data.

        Returns:
            Dictionary with distribution metrics for each categorical column
        """
        comparison = {}

        for col in self.categorical_cols:
            orig_counts = self.original_df[col].value_counts(normalize=True, dropna=True)
            synth_counts = self.synthetic_df[col].value_counts(normalize=True, dropna=True)

            # Get all unique categories from both datasets
            all_categories = set(orig_counts.index) | set(synth_counts.index)

            orig_dist = {str(cat): float(orig_counts.get(cat, 0)) for cat in all_categories}
            synth_dist = {str(cat): float(synth_counts.get(cat, 0)) for cat in all_categories}

            # Chi-square test for categorical distribution
            orig_vals = [orig_counts.get(cat, 0) for cat in all_categories]
            synth_vals = [synth_counts.get(cat, 0) for cat in all_categories]

            # Normalize to counts
            orig_counts_total = self.original_df[col].value_counts(dropna=True)
            synth_counts_total = self.synthetic_df[col].value_counts(dropna=True)
            orig_counts_norm = [orig_counts_total.get(cat, 0) for cat in all_categories]
            synth_counts_norm = [synth_counts_total.get(cat, 0) for cat in all_categories]

            # Calculate Jensen-Shannon divergence
            js_div = float(self._jensen_shannon_divergence(orig_dist, synth_dist))

            comparison[col] = {
                "original_distribution": orig_dist,
                "synthetic_distribution": synth_dist,
                "categories_in_original": len(orig_counts),
                "categories_in_synthetic": len(synth_counts),
                "categories_coverage": float(len(set(orig_counts.index) & set(synth_counts.index)) / max(len(orig_counts), 1)),
                "jensen_shannon_divergence": js_div,  # 0 = identical, 1 = completely different
                "similarity_score": float(max(0, 1 - js_div)),  # Higher is better (0-1)
            }

        return comparison

    def _jensen_shannon_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """Calculate Jensen-Shannon divergence between two distributions."""
        all_keys = set(p.keys()) | set(q.keys())
        p_arr = np.array([p.get(k, 0) for k in all_keys])
        q_arr = np.array([q.get(k, 0) for k in all_keys])

        # Normalize
        p_arr = p_arr / (p_arr.sum() + 1e-10)
        q_arr = q_arr / (q_arr.sum() + 1e-10)

        m = 0.5 * (p_arr + q_arr)
        divergence = 0.5 * stats.entropy(p_arr, m) + 0.5 * stats.entropy(q_arr, m)
        return float(divergence)

    def compare_missing_values(self) -> Dict[str, Any]:
        """Compare missing value patterns between datasets."""
        comparison = {}

        for col in self.original_df.columns:
            if col not in self.synthetic_df.columns:
                continue

            orig_missing = float(self.original_df[col].isna().sum())
            synth_missing = float(self.synthetic_df[col].isna().sum())

            comparison[col] = {
                "original_missing_count": int(orig_missing),
                "original_missing_pct": float(orig_missing / len(self.original_df) * 100),
                "synthetic_missing_count": int(synth_missing),
                "synthetic_missing_pct": float(synth_missing / len(self.synthetic_df) * 100),
                "difference": float(abs(orig_missing / len(self.original_df) - synth_missing / len(self.synthetic_df)) * 100),
            }

        return comparison

    def detect_outliers_comparison(self) -> Dict[str, Any]:
        """Compare outlier patterns between datasets using IQR method."""
        comparison = {}

        for col in self.numeric_cols:
            orig_series = self.original_df[col].dropna()
            synth_series = self.synthetic_df[col].dropna()

            def count_outliers(series):
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((series < lower_bound) | (series > upper_bound)).sum()
                return int(outliers), float(outliers / len(series) * 100)

            orig_outliers, orig_outlier_pct = count_outliers(orig_series)
            synth_outliers, synth_outlier_pct = count_outliers(synth_series)

            comparison[col] = {
                "original_outlier_count": orig_outliers,
                "original_outlier_pct": orig_outlier_pct,
                "synthetic_outlier_count": synth_outliers,
                "synthetic_outlier_pct": synth_outlier_pct,
                "difference_pct": float(abs(orig_outlier_pct - synth_outlier_pct)),
            }

        return comparison

    def calculate_overall_quality_score(self) -> Dict[str, Any]:
        """
        Calculate an overall quality score for the synthetic dataset (0-100).

        Returns:
            Dictionary with overall metrics and recommendations
        """
        numeric_comp = self.compare_numeric_distributions()
        categorical_comp = self.compare_categorical_distributions()

        # Numeric similarity scores (average)
        numeric_scores = [m.get("similarity_score", 0) for m in numeric_comp.values()]
        numeric_avg = float(np.mean(numeric_scores)) if numeric_scores else 0.5

        # Categorical similarity scores (average)
        categorical_scores = [m.get("similarity_score", 0) for m in categorical_comp.values()]
        categorical_avg = float(np.mean(categorical_scores)) if categorical_scores else 0.5

        # Overall score (weighted average, more weight on numeric for manufacturing data)
        total_cols = len(self.numeric_cols) + len(self.categorical_cols)
        if total_cols == 0:
            overall_score = 50.0
        else:
            weight_numeric = len(self.numeric_cols) / total_cols
            weight_categorical = len(self.categorical_cols) / total_cols
            overall_score = (numeric_avg * weight_numeric + categorical_avg * weight_categorical) * 100

        # Classification
        if overall_score >= 85:
            quality_level = "Excellent"
            recommendation = "Synthetic data is highly representative of original distribution"
        elif overall_score >= 70:
            quality_level = "Good"
            recommendation = "Synthetic data captures main characteristics of original distribution"
        elif overall_score >= 50:
            quality_level = "Fair"
            recommendation = "Synthetic data has acceptable similarity but some differences exist"
        else:
            quality_level = "Poor"
            recommendation = "Synthetic data differs significantly from original distribution"

        missing_comp = self.compare_missing_values()
        missing_healthy = all(
            v["difference"] < 10 for v in missing_comp.values()
        )  # If difference is <10%, it's healthy

        outlier_comp = self.detect_outliers_comparison()
        outlier_healthy = all(
            v["difference_pct"] < 15 for v in outlier_comp.values()
        )  # If difference is <15%, it's healthy

        return {
            "overall_quality_score": float(overall_score),
            "quality_level": quality_level,
            "recommendation": recommendation,
            "numeric_similarity": float(numeric_avg),
            "categorical_similarity": float(categorical_avg),
            "missing_values_healthy": bool(missing_healthy),
            "outlier_pattern_healthy": bool(outlier_healthy),
            "numeric_columns_analyzed": len(self.numeric_cols),
            "categorical_columns_analyzed": len(self.categorical_cols),
            "total_original_rows": int(len(self.original_df)),
            "total_synthetic_rows": int(len(self.synthetic_df)),
        }

    def generate_full_report(self) -> Dict[str, Any]:
        """Generate complete analysis report."""
        return {
            "overall_quality": self.calculate_overall_quality_score(),
            "numeric_comparison": self.compare_numeric_distributions(),
            "categorical_comparison": self.compare_categorical_distributions(),
            "missing_values_comparison": self.compare_missing_values(),
            "outlier_comparison": self.detect_outliers_comparison(),
        }

    def get_summary_for_display(self) -> Dict[str, Any]:
        """Get a concise summary suitable for frontend display."""
        full_report = self.generate_full_report()
        overall = full_report["overall_quality"]

        # Top differences in numeric columns
        numeric_comp = full_report["numeric_comparison"]
        top_differences = sorted(
            [(col, m["differences"]["mean_diff_pct"]) for col, m in numeric_comp.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return {
            "quality_score": overall["overall_quality_score"],
            "quality_level": overall["quality_level"],
            "recommendation": overall["recommendation"],
            "summary_metrics": {
                "numeric_similarity": overall["numeric_similarity"],
                "categorical_similarity": overall["categorical_similarity"],
                "missing_values_healthy": overall["missing_values_healthy"],
                "outlier_pattern_healthy": overall["outlier_pattern_healthy"],
                "total_original_rows": overall["total_original_rows"],
                "total_synthetic_rows": overall["total_synthetic_rows"],
            },
            "top_column_differences": [
                {"column": col, "mean_difference_pct": diff} for col, diff in top_differences
            ],
            "health_indicators": {
                "missing_values": "✓ Healthy" if overall["missing_values_healthy"] else "⚠ Attention needed",
                "outlier_patterns": "✓ Healthy" if overall["outlier_pattern_healthy"] else "⚠ Attention needed",
                "numeric_distribution": "✓ Good" if overall["numeric_similarity"] > 0.7 else "⚠ Check differences",
                "categorical_distribution": "✓ Good" if overall["categorical_similarity"] > 0.7 else "⚠ Check differences",
            },
        }
