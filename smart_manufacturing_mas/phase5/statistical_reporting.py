"""Statistical Bootstrap Reporting Engine (Part 19).

Computes non-parametric bootstrap confidence intervals (1,000 resamples),
standard errors, and paired significance tests for all ranking and event metrics.
Guarantees reproducible seeds and rigorous uncertainty quantification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class MetricBootstrapSummary:
    metric_name: str
    sample_size: int
    point_estimate: float
    mean: float
    std_error: float
    ci_lower_95: float
    ci_upper_95: float
    median: float
    iqr: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_latex_str(self, decimals: int = 4) -> str:
        r"""Returns LaTeX formatted string: point_est \pm std_error [ci_lower, ci_upper]."""
        fmt = f"{{:.{decimals}f}}"
        p = fmt.format(self.point_estimate)
        se = fmt.format(self.std_error)
        lo = fmt.format(self.ci_lower_95)
        hi = fmt.format(self.ci_upper_95)
        return f"{p} \\pm {se} [{lo}, {hi}]"


class BootstrapStatisticalReporter:
    """Computes bootstrap statistics and paired hypothesis tests for evaluation metrics."""

    def __init__(self, n_resamples: int = 1000, random_state: int = 42):
        self.n_resamples = n_resamples
        self.rng = np.random.RandomState(random_state)

    def bootstrap_metric(
        self,
        values: Union[List[float], np.ndarray],
        metric_name: str,
        statistic_fn: str = "mean",  # "mean" or "median"
    ) -> MetricBootstrapSummary:
        """Computes bootstrap distribution and 95% CI for a metric series."""
        arr = np.asarray(values, dtype=float)
        n = len(arr)
        if n == 0:
            return MetricBootstrapSummary(
                metric_name=metric_name,
                sample_size=0,
                point_estimate=0.0,
                mean=0.0,
                std_error=0.0,
                ci_lower_95=0.0,
                ci_upper_95=0.0,
                median=0.0,
                iqr=0.0,
            )

        if statistic_fn == "median":
            point_est = float(np.median(arr))
            fn = np.median
        else:
            point_est = float(np.mean(arr))
            fn = np.mean

        # Vectorized or sampled bootstrap
        indices = self.rng.choice(n, size=(self.n_resamples, n), replace=True)
        resampled_data = arr[indices]
        boot_stats = fn(resampled_data, axis=1)

        b_mean = float(np.mean(boot_stats))
        b_se = float(np.std(boot_stats, ddof=1)) if len(boot_stats) > 1 else 0.0
        ci_lo = float(np.percentile(boot_stats, 2.5))
        ci_hi = float(np.percentile(boot_stats, 97.5))
        med = float(np.median(boot_stats))
        q75, q25 = np.percentile(boot_stats, [75, 25])
        iqr = float(q75 - q25)

        return MetricBootstrapSummary(
            metric_name=metric_name,
            sample_size=n,
            point_estimate=round(point_est, 5),
            mean=round(b_mean, 5),
            std_error=round(b_se, 5),
            ci_lower_95=round(ci_lo, 5),
            ci_upper_95=round(ci_hi, 5),
            median=round(med, 5),
            iqr=round(iqr, 5),
        )

    def bootstrap_comparison(
        self,
        values_a: Union[List[float], np.ndarray],
        values_b: Union[List[float], np.ndarray],
        metric_name: str,
        model_a_name: str = "Model_A",
        model_b_name: str = "Model_B",
    ) -> Dict[str, Any]:
        """Paired bootstrap comparison (Difference: B - A).

        Returns estimated difference, 95% CI of the difference, and empirical p-value.
        """
        arr_a = np.asarray(values_a, dtype=float)
        arr_b = np.asarray(values_b, dtype=float)
        if len(arr_a) != len(arr_b) or len(arr_a) == 0:
            return {"error": "Sample sizes must be equal and non-zero for paired bootstrap"}

        n = len(arr_a)
        diff_orig = float(np.mean(arr_b) - np.mean(arr_a))

        indices = self.rng.choice(n, size=(self.n_resamples, n), replace=True)
        boot_diffs = np.mean(arr_b[indices], axis=1) - np.mean(arr_a[indices], axis=1)

        diff_mean = float(np.mean(boot_diffs))
        diff_se = float(np.std(boot_diffs, ddof=1))
        ci_lo = float(np.percentile(boot_diffs, 2.5))
        ci_hi = float(np.percentile(boot_diffs, 97.5))

        # Two-sided empirical p-value for H0: diff == 0
        p_val = float(2.0 * min(np.mean(boot_diffs <= 0.0), np.mean(boot_diffs >= 0.0)))
        p_val = min(max(p_val, 1.0 / self.n_resamples), 1.0)

        return {
            "metric_name": metric_name,
            "comparison": f"{model_b_name} vs {model_a_name}",
            "observed_difference": round(diff_orig, 5),
            "boot_difference_mean": round(diff_mean, 5),
            "difference_std_error": round(diff_se, 5),
            "ci_95_difference": [round(ci_lo, 5), round(ci_hi, 5)],
            "statistically_significant": bool(ci_lo > 0.0 or ci_hi < 0.0),
            "empirical_p_value": round(p_val, 4),
        }

    def compute_full_report(
        self,
        records: List[Dict[str, Any]],
        key_metrics: Optional[List[str]] = None,
    ) -> Dict[str, MetricBootstrapSummary]:
        """Runs bootstrap across all evaluation metrics provided in per-instance records."""
        if not records:
            return {}

        df = pd.DataFrame(records)
        default_metrics = [
            "top1_exact", "top3_recall", "mrr", "ndcg3", "ndcg5",
            "graded_relevance", "category_match", "lead_time_hours"
        ]
        metrics_to_eval = key_metrics or [c for c in default_metrics if c in df.columns]

        report = {}
        for m in metrics_to_eval:
            if m in df.columns and pd.api.types.is_numeric_dtype(df[m]):
                stat_fn = "median" if m == "lead_time_hours" else "mean"
                summary = self.bootstrap_metric(df[m].dropna().values, metric_name=m, statistic_fn=stat_fn)
                report[m] = summary

        return report
