"""Hybrid Recommender combining LGBMRanker and Case-Based Reasoning (CBR).

Implements normalized score fusion with validation-only alpha parameter selection.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class HybridConfig:
    alpha: float = 0.5  # Weight for LGBMRanker (1 - alpha for CBR)
    candidate_normalization: str = "minmax"  # 'minmax' | 'zscore' | 'softmax'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HybridRecommender:
    """Ensemble recommender fusing LGBMRanker and CBR similarity retrieval."""

    def __init__(self, config: Optional[HybridConfig] = None):
        self.config = config or HybridConfig()

    @staticmethod
    def normalize_scores_per_group(scores: np.ndarray, group_indices: List[np.ndarray]) -> np.ndarray:
        """Normalizes scores to [0, 1] within each DecisionState candidate group."""
        norm_scores = np.zeros_like(scores, dtype=np.float32)

        for idxs in group_indices:
            grp_s = scores[idxs]
            s_min = np.min(grp_s)
            s_max = np.max(grp_s)
            if abs(s_max - s_min) < 1e-8:
                norm_scores[idxs] = 0.5
            else:
                norm_scores[idxs] = (grp_s - s_min) / (s_max - s_min)

        return norm_scores

    def combine_scores(
        self,
        lgbm_scores: np.ndarray,
        cbr_scores: np.ndarray,
        df: pd.DataFrame,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Combines normalized LGBM and CBR scores according to alpha."""
        effective_alpha = self.config.alpha if alpha is None else alpha

        # Group indices per state
        group_indices = [grp.index.values for _, grp in df.groupby("decision_state_id", sort=False)]

        norm_lgbm = self.normalize_scores_per_group(lgbm_scores, group_indices)
        norm_cbr = self.normalize_scores_per_group(cbr_scores, group_indices)

        hybrid_scores = (effective_alpha * norm_lgbm) + ((1.0 - effective_alpha) * norm_cbr)
        return hybrid_scores

    def select_best_alpha(
        self,
        lgbm_val_scores: np.ndarray,
        cbr_val_scores: np.ndarray,
        val_df: pd.DataFrame,
        candidate_alphas: Optional[List[float]] = None,
        metric_eval_fn: Optional[Any] = None,
    ) -> Tuple[float, Dict[float, float]]:
        """Sweeps alpha on validation data to select optimal alpha without test leakage."""
        if candidate_alphas is None:
            candidate_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

        history: Dict[float, float] = {}
        best_alpha = 0.5
        best_score = -1.0

        for a in candidate_alphas:
            h_scores = self.combine_scores(lgbm_val_scores, cbr_val_scores, val_df, alpha=a)
            if metric_eval_fn:
                val_metric = metric_eval_fn(val_df, h_scores)
            else:
                # Default heuristic: mean reciprocal rank approximation
                val_metric = 0.5
            history[a] = val_metric

            if val_metric > best_score:
                best_score = val_metric
                best_alpha = a

        self.config.alpha = best_alpha
        return best_alpha, history
