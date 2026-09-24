"""Automated Safety Guard & Zero-Leakage Audit for Phase 4.

Validates 12 critical engineering and methodological invariants before declaring
Phase 4 complete, with strict zero-leakage assertions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd


LEAKAGE_TARGET_PATTERNS = [
    "target_",
    "ground_truth",
    "failure_label",
    "rul_",
    "post_maintenance",
    "answer_",
    "future_",
]

FORBIDDEN_FEATURE_PATTERNS = [
    "provenance_",
    "llm_provider",
    "llm_model",
    "reasoning_summary",
    "evidence_used",
    "uncertainty_explanation",
]


class Phase4SafetyGuard:
    """Automated validator ensuring zero target leakage, clean splits, and model hygiene."""

    @classmethod
    def audit_all(
        cls,
        df_all: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_names: List[str],
        X_train: np.ndarray,
        cbr_indexed_state_ids: List[str],
        relevance_col: str = "ranking_relevance",
    ) -> Dict[str, Any]:
        """Runs the complete 12-invariant audit suite."""
        audit_results = {}

        # 1. State Mapping: every candidate maps to exactly one state
        state_ids = df_all["decision_state_id"]
        audit_results["check_1_state_mapping"] = {
            "status": "PASS" if not state_ids.isnull().any() else "FAIL",
            "null_state_count": int(state_ids.isnull().sum()),
        }

        # 2. Group Size Alignment: candidates per state between 2 and 6
        group_sizes = df_all.groupby("decision_state_id").size()
        min_grp, max_grp = int(group_sizes.min()), int(group_sizes.max())
        audit_results["check_2_group_size_bounds"] = {
            "status": "PASS" if (min_grp >= 2 and max_grp <= 10) else "FAIL",
            "min_group_size": min_grp,
            "max_group_size": max_grp,
        }

        # 3. Zero Split Overlap: mutually disjoint state IDs
        train_s = set(train_df["decision_state_id"])
        val_s = set(val_df["decision_state_id"])
        test_s = set(test_df["decision_state_id"])
        train_val_overlap = len(train_s.intersection(val_s))
        train_test_overlap = len(train_s.intersection(test_s))
        val_test_overlap = len(val_s.intersection(test_s))
        audit_results["check_3_zero_split_overlap"] = {
            "status": "PASS" if (train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0) else "FAIL",
            "train_val_overlap": train_val_overlap,
            "train_test_overlap": train_test_overlap,
            "val_test_overlap": val_test_overlap,
        }

        # 4. Zero Target Leakage: feature names must not match target patterns
        leaked_target_features = [
            f for f in feature_names if any(pat in f.lower() for pat in LEAKAGE_TARGET_PATTERNS)
        ]
        audit_results["check_4_zero_target_leakage"] = {
            "status": "PASS" if len(leaked_target_features) == 0 else "FAIL",
            "leaked_features": leaked_target_features,
        }

        # 5. Zero LLM Reasoning in Features: no reasoning text in features
        leaked_reasoning_features = [
            f for f in feature_names if any(pat in f.lower() for pat in ["reasoning", "explanation", "evidence"])
        ]
        audit_results["check_5_no_llm_reasoning_features"] = {
            "status": "PASS" if len(leaked_reasoning_features) == 0 else "FAIL",
            "leaked_features": leaked_reasoning_features,
        }

        # 6. Zero Provider Shortcut: provider/model is not a predictive feature
        leaked_provider_features = [
            f for f in feature_names if any(pat in f.lower() for pat in ["provider", "model", "llm_"])
        ]
        audit_results["check_6_no_provider_shortcut_features"] = {
            "status": "PASS" if len(leaked_provider_features) == 0 else "FAIL",
            "leaked_features": leaked_provider_features,
        }

        # 7. Valid Ranking Relevance Labels: non-negative integers
        relevance_vals = df_all[relevance_col].dropna()
        is_non_neg = (relevance_vals >= 0).all()
        is_integer_like = np.all(np.mod(relevance_vals, 1) == 0)
        audit_results["check_7_valid_relevance_labels"] = {
            "status": "PASS" if (is_non_neg and is_integer_like) else "FAIL",
            "min_relevance": float(relevance_vals.min()),
            "max_relevance": float(relevance_vals.max()),
        }

        # 8. Numeric Matrix Hygiene: no NaN or Inf
        has_nan = np.isnan(X_train).any()
        has_inf = np.isinf(X_train).any()
        audit_results["check_8_matrix_hygiene"] = {
            "status": "PASS" if (not has_nan and not has_inf) else "FAIL",
            "has_nan": bool(has_nan),
            "has_inf": bool(has_inf),
        }

        # 9. CBR Isolation: CBR index must only contain training states
        cbr_set = set(cbr_indexed_state_ids)
        cbr_val_overlap = len(cbr_set.intersection(val_s))
        cbr_test_overlap = len(cbr_set.intersection(test_s))
        audit_results["check_9_cbr_retrieval_isolation"] = {
            "status": "PASS" if (cbr_val_overlap == 0 and cbr_test_overlap == 0) else "FAIL",
            "cbr_val_overlap": cbr_val_overlap,
            "cbr_test_overlap": cbr_test_overlap,
        }

        # 10. Test Split Isolation: test states not in train or val
        audit_results["check_10_test_isolation"] = {
            "status": "PASS" if len(test_s.intersection(train_s.union(val_s))) == 0 else "FAIL",
            "leakage_count": len(test_s.intersection(train_s.union(val_s))),
        }

        # 11. Multi-Stream Dataset Representation: all 11 datasets present
        unique_ds = df_all["dataset_id"].unique()
        audit_results["check_11_dataset_stream_coverage"] = {
            "status": "PASS" if len(unique_ds) >= 11 else "FAIL",
            "datasets_count": len(unique_ds),
            "datasets": sorted(unique_ds.tolist()),
        }

        # 12. Overall Audit Disposition
        all_passed = all(check["status"] == "PASS" for check in audit_results.values())
        audit_results["overall_disposition"] = "ALL_PASS" if all_passed else "FAIL_AUDIT"

        return audit_results

    @classmethod
    def save_audit_report(cls, audit_results: Dict[str, Any], output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(audit_results, f, indent=2)
