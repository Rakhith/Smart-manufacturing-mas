"""Data Splitters for Phase 4 Recommender.

Guarantees leakage-safe DecisionState-level grouping and stratified splitting across
industrial datasets and severity levels.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


@dataclass
class SplitManifest:
    train_state_ids: List[str]
    val_state_ids: List[str]
    test_state_ids: List[str]
    train_record_count: int
    val_record_count: int
    test_record_count: int
    random_seed: int
    stratification_keys: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class GroupedStratifiedSplitter:
    """Performs deterministic DecisionState-level grouped splitting stratified by dataset and severity."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitManifest]:
        """Splits DataFrame into train, val, and test partitions grouped by state."""
        # 1. Deduplicate to state-level metadata for stratification
        state_meta = (
            df[["decision_state_id", "dataset_id", "decision_severity"]]
            .drop_duplicates(subset=["decision_state_id"])
            .reset_index(drop=True)
        )

        # Create composite stratification label
        state_meta["strat_label"] = (
            state_meta["dataset_id"].astype(str) + "___" + state_meta["decision_severity"].astype(str)
        )

        # Handle singleton strata by merging or fallback
        strata_counts = state_meta["strat_label"].value_counts()
        rare_strata = strata_counts[strata_counts < 3].index
        if len(rare_strata) > 0:
            state_meta.loc[state_meta["strat_label"].isin(rare_strata), "strat_label"] = (
                state_meta.loc[state_meta["strat_label"].isin(rare_strata), "dataset_id"]
            )

        # Re-check rare strata
        strata_counts2 = state_meta["strat_label"].value_counts()
        rare_strata2 = strata_counts2[strata_counts2 < 2].index
        if len(rare_strata2) > 0:
            state_meta.loc[state_meta["strat_label"].isin(rare_strata2), "strat_label"] = "common"

        # 2. Split states: first train vs (val + test)
        eval_ratio = self.val_ratio + self.test_ratio
        train_states, eval_states = train_test_split(
            state_meta,
            test_size=eval_ratio,
            random_state=self.random_seed,
            stratify=state_meta["strat_label"],
        )

        # 3. Split eval into val and test (50/50 of eval_ratio if val_ratio == test_ratio)
        val_frac_of_eval = self.val_ratio / eval_ratio
        val_strat = eval_states["strat_label"]
        # Fallback if any eval stratum has < 2 samples
        if any(val_strat.value_counts() < 2):
            val_strat = None

        val_states, test_states = train_test_split(
            eval_states,
            test_size=(1.0 - val_frac_of_eval),
            random_state=self.random_seed,
            stratify=val_strat,
        )

        train_ids = set(train_states["decision_state_id"])
        val_ids = set(val_states["decision_state_id"])
        test_ids = set(test_states["decision_state_id"])

        # Invariant checks: zero overlap
        assert len(train_ids.intersection(val_ids)) == 0, "Train and Val state overlap detected!"
        assert len(train_ids.intersection(test_ids)) == 0, "Train and Test state overlap detected!"
        assert len(val_ids.intersection(test_ids)) == 0, "Val and Test state overlap detected!"

        # 4. Filter original DataFrame
        train_df = df[df["decision_state_id"].isin(train_ids)].copy().reset_index(drop=True)
        val_df = df[df["decision_state_id"].isin(val_ids)].copy().reset_index(drop=True)
        test_df = df[df["decision_state_id"].isin(test_ids)].copy().reset_index(drop=True)

        manifest = SplitManifest(
            train_state_ids=sorted(list(train_ids)),
            val_state_ids=sorted(list(val_ids)),
            test_state_ids=sorted(list(test_ids)),
            train_record_count=len(train_df),
            val_record_count=len(val_df),
            test_record_count=len(test_df),
            random_seed=self.random_seed,
            stratification_keys=["dataset_id", "decision_severity"],
        )

        return train_df, val_df, test_df, manifest


class LeaveOneDatasetOutSplitter:
    """Generates cross-domain generalization splits where one entire dataset is held out."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def generate_splits(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        datasets = sorted(df["dataset_id"].unique().tolist())
        splits = []

        for held_out_ds in datasets:
            held_out_mask = df["dataset_id"] == held_out_ds
            test_df = df[held_out_mask].copy().reset_index(drop=True)
            remaining_df = df[~held_out_mask].copy().reset_index(drop=True)

            # Split remaining into train (85%) and val (15%) for early stopping
            state_meta = (
                remaining_df[["decision_state_id", "dataset_id", "decision_severity"]]
                .drop_duplicates(subset=["decision_state_id"])
                .reset_index(drop=True)
            )

            strat = state_meta["dataset_id"].astype(str)
            train_states, val_states = train_test_split(
                state_meta,
                test_size=0.15,
                random_state=self.random_seed,
                stratify=strat if any(strat.value_counts() >= 2) else None,
            )

            train_ids = set(train_states["decision_state_id"])
            val_ids = set(val_states["decision_state_id"])

            train_df = remaining_df[remaining_df["decision_state_id"].isin(train_ids)].copy().reset_index(drop=True)
            val_df = remaining_df[remaining_df["decision_state_id"].isin(val_ids)].copy().reset_index(drop=True)

            splits.append({
                "held_out_dataset": held_out_ds,
                "train_df": train_df,
                "val_df": val_df,
                "test_df": test_df,
                "train_states_count": len(train_ids),
                "val_states_count": len(val_ids),
                "test_states_count": len(test_df["decision_state_id"].unique()),
            })

        return splits
