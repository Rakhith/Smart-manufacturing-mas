"""Chronological dataset partitioning and reproducible budget sampling for Phase 6."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from phase6.data_loader import AdaptationInstance


@dataclass
class SplitResult:
    """Container for chronologically partitioned instances."""
    train_pool: List[AdaptationInstance]
    val_pool: List[AdaptationInstance]
    test_pool: List[AdaptationInstance]


@dataclass
class BudgetSample:
    """Sampled adaptation subset for a given fraction and seed."""
    fraction: float
    seed: int
    instances: List[AdaptationInstance]
    num_states: int
    num_candidates: int
    num_assets: int
    unique_actions: List[str]


class ChronologicalSplitter:
    """Partitions instances chronologically into early adaptation pool, middle validation, and late test."""

    def __init__(
        self,
        train_ratio: float = 0.40,
        val_ratio: float = 0.30,
        test_ratio: float = 0.30,
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(self, instances: List[AdaptationInstance]) -> SplitResult:
        """Splits instances chronologically."""
        n = len(instances)
        n_train = int(np.floor(n * self.train_ratio))
        n_val = int(np.floor(n * self.val_ratio))

        train = instances[:n_train]
        val = instances[n_train : n_train + n_val]
        test = instances[n_train + n_val :]

        return SplitResult(train_pool=train, val_pool=val, test_pool=test)


class BudgetSampler:
    """Reproducibly samples local data subsets from the permitted adaptation pool."""

    @staticmethod
    def sample(
        train_pool: List[AdaptationInstance],
        fraction: float,
        seed: int = 42,
        sampling_mode: str = "chronological_prefix",  # 'chronological_prefix' | 'reproducible_stochastic'
    ) -> BudgetSample:
        """Samples an adaptation subset at the specified budget fraction.
        
        Args:
            train_pool: Permitted adaptation pool (must NOT include val or test instances).
            fraction: Float in [0.0, 1.0].
            seed: Random seed for stochastic sampling.
            sampling_mode: 'chronological_prefix' uses the earliest k instances;
                           'reproducible_stochastic' uses seed-controlled sampling for uncertainty analysis.
        """
        n_pool = len(train_pool)
        if fraction <= 0.0 or n_pool == 0:
            return BudgetSample(
                fraction=0.0,
                seed=seed,
                instances=[],
                num_states=0,
                num_candidates=0,
                num_assets=0,
                unique_actions=[],
            )

        k = max(1, int(np.ceil(n_pool * fraction)))
        k = min(k, n_pool)

        if sampling_mode == "chronological_prefix":
            sampled_instances = train_pool[:k]
        elif sampling_mode == "reproducible_stochastic":
            rng = np.random.RandomState(seed)
            indices = sorted(rng.choice(n_pool, size=k, replace=False).tolist())
            sampled_instances = [train_pool[idx] for idx in indices]
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")

        num_cands = sum(len(inst.candidate_actions) for inst in sampled_instances)
        num_assets = len(set(inst.asset_id for inst in sampled_instances))
        uniq_actions = sorted(list(set(inst.target_action for inst in sampled_instances)))

        return BudgetSample(
            fraction=fraction,
            seed=seed,
            instances=sampled_instances,
            num_states=len(sampled_instances),
            num_candidates=num_cands,
            num_assets=num_assets,
            unique_actions=uniq_actions,
        )


class SplitManifestBuilder:
    """Generates verifiable split provenance manifests with hashes and boundaries."""

    @staticmethod
    def _compute_sha256(data_str: str) -> str:
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

    @classmethod
    def build_domain_manifest(
        cls,
        domain_id: str,
        split_result: SplitResult,
        budget_fractions: List[float],
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Builds a JSON-serializable manifest for a domain split."""
        train_ids = [inst.instance_id for inst in split_result.train_pool]
        val_ids = [inst.instance_id for inst in split_result.val_pool]
        test_ids = [inst.instance_id for inst in split_result.test_pool]

        train_ts = [inst.prediction_timestamp for inst in split_result.train_pool]
        val_ts = [inst.prediction_timestamp for inst in split_result.val_pool]
        test_ts = [inst.prediction_timestamp for inst in split_result.test_pool]

        budgets_meta = {}
        for f in budget_fractions:
            b_sample = BudgetSampler.sample(split_result.train_pool, fraction=f, seed=seed)
            budgets_meta[f"{int(round(f * 100))}%"] = {
                "fraction": f,
                "num_states": b_sample.num_states,
                "num_candidates": b_sample.num_candidates,
                "num_assets": b_sample.num_assets,
                "unique_actions": b_sample.unique_actions,
                "instance_ids": [inst.instance_id for inst in b_sample.instances],
            }

        manifest = {
            "domain_id": domain_id,
            "total_instances": len(split_result.train_pool) + len(split_result.val_pool) + len(split_result.test_pool),
            "split_counts": {
                "train_pool": len(split_result.train_pool),
                "val_pool": len(split_result.val_pool),
                "test_pool": len(split_result.test_pool),
            },
            "temporal_boundaries": {
                "train_min": min(train_ts) if train_ts else "N/A",
                "train_max": max(train_ts) if train_ts else "N/A",
                "val_min": min(val_ts) if val_ts else "N/A",
                "val_max": max(val_ts) if val_ts else "N/A",
                "test_min": min(test_ts) if test_ts else "N/A",
                "test_max": max(test_ts) if test_ts else "N/A",
            },
            "hashes": {
                "train_pool_sha256": cls._compute_sha256(",".join(train_ids)),
                "val_pool_sha256": cls._compute_sha256(",".join(val_ids)),
                "test_pool_sha256": cls._compute_sha256(",".join(test_ids)),
            },
            "budgets": budgets_meta,
            "instance_ids": {
                "train_pool": train_ids,
                "val_pool": val_ids,
                "test_pool": test_ids,
            },
        }

        return manifest
