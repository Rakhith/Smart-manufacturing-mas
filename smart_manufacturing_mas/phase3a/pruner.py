"""Redundancy elimination and medoid clustering for Phase 3A candidate decision states."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

from phase3a.schema import DecisionState


class StatePruner:
    def __init__(self, distance_threshold: float = 0.15, seed: int = 42):
        self.distance_threshold = distance_threshold
        self.seed = seed

    def _extract_feature_vector(self, state: DecisionState) -> Tuple[List[str], np.ndarray]:
        """Extracts a flat numeric vector from input_machine_state only (zero target leakage)."""
        keys = []
        vals = []
        for modality, features in sorted(state.input_machine_state.items()):
            for k, v in sorted(features.items()):
                if isinstance(v, (int, float, np.number)) and not np.isnan(v):
                    keys.append(f"{modality}__{k}")
                    vals.append(float(v))
        return keys, np.array(vals, dtype=float)

    def prune_stratum(self, states: List[DecisionState], target_max: int = 150) -> Tuple[List[DecisionState], int]:
        """
        Prunes near-identical states within a single stratum (dataset_id, severity)
        using distance-based leader clustering to select diverse medoid states.
        """
        if len(states) <= 1:
            return states, 0

        # Extract aligned vectors
        all_keys_set = set()
        state_vectors_raw = []
        for s in states:
            k, v = self._extract_feature_vector(s)
            all_keys_set.update(k)
            state_vectors_raw.append((s, dict(zip(k, v))))

        sorted_keys = sorted(list(all_keys_set))
        if not sorted_keys:
            # No numeric features to compare; fall back to diverse sampling
            return states[:target_max], max(0, len(states) - target_max)

        X = np.zeros((len(states), len(sorted_keys)), dtype=float)
        for i, (_, feat_dict) in enumerate(state_vectors_raw):
            for j, key in enumerate(sorted_keys):
                X[i, j] = feat_dict.get(key, 0.0)

        # Standardize features
        scaler = StandardScaler()
        try:
            X_scaled = scaler.fit_transform(X)
        except Exception:
            X_scaled = X

        # Compute normalized pairwise Euclidean distances
        dist_matrix = pairwise_distances(X_scaled, metric="euclidean")
        max_dist = dist_matrix.max()
        if max_dist > 0:
            dist_matrix = dist_matrix / max_dist

        # Leader / Medoid selection
        selected_indices: List[int] = []
        visited = np.zeros(len(states), dtype=bool)
        np.random.seed(self.seed)

        # Shuffle visit order deterministically
        order = np.random.permutation(len(states))

        for idx in order:
            if visited[idx]:
                continue

            selected_indices.append(idx)
            visited[idx] = True

            # Mark close neighbors within distance_threshold as redundant
            close_neighbors = np.where(dist_matrix[idx] < self.distance_threshold)[0]
            visited[close_neighbors] = True

            if len(selected_indices) >= target_max:
                break

        pruned_states = [states[i] for i in sorted(selected_indices)]
        removed_count = len(states) - len(pruned_states)
        return pruned_states, removed_count

    def prune_all(
        self, candidate_dict: Dict[str, List[DecisionState]], max_per_stratum: int = 80
    ) -> Tuple[Dict[str, List[DecisionState]], Dict[str, Any]]:
        """Prunes redundancy across all datasets partitioned by (dataset_id, severity)."""
        pruned_dict: Dict[str, List[DecisionState]] = {}
        pruning_stats: Dict[str, Any] = {
            "total_candidates": 0,
            "total_retained_after_pruning": 0,
            "total_redundant_removed": 0,
            "dataset_pruning_breakdown": {},
        }

        for d_id, candidates in candidate_dict.items():
            pruning_stats["total_candidates"] += len(candidates)
            dataset_retained = []
            dataset_removed = 0

            # Partition by severity
            severity_groups: Dict[str, List[DecisionState]] = {}
            for s in candidates:
                severity_groups.setdefault(s.decision_severity, []).append(s)

            for sev, sev_states in severity_groups.items():
                retained, removed = self.prune_stratum(sev_states, target_max=max_per_stratum)
                dataset_retained.extend(retained)
                dataset_removed += removed

            pruned_dict[d_id] = dataset_retained
            pruning_stats["total_retained_after_pruning"] += len(dataset_retained)
            pruning_stats["total_redundant_removed"] += dataset_removed
            pruning_stats["dataset_pruning_breakdown"][d_id] = {
                "candidates": len(candidates),
                "retained": len(dataset_retained),
                "redundant_removed": dataset_removed,
            }

        return pruned_dict, pruning_stats
