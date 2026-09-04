"""Weighted stratified representative sampling for Phase 3A decision states."""

from __future__ import annotations

import logging
from typing import Any, Dict, List
import numpy as np

from phase3a.schema import DecisionState, StateSeverity

# Complexity- and feature-aware weighting configuration:
# High-column unmapped datasets (e.g. Tennessee Eastman with 139 columns) are weighted lower;
# Multi-modal verified datasets with rich failure modes (AI4I, C-MAPSS, Hydraulic) are weighted proportionally higher.
DATASET_WEIGHTS: Dict[str, float] = {
    "ai4i_2020": 1.10,                     # CNC machining: 5 distinct failure modes + wear
    "cmapss": 1.15,                        # Turbofan: 4 sub-fleets FD001-FD004, full RUL arcs
    "smart_maintenance_static": 0.95,      # Factory fleet: static maintenance priority + cost
    "smart_maintenance_timeseries": 1.05,  # Continuous telemetry: anomalies + multi-class failure
    "iiot_6g": 1.00,                       # IIoT: network latency, defect rates, efficiency
    "metal_etch": 0.85,                    # Etch tool: 21 anonymous features, binary endpoint
    "metropt3": 1.05,                      # Train compressor: continuous pneumatic/thermal dynamics
    "uci_hydraulic": 1.15,                 # Hydraulic rig: 4 distinct component degradation targets
    "tennessee_eastman": 0.70,             # 139 cols, unmapped dictionary -> weighted lower
    "nasa_ims": 0.85,                      # Bearing test rig: vibration waveform harmonics
    "nasa_milling": 0.70,                  # Milling tool wear: signal statistics
}


class StratifiedSampler:
    def __init__(self, target_budget: int = 1100, seed: int = 42):
        self.target_budget = target_budget
        self.seed = seed
        self.weights = DATASET_WEIGHTS

    def compute_dataset_quotas(self, available_counts: Dict[str, int]) -> Dict[str, int]:
        """Calculates proportional quotas across datasets based on complexity weights."""
        total_weight = sum(self.weights.get(d_id, 1.0) for d_id in available_counts)
        quotas: Dict[str, int] = {}
        allocated = 0

        for d_id, count in available_counts.items():
            w = self.weights.get(d_id, 1.0)
            raw_quota = int(round((w / total_weight) * self.target_budget))
            # Quota cannot exceed available pruned states
            quota = min(count, max(10, raw_quota))
            quotas[d_id] = quota
            allocated += quota

        # Rebalance slight rounding differences
        diff = self.target_budget - allocated
        if diff != 0:
            for d_id in sorted(quotas.keys(), key=lambda k: self.weights.get(k, 1.0), reverse=(diff > 0)):
                if diff > 0 and quotas[d_id] < available_counts[d_id]:
                    quotas[d_id] += 1
                    diff -= 1
                elif diff < 0 and quotas[d_id] > 20:
                    quotas[d_id] -= 1
                    diff += 1
                if diff == 0:
                    break

        return quotas

    def sample_dataset(self, states: List[DecisionState], quota: int) -> List[DecisionState]:
        """Samples states within a single dataset stratified by decision severity."""
        if len(states) <= quota:
            return states

        # Group by severity
        sev_groups: Dict[str, List[DecisionState]] = {}
        for s in states:
            sev_groups.setdefault(s.decision_severity, []).append(s)

        num_tiers = len(sev_groups)
        base_per_tier = quota // num_tiers
        remainder = quota % num_tiers

        selected: List[DecisionState] = []
        np.random.seed(self.seed)

        # Distribute quota across severity tiers
        for i, (sev, tier_states) in enumerate(sorted(sev_groups.items())):
            tier_quota = base_per_tier + (1 if i < remainder else 0)
            if len(tier_states) <= tier_quota:
                selected.extend(tier_states)
            else:
                indices = np.random.choice(len(tier_states), size=tier_quota, replace=False)
                selected.extend([tier_states[idx] for idx in sorted(indices)])

        # If any tier was under-represented, backfill from remaining
        if len(selected) < quota:
            selected_ids = set(s.decision_state_id for s in selected)
            remaining = [s for s in states if s.decision_state_id not in selected_ids]
            needed = quota - len(selected)
            if remaining:
                extra_indices = np.random.choice(len(remaining), size=min(needed, len(remaining)), replace=False)
                selected.extend([remaining[idx] for idx in sorted(extra_indices)])

        return selected

    def sample_all(self, pruned_dict: Dict[str, List[DecisionState]]) -> Tuple[List[DecisionState], Dict[str, Any]]:
        """Applies weighted stratified sampling across all datasets to form the final corpus."""
        available_counts = {d_id: len(states) for d_id, states in pruned_dict.items()}
        quotas = self.compute_dataset_quotas(available_counts)

        final_corpus: List[DecisionState] = []
        sampling_breakdown: Dict[str, Any] = {}

        for d_id, states in pruned_dict.items():
            quota = quotas[d_id]
            dataset_selected = self.sample_dataset(states, quota)
            final_corpus.extend(dataset_selected)

            # Record severity distribution
            sev_dist: Dict[str, int] = {}
            for s in dataset_selected:
                sev_dist[s.decision_severity] = sev_dist.get(s.decision_severity, 0) + 1

            sampling_breakdown[d_id] = {
                "available": len(states),
                "quota": quota,
                "selected": len(dataset_selected),
                "weight": self.weights.get(d_id, 1.0),
                "severity_distribution": sev_dist,
            }

        stats = {
            "target_budget": self.target_budget,
            "total_selected": len(final_corpus),
            "datasets_count": len(pruned_dict),
            "breakdown": sampling_breakdown,
        }

        return final_corpus, stats
