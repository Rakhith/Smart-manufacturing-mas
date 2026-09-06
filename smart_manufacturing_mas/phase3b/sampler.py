"""Stratified pilot sampler for selecting 100-200 balanced DecisionStates for Phase 3B."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PilotManifest:
    generated_at: str
    phase: str
    seed: int
    target_pilot_size: int
    total_corpus_size: int
    sampled_count: int
    dataset_distribution: Dict[str, int]
    archetype_distribution: Dict[str, int]
    severity_distribution: Dict[str, int]
    temporal_distribution: Dict[str, int]
    modality_distribution: Dict[str, int]
    stratum_allocation: Dict[str, int]
    sampled_decision_state_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PilotSampler:
    """Selects a balanced, reproducible pilot corpus across datasets, archetypes, severities, and temporal modes."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def sample_pilot_corpus(
        self,
        corpus: List[Dict[str, Any]],
        target_size: int = 150,
    ) -> Tuple[List[Dict[str, Any]], PilotManifest]:
        random.seed(self.seed)

        # Sort states deterministically by decision_state_id
        sorted_corpus = sorted(corpus, key=lambda s: s.get("decision_state_id", ""))

        # Group by composite stratum: (dataset_id, decision_severity)
        strata: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for s in sorted_corpus:
            dataset_id = s.get("dataset_id", "unknown")
            severity = s.get("decision_severity", "WATCH").upper()
            stratum_key = f"{dataset_id}__{severity}"
            strata[stratum_key].append(s)

        num_strata = len(strata)
        base_per_stratum = target_size // num_strata
        remainder = target_size % num_strata

        sampled_states: List[Dict[str, Any]] = []
        stratum_allocation: Dict[str, int] = {}

        # First pass: Allocate base_per_stratum (or min available in stratum)
        sorted_keys = sorted(list(strata.keys()))
        remaining_slots = target_size

        for key in sorted_keys:
            items = strata[key]
            # Deterministic shuffle within stratum
            shuffled_items = list(items)
            rng = random.Random(self.seed + hash(key) % 10000)
            rng.shuffle(shuffled_items)

            quota = min(len(shuffled_items), base_per_stratum)
            chosen = shuffled_items[:quota]
            sampled_states.extend(chosen)
            stratum_allocation[key] = len(chosen)
            remaining_slots -= len(chosen)

        # Second pass: distribute remaining slots round-robin across strata with available candidates
        idx = 0
        while remaining_slots > 0:
            key = sorted_keys[idx % len(sorted_keys)]
            items = strata[key]
            current_allocated = stratum_allocation[key]
            if current_allocated < len(items):
                rng = random.Random(self.seed + hash(key) % 10000)
                shuffled_items = list(items)
                rng.shuffle(shuffled_items)
                sampled_states.append(shuffled_items[current_allocated])
                stratum_allocation[key] += 1
                remaining_slots -= 1
            idx += 1
            if idx > len(sorted_keys) * 10:  # safety breakout
                break

        # Re-sort sampled states deterministically
        sampled_states = sorted(sampled_states, key=lambda s: s.get("decision_state_id", ""))

        # Distribution profiling
        dataset_dist = Counter(s.get("dataset_id", "unknown") for s in sampled_states)
        archetype_dist = Counter(
            s.get("asset_context", {}).get("machine_archetype", "unknown") for s in sampled_states
        )
        severity_dist = Counter(s.get("decision_severity", "unknown") for s in sampled_states)
        temporal_dist = Counter(
            s.get("asset_context", {}).get("temporal_type", "unknown") for s in sampled_states
        )

        modality_dist: Dict[str, int] = defaultdict(int)
        for s in sampled_states:
            for mod, feats in s.get("input_machine_state", {}).items():
                if feats:
                    modality_dist[mod] += 1

        manifest = PilotManifest(
            generated_at=datetime.now(timezone.utc).isoformat(),
            phase="3B",
            seed=self.seed,
            target_pilot_size=target_size,
            total_corpus_size=len(corpus),
            sampled_count=len(sampled_states),
            dataset_distribution=dict(dataset_dist),
            archetype_distribution=dict(archetype_dist),
            severity_distribution=dict(severity_dist),
            temporal_distribution=dict(temporal_dist),
            modality_distribution=dict(modality_dist),
            stratum_allocation=stratum_allocation,
            sampled_decision_state_ids=[s.get("decision_state_id", "") for s in sampled_states],
        )

        return sampled_states, manifest
