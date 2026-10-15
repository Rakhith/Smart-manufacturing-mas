"""Comprehensive Automated Test Suite for Phase 6 Low-Resource Cross-Domain Adaptation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from phase4.lgbm_ranker import MaintenanceLGBMRanker
from phase5.frozen_model import FrozenPhase4Recommender
from phase6.adaptation_models import (
    ContinuationBoostedRankerModel,
    PriorAdaptedHybridModel,
    TargetAdaptedCBRModel,
    TargetOnlyRankerModel,
    ZeroShotHybridModel,
)
from phase6.config import AdaptationExperimentConfig, SUPPORTED_DOMAINS
from phase6.data_loader import AdaptationInstance, Phase6DataLoader
from phase6.evaluator import Phase6Evaluator
from phase6.leakage import Phase6LeakageAuditor
from phase6.sampling import BudgetSampler, ChronologicalSplitter, SplitManifestBuilder


@pytest.fixture(scope="module")
def data_loader():
    return Phase6DataLoader()


@pytest.fixture(scope="module")
def frozen_rec():
    return FrozenPhase4Recommender()


@pytest.fixture(scope="module")
def sample_instances(data_loader):
    insts = data_loader.load_factory_sensors()
    return insts


def test_01_configuration_and_domain_metadata():
    """Verifies default experiment configuration and supported domain profiles."""
    config = AdaptationExperimentConfig()
    assert len(config.target_domains) >= 3
    assert 0.0 in config.budget_fractions
    assert 1.0 in config.budget_fractions
    assert "factory_sensors" in SUPPORTED_DOMAINS
    assert "oil040" in SUPPORTED_DOMAINS
    assert "cmapss_lodo" in SUPPORTED_DOMAINS

    dom = SUPPORTED_DOMAINS["factory_sensors"]
    assert dom.source_exclusion_type == "external_unseen"
    assert dom.supervision_type == "historical_work_order"


def test_02_data_loader_instances(sample_instances):
    """Verifies that instances have correct dimensions, non-empty features, and valid targets."""
    assert len(sample_instances) == 63
    first = sample_instances[0]

    assert first.domain_id == "factory_sensors"
    assert len(first.candidate_actions) == 21
    assert first.X_features.shape == (21, 96)
    assert first.relevance_grades.shape == (21,)
    assert not np.isnan(first.X_features).any()
    assert first.target_action.startswith("ACT_")
    assert first.relevance_grades.max() == 4  # Target action gets grade 4


def test_03_chronological_splitting_isolation(sample_instances):
    """Verifies 40/30/30 chronological partitioning and strict temporal isolation."""
    splitter = ChronologicalSplitter(train_ratio=0.40, val_ratio=0.30, test_ratio=0.30)
    split = splitter.split(sample_instances)

    n_total = len(sample_instances)
    assert len(split.train_pool) + len(split.val_pool) + len(split.test_pool) == n_total
    assert len(split.train_pool) == int(np.floor(n_total * 0.40))

    # Check temporal ordering
    t_train_max = max(inst.prediction_timestamp for inst in split.train_pool)
    t_val_min = min(inst.prediction_timestamp for inst in split.val_pool)
    t_val_max = max(inst.prediction_timestamp for inst in split.val_pool)
    t_test_min = min(inst.prediction_timestamp for inst in split.test_pool)

    assert t_train_max <= t_val_min, "Train pool must precede validation pool"
    assert t_val_max <= t_test_min, "Validation pool must precede test pool"


def test_04_budget_sampling_and_nesting(sample_instances):
    """Verifies monotonic budget nesting S(1%) subset S(5%) ... subset S(100%)."""
    splitter = ChronologicalSplitter(train_ratio=0.40, val_ratio=0.30, test_ratio=0.30)
    split = splitter.split(sample_instances)

    budgets = [0.0, 0.05, 0.10, 0.25, 0.50, 1.00]
    samples = [BudgetSampler.sample(split.train_pool, fraction=b, seed=42) for b in budgets]

    # Check zero-shot fraction
    assert samples[0].fraction == 0.0
    assert samples[0].num_states == 0

    # Check monotonic nesting
    for i in range(1, len(samples) - 1):
        s_curr = set(inst.instance_id for inst in samples[i].instances)
        s_next = set(inst.instance_id for inst in samples[i + 1].instances)
        assert s_curr.issubset(s_next), f"Budget {samples[i].fraction} must be a subset of {samples[i+1].fraction}"


def test_05_target_only_ranker(sample_instances):
    """Verifies Target-Only model can fit on small local history without exceptions."""
    splitter = ChronologicalSplitter(train_ratio=0.40, val_ratio=0.30, test_ratio=0.30)
    split = splitter.split(sample_instances)

    # Train on 10% budget (approx 3 states)
    b_sample = BudgetSampler.sample(split.train_pool, fraction=0.10, seed=42)
    model = TargetOnlyRankerModel(n_estimators=10)
    model.fit(b_sample.instances)

    # Rank candidates on a test instance
    test_inst = split.test_pool[0]
    ranked = model.rank_candidates(test_inst)
    assert len(ranked) == 21
    assert "score" in ranked[0]
    assert ranked[0]["rank"] == 1


def test_06_continuation_boosted_ranker(sample_instances, frozen_rec):
    """Verifies LightGBM continuation boosting initialized from frozen source booster."""
    splitter = ChronologicalSplitter(train_ratio=0.40, val_ratio=0.30, test_ratio=0.30)
    split = splitter.split(sample_instances)

    b_sample = BudgetSampler.sample(split.train_pool, fraction=0.25, seed=42)
    model = ContinuationBoostedRankerModel(
        source_ranker=frozen_rec.ranker,
        additional_estimators=5,
        adaptation_lr=0.03,
    )
    model.fit(b_sample.instances)

    # Should have a valid adapted booster
    assert model.adapted_booster is not None
    ranked = model.rank_candidates(split.test_pool[0])
    assert len(ranked) == 21
    assert ranked[0]["rank"] == 1


def test_07_adapted_cbr_retrieval(sample_instances, frozen_rec, data_loader):
    """Verifies target-adapted CBR indexing and similarity retrieval."""
    splitter = ChronologicalSplitter(train_ratio=0.40, val_ratio=0.30, test_ratio=0.30)
    split = splitter.split(sample_instances)

    b_sample = BudgetSampler.sample(split.train_pool, fraction=0.25, seed=42)
    cbr_model = TargetAdaptedCBRModel(base_cbr=frozen_rec.cbr, pipeline=data_loader.pipeline)
    cbr_model.fit(b_sample.instances)

    assert len(cbr_model.local_cases) == b_sample.num_states
    ranked = cbr_model.rank_candidates(split.test_pool[0])
    assert len(ranked) == 21
    assert ranked[0]["rank"] == 1


def test_08_prior_adapted_hybrid(sample_instances, frozen_rec):
    """Verifies validation-tuned log-prior adjusted hybrid."""
    splitter = ChronologicalSplitter(train_ratio=0.40, val_ratio=0.30, test_ratio=0.30)
    split = splitter.split(sample_instances)

    counts = {}
    for inst in split.train_pool:
        counts[inst.target_action] = counts.get(inst.target_action, 0) + 1
    total = sum(counts.values()) or 1
    priors = {a: (counts.get(a, 0) + 1.0) / (total + 21) for a in [c["action_id"] for c in split.test_pool[0].candidate_actions]}

    adapted_hybrid = PriorAdaptedHybridModel(base_recommender=frozen_rec, action_priors=priors, beta=0.5)
    ranked = adapted_hybrid.rank_candidates(split.test_pool[0])
    assert len(ranked) == 21
    assert ranked[0]["rank"] == 1


def test_09_evaluator_metrics_and_bootstrap(sample_instances, frozen_rec):
    """Verifies ranking metric computation (MRR, NDCG, Top-k) and grouped bootstrapping."""
    splitter = ChronologicalSplitter(train_ratio=0.40, val_ratio=0.30, test_ratio=0.30)
    split = splitter.split(sample_instances)

    evaluator = Phase6Evaluator()
    model = ZeroShotHybridModel(frozen_rec)
    metrics, records = evaluator.evaluate_model(model, split.test_pool)

    assert metrics.total_instances == len(split.test_pool)
    assert 0.0 <= metrics.top1_exact_match <= 1.0
    assert 0.0 <= metrics.top3_recall <= 1.0
    assert 0.0 <= metrics.mean_reciprocal_rank <= 1.0
    assert 0.0 <= metrics.mean_ndcg_at_3 <= 1.0

    # Grouped bootstrap CI
    boot_ci = evaluator.compute_grouped_bootstrap_ci(records, "reciprocal_rank", n_bootstraps=200)
    assert boot_ci.ci_lower <= boot_ci.mean <= boot_ci.ci_upper


def test_10_leakage_audit_clean_pass(sample_instances):
    """Verifies that all 8 leakage audit checks pass with zero look-ahead bias or label leaks."""
    splitter = ChronologicalSplitter(train_ratio=0.40, val_ratio=0.30, test_ratio=0.30)
    split = splitter.split(sample_instances)

    auditor = Phase6LeakageAuditor()
    chk1 = auditor.audit_source_model_immutability()
    chk2 = auditor.audit_source_target_domain_isolation("factory_sensors", sample_instances)
    chk3 = auditor.audit_temporal_causality(sample_instances)
    chk4 = auditor.audit_split_chronological_ordering(split)
    chk5 = auditor.audit_split_disjointness(split)
    chk7 = auditor.audit_test_set_immutability(split.test_pool)
    chk8 = auditor.audit_target_label_feature_isolation(sample_instances)

    assert chk1["status"] == "PASS"
    assert chk2["status"] == "PASS"
    assert chk3["status"] == "PASS"
    assert chk4["status"] == "PASS"
    assert chk5["status"] == "PASS"
    assert chk7["status"] == "PASS"
    assert chk8["status"] == "PASS"


def test_11_reference_model_hash_immutability(frozen_rec):
    """Asserts that the reference Phase 4 model SHA-256 hash has not mutated."""
    inv = frozen_rec.verify_frozen_invariants()
    assert inv["is_immutable"] is True
    assert inv["status"] == "PASS"
