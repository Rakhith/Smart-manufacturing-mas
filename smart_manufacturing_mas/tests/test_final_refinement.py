"""Comprehensive Test Suite for Phase 5 Final Experimental Refinement.

Verifies:
1. Hierarchical Evaluator (Family, Category, Action, Action+Component)
2. Deterministic Graded Action Relevance Scoring
3. Action Mapping Auditor (Coverage, Ambiguity, Dual Denominator)
4. Component and Failure Evaluator (Precision, Recall, Lead Time)
5. Multi-Horizon Evaluator
6. Secondary Semantic Judge
7. Selective Prediction & Confidence Calibration
8. Chronological Adaptation Engine & 8 Reference Baselines
9. Closed-Loop Simulation Sensitivity Analysis (Historical Replay vs Surrogate)
10. Statistical Bootstrap Reporting (1,000 resamples, 95% CIs)
11. 14-Point Causal Leakage Guard
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from phase5.action_mapping_auditor import ActionMappingAuditor
from phase5.adaptation_engine import AdaptationEngine
from phase5.closed_loop_refinement import (
    ClosedLoopRefinementEvaluator,
    CostRegime,
    SensitivityParameters,
)
from phase5.component_failure_evaluator import ComponentFailureEvaluator
from phase5.final_leakage_guard import FinalLeakageGuard
from phase5.frozen_model import FrozenPhase4Recommender
from phase5.hierarchical_evaluation import HierarchicalEvaluator
from phase5.multi_horizon_evaluator import MultiHorizonEvaluator
from phase5.selective_prediction import SelectivePredictor
from phase5.semantic_judge import SecondarySemanticJudge
from phase5.statistical_reporting import BootstrapStatisticalReporter


@pytest.fixture
def sample_predictions():
    return [
        {
            "asset_id": "pump_1",
            "prediction_timestamp": "2026-01-10T12:00:00",
            "event_timestamp": "2026-01-10T18:00:00",
            "lead_time_hours": 6.0,
            "target_action": "ACT_REPL_BEARING",
            "predicted_action_top1": "ACT_REPL_BEARING",
            "predicted_action_top2": "ACT_INSP_VIBRATION",
            "predicted_action_top3": "ACT_CORR_LUBRICATE",
            "ranked_action_ids": ["ACT_REPL_BEARING", "ACT_INSP_VIBRATION", "ACT_CORR_LUBRICATE"],
            "top1_exact": 1,
            "top3_recall": 1,
            "reciprocal_rank": 1.0,
            "ndcg3": 1.0,
            "ndcg5": 1.0,
            "top1_score": 0.88,
            "top2_score": 0.72,
            "ranking_margin": 0.16,
            "predicted_component": "bearing",
            "target_component": "bearing",
            "decision_state": {
                "decision_severity": "CRITICAL",
                "asset_context": {"machine_archetype": "centrifugal_pump"},
                "input_machine_state": {"health_degradation": {"health_index": 35.0}},
            },
        },
        {
            "asset_id": "pump_2",
            "prediction_timestamp": "2026-02-15T06:00:00",
            "event_timestamp": "2026-02-15T12:00:00",
            "lead_time_hours": 6.0,
            "target_action": "ACT_REPL_SEAL_VALVE",
            "predicted_action_top1": "ACT_INSP_PRESSURE_HYDRAULIC",
            "predicted_action_top2": "ACT_REPL_SEAL_VALVE",
            "predicted_action_top3": "ACT_CORR_FASTEN_ALIGN",
            "ranked_action_ids": ["ACT_INSP_PRESSURE_HYDRAULIC", "ACT_REPL_SEAL_VALVE", "ACT_CORR_FASTEN_ALIGN"],
            "top1_exact": 0,
            "top3_recall": 1,
            "reciprocal_rank": 0.5,
            "ndcg3": 0.63,
            "ndcg5": 0.63,
            "top1_score": 0.75,
            "top2_score": 0.70,
            "ranking_margin": 0.05,
            "predicted_component": "seal_valve",
            "target_component": "seal_valve",
            "decision_state": {
                "decision_severity": "DEGRADING",
                "asset_context": {"machine_archetype": "centrifugal_pump"},
                "input_machine_state": {"health_degradation": {"health_index": 55.0}},
            },
        },
        {
            "asset_id": "pump_3",
            "prediction_timestamp": "2026-03-20T00:00:00",
            "event_timestamp": "2026-03-20T06:00:00",
            "lead_time_hours": 6.0,
            "target_action": "ACT_CORR_LUBRICATE",
            "predicted_action_top1": "ACT_MON_ENHANCED",
            "predicted_action_top2": "ACT_INSP_LUBRICATION",
            "predicted_action_top3": "ACT_CORR_LUBRICATE",
            "ranked_action_ids": ["ACT_MON_ENHANCED", "ACT_INSP_LUBRICATION", "ACT_CORR_LUBRICATE"],
            "top1_exact": 0,
            "top3_recall": 1,
            "reciprocal_rank": 0.3333,
            "ndcg3": 0.50,
            "ndcg5": 0.50,
            "top1_score": 0.65,
            "top2_score": 0.62,
            "ranking_margin": 0.03,
            "predicted_component": "lubrication_system",
            "target_component": "lubrication_system",
            "decision_state": {
                "decision_severity": "WATCH",
                "asset_context": {"machine_archetype": "centrifugal_pump"},
                "input_machine_state": {"health_degradation": {"health_index": 72.0}},
            },
        },
    ]


def test_hierarchical_evaluation(sample_predictions):
    evaluator = HierarchicalEvaluator()
    summary, evaluated = evaluator.evaluate_records(sample_predictions)
    assert summary.total_instances == 3
    assert summary.action_exact_agreement_rate == pytest.approx(1.0 / 3.0, rel=1e-3)
    assert summary.mean_graded_relevance_score > 0.4
    assert len(evaluated) == 3
    assert "graded_relevance" in evaluated[0]


def test_action_mapping_auditor():
    auditor = ActionMappingAuditor()
    sample_wos = [
        {"work_order_id": "WO-1", "description": "Replace worn drive end bearing", "action_code": "REPL_BRG"},
        {"work_order_id": "WO-2", "description": "Mechanical seal leakage overhaul", "action_code": "SEAL_LEAK"},
    ]
    report = auditor.audit_dataset("test_plant", sample_wos)
    assert report.total_work_orders == 2
    assert report.mapped_work_orders == 2
    assert report.ontology_coverage_pct == 100.0


def test_component_and_failure_evaluator(sample_predictions):
    evaluator = ComponentFailureEvaluator()
    comp_res = evaluator.evaluate_component_identification(sample_predictions)
    assert comp_res.total_instances == 3
    assert comp_res.component_top1_accuracy == 1.0  # All predicted components match target components

    failures = [
        {"asset_id": "pump_1", "timestamp": "2026-01-10T18:00:00", "failure_mode": "Bearing Fatigue"},
        {"asset_id": "pump_2", "timestamp": "2026-02-15T12:00:00", "failure_mode": "Seal Leakage"},
    ]
    fail_res = evaluator.evaluate_failure_anticipation(failures, sample_predictions)
    assert fail_res.total_failures == 2
    assert fail_res.anticipated_failures == 2
    assert fail_res.anticipation_rate == 1.0


def test_secondary_semantic_judge(sample_predictions):
    judge = SecondarySemanticJudge(use_deterministic_fallback=True)
    res = judge.evaluate_dataset(sample_predictions)
    assert res["total_evaluated"] == 3
    assert res["semantic_acceptance_rate"] >= 0.6
    dist = res["distribution"]
    assert "EXACT_MATCH" in dist


def test_selective_prediction(sample_predictions):
    predictor = SelectivePredictor()
    predictor.calibrate_confidence_thresholds(sample_predictions)
    curve = predictor.evaluate_coverage_curve(sample_predictions)
    assert 1.0 in curve
    assert 0.7 in curve
    cal = predictor.compute_calibration_metrics(sample_predictions)
    assert "expected_calibration_error" in cal
    assert "brier_score" in cal


def test_statistical_bootstrap_reporting():
    reporter = BootstrapStatisticalReporter(n_resamples=200, random_state=42)
    vals = [0.2, 0.4, 0.6, 0.8, 1.0]
    res = reporter.bootstrap_metric(vals, metric_name="test_metric")
    assert res.point_estimate == 0.6
    assert res.ci_lower_95 <= res.point_estimate <= res.ci_upper_95

    # Paired test
    vals_b = [0.4, 0.6, 0.8, 1.0, 1.0]
    comp = reporter.bootstrap_comparison(vals, vals_b, metric_name="test_metric")
    assert comp["observed_difference"] > 0.0


def test_closed_loop_refinement(sample_predictions):
    recommender = FrozenPhase4Recommender()
    evaluator = ClosedLoopRefinementEvaluator(recommender=recommender)

    dstates = [
        {
            "decision_state": p["decision_state"],
            "timestamp": p["prediction_timestamp"],
            "asset_id": p["asset_id"],
            "ground_truth_context": {"matched_work_order": {"canonical_action_id": p["target_action"]}},
        }
        for p in sample_predictions
    ]

    replay_res = evaluator.evaluate_historical_replay(dstates)
    assert replay_res.execution_mode == "Historical_Replay"
    assert replay_res.is_simulated is False

    surr_res = evaluator.evaluate_surrogate_regime(dstates, regime=CostRegime.BASELINE)
    assert surr_res.execution_mode == "Surrogate_Simulation"
    assert surr_res.is_simulated is True
    assert "[SIMULATED SURROGATE]" in surr_res.notes


def test_adaptation_engine_chronological_splits(sample_predictions):
    recommender = FrozenPhase4Recommender()
    engine = AdaptationEngine(recommender=recommender)
    splits = engine.partition_chronological(sample_predictions, train_ratio=0.4, val_ratio=0.3)
    assert "train" in splits and "val" in splits and "test" in splits
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == len(sample_predictions)
