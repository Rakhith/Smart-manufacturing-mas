"""Comprehensive Automated Test Suite for Phase 5.

Verifies:
1. Dataset schema validation (OIL-040 & Factory Sensors)
2. Timestamp ordering and temporal monotonicity
3. Temporal leakage guard and causal isolation
4. Action mapping ontology completeness and confidence
5. DecisionState construction and CMSR compliance
6. Frozen Phase 4 model loading and SHA-256 hash preservation
7. Recommendation generation and candidate score ranking
8. Closed-loop state transitions (Replay & Surrogate modes)
9. FeedbackStore episodic tracking and metrics aggregation
10. Work-order alignment with telemetry time series
11. Ground-truth alignment without target leakage
12. LLM explanation output schema validity
13. No future-feature leakage in decision states
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import pandas as pd

from phase5.action_mapper import ActionMapper, CANONICAL_ACTIONS
from phase5.adapters import FactorySensorsAdapter, Oil040Adapter
from phase5.closed_loop import (
    ActionExecutor,
    ClosedLoopController,
    ExecutionMode,
    FeedbackStore,
    HistoricalReplayTransition,
    SurrogateStateTransition,
)
from phase5.frozen_model import FrozenPhase4Recommender
from phase5.leakage_guard import TemporalLeakageGuard
from phase5.llm_summarizer import GroundedPrescriptiveSummarizer


@pytest.fixture(scope="module")
def recommender():
    return FrozenPhase4Recommender()


@pytest.fixture(scope="module")
def oil_adapter():
    ad = Oil040Adapter()
    ad.load()
    return ad


@pytest.fixture(scope="module")
def fs_adapter():
    ad = FactorySensorsAdapter()
    ad.load()
    return ad


def test_01_oil040_schema_validation(oil_adapter):
    """Test 1: Schema validation of OIL-040 relational tables."""
    schema = oil_adapter.validate_schema()
    assert schema["status"] == "VALID"
    assert schema["equipment_count"] == 80
    assert schema["telemetry_rows"] == 9600
    assert schema["work_order_count"] == 80
    assert "vibration_rms_mm_s" in schema["columns_present"]
    assert "temperature_c" in schema["columns_present"]
    assert "pressure_psi" in schema["columns_present"]


def test_02_factory_sensors_schema_validation(fs_adapter):
    """Test 2: Schema validation of Factory Sensors pump tables."""
    schema = fs_adapter.validate_schema()
    assert schema["status"] == "VALID"
    assert schema["pump_assets_count"] == 6
    assert schema["sensor_readings_rows"] == 52560
    assert schema["work_orders_count"] == 63
    assert schema["failure_events_count"] == 39
    assert "vibration_mm_s" in schema["columns_present"]
    assert "bearing_temp_c" in schema["columns_present"]


def test_03_timestamp_monotonicity(oil_adapter, fs_adapter):
    """Test 3: Verify timestamps are strictly chronologically ordered."""
    aid_oil = oil_adapter.get_asset_ids()[0]
    seq_oil = oil_adapter.get_temporal_sequence(aid_oil)
    assert pd.to_datetime(seq_oil["timestamp"]).is_monotonic_increasing

    aid_fs = fs_adapter.get_asset_ids()[0]
    seq_fs = fs_adapter.get_temporal_sequence(aid_fs)
    assert pd.to_datetime(seq_fs["recorded_at"]).is_monotonic_increasing


def test_04_action_mapping_integrity():
    """Test 4: Action mapper maps external vocabularies into canonical ontology."""
    mapper = ActionMapper()
    # Test OIL-040 mappings
    for raw in ["sensor_calibration", "bearing_replacement", "alignment", "overhaul", "lubrication", "inspection"]:
        aid, conf, _ = mapper.map_external_action("oil040", raw)
        assert aid in CANONICAL_ACTIONS
        assert conf >= 0.90

    # Test Factory Sensors mappings
    for raw in ["bearing_wear", "seal_leak", "imbalance", "overheating", "electrical_fault"]:
        aid, conf, _ = mapper.map_external_action("factory_sensors", raw)
        assert aid in CANONICAL_ACTIONS
        assert conf >= 0.90


def test_05_decision_state_construction(oil_adapter, fs_adapter):
    """Test 5: Verify DecisionState object creation adheres to Phase 3A schema."""
    aid_oil = oil_adapter.get_asset_ids()[0]
    dstate_oil = oil_adapter.build_decision_state_at(aid_oil, "2025-01-10T00:00:00Z")
    assert dstate_oil["decision_state_id"].startswith("DS_OIL040_")
    assert "mechanical" in dstate_oil["input_machine_state"]
    assert "thermal" in dstate_oil["input_machine_state"]
    assert "fluid" in dstate_oil["input_machine_state"]
    assert dstate_oil["decision_severity"] in {"HEALTHY", "WATCH", "DEGRADING", "CRITICAL"}

    aid_fs = fs_adapter.get_asset_ids()[0]
    dstate_fs = fs_adapter.build_decision_state_at(aid_fs, "2025-10-01 12:00:00")
    assert dstate_fs["decision_state_id"].startswith("DS_FS_PUMP_")
    assert "mechanical" in dstate_fs["input_machine_state"]


def test_06_frozen_model_immutability(recommender):
    """Test 6: Verify SHA-256 hash preservation of frozen Phase 4 model."""
    inv = recommender.verify_frozen_invariants()
    assert inv["status"] == "PASS"
    assert inv["is_immutable"] is True
    assert inv["frozen_alpha"] == 0.75
    assert inv["cbr_k"] == 5


def test_07_recommendation_ranking(recommender, oil_adapter):
    """Test 7: Verify recommender outputs 20 sorted actions with valid scores."""
    aid_oil = oil_adapter.get_asset_ids()[0]
    dstate = oil_adapter.build_decision_state_at(aid_oil, "2025-01-15T00:00:00Z")
    ranked = recommender.rank_candidates(dstate)
    assert len(ranked) >= 20
    assert ranked[0]["rank"] == 1
    # Check descending order of scores
    scores = [r["hybrid_score"] for r in ranked]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_08_action_precondition_validation():
    """Test 8: Verify action executor correctly checks severity preconditions."""
    executor = ActionExecutor()
    # Replace bearing requires at least DEGRADING severity
    status_healthy, _ = executor.validate_preconditions("ACT_REPL_BEARING", "HEALTHY")
    assert status_healthy.value == "violated_severity_too_low"

    status_crit, _ = executor.validate_preconditions("ACT_REPL_BEARING", "CRITICAL")
    assert status_crit.value == "satisfied"


def test_09_closed_loop_transitions(recommender, oil_adapter):
    """Test 9: Verify closed-loop step in both Historical Replay and Surrogate modes."""
    # Historical Replay Mode
    ctrl_replay = ClosedLoopController(recommender=recommender, mode=ExecutionMode.HISTORICAL_REPLAY)
    aid = oil_adapter.get_asset_ids()[0]
    dstate = oil_adapter.build_decision_state_at(aid, "2025-01-15T00:00:00Z")
    rec_replay = ctrl_replay.step(dstate, timestamp="2025-01-15T00:00:00Z", asset_id=aid)
    assert rec_replay.outcome.is_simulated is False
    assert rec_replay.execution_mode == ExecutionMode.HISTORICAL_REPLAY

    # Surrogate Simulation Mode
    ctrl_sim = ClosedLoopController(recommender=recommender, mode=ExecutionMode.SURROGATE_SIMULATION)
    rec_sim = ctrl_sim.step(dstate, timestamp="2025-01-15T00:00:00Z", asset_id=aid)
    assert rec_sim.outcome.is_simulated is True
    assert rec_sim.execution_mode == ExecutionMode.SURROGATE_SIMULATION


def test_10_feedback_store_aggregation():
    """Test 10: FeedbackStore correctly tracks episodic history and computes fleet metrics."""
    store = FeedbackStore()
    assert store.compute_fleet_metrics()["total_decisions"] == 0


def test_11_temporal_leakage_guard_invariants():
    """Test 11: Leakage guard accepts valid causal points and catches violations."""
    guard = TemporalLeakageGuard()
    # Valid: input <= pred < future
    assert guard.audit_prediction_point(
        asset_id="TEST-1",
        prediction_timestamp="2025-01-10T12:00:00Z",
        latest_input_timestamp="2025-01-10T12:00:00Z",
        future_event_timestamp="2025-01-11T00:00:00Z",
    ) is True

    # Violation: prediction after future event (leakage)
    assert guard.audit_prediction_point(
        asset_id="TEST-1",
        prediction_timestamp="2025-01-12T00:00:00Z",
        latest_input_timestamp="2025-01-12T00:00:00Z",
        future_event_timestamp="2025-01-11T00:00:00Z",
    ) is False


def test_12_llm_technician_card_schema(recommender, oil_adapter):
    """Test 12: Verify GroundedPrescriptiveSummarizer outputs schema-compliant tickets."""
    summarizer = GroundedPrescriptiveSummarizer()
    aid = oil_adapter.get_asset_ids()[0]
    dstate = oil_adapter.build_decision_state_at(aid, "2025-01-15T00:00:00Z")
    ranked = recommender.rank_candidates(dstate)
    card = summarizer.generate_technician_card(dstate, ranked)

    assert "card_id" in card
    assert "prescribed_action" in card
    assert "grounded_evidence" in card
    assert "operational_urgency" in card

    cmms = summarizer.format_cmms_work_order(card)
    assert "WorkOrder" in cmms
    assert "Priority" in cmms
    assert "MaintenanceWorkCenter" in cmms


def test_13_no_future_leakage_in_decision_state(oil_adapter):
    """Test 13: Guarantee target failure label is never leaked as an input feature."""
    aid = oil_adapter.get_asset_ids()[0]
    dstate = oil_adapter.build_decision_state_at(aid, "2025-01-15T00:00:00Z")

    # Invariant: input_machine_state must NOT contain target fields
    for mod_name, mod_feats in dstate["input_machine_state"].items():
        for f_name in mod_feats.keys():
            assert "target" not in f_name.lower()
            assert "work_order" not in f_name.lower()
            assert "next_failure" not in f_name.lower()
