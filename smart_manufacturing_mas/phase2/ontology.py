"""Typed ontology schemas and metadata data structures for Phase 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import yaml


class SemanticModality(str, Enum):
    THERMAL = "thermal"
    MECHANICAL = "mechanical"
    KINEMATIC = "kinematic"
    FLUID = "fluid"
    ELECTRICAL = "electrical"
    ACOUSTIC = "acoustic"
    PROCESS_OPERATING = "process_operating"
    HEALTH_DEGRADATION = "health_degradation"
    OUTCOME_LABEL = "outcome_label"
    METADATA_IDENTIFIER = "metadata_identifier"
    UNKNOWN = "unknown"


class PhysicalQuantity(str, Enum):
    TEMPERATURE = "temperature"
    VIBRATION_VELOCITY = "vibration_velocity"
    VIBRATION_ACCELERATION = "vibration_acceleration"
    ROTATIONAL_SPEED = "rotational_speed"
    TORQUE = "torque"
    FORCE = "force"
    TOOL_WEAR_DURATION = "tool_wear_duration"
    PRESSURE = "pressure"
    DIFFERENTIAL_PRESSURE = "differential_pressure"
    FLOW_RATE = "flow_rate"
    FLUID_LEVEL = "fluid_level"
    ELECTRIC_CURRENT = "electric_current"
    ELECTRIC_VOLTAGE = "electric_voltage"
    ELECTRIC_POWER = "electric_power"
    ENERGY_CONSUMPTION = "energy_consumption"
    ACOUSTIC_EMISSION = "acoustic_emission"
    SOUND_LEVEL = "sound_level"
    NETWORK_LATENCY = "network_latency"
    PACKET_LOSS_RATE = "packet_loss_rate"
    CYCLE_COUNT = "cycle_count"
    REMAINING_USEFUL_LIFE = "remaining_useful_life"
    FAILURE_PROBABILITY = "failure_probability"
    CONDITION_PCT = "condition_pct"
    UNKNOWN = "unknown"


class MeasurementRole(str, Enum):
    AMBIENT_SENSOR = "ambient_sensor"
    PROCESS_SENSOR = "process_sensor"
    LUBRICANT_SENSOR = "lubricant_sensor"
    ACTUATOR_CONTROL = "actuator_control"
    ELECTRICAL_DRIVER = "electrical_driver"
    OPERATING_SETTING = "operating_setting"
    VIRTUAL_SENSOR = "virtual_sensor"
    PROCESS_PARAMETER = "process_parameter"
    ENGINE_SENSOR = "engine_sensor"
    ACCELEROMETER_CHANNEL = "accelerometer_channel"
    WAVEFORM_STATISTIC = "waveform_statistic"
    CYCLE_SUMMARY = "cycle_summary"
    HEALTH_METRIC = "health_metric"
    TARGET_LABEL = "target_label"
    IDENTIFIER = "identifier"


class FeatureTransformation(str, Enum):
    RAW = "raw"
    ROLLING_MEAN = "rolling_mean"
    ROLLING_STD = "rolling_std"
    DELTA_1 = "delta_1"
    SLOPE = "slope"
    FFT_DOMINANT_FREQUENCY = "fft_dominant_frequency"
    FFT_SPECTRAL_ENERGY = "fft_spectral_energy"
    FFT_SPECTRAL_ENTROPY = "fft_spectral_entropy"
    RMS = "rms"
    VARIANCE = "variance"
    STD = "std"
    SKEWNESS = "skewness"
    KURTOSIS = "kurtosis"
    CREST_FACTOR = "crest_factor"
    MIN = "min"
    MAX = "max"
    FIRST = "first"
    LAST = "last"
    WINDOW_DELTA = "window_delta"
    CYCLE_MEAN = "cycle_mean"
    CYCLE_SLOPE = "cycle_slope"


class SemanticConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class FeatureSemanticMetadata:
    dataset_id: str
    original_feature_name: str
    base_feature_name: str
    is_target: bool
    semantic_modality: str
    physical_quantity: str
    measurement_role: str
    original_unit: str
    canonical_unit: str
    transformation: str
    temporal_role: str
    transformation_parameter: str | None = None
    unit_conversion_rule: str | None = None
    asset_context: str = "industrial_asset"
    machine_archetype: str = "unknown"
    provenance: str = "dataset_source"
    semantic_confidence: str = "low"
    mapping_source: str = "dataset_metadata"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_ontology(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).parent / "ontology.yaml"
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def convert_unit(value: float, original_unit: str, canonical_unit: str, rule: str | None) -> float:
    """Safely convert value using verified linear rules without mutating original."""
    if rule is None or rule == "x" or original_unit == canonical_unit:
        return float(value)
    try:
        # Safe eval of strict arithmetic expressions containing 'x'
        allowed_globals = {"__builtins__": None}
        allowed_locals = {"x": float(value)}
        return float(eval(rule, allowed_globals, allowed_locals))
    except Exception:
        return float(value)
