"""Pre-LLM payload sanitization, zero-leakage validator, and audit logger for Phase 3B."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class LeakageViolationError(Exception):
    """Raised when ground truth or future target labels leak into an LLM payload."""
    pass


class LeakageGuard:
    """Guarantees zero future outcome, target label, or ground truth leakage into LLM inputs."""

    # Forbidden ground-truth keys and sensitive target names
    FORBIDDEN_KEYS: Set[str] = {
        "ground_truth_context",
        "evaluation_ground_truth_context",
        "target_ground_truth",
        "ground_truth",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
        "fault_type",
        "failure_type",
        "failure_mode",
        "target_label",
        "target",
        "label",
        "y_true",
        "rul",
        "remaining_useful_life",
    }

    # Forbidden substrings / regexes in text representations
    FORBIDDEN_PATTERNS: List[re.Pattern] = [
        re.compile(r"observed\s+machine\s+failure\s+mode", re.IGNORECASE),
        re.compile(r"\b(TWF|HDF|PWF|OSF|RNF)\b"),
        re.compile(r"tool\s+wear\s+failure", re.IGNORECASE),
        re.compile(r"heat\s+dissipation\s+failure", re.IGNORECASE),
        re.compile(r"power\s+failure", re.IGNORECASE),
        re.compile(r"overstrain\s+failure", re.IGNORECASE),
        re.compile(r"random\s+failure", re.IGNORECASE),
        re.compile(r"quarantined_ground_truth", re.IGNORECASE),
        re.compile(r"target_ground_truth", re.IGNORECASE),
    ]

    @classmethod
    def sanitize_state_for_prompt(cls, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """Strips all ground truth context and synthesizes an objective, non-leaking diagnostic summary."""
        state_id = raw_state.get("decision_state_id", "UNKNOWN")
        asset_ctx = raw_state.get("asset_context", {})
        severity = raw_state.get("decision_severity", "WATCH")
        temporal_ctx = raw_state.get("temporal_context", {})
        input_state = raw_state.get("input_machine_state", {})
        trends = raw_state.get("trend_summary", {})

        # Sanitize input modalities (strictly remove any outcome_label modality if present)
        sanitized_input: Dict[str, Dict[str, Any]] = {}
        has_unmapped = False
        active_sensor_count = 0

        for mod, feats in input_state.items():
            if mod in ["outcome_label", "target"]:
                continue  # Never include
            if mod == "unmapped" and feats:
                has_unmapped = True
            if isinstance(feats, dict) and feats:
                sanitized_input[mod] = feats
                active_sensor_count += len(feats)

        # Build explicit data limitations description
        temporal_type = asset_ctx.get("temporal_type", "static_tabular")
        limitations: List[str] = []
        if temporal_type in ["static_tabular", "signal_snapshot"] or not trends:
            limitations.append("Snapshot observation only; no longitudinal causal temporal trend available.")
        if has_unmapped:
            limitations.append("Process contains unmapped or anonymous feature channels without verified engineering units.")
        limitations.append(f"Observed severity tier ({severity}) is distribution-calibrated relative to operational baselines.")

        # Objective, non-leaking diagnostic context
        objective_rationale = (
            f"Asset exhibiting operational telemetry in calibrated {severity} severity tier "
            f"across {len(sanitized_input)} active sensor modalities ({active_sensor_count} total channels)."
        )

        sanitized_payload = {
            "decision_state_id": state_id,
            "asset_context": {
                "machine_archetype": asset_ctx.get("machine_archetype", "unknown"),
                "operational_domain": asset_ctx.get("operational_domain", "unknown"),
                "temporal_type": temporal_type,
            },
            "temporal_context": {
                "cycle": temporal_ctx.get("cycle"),
                "sequence_step": temporal_ctx.get("sequence_step"),
                "trajectory_phase": temporal_ctx.get("trajectory_phase"),
            },
            "decision_severity": severity,
            "diagnostic_summary": objective_rationale,
            "data_limitations": limitations,
            "input_machine_state": sanitized_input,
            "trend_summary": trends,
        }

        return sanitized_payload

    @classmethod
    def validate_zero_leakage(cls, payload: Any, path: str = "") -> Tuple[bool, List[str]]:
        """Recursively checks any dict, list, or string for forbidden ground-truth tokens."""
        violations: List[str] = []

        if isinstance(payload, dict):
            for k, v in payload.items():
                curr_path = f"{path}.{k}" if path else k
                # Check key against forbidden keys
                if k in cls.FORBIDDEN_KEYS:
                    violations.append(f"Forbidden key detected: '{curr_path}'")
                for pat in cls.FORBIDDEN_PATTERNS:
                    if pat.search(str(k)):
                        violations.append(f"Forbidden key pattern in '{curr_path}'")
                # Recurse value
                child_clean, child_violations = cls.validate_zero_leakage(v, curr_path)
                violations.extend(child_violations)

        elif isinstance(payload, list):
            for idx, item in enumerate(payload):
                curr_path = f"{path}[{idx}]"
                child_clean, child_violations = cls.validate_zero_leakage(item, curr_path)
                violations.extend(child_violations)

        elif isinstance(payload, str):
            for pat in cls.FORBIDDEN_PATTERNS:
                if pat.search(payload):
                    violations.append(f"Forbidden text pattern '{pat.pattern}' at '{path}': {payload[:60]}")

        return (len(violations) == 0, violations)

    @classmethod
    def assert_zero_leakage(cls, payload: Any) -> None:
        """Throws LeakageViolationError if any forbidden token or key exists."""
        is_clean, violations = cls.validate_zero_leakage(payload)
        if not is_clean:
            err_msg = "\n".join(violations[:10])
            raise LeakageViolationError(f"CRITICAL LEAKAGE DETECTED in payload:\n{err_msg}")


class LeakageAuditLogger:
    """Maintains an auditable verification trail of zero-leakage checks."""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def record_check(self, state_id: str, is_clean: bool, violations: List[str]) -> None:
        self.records.append({
            "decision_state_id": state_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "passed": is_clean,
            "violation_count": len(violations),
            "violations": violations,
        })

    def export_audit_log(self, target_path: Path) -> Dict[str, Any]:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        total = len(self.records)
        passed = sum(1 for r in self.records if r["passed"])
        failed = total - passed

        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_payloads_audited": total,
            "passed_zero_leakage": passed,
            "failed_leakage_violations": failed,
            "zero_leakage_rate_pct": 100.0 * passed / max(1, total),
            "audit_records": self.records,
        }
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary
