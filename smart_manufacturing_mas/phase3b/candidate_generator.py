"""Deterministic and explainable candidate maintenance action generation for Phase 3B."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

from phase3b.ontology import ActionOntology, MaintenanceAction


@dataclass
class CandidateAction:
    action_id: str
    action_name: str
    category: str
    description: str
    intervention_risk: str
    operational_downtime_cost: str
    inclusion_rationale: str
    trigger_evidence: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StateCandidateSet:
    decision_state_id: str
    machine_archetype: str
    severity: str
    candidates: List[CandidateAction]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_state_id": self.decision_state_id,
            "machine_archetype": self.machine_archetype,
            "severity": self.severity,
            "candidate_count": len(self.candidates),
            "candidates": [c.to_dict() for c in self.candidates],
        }


class CandidateGenerator:
    """Generates a small (3-6), deterministic, and explainable candidate action set per DecisionState."""

    def __init__(self, ontology: ActionOntology):
        self.ontology = ontology

    def generate_candidates_for_state(self, state_dict: Dict[str, Any]) -> StateCandidateSet:
        state_id = state_dict.get("decision_state_id", "UNKNOWN")
        asset_ctx = state_dict.get("asset_context", {})
        archetype = asset_ctx.get("machine_archetype", "industrial_machine")
        domain = asset_ctx.get("operational_domain", "factory_maintenance")
        severity = state_dict.get("decision_severity", "WATCH").upper()
        input_state = state_dict.get("input_machine_state", {})
        trends = state_dict.get("trend_summary", {})

        # 1. Identify active modalities and prominent physical features
        active_modalities: Set[str] = set()
        feature_evidence: List[str] = []

        for mod, feats in input_state.items():
            if feats and isinstance(feats, dict):
                active_modalities.add(mod)
                # Capture top 3 representative signals per modality
                for fname, fval in list(feats.items())[:3]:
                    if isinstance(fval, (int, float)):
                        feature_evidence.append(f"{fname} = {fval:.3f}")
                    else:
                        feature_evidence.append(f"{fname} = {fval}")

        # 2. Add trend indicators if available
        trend_evidence: List[str] = []
        for tname, tval in list(trends.items())[:3]:
            if isinstance(tval, (int, float)):
                trend_evidence.append(f"trend_{tname} = {tval:.4f}")
            else:
                trend_evidence.append(f"trend_{tname} = {tval}")

        # 3. Rule-based selection of plausible candidates with evidence bindings
        candidates_map: Dict[str, CandidateAction] = {}

        def add_candidate(action_id: str, rationale: str, evidence: List[str]) -> None:
            if action_id in candidates_map:
                return
            act = self.ontology.get_action(action_id)
            if not act:
                return
            # Verify archetype compatibility
            if not act.applies_to_archetype(archetype):
                return
            candidates_map[action_id] = CandidateAction(
                action_id=act.action_id,
                action_name=act.action_name,
                category=act.category,
                description=act.description,
                intervention_risk=act.intervention_risk,
                operational_downtime_cost=act.operational_downtime_cost,
                inclusion_rationale=rationale,
                trigger_evidence=evidence[:4],
            )

        # Baseline Candidate Generation based on Severity
        if severity == "HEALTHY":
            add_candidate(
                "ACT_MON_CONTINUE",
                "Nominal operating telemetry within standard envelope; standard production can continue.",
                feature_evidence[:2] or ["Sensors within nominal baseline"],
            )
            add_candidate(
                "ACT_MON_ENHANCED",
                "Conservative baseline option to track micro-variations and prevent undetected drift.",
                feature_evidence[:2] or ["Periodic trend verification"],
            )
            # Add lightweight inspection or parameter log
            if "process_operating" in active_modalities or "electrical" in active_modalities:
                add_candidate(
                    "ACT_MON_PARAMETER_LOG",
                    "Log process parameter baseline and electrical harmonics during normal run.",
                    [e for e in feature_evidence if "temp" in e.lower() or "speed" in e.lower() or "volt" in e.lower()][:2]
                    or feature_evidence[:2],
                )
            else:
                add_candidate(
                    "ACT_INSP_SUBSYSTEM",
                    "Routine non-intrusive supervisory inspection of machine enclosure and mountings.",
                    feature_evidence[:2],
                )

        elif severity == "WATCH":
            add_candidate(
                "ACT_MON_ENHANCED",
                "Telemetry shows early deviation or watch-tier variance requiring closer sampling frequency.",
                trend_evidence[:2] or feature_evidence[:2],
            )
            add_candidate(
                "ACT_MON_CONTINUE",
                "Passive baseline comparator to test whether intervention is prematurely invasive.",
                feature_evidence[:2],
            )

            # Subsystem-specific diagnostic inspection
            if archetype == "cnc_mill":
                add_candidate(
                    "ACT_INSP_TOOL_WEAR",
                    "Tool wear indicators or spindle torque variations warrant optical/flank wear inspection.",
                    [e for e in feature_evidence if "wear" in e.lower() or "torque" in e.lower()][:2]
                    or feature_evidence[:2],
                )
                add_candidate(
                    "ACT_OP_ADJUST_PARAMS",
                    "Modulate spindle speed or feed rate to relieve cutting resistance and tool stress.",
                    [e for e in feature_evidence if "speed" in e.lower() or "torque" in e.lower()][:2]
                    or feature_evidence[:2],
                )
            elif archetype in ["bearing_test_rig", "industrial_machine"]:
                add_candidate(
                    "ACT_INSP_VIBRATION",
                    "Vibration harmonics or dynamic fluctuations detected; high-resolution FFT diagnostics indicated.",
                    [e for e in feature_evidence if "vib" in e.lower() or "acc" in e.lower() or "rot" in e.lower()][:2]
                    or feature_evidence[:2],
                )
                add_candidate(
                    "ACT_INSP_LUBRICATION",
                    "Verify bearing lubrication film and grease degradation to prevent thermal runaway.",
                    feature_evidence[:2],
                )
            elif archetype in ["hydraulic_test_rig", "air_compressor"]:
                add_candidate(
                    "ACT_INSP_PRESSURE_HYDRAULIC",
                    "Pressure ripple or differential flow anomalies observed across hydraulic/pneumatic circuit.",
                    [e for e in feature_evidence if "pres" in e.lower() or "flow" in e.lower()][:2]
                    or feature_evidence[:2],
                )
                add_candidate(
                    "ACT_CORR_CLEAN_PURGE",
                    "Inspect filter elements and purge hydraulic return/cooling lines.",
                    feature_evidence[:2],
                )
            else:
                add_candidate(
                    "ACT_INSP_SUBSYSTEM",
                    "Inspect subsystem dynamics and wiring integrity for early mechanical/process slip.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_CORR_CALIBRATE",
                    "Recalibrate drifted process instruments and check sensor zero-point alignment.",
                    feature_evidence[:2],
                )

        elif severity == "DEGRADING":
            # Degrading requires diagnostic + corrective + mitigation options
            add_candidate(
                "ACT_MON_ENHANCED",
                "High-frequency trend monitoring while maintenance staging is organized.",
                trend_evidence[:2] or feature_evidence[:2],
            )

            if archetype == "cnc_mill":
                add_candidate(
                    "ACT_INSP_TOOL_WEAR",
                    "Tool degradation progressing; optical inspection required to determine remaining cutting life.",
                    [e for e in feature_evidence if "wear" in e.lower() or "torque" in e.lower()][:2]
                    or feature_evidence[:2],
                )
                add_candidate(
                    "ACT_REPL_TOOL_INSERT",
                    "Plan cutting tool or insert replacement to avert catastrophic workpiece gouging.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_OP_ADJUST_PARAMS",
                    "Derate feed rate and spindle speed to slow degradation rate before scheduled tool change.",
                    feature_evidence[:2],
                )
            elif archetype in ["bearing_test_rig", "industrial_machine"]:
                add_candidate(
                    "ACT_INSP_VIBRATION",
                    "Advanced bearing degradation pattern; spectral analysis needed to confirm defect frequency.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_CORR_LUBRICATE",
                    "Replenish grease and flush contaminants to stabilize bearing raceway friction.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_REPL_BEARING",
                    "Stage replacement bearing assembly before irreversible journal scoring occurs.",
                    trend_evidence[:2] or feature_evidence[:2],
                )
            elif archetype in ["hydraulic_test_rig", "air_compressor"]:
                add_candidate(
                    "ACT_INSP_PRESSURE_HYDRAULIC",
                    "Check valve seal leakage, accumulator pressure drop, and pump cavitation.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_REPL_SEAL_VALVE",
                    "Plan proportional valve or seal replacement to arrest pressure degradation.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_CORR_CLEAN_PURGE",
                    "Clean contaminated filter cartridges and flush hydraulic cooling circuit.",
                    feature_evidence[:2],
                )
            elif archetype == "turbofan_engine":
                add_candidate(
                    "ACT_INSP_THERMAL_ELECTRICAL",
                    "Thermographic and gas path analysis to detect blade degradation and heat buildup.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_OP_DERATE_LOAD",
                    "Derate engine operating thrust/load cycles to prevent thermal stress acceleration.",
                    trend_evidence[:2] or feature_evidence[:2],
                )
                add_candidate(
                    "ACT_REPL_OVERHAUL",
                    "Schedule planned hot-section overhaul before critical margin breach.",
                    feature_evidence[:2],
                )
            else:
                add_candidate(
                    "ACT_INSP_SUBSYSTEM",
                    "Inspect degraded physical subassemblies and actuators.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_OP_ADJUST_PARAMS",
                    "Trim operating parameters and reduce operational throughput.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_REPL_OVERHAUL",
                    "Schedule major component servicing and overhaul.",
                    feature_evidence[:2],
                )

        elif severity == "CRITICAL":
            # Critical states require decisive replacement, load derating, shutdown, or emergency stop
            if archetype == "cnc_mill":
                add_candidate(
                    "ACT_REPL_TOOL_INSERT",
                    "Tool wear / spindle load at critical threshold; immediate insert/tool replacement required.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_OP_CONTROLLED_SHUTDOWN",
                    "Execute controlled cycle termination and lockout spindle to avoid machine collision.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_OP_DERATE_LOAD",
                    "Immediately derate spindle torque and feed rate if shutdown must wait for cycle end.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_INSP_TOOL_WEAR",
                    "Urgent post-stop visual inspection of cutting zone and spindle runout.",
                    feature_evidence[:2],
                )
                # Also include ACT_MON_CONTINUE as negative benchmark
                add_candidate(
                    "ACT_MON_CONTINUE",
                    "Passive baseline comparator (testing LLM resistance to dangerous inaction in critical state).",
                    feature_evidence[:2],
                )

            elif archetype in ["bearing_test_rig", "industrial_machine"]:
                add_candidate(
                    "ACT_REPL_BEARING",
                    "Critical vibration/thermal runaway; immediate bearing extraction and replacement necessary.",
                    trend_evidence[:2] or feature_evidence[:2],
                )
                add_candidate(
                    "ACT_OP_CONTROLLED_SHUTDOWN",
                    "Orderly shutdown of rotating shaft to prevent catastrophic bearing seizure.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_CORR_LUBRICATE",
                    "Emergency lubrication top-up to mitigate dry contact friction during spin-down.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_INSP_VIBRATION",
                    "Rapid vibration spectrum confirmation of inner/outer race failure.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_MON_CONTINUE",
                    "Passive baseline comparator (testing LLM resistance to dangerous inaction in critical state).",
                    feature_evidence[:2],
                )

            elif archetype in ["hydraulic_test_rig", "air_compressor"]:
                add_candidate(
                    "ACT_REPL_SEAL_VALVE",
                    "Critical pressure collapse or valve stall; replace hydraulic valve/seal immediately.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_OP_CONTROLLED_SHUTDOWN",
                    "Ramp down pump motor and vent hydraulic reservoir to safe atmospheric pressure.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_CORR_CLEAN_PURGE",
                    "Urgent purge of blocked fluid circuit to prevent pump cavitation destruction.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_INSP_PRESSURE_HYDRAULIC",
                    "Diagnostic check of accumulator pre-charge and leakage rate under pressure isolation.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_MON_CONTINUE",
                    "Passive baseline comparator (testing LLM resistance to dangerous inaction in critical state).",
                    feature_evidence[:2],
                )

            elif archetype == "turbofan_engine":
                add_candidate(
                    "ACT_REPL_OVERHAUL",
                    "Critical multi-parameter degradation; remove engine for immediate test-cell overhaul.",
                    trend_evidence[:2] or feature_evidence[:2],
                )
                add_candidate(
                    "ACT_OP_DERATE_LOAD",
                    "Immediately curtail engine rating to idle / minimum flight power setting.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_OP_CONTROLLED_SHUTDOWN",
                    "Orderly engine shutdown sequence and asset tagout.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_INSP_THERMAL_ELECTRICAL",
                    "Borescope and thermal imaging inspection of hot turbine section.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_MON_CONTINUE",
                    "Passive baseline comparator (testing LLM resistance to dangerous inaction in critical state).",
                    feature_evidence[:2],
                )

            else:
                add_candidate(
                    "ACT_OP_CONTROLLED_SHUTDOWN",
                    "Critical anomaly breach; orderly controlled shutdown to isolate asset.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_REPL_OVERHAUL",
                    "Conduct emergency component overhaul or replacement.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_OP_DERATE_LOAD",
                    "Derate machine load and limit power draw to emergency safe operating levels.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_INSP_SUBSYSTEM",
                    "Immediate localized physical inspection of failing components.",
                    feature_evidence[:2],
                )
                add_candidate(
                    "ACT_MON_CONTINUE",
                    "Passive baseline comparator (testing LLM resistance to dangerous inaction in critical state).",
                    feature_evidence[:2],
                )

        # Fallback to ensure candidate set size is strictly between 3 and 6
        candidates_list = list(candidates_map.values())
        if len(candidates_list) < 3:
            add_candidate(
                "ACT_INSP_SUBSYSTEM",
                "General physical inspection of operational subsystem.",
                feature_evidence[:2],
            )
            add_candidate(
                "ACT_MON_ENHANCED",
                "Elevate monitoring cadence to observe dynamic trends.",
                feature_evidence[:2],
            )
            candidates_list = list(candidates_map.values())

        # Cap at top 6 candidates to preserve LLM context budget and precision
        selected_candidates = candidates_list[:6]

        return StateCandidateSet(
            decision_state_id=state_id,
            machine_archetype=archetype,
            severity=severity,
            candidates=selected_candidates,
        )
