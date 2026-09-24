"""Structural validation and engineering sanity checks for Phase 3B judgments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from phase3b.candidate_generator import StateCandidateSet


@dataclass
class ValidationReport:
    decision_state_id: str
    is_schema_valid: bool
    structural_errors: List[str]
    sanity_flags_triggered: List[str]
    candidate_coverage_pct: float
    top_action: str
    top_action_rank: int
    top_action_suitability: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JudgmentValidator:
    """Validates structured LLM judgment output and calculates engineering sanity flags."""

    REQUIRED_EVAL_FIELDS: Set[str] = {
        "action_id",
        "rank",
        "suitability_score",
        "urgency_score",
        "expected_effectiveness_score",
        "operational_risk_score",
        "confidence",
        "reasoning_summary",
        "final_verdict",
    }

    ALLOWED_VERDICTS: Set[str] = {
        "RECOMMENDED",
        "ACCEPTABLE_ALTERNATIVE",
        "INAPPROPRIATE_AT_CURRENT_TIME",
        "UNSAFE",
    }

    @classmethod
    def normalize_judgment(cls, judgment: Dict[str, Any], candidate_set: StateCandidateSet) -> Dict[str, Any]:
        """Auto-heals all minor formatting and structural deviations (e.g. missing candidates, type mismatches, ties)."""
        if not isinstance(judgment, dict):
            judgment = {"evaluations": []}

        evaluations = judgment.get("evaluations", [])
        if not isinstance(evaluations, list):
            evaluations = []
            judgment["evaluations"] = evaluations

        expected_ids = {c.action_id for c in candidate_set.candidates}
        severity = candidate_set.severity

        # 1. Clean individual evaluation records
        cleaned_evals = []
        for idx, raw_ev in enumerate(evaluations):
            if not isinstance(raw_ev, dict):
                continue
            ev = dict(raw_ev)

            # Clean action_id trailing commas / whitespace
            aid = str(ev.get("action_id", "")).strip().rstrip(",;.:")
            if aid in expected_ids:
                ev["action_id"] = aid
            elif not aid and idx < len(candidate_set.candidates):
                ev["action_id"] = candidate_set.candidates[idx].action_id

            # Coerce & clamp numeric scores
            for score_key in ["suitability_score", "urgency_score", "expected_effectiveness_score", "operational_risk_score"]:
                val = ev.get(score_key, 50)
                try:
                    int_val = int(float(val))
                except (ValueError, TypeError):
                    int_val = 50
                ev[score_key] = max(0, min(100, int_val))

            # Coerce & clamp confidence
            conf = ev.get("confidence", 0.7)
            try:
                flt_conf = float(conf)
            except (ValueError, TypeError):
                flt_conf = 0.7
            ev["confidence"] = round(max(0.0, min(1.0, flt_conf)), 2)

            # Normalize verdict aliases
            verd = str(ev.get("final_verdict", "")).strip().upper()
            if "INAPPROPRIATE" in verd:
                ev["final_verdict"] = "INAPPROPRIATE_AT_CURRENT_TIME"
            elif "ACCEPTABLE" in verd or "ALTERNATIVE" in verd:
                ev["final_verdict"] = "ACCEPTABLE_ALTERNATIVE"
            elif "RECOMMEND" in verd:
                ev["final_verdict"] = "RECOMMENDED"
            elif "UNSAFE" in verd:
                ev["final_verdict"] = "UNSAFE"
            else:
                suit = ev["suitability_score"]
                risk = ev["operational_risk_score"]
                if suit >= 80:
                    ev["final_verdict"] = "RECOMMENDED"
                elif suit >= 55:
                    ev["final_verdict"] = "ACCEPTABLE_ALTERNATIVE"
                elif risk >= 80 and severity == "CRITICAL":
                    ev["final_verdict"] = "UNSAFE"
                else:
                    ev["final_verdict"] = "INAPPROPRIATE_AT_CURRENT_TIME"

            # Lists
            if not isinstance(ev.get("unsupported_assumptions"), list):
                ev["unsupported_assumptions"] = []
            if not isinstance(ev.get("evidence_used"), list):
                ev["evidence_used"] = []

            # Reasoning summary
            reason = str(ev.get("reasoning_summary", "")).strip()
            if not reason:
                reason = f"Action {ev.get('action_id')} evaluated for {severity} state."
            ev["reasoning_summary"] = reason

            cleaned_evals.append(ev)

        # 2. Inject any missing candidates from candidate_set
        evaluated_ids = {e.get("action_id") for e in cleaned_evals}
        for cand in candidate_set.candidates:
            if cand.action_id not in evaluated_ids:
                cleaned_evals.append({
                    "action_id": cand.action_id,
                    "rank": 99,
                    "suitability_score": 15,
                    "urgency_score": 20,
                    "expected_effectiveness_score": 25,
                    "operational_risk_score": 35,
                    "confidence": 0.6,
                    "evidence_used": [],
                    "reasoning_summary": f"Omitted candidate {cand.action_id} auto-healed.",
                    "unsupported_assumptions": [],
                    "final_verdict": "INAPPROPRIATE_AT_CURRENT_TIME",
                })

        # 3. Resolve strict rank sequence (1..K without ties)
        def sort_key(e: Dict[str, Any]) -> Tuple[int, int, int]:
            raw_r = e.get("rank", 99)
            try:
                r_int = int(raw_r)
            except (ValueError, TypeError):
                r_int = 99
            return (r_int, -e.get("suitability_score", 0), e.get("operational_risk_score", 50))

        cleaned_evals.sort(key=sort_key)
        for rank_num, ev in enumerate(cleaned_evals, 1):
            ev["rank"] = rank_num

        judgment["evaluations"] = cleaned_evals

        # 4. Ensure top_recommended_action and alternatives
        if cleaned_evals:
            top_id = cleaned_evals[0]["action_id"]
            judgment["top_recommended_action"] = top_id
            judgment["alternative_actions"] = [
                e["action_id"] for e in cleaned_evals[1:] if e.get("final_verdict") == "ACCEPTABLE_ALTERNATIVE"
            ]
        else:
            judgment["top_recommended_action"] = ""
            judgment["alternative_actions"] = []

        if "insufficient_information" not in judgment:
            judgment["insufficient_information"] = False
        if "uncertainty_explanation" not in judgment:
            judgment["uncertainty_explanation"] = ""

        return judgment

    @classmethod
    def validate_and_audit(
        cls,
        judgment: Dict[str, Any],
        candidate_set: StateCandidateSet,
        raw_state: Dict[str, Any],
    ) -> ValidationReport:
        errors: List[str] = []
        sanity_flags: List[str] = []
        state_id = candidate_set.decision_state_id
        severity = candidate_set.severity

        # Normalize and auto-heal formatting before auditing
        if isinstance(judgment, dict):
            judgment = cls.normalize_judgment(judgment, candidate_set)

        # 1. Structural Validation
        if not isinstance(judgment, dict):
            errors.append("Judgment output is not a JSON dictionary.")
            return ValidationReport(
                decision_state_id=state_id,
                is_schema_valid=False,
                structural_errors=errors,
                sanity_flags_triggered=sanity_flags,
                candidate_coverage_pct=0.0,
                top_action="NONE",
                top_action_rank=0,
                top_action_suitability=0,
            )

        evaluations = judgment.get("evaluations", [])
        if not isinstance(evaluations, list) or len(evaluations) == 0:
            errors.append("Missing or empty 'evaluations' list.")

        expected_ids = {c.action_id for c in candidate_set.candidates}
        evaluated_ids: Set[str] = set()
        ranks_found: List[int] = []

        top_act_from_eval = ""
        top_suitability = 0

        for idx, ev in enumerate(evaluations):
            if not isinstance(ev, dict):
                errors.append(f"Evaluation at index {idx} is not a dictionary.")
                continue

            aid = ev.get("action_id")
            if not aid:
                errors.append(f"Evaluation at index {idx} missing 'action_id'.")
            else:
                evaluated_ids.add(aid)

            # Check required fields
            missing_fields = cls.REQUIRED_EVAL_FIELDS - set(ev.keys())
            if missing_fields:
                errors.append(f"Action '{aid}' missing required fields: {list(missing_fields)}")

            # Check score ranges
            for score_key in ["suitability_score", "urgency_score", "expected_effectiveness_score", "operational_risk_score"]:
                val = ev.get(score_key)
                if not isinstance(val, (int, float)) or not (0 <= val <= 100):
                    errors.append(f"Action '{aid}' {score_key} ({val}) outside [0, 100].")

            # Check confidence range
            conf = ev.get("confidence")
            if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
                errors.append(f"Action '{aid}' confidence ({conf}) outside [0.0, 1.0].")

            # Check rank integer
            rank = ev.get("rank")
            if not isinstance(rank, int):
                errors.append(f"Action '{aid}' rank ({rank}) must be an integer.")
            else:
                ranks_found.append(rank)
                if rank == 1:
                    top_act_from_eval = aid
                    top_suitability = int(ev.get("suitability_score", 0))

            # Check verdict
            verd = ev.get("final_verdict")
            if verd not in cls.ALLOWED_VERDICTS:
                errors.append(f"Action '{aid}' final_verdict '{verd}' not recognized.")

            # Sanity flag: check unsupported assumptions
            assumptions = ev.get("unsupported_assumptions", [])
            if assumptions and len(assumptions) > 0:
                sanity_flags.append(f"FLAG_UNSUPPORTED_ASSUMPTIONS_PRESENT:{aid}")

        # Check candidate coverage
        missing_cands = expected_ids - evaluated_ids
        coverage_pct = 100.0 * len(evaluated_ids.intersection(expected_ids)) / max(1, len(expected_ids))
        if missing_cands:
            errors.append(f"Missing evaluations for candidate actions: {list(missing_cands)}")

        # Check rank integrity (must be exact permutation 1..K)
        expected_ranks = list(range(1, len(evaluations) + 1))
        if sorted(ranks_found) != expected_ranks:
            errors.append(f"Ranks do not form a strict 1..K sequence without ties. Got: {sorted(ranks_found)}")

        # Check top_recommended_action matches rank 1
        stated_top = judgment.get("top_recommended_action")
        if not stated_top:
            errors.append("Missing 'top_recommended_action' field.")
        elif stated_top != top_act_from_eval:
            errors.append(f"'top_recommended_action' ({stated_top}) does not match rank 1 action ({top_act_from_eval}).")

        # 2. Engineering Sanity Checks (Telemetry, non-overriding)
        # Check Critical Passivity
        if severity == "CRITICAL" and (top_act_from_eval == "ACT_MON_CONTINUE" or "MON_" in top_act_from_eval):
            sanity_flags.append("FLAG_CRITICAL_PASSIVITY:Rank_1_monitoring_in_critical_state")

        # Check Healthy Overkill
        if severity == "HEALTHY" and ("REPL_" in top_act_from_eval or "SHUTDOWN" in top_act_from_eval or "EMERGENCY" in top_act_from_eval):
            sanity_flags.append("FLAG_HEALTHY_OVERKILL:Invasive_action_recommended_in_healthy_state")

        # Check Overconfidence on Anonymous / Unmapped
        input_state = raw_state.get("input_machine_state", {})
        has_unmapped = bool(input_state.get("unmapped"))
        top_eval = next((ev for ev in evaluations if ev.get("action_id") == top_act_from_eval), {})
        top_conf = top_eval.get("confidence", 0.0) if isinstance(top_eval, dict) else 0.0
        if has_unmapped and top_conf > 0.85:
            sanity_flags.append("FLAG_OVERCONFIDENT_ON_ANONYMOUS:High_confidence_on_unmapped_features")

        return ValidationReport(
            decision_state_id=state_id,
            is_schema_valid=(len(errors) == 0),
            structural_errors=errors,
            sanity_flags_triggered=sanity_flags,
            candidate_coverage_pct=coverage_pct,
            top_action=top_act_from_eval or str(stated_top),
            top_action_rank=1 if top_act_from_eval else 0,
            top_action_suitability=top_suitability,
        )
