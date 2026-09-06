"""LLM-as-Judge comparative evaluation engine for Phase 3B."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from phase3b.candidate_generator import StateCandidateSet
from phase3b.leakage_guard import LeakageGuard
from phase3b.llm_client import BaseLLMClient


SYSTEM_INSTRUCTION = """You are an expert Senior Industrial Reliability and Prescriptive Maintenance Judge.
Your role is to comparatively evaluate and rank a small set of candidate maintenance actions against an industrial machine's observed decision-time telemetry.

Core Directives:
1. Base your evaluations strictly and exclusively on the provided machine state, sensor readings, and engineering physics.
2. Never invent, hallucinate, or assume unobserved components, sensors, or failure history.
3. Explicitly respect all stated data limitations: note when features are unmapped/anonymous, when observations are instantaneous snapshots without longitudinal history, and when severity is distribution-calibrated.
4. If telemetry is insufficient or ambiguous to justify an invasive intervention, flag `insufficient_information: true` and document your uncertainty.
5. Provide strict, valid JSON matching the requested schema. Do not output markdown commentary or text outside the JSON structure."""


class MaintenanceJudgePromptBuilder:
    """Constructs rigorous, leakage-free prompt payloads for LLM comparative evaluation."""

    @staticmethod
    def build_prompt(
        sanitized_state: Dict[str, Any],
        candidate_set: StateCandidateSet,
        candidate_order: Optional[List[int]] = None,
    ) -> str:
        state_id = sanitized_state.get("decision_state_id", "UNKNOWN")
        asset_ctx = sanitized_state.get("asset_context", {})
        temp_ctx = sanitized_state.get("temporal_context", {})
        severity = sanitized_state.get("decision_severity", "WATCH")
        limitations = sanitized_state.get("data_limitations", [])
        input_state = sanitized_state.get("input_machine_state", {})
        trends = sanitized_state.get("trend_summary", {})

        # Order candidates (allowing permuted orders for stability tests)
        raw_candidates = candidate_set.candidates
        if candidate_order and len(candidate_order) == len(raw_candidates):
            ordered_candidates = [raw_candidates[i] for i in candidate_order]
        else:
            ordered_candidates = raw_candidates

        # Format candidates as compact JSON array
        cand_dicts = []
        for c in ordered_candidates:
            cand_dicts.append({
                "action_id": c.action_id,
                "action_name": c.action_name,
                "category": c.category,
                "description": c.description,
                "intervention_risk": c.intervention_risk,
                "operational_downtime_cost": c.operational_downtime_cost,
                "inclusion_rationale": c.inclusion_rationale,
                "trigger_evidence": c.trigger_evidence,
            })

        prompt_lines = [
            f"# Maintenance Action Evaluation Request: State `{state_id}`",
            "",
            "## 1. Machine Asset & Operational Context",
            f"- **Asset Archetype**: `{asset_ctx.get('machine_archetype', 'unknown')}`",
            f"- **Industrial Domain**: `{asset_ctx.get('operational_domain', 'unknown')}`",
            f"- **Data Dynamics**: `{asset_ctx.get('temporal_type', 'unknown')}`",
            f"- **Calibrated Severity Tier**: **{severity}**",
            "",
            "## 2. Operating Telemetry & Subsystem Conditions",
        ]

        # Present active modalities
        for mod, feats in input_state.items():
            mod_title = mod.replace("_", " ").title()
            prompt_lines.append(f"### Subsystem: {mod_title}")
            for fname, fval in feats.items():
                val_str = f"{fval:.4f}" if isinstance(fval, float) else str(fval)
                prompt_lines.append(f"- `{fname}`: {val_str}")

        # Temporal Context & Trends
        prompt_lines.append("")
        prompt_lines.append("## 3. Temporal Dynamics & Trend Summary")
        if temp_ctx.get("cycle") is not None:
            prompt_lines.append(f"- Cycle: {temp_ctx['cycle']}")
        if temp_ctx.get("sequence_step") is not None:
            prompt_lines.append(f"- Sequence Step: {temp_ctx['sequence_step']}")
        if temp_ctx.get("trajectory_phase") is not None:
            prompt_lines.append(f"- Trajectory Phase: {temp_ctx['trajectory_phase']}")

        if trends:
            for tname, tval in trends.items():
                tval_str = f"{tval:.4f}" if isinstance(tval, float) else str(tval)
                prompt_lines.append(f"- Trend `{tname}`: {tval_str}")
        else:
            prompt_lines.append("- (Instantaneous snapshot only — no longitudinal trend statistics)")

        # Explicit Data Limitations
        prompt_lines.append("")
        prompt_lines.append("## 4. Stated Data Limitations & Epistemic Boundaries")
        for lim in limitations:
            prompt_lines.append(f"- ⚠️ {lim}")

        # Candidates to evaluate
        prompt_lines.append("")
        prompt_lines.append("## 5. Candidate Maintenance Actions to Evaluate")
        prompt_lines.append("Evaluate each of the following candidate actions comparatively:")
        prompt_lines.append("```json")
        prompt_lines.append(json.dumps(cand_dicts, indent=2))
        prompt_lines.append("```")

        # Instructions & Output Schema
        prompt_lines.append("")
        prompt_lines.append("## 6. Evaluation Instructions & Scoring Rubric")
        prompt_lines.append(
            "Comparatively rank and score all candidates against the observed machine state. "
            "Scores must be integers from 0 to 100:\n"
            "- `suitability_score` [0-100]: Technical fit for observed physical state & archetype.\n"
            "- `urgency_score` [0-100]: Time criticality (90-100: immediate action required; 0-20: elective/routine).\n"
            "- `expected_effectiveness_score` [0-100]: Expected success in arresting degradation or protecting asset.\n"
            "- `operational_risk_score` [0-100]: Risk of unnecessary downtime, collateral damage, or waste.\n"
            "- `confidence` [0.0-1.0]: Certainty of rating given the sufficiency and clarity of sensor telemetry.\n"
            "- `rank`: Unique integer from 1 to K (1 = top recommended action; no tied ranks).\n"
            "- `final_verdict`: Exactly one of 'RECOMMENDED', 'ACCEPTABLE_ALTERNATIVE', 'INAPPROPRIATE_AT_CURRENT_TIME', 'UNSAFE'."
        )

        prompt_lines.append("")
        prompt_lines.append("Respond ONLY with a JSON object matching this exact structure:")
        prompt_lines.append("""```json
{
  "evaluations": [
    {
      "action_id": "<action_id>",
      "rank": 1,
      "suitability_score": <0-100>,
      "urgency_score": <0-100>,
      "expected_effectiveness_score": <0-100>,
      "operational_risk_score": <0-100>,
      "confidence": <0.0-1.0>,
      "evidence_used": ["<sensor or trend cited>"],
      "reasoning_summary": "<concise engineering justification>",
      "unsupported_assumptions": ["<any unverified assumptions or empty list>"],
      "final_verdict": "RECOMMENDED"
    }
  ],
  "top_recommended_action": "<action_id of rank 1>",
  "alternative_actions": ["<action_id of acceptable alternatives>"],
  "insufficient_information": false,
  "uncertainty_explanation": "<explanation if confidence is low, else empty string>"
}
```""")

        return "\n".join(prompt_lines)


class MaintenanceJudge:
    """Executes comparative LLM-as-Judge evaluations on DecisionStates."""

    def __init__(self, client: BaseLLMClient):
        self.client = client

    def evaluate_state(
        self,
        raw_state: Dict[str, Any],
        candidate_set: StateCandidateSet,
        temperature: float = 0.1,
        candidate_order: Optional[List[int]] = None,
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        # 1. Sanitize state to guarantee zero leakage
        sanitized_state = LeakageGuard.sanitize_state_for_prompt(raw_state)
        LeakageGuard.assert_zero_leakage(sanitized_state)

        # 2. Build prompt
        prompt = MaintenanceJudgePromptBuilder.build_prompt(
            sanitized_state=sanitized_state,
            candidate_set=candidate_set,
            candidate_order=candidate_order,
        )

        # 3. Double-check entire prompt string for any leakage
        LeakageGuard.assert_zero_leakage(prompt)

        # 4. Dispatch to LLM client
        raw_response, meta = self.client.generate_structured_evaluation(
            prompt=prompt,
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=temperature,
        )

        # 5. Robust JSON parse
        parsed_json = self._parse_json_response(raw_response)

        return parsed_json, raw_response, meta

    @staticmethod
    def _parse_json_response(text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        # Strip markdown json codeblock if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback regex extraction of outermost JSON object
            match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise ValueError(f"Could not parse valid JSON from LLM response:\n{text[:200]}")
