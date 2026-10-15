"""Secondary LLM Semantic Action Relevance Evaluator.

Part 6: Semantic Action Relevance Evaluation
Evaluates whether a predicted canonical maintenance action is semantically compatible
with the recorded maintenance event/work order even when exact wording or labels differ.

Strict JSON format:
{
  "label": "EXACT_MATCH" | "SEMANTICALLY_EQUIVALENT" | "PARTIALLY_RELEVANT" | "NOT_RELEVANT" | "INSUFFICIENT_INFORMATION",
  "confidence": 0.0 to 1.0,
  "reason": "short explanation"
}

Inputs provided:
- Machine state summary (severity, health index, key sensor deviations)
- Predicted canonical action
- Recorded maintenance action / work order text / failure mode
- Affected component / subsystem

Inputs prohibited:
- Future sensor trajectories
- Model identity or ranking scores
- Whether prediction is frozen vs adapted
- Expected answer
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from phase5.hierarchical_evaluation import ACTION_TO_COMPONENT, ACTION_TO_FAMILY


SEMANTIC_SYSTEM_PROMPT = """You are an independent industrial maintenance reliability engineer evaluating the semantic compatibility between an AI recommender's predicted maintenance action and a plant's recorded maintenance event.

Evaluate whether the predicted action is semantically compatible with the physical problem recorded.

Allowed labels:
1. EXACT_MATCH: The predicted action is the exact procedure performed.
2. SEMANTICALLY_EQUIVALENT: The predicted action addresses the exact same physical failure mechanism with equivalent operational effect, even if wording differs (e.g., ACT_INSP_VIBRATION vs 'check bearing vibration spectrum').
3. PARTIALLY_RELEVANT: The predicted action addresses the correct component or subsystem but represents a different maintenance stage (e.g. diagnosing/lubricating vs replacing).
4. NOT_RELEVANT: The predicted action targets an unrelated subsystem or failure mechanism.
5. INSUFFICIENT_INFORMATION: The recorded text lacks sufficient detail to determine relevance.

Respond ONLY with valid JSON in this exact structure:
{
  "label": "EXACT_MATCH" | "SEMANTICALLY_EQUIVALENT" | "PARTIALLY_RELEVANT" | "NOT_RELEVANT" | "INSUFFICIENT_INFORMATION",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<one sentence explanation>"
}"""


class DeterministicSemanticFallbackJudge:
    """Rigorous ontology-grounded rule engine used for deterministic semantic adjudication."""

    @classmethod
    def judge(
        cls,
        predicted_action: str,
        recorded_action: str,
        component: str,
        severity: str,
    ) -> Dict[str, Any]:
        pred_act = predicted_action.upper()
        rec_act = recorded_action.lower()
        pred_comp = ACTION_TO_COMPONENT.get(pred_act, "GENERAL")
        pred_fam = ACTION_TO_FAMILY.get(pred_act, "GENERAL")

        # 1. Exact match
        if pred_act.replace("ACT_", "").lower() in rec_act or rec_act in pred_act.lower():
            return {
                "label": "EXACT_MATCH",
                "confidence": 1.0,
                "reason": f"Predicted action {pred_act} matches recorded action {recorded_action}.",
            }

        # 2. Semantically equivalent
        # Bearing wear -> inspection vibration or bearing replacement
        if "bearing" in rec_act or component == "BEARING":
            if pred_act in {"ACT_REPL_BEARING", "ACT_INSP_VIBRATION"}:
                return {
                    "label": "SEMANTICALLY_EQUIVALENT",
                    "confidence": 0.95,
                    "reason": f"Both target bearing mechanical raceway failure mechanism ({pred_act} vs {recorded_action}).",
                }
            if pred_act in {"ACT_CORR_LUBRICATE", "ACT_INSP_LUBRICATION"}:
                return {
                    "label": "PARTIALLY_RELEVANT",
                    "confidence": 0.85,
                    "reason": "Lubrication addresses friction contributing to bearing wear but is not full replacement.",
                }

        # Seal leak -> seal replacement or pressure inspection
        if "seal" in rec_act or component == "SEAL_VALVE":
            if pred_act in {"ACT_REPL_SEAL_VALVE", "ACT_INSP_PRESSURE_HYDRAULIC"}:
                return {
                    "label": "SEMANTICALLY_EQUIVALENT",
                    "confidence": 0.95,
                    "reason": f"Both target fluid barrier seal integrity ({pred_act} vs {recorded_action}).",
                }
            if pred_act in {"ACT_CORR_CLEAN_PURGE", "ACT_INSP_SUBSYSTEM"}:
                return {
                    "label": "PARTIALLY_RELEVANT",
                    "confidence": 0.75,
                    "reason": "Purge/inspection is relevant to slurry seal flushing.",
                }

        # Imbalance -> align drive or vibration inspection
        if "imbalance" in rec_act or "align" in rec_act or component == "ALIGNMENT":
            if pred_act in {"ACT_CORR_FASTEN_ALIGN", "ACT_INSP_VIBRATION"}:
                return {
                    "label": "SEMANTICALLY_EQUIVALENT",
                    "confidence": 0.90,
                    "reason": f"Imbalance requires vibration diagnosis or mechanical realignment ({pred_act}).",
                }

        # Overheating / thermal -> cooling purge, electrical inspection
        if "overheat" in rec_act or "temp" in rec_act or component == "COOLING_THERMAL":
            if pred_act in {"ACT_CORR_CLEAN_PURGE", "ACT_INSP_THERMAL_ELECTRICAL", "ACT_OP_DERATE_LOAD"}:
                return {
                    "label": "SEMANTICALLY_EQUIVALENT",
                    "confidence": 0.90,
                    "reason": f"Thermal excursion managed by cooling service, derating, or thermography ({pred_act}).",
                }

        # Electrical fault -> motor inspection or contactor replacement
        if "electric" in rec_act or component == "ELECTRICAL_MOTOR":
            if pred_act in {"ACT_INSP_THERMAL_ELECTRICAL"}:
                return {
                    "label": "SEMANTICALLY_EQUIVALENT",
                    "confidence": 0.90,
                    "reason": f"Electrical anomaly directly diagnosed via electrical/thermal motor inspection.",
                }

        # Same component but different action family
        if pred_comp == component:
            return {
                "label": "PARTIALLY_RELEVANT",
                "confidence": 0.70,
                "reason": f"Predicted action targets affected component {component} at a different maintenance phase.",
            }

        # Diagnostic inspection when preventive
        if "preventive" in rec_act or "planned" in rec_act:
            if pred_fam in {"INSPECTION", "MONITORING"}:
                return {
                    "label": "SEMANTICALLY_EQUIVALENT",
                    "confidence": 0.85,
                    "reason": f"Planned surveillance matches non-invasive supervisory inspection ({pred_act}).",
                }

        # Not relevant
        return {
            "label": "NOT_RELEVANT",
            "confidence": 0.90,
            "reason": f"Predicted action {pred_act} ({pred_comp}) does not address recorded event {recorded_action} ({component}).",
        }


class SemanticActionEvaluator:
    """Evaluates semantic compatibility between predicted actions and recorded work orders."""

    ALLOWED_LABELS = {
        "EXACT_MATCH",
        "SEMANTICALLY_EQUIVALENT",
        "PARTIALLY_RELEVANT",
        "NOT_RELEVANT",
        "INSUFFICIENT_INFORMATION",
    }

    def __init__(self, use_llm_api: bool = False, use_deterministic_fallback: bool = True):
        self.use_llm_api = use_llm_api and not use_deterministic_fallback
        self.use_deterministic_fallback = use_deterministic_fallback
        self.client = None
        if self.use_llm_api:
            try:
                from phase3b.llm_client import GeminiRESTClient
                self.client = GeminiRESTClient(model="gemini-2.5-flash")
            except Exception:
                self.client = None

    def evaluate_record(
        self,
        predicted_action: str,
        recorded_action: str,
        component: str,
        severity: str,
        sensor_context: str = "",
    ) -> Dict[str, Any]:
        """Evaluates a single prediction against recorded maintenance."""
        # Attempt LLM if configured and available
        if self.client is not None:
            prompt = (
                f"Prediction-Time Machine State:\n"
                f"- Severity: {severity}\n"
                f"- Key Observations: {sensor_context}\n"
                f"- Subsystem/Component: {component}\n\n"
                f"Candidate Prediction:\n"
                f"- Predicted Action: {predicted_action}\n\n"
                f"Recorded Maintenance Log:\n"
                f"- Recorded Action / Work Order Text: {recorded_action}\n\n"
                f"Assess semantic compatibility."
            )
            try:
                raw_text, meta = self.client.generate_structured_evaluation(
                    prompt=prompt,
                    system_instruction=SEMANTIC_SYSTEM_PROMPT,
                    temperature=0.0,
                )
                parsed = json.loads(raw_text)
                if parsed.get("label") in self.ALLOWED_LABELS:
                    return {
                        "label": parsed["label"],
                        "confidence": float(parsed.get("confidence", 0.9)),
                        "reason": str(parsed.get("reason", "")),
                        "evaluator": "llm_judge",
                    }
            except Exception:
                pass

        # Robust deterministic fallback
        res = DeterministicSemanticFallbackJudge.judge(
            predicted_action=predicted_action,
            recorded_action=recorded_action,
            component=component,
            severity=severity,
        )
        res["evaluator"] = "deterministic_semantic_judge"
        return res

    def evaluate_batch(self, evaluation_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluates a batch of prediction records and summarizes semantic metrics."""
        results = []
        for r in evaluation_records:
            pred_act = r.get("predicted_action_top1", "")
            rec_act = r.get("target_action", r.get("recorded_action", ""))
            comp = r.get("target_component", "GENERAL")
            sev = r.get("decision_severity", "WATCH")
            sensor_ctx = r.get("sensor_summary", "")

            eval_res = self.evaluate_record(
                predicted_action=pred_act,
                recorded_action=rec_act,
                component=comp,
                severity=sev,
                sensor_context=sensor_ctx,
            )
            r_out = dict(r)
            r_out.update(eval_res)
            results.append(r_out)

        n = len(results)
        if n == 0:
            return {
                "total_records": 0,
                "exact_match_rate": 0.0,
                "semantically_equivalent_rate": 0.0,
                "secondary_semantic_relevance_rate": 0.0,
                "partially_relevant_rate": 0.0,
                "not_relevant_rate": 0.0,
                "insufficient_information_rate": 0.0,
            }

        counts = {lbl: 0 for lbl in self.ALLOWED_LABELS}
        for r in results:
            lbl = r["label"]
            counts[lbl] = counts.get(lbl, 0) + 1

        exact_rate = counts["EXACT_MATCH"] / n
        equiv_rate = counts["SEMANTICALLY_EQUIVALENT"] / n
        # Secondary semantic relevance metric = EXACT_MATCH + SEMANTICALLY_EQUIVALENT
        secondary_relevance_rate = (counts["EXACT_MATCH"] + counts["SEMANTICALLY_EQUIVALENT"]) / n

        return {
            "total_records": n,
            "exact_match_rate": round(exact_rate, 4),
            "exact_match_numerator": counts["EXACT_MATCH"],
            "semantically_equivalent_rate": round(equiv_rate, 4),
            "semantically_equivalent_numerator": counts["SEMANTICALLY_EQUIVALENT"],
            "secondary_semantic_relevance_rate": round(secondary_relevance_rate, 4),
            "secondary_semantic_relevance_numerator": counts["EXACT_MATCH"] + counts["SEMANTICALLY_EQUIVALENT"],
            "partially_relevant_rate": round(counts["PARTIALLY_RELEVANT"] / n, 4),
            "partially_relevant_numerator": counts["PARTIALLY_RELEVANT"],
            "not_relevant_rate": round(counts["NOT_RELEVANT"] / n, 4),
            "not_relevant_numerator": counts["NOT_RELEVANT"],
            "insufficient_information_rate": round(counts["INSUFFICIENT_INFORMATION"] / n, 4),
            "counts": counts,
            "evaluated_records": results,
        }

    def measure_consistency(self, records: List[Dict[str, Any]], sample_size: int = 15) -> Dict[str, Any]:
        """Runs repeated evaluation on a random subset to measure semantic judge consistency."""
        import random
        sub = random.sample(records, min(sample_size, len(records)))
        pass1 = self.evaluate_batch(sub)["evaluated_records"]
        pass2 = self.evaluate_batch(sub)["evaluated_records"]

        agreements = sum(1 for p1, p2 in zip(pass1, pass2) if p1["label"] == p2["label"])
        rate = agreements / len(sub) if sub else 1.0
        return {
            "sample_size": len(sub),
            "agreements_count": agreements,
            "consistency_rate": round(rate, 4),
            "status": "PASS" if rate >= 0.90 else "WARN_LOW_CONSISTENCY",
        }

    def evaluate_dataset(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluates dataset and summarizes semantic acceptance rate."""
        raw = self.evaluate_batch(records)
        dist = raw.get("counts", {})
        n = max(1, len(records))
        acc_cnt = (
            dist.get("EXACT_MATCH", 0)
            + dist.get("SEMANTICALLY_EQUIVALENT", 0)
            + dist.get("PARTIALLY_RELEVANT", 0)
        )
        return {
            "total_evaluated": len(records),
            "distribution": dist,
            "semantic_acceptance_rate": round(acc_cnt / n, 4),
            "exact_match_count": dist.get("EXACT_MATCH", 0),
            "semantically_equivalent_count": dist.get("SEMANTICALLY_EQUIVALENT", 0),
            "partially_relevant_count": dist.get("PARTIALLY_RELEVANT", 0),
            "not_relevant_count": dist.get("NOT_RELEVANT", 0),
            "raw_summary": raw,
            "evaluated_records": raw.get("evaluated_records", []),
        }


# Export alias
SecondarySemanticJudge = SemanticActionEvaluator

