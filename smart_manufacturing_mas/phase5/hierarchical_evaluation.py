"""Hierarchical and Graded Action Relevance Evaluation Engine.

Evaluates recommender predictions across 4 hierarchical tiers:
LEVEL 1: Action Family (MONITORING, INSPECTION, CORRECTIVE, REPLACEMENT, OPERATIONAL)
LEVEL 2: Action Category
LEVEL 3: Exact Canonical Action ID
LEVEL 4: Action + Component (Procedure + Physical Subsystem)

Also implements the deterministic ontology-derived Graded Action Relevance Score:
- Exact Action + Component: 1.0
- Exact Action: 0.85
- Same Category + Same Component: 0.70
- Same Family + Same Component: 0.55
- Same Family: 0.40
- Unrelated: 0.0
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class HierarchicalSummary:
    total_instances: int
    family_agreement_rate: float
    category_agreement_rate: float
    action_exact_agreement_rate: float
    action_component_agreement_rate: float
    mean_graded_relevance_score: float
    graded_distribution: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


ACTION_TO_FAMILY = {
    "ACT_MON_CONTINUE": "MONITORING",
    "ACT_MON_ENHANCED": "MONITORING",
    "ACT_MON_PARAMETER_LOG": "MONITORING",
    "ACT_INSP_VIBRATION": "INSPECTION",
    "ACT_INSP_LUBRICATION": "INSPECTION",
    "ACT_INSP_THERMAL_ELECTRICAL": "INSPECTION",
    "ACT_INSP_TOOL_WEAR": "INSPECTION",
    "ACT_INSP_PRESSURE_HYDRAULIC": "INSPECTION",
    "ACT_INSP_SUBSYSTEM": "INSPECTION",
    "ACT_CORR_LUBRICATE": "CORRECTIVE",
    "ACT_CORR_CLEAN_PURGE": "CORRECTIVE",
    "ACT_CORR_CALIBRATE": "CORRECTIVE",
    "ACT_CORR_FASTEN_ALIGN": "CORRECTIVE",
    "ACT_REPL_TOOL_INSERT": "REPLACEMENT",
    "ACT_REPL_BEARING": "REPLACEMENT",
    "ACT_REPL_SEAL_VALVE": "REPLACEMENT",
    "ACT_REPL_OVERHAUL": "REPLACEMENT",
    "ACT_OP_ADJUST_PARAMS": "OPERATIONAL",
    "ACT_OP_DERATE_LOAD": "OPERATIONAL",
    "ACT_OP_CONTROLLED_SHUTDOWN": "OPERATIONAL",
    "ACT_OP_EMERGENCY_STOP": "OPERATIONAL",
}

ACTION_TO_COMPONENT = {
    "ACT_REPL_BEARING": "BEARING",
    "ACT_INSP_VIBRATION": "BEARING",
    "ACT_REPL_SEAL_VALVE": "SEAL_VALVE",
    "ACT_INSP_PRESSURE_HYDRAULIC": "SEAL_VALVE",
    "ACT_CORR_FASTEN_ALIGN": "ALIGNMENT",
    "ACT_CORR_CLEAN_PURGE": "COOLING_THERMAL",
    "ACT_INSP_THERMAL_ELECTRICAL": "ELECTRICAL_MOTOR",
    "ACT_CORR_LUBRICATE": "LUBRICATION",
    "ACT_INSP_LUBRICATION": "LUBRICATION",
    "ACT_CORR_CALIBRATE": "SENSORS",
    "ACT_REPL_TOOL_INSERT": "TOOL",
    "ACT_INSP_TOOL_WEAR": "TOOL",
    "ACT_REPL_OVERHAUL": "SUBSYSTEM",
    "ACT_INSP_SUBSYSTEM": "SUBSYSTEM",
    "ACT_OP_ADJUST_PARAMS": "PROCESS",
    "ACT_OP_DERATE_LOAD": "PROCESS",
    "ACT_OP_CONTROLLED_SHUTDOWN": "SYSTEM",
    "ACT_OP_EMERGENCY_STOP": "SYSTEM",
    "ACT_MON_CONTINUE": "SYSTEM",
    "ACT_MON_ENHANCED": "SYSTEM",
    "ACT_MON_PARAMETER_LOG": "SYSTEM",
}


class HierarchicalActionEvaluator:
    """Evaluates recommendations across family, category, exact action, and component."""

    @staticmethod
    def get_family(action_id: str) -> str:
        return ACTION_TO_FAMILY.get(action_id, "UNKNOWN")

    @staticmethod
    def get_component(action_id: str) -> str:
        return ACTION_TO_COMPONENT.get(action_id, "GENERAL")

    @classmethod
    def normalize_component(cls, comp: Optional[str]) -> str:
        if not comp:
            return "GENERAL"
        c = str(comp).strip().upper()
        aliases = {
            "LUBRICATION_SYSTEM": "LUBRICATION",
            "SEAL": "SEAL_VALVE",
            "VALVE": "SEAL_VALVE",
            "MOTOR": "ELECTRICAL_MOTOR",
            "COOLING": "COOLING_THERMAL",
        }
        return aliases.get(c, c)

    @classmethod
    def evaluate_pair(
        cls,
        predicted_action: str,
        target_action: str,
        target_component: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluates hierarchical agreement between a single predicted action and target."""
        pred_fam = cls.get_family(predicted_action)
        tgt_fam = cls.get_family(target_action)

        pred_comp = cls.normalize_component(cls.get_component(predicted_action))
        tgt_comp = cls.normalize_component(target_component or cls.get_component(target_action))

        is_exact_action = (predicted_action == target_action)
        is_family_match = (pred_fam == tgt_fam)
        is_component_match = (pred_comp == tgt_comp)
        is_action_and_component = (is_exact_action and is_component_match)

        # Deterministic graded relevance score
        if is_action_and_component:
            graded_score = 1.00
            graded_level = "EXACT_ACTION_AND_COMPONENT"
        elif is_exact_action:
            graded_score = 0.85
            graded_level = "EXACT_ACTION"
        elif is_family_match and is_component_match:
            graded_score = 0.70
            graded_level = "FAMILY_AND_COMPONENT"
        elif is_component_match:
            graded_score = 0.55
            graded_level = "COMPONENT_ONLY"
        elif is_family_match:
            graded_score = 0.40
            graded_level = "FAMILY_ONLY"
        else:
            graded_score = 0.00
            graded_level = "UNRELATED"

        return {
            "predicted_action": predicted_action,
            "target_action": target_action,
            "predicted_family": pred_fam,
            "target_family": tgt_fam,
            "predicted_component": pred_comp,
            "target_component": tgt_comp,
            "exact_action_match": is_exact_action,
            "family_match": is_family_match,
            "component_match": is_component_match,
            "action_and_component_match": is_action_and_component,
            "graded_relevance_score": graded_score,
            "graded_relevance_level": graded_level,
        }

    @classmethod
    def evaluate_batch(
        cls,
        predictions: List[str],
        targets: List[str],
        target_components: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Computes summary hierarchical metrics across a list of predictions."""
        n = len(predictions)
        if n == 0:
            return {
                "total_records": 0,
                "exact_action_accuracy": 0.0,
                "family_accuracy": 0.0,
                "component_accuracy": 0.0,
                "action_and_component_accuracy": 0.0,
                "mean_graded_score": 0.0,
                "graded_distribution": {},
            }

        comps = target_components or [None] * n
        results = [
            cls.evaluate_pair(pred, tgt, comp)
            for pred, tgt, comp in zip(predictions, targets, comps)
        ]

        exact_cnt = sum(1 for r in results if r["exact_action_match"])
        fam_cnt = sum(1 for r in results if r["family_match"])
        comp_cnt = sum(1 for r in results if r["component_match"])
        both_cnt = sum(1 for r in results if r["action_and_component_match"])
        scores = [r["graded_relevance_score"] for r in results]

        level_dist = {}
        for r in results:
            lvl = r["graded_relevance_level"]
            level_dist[lvl] = level_dist.get(lvl, 0) + 1

        return {
            "total_records": n,
            "exact_action_accuracy": round(exact_cnt / n, 4),
            "exact_action_numerator": exact_cnt,
            "family_accuracy": round(fam_cnt / n, 4),
            "family_numerator": fam_cnt,
            "component_accuracy": round(comp_cnt / n, 4),
            "component_numerator": comp_cnt,
            "action_and_component_accuracy": round(both_cnt / n, 4),
            "action_and_component_numerator": both_cnt,
            "mean_graded_score": round(float(np.mean(scores)), 4),
            "graded_distribution": {k: round(v / n, 4) for k, v in level_dist.items()},
            "graded_counts": level_dist,
            "detailed_records": results,
        }

    def evaluate_records(
        self,
        records: List[Dict[str, Any]],
    ) -> Tuple[HierarchicalSummary, List[Dict[str, Any]]]:
        """Evaluates a list of prediction records and returns summary and enriched records."""
        enriched = []
        fam_matches = 0
        cat_matches = 0
        act_matches = 0
        act_comp_matches = 0
        graded_scores = []
        graded_dist: Dict[str, int] = {}

        for r in records:
            r_copy = dict(r)
            pred = r.get("predicted_action_top1", r.get("predicted_action", "ACT_MON_CONTINUE"))
            tgt = r.get("target_action", "ACT_MON_CONTINUE")
            tgt_comp = r.get("target_component")
            eval_res = self.evaluate_pair(pred, tgt, tgt_comp)

            # Check category match
            pred_fam = eval_res["predicted_family"]
            tgt_fam = eval_res["target_family"]
            # Category match: family match or exact action or category if mapped
            is_cat = (pred_fam == tgt_fam)

            is_fam = eval_res["family_match"]
            is_act = eval_res["exact_action_match"]
            is_both = eval_res["action_and_component_match"]
            g_score = eval_res["graded_relevance_score"]
            g_lvl = eval_res["graded_relevance_level"]

            if is_fam:
                fam_matches += 1
            if is_cat:
                cat_matches += 1
            if is_act:
                act_matches += 1
            if is_both:
                act_comp_matches += 1

            graded_scores.append(g_score)
            graded_dist[g_lvl] = graded_dist.get(g_lvl, 0) + 1

            r_copy["hierarchical_family_match"] = is_fam
            r_copy["hierarchical_action_match"] = is_act
            r_copy["action_component_match"] = is_both
            r_copy["graded_relevance"] = g_score
            r_copy["graded_relevance_score"] = g_score
            r_copy["graded_level"] = g_lvl
            enriched.append(r_copy)

        n = max(1, len(records))
        summary = HierarchicalSummary(
            total_instances=len(records),
            family_agreement_rate=round(fam_matches / n, 4),
            category_agreement_rate=round(cat_matches / n, 4),
            action_exact_agreement_rate=round(act_matches / n, 4),
            action_component_agreement_rate=round(act_comp_matches / n, 4),
            mean_graded_relevance_score=round(float(np.mean(graded_scores)), 4) if graded_scores else 0.0,
            graded_distribution=graded_dist,
        )

        return summary, enriched


# Alias
HierarchicalEvaluator = HierarchicalActionEvaluator

