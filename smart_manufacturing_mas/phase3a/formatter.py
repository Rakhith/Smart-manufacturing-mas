"""LLM-ready prompt serializer and Machine Health Card formatter for Phase 3A."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from phase3a.schema import DecisionState


class LLMStateFormatter:
    """Formats DecisionState objects into compact, interpretable representations for future LLM evaluation."""

    @staticmethod
    def to_markdown_health_card(state: DecisionState) -> str:
        """Formats a compact human- and LLM-readable Markdown Machine Health Card (~350-650 tokens)."""
        lines = [
            f"### Machine Health State: `{state.decision_state_id}`",
            f"- **Asset Archetype**: `{state.asset_context.get('machine_archetype', 'unknown')}`",
            f"- **Asset ID**: `{state.asset_context.get('asset_id', 'unknown')}`",
            f"- **Domain**: {state.asset_context.get('operational_domain', 'unknown')}",
            f"- **Observed Severity Tier**: **{state.decision_severity}**",
            f"- **Decision Rationale**: {state.decision_relevance_rationale}",
            "",
            "#### Current Subsystem Conditions",
        ]

        # Group by semantic modalities
        active_modalities = {
            mod: feats for mod, feats in state.input_machine_state.items() if feats and mod != "unmapped"
        }

        if not active_modalities:
            # For unmapped datasets (e.g. Metal Etch)
            unmapped = state.input_machine_state.get("unmapped", {})
            if unmapped:
                lines.append(f"- **Process Parameters (Unmapped)**: {len(unmapped)} channels active")
                sample_feats = list(unmapped.items())[:6]
                for k, v in sample_feats:
                    val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                    lines.append(f"  - `{k}`: {val_str}")
                if len(unmapped) > 6:
                    lines.append(f"  - *(and {len(unmapped) - 6} additional parameters)*")
        else:
            for mod, feats in active_modalities.items():
                mod_title = mod.replace("_", " ").title()
                lines.append(f"- **{mod_title} State**:")
                # Show top 5 features per modality to prevent prompt bloat
                sample_feats = list(feats.items())[:5]
                for k, v in sample_feats:
                    val_str = f"{v:.3f}" if isinstance(v, float) else str(v)
                    lines.append(f"  - `{k}`: {val_str}")
                if len(feats) > 5:
                    lines.append(f"  - *(+{len(feats) - 5} additional {mod} metrics)*")

        # Temporal Context & Trends
        lines.append("")
        lines.append("#### Temporal Context & Dynamics")
        temp = state.temporal_context
        if temp.get("cycle") is not None:
            lines.append(f"- Operating Cycle: {temp['cycle']}")
        if temp.get("sequence_step") is not None:
            lines.append(f"- Sequence Step: {temp['sequence_step']}")
        if temp.get("trajectory_phase") is not None:
            lines.append(f"- Trajectory Phase: {temp['trajectory_phase']}")

        trends = state.trend_summary
        if trends:
            lines.append("- Active Trend & Delta Indicators:")
            for k, v in list(trends.items())[:4]:
                v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                lines.append(f"  - `{k}`: {v_str}")
        else:
            lines.append("- Instantaneous Operating Snapshot (Static / steady-state baseline)")

        return "\n".join(lines) + "\n"

    @staticmethod
    def to_jsonl_payload(state: DecisionState, include_ground_truth: bool = False) -> Dict[str, Any]:
        """Produces a structured JSON payload for programmatic LLM prompt templating."""
        payload = {
            "decision_state_id": state.decision_state_id,
            "dataset_id": state.dataset_id,
            "asset_context": state.asset_context,
            "temporal_context": state.temporal_context,
            "decision_severity": state.decision_severity,
            "decision_relevance_rationale": state.decision_relevance_rationale,
            "input_machine_state": state.input_machine_state,
            "trend_summary": state.trend_summary,
            "provenance": state.provenance,
        }
        if include_ground_truth:
            # Quarantined under explicit evaluation key
            payload["evaluation_ground_truth_context"] = state.ground_truth_context
        return payload

    @classmethod
    def export_all(
        cls,
        states: List[DecisionState],
        output_dir: Path,
        max_md_samples: int = 50,
    ) -> None:
        """Exports JSONL corpus and sample Markdown cards into output directory."""
        jsonl_dir = output_dir / "state_cards_jsonl"
        md_dir = output_dir / "state_cards_md"
        jsonl_dir.mkdir(parents=True, exist_ok=True)
        md_dir.mkdir(parents=True, exist_ok=True)

        # 1. Full JSONL corpus (clean inputs only, zero leakage)
        jsonl_path = jsonl_dir / "decision_state_cards.jsonl"
        eval_jsonl_path = jsonl_dir / "decision_state_cards_with_eval_truth.jsonl"

        with open(jsonl_path, "w", encoding="utf-8") as f_in, open(eval_jsonl_path, "w", encoding="utf-8") as f_eval:
            for s in states:
                f_in.write(json.dumps(cls.to_jsonl_payload(s, include_ground_truth=False)) + "\n")
                f_eval.write(json.dumps(cls.to_jsonl_payload(s, include_ground_truth=True)) + "\n")

        # 2. Export sample Markdown cards for inspection
        for i, s in enumerate(states[:max_md_samples]):
            md_card = cls.to_markdown_health_card(s)
            safe_name = f"{s.decision_state_id}.md"
            (md_dir / safe_name).write_text(md_card, encoding="utf-8")
