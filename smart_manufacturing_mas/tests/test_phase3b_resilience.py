import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phase3b.judge import MaintenanceJudge
from phase3b.validator import JudgmentValidator
from phase3b.candidate_generator import CandidateGenerator, StateCandidateSet, CandidateAction
from phase3b.ontology import ActionOntology
from phase3b.llm_client import (
    HeuristicMockClient,
    MultiProviderDispatcherClient,
    GeminiRESTClient,
    BaseLLMClient,
)


class TestPhase3BResilience(unittest.TestCase):

    def test_json_parser_markdown_and_thinking_tags(self):
        raw_input = """<think>Let me evaluate this carefully...</think>
```json
{
  "evaluations": [
    {
      "action_id": "ACT_MON_CONTINUE",
      "rank": 1,
      "suitability_score": 90,
      "urgency_score": 10,
      "expected_effectiveness_score": 85,
      "operational_risk_score": 5,
      "confidence": 0.95,
      "evidence_used": ["temp_normal"],
      "reasoning_summary": "Machine is running within normal parameters.",
      "unsupported_assumptions": [],
      "final_verdict": "RECOMMENDED"
    }
  ],
  "top_recommended_action": "ACT_MON_CONTINUE",
  "alternative_actions": [],
  "insufficient_information": false,
  "uncertainty_explanation": ""
}
```
Some trailing commentary here."""
        parsed = MaintenanceJudge._parse_json_response(raw_input)
        self.assertEqual(parsed["top_recommended_action"], "ACT_MON_CONTINUE")
        self.assertEqual(len(parsed["evaluations"]), 1)

    def test_json_parser_trailing_commas(self):
        raw_input = """{
  "evaluations": [
    {
      "action_id": "ACT_MON_CONTINUE",
      "rank": 1,
      "suitability_score": 90,
      "urgency_score": 10,
      "expected_effectiveness_score": 85,
      "operational_risk_score": 5,
      "confidence": 0.95,
      "evidence_used": ["temp_normal",],
      "reasoning_summary": "Machine is running within normal parameters.",
      "unsupported_assumptions": [],
      "final_verdict": "RECOMMENDED",
    },
  ],
  "top_recommended_action": "ACT_MON_CONTINUE",
}"""
        parsed = MaintenanceJudge._parse_json_response(raw_input)
        self.assertEqual(parsed["top_recommended_action"], "ACT_MON_CONTINUE")

    def test_json_parser_truncated_json(self):
        raw_input = """{
  "evaluations": [
    {
      "action_id": "ACT_MON_CONTINUE",
      "rank": 1,
      "suitability_score": 90,
      "urgency_score": 10,
      "expected_effectiveness_score": 85,
      "operational_risk_score": 5,
      "confidence": 0.95,
      "evidence_used": [],
      "reasoning_summary": "Valid action.",
      "unsupported_assumptions": [],
      "final_verdict": "RECOMMENDED"
    }"""
        parsed = MaintenanceJudge._parse_json_response(raw_input)
        self.assertIn("evaluations", parsed)
        self.assertEqual(parsed["evaluations"][0]["action_id"], "ACT_MON_CONTINUE")

    def test_validator_auto_healing_missing_candidate_and_rank_ties(self):
        cands = [
            CandidateAction(action_id="ACT_MON_CONTINUE", action_name="Continue", category="MONITORING", description="", intervention_risk="LOW", operational_downtime_cost="ZERO", inclusion_rationale="test", trigger_evidence=[]),
            CandidateAction(action_id="ACT_INSP_TOOL_WEAR", action_name="Inspect", category="INSPECTION", description="", intervention_risk="LOW", operational_downtime_cost="LOW", inclusion_rationale="test", trigger_evidence=[]),
            CandidateAction(action_id="ACT_REPL_TOOL_INSERT", action_name="Replace", category="REPLACEMENT", description="", intervention_risk="MEDIUM", operational_downtime_cost="MEDIUM", inclusion_rationale="test", trigger_evidence=[]),
        ]
        cset = StateCandidateSet(
            decision_state_id="DS_TEST_001",
            machine_archetype="cnc_mill",
            severity="WATCH",
            candidates=cands,
        )

        # Malformed judgment: missing ACT_REPL_TOOL_INSERT, tied ranks, string scores
        bad_judgment = {
            "evaluations": [
                {
                    "action_id": "ACT_MON_CONTINUE,",  # trailing comma
                    "rank": "1",                        # string rank
                    "suitability_score": "88",          # string score
                    "urgency_score": 20,
                    "expected_effectiveness_score": 75,
                    "operational_risk_score": 10,
                    "confidence": "0.9",               # string confidence
                    "reasoning_summary": "Normal.",
                    "final_verdict": "ACCEPTABLE",     # alias
                },
                {
                    "action_id": "ACT_INSP_TOOL_WEAR",
                    "rank": 1,                         # TIED RANK!
                    "suitability_score": 92,
                    "urgency_score": 40,
                    "expected_effectiveness_score": 85,
                    "operational_risk_score": 15,
                    "confidence": 0.85,
                    "reasoning_summary": "Inspect soon.",
                    "final_verdict": "RECOMMEND",      # alias
                },
            ],
            "top_recommended_action": "",
        }

        raw_state = {
            "decision_state_id": "DS_TEST_001",
            "dataset_id": "ai4i_2020",
            "decision_severity": "WATCH",
            "asset_context": {"machine_archetype": "cnc_mill"},
        }

        healed = JudgmentValidator.normalize_judgment(bad_judgment, cset)
        report = JudgmentValidator.validate_and_audit(healed, cset, raw_state)

        self.assertTrue(report.is_schema_valid, f"Validation failed with errors: {report.structural_errors}")
        self.assertEqual(len(healed["evaluations"]), 3, "Missing candidate was not injected")
        ranks = [e["rank"] for e in healed["evaluations"]]
        self.assertEqual(ranks, [1, 2, 3], f"Ranks are not strictly 1..3: {ranks}")
        self.assertEqual(healed["top_recommended_action"], "ACT_INSP_TOOL_WEAR")

    def test_multi_provider_dispatcher_with_failing_client(self):
        class FailingClient(BaseLLMClient):
            def generate_structured_evaluation(self, prompt, system_instruction=None, temperature=0.1):
                raise RuntimeError("API connection timeout to remote endpoint")

        mock = HeuristicMockClient()
        dispatcher = MultiProviderDispatcherClient(clients=[FailingClient(), mock])

        text, meta = dispatcher.generate_structured_evaluation("Test prompt")
        self.assertEqual(meta["failover_hops"], 1)
        self.assertIn("evaluations", json.loads(text))

    def test_json_parser_mangled_unparseable_json_heuristic_recovery(self):
        # Text has valid evaluation key-values but broken JSON syntax (missing quotes, unescaped commas)
        mangled = """
        Here is the evaluation:
        { "action_id": "ACT_CORR_LUBRICATE", "rank": 1, "suitability_score": 95, "urgency_score": 80, "expected_effectiveness_score": 90, "operational_risk_score": 10, "confidence": 0.90, "final_verdict": "RECOMMENDED", "reasoning_summary": "Lubrication needed immediately." }
        and also:
        { "action_id": "ACT_MON_CONTINUE", "rank": 2, "suitability_score": 20, "urgency_score": 10, "expected_effectiveness_score": 15, "operational_risk_score": 80, "confidence": 0.85, "final_verdict": "UNSAFE", "reasoning_summary": "Do not continue." }
        """
        parsed = MaintenanceJudge._parse_json_response(mangled)
        self.assertIn("evaluations", parsed)
        self.assertEqual(len(parsed["evaluations"]), 2)
        self.assertEqual(parsed["top_recommended_action"], "ACT_CORR_LUBRICATE")

    def test_atomic_checkpoint_replace(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "checkpoints.json"
            tmp_p = p.with_suffix(".tmp")
            data = {"state_1": {"status": "valid"}}
            with open(tmp_p, "w", encoding="utf-8") as f:
                json.dump(data, f)
            tmp_p.replace(p)
            self.assertTrue(p.exists())
            loaded = json.load(open(p))
            self.assertEqual(loaded["state_1"]["status"], "valid")


if __name__ == "__main__":
    unittest.main()
