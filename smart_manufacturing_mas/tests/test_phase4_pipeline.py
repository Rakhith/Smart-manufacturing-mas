"""Unit and Pipeline Tests for Phase 4 Prescriptive Maintenance Recommender."""

import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase4.dataset_builder import Phase4DatasetBuilder
from phase4.targets import TargetBuilder, TargetConfig
from phase4.feature_pipeline import FeaturePipeline
from phase4.splitters import GroupedStratifiedSplitter
from phase4.lgbm_ranker import MaintenanceLGBMRanker, RankerHyperparameters
from phase4.cbr_recommender import CaseBasedRecommender, CBRConfig
from phase4.hybrid_recommender import HybridRecommender
from phase4.evaluator import PrescriptiveEvaluator
from phase4.safety_guard import Phase4SafetyGuard


class TestPhase4Pipeline(unittest.TestCase):

    def setUp(self):
        self.p3a_corpus = PROJECT_ROOT / "outputs/phase3a/selected_decision_states/decision_states_corpus.json"
        self.p3b_judgments = PROJECT_ROOT / "outputs/phase3b/parsed_judgments/parsed_judgments.json"
        self.p3b_ontology = PROJECT_ROOT / "outputs/phase3b/action_ontology/action_ontology.json"

    def test_dataset_builder_and_targets(self):
        builder = Phase4DatasetBuilder(
            phase3a_corpus_path=self.p3a_corpus,
            phase3b_judgments_path=self.p3b_judgments,
            phase3b_ontology_path=self.p3b_ontology,
        )
        df, meta = builder.build_training_table()
        self.assertEqual(meta["total_decision_states"], 1023)
        self.assertGreater(len(df), 3500)
        self.assertIn("decision_state_id", df.columns)
        self.assertIn("action_id", df.columns)

        # Apply target
        tb = TargetBuilder(TargetConfig(strategy="inverted_rank"))
        df_target = tb.apply_to_dataframe(df)
        self.assertIn("ranking_relevance", df_target.columns)
        self.assertTrue((df_target["ranking_relevance"] >= 0).all())
        self.assertTrue((df_target["ranking_relevance"] <= 4).all())

    def test_feature_pipeline(self):
        pipeline = FeaturePipeline(corpus_path=self.p3a_corpus)
        sample_df = pd.DataFrame([{
            "decision_state_id": "DS_AI4I_168",
            "action_id": "ACT_OP_CONTROLLED_SHUTDOWN",
            "action_category": "operational_mitigation",
            "intervention_risk": "low",
            "operational_downtime_cost": "high",
            "decision_severity": "CRITICAL",
            "ranking_relevance": 4,
        }])
        X, fnames = pipeline.transform_dataframe(sample_df)
        self.assertEqual(X.shape[0], 1)
        self.assertGreater(len(fnames), 20)
        self.assertFalse(np.isnan(X).any())

    def test_grouped_stratified_splitter_no_leakage(self):
        df_sample = pd.DataFrame([
            {"decision_state_id": f"DS_{i}", "dataset_id": "ai4i_2020" if i < 50 else "cmapss", "decision_severity": "WATCH"}
            for i in range(100)
        ])
        # Repeat to simulate multiple candidates per state
        df_full = pd.concat([df_sample, df_sample], ignore_index=True)

        splitter = GroupedStratifiedSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)
        train_df, val_df, test_df, manifest = splitter.split(df_full)

        train_s = set(train_df["decision_state_id"])
        val_s = set(val_df["decision_state_id"])
        test_s = set(test_df["decision_state_id"])

        self.assertEqual(len(train_s.intersection(val_s)), 0)
        self.assertEqual(len(train_s.intersection(test_s)), 0)
        self.assertEqual(len(val_s.intersection(test_s)), 0)

    def test_evaluator_ndcg_and_mrr(self):
        evaluator = PrescriptiveEvaluator()
        # Perfect ranking
        true_rel = [4.0, 3.0, 2.0, 1.0]
        pred_scores = [0.9, 0.7, 0.4, 0.1]
        ndcg1 = evaluator.ndcg_at_k(true_rel, pred_scores, k=1)
        ndcg3 = evaluator.ndcg_at_k(true_rel, pred_scores, k=3)
        self.assertAlmostEqual(ndcg1, 1.0, places=4)
        self.assertAlmostEqual(ndcg3, 1.0, places=4)

        # Reversed ranking
        pred_rev = [0.1, 0.4, 0.7, 0.9]
        ndcg_rev = evaluator.ndcg_at_k(true_rel, pred_rev, k=3)
        self.assertLess(ndcg_rev, 0.8)

    def test_safety_guard_leakage_detection(self):
        # Verify safety guard catches leaked target feature
        audit = Phase4SafetyGuard.audit_all(
            df_all=pd.DataFrame({"decision_state_id": ["S1", "S1"], "ranking_relevance": [4, 2], "dataset_id": ["D1", "D1"]}),
            train_df=pd.DataFrame({"decision_state_id": ["S1"]}),
            val_df=pd.DataFrame({"decision_state_id": ["S2"]}),
            test_df=pd.DataFrame({"decision_state_id": ["S3"]}),
            feature_names=["state__severity", "target_ground_truth_failure"],  # Injected leakage
            X_train=np.array([[1.0, 0.0]]),
            cbr_indexed_state_ids=["S1"],
            relevance_col="ranking_relevance",
        )
        self.assertEqual(audit["check_4_zero_target_leakage"]["status"], "FAIL")
        self.assertEqual(audit["overall_disposition"], "FAIL_AUDIT")


if __name__ == "__main__":
    unittest.main()
