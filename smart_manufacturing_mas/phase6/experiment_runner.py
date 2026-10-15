"""Experiment runner orchestrating multi-domain low-resource cross-domain adaptation benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from phase4.cbr_recommender import CaseBasedRecommender, CBRConfig
from phase4.hybrid_recommender import HybridRecommender, HybridConfig
from phase4.lgbm_ranker import MaintenanceLGBMRanker, RankerHyperparameters
from phase5.action_mapper import CANONICAL_ACTIONS
from phase5.frozen_model import FrozenPhase4Recommender
from phase6.adaptation_models import (
    BaseRankingModel,
    ComponentMajorityPolicy,
    ContinuationBoostedRankerModel,
    FrozenSourceLGBMModel,
    FullAdaptedHybridModel,
    GlobalMajorityPolicy,
    PriorAdaptedHybridModel,
    RandomRankingPolicy,
    ResidualCorrectionRankerModel,
    SeverityHeuristicPolicy,
    TargetAdaptedCBRModel,
    TargetOnlyRankerModel,
    ZeroShotHybridModel,
)
from phase6.config import AdaptationExperimentConfig, SUPPORTED_DOMAINS
from phase6.data_loader import AdaptationInstance, Phase6DataLoader
from phase6.evaluator import EvaluationMetrics, Phase6Evaluator
from phase6.leakage import Phase6LeakageAuditor
from phase6.sampling import BudgetSample, BudgetSampler, ChronologicalSplitter, SplitManifestBuilder, SplitResult


class AdaptationExperimentRunner:
    """Orchestrates multi-domain data-efficiency benchmark across budgets and comparison methods."""

    def __init__(self, config: Optional[AdaptationExperimentConfig] = None):
        self.config = config or AdaptationExperimentConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Initialize data loader and evaluator
        self.data_loader = Phase6DataLoader(
            corpus_path=self.config.corpus_path,
            ontology_path=self.config.ontology_path,
            training_table_path=self.config.training_table_path,
        )
        self.evaluator = Phase6Evaluator()
        self.splitter = ChronologicalSplitter(
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
        )
        self.leakage_auditor = Phase6LeakageAuditor(
            source_model_joblib_path=self.config.model_dir / "lgbm_ranker.joblib",
            phase4_training_table_path=self.config.training_table_path,
        )

        # 2. Load frozen Phase 4 reference recommender
        self.frozen_recommender = FrozenPhase4Recommender(
            model_dir=self.config.model_dir,
            training_table_path=self.config.training_table_path,
            corpus_path=self.config.corpus_path,
            ontology_path=self.config.ontology_path,
            alpha=self.config.reference_alpha,
            cbr_k=self.config.cbr_k,
        )

    def _train_lodo_source_model(self, lodo_train_df: pd.DataFrame) -> Tuple[MaintenanceLGBMRanker, CaseBasedRecommender]:
        """Trains strict Leave-One-Domain-Out source ranker and CBR excluding held-out domain."""
        X, _ = self.data_loader.pipeline.transform_dataframe(lodo_train_df, feature_subset="full_semantic_trend")
        y = lodo_train_df["ranking_relevance"].values
        groups = lodo_train_df.groupby("decision_state_id", sort=False).size().tolist()

        lodo_ranker = MaintenanceLGBMRanker(
            params=RankerHyperparameters(
                n_estimators=50,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=4,
                min_child_samples=5,
                random_state=42,
            )
        )
        lodo_ranker.fit(X, y, groups)

        # LODO CBR
        state_vectors = {
            s["decision_state_id"]: self.data_loader.pipeline.get_state_vector(s)
            for s in self.data_loader.pipeline.state_details.values()
            if s["decision_state_id"] in set(lodo_train_df["decision_state_id"])
        }
        lodo_cbr = CaseBasedRecommender(config=CBRConfig(k_neighbors=self.config.cbr_k))
        lodo_cbr.fit(train_df=lodo_train_df, state_vectors=state_vectors, relevance_col="ranking_relevance")

        return lodo_ranker, lodo_cbr

    def _tune_hyperparameters_on_val(
        self,
        base_rec: Any,
        val_instances: List[AdaptationInstance],
        action_priors: Dict[str, float],
    ) -> float:
        """Selects optimal log-prior adaptation beta strictly on the validation pool."""
        if not val_instances:
            return 0.4

        candidate_betas = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
        # Precompute base rankings on validation instances once
        val_base_ranked = [
            base_rec.rank_candidates(inst.decision_state, candidate_actions=inst.candidate_actions)
            for inst in val_instances
        ]
        target_actions = [inst.target_action for inst in val_instances]

        best_beta = 0.4
        best_mrr = -1.0

        for b in candidate_betas:
            mrrs = []
            for i, ranked_base in enumerate(val_base_ranked):
                tgt = target_actions[i]
                if b <= 0.0 or not action_priors:
                    ranked_ids = [item["action_id"] for item in ranked_base]
                else:
                    scores = {}
                    for j, item in enumerate(ranked_base):
                        aid = item["action_id"]
                        orig_s = float(item.get("score", item.get("hybrid_score", 1.0 - (j * 0.04))))
                        prior = float(action_priors.get(aid, 1e-4))
                        scores[aid] = orig_s + (b * np.log(prior + 1e-4))
                    ranked_ids = sorted(scores.keys(), key=lambda a: scores[a], reverse=True)
                rr = self.evaluator.compute_reciprocal_rank(ranked_ids, tgt)
                mrrs.append(rr)

            avg_mrr = float(np.mean(mrrs)) if mrrs else 0.0
            if avg_mrr > best_mrr:
                best_mrr = avg_mrr
                best_beta = b

        return best_beta

    def run_benchmark_for_domain(
        self,
        domain_id: str,
        instances: List[AdaptationInstance],
        lodo_source_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Runs multi-budget adaptation experiment for a single domain."""
        # 1. Chronological partition
        split = self.splitter.split(instances)

        # 2. Leakage checks
        self.leakage_auditor.audit_source_model_immutability(self.frozen_recommender.initial_model_hash)
        self.leakage_auditor.audit_source_target_domain_isolation(domain_id, instances)
        self.leakage_auditor.audit_temporal_causality(instances)
        self.leakage_auditor.audit_split_chronological_ordering(split)
        self.leakage_auditor.audit_split_disjointness(split)
        self.leakage_auditor.audit_test_set_immutability(split.test_pool)
        self.leakage_auditor.audit_target_label_feature_isolation(instances)

        # 3. Source model configuration
        if domain_id == "cmapss_lodo" and lodo_source_data is not None:
            src_ranker, src_cbr = self._train_lodo_source_model(lodo_source_data)
            # Create a mock wrapper for LODO recommender
            class LodoFrozenWrapper:
                def __init__(self, ranker, cbr, pipeline, alpha):
                    self.ranker = ranker
                    self.cbr = cbr
                    self.pipeline = pipeline
                    self.alpha = alpha

                def rank_candidates(self, decision_state, candidate_actions):
                    sid = decision_state.get("decision_state_id", "Q")
                    self.pipeline.state_details[sid] = decision_state
                    # Evaluate candidates
                    rows = []
                    for c in candidate_actions:
                        rows.append({
                            "decision_state_id": sid,
                            "decision_severity": decision_state.get("decision_severity", "WATCH"),
                            "machine_archetype": decision_state.get("asset_context", {}).get("machine_archetype", "industrial_machine"),
                            "temporal_type": decision_state.get("asset_context", {}).get("temporal_type", "time_series"),
                            "cycle": decision_state.get("temporal_context", {}).get("cycle", 0),
                            "sequence_step": decision_state.get("temporal_context", {}).get("sequence_step", 0),
                            "action_id": c["action_id"],
                            "action_name": c.get("action_name", c["action_id"]),
                            "action_category": c.get("action_category", "unknown"),
                            "intervention_risk": c.get("intervention_risk", "low"),
                            "operational_downtime_cost": c.get("operational_downtime_cost", "low"),
                            "precondition_min_severity": c.get("precondition_min_severity", "HEALTHY"),
                            "precondition_max_severity": c.get("precondition_max_severity", "CRITICAL"),
                        })
                    df_q = pd.DataFrame(rows)
                    X_q, _ = self.pipeline.transform_dataframe(df_q, feature_subset="full_semantic_trend")
                    self.pipeline.state_details.pop(sid, None)

                    lgbm_s = self.ranker.predict(X_q)
                    q_vec = self.pipeline.get_state_vector(decision_state)
                    cbr_s, _ = self.cbr.score_candidates_for_state(sid, q_vec, candidate_actions, exclude_query_state=False)

                    # Norm
                    s_min, s_max = np.min(lgbm_s), np.max(lgbm_s)
                    norm_lgbm = (lgbm_s - s_min) / (s_max - s_min + 1e-8) if s_max > s_min else np.full_like(lgbm_s, 0.5)
                    c_min, c_max = np.min(cbr_s), np.max(cbr_s)
                    norm_cbr = (cbr_s - c_min) / (c_max - c_min + 1e-8) if c_max > c_min else np.full_like(cbr_s, 0.5)

                    fused = (self.alpha * norm_lgbm) + ((1.0 - self.alpha) * norm_cbr)
                    ranked = []
                    for i, cand in enumerate(candidate_actions):
                        ranked.append({
                            "action_id": cand["action_id"],
                            "action_name": cand.get("action_name", cand["action_id"]),
                            "action_category": cand.get("action_category", "unknown"),
                            "hybrid_score": float(fused[i]),
                            "score": float(fused[i]),
                        })
                    ranked.sort(key=lambda x: x["score"], reverse=True)
                    for r, item in enumerate(ranked, start=1):
                        item["rank"] = r
                    return ranked

            ref_model = LodoFrozenWrapper(src_ranker, src_cbr, self.data_loader.pipeline, self.config.reference_alpha)
            ref_ranker = src_ranker
            ref_cbr = src_cbr
        else:
            ref_model = self.frozen_recommender
            ref_ranker = self.frozen_recommender.ranker
            ref_cbr = self.frozen_recommender.cbr

        # 4. Generate domain manifest
        budget_samples = [
            BudgetSampler.sample(split.train_pool, fraction=f, seed=42)
            for f in self.config.budget_fractions
        ]
        self.leakage_auditor.audit_budget_nesting(budget_samples)
        domain_manifest = SplitManifestBuilder.build_domain_manifest(
            domain_id=domain_id,
            split_result=split,
            budget_fractions=self.config.budget_fractions,
            seed=42,
        )

        domain_results = []
        detailed_metrics = {}

        # 5. Sweep through budget fractions
        for b_sample in budget_samples:
            f = b_sample.fraction
            budget_label = f"{int(round(f * 100))}%"
            sub_train = b_sample.instances

            # Compute empirical action priors on this budget subset
            counts: Dict[str, int] = {}
            for inst in sub_train:
                counts[inst.target_action] = counts.get(inst.target_action, 0) + 1
            total_counts = sum(counts.values()) or 1
            smoothed_priors = {
                a: (counts.get(a, 0) + 1.0) / (total_counts + len(CANONICAL_ACTIONS))
                for a in CANONICAL_ACTIONS
            }

            # Tune validation beta
            best_beta = self._tune_hyperparameters_on_val(ref_model, split.val_pool, smoothed_priors)

            # Build and fit models for this budget
            models: Dict[str, BaseRankingModel] = {
                "Zero_Shot": ZeroShotHybridModel(ref_model),
                "Frozen_Source": FrozenSourceLGBMModel(ref_ranker),
                "Target_Only": TargetOnlyRankerModel().fit(sub_train),
                "Continuation_FT": ContinuationBoostedRankerModel(ref_ranker).fit(sub_train),
                "Residual_Correction": ResidualCorrectionRankerModel(ref_ranker).fit(sub_train, gamma=0.5),
                "Adapted_CBR": TargetAdaptedCBRModel(ref_cbr, self.data_loader.pipeline).fit(sub_train),
                "Adapted_Hybrid": PriorAdaptedHybridModel(ref_model, smoothed_priors, beta=best_beta),
                "Full_Adapted": FullAdaptedHybridModel(
                    continuation_model=ContinuationBoostedRankerModel(ref_ranker).fit(sub_train),
                    adapted_cbr=TargetAdaptedCBRModel(ref_cbr, self.data_loader.pipeline).fit(sub_train),
                    action_priors=smoothed_priors,
                    alpha=self.config.reference_alpha,
                    beta=best_beta,
                ),
                "Random": RandomRankingPolicy(random_seed=42),
                "Global_Majority": GlobalMajorityPolicy(counts),
                "Component_Majority": ComponentMajorityPolicy(
                    comp_counts={}, global_counts=counts
                ),
                "Severity_Heuristic": SeverityHeuristicPolicy(),
            }

            detailed_metrics[budget_label] = {
                "fraction": f,
                "num_states": b_sample.num_states,
                "num_candidates": b_sample.num_candidates,
                "best_val_beta": best_beta,
                "methods": {},
            }

            zero_shot_records = None

            for m_name, model in models.items():
                metrics, records = self.evaluator.evaluate_model(model, split.test_pool)
                if m_name == "Zero_Shot":
                    zero_shot_records = records

                # Bootstrap CI
                boot_ci_top1 = self.evaluator.compute_grouped_bootstrap_ci(records, "top1_match")
                boot_ci_mrr = self.evaluator.compute_grouped_bootstrap_ci(records, "reciprocal_rank")

                # Paired test against zero shot
                paired_test = {}
                if zero_shot_records is not None and m_name != "Zero_Shot":
                    paired_test = self.evaluator.compute_paired_bootstrap_test(
                        records_adapted=records,
                        records_baseline=zero_shot_records,
                        metric_key="reciprocal_rank",
                    )

                rec = {
                    "domain_id": domain_id,
                    "budget_fraction": f,
                    "budget_label": budget_label,
                    "adaptation_states": b_sample.num_states,
                    "adaptation_candidates": b_sample.num_candidates,
                    "test_states": len(split.test_pool),
                    "method": m_name,
                    "top1_exact_match": metrics.top1_exact_match,
                    "top3_recall": metrics.top3_recall,
                    "mrr": metrics.mean_reciprocal_rank,
                    "ndcg_at_1": metrics.mean_ndcg_at_1,
                    "ndcg_at_3": metrics.mean_ndcg_at_3,
                    "ndcg_at_5": metrics.mean_ndcg_at_5,
                    "category_agreement": metrics.category_agreement_rate,
                    "component_accuracy": metrics.component_accuracy_rate,
                    "graded_relevance": metrics.mean_graded_relevance,
                    "mrr_ci_lower": boot_ci_mrr.ci_lower,
                    "mrr_ci_upper": boot_ci_mrr.ci_upper,
                    "top1_ci_lower": boot_ci_top1.ci_lower,
                    "top1_ci_upper": boot_ci_top1.ci_upper,
                    "delta_mrr_vs_zero_shot": paired_test.get("delta_mean", 0.0),
                    "p_value_vs_zero_shot": paired_test.get("p_value", 1.0),
                }
                domain_results.append(rec)

                detailed_metrics[budget_label]["methods"][m_name] = {
                    "metrics": metrics.to_dict(),
                    "mrr_bootstrap_ci": boot_ci_mrr.to_dict(),
                    "top1_bootstrap_ci": boot_ci_top1.to_dict(),
                    "paired_test_vs_zero_shot": paired_test,
                }

        # Determine observed crossover budget B*
        # Crossover is the lowest budget where Adapted_Hybrid beats Zero_Shot and Target_Only
        df_domain = pd.DataFrame(domain_results)
        crossover_budget = self._find_crossover_budget(df_domain)

        return {
            "domain_id": domain_id,
            "manifest": domain_manifest,
            "results": domain_results,
            "detailed_metrics": detailed_metrics,
            "crossover_budget": crossover_budget,
            "test_instances_count": len(split.test_pool),
        }

    @staticmethod
    def _find_crossover_budget(df_domain: pd.DataFrame) -> Dict[str, Any]:
        """Finds the lowest observed budget where adaptation exceeds zero-shot and target-only."""
        budgets = sorted(df_domain["budget_fraction"].unique().tolist())
        crossover = {
            "mrr_crossover_budget": "None",
            "top3_crossover_budget": "None",
            "top1_crossover_budget": "None",
        }

        # Check MRR
        for b in budgets:
            sub = df_domain[df_domain["budget_fraction"] == b]
            adapt_mrr = sub[sub["method"] == "Adapted_Hybrid"]["mrr"].values
            zero_mrr = sub[sub["method"] == "Zero_Shot"]["mrr"].values
            tgt_mrr = sub[sub["method"] == "Target_Only"]["mrr"].values

            if len(adapt_mrr) > 0 and len(zero_mrr) > 0 and len(tgt_mrr) > 0:
                if adapt_mrr[0] > zero_mrr[0] and adapt_mrr[0] > tgt_mrr[0]:
                    crossover["mrr_crossover_budget"] = f"{int(round(b * 100))}%"
                    break

        # Check Top-3
        for b in budgets:
            sub = df_domain[df_domain["budget_fraction"] == b]
            adapt_t3 = sub[sub["method"] == "Adapted_Hybrid"]["top3_recall"].values
            zero_t3 = sub[sub["method"] == "Zero_Shot"]["top3_recall"].values
            tgt_t3 = sub[sub["method"] == "Target_Only"]["top3_recall"].values

            if len(adapt_t3) > 0 and len(zero_t3) > 0 and len(tgt_t3) > 0:
                if adapt_t3[0] > zero_t3[0] and adapt_t3[0] >= tgt_t3[0]:
                    crossover["top3_crossover_budget"] = f"{int(round(b * 100))}%"
                    break

        return crossover

    def run_all_domains(self) -> Dict[str, Any]:
        """Executes full benchmark across Factory Sensors, OIL-040, and C-MAPSS LODO."""
        all_results = []
        all_manifests = {}
        all_metrics_by_budget = {}
        domain_crossovers = {}

        # 1. Factory Sensors
        if "factory_sensors" in self.config.target_domains:
            print(">>> Running Phase 6 Benchmark: Factory Sensors (Centrifugal Pumps)...")
            fs_instances = self.data_loader.load_factory_sensors()
            res_fs = self.run_benchmark_for_domain("factory_sensors", fs_instances)
            all_results.extend(res_fs["results"])
            all_manifests["factory_sensors"] = res_fs["manifest"]
            all_metrics_by_budget["factory_sensors"] = res_fs["detailed_metrics"]
            domain_crossovers["factory_sensors"] = res_fs["crossover_budget"]

        # 2. OIL-040
        if "oil040" in self.config.target_domains:
            print(">>> Running Phase 6 Benchmark: OIL-040 (Rotating Equipment)...")
            oil_instances = self.data_loader.load_oil040()
            res_oil = self.run_benchmark_for_domain("oil040", oil_instances)
            all_results.extend(res_oil["results"])
            all_manifests["oil040"] = res_oil["manifest"]
            all_metrics_by_budget["oil040"] = res_oil["detailed_metrics"]
            domain_crossovers["oil040"] = res_oil["crossover_budget"]

        # 3. C-MAPSS LODO
        if "cmapss_lodo" in self.config.target_domains:
            print(">>> Running Phase 6 Benchmark: C-MAPSS (Turbofan Engines, Leave-One-Domain-Out)...")
            cmapss_instances, lodo_df = self.data_loader.load_cmapss_lodo()
            res_cmapss = self.run_benchmark_for_domain("cmapss_lodo", cmapss_instances, lodo_source_data=lodo_df)
            all_results.extend(res_cmapss["results"])
            all_manifests["cmapss_lodo"] = res_cmapss["manifest"]
            all_metrics_by_budget["cmapss_lodo"] = res_cmapss["detailed_metrics"]
            domain_crossovers["cmapss_lodo"] = res_cmapss["crossover_budget"]

        # 4. Save results to disk
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(self.output_dir / "results.csv", index=False)

        # Aggregate results across domains and budgets
        agg_cols = ["budget_fraction", "budget_label", "method"]
        numeric_cols = [
            "top1_exact_match", "top3_recall", "mrr", "ndcg_at_1", "ndcg_at_3",
            "category_agreement", "component_accuracy", "graded_relevance"
        ]
        df_agg = df_results.groupby(agg_cols)[numeric_cols].agg(["mean", "std"]).reset_index()
        # Flatten MultiIndex columns
        df_agg.columns = [
            f"{c[0]}_{c[1]}" if c[1] else c[0] for c in df_agg.columns
        ]
        df_agg.to_csv(self.output_dir / "aggregate_results.csv", index=False)

        # Save manifests and audit
        with open(self.output_dir / "split_manifest.json", "w", encoding="utf-8") as f:
            json.dump(all_manifests, f, indent=2)

        with open(self.output_dir / "metrics_by_budget.json", "w", encoding="utf-8") as f:
            json.dump({
                "crossovers": domain_crossovers,
                "domains": all_metrics_by_budget,
            }, f, indent=2)

        self.config.save(self.output_dir / "experiment_config.json")
        self.leakage_auditor.generate_audit_report(self.output_dir / "leakage_audit.json")

        print(f"Benchmark completed successfully! Artifacts written to {self.output_dir}")
        return {
            "results_df": df_results,
            "aggregate_df": df_agg,
            "crossovers": domain_crossovers,
            "manifests": all_manifests,
            "metrics_by_budget": all_metrics_by_budget,
        }
