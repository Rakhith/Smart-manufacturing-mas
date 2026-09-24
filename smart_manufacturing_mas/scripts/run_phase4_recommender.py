"""Master Orchestrator for Phase 4 Prescriptive Maintenance Recommender.

Executes the complete Phase 4 pipeline:
1. Data Extraction & Zero-Leakage Dataset Construction
2. Target Definition & Transformation
3. Cross-Domain Semantic Feature Engineering
4. Grouped Stratified Train/Val/Test Splitting
5. LGBMRanker Training with Early Stopping
6. Case-Based Reasoning (CBR) Retrieval Indexing
7. Hybrid Fusion & Validation-Driven Alpha Selection
8. Baselines & Feature Ablations
9. Cross-Domain Generalization (Leave-One-Dataset-Out)
10. Provider Robustness & Silver Uncertainty Analysis
11. Explainability (SHAP & Sample Explanation Cards)
12. 12-Invariant Safety & Leakage Audit
13. Research Report Generation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase4.dataset_builder import Phase4DatasetBuilder
from phase4.targets import TargetBuilder, TargetConfig
from phase4.feature_pipeline import FeaturePipeline
from phase4.splitters import GroupedStratifiedSplitter, LeaveOneDatasetOutSplitter
from phase4.lgbm_ranker import MaintenanceLGBMRanker, RankerHyperparameters
from phase4.cbr_recommender import CaseBasedRecommender, CBRConfig
from phase4.hybrid_recommender import HybridRecommender, HybridConfig
from phase4.explainability import PrescriptiveExplainer
from phase4.evaluator import PrescriptiveEvaluator
from phase4.safety_guard import Phase4SafetyGuard


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 4 Prescriptive Maintenance Recommender Pipeline")
    parser.add_argument("--phase3a-dir", type=str, default="outputs/phase3a", help="Path to Phase 3A outputs directory")
    parser.add_argument("--phase3b-dir", type=str, default="outputs/phase3b", help="Path to Phase 3B outputs directory")
    parser.add_argument("--output-dir", type=str, default="outputs/phase4", help="Path to Phase 4 outputs directory")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--cbr-k", type=int, default=5, help="Number of neighbors for CBR retrieval")
    parser.add_argument("--target-strategy", type=str, default="inverted_rank", choices=["inverted_rank", "composite_suitability"], help="Ranking target strategy")
    parser.add_argument("--skip-cross-domain", action="store_true", help="Skip leave-one-dataset-out cross-domain evaluation")
    return parser.parse_args()


def main():
    start_time = time.time()
    args = parse_args()

    phase3a_dir = Path(args.phase3a_dir)
    phase3b_dir = Path(args.phase3b_dir)
    output_dir = Path(args.output_dir)

    # Subdirectories
    train_dir = output_dir / "training"
    models_dir = output_dir / "models"
    preds_dir = output_dir / "predictions"
    eval_dir = output_dir / "evaluation"
    expl_dir = output_dir / "explainability"
    splits_dir = output_dir / "splits"
    reports_dir = output_dir / "reports"
    meta_dir = output_dir / "metadata"

    for d in [train_dir, models_dir, preds_dir, eval_dir, expl_dir, splits_dir, reports_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 4: PRESCRIPTIVE MAINTENANCE RECOMMENDER PIPELINE")
    print("=" * 70)
    print(f"Phase 3A Dir   : {phase3a_dir.resolve()}")
    print(f"Phase 3B Dir   : {phase3b_dir.resolve()}")
    print(f"Output Dir     : {output_dir.resolve()}")
    print(f"Random Seed    : {args.random_seed}")
    print(f"Target Strategy: {args.target_strategy}")
    print("=" * 70)

    # -------------------------------------------------------------
    # STEP 1: Construct Training Table
    # -------------------------------------------------------------
    print("\n[Step 1/13] Building leakage-safe training table...")
    corpus_json = phase3a_dir / "selected_decision_states" / "decision_states_corpus.json"
    judgments_json = phase3b_dir / "parsed_judgments" / "parsed_judgments.json"
    ontology_json = phase3b_dir / "action_ontology" / "action_ontology.json"
    pref_parquet = phase3b_dir / "preference_dataset" / "maintenance_preference_dataset.parquet"

    builder = Phase4DatasetBuilder(
        phase3a_corpus_path=corpus_json,
        phase3b_judgments_path=judgments_json,
        phase3b_ontology_path=ontology_json,
        phase3b_pref_path=pref_parquet,
    )
    df_raw, build_meta = builder.build_training_table()
    print(f"  Total DecisionStates: {build_meta['total_decision_states']}")
    print(f"  Total Action Pairs  : {build_meta['total_candidate_pairs']}")
    print(f"  Datasets ({len(build_meta['datasets_represented'])}): {build_meta['datasets_represented']}")

    # -------------------------------------------------------------
    # STEP 2: Compute Learning Targets
    # -------------------------------------------------------------
    print("\n[Step 2/13] Defining ranking learning target...")
    target_config = TargetConfig(strategy=args.target_strategy)
    target_builder = TargetBuilder(config=target_config)
    df_all = target_builder.apply_to_dataframe(df_raw, target_col="ranking_relevance")
    target_config.save_json(meta_dir / "target_definition.json")
    print(f"  Target strategy applied: {args.target_strategy}")
    print(f"  Relevance distribution:\n{df_all['ranking_relevance'].value_counts().sort_index()}")

    # Save training table
    df_all.to_parquet(train_dir / "phase4_training_table.parquet", index=False)
    df_all.to_csv(train_dir / "phase4_training_table.csv", index=False)
    print(f"  Saved master training table to {train_dir}")

    # -------------------------------------------------------------
    # STEP 3: Feature Engineering
    # -------------------------------------------------------------
    print("\n[Step 3/13] Engineering cross-domain semantic features...")
    pipeline = FeaturePipeline(corpus_path=corpus_json)
    X_full, feature_names = pipeline.transform_dataframe(df_all, feature_subset="full_semantic_trend")
    pipeline.save_feature_schema(meta_dir / "feature_schema.json")
    print(f"  Total engineered features: {len(feature_names)}")
    for grp, fnames in pipeline.feature_groups.items():
        print(f"    {grp:28s}: {len(fnames)} features")

    # Extract state-level vectors for CBR
    with open(corpus_json, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    state_vectors = {s["decision_state_id"]: pipeline.get_state_vector(s) for s in corpus}

    # -------------------------------------------------------------
    # STEP 4: Grouped Stratified Train/Val/Test Split
    # -------------------------------------------------------------
    print("\n[Step 4/13] Partitioning grouped train/val/test splits...")
    splitter = GroupedStratifiedSplitter(
        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=args.random_seed
    )
    train_df, val_df, test_df, split_manifest = splitter.split(df_all)
    split_manifest.save_json(splits_dir / "split_manifest.json")
    print(f"  Train states: {len(split_manifest.train_state_ids):4d} ({len(train_df):4d} candidate rows)")
    print(f"  Val states  : {len(split_manifest.val_state_ids):4d} ({len(val_df):4d} candidate rows)")
    print(f"  Test states : {len(split_manifest.test_state_ids):4d} ({len(test_df):4d} candidate rows)")

    # Prepare feature matrices
    X_train, _ = pipeline.transform_dataframe(train_df, feature_subset="full_semantic_trend")
    y_train = train_df["ranking_relevance"].to_numpy(dtype=np.int32)
    groups_train = train_df.groupby("decision_state_id", sort=False).size().tolist()

    X_val, _ = pipeline.transform_dataframe(val_df, feature_subset="full_semantic_trend")
    y_val = val_df["ranking_relevance"].to_numpy(dtype=np.int32)
    groups_val = val_df.groupby("decision_state_id", sort=False).size().tolist()

    X_test, _ = pipeline.transform_dataframe(test_df, feature_subset="full_semantic_trend")
    y_test = test_df["ranking_relevance"].to_numpy(dtype=np.int32)
    groups_test = test_df.groupby("decision_state_id", sort=False).size().tolist()

    # -------------------------------------------------------------
    # STEP 5: Train LGBMRanker
    # -------------------------------------------------------------
    print("\n[Step 5/13] Training Primary Recommender (LGBMRanker)...")
    ranker_params = RankerHyperparameters(random_state=args.random_seed)
    ranker = MaintenanceLGBMRanker(params=ranker_params)
    ranker.fit(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        X_val=X_val,
        y_val=y_val,
        groups_val=groups_val,
        feature_names=feature_names,
    )
    ranker.save(models_dir)
    print(f"  LGBMRanker trained successfully (best iteration: {ranker.best_iteration_})")
    top_importances = list(ranker.get_feature_importance().items())[:5]
    print("  Top 5 features by gain:")
    for fn, vals in top_importances:
        print(f"    - {fn:32s}: gain={vals['gain']:.2f}, split={vals['split']}")

    # -------------------------------------------------------------
    # STEP 6: Index Case-Based Reasoning (CBR) Recommender
    # -------------------------------------------------------------
    print("\n[Step 6/13] Indexing Case-Based Reasoning (CBR) retriever...")
    cbr_config = CBRConfig(k_neighbors=args.cbr_k)
    cbr = CaseBasedRecommender(config=cbr_config)
    cbr.fit(train_df=train_df, state_vectors=state_vectors, relevance_col="ranking_relevance")
    print(f"  CBR indexed {len(cbr.indexed_state_ids)} historical training states (k={cbr_config.k_neighbors})")

    # -------------------------------------------------------------
    # STEP 7: Hybrid Fusion & Alpha Selection
    # -------------------------------------------------------------
    print("\n[Step 7/13] Optimizing Hybrid ensemble alpha on validation set...")
    hybrid = HybridRecommender(config=HybridConfig())
    evaluator = PrescriptiveEvaluator(total_ontology_actions=21)

    lgbm_val_preds = ranker.predict(X_val)
    cbr_val_preds = cbr.predict_dataframe(val_df, state_vectors=state_vectors)

    def val_metric_fn(df_subset, preds):
        res = evaluator.evaluate_predictions(df_subset, preds)
        return res["ndcg_at_3"]

    best_alpha, alpha_history = hybrid.select_best_alpha(
        lgbm_val_scores=lgbm_val_preds,
        cbr_val_scores=cbr_val_preds,
        val_df=val_df,
        candidate_alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
        metric_eval_fn=val_metric_fn,
    )
    print(f"  Alpha sweep results (validation NDCG@3):")
    for a_val, score in alpha_history.items():
        star = " *" if a_val == best_alpha else ""
        print(f"    alpha = {a_val:4.2f} -> NDCG@3 = {score:.4f}{star}")
    print(f"  Selected optimal alpha: {best_alpha}")

    # -------------------------------------------------------------
    # STEP 8: Model Predictions on Test Set
    # -------------------------------------------------------------
    print("\n[Step 8/13] Generating test set predictions...")
    lgbm_test_preds = ranker.predict(X_test)
    cbr_test_preds = cbr.predict_dataframe(test_df, state_vectors=state_vectors)
    hybrid_test_preds = hybrid.combine_scores(
        lgbm_test_preds, cbr_test_preds, test_df, alpha=best_alpha
    )

    test_preds_df = test_df[[
        "decision_state_id", "action_id", "dataset_id", "decision_severity",
        "silver_rank", "ranking_relevance", "silver_confidence", "provenance_llm_provider"
    ]].copy()
    test_preds_df["lgbm_score"] = lgbm_test_preds
    test_preds_df["cbr_score"] = cbr_test_preds
    test_preds_df["hybrid_score"] = hybrid_test_preds
    test_preds_df.to_parquet(preds_dir / "test_predictions.parquet", index=False)
    test_preds_df.to_csv(preds_dir / "test_predictions.csv", index=False)
    print(f"  Saved test predictions to {preds_dir}")

    # -------------------------------------------------------------
    # STEP 9: Comprehensive Test Evaluation & Baselines
    # -------------------------------------------------------------
    print("\n[Step 9/13] Evaluating models and baselines on test set...")
    # Baselines
    baselines = evaluator.run_baselines(train_df, test_df, random_seed=args.random_seed)

    # Core models
    eval_cbr = evaluator.evaluate_predictions(test_df, cbr_test_preds)
    eval_lgbm = evaluator.evaluate_predictions(test_df, lgbm_test_preds)
    eval_hybrid = evaluator.evaluate_predictions(test_df, hybrid_test_preds)

    model_evaluations = {
        "baseline_random": baselines["random_ranking"],
        "baseline_majority": baselines["majority_popularity"],
        "baseline_severity_heuristic": baselines["severity_heuristic"],
        "cbr_only": eval_cbr,
        "lgbm_only": eval_lgbm,
        "hybrid_optimal": eval_hybrid,
    }

    with open(eval_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(model_evaluations, f, indent=2)

    print("\n  ================ TEST EVALUATION RESULTS ================")
    print(f"  {'Model / Baseline':<28s} | {'NDCG@1':<7s} | {'NDCG@3':<7s} | {'Rec@1':<7s} | {'Rec@3':<7s} | {'MRR':<7s} | {'Cov%':<6s}")
    print("  " + "-" * 75)
    for m_name, m_res in model_evaluations.items():
        print(f"  {m_name:<28s} | {m_res['ndcg_at_1']:.4f}  | {m_res['ndcg_at_3']:.4f}  | {m_res['recall_at_1']:.4f}  | {m_res['recall_at_3']:.4f}  | {m_res['mrr']:.4f}  | {m_res['action_coverage_pct']:.1f}%")

    # -------------------------------------------------------------
    # STEP 10: Feature Group Ablations
    # -------------------------------------------------------------
    print("\n[Step 10/13] Running feature group ablations...")
    ablation_results = {"alpha_sweep": alpha_history}

    for subset in ["raw_structured", "canonical"]:
        X_tr_sub, fn_sub = pipeline.transform_dataframe(train_df, feature_subset=subset)
        X_te_sub, _ = pipeline.transform_dataframe(test_df, feature_subset=subset)
        ranker_sub = MaintenanceLGBMRanker(params=ranker_params)
        ranker_sub.fit(X_tr_sub, y_train, groups_train, feature_names=fn_sub)
        sub_preds = ranker_sub.predict(X_te_sub)
        ablation_results[f"lgbm_feature_subset__{subset}"] = evaluator.evaluate_predictions(test_df, sub_preds)

    ablation_results["lgbm_feature_subset__full_semantic_trend"] = eval_lgbm

    with open(eval_dir / "ablations.json", "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    print("  Feature Ablation Summary (NDCG@3 / MRR):")
    print(f"    Raw Structured     : NDCG@3 = {ablation_results['lgbm_feature_subset__raw_structured']['ndcg_at_3']:.4f}, MRR = {ablation_results['lgbm_feature_subset__raw_structured']['mrr']:.4f}")
    print(f"    Canonical Physical : NDCG@3 = {ablation_results['lgbm_feature_subset__canonical']['ndcg_at_3']:.4f}, MRR = {ablation_results['lgbm_feature_subset__canonical']['mrr']:.4f}")
    print(f"    Full Semantic+Trend: NDCG@3 = {ablation_results['lgbm_feature_subset__full_semantic_trend']['ndcg_at_3']:.4f}, MRR = {ablation_results['lgbm_feature_subset__full_semantic_trend']['mrr']:.4f}")

    # -------------------------------------------------------------
    # STEP 11: Stratified & Robustness Slices
    # -------------------------------------------------------------
    print("\n[Step 11/13] Computing stratified slices (dataset, severity, provider, confidence)...")
    per_dataset_metrics = evaluator.evaluate_slices(test_df, hybrid_test_preds, slice_column="dataset_id")
    per_severity_metrics = evaluator.evaluate_slices(test_df, hybrid_test_preds, slice_column="decision_severity")
    provider_robustness_metrics = evaluator.evaluate_slices(test_df, hybrid_test_preds, slice_column="provenance_llm_provider")

    # High vs Low confidence slice
    test_df_conf = test_df.copy()
    test_df_conf["confidence_tier"] = np.where(test_df_conf["silver_confidence"] >= 0.85, "HIGH_CONFIDENCE (>=0.85)", "LOW_CONFIDENCE (<0.85)")
    confidence_slice_metrics = evaluator.evaluate_slices(test_df_conf, hybrid_test_preds, slice_column="confidence_tier")

    with open(eval_dir / "per_dataset_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_dataset_metrics, f, indent=2)
    with open(eval_dir / "per_severity_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_severity_metrics, f, indent=2)
    with open(eval_dir / "provider_robustness_metrics.json", "w", encoding="utf-8") as f:
        json.dump(provider_robustness_metrics, f, indent=2)
    with open(eval_dir / "confidence_slice_metrics.json", "w", encoding="utf-8") as f:
        json.dump(confidence_slice_metrics, f, indent=2)

    # -------------------------------------------------------------
    # STEP 12: Cross-Domain Generalization (Leave-One-Dataset-Out)
    # -------------------------------------------------------------
    cross_domain_metrics = {}
    if not args.skip_cross_domain:
        print("\n[Step 12/13] Running Leave-One-Dataset-Out cross-domain evaluation...")
        lodo = LeaveOneDatasetOutSplitter(random_seed=args.random_seed)
        lodo_splits = lodo.generate_splits(df_all)

        for sp in lodo_splits:
            ds_name = sp["held_out_dataset"]
            tr_sp, va_sp, te_sp = sp["train_df"], sp["val_df"], sp["test_df"]

            X_tr_sp, fn_sp = pipeline.transform_dataframe(tr_sp, feature_subset="full_semantic_trend")
            y_tr_sp = tr_sp["ranking_relevance"].to_numpy(dtype=np.int32)
            grp_tr_sp = tr_sp.groupby("decision_state_id", sort=False).size().tolist()

            X_te_sp, _ = pipeline.transform_dataframe(te_sp, feature_subset="full_semantic_trend")

            # Train ranker on 10 datasets
            lgb_sp = MaintenanceLGBMRanker(params=ranker_params)
            lgb_sp.fit(X_tr_sp, y_tr_sp, grp_tr_sp, feature_names=fn_sp)
            preds_lgb_sp = lgb_sp.predict(X_te_sp)

            # Fit CBR on 10 datasets
            cbr_sp = CaseBasedRecommender(config=cbr_config)
            cbr_sp.fit(tr_sp, state_vectors)
            preds_cbr_sp = cbr_sp.predict_dataframe(te_sp, state_vectors)

            # Combine using standard alpha=0.5
            hybrid_sp_preds = (0.5 * HybridRecommender.normalize_scores_per_group(preds_lgb_sp, [g.index.values for _, g in te_sp.groupby("decision_state_id", sort=False)])) + \
                              (0.5 * HybridRecommender.normalize_scores_per_group(preds_cbr_sp, [g.index.values for _, g in te_sp.groupby("decision_state_id", sort=False)]))

            cross_res = evaluator.evaluate_predictions(te_sp, hybrid_sp_preds)
            cross_domain_metrics[ds_name] = cross_res
            print(f"    Held-Out: {ds_name:28s} (N={sp['test_states_count']:3d}) -> NDCG@3: {cross_res['ndcg_at_3']:.4f}, Recall@1: {cross_res['recall_at_1']:.4f}")

        with open(eval_dir / "cross_domain_metrics.json", "w", encoding="utf-8") as f:
            json.dump(cross_domain_metrics, f, indent=2)
    else:
        print("\n[Step 12/13] Skipped cross-domain evaluation (--skip-cross-domain specified).")

    # -------------------------------------------------------------
    # STEP 13: Explainability & Audit Suite
    # -------------------------------------------------------------
    print("\n[Step 13/13] Generating explainability artifacts & running audit...")
    explainer = PrescriptiveExplainer(ranker_model=ranker, feature_names=feature_names)
    shap_results = explainer.compute_global_shap(X_test, max_samples=150)
    with open(expl_dir / "global_shap_results.json", "w", encoding="utf-8") as f:
        json.dump(shap_results, f, indent=2)

    # Generate sample explanation cards for 6 diverse test states
    sample_cards = []
    test_state_ids = test_df["decision_state_id"].unique()
    sample_sids = test_state_ids[:6]

    for sid in sample_sids:
        group_rows = test_df[test_df["decision_state_id"] == sid].copy().reset_index(drop=True)
        grp_preds = hybrid_test_preds[test_df[test_df["decision_state_id"] == sid].index.values]

        q_vec = state_vectors[sid]
        _, cbr_neighbors = cbr.score_candidates_for_state(
            sid, q_vec, group_rows.to_dict(orient="records"), exclude_query_state=True
        )

        state_meta = {
            "dataset_id": group_rows.iloc[0]["dataset_id"],
            "machine_archetype": group_rows.iloc[0]["machine_archetype"],
            "decision_severity": group_rows.iloc[0]["decision_severity"],
            "cycle": group_rows.iloc[0].get("cycle"),
        }
        card = explainer.generate_recommendation_card(
            state_id=sid,
            state_metadata=state_meta,
            candidate_rows=group_rows,
            predicted_scores=grp_preds,
            cbr_neighbors=cbr_neighbors,
        )
        sample_cards.append(card)

    explainer.save_sample_cards(sample_cards, expl_dir)
    print(f"  Generated {len(sample_cards)} sample explanation cards under {expl_dir}")

    # Safety & Leakage Audit
    print("\n[Audit] Executing 12-invariant safety & leakage audit...")
    audit_results = Phase4SafetyGuard.audit_all(
        df_all=df_all,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_names=feature_names,
        X_train=X_train,
        cbr_indexed_state_ids=cbr.indexed_state_ids,
        relevance_col="ranking_relevance",
    )
    Phase4SafetyGuard.save_audit_report(audit_results, reports_dir / "leakage_audit_report.json")

    for k, v in audit_results.items():
        if k != "overall_disposition":
            st = v["status"]
            print(f"  {k:38s}: [{st}]")
    print(f"  Overall Audit Disposition: {audit_results['overall_disposition']}")

    if audit_results["overall_disposition"] != "ALL_PASS":
        print("\n[ERROR] Safety audit failed! Aborting phase completion.")
        sys.exit(1)

    # -------------------------------------------------------------
    # Generate Research Reports
    # -------------------------------------------------------------
    print("\n[Reports] Generating research-quality Phase 4 report...")
    elapsed = time.time() - start_time

    # Generate Markdown Report
    generate_markdown_report(
        output_path=reports_dir / "PHASE_4_REPORT.md",
        summary_path=reports_dir / "PHASE_4_SUMMARY.md",
        build_meta=build_meta,
        split_manifest=split_manifest,
        model_evaluations=model_evaluations,
        ablation_results=ablation_results,
        per_dataset_metrics=per_dataset_metrics,
        per_severity_metrics=per_severity_metrics,
        provider_robustness_metrics=provider_robustness_metrics,
        confidence_slice_metrics=confidence_slice_metrics,
        cross_domain_metrics=cross_domain_metrics,
        shap_results=shap_results,
        audit_results=audit_results,
        elapsed_sec=elapsed,
        best_alpha=best_alpha,
    )

    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE AND VERIFIED!")
    print(f"Total Pipeline Runtime: {elapsed:.2f} seconds")
    print(f"Report Location: {reports_dir / 'PHASE_4_REPORT.md'}")
    print("=" * 70)


def generate_markdown_report(
    output_path: Path,
    summary_path: Path,
    build_meta: Dict[str, Any],
    split_manifest: Any,
    model_evaluations: Dict[str, Any],
    ablation_results: Dict[str, Any],
    per_dataset_metrics: Dict[str, Any],
    per_severity_metrics: Dict[str, Any],
    provider_robustness_metrics: Dict[str, Any],
    confidence_slice_metrics: Dict[str, Any],
    cross_domain_metrics: Dict[str, Any],
    shap_results: Dict[str, Any],
    audit_results: Dict[str, Any],
    elapsed_sec: float,
    best_alpha: float,
):
    """Generates rigorous research-quality Phase 4 documentation."""
    lines = [
        "# Phase 4 Research Report: Learned Prescriptive Maintenance Recommender",
        "",
        "## 1. Executive Summary & Epistemic Framing",
        "- **Phase Objective**: Build, validate, and evaluate the first learned prescriptive-maintenance recommender for the Smart Manufacturing Multi-Agent System (MAS).",
        "- **Epistemic Framing**: The training and evaluation labels are **silver-standard preference rankings** generated by LLM-as-a-Judge in Phase 3B across 1,023 representative Phase 3A DecisionStates. They are **not** ground-truth operational labels.",
        "- **Target Architecture**: DecisionState -> Semantic Feature Pipeline -> LGBMRanker (LambdaRank) -> Case-Based Reasoning (CBR) -> Hybrid Recommender -> Explainability & Invariant Audit.",
        f"- **Corpus Scale**: 1,023 DecisionStates across 11 heterogeneous industrial dataset streams yielding **{build_meta['total_candidate_pairs']}** candidate action evaluation pairs.",
        f"- **Optimal Hybrid Alpha**: `alpha = {best_alpha:.2f}` (selected strictly on validation split without test leakage).",
        f"- **Primary Test Performance (Hybrid)**: **NDCG@1 = {model_evaluations['hybrid_optimal']['ndcg_at_1']:.4f}**, **NDCG@3 = {model_evaluations['hybrid_optimal']['ndcg_at_3']:.4f}**, **Recall@1 = {model_evaluations['hybrid_optimal']['recall_at_1']:.4f}**, **MRR = {model_evaluations['hybrid_optimal']['mrr']:.4f}**.",
        f"- **Safety & Leakage Audit**: **{audit_results['overall_disposition']}** across all 12 invariant tests.",
        "",
        "## 2. Dataset & Split Distributions",
        f"- **Total DecisionStates**: {build_meta['total_decision_states']}",
        f"- **Total Evaluated Action Pairs**: {build_meta['total_candidate_pairs']}",
        f"- **Train Partition**: {len(split_manifest.train_state_ids)} states ({split_manifest.train_record_count} candidate rows, 70%)",
        f"- **Validation Partition**: {len(split_manifest.val_state_ids)} states ({split_manifest.val_record_count} candidate rows, 15%)",
        f"- **Test Partition**: {len(split_manifest.test_state_ids)} states ({split_manifest.test_record_count} candidate rows, 15%)",
        "- **Grouping**: Strictly grouped by `decision_state_id` to eliminate state leakage.",
        "",
        "## 3. Comparative Test Performance vs. Baselines",
        "",
        "| Model / Baseline | NDCG@1 | NDCG@3 | NDCG@5 | Recall@1 | Recall@3 | MRR | Action Coverage | Diversity Entropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for m_name, m_res in model_evaluations.items():
        lines.append(
            f"| `{m_name}` | {m_res['ndcg_at_1']:.4f} | {m_res['ndcg_at_3']:.4f} | {m_res['ndcg_at_5']:.4f} | "
            f"{m_res['recall_at_1']:.4f} | {m_res['recall_at_3']:.4f} | {m_res['mrr']:.4f} | "
            f"{m_res['action_coverage_pct']:.1f}% | {m_res['recommendation_diversity_entropy']:.3f} |"
        )

    lines.extend([
        "",
        "## 4. Feature Representation & Ablations",
        "Investigates whether semantic normalization and trend indicators yield empirical gains over raw structured features:",
        "",
        "| Feature Configuration | NDCG@1 | NDCG@3 | Recall@1 | MRR | Action Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ])

    for sub_key in ["lgbm_feature_subset__raw_structured", "lgbm_feature_subset__canonical", "lgbm_feature_subset__full_semantic_trend"]:
        if sub_key in ablation_results:
            sub_res = ablation_results[sub_key]
            lbl = sub_key.replace("lgbm_feature_subset__", "").replace("_", " ").title()
            lines.append(
                f"| `{lbl}` | {sub_res['ndcg_at_1']:.4f} | {sub_res['ndcg_at_3']:.4f} | "
                f"{sub_res['recall_at_1']:.4f} | {sub_res['mrr']:.4f} | {sub_res['action_coverage_pct']:.1f}% |"
            )

    lines.extend([
        "",
        "## 5. Cross-Domain Generalization (Leave-One-Dataset-Out)",
        "Evaluates whether the learned semantic representation transfers across heterogeneous industrial domains under complete dataset hold-out:",
        "",
        "| Held-Out Dataset | Evaluated States | NDCG@1 | NDCG@3 | Recall@1 | Recall@3 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])

    if cross_domain_metrics:
        for ds, c_res in sorted(cross_domain_metrics.items()):
            lines.append(
                f"| `{ds}` | {c_res['total_eval_states']} | {c_res['ndcg_at_1']:.4f} | {c_res['ndcg_at_3']:.4f} | "
                f"{c_res['recall_at_1']:.4f} | {c_res['recall_at_3']:.4f} | {c_res['mrr']:.4f} |"
            )
    else:
        lines.append("| *(Cross-domain evaluation skipped)* | - | - | - | - | - | - |")

    lines.extend([
        "",
        "## 6. LLM Provider Provenance & Robustness",
        "Evaluates recommender alignment across the 4 Phase 3B LLM inference tiers (sample size caveat: Groq n=9):",
        "",
        "| LLM Provider | Test States | NDCG@1 | NDCG@3 | Recall@1 | MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ])

    for prov, p_res in sorted(provider_robustness_metrics.items()):
        lines.append(
            f"| `{prov}` | {p_res['total_eval_states']} | {p_res['ndcg_at_1']:.4f} | {p_res['ndcg_at_3']:.4f} | "
            f"{p_res['recall_at_1']:.4f} | {p_res['mrr']:.4f} |"
        )

    # Ensure Groq is explicitly reported even if 0 test states in random split
    if "groq" not in provider_robustness_metrics:
        lines.append("| `groq` | 0 (all 9 in train/val) | N/A | N/A | N/A | N/A |")

    lines.extend([
        "",
        "## 7. Silver-Label Uncertainty Analysis",
        "Evaluates performance stratified by LLM silver-preference confidence:",
        "",
        "| Confidence Tier | Test States | NDCG@1 | NDCG@3 | Recall@1 | MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ])

    for tier, t_res in sorted(confidence_slice_metrics.items()):
        lines.append(
            f"| `{tier}` | {t_res['total_eval_states']} | {t_res['ndcg_at_1']:.4f} | {t_res['ndcg_at_3']:.4f} | "
            f"{t_res['recall_at_1']:.4f} | {t_res['mrr']:.4f} |"
        )

    lines.extend([
        "",
        "## 8. Safety & Zero-Leakage Audit",
        "",
        "| Invariant Check | Status | Verification Detail |",
        "|---|---|---|",
    ])

    for chk, dtl in audit_results.items():
        if chk != "overall_disposition":
            lines.append(f"| `{chk}` | **{dtl['status']}** | `{json.dumps(dtl)}` |")

    lines.extend([
        "",
        "## 9. Limitations & Research Caveats",
        "1. **Silver-Standard Proxy**: Recommenders are trained to emulate LLM-as-a-Judge preference patterns under scarce action labels, which cannot be assumed to be identical to plant ground truth without closed-loop field validation.",
        "2. **Groq Sample Size**: The Groq LPU tier comprises only 9 DecisionStates in the corpus and cannot support standalone statistical inference.",
        "3. **Heterogeneous Modality Sparsity**: Datasets vary substantially in instrumented physical channels (e.g. chemical plant vs single bearing rig), requiring modality presence indicators and hierarchical pooling.",
        "",
        "## 10. Recommended Next Step: Phase 5 Closed-Loop Simulation",
        "Phase 5 will deploy this hybrid recommender into closed-loop simulation environments (e.g. continuous CMAPSS turbofan wear trajectory and Tennessee Eastman chemical plant faults) to measure physical maintenance outcomes: avoided downtime, MTBF extension, and intervention ROI.",
    ])

    content = "\n".join(lines) + "\n"
    output_path.write_text(content, encoding="utf-8")

    # Short executive summary
    # Find table end index
    table_end_idx = 30
    for idx, l in enumerate(lines):
        if "## 4. Feature Representation" in l:
            table_end_idx = idx - 1
            break

    summary_lines = lines[:table_end_idx] + [
        "",
        "### Key Metrics Summary",
        f"- Hybrid Test NDCG@3: **{model_evaluations['hybrid_optimal']['ndcg_at_3']:.4f}**",
        f"- Hybrid Test Recall@1: **{model_evaluations['hybrid_optimal']['recall_at_1']:.4f}**",
        f"- Hybrid Test MRR: **{model_evaluations['hybrid_optimal']['mrr']:.4f}**",
        f"- Action Coverage: **{model_evaluations['hybrid_optimal']['action_coverage_pct']:.1f}%**",
        f"- Audit Disposition: **{audit_results['overall_disposition']}**",
        "",
        f"See full research report in `{output_path.name}`.",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
