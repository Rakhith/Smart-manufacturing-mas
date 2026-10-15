#!/usr/bin/env python3
"""Master Execution Entry Point for Phase 5 of the Smart Manufacturing MAS.

Executes:
1. Dataset Screening Gate & Feasibility Table (20 criteria)
2. Frozen Phase 4 Recommender verification & Invariant check
3. Action Ontology Mapping & Provenance Tracking
4. External OOD Prescriptive Evaluation (OIL-040 & Factory Sensors)
5. Industrial Maintenance Synthetic CMMS Work-Order Text Evaluation
6. Closed-Loop Simulation (Historical Replay vs. Surrogate Simulation)
7. Prescriptive Baselines Evaluation
8. Systematic Ablation Studies
9. Multi-Agent Orchestration & Grounded Technician Cards
10. Scientific Publication Figures (15 PNGs)
11. Comprehensive Phase 5 Final Report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from phase5.action_mapper import ActionMapper
from phase5.adapters import FactorySensorsAdapter, IndustrialSyntheticAdapter, Oil040Adapter
from phase5.ablations import Phase5AblationRunner
from phase5.baselines import NoMaintenancePolicy, RandomRankingPolicy, SeverityHeuristicPolicy
from phase5.closed_loop import ClosedLoopController, ExecutionMode, FeedbackStore
from phase5.evaluator import ExternalPrescriptiveEvaluator
from phase5.frozen_model import FrozenPhase4Recommender
from phase5.leakage_guard import TemporalLeakageGuard
from phase5.llm_summarizer import GroundedPrescriptiveSummarizer
from phase5.mas_coordinator import MultiAgentClosedLoopCoordinator
from phase5.report_compiler import save_phase5_report
from phase5.screening import run_screening
from phase5.visualizations import generate_all_visualizations


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 5 Prescriptive Validation Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "oil040", "factory_sensors", "industrial_synthetic"],
        help="Target dataset to evaluate",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "replay", "simulation"],
        help="Closed-loop evaluation mode",
    )
    parser.add_argument("--skip-ablations", action="store_true", help="Skip running ablation studies")
    parser.add_argument("--skip-visualizations", action="store_true", help="Skip figure generation")
    parser.add_argument("--output-dir", type=str, default="outputs/phase5", help="Root output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
    logging.info("===================================================================")
    logging.info("  STARTING PHASE 5: CLOSED-LOOP PRESCRIPTIVE VALIDATION")
    logging.info("===================================================================")

    # -------------------------------------------------------------
    # STEP 1: Dataset Screening Gate
    # -------------------------------------------------------------
    logging.info("\n[Step 1/11] Running Dataset Screening Gate & Feasibility Audit...")
    screening_dir = output_dir / "dataset_screening"
    df_screen, screen_report = run_screening(screening_dir)
    logging.info(f"  ✓ Screening matrix generated for {len(df_screen)} candidate profiles.")

    # -------------------------------------------------------------
    # STEP 2: Verify Frozen Phase 4 Model
    # -------------------------------------------------------------
    logging.info("\n[Step 2/11] Verifying and Loading Frozen Phase 4 Recommender...")
    recommender = FrozenPhase4Recommender()
    inv = recommender.verify_frozen_invariants()
    logging.info(f"  ✓ Model Hash: {inv['initial_sha256'][:16]}... Status: {inv['status']}")
    logging.info(f"  ✓ Invariants: Frozen Alpha = {inv['frozen_alpha']}, CBR Neighbors = {inv['cbr_k']}")

    # -------------------------------------------------------------
    # STEP 3: Action Ontology Mapping
    # -------------------------------------------------------------
    logging.info("\n[Step 3/11] Exporting Action Ontology Mappings & Provenance...")
    action_mapper = ActionMapper()
    mapping_dir = output_dir / "action_mapping"
    action_mapper.export_action_mapping_artifacts(mapping_dir)
    logging.info("  ✓ Action mapping artifacts saved under outputs/phase5/action_mapping/")

    evaluator = ExternalPrescriptiveEvaluator(recommender=recommender)

    # -------------------------------------------------------------
    # STEP 4: External OOD Evaluation: OIL-040
    # -------------------------------------------------------------
    eval_oil: Dict[str, Any] = {"summary_metrics": {}, "evaluation_records": []}
    if args.dataset in {"all", "oil040"}:
        logging.info("\n[Step 4/11] Evaluating on OIL-040 Rotating Equipment Benchmark...")
        oil_adapter = Oil040Adapter()
        oil_adapter.load()
        eval_oil = evaluator.evaluate_dataset(oil_adapter)
        oil_summary = eval_oil["summary_metrics"]
        logging.info(f"  ✓ OIL-040 Evaluation Points: {oil_summary['total_evaluation_points']}")
        logging.info(f"  ✓ Top-1 Exact Agreement: {oil_summary['top1_exact_agreement'] * 100:.2f}%")
        logging.info(f"  ✓ Top-3 Action Recall:   {oil_summary['top3_recall'] * 100:.2f}%")
        logging.info(f"  ✓ Category Agreement:   {oil_summary['category_agreement'] * 100:.2f}%")
        logging.info(f"  ✓ NDCG@1: {oil_summary['mean_ndcg_at_1']:.4f}, NDCG@3: {oil_summary['mean_ndcg_at_3']:.4f}, MRR: {oil_summary['mean_reciprocal_rank']:.4f}")
        logging.info(f"  ✓ Median Warning Lead Time: {oil_summary['lead_time_hours_median']:.1f} hours")

        eval_dir = output_dir / "external_evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_dir / "oil040_evaluation.json", "w", encoding="utf-8") as f:
            json.dump(eval_oil, f, indent=2, default=str)
        pd.DataFrame(eval_oil["evaluation_records"]).to_csv(eval_dir / "oil040_predictions.csv", index=False)

    # -------------------------------------------------------------
    # STEP 5: External OOD Evaluation: Factory Sensors
    # -------------------------------------------------------------
    eval_fs: Dict[str, Any] = {"summary_metrics": {}, "evaluation_records": []}
    if args.dataset in {"all", "factory_sensors"}:
        logging.info("\n[Step 5/11] Evaluating on Factory Sensors Centrifugal Pump Benchmark...")
        fs_adapter = FactorySensorsAdapter()
        fs_adapter.load()
        eval_fs = evaluator.evaluate_dataset(fs_adapter)
        fs_summary = eval_fs["summary_metrics"]
        logging.info(f"  ✓ Factory Sensors Points: {fs_summary['total_evaluation_points']}")
        logging.info(f"  ✓ Top-1 Exact Agreement: {fs_summary['top1_exact_agreement'] * 100:.2f}%")
        logging.info(f"  ✓ Top-3 Action Recall:   {fs_summary['top3_recall'] * 100:.2f}%")
        logging.info(f"  ✓ Category Agreement:   {fs_summary['category_agreement'] * 100:.2f}%")
        logging.info(f"  ✓ NDCG@1: {fs_summary['mean_ndcg_at_1']:.4f}, NDCG@3: {fs_summary['mean_ndcg_at_3']:.4f}, MRR: {fs_summary['mean_reciprocal_rank']:.4f}")

        eval_dir = output_dir / "external_evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_dir / "factory_sensors_evaluation.json", "w", encoding="utf-8") as f:
            json.dump(eval_fs, f, indent=2, default=str)
        pd.DataFrame(eval_fs["evaluation_records"]).to_csv(eval_dir / "factory_sensors_predictions.csv", index=False)

    # Save consolidated leakage audit
    evaluator.leakage_guard.save_audit_report(output_dir / "external_evaluation" / "temporal_leakage_audit.json")

    # -------------------------------------------------------------
    # STEP 6: Industrial Maintenance Synthetic Corpus Evaluation
    # -------------------------------------------------------------
    if args.dataset in {"all", "industrial_synthetic"}:
        logging.info("\n[Step 6/11] Auditing Industrial Maintenance Synthetic CMMS Text Corpus...")
        ind_adapter = IndustrialSyntheticAdapter()
        ind_adapter.load(sample_limit=20000)
        text_dist = ind_adapter.get_action_distribution(n_sample=1000)
        logging.info(f"  ✓ Mapped 1,000 Sample Descriptions to Canonical Actions: {text_dist['canonical_action_distribution']}")
        with open(output_dir / "external_evaluation" / "industrial_synthetic_text_evaluation.json", "w", encoding="utf-8") as f:
            json.dump(text_dist, f, indent=2, default=str)

    # -------------------------------------------------------------
    # STEP 7: Closed-Loop Simulations (Mode A & Mode B)
    # -------------------------------------------------------------
    logging.info("\n[Step 7/11] Running Closed-Loop Prescriptive Simulation Experiments...")
    coordinator = MultiAgentClosedLoopCoordinator(recommender=recommender)

    closed_loop_dir = output_dir / "closed_loop"
    closed_loop_dir.mkdir(parents=True, exist_ok=True)

    # Experiment 1: Historical Replay Mode on OIL-040 Asset 1
    replay_results = coordinator.run_asset_lifecycle(
        adapter=Oil040Adapter(),
        asset_id="EQ-0000001",
        mode=ExecutionMode.HISTORICAL_REPLAY,
        max_steps=20,
    )
    logging.info(f"  ✓ Historical Replay: {replay_results['steps_executed']} steps. Agreement: {replay_results['asset_metrics']['agreement_rate'] * 100:.1f}%")

    # Experiment 2: Surrogate Simulation Mode on Slurry Pump 1
    sim_results = coordinator.run_asset_lifecycle(
        adapter=FactorySensorsAdapter(),
        asset_id="1",
        mode=ExecutionMode.SURROGATE_SIMULATION,
        max_steps=20,
    )
    logging.info(f"  ✓ Surrogate Simulation: {sim_results['steps_executed']} steps. Prevented Failures: {sim_results['asset_metrics']['prevented_failures']}")

    with open(closed_loop_dir / "closed_loop_replay_summary.json", "w", encoding="utf-8") as f:
        json.dump(replay_results, f, indent=2, default=str)
    with open(closed_loop_dir / "closed_loop_simulation_summary.json", "w", encoding="utf-8") as f:
        json.dump(sim_results, f, indent=2, default=str)

    coordinator.feedback_store.export(closed_loop_dir / "closed_loop_transitions.csv")

    # -------------------------------------------------------------
    # STEP 8: Prescriptive Baselines Evaluation
    # -------------------------------------------------------------
    logging.info("\n[Step 8/11] Evaluating Prescriptive Baselines against Ground Truth...")
    baselines_dir = output_dir / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    oil_eval_records = eval_oil.get("evaluation_records", [])
    fs_eval_records = eval_fs.get("evaluation_records", [])
    baseline_metrics = {"oil040": {}, "factory_sensors": {}}

    cands = recommender.get_canonical_candidates()
    no_maint = NoMaintenancePolicy()
    heuristic = SeverityHeuristicPolicy()
    random_pol = RandomRankingPolicy(random_seed=42)

    def eval_policy_on_records(policy, records, adapter):
        n1s, n3s, mrrs, t1, t3, cat_m = [], [], [], 0, 0, 0
        for r in records:
            st = adapter.build_decision_state_at(r["asset_id"], r["prediction_timestamp"])
            rnk = policy.rank_candidates(st, cands)
            ids = [a["action_id"] for a in rnk]
            tgt = r["target_action"]
            n1s.append(ExternalPrescriptiveEvaluator.compute_ndcg_at_k(ids, tgt, 1))
            n3s.append(ExternalPrescriptiveEvaluator.compute_ndcg_at_k(ids, tgt, 3))
            mrrs.append(ExternalPrescriptiveEvaluator.compute_reciprocal_rank(ids, tgt))
            if ids[0] == tgt:
                t1 += 1
            if tgt in ids[:3]:
                t3 += 1
            target_cat = next((c["action_category"] for c in cands if c["action_id"] == tgt), "unknown")
            top_cat = rnk[0]["action_category"]
            if target_cat == top_cat:
                cat_m += 1
        n = len(records)
        return {
            "top1_exact_agreement": round(t1 / n, 4) if n > 0 else 0.0,
            "top3_recall": round(t3 / n, 4) if n > 0 else 0.0,
            "category_agreement": round(cat_m / n, 4) if n > 0 else 0.0,
            "mean_ndcg_at_1": round(float(np.mean(n1s)), 4) if n1s else 0.0,
            "mean_ndcg_at_3": round(float(np.mean(n3s)), 4) if n3s else 0.0,
            "mean_reciprocal_rank": round(float(np.mean(mrrs)), 4) if mrrs else 0.0,
        }

    # Evaluate on OIL-040
    if oil_eval_records:
        oil_ad = Oil040Adapter()
        oil_ad.load()
        baseline_metrics["oil040"]["random"] = eval_policy_on_records(random_pol, oil_eval_records, oil_ad)
        baseline_metrics["oil040"]["heuristic"] = eval_policy_on_records(heuristic, oil_eval_records, oil_ad)
        baseline_metrics["oil040"]["no_maintenance"] = eval_policy_on_records(no_maint, oil_eval_records, oil_ad)
        baseline_metrics["oil040"]["frozen_hybrid"] = eval_oil.get("summary_metrics", {})

    # Evaluate on Factory Sensors
    if fs_eval_records:
        fs_ad = FactorySensorsAdapter()
        fs_ad.load()
        baseline_metrics["factory_sensors"]["random"] = eval_policy_on_records(random_pol, fs_eval_records, fs_ad)
        baseline_metrics["factory_sensors"]["heuristic"] = eval_policy_on_records(heuristic, fs_eval_records, fs_ad)
        baseline_metrics["factory_sensors"]["no_maintenance"] = eval_policy_on_records(no_maint, fs_eval_records, fs_ad)
        baseline_metrics["factory_sensors"]["frozen_hybrid"] = eval_fs.get("summary_metrics", {})

    logging.info(f"  ✓ Factory Sensors Baselines - Random Top-1: {baseline_metrics.get('factory_sensors', {}).get('random', {}).get('top1_exact_agreement', 0)*100:.2f}%, Heuristic Top-1: {baseline_metrics.get('factory_sensors', {}).get('heuristic', {}).get('top1_exact_agreement', 0)*100:.2f}%, Frozen Hybrid: {eval_fs.get('summary_metrics', {}).get('top1_exact_agreement', 0)*100:.2f}%")

    with open(baselines_dir / "baseline_comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(baseline_metrics, f, indent=2, default=str)

    # -------------------------------------------------------------
    # STEP 9: Systematic Ablation Studies
    # -------------------------------------------------------------
    ablation_results = {}
    if not args.skip_ablations and oil_eval_records:
        logging.info("\n[Step 9/11] Running Prescriptive Component Ablations...")
        abl_runner = Phase5AblationRunner(recommender=recommender)
        oil_ad = Oil040Adapter()
        oil_ad.load()

        alpha_abls = abl_runner.run_alpha_sweep_ablation(oil_eval_records[:40], oil_ad)
        feat_abls = abl_runner.run_feature_representation_ablation(oil_eval_records[:40], oil_ad)

        ablation_results = {
            "alpha_sweep_ablation": alpha_abls,
            "feature_representation_ablation": feat_abls,
        }
        abl_runner.save_ablation_results(ablation_results, output_dir / "ablations" / "ablation_results.json")
        logging.info("  ✓ Saved ablation results under outputs/phase5/ablations/")

    # -------------------------------------------------------------
    # STEP 10: Multi-Agent Dispatch & LLM Summarizer Cards
    # -------------------------------------------------------------
    logging.info("\n[Step 10/11] Generating Grounded Technician Cards & CMMS Tickets...")
    summarizer = GroundedPrescriptiveSummarizer()
    llm_dir = output_dir / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)

    sample_cards = []
    if oil_eval_records:
        oil_ad = Oil040Adapter()
        oil_ad.load()
        for r in oil_eval_records[:5]:
            dstate = oil_ad.build_decision_state_at(r["asset_id"], r["prediction_timestamp"])
            rnk = recommender.rank_candidates(dstate)
            card = summarizer.generate_technician_card(dstate, rnk, actual_outcome={"target_action": r["target_action"], "downtime": r["downtime_hours"]})
            cmms = summarizer.format_cmms_work_order(card)
            sample_cards.append({"technician_card": card, "cmms_ticket": cmms})

    with open(llm_dir / "sample_technician_cards.json", "w", encoding="utf-8") as f:
        json.dump(sample_cards, f, indent=2, default=str)
    logging.info(f"  ✓ Generated {len(sample_cards)} grounded technician cards and CMMS tickets.")

    # -------------------------------------------------------------
    # STEP 11: Scientific Visualizations & Final Report
    # -------------------------------------------------------------
    if not args.skip_visualizations:
        logging.info("\n[Step 11/11] Generating 15 Publication-Grade Figures & Final Report...")
        figures_dir = output_dir / "figures"
        generate_all_visualizations(
            output_dir=figures_dir,
            eval_oil=eval_oil,
            eval_fs=eval_fs,
            baseline_metrics=baseline_metrics,
            ablation_res=ablation_results,
        )
        logging.info("  ✓ Generated 15 scientific figures under outputs/phase5/figures/")

    # Build Final Comprehensive 25-Section Report
    logging.info("\nCompiling Final 25-Section Scientific Report PHASE5_REPORT.md...")
    report_path = output_dir / "reports" / "PHASE5_REPORT.md"
    save_phase5_report(
        output_path=report_path,
        screening_df=df_screen,
        eval_oil=eval_oil,
        eval_fs=eval_fs,
        baselines=baseline_metrics,
        ablations=ablation_results,
    )
    logging.info(f"  ✓ Final report written to {report_path}")

    logging.info("===================================================================")
    logging.info("  PHASE 5 EXECUTION COMPLETED SUCCESSFULLY!")
    logging.info("===================================================================")


if __name__ == "__main__":
    main()
