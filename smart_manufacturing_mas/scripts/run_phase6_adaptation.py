"""Reproducible CLI entry point for Phase 6 Low-Resource Cross-Domain Adaptation."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase6.config import AdaptationExperimentConfig
from phase6.experiment_runner import AdaptationExperimentRunner
from phase6.report_generator import Phase6ReportGenerator
from phase6.visualizations import Phase6Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 6 Low-Resource Cross-Domain Adaptation Benchmark")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["factory_sensors", "oil040", "cmapss_lodo"],
        help="Target domains to benchmark (choices: factory_sensors, oil040, cmapss_lodo)",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=[0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00],
        help="Local adaptation data budgets as fractions in [0.0, 1.0]",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/phase6",
        help="Output directory for results, manifests, figures, and report",
    )
    parser.add_argument(
        "--quick_smoke_test",
        action="store_true",
        help="Quick smoke test mode with reduced budgets for rapid verification",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()
    print("=" * 80)
    print("PHASE 6: LOW-RESOURCE CROSS-DOMAIN ADAPTATION BENCHMARK")
    print("Smart Manufacturing Multi-Agent System (MAS)")
    print("=" * 80)

    budgets = [0.0, 0.10, 1.00] if args.quick_smoke_test else args.budgets
    config = AdaptationExperimentConfig(
        target_domains=args.domains,
        budget_fractions=budgets,
        output_dir=Path(args.output_dir),
    )

    print(f"Target Domains : {config.target_domains}")
    print(f"Data Budgets   : {config.budget_fractions}")
    print(f"Output Dir     : {config.output_dir}")
    print("-" * 80)

    # 1. Run benchmark
    runner = AdaptationExperimentRunner(config=config)
    benchmark_res = runner.run_all_domains()

    df_results = benchmark_res["results_df"]
    df_agg = benchmark_res["aggregate_df"]
    manifests = benchmark_res["manifests"]
    crossovers = benchmark_res["crossovers"]
    leakage_audit = runner.leakage_auditor.generate_audit_report(config.output_dir / "leakage_audit.json")

    # 2. Generate high-resolution figures
    print("\n>>> Generating Publication-Grade Visualizations (300 DPI)...")
    visualizer = Phase6Visualizer(output_dir=config.output_dir / "figures")

    if "factory_sensors" in config.target_domains:
        visualizer.plot_domain_data_efficiency(
            df_results=df_results,
            domain_id="factory_sensors",
            title="Factory Sensors Centrifugal Slurry Pumps",
            filename="fig1_data_efficiency_factory_sensors.png",
        )

    if "oil040" in config.target_domains:
        visualizer.plot_domain_data_efficiency(
            df_results=df_results,
            domain_id="oil040",
            title="OIL-040 Industrial Rotating Turbomachinery",
            filename="fig2_data_efficiency_oil040.png",
        )

    if "cmapss_lodo" in config.target_domains:
        visualizer.plot_domain_data_efficiency(
            df_results=df_results,
            domain_id="cmapss_lodo",
            title="NASA C-MAPSS Turbofan Engines (Strict LODO)",
            filename="fig3_data_efficiency_cmapss.png",
        )

    visualizer.plot_crossover_analysis(df_results=df_results, filename="fig4_crossover_analysis.png")
    visualizer.plot_method_comparison_bar(df_results=df_results, filename="fig5_method_comparison.png")
    visualizer.plot_negative_transfer_analysis(df_results=df_results, filename="fig6_negative_transfer.png")

    # 3. Generate comprehensive markdown report
    print("\n>>> Generating Comprehensive Scientific Research Report...")
    report_gen = Phase6ReportGenerator(output_dir=config.output_dir)
    report_path = report_gen.generate_report(
        df_results=df_results,
        df_aggregate=df_agg,
        manifests=manifests,
        crossovers=crossovers,
        leakage_audit=leakage_audit,
        experiment_config=config.to_dict(),
    )

    elapsed = time.time() - t_start
    print("=" * 80)
    print(f"PHASE 6 EXECUTION COMPLETED in {elapsed:.2f} seconds!")
    print(f"Results CSV       : {config.output_dir / 'results.csv'}")
    print(f"Aggregate CSV     : {config.output_dir / 'aggregate_results.csv'}")
    print(f"Split Manifest    : {config.output_dir / 'split_manifest.json'}")
    print(f"Leakage Audit     : {config.output_dir / 'leakage_audit.json'} ({leakage_audit['overall_status']})")
    print(f"Research Report   : {report_path}")
    print(f"Figures Directory : {config.output_dir / 'figures'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
