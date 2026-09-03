# Phase 1 Summary - Dataset Preparation

Generated: 2026-09-03T07:14:58.480432+00:00

## Scope

Dataset-specific preparation is complete through prepared observations. No semantic normalization, ontology mapping, severity mapping, action generation, recommender, LLM-as-a-Judge, LGBMRanker, collaborative filtering, or closed-loop functionality is included.

## Dataset Status

| Dataset | Type | Status | Raw Records | Prepared Observations | Labels/Targets | Notes |
|---|---|---|---:|---:|---|---|
| AI4I 2020 | static_tabular | SUCCESS | 10000 | 10000 | Machine failure, TWF, HDF, PWF, OSF, RNF | Original source columns are retained. Derived columns use the __ suffix. |
| NASA C-MAPSS | run_to_failure | SUCCESS | 265256 | 265256 | RUL, fault_regime_metadata | FD001/FD002 contain HPC degradation; FD003/FD004 include HPC and fan degradation according to the included readme. |
| Smart Manufacturing Maintenance | mixed_tabular_time_series | SUCCESS | 101430 | 101430 | maintenance_priority, anomaly, failure_type, maintenance_required | The two variants are prepared separately and must not be merged in Phase 1. |
| Intelligent Manufacturing 6G | low_frequency_time_series | SUCCESS | 100000 | 100000 | Efficiency_Status | Original source columns are retained. Derived columns use the __ suffix. |
| NASA Milling Tool Wear | run_to_failure_signal | SUCCESS_WITH_LIMITATIONS | 1002 | 1002 | tool_wear_available | No physical field name is assumed from the nested MATLAB structure. Signal sampling rate is not assigned without a supported source.; Prepared observations retain source_array as provenance. |
| UCI MetroPT-3 | low_frequency_time_series | SUCCESS_WITH_LIMITATIONS | 1516948 | 306960 | failure_reports_documented | One-minute windows are used to keep Phase 1 artifacts tractable; raw 1 Hz records remain untouched.; Failure reports in the PDF are not expanded into row-level labels in this phase. |
| UCI Hydraulic Systems | multirate_cycle_signal | SUCCESS | 2205 | 2205 | cooler_condition, valve_condition, pump_leakage, accumulator_condition | Raw sensor matrices are preserved in data/. Prepared output contains per-cycle summaries only. |
| Tennessee Eastman Process | multivariate_process_time_series | SUCCESS_WITH_LIMITATIONS | 5730000 | 5730000 | fault_number | The repository does not contain the Tennessee Eastman variable dictionary; physical variable meanings remain UNKNOWN/UNVERIFIED pending source documentation. |
| NASA IMS Bearings | high_frequency_signal | SUCCESS_WITH_LIMITATIONS | 3140 | 3140 | end_of_test_failure_description | Known failure descriptions apply only at test end, not as per-snapshot labels.; 0 waveform files were unreadable. |
| Metal Etch | static_tabular | SUCCESS_WITH_LIMITATIONS | 10770 | 10663 | Target | Anonymous features; physical meanings and units are unverified.; Original source columns are retained. Derived columns use the __ suffix. |
| Digital Manufacturing (deprecated) | static_tabular | SKIPPED_NOT_AVAILABLE | 9800 | 0 | demand_class | Retained for inventory only; excluded from active PreparedObservation corpus.; Excluded from the active Phase 1 corpus because its source documentation categorizes it as deprecated and its non-engineering business fields dominate. |

## Overall Summary

- Available catalogue entries: 11 (10 active datasets and 1 deprecated inventory-only entry).
- Successfully processed: 10.
- Success with limitations: 5.
- Failed: 0.
- Skipped: 1 deprecated dataset, retained for inventory only.
- Modalities: thermal, mechanical/vibration, electrical, hydraulic/fluid, acoustic, process, operating context, degradation/health, and business/economic fields where documented.
- Processing types: static tabular, low-frequency time series, run-to-failure trajectories, high-frequency signal snapshots, multirate cycle signals, and multivariate process time series.

## Major Limitations

- Most datasets contain condition, fault, degradation, RUL, or process labels rather than maintenance action/outcome histories.
- Tennessee Eastman xmeas/xmv physical mappings are unavailable in the repository and remain UNKNOWN/UNVERIFIED.
- Metal Etch features are anonymous and have no verified physical interpretation.
- MetroPT-3 source failure reports are not converted into row-level labels.
- NASA IMS failure descriptions apply at test end, not to every preceding snapshot.
- NASA Milling is conservatively extracted from nested MATLAB variables without unsupported semantics.

## Next-phase Input

Every active dataset has separate prepared observations, quality/preprocessing reports, catalogues, derived features where applicable, provenance, and representative visualizations. These artifacts are ready for semantic normalization while preserving dataset-specific physical meaning.
