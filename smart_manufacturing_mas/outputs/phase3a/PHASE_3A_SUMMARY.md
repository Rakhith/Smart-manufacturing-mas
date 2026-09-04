# Phase 3A Summary — Decision-State Construction & Representative State Selection

Generated: 2026-09-04T09:29:17.923874+00:00

## 1. Executive Overview
Phase 3A has successfully converted the multi-million-row Phase 2 telemetry corpus into a compact, diverse, non-redundant corpus of **1,023 representative DecisionStates** (target budget: ~1,100).
It eliminates redundant steady-state observations while preserving decision-relevant degradation milestones, failure-adjacent transitions, and cross-industry asset variety.

## 2. Corpus Funnel & Reduction Statistics
- **Phase 2 Raw Telemetry Observations**: ~**6,530,000** records across 11 dataset streams.
- **Candidate Decision States Harvested**: **2,915** (focused on degradation steps, change points, and failure windows).
- **Redundant States Pruned (Medoid Clustering)**: **1,101** duplicate/near-identical observations eliminated.
- **Final Representative Decision States Selected**: **1,023** states.
- **Overall Corpus Compression Ratio**: **0.0157%** of raw telemetry retained (over 99.98% compression with 100% critical state retention).

## 3. Dataset Distribution & Weighted Allocation

| Dataset ID | Asset Archetype | Temporal Type | Harvested | Redundant Removed | Selected Quota | Weighting Rationale |
|---|---|---|---:|---:|---:|---|
| `ai4i_2020` | `cnc_mill` | `static_tabular` | 599 | 391 | **116** | Standard multi-modal physical allocation (w=1.10) |
| `cmapss` | `turbofan_engine` | `time_series` | 560 | 357 | **121** | Standard multi-modal physical allocation (w=1.15) |
| `smart_maintenance_static` | `industrial_machine` | `static_tabular` | 195 | 0 | **100** | Standard multi-modal physical allocation (w=0.95) |
| `smart_maintenance_timeseries` | `industrial_machine` | `time_series` | 220 | 0 | **110** | Standard multi-modal physical allocation (w=1.05) |
| `iiot_6g` | `iiot_machine` | `time_series` | 200 | 0 | **105** | Standard multi-modal physical allocation (w=1.00) |
| `metal_etch` | `etch_tool` | `static_tabular` | 165 | 0 | **90** | Standard multi-modal physical allocation (w=0.85) |
| `metropt3` | `air_compressor` | `time_series` | 140 | 77 | **63** | Standard multi-modal physical allocation (w=1.05) |
| `uci_hydraulic` | `hydraulic_test_rig` | `multirate_cycle` | 216 | 50 | **121** | Standard multi-modal physical allocation (w=1.15) |
| `tennessee_eastman` | `chemical_plant` | `time_series` | 200 | 26 | **74** | Weighted lower (139 cols, anonymous physical mapping) (w=0.70) |
| `nasa_ims` | `bearing_test_rig` | `high_frequency_vibration` | 270 | 83 | **90** | Standard multi-modal physical allocation (w=0.85) |
| `nasa_milling` | `cnc_mill` | `signal_snapshot` | 150 | 117 | **33** | Standard multi-modal physical allocation (w=0.70) |

## 4. Severity & Health State Distribution

| Dataset ID | HEALTHY | WATCH | DEGRADING | CRITICAL | Total Selected |
|---|---:|---:|---:|---:|---:|
| `ai4i_2020` | 29 | 29 | 29 | 29 | **116** |
| `cmapss` | 30 | 30 | 30 | 31 | **121** |
| `iiot_6g` | 35 | 35 | 0 | 35 | **105** |
| `metal_etch` | 45 | 0 | 0 | 45 | **90** |
| `metropt3` | 30 | 0 | 0 | 33 | **63** |
| `nasa_ims` | 37 | 32 | 0 | 21 | **90** |
| `nasa_milling` | 6 | 13 | 0 | 14 | **33** |
| `smart_maintenance_static` | 33 | 33 | 0 | 34 | **100** |
| `smart_maintenance_timeseries` | 37 | 36 | 0 | 37 | **110** |
| `tennessee_eastman` | 25 | 24 | 0 | 25 | **74** |
| `uci_hydraulic` | 21 | 34 | 36 | 30 | **121** |

## 5. Invariant Validation & Safety Assertions
- **Zero Target Leakage**: **PASSED** (0 target/label columns exist in `input_machine_state`).
- **All 11 Datasets Represented**: **PASSED** (every active stream has a dedicated quota).
- **Bounded Representation**: Tennessee Eastman (5.73M rows) contributed only **74** states, strictly preventing dataset domination.
- **Target Budget Compliance**: Final count is **1023** states (target: 1100).
- **Deterministic Reproducibility**: Seed fixed at 42 for all clustering and sampling operations.

## 6. Downstream Interface & Phase Boundary
- **LLM-Ready Formatting**: Each DecisionState is exportable into a structured Markdown Machine Health Card (~350–650 tokens) and clean JSONL payload under `outputs/phase3a/llm_ready_states/`.
- **Strict Phase Boundary**: No LLM calls were made, no maintenance recommendations were generated, and no rankers were trained.
- This decision-state corpus is now fully prepared for Phase 3B/Phase 4 (LLM-as-a-Judge for maintenance action ranking).
