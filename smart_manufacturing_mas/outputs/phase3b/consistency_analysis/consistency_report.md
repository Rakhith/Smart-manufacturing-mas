# LLM-as-Judge Consistency & Ranking Stability Evaluation

**Generated**: 2026-09-06T02:21:48.254665+00:00  
**States Evaluated**: 20  
**Repeats per State**: 3  

## 1. Executive Summary & Assessment
> **MODERATE STABILITY: LLM Judge displays acceptable top-action consistency with moderate variance in lower-tier candidate ordering.**

## 2. Quantitative Stability Metrics

| Metric | Value | Target Benchmark | Status |
|---|---:|---:|:---:|
| **Top-1 Action Agreement** | **76.67%** | ≥ 80.0% | ⚠️ ACCEPTABLE |
| **Unanimous Choice Rate** | **35.00%** | ≥ 70.0% | ⚠️ ACCEPTABLE |
| **Spearman Rank Correlation ($\rho$)** | **0.7500** | ≥ 0.75 | ⚠️ ACCEPTABLE |
| **Suitability Score StdDev (MAD)** | **0.00 pts** | ≤ 8.0 pts | ✅ PASSED |
| **Urgency Score StdDev (MAD)** | **0.00 pts** | ≤ 8.0 pts | ✅ PASSED |
| **Confidence StdDev (MAD)** | **0.0000** | ≤ 0.10 | ✅ PASSED |
| **Positional Permutation Sensitivity** | **65.00%** | ≤ 20.0% | ⚠️ ACCEPTABLE |

## 3. Per-State Stability Breakdown

| State ID | Archetype | Severity | Top Action(s) | Agreement | Spearman $\rho$ | Permutation Sensitive? |
|---|---|---|---|---:|---:|:---:|
| `DS_AI4I_1013` | `cnc_mill` | HEALTHY | ACT_MON_ENHANCED, ACT_MON_PARAMETER_LOG, ACT_MON_ENHANCED | 66.7% | 0.667 | Yes |
| `DS_AI4I_1124` | `cnc_mill` | CRITICAL | ACT_INSP_TOOL_WEAR, ACT_INSP_TOOL_WEAR, ACT_INSP_TOOL_WEAR | 100.0% | 0.733 | No |
| `DS_AI4I_2763` | `cnc_mill` | DEGRADING | ACT_MON_ENHANCED, ACT_INSP_TOOL_WEAR, ACT_INSP_TOOL_WEAR | 66.7% | 0.867 | Yes |
| `DS_AI4I_7729` | `cnc_mill` | WATCH | ACT_MON_ENHANCED, ACT_INSP_TOOL_WEAR, ACT_INSP_TOOL_WEAR | 66.7% | 0.867 | Yes |
| `DS_CMAPSS_FD002_U8_C67` | `turbofan_engine` | WATCH | ACT_MON_ENHANCED, ACT_INSP_SUBSYSTEM, ACT_INSP_SUBSYSTEM | 66.7% | 0.867 | Yes |
| `DS_CMAPSS_FD003_U2_C233` | `turbofan_engine` | CRITICAL | ACT_INSP_THERMAL_ELECTRICAL, ACT_INSP_THERMAL_ELECTRICAL, ACT_INSP_THERMAL_ELECTRICAL | 100.0% | 0.733 | No |
| `DS_CMAPSS_FD004_U10_C304` | `turbofan_engine` | DEGRADING | ACT_MON_ENHANCED, ACT_INSP_THERMAL_ELECTRICAL, ACT_INSP_THERMAL_ELECTRICAL | 66.7% | 0.733 | Yes |
| `DS_CMAPSS_FD004_U6_C12` | `turbofan_engine` | HEALTHY | ACT_MON_ENHANCED, ACT_MON_PARAMETER_LOG, ACT_MON_ENHANCED | 66.7% | 0.667 | Yes |
| `DS_ETCH_CRIT_5916` | `etch_tool` | CRITICAL | ACT_INSP_SUBSYSTEM, ACT_INSP_SUBSYSTEM, ACT_INSP_SUBSYSTEM | 100.0% | 0.733 | No |
| `DS_ETCH_HEALTHY_5511` | `etch_tool` | HEALTHY | ACT_MON_ENHANCED, ACT_INSP_SUBSYSTEM, ACT_MON_ENHANCED | 66.7% | 0.667 | Yes |
| `DS_IIOT_CRIT_62692` | `iiot_machine` | CRITICAL | ACT_INSP_SUBSYSTEM, ACT_INSP_SUBSYSTEM, ACT_INSP_SUBSYSTEM | 100.0% | 0.867 | No |
| `DS_IIOT_HEALTHY_84639` | `iiot_machine` | HEALTHY | ACT_MON_ENHANCED, ACT_MON_PARAMETER_LOG, ACT_MON_ENHANCED | 66.7% | 0.667 | Yes |
| `DS_IIOT_WATCH_8677` | `iiot_machine` | WATCH | ACT_MON_ENHANCED, ACT_INSP_SUBSYSTEM, ACT_INSP_SUBSYSTEM | 66.7% | 0.867 | Yes |
| `DS_IMS_2nd_test_CRIT_916` | `bearing_test_rig` | CRITICAL | ACT_INSP_VIBRATION, ACT_INSP_VIBRATION, ACT_INSP_VIBRATION | 100.0% | 0.933 | No |
| `DS_IMS_2nd_test_HEALTHY_73` | `bearing_test_rig` | HEALTHY | ACT_MON_ENHANCED, ACT_INSP_SUBSYSTEM, ACT_MON_ENHANCED | 66.7% | 0.667 | Yes |
| `DS_IMS_2nd_test_WATCH_514` | `bearing_test_rig` | WATCH | ACT_MON_ENHANCED, ACT_INSP_LUBRICATION, ACT_INSP_VIBRATION | 33.3% | 0.467 | Yes |
| `DS_METRO_CRIT_237752` | `air_compressor` | CRITICAL | ACT_INSP_PRESSURE_HYDRAULIC, ACT_INSP_PRESSURE_HYDRAULIC, ACT_INSP_PRESSURE_HYDRAULIC | 100.0% | 0.933 | No |
| `DS_METRO_HEALTHY_265516` | `air_compressor` | HEALTHY | ACT_MON_ENHANCED, ACT_MON_PARAMETER_LOG, ACT_MON_ENHANCED | 66.7% | 0.667 | Yes |
| `DS_MILL_CRIT_966` | `cnc_mill` | CRITICAL | ACT_INSP_TOOL_WEAR, ACT_INSP_TOOL_WEAR, ACT_INSP_TOOL_WEAR | 100.0% | 0.733 | No |
| `DS_MILL_HEALTHY_281` | `cnc_mill` | HEALTHY | ACT_MON_ENHANCED, ACT_INSP_SUBSYSTEM, ACT_MON_ENHANCED | 66.7% | 0.667 | Yes |
