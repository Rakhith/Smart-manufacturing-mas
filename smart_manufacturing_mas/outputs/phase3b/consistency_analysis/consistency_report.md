# LLM-as-Judge Consistency & Ranking Stability Evaluation

**Generated**: 2026-09-22T11:02:16.433571+00:00  
**States Evaluated**: 10  
**Repeats per State**: 3  

## 1. Executive Summary & Assessment
> **MODERATE STABILITY: LLM Judge displays acceptable top-action consistency with moderate variance in lower-tier candidate ordering.**

## 2. Quantitative Stability Metrics

| Metric | Value | Target Benchmark | Status |
|---|---:|---:|:---:|
| **Top-1 Action Agreement** | **80.00%** | ≥ 80.0% | ✅ PASSED |
| **Unanimous Choice Rate** | **40.00%** | ≥ 70.0% | ⚠️ ACCEPTABLE |
| **Spearman Rank Correlation ($\rho$)** | **0.7267** | ≥ 0.75 | ⚠️ ACCEPTABLE |
| **Suitability Score StdDev (MAD)** | **2.27 pts** | ≤ 8.0 pts | ✅ PASSED |
| **Urgency Score StdDev (MAD)** | **2.58 pts** | ≤ 8.0 pts | ✅ PASSED |
| **Confidence StdDev (MAD)** | **0.0092** | ≤ 0.10 | ✅ PASSED |
| **Positional Permutation Sensitivity** | **60.00%** | ≤ 20.0% | ⚠️ ACCEPTABLE |

## 3. Per-State Stability Breakdown

| State ID | Archetype | Severity | Top Action(s) | Agreement | Spearman $\rho$ | Permutation Sensitive? |
|---|---|---|---|---:|---:|:---:|
| `DS_AI4I_2047` | `cnc_mill` | WATCH | ACT_INSP_TOOL_WEAR, ACT_INSP_TOOL_WEAR, ACT_INSP_TOOL_WEAR | 100.0% | 0.867 | No |
| `DS_AI4I_2166` | `cnc_mill` | CRITICAL | ACT_INSP_TOOL_WEAR, ACT_INSP_TOOL_WEAR, ACT_INSP_TOOL_WEAR | 100.0% | 0.733 | No |
| `DS_AI4I_2519` | `cnc_mill` | HEALTHY | ACT_MON_ENHANCED, ACT_MON_PARAMETER_LOG, ACT_MON_ENHANCED | 66.7% | 0.667 | Yes |
| `DS_AI4I_7502` | `cnc_mill` | DEGRADING | ACT_MON_ENHANCED, ACT_INSP_TOOL_WEAR, ACT_INSP_TOOL_WEAR | 66.7% | 0.867 | Yes |
| `DS_CMAPSS_FD004_U11_C289` | `turbofan_engine` | CRITICAL | ACT_INSP_THERMAL_ELECTRICAL, ACT_INSP_THERMAL_ELECTRICAL, ACT_INSP_THERMAL_ELECTRICAL | 100.0% | 0.733 | No |
| `DS_CMAPSS_FD004_U12_C177` | `turbofan_engine` | WATCH | ACT_MON_ENHANCED, ACT_INSP_SUBSYSTEM, ACT_INSP_SUBSYSTEM | 66.7% | 0.867 | Yes |
| `DS_CMAPSS_FD004_U2_C12` | `turbofan_engine` | HEALTHY | ACT_MON_ENHANCED, ACT_MON_PARAMETER_LOG, ACT_MON_ENHANCED | 66.7% | 0.667 | Yes |
| `DS_CMAPSS_FD004_U4_C224` | `turbofan_engine` | DEGRADING | ACT_MON_ENHANCED, ACT_INSP_THERMAL_ELECTRICAL, ACT_INSP_THERMAL_ELECTRICAL | 66.7% | 0.733 | Yes |
| `DS_IIOT_CRIT_19469` | `iiot_machine` | CRITICAL | ACT_INSP_SUBSYSTEM, ACT_INSP_SUBSYSTEM, ACT_INSP_SUBSYSTEM | 100.0% | 0.467 | No |
| `DS_IIOT_HEALTHY_31354` | `iiot_machine` | HEALTHY | ACT_MON_ENHANCED, ACT_MON_PARAMETER_LOG, ACT_MON_ENHANCED | 66.7% | 0.667 | Yes |
