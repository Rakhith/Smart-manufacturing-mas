# Prescriptive Maintenance Recommendation Explanation Cards

## State `DS_AI4I_1969`
- **Recommended Action**: **`ACT_INSP_TOOL_WEAR`** (Inspect Cutting Tool & Insert Flank Wear)
- **Learned Recommender Score**: `1.0000`
- **Asset & Severity**: `cnc_mill` | Severity: **WATCH**
### Candidate Ranked Order:
  1. `ACT_INSP_TOOL_WEAR` (Score: 1.0000)
  2. `ACT_MON_ENHANCED` (Score: 0.9236)
  3. `ACT_OP_ADJUST_PARAMS` (Score: 0.1541)
  4. `ACT_MON_CONTINUE` (Score: 0.0000)
### Case-Based Reasoning Precedents:
  - `DS_AI4I_3586` (ai4i_2020, Sim: 0.999) -> Historical Best: `ACT_MON_ENHANCED`
  - `DS_AI4I_983` (ai4i_2020, Sim: 0.999) -> Historical Best: `ACT_INSP_TOOL_WEAR`
  - `DS_AI4I_2902` (ai4i_2020, Sim: 0.998) -> Historical Best: `ACT_INSP_TOOL_WEAR`

---

## State `DS_AI4I_4730`
- **Recommended Action**: **`ACT_REPL_TOOL_INSERT`** (Replace Cutting Tool / Carbide Insert)
- **Learned Recommender Score**: `1.0000`
- **Asset & Severity**: `cnc_mill` | Severity: **DEGRADING**
### Candidate Ranked Order:
  1. `ACT_REPL_TOOL_INSERT` (Score: 1.0000)
  2. `ACT_INSP_TOOL_WEAR` (Score: 0.7992)
  3. `ACT_MON_ENHANCED` (Score: 0.4836)
  4. `ACT_OP_ADJUST_PARAMS` (Score: 0.0278)
### Case-Based Reasoning Precedents:
  - `DS_AI4I_6076` (ai4i_2020, Sim: 0.999) -> Historical Best: `ACT_REPL_TOOL_INSERT`
  - `DS_AI4I_1590` (ai4i_2020, Sim: 0.999) -> Historical Best: `ACT_INSP_TOOL_WEAR`
  - `DS_AI4I_8282` (ai4i_2020, Sim: 0.999) -> Historical Best: `ACT_REPL_TOOL_INSERT`

---

## State `DS_AI4I_4731`
- **Recommended Action**: **`ACT_REPL_TOOL_INSERT`** (Replace Cutting Tool / Carbide Insert)
- **Learned Recommender Score**: `1.0000`
- **Asset & Severity**: `cnc_mill` | Severity: **CRITICAL**
### Candidate Ranked Order:
  1. `ACT_REPL_TOOL_INSERT` (Score: 1.0000)
  2. `ACT_INSP_TOOL_WEAR` (Score: 0.7371)
  3. `ACT_OP_CONTROLLED_SHUTDOWN` (Score: 0.6858)
  4. `ACT_OP_DERATE_LOAD` (Score: 0.4711)
  5. `ACT_MON_CONTINUE` (Score: 0.0000)
### Case-Based Reasoning Precedents:
  - `DS_AI4I_2864` (ai4i_2020, Sim: 0.999) -> Historical Best: `ACT_REPL_TOOL_INSERT`
  - `DS_AI4I_4384` (ai4i_2020, Sim: 0.998) -> Historical Best: `ACT_REPL_TOOL_INSERT`
  - `DS_AI4I_7510` (ai4i_2020, Sim: 0.998) -> Historical Best: `ACT_REPL_TOOL_INSERT`

---

## State `DS_CMAPSS_FD002_U17_C217`
- **Recommended Action**: **`ACT_MON_ENHANCED`** (Increase Monitoring Frequency & Alert Sensitivity)
- **Learned Recommender Score**: `1.0000`
- **Asset & Severity**: `turbofan_engine` | Severity: **WATCH**
### Candidate Ranked Order:
  1. `ACT_MON_ENHANCED` (Score: 1.0000)
  2. `ACT_MON_CONTINUE` (Score: 0.6694)
  3. `ACT_INSP_SUBSYSTEM` (Score: 0.5164)
  4. `ACT_CORR_CALIBRATE` (Score: 0.0000)
### Case-Based Reasoning Precedents:
  - `DS_CMAPSS_FD002_U1_C161` (cmapss, Sim: 1.000) -> Historical Best: `ACT_MON_ENHANCED`
  - `DS_CMAPSS_FD002_U16_C57` (cmapss, Sim: 0.999) -> Historical Best: `ACT_MON_ENHANCED`
  - `DS_CMAPSS_FD002_U8_C67` (cmapss, Sim: 0.999) -> Historical Best: `ACT_MON_ENHANCED`

---

## State `DS_CMAPSS_FD002_U4_C215`
- **Recommended Action**: **`ACT_REPL_OVERHAUL`** (Schedule Planned Subsystem Overhaul)
- **Learned Recommender Score**: `0.9780`
- **Asset & Severity**: `turbofan_engine` | Severity: **CRITICAL**
### Candidate Ranked Order:
  1. `ACT_REPL_OVERHAUL` (Score: 0.9780)
  2. `ACT_OP_CONTROLLED_SHUTDOWN` (Score: 0.8067)
  3. `ACT_OP_DERATE_LOAD` (Score: 0.7501)
  4. `ACT_INSP_THERMAL_ELECTRICAL` (Score: 0.4567)
  5. `ACT_MON_CONTINUE` (Score: 0.0000)
### Case-Based Reasoning Precedents:
  - `DS_CMAPSS_FD002_U14_C132` (cmapss, Sim: 0.997) -> Historical Best: `ACT_OP_DERATE_LOAD`
  - `DS_CMAPSS_FD004_U11_C296` (cmapss, Sim: 0.995) -> Historical Best: `ACT_REPL_OVERHAUL`
  - `DS_CMAPSS_FD002_U15_C208` (cmapss, Sim: 0.995) -> Historical Best: `ACT_REPL_OVERHAUL`

---

## State `DS_CMAPSS_FD002_U4_C135`
- **Recommended Action**: **`ACT_MON_ENHANCED`** (Increase Monitoring Frequency & Alert Sensitivity)
- **Learned Recommender Score**: `1.0000`
- **Asset & Severity**: `turbofan_engine` | Severity: **WATCH**
### Candidate Ranked Order:
  1. `ACT_MON_ENHANCED` (Score: 1.0000)
  2. `ACT_MON_CONTINUE` (Score: 0.6850)
  3. `ACT_INSP_SUBSYSTEM` (Score: 0.5114)
  4. `ACT_CORR_CALIBRATE` (Score: 0.0000)
### Case-Based Reasoning Precedents:
  - `DS_CMAPSS_FD004_U16_C87` (cmapss, Sim: 0.998) -> Historical Best: `ACT_MON_ENHANCED`
  - `DS_CMAPSS_FD002_U5_C61` (cmapss, Sim: 0.998) -> Historical Best: `ACT_MON_ENHANCED`
  - `DS_CMAPSS_FD004_U3_C205` (cmapss, Sim: 0.996) -> Historical Best: `ACT_MON_ENHANCED`

---
