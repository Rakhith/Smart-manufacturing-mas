# Maintenance Action Evaluation Request: State `DS_AI4I_2242`

## 1. Machine Asset & Operational Context
- **Asset Archetype**: `cnc_mill`
- **Industrial Domain**: `cnc_machining`
- **Data Dynamics**: `static_tabular`
- **Calibrated Severity Tier**: **DEGRADING**

## 2. Operating Telemetry & Subsystem Conditions
### Subsystem: Thermal
- `Air temperature [K]`: 299.2000
- `Process temperature [K]`: 308.4000
- `Air temperature [K]__canonical_degC`: 26.0500
- `Process temperature [K]__canonical_degC`: 35.2500
### Subsystem: Mechanical
- `Torque [Nm]`: 29.0000
### Subsystem: Kinematic
- `Rotational speed [rpm]`: 1667
### Subsystem: Process Operating
- `Type`: L
### Subsystem: Health Degradation
- `Tool wear [min]`: 199

## 3. Temporal Dynamics & Trend Summary
- Sequence Step: 2242
- (Instantaneous snapshot only — no longitudinal trend statistics)

## 4. Stated Data Limitations & Epistemic Boundaries
- ⚠️ Snapshot observation only; no longitudinal causal temporal trend available.
- ⚠️ Observed severity tier (DEGRADING) is distribution-calibrated relative to operational baselines.

## 5. Candidate Maintenance Actions to Evaluate
Evaluate each of the following candidate actions comparatively:
```json
[
  {
    "action_id": "ACT_MON_ENHANCED",
    "action_name": "Increase Monitoring Frequency & Alert Sensitivity",
    "category": "monitoring_observation",
    "description": "Shorten sensor sampling / logging interval, tighten statistical alarm thresholds, and track trailing trends closely.",
    "intervention_risk": "low",
    "operational_downtime_cost": "negligible",
    "inclusion_rationale": "High-frequency trend monitoring while maintenance staging is organized.",
    "trigger_evidence": [
      "Air temperature [K] = 299.200",
      "Process temperature [K] = 308.400"
    ]
  },
  {
    "action_id": "ACT_INSP_TOOL_WEAR",
    "action_name": "Inspect Cutting Tool & Insert Flank Wear",
    "category": "inspection_diagnosis",
    "description": "Perform optical / microscopic inspection of cutting edge, measure flank wear (VB), chip loading, and tool runout.",
    "intervention_risk": "low",
    "operational_downtime_cost": "low",
    "inclusion_rationale": "Tool degradation progressing; optical inspection required to determine remaining cutting life.",
    "trigger_evidence": [
      "Torque [Nm] = 29.000",
      "Tool wear [min] = 199.000"
    ]
  },
  {
    "action_id": "ACT_REPL_TOOL_INSERT",
    "action_name": "Replace Cutting Tool / Carbide Insert",
    "category": "replacement_overhaul",
    "description": "Index worn insert or swap out cutting tool holder, remeasure tool offset, and verify workpiece clearance.",
    "intervention_risk": "medium",
    "operational_downtime_cost": "medium",
    "inclusion_rationale": "Plan cutting tool or insert replacement to avert catastrophic workpiece gouging.",
    "trigger_evidence": [
      "Air temperature [K] = 299.200",
      "Process temperature [K] = 308.400"
    ]
  },
  {
    "action_id": "ACT_OP_ADJUST_PARAMS",
    "action_name": "Adjust Operational Speed, Feed & Process Parameters",
    "category": "operational_mitigation",
    "description": "Dial back spindle RPM, feed rate, chamber RF power, or reactor flow rate to relieve mechanical/thermal stress.",
    "intervention_risk": "medium",
    "operational_downtime_cost": "low",
    "inclusion_rationale": "Derate feed rate and spindle speed to slow degradation rate before scheduled tool change.",
    "trigger_evidence": [
      "Air temperature [K] = 299.200",
      "Process temperature [K] = 308.400"
    ]
  }
]
```

## 6. Evaluation Instructions & Scoring Rubric
Comparatively rank and score all candidates against the observed machine state. Scores must be integers from 0 to 100:
- `suitability_score` [0-100]: Technical fit for observed physical state & archetype.
- `urgency_score` [0-100]: Time criticality (90-100: immediate action required; 0-20: elective/routine).
- `expected_effectiveness_score` [0-100]: Expected success in arresting degradation or protecting asset.
- `operational_risk_score` [0-100]: Risk of unnecessary downtime, collateral damage, or waste.
- `confidence` [0.0-1.0]: Certainty of rating given the sufficiency and clarity of sensor telemetry.
- `rank`: Unique integer from 1 to K (1 = top recommended action; no tied ranks).
- `final_verdict`: Exactly one of 'RECOMMENDED', 'ACCEPTABLE_ALTERNATIVE', 'INAPPROPRIATE_AT_CURRENT_TIME', 'UNSAFE'.

Respond ONLY with a JSON object matching this exact structure:
```json
{
  "evaluations": [
    {
      "action_id": "<action_id>",
      "rank": 1,
      "suitability_score": <0-100>,
      "urgency_score": <0-100>,
      "expected_effectiveness_score": <0-100>,
      "operational_risk_score": <0-100>,
      "confidence": <0.0-1.0>,
      "evidence_used": ["<sensor or trend cited>"],
      "reasoning_summary": "<concise engineering justification>",
      "unsupported_assumptions": ["<any unverified assumptions or empty list>"],
      "final_verdict": "RECOMMENDED"
    }
  ],
  "top_recommended_action": "<action_id of rank 1>",
  "alternative_actions": ["<action_id of acceptable alternatives>"],
  "insufficient_information": false,
  "uncertainty_explanation": "<explanation if confidence is low, else empty string>"
}
```