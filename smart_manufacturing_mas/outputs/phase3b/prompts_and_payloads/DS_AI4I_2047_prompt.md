# Maintenance Action Evaluation Request: State `DS_AI4I_2047`

## 1. Machine Asset & Operational Context
- **Asset Archetype**: `cnc_mill`
- **Industrial Domain**: `cnc_machining`
- **Data Dynamics**: `static_tabular`
- **Calibrated Severity Tier**: **WATCH**

## 2. Operating Telemetry & Subsystem Conditions
### Subsystem: Thermal
- `Air temperature [K]`: 299.3000
- `Process temperature [K]`: 309.2000
- `Air temperature [K]__canonical_degC`: 26.1500
- `Process temperature [K]__canonical_degC`: 36.0500
### Subsystem: Mechanical
- `Torque [Nm]`: 24.2000
### Subsystem: Kinematic
- `Rotational speed [rpm]`: 1775
### Subsystem: Process Operating
- `Type`: L
### Subsystem: Health Degradation
- `Tool wear [min]`: 125

## 3. Temporal Dynamics & Trend Summary
- Sequence Step: 2047
- (Instantaneous snapshot only — no longitudinal trend statistics)

## 4. Stated Data Limitations & Epistemic Boundaries
- ⚠️ Snapshot observation only; no longitudinal causal temporal trend available.
- ⚠️ Observed severity tier (WATCH) is distribution-calibrated relative to operational baselines.

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
    "inclusion_rationale": "Telemetry shows early deviation or watch-tier variance requiring closer sampling frequency.",
    "trigger_evidence": [
      "Air temperature [K] = 299.300",
      "Process temperature [K] = 309.200"
    ]
  },
  {
    "action_id": "ACT_MON_CONTINUE",
    "action_name": "Continue Normal Operation",
    "category": "monitoring_observation",
    "description": "Maintain current production schedule and routine supervisory monitoring without manual intervention.",
    "intervention_risk": "low",
    "operational_downtime_cost": "negligible",
    "inclusion_rationale": "Passive baseline comparator to test whether intervention is prematurely invasive.",
    "trigger_evidence": [
      "Air temperature [K] = 299.300",
      "Process temperature [K] = 309.200"
    ]
  },
  {
    "action_id": "ACT_INSP_TOOL_WEAR",
    "action_name": "Inspect Cutting Tool & Insert Flank Wear",
    "category": "inspection_diagnosis",
    "description": "Perform optical / microscopic inspection of cutting edge, measure flank wear (VB), chip loading, and tool runout.",
    "intervention_risk": "low",
    "operational_downtime_cost": "low",
    "inclusion_rationale": "Tool wear indicators or spindle torque variations warrant optical/flank wear inspection.",
    "trigger_evidence": [
      "Torque [Nm] = 24.200",
      "Tool wear [min] = 125.000"
    ]
  },
  {
    "action_id": "ACT_OP_ADJUST_PARAMS",
    "action_name": "Adjust Operational Speed, Feed & Process Parameters",
    "category": "operational_mitigation",
    "description": "Dial back spindle RPM, feed rate, chamber RF power, or reactor flow rate to relieve mechanical/thermal stress.",
    "intervention_risk": "medium",
    "operational_downtime_cost": "low",
    "inclusion_rationale": "Modulate spindle speed or feed rate to relieve cutting resistance and tool stress.",
    "trigger_evidence": [
      "Torque [Nm] = 24.200",
      "Rotational speed [rpm] = 1775.000"
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