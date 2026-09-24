# Maintenance Action Evaluation Request: State `DS_AI4I_162`

## 1. Machine Asset & Operational Context
- **Asset Archetype**: `cnc_mill`
- **Industrial Domain**: `cnc_machining`
- **Data Dynamics**: `static_tabular`
- **Calibrated Severity Tier**: **HEALTHY**

## 2. Operating Telemetry & Subsystem Conditions
### Subsystem: Thermal
- `Air temperature [K]`: 298.3000
- `Process temperature [K]`: 308.1000
- `Air temperature [K]__canonical_degC`: 25.1500
- `Process temperature [K]__canonical_degC`: 34.9500
### Subsystem: Mechanical
- `Torque [Nm]`: 35.5000
### Subsystem: Kinematic
- `Rotational speed [rpm]`: 1586
### Subsystem: Process Operating
- `Type`: L
### Subsystem: Health Degradation
- `Tool wear [min]`: 0

## 3. Temporal Dynamics & Trend Summary
- Sequence Step: 162
- (Instantaneous snapshot only — no longitudinal trend statistics)

## 4. Stated Data Limitations & Epistemic Boundaries
- ⚠️ Snapshot observation only; no longitudinal causal temporal trend available.
- ⚠️ Observed severity tier (HEALTHY) is distribution-calibrated relative to operational baselines.

## 5. Candidate Maintenance Actions to Evaluate
Evaluate each of the following candidate actions comparatively:
```json
[
  {
    "action_id": "ACT_MON_CONTINUE",
    "action_name": "Continue Normal Operation",
    "category": "monitoring_observation",
    "description": "Maintain current production schedule and routine supervisory monitoring without manual intervention.",
    "intervention_risk": "low",
    "operational_downtime_cost": "negligible",
    "inclusion_rationale": "Nominal operating telemetry within standard envelope; standard production can continue.",
    "trigger_evidence": [
      "Air temperature [K] = 298.300",
      "Process temperature [K] = 308.100"
    ]
  },
  {
    "action_id": "ACT_MON_ENHANCED",
    "action_name": "Increase Monitoring Frequency & Alert Sensitivity",
    "category": "monitoring_observation",
    "description": "Shorten sensor sampling / logging interval, tighten statistical alarm thresholds, and track trailing trends closely.",
    "intervention_risk": "low",
    "operational_downtime_cost": "negligible",
    "inclusion_rationale": "Conservative baseline option to track micro-variations and prevent undetected drift.",
    "trigger_evidence": [
      "Air temperature [K] = 298.300",
      "Process temperature [K] = 308.100"
    ]
  },
  {
    "action_id": "ACT_MON_PARAMETER_LOG",
    "action_name": "Log Process Parameter & Latency Drift",
    "category": "monitoring_observation",
    "description": "Flag operational parameter drift, network packet latency, or minor thermodynamic deviations for shift review.",
    "intervention_risk": "low",
    "operational_downtime_cost": "negligible",
    "inclusion_rationale": "Log process parameter baseline and electrical harmonics during normal run.",
    "trigger_evidence": [
      "Air temperature [K] = 298.300",
      "Process temperature [K] = 308.100"
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
- `reasoning_summary`: Strictly ONE brief sentence (maximum 15 words). Do not write multi-sentence paragraphs.

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
      "reasoning_summary": "<strictly 1 brief sentence, max 15 words>",
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