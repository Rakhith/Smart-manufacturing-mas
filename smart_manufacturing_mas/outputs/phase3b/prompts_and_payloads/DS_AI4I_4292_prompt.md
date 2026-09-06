# Maintenance Action Evaluation Request: State `DS_AI4I_4292`

## 1. Machine Asset & Operational Context
- **Asset Archetype**: `cnc_mill`
- **Industrial Domain**: `cnc_machining`
- **Data Dynamics**: `static_tabular`
- **Calibrated Severity Tier**: **CRITICAL**

## 2. Operating Telemetry & Subsystem Conditions
### Subsystem: Thermal
- `Air temperature [K]`: 301.8000
- `Process temperature [K]`: 310.1000
- `Air temperature [K]__canonical_degC`: 28.6500
- `Process temperature [K]__canonical_degC`: 36.9500
### Subsystem: Mechanical
- `Torque [Nm]`: 13.9000
### Subsystem: Kinematic
- `Rotational speed [rpm]`: 2372
### Subsystem: Process Operating
- `Type`: M
### Subsystem: Health Degradation
- `Tool wear [min]`: 205

## 3. Temporal Dynamics & Trend Summary
- Sequence Step: 4292
- (Instantaneous snapshot only — no longitudinal trend statistics)

## 4. Stated Data Limitations & Epistemic Boundaries
- ⚠️ Snapshot observation only; no longitudinal causal temporal trend available.
- ⚠️ Observed severity tier (CRITICAL) is distribution-calibrated relative to operational baselines.

## 5. Candidate Maintenance Actions to Evaluate
Evaluate each of the following candidate actions comparatively:
```json
[
  {
    "action_id": "ACT_REPL_TOOL_INSERT",
    "action_name": "Replace Cutting Tool / Carbide Insert",
    "category": "replacement_overhaul",
    "description": "Index worn insert or swap out cutting tool holder, remeasure tool offset, and verify workpiece clearance.",
    "intervention_risk": "medium",
    "operational_downtime_cost": "medium",
    "inclusion_rationale": "Tool wear / spindle load at critical threshold; immediate insert/tool replacement required.",
    "trigger_evidence": [
      "Air temperature [K] = 301.800",
      "Process temperature [K] = 310.100"
    ]
  },
  {
    "action_id": "ACT_OP_CONTROLLED_SHUTDOWN",
    "action_name": "Schedule Orderly Controlled Shutdown",
    "category": "operational_mitigation",
    "description": "Complete current production cycle/batch safely, ramp down speeds gracefully, and lock out asset for urgent maintenance.",
    "intervention_risk": "high",
    "operational_downtime_cost": "high",
    "inclusion_rationale": "Execute controlled cycle termination and lockout spindle to avoid machine collision.",
    "trigger_evidence": [
      "Air temperature [K] = 301.800",
      "Process temperature [K] = 310.100"
    ]
  },
  {
    "action_id": "ACT_OP_DERATE_LOAD",
    "action_name": "Derate Load & Curtail High-Stress Duty Cycles",
    "category": "operational_mitigation",
    "description": "Impose load cap (e.g. 70% nominal), restrict aggressive acceleration ramps, and avoid peak duty cycles until maintenance.",
    "intervention_risk": "medium",
    "operational_downtime_cost": "medium",
    "inclusion_rationale": "Immediately derate spindle torque and feed rate if shutdown must wait for cycle end.",
    "trigger_evidence": [
      "Air temperature [K] = 301.800",
      "Process temperature [K] = 310.100"
    ]
  },
  {
    "action_id": "ACT_INSP_TOOL_WEAR",
    "action_name": "Inspect Cutting Tool & Insert Flank Wear",
    "category": "inspection_diagnosis",
    "description": "Perform optical / microscopic inspection of cutting edge, measure flank wear (VB), chip loading, and tool runout.",
    "intervention_risk": "low",
    "operational_downtime_cost": "low",
    "inclusion_rationale": "Urgent post-stop visual inspection of cutting zone and spindle runout.",
    "trigger_evidence": [
      "Air temperature [K] = 301.800",
      "Process temperature [K] = 310.100"
    ]
  },
  {
    "action_id": "ACT_MON_CONTINUE",
    "action_name": "Continue Normal Operation",
    "category": "monitoring_observation",
    "description": "Maintain current production schedule and routine supervisory monitoring without manual intervention.",
    "intervention_risk": "low",
    "operational_downtime_cost": "negligible",
    "inclusion_rationale": "Passive baseline comparator (testing LLM resistance to dangerous inaction in critical state).",
    "trigger_evidence": [
      "Air temperature [K] = 301.800",
      "Process temperature [K] = 310.100"
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