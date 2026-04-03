# Architecture and Workflow

## Three-Tier Intelligence Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 1 — Cloud LLM (Gemini 2.5-Flash)                             │
│  Strategic Orchestration OR Reflexion Summary                       │
│  • Reasons about full workflow state                                │
│  • Chain-of-thought planning and error recovery (LLM mode)         │
│  • OR: interprets structured results via Reflexion (rules-first)   │
├─────────────────────────────────────────────────────────────────────┤
│  TIER 2 — Local SLM (Qwen3:4B via Ollama / LlamaCpp)              │
│  Tactical Edge Inference — ONE position retained                    │
│  • SLM 3b only: suggests IsolationForest contamination param       │
│  • Runs on factory edge node — no cloud dependency for this call   │
│  • HITL gate immediately follows; sklearn fallback if unavailable  │
├─────────────────────────────────────────────────────────────────────┤
│  TIER 3 — Rule-Based ToolDecider                                    │
│  Deterministic Decisions — zero latency, zero hallucination risk   │
│  • Imputer choice (KNN vs SimpleImputer based on missing %)        │
│  • Scaler choice (Robust vs Standard based on dataset size)        │
│  • Model family selection (3 tasks × 5 families = 15 outcomes)    │
└─────────────────────────────────────────────────────────────────────┘
```

## Five Agent Layers

1. **DataLoaderAgent** — CSV ingestion, schema discovery, data quality report.
2. **PreprocessingAgent** — typing, feature analysis, sklearn pipeline, optional PCA.
3. **DynamicAnalysisAgent** — model selection, training, adaptive retry, model cache.
4. **OptimizationAgent** — priority scoring, prescriptive recommendations.
5. **LLMPlannerAgent / RulesFirstPlannerAgent** — orchestration layer.

## Two Orchestration Modes

### LLM Mode (original)
```
User → LLMPlannerAgent.run_workflow_with_llm()
         ↓ (each step)
       Gemini decides tool → execute → update context → repeat
         ↓ (after all steps)
       IntelligentSummarizer (Reflexion loop)
```

### Rules-First Mode (NEW — Proposed Next Architecture)
```
User → RulesFirstPlannerAgent.run_workflow()
         ↓ Step 0: auto-detect problem type (HITL if confidence < 75%)
         ↓ Step 1: DataLoaderAgent
         ↓ Step 2: PreprocessingAgent (+ optional PCA)
         ↓ Step 3: DynamicAnalysisAgent (+ model cache)
         ↓ Step 4: OptimizationAgent + HITL review
         ↓ Step 5: Gemini Reflexion summary (draft → critique → revise)
```

## SLM Reduction Detail (4 → 1)

| Position | Task | Status | Replacement |
|----------|------|--------|-------------|
| SLM 1 | Perception / schema discovery | ELIMINATED | pandas dtypes + HITL operator selection |
| SLM 2 | Preprocessing strategy | ELIMINATED | ToolDecider if-else rules |
| SLM 3a | Model family selection | ELIMINATED | ToolDecider rule table |
| SLM 3b | Anomaly params (contamination) | **RETAINED** | irreplaceable: multi-signal interaction |
| SLM 4 | Narrative summary | ELIMINATED | Cloud LLM Reflexion loop |

### Why SLM 3b is the only irreplaceable position

`contamination` depends on a multi-signal interaction: outlier fraction + feature distribution shape + FP vs FN trade-off. Two datasets with identical outlier percentages may need very different values — a scalar if-else rule cannot generalise. The SLM provides an auditable `reason` field the operator can read and challenge. Output is bounded: contamination clipped to [0.001, 0.2], n_estimators floored at 50. A HITL gate immediately follows before IsolationForest runs.

## Adaptive Intelligence

Triggered when initial model performance falls below threshold:
- Classification: accuracy < 0.60
- Regression: R² < 0.10

The Analysis Agent systematically tries all model families, records all outcomes, and selects the best-performing configuration. All runs are logged for traceability and cumulative learning.

## Model Cache (New)

Cache key = SHA-256(dataset_basename + sorted_features + target + problem_type)[:16]

- Same column set + same dataset → **cache HIT** → instant load.
- Different column set → **cache MISS** → fresh train + save.
- Moving the project directory does not invalidate the cache (basename only used).

## Reflexion Loop (Cloud LLM)

Based on Shinn et al. 2023. Three LLM calls, same model:
1. **Draft**: generate narrative summary from structured context.
2. **Critique**: identify inaccuracies or missing numbers vs. actual metrics.
3. **Revise**: apply critique, produce final ≤200-word summary.

Cost: 3 additional API turns (not local inference). No new model dependency.
