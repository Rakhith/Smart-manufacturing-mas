# Phase 3A Reproducibility Guide

This guide describes how to reproduce **Phase 3A: Decision-State Construction and Representative State Selection**.

---

## 1. Environment & Setup

- **Python Version**: 3.10+ (Current project environment: Python 3.14)
- **Virtual Environment**:
  ```powershell
  C:\Users\Darsh Veer Singh\Documents\Virtual_Environments\mas_venv
  ```
- **Dependencies**:
  ```powershell
  & "C:\Users\Darsh Veer Singh\Documents\Virtual_Environments\mas_venv\Scripts\python.exe" -m pip install pandas pyarrow scikit-learn matplotlib
  ```

---

## 2. Input Dependencies

Phase 3A consumes the validated outputs of Phase 2:
- Canonical Machine State tables: `outputs/phase2/canonical_machine_states/*.parquet`
- Unified Semantic Feature Catalogue: `outputs/phase2/semantic_feature_catalogue/semantic_feature_catalogue.parquet`

---

## 3. Execution Commands

### A. Full Pipeline Run (Target Budget: ~1,100 Decision States)
```powershell
& "C:\Users\Darsh Veer Singh\Documents\Virtual_Environments\mas_venv\Scripts\python.exe" scripts/run_phase3a_decision_states.py --clean --budget 1100 --seed 42
```

### B. Custom Sampling Budget (e.g. 2,000 States for Scaled POC)
```powershell
& "C:\Users\Darsh Veer Singh\Documents\Virtual_Environments\mas_venv\Scripts\python.exe" scripts/run_phase3a_decision_states.py --clean --budget 2000 --seed 42
```

### C. Custom Distance Threshold for Stricter/Looser Pruning
```powershell
& "C:\Users\Darsh Veer Singh\Documents\Virtual_Environments\mas_venv\Scripts\python.exe" scripts/run_phase3a_decision_states.py --clean --budget 1100 --distance-threshold 0.15
```

---

## 4. Output Artifacts Layout

A complete run creates the following structure under `outputs/phase3a/`:

```
outputs/phase3a/
├── selected_decision_states/
│   ├── decision_states_corpus.parquet     # Master tabular Parquet representation of selected states
│   ├── decision_states_corpus.json        # Full JSON structured DecisionState objects
│   └── [dataset_id]_selected_states.json  # Per-dataset selected state subsets
├── llm_ready_states/
│   ├── state_cards_jsonl/
│   │   ├── decision_state_cards.jsonl                    # Clean inputs only (zero leakage)
│   │   └── decision_state_cards_with_eval_truth.jsonl    # With ground-truth context for offline eval
│   └── state_cards_md/
│       └── [decision_state_id].md         # Formatted Markdown Machine Health Cards (~500 tokens)
├── coverage_reports/
│   ├── dataset_severity_coverage.csv      # Dataset × Severity distribution matrix
│   ├── dataset_modality_coverage.csv      # Dataset × Modality presence matrix
│   └── dataset_weighting_summary.csv      # Weight, quota, and selection breakdown
├── validation_reports/
│   ├── phase3a_validation_report.json     # Automated assertion outcomes
│   └── phase3a_validation_report.md       # Human-readable safety check
├── visualizations/
│   ├── dataset_sample_distribution.png    # Selected states per dataset
│   ├── severity_distribution_by_dataset.png # Stacked severity breakdown
│   └── temporal_vs_static_split.png       # Temporal vs. static telemetry ratio
├── PHASE_3A_SUMMARY.md                    # Detailed executive audit summary
└── run_manifest.json                      # Reproducibility manifest (Git SHA, seeds, quotas)
```

---

## 5. Weighting and Quota Configuration

To balance datasets by feature complexity and prevent unmapped high-column datasets from dominating:
- **Weights** are defined in `phase3a/sampler.py` (`DATASET_WEIGHTS`):
  - Datasets with verified multi-modal dynamics and distinct failure modes (AI4I, C-MAPSS, Hydraulic) receive higher weights (~$1.10\text{--}1.15$).
  - Datasets with very high column counts and unmapped semantics (e.g. Tennessee Eastman with 139 columns) receive lower weights (~$0.70$).
  - Sum of quotas aligns with the user's target budget (~1,100 states).

---

## 6. How to Add a New Dataset into Phase 3A

1. **Verify Phase 2 Output**: Ensure the new dataset is normalized and written to `outputs/phase2/canonical_machine_states/<new_id>.parquet`.
2. **Add Harvester Method**: In `phase3a/harvester.py`, implement `harvest_<new_id>()` defining domain-grounded criteria for `HEALTHY`, `WATCH`, `DEGRADING`, and `CRITICAL`.
3. **Register in Harvester**: Add `<new_id>` to `harvest_funcs` in `CandidateHarvester.harvest_all()`.
4. **Assign Complexity Weight**: In `phase3a/sampler.py`, assign a weight in `DATASET_WEIGHTS`.
5. **Run & Verify**: Execute `python scripts/run_phase3a_decision_states.py --clean` and inspect `outputs/phase3a/validation_reports/phase3a_validation_report.md`.
