# Phase 2 Reproducibility Guide

This guide describes how to reproduce the **Phase 2: Semantic Normalization and Canonical Machine State Representation (CMSR)** pipeline.

---

## 1. Environment & Setup

- **Python Version**: 3.10+ (Current project environment: Python 3.14)
- **Virtual Environment Location**:
  ```powershell
  C:\Users\Darsh Veer Singh\Documents\Virtual_Environments\mas_venv
  ```
- **Dependencies**:
  ```powershell
  & "C:\Users\Darsh Veer Singh\Documents\Virtual_Environments\mas_venv\Scripts\python.exe" -m pip install pandas pyarrow pyyaml matplotlib
  ```

---

## 2. Input Dependencies

Phase 2 consumes the prepared Parquet datasets produced during Phase 1:
- Location: `processed_datasets/` (or `outputs/phase1/processed_datasets/`)
- Active input datasets:
  - `ai4i_2020.parquet`
  - `cmapss.parquet`
  - `smart_maintenance_static.parquet`
  - `smart_maintenance_timeseries.parquet`
  - `iiot_6g.parquet`
  - `metal_etch.parquet`
  - `metropt3_minute_windows.parquet`
  - `uci_hydraulic.parquet`
  - `tennessee_eastman.parquet`
  - `nasa_ims.parquet`
  - `nasa_milling.parquet`

---

## 3. Execution Commands

### A. Full Pipeline Rerun (Clean)
```powershell
& "C:\Users\Darsh Veer Singh\Documents\Virtual_Environments\mas_venv\Scripts\python.exe" scripts/run_phase2_normalization.py --clean
```

### B. Targeted Execution on Specific Datasets
```powershell
& "C:\Users\Darsh Veer Singh\Documents\Virtual_Environments\mas_venv\Scripts\python.exe" scripts/run_phase2_normalization.py --clean --datasets ai4i_2020 uci_hydraulic
```

### C. Custom Output Directory
```powershell
& "C:\Users\Darsh Veer Singh\Documents\Virtual_Environments\mas_venv\Scripts\python.exe" scripts/run_phase2_normalization.py --output-dir custom_outputs/phase2
```

---

## 4. Output Structure

A complete run writes the following artifacts under `outputs/phase2/`:
```
outputs/phase2/
├── dataset_mappings/                  # YAML mapping specifications for auditability
│   ├── ai4i_2020.yaml
│   ├── cmapss.yaml
│   └── ...
├── semantic_feature_catalogue/        # Global & per-dataset semantic feature catalogs
│   ├── semantic_feature_catalogue.json
│   ├── semantic_feature_catalogue.parquet
│   └── [dataset_id]_catalogue.json
├── canonical_machine_states/          # Dataset-partitioned Parquet tables with CMSR schemas
│   ├── ai4i_2020.parquet
│   ├── cmapss.parquet
│   └── ...
├── validation_reports/                # Automated audit reports and assertion logs
│   ├── semantic_validation_report.json
│   └── semantic_validation_report.md
├── visualizations/                    # Modality coverage matrix, confidence distributions
│   ├── dataset_modality_matrix.png
│   ├── confidence_distribution.png
│   └── feature_modality_distribution.png
├── PHASE_2_SUMMARY.md                 # Human-readable executive summary
└── run_manifest.json                  # Machine-readable provenance manifest
```

---

## 5. How to Add Mappings for a New Dataset

To onboard a new industrial dataset into the Phase 2 CMSR pipeline:

1. **Verify Phase 1 Prepared Observation**: Ensure the dataset is prepared as a Parquet table under `processed_datasets/<new_dataset_id>.parquet`.
2. **Create a Declarative Mapping File**:
   Create `phase2/dataset_mappings/<new_dataset_id>.yaml`:
   ```yaml
   dataset_id: "new_dataset_id"
   asset_context: "Brief description of the physical equipment"
   machine_archetype: "archetype_name" # e.g. pump, cnc_mill, compressor
   provenance: "Dataset source"

   features:
     raw_sensor_name:
       modality: "thermal"             # thermal | mechanical | fluid | electrical | ...
       physical_quantity: "temperature"
       role: "process_sensor"
       unit: "C"
       canonical_unit: "degC"
       conversion_rule: "x"            # or linear formula like "x - 273.15"
       confidence: "high"              # high | medium | low | unknown
       is_target: false                # true if it is a failure label or future outcome

     target_failure_flag:
       modality: "outcome_label"
       physical_quantity: "unknown"
       role: "target_label"
       unit: "binary"
       canonical_unit: "binary"
       confidence: "high"
       is_target: true
   ```
3. **Register in Runner**: Add the key-value mapping to `DATASET_FILE_MAP` in `scripts/run_phase2_normalization.py`.
4. **Execute Normalization**: Run `python scripts/run_phase2_normalization.py --datasets <new_dataset_id>`.
5. **Verify**: Check `outputs/phase2/validation_reports/semantic_validation_report.md` to confirm 100% feature coverage and target isolation.
