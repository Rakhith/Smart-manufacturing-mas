# Phase 1: Dataset Preparation

This package prepares each installed industrial dataset independently for the next semantic-normalization phase. It does not merge datasets or create a common machine-state schema.

## Scope

The runner performs:

- source inventory and file manifests with hashes where practical
- dataset-specific catalogues and sensor metadata
- quality checks for missing, duplicate, constant, infinite, and labelled values
- timestamp parsing, stable sorting, duplicate removal, and source-specific organization
- causal temporal features that use only current and prior observations
- per-cycle, per-window, trajectory-point, and signal-snapshot observations
- inspectable reports and non-semantic visualizations

The runner intentionally does **not** implement semantic normalization, severity mapping, action generation, ranking, collaborative filtering, LLM judging, or closed-loop simulation.

## Run later

From `smart_manufacturing_mas/`:

```bash
./mas_venv/bin/python -m pip install -r requirements.txt
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. ./mas_venv/bin/python scripts/run_phase1_preparation.py --clean
```

Use `--datasets DATASET_ID ...` for a targeted rerun. Valid IDs are listed in `phase1/catalogue.py`. Use `--output-dir PATH` to write artifacts outside the repository. Raw files under `data/` are never modified.

The runner prefers Parquet through `pyarrow`. If no Parquet engine is available, it falls back to CSV and updates catalogue/manifest paths to the files actually written. The three artifact directories use hard links when possible, so identical representations do not consume three times the disk space.

## Output layout

The default output is `outputs/phase1/`:

- `dataset_inventory/`: actual source files, counts, sizes, and formats
- `dataset_catalogue/`: one JSON and Markdown catalogue per dataset
- `data_quality_reports/`: before/after quality profiles
- `preprocessing_reports/`: explicit operation logs and record counts
- `processed_datasets/`: cleaned dataset-specific tables
- `derived_features/`: source-specific engineered features
- `prepared_observations/`: pre-semantic observation units with provenance
- `visualizations/`: representative quality, trajectory, label, and signal plots
- `run_manifest.json`: machine-readable run status and output paths
- `PHASE_1_SUMMARY.md`: human-readable cross-dataset summary

Generated outputs are ignored by Git because several installed datasets produce multi-gigabyte artifacts. Review them locally after running the pipeline; commit code, reports, or small samples separately when needed.

## Dataset-specific handling

- AI4I and Metal Etch remain row-level tabular observations.
- Smart Manufacturing preserves its static and timestamped variants separately.
- C-MAPSS retains unit/cycle trajectories and derives causal RUL labels.
- IIoT 6G receives timestamp sorting and causal temporal features.
- NASA Milling extracts conservative signal features from the MATLAB archive without unsupported physical mappings.
- MetroPT-3 is reduced to one-minute windows using vectorized resampling.
- UCI Hydraulic Systems becomes one observation per synchronized load cycle with multirate signal summaries.
- Tennessee Eastman reads RData objects, retains run/sample context, and leaves undocumented variable meanings explicitly unverified.
- NASA IMS converts each vibration snapshot into time/frequency-domain features and keeps end-of-test failure descriptions as non-row-level context.
- The deprecated Digital Manufacturing file is inventoried and catalogued but not included in the active observation corpus.

## Review order

1. `outputs/phase1/dataset_inventory/inventory.json`
2. `outputs/phase1/PHASE_1_SUMMARY.md`
3. `outputs/phase1/run_manifest.json`
4. a dataset catalogue JSON/Markdown pair
5. its quality and preprocessing reports
6. its prepared observation artifact and visualization
