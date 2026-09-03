# Phase 1 Reproducibility

## 1. Environment

Use Python 3.13 or newer with the project environment. The installed corpus is large, so allow several GB of free disk space. Raw data is read from `data/` and never modified.

## 2. Install

From `smart_manufacturing_mas/`:

```bash
./mas_venv/bin/python -m pip install -r requirements.txt
```

The Phase 1 reader dependencies include `pyreadr` for Tennessee Eastman RData and `pyarrow` for compact Parquet output.

## 3. Run all available datasets

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. ./mas_venv/bin/python scripts/run_phase1_preparation.py --clean --output-dir outputs/phase1
```

To regenerate only the audit manifest and summary from completed artifacts:

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. ./mas_venv/bin/python scripts/run_phase1_preparation.py --resume --output-dir outputs/phase1
```

## 4. Run selected datasets

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. ./mas_venv/bin/python scripts/run_phase1_preparation.py --clean --datasets ai4i_2020 cmapss metropt3 --output-dir outputs/phase1
```

Valid dataset IDs are listed by `scripts/run_phase1_preparation.py --help` and defined in `phase1/catalogue.py`.

## 5. Output structure

`outputs/phase1/` contains inventory, JSON/Markdown catalogues, quality reports, preprocessing logs, processed datasets, derived features, prepared observations, visualizations, `run_manifest.json`, and `PHASE_1_SUMMARY.md`.

The three tabular artifact directories may use safe symlinks when they contain the same representation, reducing disk usage without removing any inspectable path.

## 6. Verify a successful run

1. Open `outputs/phase1/run_manifest.json` and confirm `errors` is empty.
2. Confirm every active dataset has `status: completed` and an existing prepared output.
3. Confirm `outputs/phase1/PHASE_1_SUMMARY.md` reports SUCCESS or SUCCESS_WITH_LIMITATIONS.
4. Compare catalogue `record_count` and manifest `prepared_records` with the quality and preprocessing reports.
5. Inspect a representative prepared artifact and its visualization. Prepared rows retain `dataset_id`, source context, labels, original features, derived features, and provenance.

## 7. Add another dataset

Add a `DatasetDefinition` in `phase1/catalogue.py`, add supported physical metadata, implement a dedicated processor in `scripts/run_phase1_preparation.py`, register it in `processors`, and add a focused representative plot/report. Keep the raw source under `data/`, preserve unknown meanings as `UNKNOWN/UNVERIFIED`, and run the selected-dataset command before including it in the all-datasets run.
