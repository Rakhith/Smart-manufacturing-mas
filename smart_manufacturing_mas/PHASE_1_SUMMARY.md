# Phase 1 Summary

Phase 1 implementation is complete as reusable code. It prepares the datasets independently and stops before semantic normalization.

## Implementation

The pipeline entry point is `scripts/run_phase1_preparation.py`. Dataset definitions and quality/catalogue helpers live in `phase1/catalogue.py`; causal temporal and signal features live in `phase1/features.py`; NASA Milling MATLAB extraction lives in `phase1/milling_extract.py`.

The default output root is `outputs/phase1/`, which is intentionally Git-ignored because the active corpus includes multi-gigabyte source-derived artifacts. A completed run writes inventory, Markdown and JSON catalogues, quality reports, preprocessing logs, processed tables, derived features, prepared observations, visualizations, a machine-readable manifest, and this summary generated from the actual run.

## Rerun

```bash
cd smart_manufacturing_mas
./mas_venv/bin/python -m pip install -r requirements.txt
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. ./mas_venv/bin/python scripts/run_phase1_preparation.py --clean
```

For development, process a subset:

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. ./mas_venv/bin/python scripts/run_phase1_preparation.py --clean --datasets ai4i_2020 metropt3
```

The actual record counts, labels, limitations, and statuses must be read from `outputs/phase1/PHASE_1_SUMMARY.md` and `outputs/phase1/run_manifest.json` after the run. This repository copy does not claim a completed corpus because the final multi-gigabyte run was intentionally deferred to avoid unnecessary resource use.

## Next phase boundary

Prepared observations retain dataset IDs, source/window context, asset and time/cycle identifiers where available, original feature names, derived features, labels, and provenance. They are ready to be mapped by a future semantic-normalization phase. No common ontology, severity level, action recommendation, ranker, collaborative filtering, LLM judge, or closed-loop system is included here.
