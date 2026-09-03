"""Extract modest, inspectable NASA milling observations from mill.mat."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from phase1.features import waveform_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = args.output / "_milling_source"
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        archive.extract("mill.mat", work)
    raw = loadmat(work / "mill.mat", squeeze_me=True, struct_as_record=False)
    public = {key: value for key, value in raw.items() if not key.startswith("__")}
    rows: list[dict[str, object]] = []
    manifest: dict[str, object] = {"mat_variables": {key: list(np.shape(value)) for key, value in public.items()}, "notes": []}
    # The NASA release is a nested MATLAB structure.  Flatten numeric arrays into
    # per-run observations without assigning unsupported physical semantics.
    def visit(value: object, path: str) -> None:
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            if value.ndim == 1 and value.size >= 32:
                item = {"source_array": path, "sample_count": int(value.size), **{f"signal__{k}": v for k, v in waveform_features(value, 1.0).items()}}
                rows.append(item)
            elif value.ndim == 2 and value.shape[0] >= 1 and value.shape[1] >= 32:
                for index, sample in enumerate(value[:5000]):
                    item = {"source_array": path, "row_index": index, "sample_count": int(sample.size), **{f"signal__{k}": v for k, v in waveform_features(sample, 1.0).items()}}
                    rows.append(item)
            return
        if isinstance(value, np.ndarray) and value.dtype == object:
            for index, child in np.ndenumerate(value):
                visit(child, f"{path}[{','.join(map(str, index))}]")
        elif hasattr(value, "_fieldnames"):
            for field in value._fieldnames:
                visit(getattr(value, field), f"{path}.{field}")
    for key, value in public.items():
        visit(value, key)
    output = pd.DataFrame(rows)
    if output.empty:
        manifest["notes"].append("No numeric signal arrays could be flattened automatically; retained MAT-variable inventory only.")
    # The project virtual environment supplies SciPy for MAT parsing but not a
    # Parquet engine.  Hand the compact intermediary to the main runner as CSV;
    # it converts it to the standard Phase 1 Parquet artifacts.
    output.to_csv(args.output / "nasa_milling_prepared_observations.csv", index=False)
    (args.output / "nasa_milling_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
