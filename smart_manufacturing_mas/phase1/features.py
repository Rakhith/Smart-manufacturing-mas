"""Physical, dataset-specific feature extraction utilities.

The names emitted here deliberately remain dataset/domain specific.  This is
pre-semantic preparation, not a common feature ontology.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return np.nan
    x = np.arange(values.size, dtype=float)[valid]
    return float(np.polyfit(x, values[valid], 1)[0])


def waveform_features(values: np.ndarray, sampling_hz: float) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: np.nan for key in ("mean", "rms", "variance", "std", "skewness", "kurtosis", "crest_factor", "dominant_frequency_hz", "spectral_energy", "spectral_entropy")}
    mean = float(values.mean())
    centered = values - mean
    std = float(centered.std())
    rms = float(np.sqrt(np.mean(values ** 2)))
    skewness = float(np.mean((centered / std) ** 3)) if std else 0.0
    kurtosis = float(np.mean((centered / std) ** 4) - 3) if std else 0.0
    crest = float(np.max(np.abs(values)) / rms) if rms else np.nan
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(values.size, d=1 / sampling_hz)
    if spectrum.size > 1:
        dominant = float(frequencies[1 + np.argmax(spectrum[1:])])
    else:
        dominant = np.nan
    spectral_energy = float(spectrum.sum() / max(values.size, 1))
    probability = spectrum / spectrum.sum() if spectrum.sum() > 0 else np.zeros_like(spectrum)
    spectral_entropy = float(-np.sum(probability[probability > 0] * np.log2(probability[probability > 0]))) if probability.size else np.nan
    return {
        "mean": mean, "rms": rms, "variance": float(values.var()), "std": std, "skewness": skewness,
        "kurtosis": kurtosis, "crest_factor": crest, "dominant_frequency_hz": dominant,
        "spectral_energy": spectral_energy, "spectral_entropy": spectral_entropy,
    }


def matrix_cycle_features(matrix: np.ndarray, prefix: str) -> pd.DataFrame:
    matrix = np.asarray(matrix, dtype=float)
    n_rows = matrix.shape[0]
    x = np.arange(matrix.shape[1], dtype=float)
    centered_x = x - x.mean()
    denominator = float(np.sum(centered_x**2))
    row_mean = np.nanmean(matrix, axis=1)
    slopes = np.nansum((matrix - row_mean[:, None]) * centered_x, axis=1) / denominator
    return pd.DataFrame({
        f"{prefix}__mean": row_mean, f"{prefix}__std": np.nanstd(matrix, axis=1),
        f"{prefix}__min": np.nanmin(matrix, axis=1), f"{prefix}__max": np.nanmax(matrix, axis=1),
        f"{prefix}__first": matrix[:, 0], f"{prefix}__last": matrix[:, -1], f"{prefix}__slope_per_sample": slopes,
    }, index=pd.RangeIndex(n_rows, name="cycle_index"))


def add_causal_temporal_features(df: pd.DataFrame, group_column: str | None, numeric_columns: list[str], window: int = 10) -> pd.DataFrame:
    """Append causal rolling summaries; no future rows are used."""
    output = df.copy()
    groups = output.groupby(group_column, sort=False, dropna=False) if group_column and group_column in output else [(None, output)]
    for _, group in groups:
        idx = group.index
        for column in numeric_columns:
            series = pd.to_numeric(group[column], errors="coerce")
            output.loc[idx, f"{column}__rolling_mean_{window}"] = series.rolling(window, min_periods=1).mean().to_numpy()
            output.loc[idx, f"{column}__rolling_std_{window}"] = series.rolling(window, min_periods=2).std().to_numpy()
            output.loc[idx, f"{column}__delta_1"] = series.diff().to_numpy()
            # Least-squares slope can be computed from rolling sums.  This is
            # equivalent to fitting y~time in each causal window but avoids a
            # Python callback for every sample (critical for C-MAPSS/TEP).
            x = pd.Series(np.arange(len(series), dtype=float), index=series.index)
            valid = series.notna()
            n = valid.astype(float).rolling(window, min_periods=1).sum()
            sx = x.where(valid).rolling(window, min_periods=1).sum()
            sy = series.rolling(window, min_periods=1).sum()
            sxx = (x * x).where(valid).rolling(window, min_periods=1).sum()
            sxy = (x * series).rolling(window, min_periods=1).sum()
            denominator = n * sxx - sx * sx
            rolling_slope = (n * sxy - sx * sy) / denominator.where((n >= 2) & (denominator != 0))
            output.loc[idx, f"{column}__slope_{window}"] = rolling_slope.to_numpy()
    return output
