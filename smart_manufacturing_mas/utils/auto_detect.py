"""
utils/auto_detect.py
--------------------
Auto-detection of ML problem type from dataset characteristics.

Implements the "Auto-Detect Problem Type" improvement from the Proposed Next Architecture.
Eliminates the need for the user to know ML terminology when using --mode rules-first.

Detection logic:
  No target column specified       → anomaly_detection  (confidence 0.95)
  String / category dtype          → classification     (confidence 0.97)
  Boolean dtype                    → classification     (confidence 0.99)
  Integer, ≤ threshold unique vals → classification     (confidence 0.82–0.95)
  Integer, > threshold unique vals → regression         (confidence 0.75)
  Float, very few unique vals (≤5) → classification     (confidence 0.70)
  Float, many unique vals          → regression         (confidence 0.78–0.92)
  Unrecognised dtype               → classification     (confidence 0.50)

Returns (problem_type, confidence, reasoning_dict).
HITL confirmation is triggered when confidence < CONFIDENCE_THRESHOLD_FOR_HITL.
"""

import logging
from typing import Dict, Any, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CLASSIFICATION_THRESHOLD = 20   # max unique values in int column → classification
CONFIDENCE_THRESHOLD_FOR_HITL = 0.75    # below this → ask operator to confirm


def auto_detect_problem_type(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    classification_threshold: int = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Infer the ML problem type from the target column's statistics.

    Args:
        df: Full or sample DataFrame.
        target_col: Intended target column name. Pass None → anomaly_detection.
        classification_threshold: Max unique integer values before → regression.

    Returns:
        (problem_type, confidence, reasoning)
    """
    # ── No target column ────────────────────────────────────────────────────
    if target_col is None or target_col not in df.columns:
        return (
            "anomaly_detection",
            0.95,
            {
                "reason": "No target column specified — treating as unsupervised anomaly detection.",
                "evidence": [f"Available columns: {list(df.columns)}"],
            },
        )

    col = df[target_col].dropna()
    dtype_str = str(col.dtype)
    n_unique = int(col.nunique())
    n_rows = len(col)
    evidence = [f"dtype={dtype_str}", f"n_unique={n_unique}", f"n_rows={n_rows}"]

    # ── String / category ───────────────────────────────────────────────────
    if dtype_str in ("object", "category"):
        return (
            "classification",
            0.97,
            {"reason": f'Target "{target_col}" has string/category dtype → classification.', "evidence": evidence},
        )

    # ── Boolean ─────────────────────────────────────────────────────────────
    if dtype_str == "bool":
        return (
            "classification",
            0.99,
            {"reason": f'Target "{target_col}" is boolean → classification.', "evidence": evidence},
        )

    # ── Integer ─────────────────────────────────────────────────────────────
    if dtype_str in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
        if n_unique <= classification_threshold:
            confidence = 0.95 if n_unique <= 10 else 0.82
            return (
                "classification",
                confidence,
                {
                    "reason": (
                        f'Integer target with {n_unique} unique value(s) '
                        f"(≤ threshold {classification_threshold}) → classification."
                    ),
                    "evidence": evidence,
                },
            )
        else:
            evidence.append(f"Integer but {n_unique} unique values exceeds threshold {classification_threshold}.")
            return (
                "regression",
                0.75,
                {
                    "reason": (
                        f'Integer target with {n_unique} unique values '
                        f"(> {classification_threshold}) — likely ordinal/count → regression."
                    ),
                    "evidence": evidence,
                },
            )

    # ── Float ────────────────────────────────────────────────────────────────
    if dtype_str in ("float16", "float32", "float64"):
        if n_unique <= 5:
            evidence.append(f"Float dtype but only {n_unique} unique values — may be encoded class labels.")
            return (
                "classification",
                0.70,
                {
                    "reason": f'Float target with very few unique values ({n_unique}) — possibly encoded classification labels.',
                    "evidence": evidence,
                },
            )
        uniqueness_ratio = n_unique / n_rows if n_rows > 0 else 0
        confidence = 0.92 if uniqueness_ratio > 0.10 else 0.78
        evidence.append(f"uniqueness_ratio={uniqueness_ratio:.3f}")
        return (
            "regression",
            confidence,
            {
                "reason": (
                    f'Continuous float target ({n_unique} unique values, '
                    f"{uniqueness_ratio:.1%} uniqueness ratio) → regression."
                ),
                "evidence": evidence,
            },
        )

    # ── Fallback ─────────────────────────────────────────────────────────────
    logger.warning(
        f"Could not confidently determine problem type for dtype={dtype_str}. "
        "Defaulting to classification with low confidence."
    )
    return (
        "classification",
        0.50,
        {
            "reason": f"Unrecognised dtype '{dtype_str}' — defaulting to classification with low confidence.",
            "evidence": evidence,
        },
    )


def suggest_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristically suggest a target column when none is specified.

    Priority:
      1. Column whose name contains a common target-like keyword.
      2. Last column in the DataFrame (common ML convention).
    """
    if df.empty or len(df.columns) == 0:
        return None

    def score_column(name: str) -> int:
        n = name.lower()
        score = 0

        # Strong target markers (highest priority)
        strong = ("target", "label", "class", "status", "outcome")
        if any(tok in n for tok in strong):
            score += 100

        # Common predictive-task markers
        medium = ("priority", "failure", "fault", "anomaly", "efficiency")
        if any(tok in n for tok in medium):
            score += 45

        # Score-like columns can be targets, but are often intermediate features
        if "score" in n:
            score += 15

        # Penalize very feature-like names when no stronger marker exists
        feature_like = ("latency", "loss", "rate", "temp", "temperature", "pressure", "vibration")
        if any(tok in n for tok in feature_like):
            score -= 20

        # Slight preference for right-most columns (common dataset convention)
        return score

    ranked = sorted(df.columns, key=lambda c: score_column(c), reverse=True)
    if ranked and score_column(ranked[0]) > 0:
        best = ranked[0]
        logger.info(f"[AutoDetect] Suggested target column by ranked keyword score: '{best}'")
        return best

    last = str(df.columns[-1])
    logger.info(f"[AutoDetect] No keyword match — suggesting last column: '{last}'")
    return last


def needs_hitl_confirmation(confidence: float) -> bool:
    """Return True when auto-detection confidence is low enough to warrant operator confirmation."""
    return confidence < CONFIDENCE_THRESHOLD_FOR_HITL


def log_detection_result(problem_type: str, confidence: float, reasoning: Dict[str, Any]) -> None:
    """Log the auto-detection result in a structured, human-readable format."""
    level = logging.INFO if confidence >= CONFIDENCE_THRESHOLD_FOR_HITL else logging.WARNING
    logger.log(
        level,
        f"[AutoDetect] problem_type='{problem_type}', confidence={confidence:.0%}\n"
        f"  Reason  : {reasoning['reason']}\n"
        f"  Evidence: {', '.join(reasoning['evidence'])}",
    )
