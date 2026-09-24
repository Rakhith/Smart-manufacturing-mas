"""Target Definition and Transformation for Phase 4 Recommender.

Defines learning-to-rank targets for LGBMRanker LambdaRank based on Phase 3B
silver-standard action rankings and judgments.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class TargetConfig:
    strategy: str = "inverted_rank"  # 'inverted_rank' | 'composite_suitability'
    max_relevance: int = 4
    penalize_unsafe: bool = True
    penalize_inappropriate: bool = True
    min_relevance: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class TargetBuilder:
    """Computes ranking relevance labels from silver-standard evaluations."""

    def __init__(self, config: Optional[TargetConfig] = None):
        self.config = config or TargetConfig()

    def compute_relevance(self, row: pd.Series) -> int:
        rank = row.get("silver_rank")
        if rank is None or pd.isna(rank):
            return self.config.min_relevance

        try:
            rank_int = int(rank)
        except (ValueError, TypeError):
            return self.config.min_relevance

        # Base inverted rank: Rank 1 -> 4, Rank 2 -> 3, Rank 3 -> 2, Rank 4 -> 1, Rank 5 -> 0
        base_relevance = max(self.config.min_relevance, self.config.max_relevance - (rank_int - 1))

        if self.config.strategy == "inverted_rank":
            return int(base_relevance)

        elif self.config.strategy == "composite_suitability":
            verdict = str(row.get("silver_final_verdict", "")).upper()
            if self.config.penalize_unsafe and "UNSAFE" in verdict:
                return self.config.min_relevance
            if self.config.penalize_inappropriate and "INAPPROPRIATE" in verdict:
                return self.config.min_relevance

            # Suitability modulation: if suitability score is very low (< 30), floor to 0
            suit = row.get("silver_suitability_score")
            if suit is not None and not pd.isna(suit) and suit < 30:
                return self.config.min_relevance

            return int(base_relevance)

        else:
            raise ValueError(f"Unknown target strategy: {self.config.strategy}")

    def apply_to_dataframe(self, df: pd.DataFrame, target_col: str = "ranking_relevance") -> pd.DataFrame:
        df_out = df.copy()
        df_out[target_col] = df_out.apply(self.compute_relevance, axis=1)
        return df_out
