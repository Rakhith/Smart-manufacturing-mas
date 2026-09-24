"""LGBMRanker Learner for Phase 4 Recommender.

Wraps LightGBM LGBMRanker with LambdaRank objective, grouped training, early stopping,
conservative hyperparameters, feature importance extraction, and model persistence.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass
class RankerHyperparameters:
    n_estimators: int = 50
    learning_rate: float = 0.03
    num_leaves: int = 15
    max_depth: int = 4
    min_child_samples: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    importance_type: str = "gain"
    early_stopping_rounds: int = 0  # 0 to train full conservative n_estimators

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MaintenanceLGBMRanker:
    """Primary learned prescriptive maintenance recommender using LightGBM LambdaRank."""

    def __init__(self, params: Optional[RankerHyperparameters] = None):
        self.params = params or RankerHyperparameters()
        self.model: Optional[lgb.LGBMRanker] = None
        self.feature_names: List[str] = []
        self.best_iteration_: Optional[int] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        groups_train: List[int],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        groups_val: Optional[List[int]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> MaintenanceLGBMRanker:
        """Trains the LGBMRanker with grouped instances and optional early stopping."""
        self.feature_names = feature_names or [f"f_{i}" for i in range(X_train.shape[1])]

        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=self.params.n_estimators,
            learning_rate=self.params.learning_rate,
            num_leaves=self.params.num_leaves,
            max_depth=self.params.max_depth,
            min_child_samples=self.params.min_child_samples,
            subsample=self.params.subsample,
            colsample_bytree=self.params.colsample_bytree,
            random_state=self.params.random_state,
            importance_type=self.params.importance_type,
            n_jobs=-1,
            verbose=-1,
        )

        callbacks = []
        eval_set = None
        eval_group = None

        if self.params.early_stopping_rounds > 0 and X_val is not None and y_val is not None and groups_val is not None:
            eval_set = [(X_val, y_val)]
            eval_group = [groups_val]
            callbacks.append(lgb.early_stopping(stopping_rounds=self.params.early_stopping_rounds, verbose=False))

        self.model.fit(
            X=X_train,
            y=y_train,
            group=groups_train,
            eval_set=eval_set,
            eval_group=eval_group,
            feature_name=self.feature_names,
            callbacks=callbacks if callbacks else None,
        )

        self.best_iteration_ = getattr(self.model, "best_iteration_", None)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts ranking scores for candidate actions."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Returns sorted feature importance by gain and split."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        gain_imp = self.model.booster_.feature_importance(importance_type="gain")
        split_imp = self.model.booster_.feature_importance(importance_type="split")

        records = {}
        for name, g, s in zip(self.feature_names, gain_imp, split_imp):
            records[name] = {"gain": float(g), "split": int(s)}

        sorted_records = dict(sorted(records.items(), key=lambda item: item[1]["gain"], reverse=True))
        return sorted_records

    def save(self, model_dir: Path) -> None:
        """Serializes model, hyperparameters, and feature metadata."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # 1. Joblib binary
        joblib.dump(self.model, model_dir / "lgbm_ranker.joblib")

        # 2. Metadata json
        meta = {
            "hyperparameters": self.params.to_dict(),
            "feature_names": self.feature_names,
            "best_iteration": self.best_iteration_,
            "feature_importance_top20": list(self.get_feature_importance().items())[:20],
        }
        with open(model_dir / "lgbm_ranker_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, model_dir: Path) -> MaintenanceLGBMRanker:
        """Loads serialized model and metadata."""
        model_dir = Path(model_dir)
        with open(model_dir / "lgbm_ranker_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        params = RankerHyperparameters(**meta.get("hyperparameters", {}))
        ranker = cls(params=params)
        ranker.model = joblib.load(model_dir / "lgbm_ranker.joblib")
        ranker.feature_names = meta.get("feature_names", [])
        ranker.best_iteration_ = meta.get("best_iteration")
        return ranker
