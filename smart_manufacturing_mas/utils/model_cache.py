"""
utils/model_cache.py
--------------------
Hash-keyed model persistence cache (Solution 1 from the presentation).

Cache key = SHA-256( dataset_basename + sorted_features + target + problem_type )

Benefits:
  - Same column set + same dataset  → instant load, no retraining.
  - Different column set            → fresh train.
  - Moving the project directory doesn't invalidate the cache (basename only).
  - Zero user-facing API change — completely transparent.

Usage:
    cache = ModelCache()

    # Before training — try to load
    entry = cache.load(dataset_path, feature_cols, target_col, problem_type)
    if entry:
        model    = entry['model']
        metadata = entry['metadata']
    else:
        model, metadata = train(...)
        cache.save(model, dataset_path, feature_cols, target_col, problem_type, metadata)
"""

import hashlib
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    import pickle as joblib  # type: ignore
    _HAS_JOBLIB = False
    logger.warning("joblib not found; falling back to pickle for model cache.")


class ModelCache:
    """
    Filesystem-backed, hash-keyed model cache.
    Thread-safety note: last-write-wins for concurrent saves with the same key.
    """

    _DEFAULT_CACHE_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model_cache",
    )

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir: str = cache_dir or self._DEFAULT_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self._registry_path = os.path.join(self.cache_dir, "registry.json")
        self._registry: Dict[str, Any] = self._load_registry()
        logger.info(f"ModelCache ready — dir='{self.cache_dir}', entries={len(self._registry)}")

    # ── Key ───────────────────────────────────────────────────────────────────

    def make_key(
        self,
        dataset_path: str,
        feature_columns: List[str],
        target_column: Optional[str],
        problem_type: str,
    ) -> str:
        """16-char hex SHA-256 key for (dataset, features, target, task)."""
        canonical = json.dumps(
            {
                "dataset":      os.path.basename(dataset_path),
                "features":     sorted(feature_columns),
                "target":       target_column,
                "problem_type": problem_type,
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def save(
        self,
        model: Any,
        dataset_path: str,
        feature_columns: List[str],
        target_column: Optional[str],
        problem_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        preprocessing_pipeline: Any = None,
    ) -> str:
        """Persist a trained model. Returns the cache key."""
        key = self.make_key(dataset_path, feature_columns, target_column, problem_type)
        model_path = os.path.join(self.cache_dir, f"{key}.pkl")

        payload = {
            "model": model,
            "preprocessing_pipeline": preprocessing_pipeline,
            "metadata": metadata or {},
            "config": {
                "dataset":      os.path.basename(dataset_path),
                "features":     sorted(feature_columns),
                "target":       target_column,
                "problem_type": problem_type,
            },
        }

        if _HAS_JOBLIB:
            joblib.dump(payload, model_path)
        else:
            import pickle
            with open(model_path, "wb") as fh:
                pickle.dump(payload, fh)

        self._registry[key] = {
            "path":     model_path,
            "config":   payload["config"],
            "metadata": metadata or {},
        }
        self._save_registry()
        logger.info(f"[ModelCache] SAVED  key={key} → {model_path}")
        return key

    def load(
        self,
        dataset_path: str,
        feature_columns: List[str],
        target_column: Optional[str],
        problem_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Load a cached model. Returns None on cache miss or load failure."""
        key = self.make_key(dataset_path, feature_columns, target_column, problem_type)
        model_path = os.path.join(self.cache_dir, f"{key}.pkl")

        if not os.path.exists(model_path):
            logger.info(f"[ModelCache] MISS   key={key}")
            return None

        try:
            if _HAS_JOBLIB:
                payload = joblib.load(model_path)
            else:
                import pickle
                with open(model_path, "rb") as fh:
                    payload = pickle.load(fh)
            logger.info(f"[ModelCache] HIT    key={key}")
            return payload
        except Exception as exc:
            logger.warning(f"[ModelCache] Load failed for key={key}: {exc}. Will retrain.")
            return None

    def invalidate(
        self,
        dataset_path: str,
        feature_columns: List[str],
        target_column: Optional[str],
        problem_type: str,
    ) -> bool:
        """Delete a specific cache entry. Returns True if something was removed."""
        key = self.make_key(dataset_path, feature_columns, target_column, problem_type)
        model_path = os.path.join(self.cache_dir, f"{key}.pkl")
        removed = False
        if os.path.exists(model_path):
            os.remove(model_path)
            removed = True
        if key in self._registry:
            del self._registry[key]
            self._save_registry()
            removed = True
        logger.info(f"[ModelCache] {'Invalidated' if removed else 'Nothing to invalidate for'} key={key}")
        return removed

    def clear(self) -> None:
        """Remove ALL cached models and reset the registry."""
        if os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self._registry = {}
        self._save_registry()
        logger.info("[ModelCache] Cache cleared.")

    def list_entries(self) -> Dict[str, Any]:
        """Return a copy of the full registry."""
        return dict(self._registry)

    def stats(self) -> Dict[str, Any]:
        total_size = sum(
            os.path.getsize(os.path.join(self.cache_dir, f))
            for f in os.listdir(self.cache_dir)
            if f.endswith(".pkl")
        )
        return {
            "entries":        len(self._registry),
            "cache_dir":      self.cache_dir,
            "total_size_mb":  round(total_size / (1024 ** 2), 2),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_registry(self) -> Dict[str, Any]:
        if os.path.exists(self._registry_path):
            try:
                with open(self._registry_path, "r") as fh:
                    return json.load(fh)
            except Exception:
                logger.warning("Registry file corrupted — starting fresh.")
        return {}

    def _save_registry(self) -> None:
        with open(self._registry_path, "w") as fh:
            json.dump(self._registry, fh, indent=2, default=str)
