"""
agents/preprocessing_agent.py
------------------------------
PreprocessingAgent — cleans, feature-engineers, and builds a reproducible sklearn pipeline.

Improvements over baseline:
  ✦ Optional PCA dimensionality reduction (use_pca=False by default).
    WARNING: enabling PCA replaces named features with anonymous PC_0, PC_1, …
    and breaks per-feature recommendations ("Vibration is the key driver").
    Only enable when dimensionality matters more than interpretability.

Architecture note (Three-Tier Intelligence Hierarchy):
  - Strategy selection: Rule-Based ToolDecider (TIER 3), NOT SLM.
  - SLM 2 (Preprocessing Strategy): ELIMINATED — deterministic if-else rules.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tool_decider import ToolDecider, create_data_summary, get_tool_decider
from utils.intelligent_feature_analysis import IntelligentFeatureAnalyzer

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')


class PreprocessingAgent:
    """
    Responsible for cleaning and preparing the dataset.

    Args:
        data                   : Raw DataFrame (target column may be included for feature analysis).
        tool_decider           : Strategy selector (defaults to rule-based).
        target_column          : Target column name (for feature-importance analysis + leakage prevention).
        problem_type           : 'classification' | 'regression' | 'anomaly_detection'.
        protected_columns      : Columns that must NOT be automatically dropped.
        use_pca                : Enable optional PCA after main pipeline (default: False).
        pca_variance_threshold : Fraction of variance to retain when use_pca=True (default: 0.95).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tool_decider: Optional[ToolDecider] = None,
        target_column: Optional[str] = None,
        problem_type: Optional[str] = None,
        protected_columns: Optional[List[str]] = None,
        use_pca: bool = False,
        pca_variance_threshold: float = 0.95,
    ):
        logging.info("Initializing Preprocessing Agent...")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        self.data = data.copy()
        self.tool_decider = tool_decider or get_tool_decider("rule_based")
        self.target_column = target_column
        self.problem_type = problem_type
        self.protected_columns = protected_columns or []

        self.use_pca = use_pca
        self.pca_variance_threshold = pca_variance_threshold
        if use_pca:
            logging.warning(
                "[PCA] PCA is ENABLED. Named feature columns will be replaced with "
                "anonymous principal components (PC_0, PC_1, …). "
                "Per-feature recommendations will NOT be available for this run."
            )

        self.feature_analyzer: Optional[IntelligentFeatureAnalyzer] = None
        self.last_feature_insights: Dict[str, Any] = {}
        self.feature_insights: Dict[str, Any] = {}
        self.fitted_pca: Optional[PCA] = None  # Exposed for downstream inspection
        self.identifier_columns: List[str] = []

        logging.info(f"Preprocessing Agent initialised — shape={self.data.shape}, use_pca={use_pca}")

    # ── Feature type detection ────────────────────────────────────────────────

    def get_feature_types(self):
        """Identify numerical and categorical features."""
        numerical_features = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

        if self.problem_type == 'anomaly_detection':
            kept_identifiers: List[str] = []
            protected_set = set(self.protected_columns)
            for feature in list(numerical_features):
                upper = feature.upper()
                if ('ID' in upper or upper.endswith('_ID') or upper == 'ID') and feature in protected_set:
                    kept_identifiers.append(feature)
                    numerical_features.remove(feature)

            if kept_identifiers:
                logging.info(f"Protected identifier columns for anomaly detection: {kept_identifiers}")
                passthrough_df = self.data[kept_identifiers].copy()
                passthrough_df.columns = [f"identifier__{col}" for col in kept_identifiers]
                self.data = pd.concat([self.data, passthrough_df], axis=1)
                self.identifier_columns = [f"identifier__{col}" for col in kept_identifiers]
            else:
                self.identifier_columns = []
        else:
            self.identifier_columns = []

        logging.info(f"Numerical features : {numerical_features}")
        logging.info(f"Categorical features: {categorical_features}")
        return numerical_features, categorical_features

    # ── Feature analysis ──────────────────────────────────────────────────────

    def perform_intelligent_feature_analysis(self) -> Dict[str, Any]:
        """Run correlation / mutual-information analysis when target is available."""
        if (
            not self.target_column
            or not self.problem_type
            or self.target_column not in self.data.columns
        ):
            logging.info("Skipping intelligent feature analysis — target column or problem type not available.")
            return {}

        logging.info("Performing intelligent feature analysis...")
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Drop non-numeric columns before feature analysis:
        # - Identifier columns (ID, Machine_ID, etc.)
        # - Timestamp columns (Timestamp, DateTime, etc.)
        # - Categorical object/string columns (Operation_Mode, Status, etc.)
        id_cols = [col for col in X.columns if 'ID' in col.upper() and col not in self.protected_columns]
        timestamp_cols = [col for col in X.columns if col.lower() in ('timestamp', 'datetime', 'date', 'time')]
        categorical_cols = [col for col in X.columns if X[col].dtype in ('object', 'str', 'string', 'category')]
        cols_to_drop = list(set(id_cols + timestamp_cols + categorical_cols))
        
        if cols_to_drop:
            logging.info(f"Dropping non-numeric columns for feature analysis: {sorted(cols_to_drop)}")
            X = X.drop(columns=cols_to_drop)

        self.feature_analyzer = IntelligentFeatureAnalyzer(self.target_column, self.problem_type)
        self.feature_insights = self.feature_analyzer.analyze_features(X, y)

        logging.info("Feature analysis completed.")
        logging.info(f"Analysis summary:\n{self.feature_insights['summary']}")
        return self.feature_insights

    # ── Pipeline construction ─────────────────────────────────────────────────

    def create_preprocessing_pipeline(
        self, numerical_features: List[str], categorical_features: List[str]
    ) -> ColumnTransformer:
        """Build a ColumnTransformer from rule-based ToolDecider decisions."""
        logging.info("Creating preprocessing pipeline...")

        data_summary = create_data_summary(self.data)
        available_tools = ["imputation", "scaling", "encoding", "normalization"]
        decision = self.tool_decider.decide_preprocessing_strategy(data_summary, available_tools)
        logging.info(f"ToolDecider strategy: {decision}")

        # ── Numerical transformer ─────────────────────────────────────────
        numerical_steps: list = []
        if "imputation" in decision.get("tools", []):
            if data_summary["missing_percentage"] > 20:
                numerical_steps.append(('imputer', KNNImputer(n_neighbors=3)))
                logging.info("Using KNNImputer (high missing %).")
            else:
                numerical_steps.append(('imputer', SimpleImputer(strategy='median')))
                logging.info("Using SimpleImputer(median).")

        if "scaling" in decision.get("tools", []):
            if data_summary["memory_usage_mb"] > 100:
                numerical_steps.append(('scaler', RobustScaler()))
                logging.info("Using RobustScaler (large dataset).")
            else:
                numerical_steps.append(('scaler', StandardScaler()))
                logging.info("Using StandardScaler.")
        elif "normalization" in decision.get("tools", []):
            numerical_steps.append(('scaler', MinMaxScaler()))
            logging.info("Using MinMaxScaler.")

        # ── Categorical transformer ───────────────────────────────────────
        categorical_steps: list = []
        low_cardinality_features: List[str] = []
        high_cardinality_features: List[str] = []

        if "encoding" in decision.get("tools", []) and categorical_features:
            categorical_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
            for feature in categorical_features:
                unique_count = self.data[feature].nunique()
                if unique_count > 50:
                    high_cardinality_features.append(feature)
                    logging.warning(
                        f"High cardinality feature '{feature}' ({unique_count} unique) — will be dropped."
                    )
                else:
                    low_cardinality_features.append(feature)
                    logging.info(f"Low cardinality feature '{feature}' ({unique_count} unique) — safe for OHE.")

            if low_cardinality_features:
                categorical_steps.append(
                    ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=50))
                )
            else:
                logging.warning("No categorical features suitable for encoding (all high-cardinality).")

        # ── Assemble ─────────────────────────────────────────────────────
        numerical_pipeline = Pipeline(steps=numerical_steps) if numerical_steps else None
        categorical_pipeline = Pipeline(steps=categorical_steps) if categorical_steps else None

        transformers = []
        if numerical_pipeline and numerical_features:
            transformers.append(('num', numerical_pipeline, numerical_features))
        if categorical_pipeline and low_cardinality_features:
            transformers.append(('cat', categorical_pipeline, low_cardinality_features))

        # Use remainder='drop' to exclude high-cardinality and other non-selected columns
        # (e.g., timestamp, high-cardinality categorical features)
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        logging.info(f"Preprocessing pipeline created with {len(transformers)} transformer(s).")
        return preprocessor

    # ── Main entry point ──────────────────────────────────────────────────────

    def preprocess(self) -> Optional[pd.DataFrame]:
        """Run the full preprocessing pipeline. Returns preprocessed DataFrame or None on failure."""
        logging.info("Starting data preprocessing...")
        try:
            feature_insights = self.perform_intelligent_feature_analysis()
            self.last_feature_insights = feature_insights

            numerical_features, categorical_features = self.get_feature_types()

            # ── Determine columns to drop ─────────────────────────────────
            cols_to_drop = [
                col for col in self.data.columns
                if 'ID' in col.upper() and col not in self.protected_columns
            ]

            # Drop timestamp columns (they're not useful for ML models)
            for col in self.data.columns:
                if col.lower() in ('timestamp', 'datetime', 'date', 'time') and col not in self.protected_columns:
                    cols_to_drop.append(col)
                    logging.info(f"Dropping timestamp column '{col}'.")

            for feature in categorical_features:
                if feature not in self.protected_columns:
                    if self.data[feature].nunique() > 50 and not feature.startswith("identifier__"):
                        cols_to_drop.append(feature)
                        logging.warning(f"Dropping high-cardinality feature '{feature}'.")

            if feature_insights and 'recommendations' in feature_insights:
                for rec in feature_insights['recommendations'].get('features_to_remove', []):
                    if rec['feature'] in self.data.columns and rec['feature'] != self.target_column:
                        cols_to_drop.append(rec['feature'])
                        logging.info(f"Removing '{rec['feature']}' per feature analysis: {rec['reason']}")
                for suggestion in feature_insights['recommendations'].get('feature_engineering_suggestions', []):
                    logging.info(f"Feature engineering suggestion: {suggestion['suggestion']} — {suggestion['details']}")

            # Always remove target to prevent data leakage
            if self.target_column and self.target_column in self.data.columns:
                cols_to_drop.append(self.target_column)
                logging.info(f"Removing target column '{self.target_column}' to prevent data leakage.")

            if cols_to_drop:
                logging.info(f"Dropping columns: {list(set(cols_to_drop))}")
                self.data = self.data.drop(columns=list(set(cols_to_drop)))
                numerical_features = [f for f in numerical_features if f not in cols_to_drop]
                categorical_features = [f for f in categorical_features if f not in cols_to_drop]

            # Keep identifier columns as pass-through (no encoding)
            passthrough_ids: List[str] = []
            for col in self.protected_columns:
                if col in categorical_features and 'ID' in col.upper():
                    passthrough_ids.append(col)
                    logging.info(f"Keeping identifier '{col}' as pass-through.")
            if passthrough_ids:
                categorical_features = [f for f in categorical_features if f not in passthrough_ids]

            # ── Fit and transform ─────────────────────────────────────────
            pipeline = self.create_preprocessing_pipeline(numerical_features, categorical_features)
            logging.info("Fitting and transforming the data with the pipeline...")
            processed_data = pipeline.fit_transform(self.data)
            feature_names = list(pipeline.get_feature_names_out())

            if sparse.issparse(processed_data):
                processed_df = pd.DataFrame.sparse.from_spmatrix(
                    processed_data, index=self.data.index, columns=feature_names
                )
            else:
                processed_df = pd.DataFrame(processed_data, columns=feature_names, index=self.data.index)

            logging.info(f"Pipeline complete — shape after main pipeline: {processed_df.shape}")

            # ── Optional PCA ──────────────────────────────────────────────
            if self.use_pca:
                processed_df = self._apply_pca(processed_df)

            logging.info(f"Data preprocessing complete. Final shape: {processed_df.shape}")
            return processed_df

        except Exception as exc:
            logging.error(f"An error occurred during preprocessing: {exc}", exc_info=True)
            return None

    # ── PCA helper ────────────────────────────────────────────────────────────

    def _apply_pca(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply PCA dimensionality reduction after the main pipeline.

        Keeps enough components to explain `pca_variance_threshold` of variance.
        Output columns are named PC_0, PC_1, … (named features are gone).
        Sets self.fitted_pca for downstream inspection.
        """
        logging.info(f"[PCA] Fitting PCA — variance_threshold={self.pca_variance_threshold}")

        if hasattr(df, "sparse"):
            X = df.sparse.to_dense().values
        else:
            X = df.values.astype(float)

        pca = PCA(n_components=self.pca_variance_threshold, svd_solver="full")
        X_pca = pca.fit_transform(X)
        self.fitted_pca = pca

        n_components = X_pca.shape[1]
        explained = pca.explained_variance_ratio_.sum()
        logging.info(
            f"[PCA] Reduced {df.shape[1]} → {n_components} components "
            f"(explains {explained:.1%} of variance)."
        )
        logging.warning(
            "[PCA] Feature names are now anonymous (PC_0 … PC_%d). "
            "Human-readable recommendations will NOT be available.",
            n_components - 1,
        )
        pc_cols = [f"PC_{i}" for i in range(n_components)]
        return pd.DataFrame(X_pca, columns=pc_cols, index=df.index)


if __name__ == '__main__':
    logging.info("--- Running Preprocessing Agent in Standalone Mode ---")
    sample_data = {
        'MachineID': ['M01', 'M02', 'M03', 'M04', 'M05'],
        'Temperature': [300.1, 301.5, 299.8, 302.1, 301.9],
        'Vibration': [1.5, 1.7, 1.4, 1.8, 1.9],
        'Failure_Type': ['None', 'Power', 'None', 'Overstrain', 'None'],
        'Downtime_Cost': [100, 5000, 90, 8000, 120],
    }
    sample_df = pd.DataFrame(sample_data)
    sample_df.iloc[2, 1] = None
    agent = PreprocessingAgent(sample_df)
    result = agent.preprocess()
    if result is not None:
        logging.info(f"Preprocessed shape: {result.shape}")
