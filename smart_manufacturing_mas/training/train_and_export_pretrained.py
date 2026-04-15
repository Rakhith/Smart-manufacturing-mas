import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BASE_DIR / "artifacts" / "pretrained_models"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SMART_MAINTENANCE_PATH = BASE_DIR / "data" / "Smart_Manufacturing_Maintenance_Dataset" / "smart_maintenance_dataset.csv"
INTELLIGENT_MANUFACTURING_PATH = BASE_DIR / "data" / "Intelligent_Manufacturing_Dataset" / "manufacturing_6G_dataset.csv"

REGRESSION_TARGET = "Failure_Prob"
CLASSIFICATION_TARGET = "Maintenance_Priority"
ALTERNATE_CLASSIFICATION_TARGET = "Efficiency_Status"


def build_preprocessor(x_df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in x_df.columns if c not in numeric_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def save_bundle(bundle: dict, bundle_file: str) -> None:
    joblib.dump(bundle, ARTIFACT_DIR / bundle_file)


def run_regression_exports() -> list[dict]:
    reg_df = pd.read_csv(SMART_MAINTENANCE_PATH)

    if REGRESSION_TARGET not in reg_df.columns:
        raise ValueError(f"Regression target '{REGRESSION_TARGET}' not found.")

    x_reg = reg_df.drop(columns=[REGRESSION_TARGET])
    for drop_col in ("Machine_ID", CLASSIFICATION_TARGET):
        if drop_col in x_reg.columns:
            x_reg = x_reg.drop(columns=[drop_col])
    y_reg = reg_df[REGRESSION_TARGET]

    x_train, x_val, y_train, y_val = train_test_split(
        x_reg, y_reg, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=42),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "SVR": SVR(kernel="rbf", C=1.0),
    }

    entries = []

    for model_name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(x_train)),
            ("model", estimator),
        ])
        pipe.fit(x_train, y_train)
        train_preds = pipe.predict(x_train)
        val_preds = pipe.predict(x_val)

        metrics = {
            "train_r2": float(r2_score(y_train, train_preds)),
            "train_mse": float(mean_squared_error(y_train, train_preds)),
            "r2": float(r2_score(y_val, val_preds)),
            "mse": float(mean_squared_error(y_val, val_preds)),
        }
        bundle_file = f"regression__{REGRESSION_TARGET}__{model_name}.joblib"
        save_bundle(
            {
                "model_name": model_name,
                "problem_type": "regression",
                "target_column": REGRESSION_TARGET,
                "feature_columns": x_train.columns.tolist(),
                "pipeline": pipe,
                "metrics": metrics,
                "train_prediction_stats": {
                    "mean": float(np.mean(train_preds)),
                    "std": float(np.std(train_preds)),
                },
            },
            bundle_file,
        )

        entries.append(
            {
                "model_name": model_name,
                "bundle_file": bundle_file,
                "target_column": REGRESSION_TARGET,
                "feature_columns": x_train.columns.tolist(),
                "metrics": metrics,
            }
        )

    return entries


def run_classification_exports(dataset_path: Path, target_column: str, drop_columns: tuple[str, ...]) -> list[dict]:
    cls_df = pd.read_csv(dataset_path)
    if target_column not in cls_df.columns:
        raise ValueError(f"Classification target '{target_column}' not found.")

    x_cls = cls_df.drop(columns=[target_column])
    for drop_col in drop_columns:
        if drop_col in x_cls.columns:
            x_cls = x_cls.drop(columns=[drop_col])
    y_cls = cls_df[target_column]

    x_train, x_val, y_train, y_val = train_test_split(
        x_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "SVC": SVC(kernel="rbf", C=1.0),
    }

    entries = []
    for model_name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(x_train)),
            ("model", estimator),
        ])
        pipe.fit(x_train, y_train)
        val_preds = pipe.predict(x_val)

        metrics = {
            "accuracy": float(accuracy_score(y_val, val_preds)),
            "classification_report": classification_report(y_val, val_preds),
        }

        bundle_file = f"classification__{target_column}__{model_name}.joblib"
        save_bundle(
            {
                "model_name": model_name,
                "problem_type": "classification",
                "target_column": target_column,
                "feature_columns": x_train.columns.tolist(),
                "pipeline": pipe,
                "metrics": metrics,
            },
            bundle_file,
        )

        entries.append(
            {
                "model_name": model_name,
                "bundle_file": bundle_file,
                "target_column": target_column,
                "feature_columns": x_train.columns.tolist(),
                "metrics": {"accuracy": metrics["accuracy"]},
            }
        )

    return entries


def main() -> None:
    regression_entries = run_regression_exports()
    classification_entries = run_classification_exports(
        SMART_MAINTENANCE_PATH,
        CLASSIFICATION_TARGET,
        ("Machine_ID", REGRESSION_TARGET),
    )
    alternate_classification_entries = run_classification_exports(
        INTELLIGENT_MANUFACTURING_PATH,
        ALTERNATE_CLASSIFICATION_TARGET,
        ("Timestamp", "Machine_ID"),
    )

    registry = {
        "regression": regression_entries,
        "classification": classification_entries + alternate_classification_entries,
    }

    registry_path = ARTIFACT_DIR / "registry.json"
    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"Saved registry: {registry_path}")
    print("Regression models:", [e["model_name"] for e in regression_entries])
    print("Classification models:", [e["model_name"] for e in registry["classification"]])


if __name__ == "__main__":
    main()
