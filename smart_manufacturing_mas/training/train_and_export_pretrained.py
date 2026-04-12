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

REGRESSION_TRAIN_PATH = BASE_DIR / "data" / "smart_manufacturing_dataset.csv"
REGRESSION_VALID_PATH = BASE_DIR / "data" / "digital_manufacturing_dataset.csv"
CLASSIFICATION_PATH = BASE_DIR / "data" / "Smart Manufacturing Maintenance Dataset" / "smart_maintenance_dataset.csv"

REGRESSION_TARGET = "Production_Efficiency"
CLASSIFICATION_TARGET = "Maintenance_Priority"


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
    reg_df = pd.read_csv(REGRESSION_TRAIN_PATH)
    reg_val_df = pd.read_csv(REGRESSION_VALID_PATH)

    if REGRESSION_TARGET not in reg_df.columns:
        raise ValueError(f"Regression target '{REGRESSION_TARGET}' not found.")

    x_reg = reg_df.drop(columns=[REGRESSION_TARGET])
    if "Agent_ID" in x_reg.columns:
        x_reg = x_reg.drop(columns=["Agent_ID"])
    y_reg = reg_df[REGRESSION_TARGET]

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=42),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "SVR": SVR(kernel="rbf", C=1.0),
    }

    common_reg_features = [c for c in x_reg.columns if c in reg_val_df.columns]
    entries = []

    for model_name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(x_reg)),
            ("model", estimator),
        ])
        pipe.fit(x_reg, y_reg)
        train_preds = pipe.predict(x_reg)

        metrics = {
            "train_r2": float(r2_score(y_reg, train_preds)),
            "train_mse": float(mean_squared_error(y_reg, train_preds)),
        }

        if common_reg_features and REGRESSION_TARGET in reg_val_df.columns:
            x_val = reg_val_df[common_reg_features].copy()
            for col in x_reg.columns:
                if col not in x_val.columns:
                    x_val[col] = np.nan
            x_val = x_val[x_reg.columns]
            y_val = reg_val_df[REGRESSION_TARGET]
            val_preds = pipe.predict(x_val)
            metrics["val_r2"] = float(r2_score(y_val, val_preds))
            metrics["val_mse"] = float(mean_squared_error(y_val, val_preds))
        else:
            metrics["val_r2"] = None
            metrics["val_mse"] = None
            metrics["val_note"] = "Validation skipped: no common feature schema or missing target in validation dataset."

        bundle_file = f"regression_{model_name}.joblib"
        save_bundle(
            {
                "model_name": model_name,
                "problem_type": "regression",
                "target_column": REGRESSION_TARGET,
                "feature_columns": x_reg.columns.tolist(),
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
                "feature_columns": x_reg.columns.tolist(),
                "metrics": metrics,
            }
        )

    return entries


def run_classification_exports() -> list[dict]:
    cls_df = pd.read_csv(CLASSIFICATION_PATH)
    if CLASSIFICATION_TARGET not in cls_df.columns:
        raise ValueError(f"Classification target '{CLASSIFICATION_TARGET}' not found.")

    x_cls = cls_df.drop(columns=[CLASSIFICATION_TARGET])
    y_cls = cls_df[CLASSIFICATION_TARGET]

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

        bundle_file = f"classification_{model_name}.joblib"
        save_bundle(
            {
                "model_name": model_name,
                "problem_type": "classification",
                "target_column": CLASSIFICATION_TARGET,
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
                "target_column": CLASSIFICATION_TARGET,
                "feature_columns": x_train.columns.tolist(),
                "metrics": {"accuracy": metrics["accuracy"]},
            }
        )

    return entries


def main() -> None:
    regression_entries = run_regression_exports()
    classification_entries = run_classification_exports()

    registry = {
        "regression": regression_entries,
        "classification": classification_entries,
    }

    registry_path = ARTIFACT_DIR / "registry.json"
    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"Saved registry: {registry_path}")
    print("Regression models:", [e["model_name"] for e in regression_entries])
    print("Classification models:", [e["model_name"] for e in classification_entries])


if __name__ == "__main__":
    main()
