"""
Prediction Analysis Module

This module analyzes model predictions, generates recommendations, and computes performance metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, precision_recall_fscore_support


class PredictionAnalyzer:
    """Analyzes predictions from trained models and generates insights."""

    def __init__(
        self,
        predictions: List[Any],
        actual_values: Optional[List[Any]] = None,
        problem_type: str = "classification",
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize prediction analyzer.

        Args:
            predictions: List of predicted values
            actual_values: List of actual/true values (optional, for accuracy calculation)
            problem_type: "classification" or "regression"
            feature_names: Names of features used in predictions
        """
        self.predictions = np.array(predictions)
        self.actual_values = np.array(actual_values) if actual_values is not None else None
        self.problem_type = problem_type
        self.feature_names = feature_names or []
        self.has_actuals = actual_values is not None

    def analyze_classification_predictions(self) -> Dict[str, Any]:
        """Analyze classification predictions and generate metrics."""
        unique_classes = np.unique(self.predictions)
        class_distribution = {str(cls): int(np.sum(self.predictions == cls)) for cls in unique_classes}

        analysis = {
            "total_predictions": int(len(self.predictions)),
            "unique_classes": len(unique_classes),
            "class_distribution": class_distribution,
            "class_percentages": {
                str(cls): float(np.sum(self.predictions == cls) / len(self.predictions) * 100)
                for cls in unique_classes
            },
            "most_common_class": str(unique_classes[np.argmax([np.sum(self.predictions == cls) for cls in unique_classes])]),
            "least_common_class": str(
                unique_classes[np.argmin([np.sum(self.predictions == cls) for cls in unique_classes])]
            ),
        }

        if self.has_actuals:
            accuracy = float(accuracy_score(self.actual_values, self.predictions))
            analysis["accuracy"] = accuracy

            # Precision, Recall, F1
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    self.actual_values, self.predictions, average="weighted", zero_division=0
                )
                analysis["precision"] = float(precision)
                analysis["recall"] = float(recall)
                analysis["f1_score"] = float(f1)

                # Confusion matrix
                cm = confusion_matrix(self.actual_values, self.predictions)
                analysis["confusion_matrix"] = cm.tolist()

                # Detailed classification report
                report = classification_report(
                    self.actual_values, self.predictions, output_dict=True, zero_division=0
                )
                analysis["classification_report"] = {
                    k: {inner_k: float(v) if isinstance(v, (int, float, np.number)) else v for inner_k, v in v.items()}
                    if isinstance(v, dict)
                    else float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in report.items()
                }
            except Exception as e:
                analysis["metrics_error"] = str(e)

        return analysis

    def analyze_regression_predictions(self) -> Dict[str, Any]:
        """Analyze regression predictions and generate metrics."""
        analysis = {
            "total_predictions": int(len(self.predictions)),
            "mean_prediction": float(np.mean(self.predictions)),
            "median_prediction": float(np.median(self.predictions)),
            "std_prediction": float(np.std(self.predictions)),
            "min_prediction": float(np.min(self.predictions)),
            "max_prediction": float(np.max(self.predictions)),
            "range": float(np.max(self.predictions) - np.min(self.predictions)),
        }

        if self.has_actuals:
            mse = mean_squared_error(self.actual_values, self.predictions)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(self.actual_values - self.predictions))
            residuals = self.actual_values - self.predictions

            analysis["mse"] = float(mse)
            analysis["rmse"] = float(rmse)
            analysis["mae"] = float(mae)
            analysis["mean_actual"] = float(np.mean(self.actual_values))
            analysis["actual_std"] = float(np.std(self.actual_values))

            # R² score
            ss_res = np.sum((self.actual_values - self.predictions) ** 2)
            ss_tot = np.sum((self.actual_values - np.mean(self.actual_values)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            analysis["r2_score"] = float(r2)

            # Residual analysis
            analysis["residual_mean"] = float(np.mean(residuals))
            analysis["residual_std"] = float(np.std(residuals))

        return analysis

    def generate_recommendations(self) -> List[Dict[str, str]]:
        """
        Generate data-driven recommendations based on predictions.

        Returns:
            List of recommendation dictionaries with severity and message
        """
        recommendations = []

        if self.problem_type == "classification":
            analysis = self.analyze_classification_predictions()

            # Check for class imbalance
            class_percentages = analysis.get("class_percentages", {})
            if class_percentages:
                max_pct = max(class_percentages.values())
                min_pct = min(class_percentages.values())
                if max_pct > 80:
                    recommendations.append({
                        "severity": "warning",
                        "category": "Class Imbalance",
                        "message": f"Highly skewed predictions: {analysis['most_common_class']} represents {max_pct:.1f}% of predictions. Consider class balancing strategies.",
                    })

                if min_pct < 5:
                    recommendations.append({
                        "severity": "warning",
                        "category": "Minority Class",
                        "message": f"Minority class {analysis['least_common_class']} is only {min_pct:.1f}% of predictions. Model may need adjustment for underrepresented classes.",
                    })

            # Check model performance if actuals available
            if self.has_actuals:
                accuracy = analysis.get("accuracy", 0)
                if accuracy < 0.7:
                    recommendations.append({
                        "severity": "error",
                        "category": "Low Accuracy",
                        "message": f"Model accuracy on synthetic data is {accuracy:.2%}. Consider retraining or data validation.",
                    })
                elif accuracy < 0.85:
                    recommendations.append({
                        "severity": "warning",
                        "category": "Moderate Accuracy",
                        "message": f"Model accuracy is {accuracy:.2%}. Performance could be improved.",
                    })
                else:
                    recommendations.append({
                        "severity": "success",
                        "category": "Good Performance",
                        "message": f"Model achieves {accuracy:.2%} accuracy on synthetic data. Good model generalization.",
                    })

        elif self.problem_type == "regression":
            analysis = self.analyze_regression_predictions()

            if self.has_actuals:
                rmse = analysis.get("rmse", float("inf"))
                mae = analysis.get("mae", float("inf"))
                r2 = analysis.get("r2_score", -1)

                if r2 < 0:
                    recommendations.append({
                        "severity": "error",
                        "category": "Poor Model Fit",
                        "message": f"R² score is {r2:.3f} (negative). Model performs worse than baseline. Investigate data quality.",
                    })
                elif r2 < 0.6:
                    recommendations.append({
                        "severity": "warning",
                        "category": "Weak Model Performance",
                        "message": f"R² score is {r2:.3f}. Model explains only {r2*100:.1f}% of variance.",
                    })
                else:
                    recommendations.append({
                        "severity": "success",
                        "category": "Good Model Performance",
                        "message": f"R² score is {r2:.3f}. Model explains {r2*100:.1f}% of variance.",
                    })

                # MAE check
                actual_range = analysis["mean_actual"] + 3 * analysis["actual_std"]
                if actual_range > 0 and mae / actual_range > 0.2:
                    recommendations.append({
                        "severity": "warning",
                        "category": "High Prediction Error",
                        "message": f"Mean Absolute Error ({mae:.3f}) is {mae/actual_range*100:.1f}% of data range. Check prediction reliability.",
                    })

        # General recommendations based on data statistics
        if len(self.predictions) < 100:
            recommendations.append({
                "severity": "info",
                "category": "Sample Size",
                "message": f"Only {len(self.predictions)} predictions available. Results may be more reliable with larger datasets.",
            })

        if not recommendations:
            recommendations.append({
                "severity": "info",
                "category": "General",
                "message": "No specific issues detected in predictions.",
            })

        return recommendations

    def get_summary(self) -> Dict[str, Any]:
        """Get concise summary of predictions for frontend display."""
        if self.problem_type == "classification":
            analysis = self.analyze_classification_predictions()
        else:
            analysis = self.analyze_regression_predictions()

        summary = {
            "problem_type": self.problem_type,
            "total_predictions": analysis.get("total_predictions"),
            "analysis": analysis,
            "recommendations": self.generate_recommendations(),
        }

        return summary
