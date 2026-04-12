
import pandas as pd
import logging
from typing import Dict, Any
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')


class OptimizationAgent:
    """
    The OptimizationAgent takes insights from the AnalysisAgent and generates
    a prescriptive maintenance action plan. It represents the final step in
    closing the loop from prediction to action.
    """

    UNIFIED_OUTPUT_COLUMNS = [
        'Problem_Type',
        'Machine_ID',
        'Predicted_Label',
        'Predicted_Value',
        'Anomaly_Count',
        'Priority_Level',
        'Priority_Score',
        'Reason_for_Action',
        'Recommended_Action',
        'Contributing_Factors',
        'Estimated_Cost',
        'Timeframe',
        'Model_Confidence',
        'Model_Warning'
    ]

    def __init__(self, analysis_results: Dict[str, Any]):
        """
        Initialize the OptimizationAgent.
        Args:
            analysis_results (dict): A dictionary containing the results from the
                                     AnalysisAgent, including predictions, original test data
                                     for context, and feature importances.
        """
        logging.info("Initializing Optimization Agent...")
        self.results = self._normalize_analysis_results(analysis_results)
        self.problem_type = self._infer_problem_type(self.results)

        # Check for required keys based on normalized analysis type
        if self.problem_type == 'anomaly_detection':
            # Anomaly detection results
            if 'results_df' not in self.results or 'anomaly_labels' not in self.results:
                raise ValueError("Anomaly detection results missing required keys: 'results_df' and 'anomaly_labels'")
        else:
            # Supervised learning results - train_predictions only needed for regression
            required_keys = ['test_data', 'test_predictions', 'feature_importances']
            if not all(k in self.results for k in required_keys):
                raise ValueError("Analysis results are missing required keys for optimization.")

    @staticmethod
    def _infer_problem_type(results: Dict[str, Any]) -> str:
        explicit_type = str(results.get('problem_type', '')).strip().lower()
        if explicit_type in {'classification', 'regression', 'anomaly_detection'}:
            return explicit_type

        if 'results_df' in results and 'anomaly_labels' in results:
            return 'anomaly_detection'

        train_predictions = results.get('train_predictions')
        if (
            results.get('r2') is not None and
            train_predictions is not None and
            np.issubdtype(np.asarray(train_predictions).dtype, np.number)
        ):
            return 'regression'

        return 'classification'

    @staticmethod
    def _to_feature_importance_df(feature_importances: Any) -> pd.DataFrame:
        if isinstance(feature_importances, pd.DataFrame):
            return feature_importances
        if isinstance(feature_importances, list) and feature_importances:
            try:
                return pd.DataFrame(feature_importances)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def _normalize_analysis_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize input into a common internal shape while remaining backward compatible.

        Supported unified input schema (single call path):
        {
            'problem_type': 'classification'|'regression'|'anomaly_detection',
            'data': <pd.DataFrame>,
            'predictions': <array-like>,
            'train_predictions': <array-like, optional for regression>,
            'feature_importances': <pd.DataFrame|list, optional>,
            'metrics': {'accuracy': float, 'r2': float, ...},
            'anomaly_scores': <array-like, optional for anomaly_detection>
        }
        """
        normalized = dict(analysis_results)

        if 'problem_type' in normalized and 'data' in normalized:
            problem_type = str(normalized.get('problem_type', '')).strip().lower()
            data = normalized.get('data')
            predictions = normalized.get('predictions')
            feature_importances = self._to_feature_importance_df(normalized.get('feature_importances'))
            metrics = normalized.get('metrics', {})

            normalized = {
                'problem_type': problem_type,
                'feature_importances': feature_importances
            }

            if isinstance(metrics, dict):
                normalized.update(metrics)

            if problem_type == 'anomaly_detection':
                results_df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
                if predictions is not None and 'anomaly_label' not in results_df.columns and 'Is_Anomaly' not in results_df.columns:
                    results_df['anomaly_label'] = predictions

                anomaly_scores = analysis_results.get('anomaly_scores')
                if anomaly_scores is not None and 'anomaly_score' not in results_df.columns and 'Anomaly_Score' not in results_df.columns:
                    results_df['anomaly_score'] = anomaly_scores

                if 'anomaly_label' in results_df.columns:
                    anomaly_labels = results_df['anomaly_label'].tolist()
                elif 'Is_Anomaly' in results_df.columns:
                    anomaly_labels = [(-1 if bool(v) else 1) for v in results_df['Is_Anomaly']]
                elif predictions is not None:
                    anomaly_labels = list(predictions)
                else:
                    anomaly_labels = []

                normalized['results_df'] = results_df
                normalized['anomaly_labels'] = anomaly_labels
            else:
                test_data = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
                normalized['test_data'] = test_data
                normalized['test_predictions'] = predictions if predictions is not None else []
                if normalized['feature_importances'].empty:
                    normalized['feature_importances'] = pd.DataFrame()

                if problem_type == 'regression':
                    normalized['train_predictions'] = analysis_results.get('train_predictions')

            return normalized

        # Backward-compatible path for existing callers
        normalized['feature_importances'] = self._to_feature_importance_df(normalized.get('feature_importances'))
        return normalized

    def _standardize_recommendations(
        self,
        recommendations_df: pd.DataFrame,
        problem_type: str,
        model_performance: Dict[str, Any]
    ) -> pd.DataFrame:
        if recommendations_df is None or recommendations_df.empty:
            return pd.DataFrame(columns=self.UNIFIED_OUTPUT_COLUMNS)

        standardized = recommendations_df.copy()

        standardized['Problem_Type'] = problem_type
        if 'Model_Confidence' not in standardized.columns:
            standardized['Model_Confidence'] = model_performance.get('recommendation_confidence', 'High')

        warning = model_performance.get('reliability_warning')
        if warning and 'Model_Warning' not in standardized.columns:
            standardized['Model_Warning'] = warning

        for col in self.UNIFIED_OUTPUT_COLUMNS:
            if col not in standardized.columns:
                standardized[col] = None

        return standardized[self.UNIFIED_OUTPUT_COLUMNS]

    def _assess_model_performance(self) -> Dict[str, Any]:
        """Assess model performance and provide context for recommendations."""
        performance = {
            'confidence_level': 'High',
            'reliability_warning': None,
            'recommendation_confidence': 'High'
        }
        
        # Check for common performance issues
        if 'accuracy' in self.results:
            accuracy = self.results['accuracy']
            if accuracy < 0.6:
                performance['confidence_level'] = 'Low'
                performance['reliability_warning'] = f"Model accuracy ({accuracy:.2%}) is below recommended threshold (60%)"
                performance['recommendation_confidence'] = 'Low'
            elif accuracy < 0.8:
                performance['confidence_level'] = 'Medium'
                performance['reliability_warning'] = f"Model accuracy ({accuracy:.2%}) suggests moderate reliability"
                performance['recommendation_confidence'] = 'Medium'
        
        if 'r2' in self.results:
            r2 = self.results['r2']
            if r2 < 0:
                performance['confidence_level'] = 'Very Low'
                performance['reliability_warning'] = f"Model performs worse than baseline (R² = {r2:.3f})"
                performance['recommendation_confidence'] = 'Very Low'
            elif r2 < 0.3:
                performance['confidence_level'] = 'Low'
                performance['reliability_warning'] = f"Model explains only {r2:.1%} of variance"
                performance['recommendation_confidence'] = 'Low'
        
        return performance

    def _infer_target_column(self, data: pd.DataFrame, predictions) -> str:
        """Attempt to infer the target column by matching prediction values."""
        try:
            prediction_strings = set(pd.Series(predictions).astype(str).unique())
        except Exception:
            prediction_strings = set(str(p) for p in set(predictions))

        for column in data.columns:
            series = data[column]
            try:
                values = set(series.astype(str).unique())
            except Exception:
                values = set(series.apply(str).unique())
            if prediction_strings.issubset(values):
                return column
        return ""

    @staticmethod
    def _is_numeric_label(label) -> bool:
        if isinstance(label, (int, float, np.number)):
            return True
        if isinstance(label, str):
            try:
                float(label)
                return True
            except ValueError:
                return False
        return False

    def _score_from_keywords(self, label: str, target_column: str) -> float:
        """Derive a priority score (1-3) based on label semantics and context."""
        label_lower = str(label).strip().lower()
        if not label_lower:
            return 2.0

        target_lower = target_column.lower() if target_column else ""

        # Context where lower textual value implies higher risk (e.g., efficiency, health)
        invert_low_high = any(
            keyword in target_lower
            for keyword in ["efficiency", "performance", "quality", "health", "uptime", "score", "yield"]
        )

        # Priority-specific cues override general rules
        if "priority" in label_lower or "risk" in label_lower or "severity" in label_lower:
            if "high" in label_lower:
                return 3.0
            if "medium" in label_lower:
                return 2.0
            if "low" in label_lower:
                return 1.0

        high_severity_keywords = [
            "critical", "urgent", "fail", "failure", "fault", "alarm",
            "down", "offline", "shutdown", "alert", "incident", "unsafe",
            "issue", "anomaly", "breach", "hazard", "poor", "degraded", "risk"
        ]
        medium_severity_keywords = [
            "medium", "moderate", "warning", "reduced", "caution",
            "watch", "elevated", "unstable", "attention"
        ]
        low_severity_keywords = [
            "normal", "ok", "good", "optimal", "stable",
            "healthy", "excellent", "nominal", "efficient"
        ]

        # Handle generic "high"/"low" depending on context
        if invert_low_high:
            high_severity_keywords.append("low")
            low_severity_keywords.append("high")
        else:
            high_severity_keywords.append("high")
            low_severity_keywords.append("low")

        for keyword in high_severity_keywords:
            if keyword in label_lower:
                return 3.0
        for keyword in medium_severity_keywords:
            if keyword in label_lower:
                return 2.0
        for keyword in low_severity_keywords:
            if keyword in label_lower:
                return 1.0

        # Default when semantics are unclear
        return 2.0

    def _derive_priority_mapping(self, predictions, target_column: str) -> Dict[Any, float]:
        """Create a mapping from prediction labels to numeric priority scores."""
        unique_seen = []
        for label in predictions:
            if label not in unique_seen:
                unique_seen.append(label)

        numeric_labels = []
        mapping: Dict[Any, float] = {}

        for label in unique_seen:
            if self._is_numeric_label(label):
                try:
                    numeric_value = float(label)
                except ValueError:
                    numeric_value = float(str(label))
                numeric_labels.append((label, numeric_value))
            else:
                mapping[label] = self._score_from_keywords(label, target_column)

        if numeric_labels:
            numeric_values = [value for _, value in numeric_labels]
            min_val = min(numeric_values)
            max_val = max(numeric_values)
            for original_label, numeric_value in numeric_labels:
                if max_val == min_val:
                    score = 2.0
                else:
                    score = 1.0 + 2.0 * ((numeric_value - min_val) / (max_val - min_val))
                mapping[original_label] = float(score)

        return mapping

    def _build_contributing_factors(self, row: pd.Series, feature_names: list[str]) -> str:
        factors = []
        for feature in feature_names:
            if feature not in row:
                continue
            value = row[feature]
            if isinstance(value, (int, float, np.number)):
                factors.append(f"{feature}={value:.2f}")
            else:
                factors.append(f"{feature}={value}")
        return ", ".join(factors)

    def _action_plan_from_score(self, score: float) -> Dict[str, str]:
        if score >= 2.5:
            return {
                "priority_level": "Critical",
                "action": "IMMEDIATE: Dispatch maintenance team to investigate and recover performance.",
                "cost": "High ($5,000+)",
                "timeframe": "Within 24-48 hours"
            }
        if score >= 1.5:
            return {
                "priority_level": "Elevated",
                "action": "Schedule targeted maintenance in the upcoming service window.",
                "cost": "Medium ($1,000-$5,000)",
                "timeframe": "Within 1-2 weeks"
            }
        return {
            "priority_level": "Low",
            "action": "Continue monitoring and maintain current operating procedures.",
            "cost": "Low (<$1,000)",
            "timeframe": "Next scheduled maintenance"
        }

    def generate_recommendations(self) -> pd.DataFrame:
        """
        Generates comprehensive recommendations based on analysis results.
        For classification: Prioritizes maintenance tasks with detailed insights
        For regression: Identifies concerning predicted values with context
        For anomaly detection: Identifies and explains anomalous behavior
        Returns:
            pd.DataFrame: A DataFrame containing recommendations and actions
        """
        logging.info("Generating comprehensive prescriptive recommendations...")
        
        # Add model performance context to recommendations
        model_performance = self._assess_model_performance()
        logging.info(f"Model performance assessment: {model_performance}")

        train_predictions = self.results.get('train_predictions')
        regression_baseline = self.results.get('regression_baseline')

        if self.problem_type == 'regression':
            # Handle regression results
            results_df = self.results['test_data'].copy()
            results_df['Predicted_Value'] = self.results['test_predictions']
            
            # Calculate prediction thresholds based on training data distribution
            if train_predictions is not None:
                train_mean = np.asarray(train_predictions).mean()
                train_std = np.asarray(train_predictions).std()
            elif isinstance(regression_baseline, dict):
                train_mean = float(regression_baseline.get('mean', results_df['Predicted_Value'].mean()))
                train_std = float(regression_baseline.get('std', results_df['Predicted_Value'].std()))
            else:
                train_mean = float(results_df['Predicted_Value'].mean())
                train_std = float(results_df['Predicted_Value'].std())
            if train_std == 0:
                train_std = 1.0
            high_threshold = train_mean + 2 * train_std
            critical_threshold = train_mean + 3 * train_std
            
            # Identify machines with concerning predicted values
            critical = results_df[results_df['Predicted_Value'] >= critical_threshold].copy()
            warning = results_df[(results_df['Predicted_Value'] >= high_threshold) & 
                               (results_df['Predicted_Value'] < critical_threshold)].copy()
            
            recommendations = []
            
            # Handle critical cases
            for _, row in critical.iterrows():
                recommendations.append({
                    'Machine_ID': row['Machine_ID'] if 'Machine_ID' in row else 'Unknown',
                    'Priority_Score': (row['Predicted_Value'] - train_mean) / train_std,
                    'Current_Value': row['Predicted_Value'],
                    'Threshold': critical_threshold,
                    'Severity': 'Critical',
                    'Reason_for_Action': f"Predicted value ({row['Predicted_Value']:.2f}) exceeds critical threshold ({critical_threshold:.2f})",
                    'Recommended_Action': "Immediate inspection and preventive maintenance required"
                })
            
            # Handle warning cases
            for _, row in warning.iterrows():
                recommendations.append({
                    'Machine_ID': row['Machine_ID'] if 'Machine_ID' in row else 'Unknown',
                    'Priority_Score': (row['Predicted_Value'] - train_mean) / train_std,
                    'Current_Value': row['Predicted_Value'],
                    'Threshold': high_threshold,
                    'Severity': 'Warning',
                    'Reason_for_Action': f"Predicted value ({row['Predicted_Value']:.2f}) exceeds warning threshold ({high_threshold:.2f})",
                    'Recommended_Action': "Schedule inspection within next maintenance window"
                })
            
            if recommendations:
                recommendations_df = pd.DataFrame(recommendations)
                recommendations_df = recommendations_df.sort_values('Priority_Score', ascending=False)
                return self._standardize_recommendations(recommendations_df, 'regression', model_performance)
            else:
                logging.info("No concerning predictions identified. All values within normal range.")
                return self._standardize_recommendations(pd.DataFrame(), 'regression', model_performance)
                
        elif self.problem_type == 'anomaly_detection':
            # Handle anomaly detection results
            results_df = self.results['results_df'].copy()

            # Normalize anomaly column names across schema versions.
            if 'Is_Anomaly' not in results_df.columns and 'anomaly_label' in results_df.columns:
                results_df['Is_Anomaly'] = results_df['anomaly_label'] == -1
            if 'Anomaly_Score' not in results_df.columns and 'anomaly_score' in results_df.columns:
                results_df['Anomaly_Score'] = results_df['anomaly_score']

            if not results_df.empty:
                identifier_cols = [col for col in results_df.columns if col.startswith("identifier__")]
                if identifier_cols:
                    for id_col in identifier_cols:
                        results_df[id_col.replace("identifier__", "")] = results_df[id_col]

            if 'Machine_ID' not in results_df.columns:
                fallback_id_col = None
                for candidate in ('MachineId', 'machine_id', 'machine', 'Asset_ID', 'asset_id'):
                    if candidate in results_df.columns:
                        fallback_id_col = candidate
                        break
                if fallback_id_col is None:
                    identifier_cols = [c for c in results_df.columns if c.startswith('identifier__')]
                    if identifier_cols:
                        fallback_id_col = identifier_cols[0]

                if fallback_id_col is not None:
                    results_df['Machine_ID'] = results_df[fallback_id_col]
                else:
                    # Final fallback: keep analysis usable even without explicit machine identifiers.
                    results_df['Machine_ID'] = results_df.index.astype(str)

            anomalous = results_df[results_df['Is_Anomaly']]
            
            if anomalous.empty:
                logging.info("No anomalies detected. All machines operating within normal parameters.")
                return self._standardize_recommendations(pd.DataFrame(), 'anomaly_detection', model_performance)
            
            # Group anomalies by Machine_ID
            agg_spec = {
                'Is_Anomaly': 'count',
                'Anomaly_Score': ['mean', 'min', 'max']
            }
            if 'Timestamp' in anomalous.columns:
                agg_spec['Timestamp'] = ['min', 'max']
            machine_anomalies = anomalous.groupby('Machine_ID').agg(agg_spec)
            machine_anomalies.columns = ['Anomaly_Count', 'Avg_Anomaly_Score', 'Min_Anomaly_Score', 'Max_Anomaly_Score'] + (
                ['First_Anomaly_Time', 'Last_Anomaly_Time'] if 'Timestamp' in anomalous.columns else []
            )
            machine_anomalies = machine_anomalies.reset_index()
            machine_anomalies = machine_anomalies.sort_values('Avg_Anomaly_Score')
            
            # Generate recommendations for each anomalous machine
            recommendations = []
            zscore_columns = [col for col in anomalous.columns if col.endswith('_zscore')]
            for _, row in machine_anomalies.iterrows():
                machine_data = anomalous[anomalous['Machine_ID'] == row['Machine_ID']]
                
                feature_signals = []
                for z_col in zscore_columns:
                    metric = z_col.replace('_zscore', '')
                    mean_z = machine_data[z_col].mean()
                    if np.isnan(mean_z):
                        continue
                    value_col = f"{metric}_Value"
                    mean_val = machine_data[value_col].mean() if value_col in machine_data else None
                    feature_signals.append((metric, mean_z, mean_val))
                feature_signals.sort(key=lambda item: abs(item[1]), reverse=True)

                top_signals = feature_signals[:3]
                has_signals = len(top_signals) > 0
                indicators = []
                max_abs_z = 0.0
                for metric, mean_z, mean_val in top_signals:
                    max_abs_z = max(max_abs_z, abs(mean_z))
                    if mean_val is not None and not np.isnan(mean_val):
                        indicators.append(f"{metric}: z={mean_z:.2f}, mean≈{mean_val:.2f}")
                    else:
                        indicators.append(f"{metric}: z={mean_z:.2f}")
                indicator_text = "; ".join(indicators) if indicators else "Signal strength insufficient to isolate drivers."

                if max_abs_z >= 3:
                    recommended_action = "Schedule immediate diagnostic assessment"
                    priority_level = "Critical"
                elif max_abs_z >= 2:
                    recommended_action = "Schedule inspection and targeted monitoring"
                    priority_level = "Elevated"
                else:
                    recommended_action = "Monitor these parameters closely"
                    priority_level = "Advisory"

                first_anomaly_time = row['First_Anomaly_Time'] if 'First_Anomaly_Time' in row else None
                last_anomaly_time = row['Last_Anomaly_Time'] if 'Last_Anomaly_Time' in row else None

                recommendations.append({
                    'Machine_ID': row['Machine_ID'],
                    'Priority_Score': abs(row['Avg_Anomaly_Score']),
                    'Anomaly_Count': row['Anomaly_Count'],
                    'Avg_Anomaly_Score': row['Avg_Anomaly_Score'],
                    'Most_Anomalous_Score': row['Min_Anomaly_Score'],
                    'Least_Anomalous_Score': row['Max_Anomaly_Score'],
                    'First_Anomaly_Time': first_anomaly_time,
                    'Last_Anomaly_Time': last_anomaly_time,
                    'Top_Indicators': indicator_text,
                    'Contributing_Factors': indicator_text if has_signals else None,
                    'Priority_Level': priority_level,
                    'Reason_for_Action': (
                        f"Detected recurrent anomalies. Top signals: {indicator_text}"
                        if has_signals else "Detected recurrent anomalies across monitored features."
                    ),
                    'Recommended_Action': recommended_action
                })
            
            recommendations_df = pd.DataFrame(recommendations)
            return self._standardize_recommendations(recommendations_df, 'anomaly_detection', model_performance)
            
        else:
            # Handle classification results with enhanced, label-aware insights
            results_df = self.results['test_data'].copy()
            predictions = pd.Series(self.results['test_predictions'], index=results_df.index, name="Predicted_Label")
            results_df['Predicted_Label'] = predictions

            target_column = self._infer_target_column(results_df, predictions)
            priority_mapping = self._derive_priority_mapping(predictions, target_column)

            default_score = 2.0 if priority_mapping else 2.0
            results_df['Priority_Score'] = results_df['Predicted_Label'].map(priority_mapping).fillna(default_score)

            # Prepare feature context
            feature_names = []
            if (
                self.results.get('feature_importances') is not None
                and not self.results['feature_importances'].empty
                and 'feature' in self.results['feature_importances']
            ):
                feature_names = [
                    f.replace('num__', '').replace('cat__', '')
                    for f in self.results['feature_importances']['feature'].head(3).tolist()
                ]
            elif 'test_data' in self.results:
                feature_names = [
                    col for col in self.results['test_data'].columns
                    if col not in {'Machine_ID', 'Timestamp'}
                ][:3]

            prioritized = results_df.sort_values('Priority_Score', ascending=False)
            if 'Machine_ID' in prioritized.columns:
                prioritized = prioritized.dropna(subset=['Machine_ID'])
                prioritized = prioritized.drop_duplicates(subset=['Machine_ID'], keep='first')

            max_recommendations = 30
            prioritized = prioritized.head(max_recommendations)

            recommendations = []
            for _, row in prioritized.iterrows():
                score = float(row.get('Priority_Score', default_score))
                label_text = str(row.get('Predicted_Label', 'Unknown'))
                action_plan = self._action_plan_from_score(score)
                contributing = self._build_contributing_factors(row, feature_names)

                reason = f"Model predicted '{label_text}' for the current operating state."
                if feature_names:
                    readable_features = ", ".join(feature_names)
                    reason += f" Key drivers include: {readable_features}."

                recommendations.append({
                    'Machine_ID': row.get('Machine_ID', 'Unknown'),
                    'Predicted_Label': label_text,
                    'Priority_Level': action_plan['priority_level'],
                    'Priority_Score': round(score, 2),
                    'Contributing_Factors': contributing if contributing else "Model-driven signals (top features unavailable).",
                    'Reason_for_Action': reason,
                    'Recommended_Action': action_plan['action'],
                    'Estimated_Cost': action_plan['cost'],
                    'Timeframe': action_plan['timeframe'],
                    'Model_Confidence': model_performance['recommendation_confidence']
                })

            if recommendations:
                recommendations_df = pd.DataFrame(recommendations).sort_values('Priority_Score', ascending=False)

                if model_performance['reliability_warning']:
                    logging.warning(f"⚠️ {model_performance['reliability_warning']}")
                    recommendations_df['Model_Warning'] = model_performance['reliability_warning']

                return self._standardize_recommendations(recommendations_df, 'classification', model_performance)

            logging.info("No maintenance recommendations generated.")
            return self._standardize_recommendations(pd.DataFrame(), 'classification', model_performance)

    def generate_summary_report(self, recommendations_df: pd.DataFrame) -> str:
        """Generate a human-readable summary report of recommendations."""
        if recommendations_df.empty:
            return "✅ All systems operating within normal parameters. No immediate maintenance actions required."
        
        report_lines = ["🎯 MAINTENANCE RECOMMENDATIONS SUMMARY", "=" * 50]
        
        # Count by priority
        if 'Priority_Level' in recommendations_df.columns:
            priority_counts = recommendations_df['Priority_Level'].value_counts()
            report_lines.append(f"\n📊 Priority Distribution:")
            for priority, count in priority_counts.items():
                report_lines.append(f"  • {priority}: {count} machines")
        
        # Cost estimation
        if 'Estimated_Cost' in recommendations_df.columns:
            cost_summary = recommendations_df['Estimated_Cost'].value_counts()
            report_lines.append(f"\n💰 Estimated Cost Distribution:")
            for cost, count in cost_summary.items():
                report_lines.append(f"  • {cost}: {count} machines")
        
        # Top recommendations
        top_recs = recommendations_df.head(3)
        report_lines.append(f"\n🚨 TOP PRIORITY ACTIONS:")
        for _, rec in top_recs.iterrows():
            report_lines.append(f"  • Machine {rec.get('Machine_ID', 'Unknown')}: {rec.get('Recommended_Action', 'N/A')}")
        
        # Model confidence warning
        if 'Model_Warning' in recommendations_df.columns and not recommendations_df['Model_Warning'].isna().all():
            warnings = recommendations_df['Model_Warning'].dropna().unique()
            if warnings:
                report_lines.append(f"\n⚠️ MODEL RELIABILITY WARNING:")
                for warning in warnings:
                    report_lines.append(f"  • {warning}")
        
        return "\n".join(report_lines)

    def run(self) -> Dict[str, Any]:
        """
        Single-call entry point with a uniform output payload for all problem types.
        """
        recommendations_df = self.generate_recommendations()
        return {
            'problem_type': self.problem_type,
            'recommendations': recommendations_df,
            'summary_report': self.generate_summary_report(recommendations_df),
            'output_schema': list(self.UNIFIED_OUTPUT_COLUMNS)
        }

if __name__ == '__main__':
    logging.info("--- Running Optimization Agent in Standalone Mode ---")
    
    # Create sample analysis results for demonstration
    sample_analysis_results = {
        'test_data': pd.DataFrame({
            'Machine_ID': ['M01', 'M02', 'M03', 'M04'],
            'Temp_C': [308.1, 300.5, 309.2, 299.8],
            'Vibration_mm_s': [2.5, 1.2, 2.8, 1.1]
        }),
        'test_predictions': [3, 1, 3, 2], # Two high-priority predictions
        'feature_importances': pd.DataFrame({
            'feature': ['num__Vibration_mm_s', 'num__Temp_C'],
            'importance': [0.6, 0.4]
        })
    }
    
    # Initialize and run the agent
    optimization_agent = OptimizationAgent(sample_analysis_results)
    optimization_agent.generate_recommendations()
    
    logging.info("--- End of Standalone Run ---")
