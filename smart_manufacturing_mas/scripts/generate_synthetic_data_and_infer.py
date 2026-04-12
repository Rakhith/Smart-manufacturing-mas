"""
Generate synthetic/artificial data based on real datasets and run inference.

This script creates artificial datasets by analyzing the statistical properties
of real data and generating new rows with similar characteristics. It then runs
inference on this synthetic data using pretrained models.

Usage:
    python generate_synthetic_data_and_infer.py --n-rows 1000
    python generate_synthetic_data_and_infer.py --n-rows 500 --seed 42
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats

# Configure paths
PROJECT_DIR = Path.cwd()
if not (PROJECT_DIR / 'data').exists():
    PROJECT_DIR = PROJECT_DIR.parent
if not (PROJECT_DIR / 'data').exists():
    PROJECT_DIR = PROJECT_DIR.parent / 'smart_manufacturing_mas'

DATA_DIR = PROJECT_DIR / 'data'
ARTIFACT_DIR = PROJECT_DIR / 'artifacts' / 'pretrained_models'
REGRESSION_DATA_PATH = DATA_DIR / 'smart_manufacturing_dataset.csv'
CLASSIFICATION_DATA_PATH = DATA_DIR / 'Smart Manufacturing Maintenance Dataset' / 'smart_maintenance_dataset.csv'

print(f"Project dir: {PROJECT_DIR}")
print(f"Data dir: {DATA_DIR}")
print(f"Artifacts dir: {ARTIFACT_DIR}")


class SyntheticDataGenerator:
    """Generate synthetic data based on statistical properties of real data."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def analyze_numeric_column(self, series: pd.Series) -> Dict:
        """Extract statistics from numeric column."""
        series_clean = series.dropna()
        return {
            'mean': float(series_clean.mean()),
            'std': float(series_clean.std()),
            'min': float(series_clean.min()),
            'max': float(series_clean.max()),
            'q25': float(series_clean.quantile(0.25)),
            'q75': float(series_clean.quantile(0.75)),
            'dtype': str(series.dtype),
        }

    def analyze_categorical_column(self, series: pd.Series) -> Dict:
        """Extract statistics from categorical column."""
        value_counts = series.value_counts(normalize=True, dropna=True)
        return {
            'categories': value_counts.index.tolist(),
            'probabilities': value_counts.values.tolist(),
            'dtype': str(series.dtype),
        }

    def generate_numeric_value(self, stats_dict: Dict) -> float:
        """Generate synthetic numeric value based on statistics."""
        # Use normal distribution clipped to observed min/max
        value = np.random.normal(
            loc=stats_dict['mean'],
            scale=stats_dict['std']
        )
        # Clip to observed range
        value = np.clip(value, stats_dict['min'], stats_dict['max'])
        
        # Convert to int if original was integer
        if 'int' in stats_dict['dtype']:
            value = int(np.round(value))
        
        return value

    def generate_categorical_value(self, stats_dict: Dict) -> str:
        """Generate synthetic categorical value based on observed proportions."""
        return np.random.choice(
            stats_dict['categories'],
            p=stats_dict['probabilities']
        )

    def generate_synthetic_data(
        self,
        real_df: pd.DataFrame,
        n_rows: int,
        target_column: str = None,
        exclude_columns: list = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic data matching the statistical properties of real data.
        
        Args:
            real_df: Real dataset to analyze
            n_rows: Number of synthetic rows to generate
            target_column: Column to exclude (e.g., the model's target)
            exclude_columns: List of columns to exclude from generation
        
        Returns:
            DataFrame with synthetic data
        """
        if exclude_columns is None:
            exclude_columns = []
        
        # Add target column to exclusion list
        if target_column:
            exclude_columns = list(set(exclude_columns + [target_column]))
        
        # Analyze each column
        column_stats = {}
        numeric_columns = []
        categorical_columns = []

        for col in real_df.columns:
            if col in exclude_columns:
                continue
            
            if pd.api.types.is_numeric_dtype(real_df[col]):
                numeric_columns.append(col)
                column_stats[col] = self.analyze_numeric_column(real_df[col])
            else:
                categorical_columns.append(col)
                column_stats[col] = self.analyze_categorical_column(real_df[col])

        print(f"  Numeric columns: {numeric_columns}")
        print(f"  Categorical columns: {categorical_columns}")

        # Generate synthetic data
        synthetic_rows = []
        for _ in range(n_rows):
            row = {}
            for col in numeric_columns:
                row[col] = self.generate_numeric_value(column_stats[col])
            for col in categorical_columns:
                row[col] = self.generate_categorical_value(column_stats[col])
            synthetic_rows.append(row)

        synthetic_df = pd.DataFrame(synthetic_rows)
        
        # Add target column with synthetic values if provided
        if target_column and target_column in real_df.columns:
            target_stats = self.analyze_numeric_column(real_df[target_column])
            synthetic_df[target_column] = [
                self.generate_numeric_value(target_stats)
                for _ in range(n_rows)
            ]

        return synthetic_df


def load_pretrained_bundle(bundle_file: Path) -> Dict:
    """Load a pretrained model bundle."""
    print(f"  Loading {bundle_file.name}...")
    bundle = joblib.load(bundle_file)
    return bundle


def run_inference_on_synthetic_data(
    synthetic_df: pd.DataFrame,
    bundle: Dict,
    bundle_name: str,
    problem_type: str,
) -> Dict:
    """
    Run inference on synthetic data using a pretrained bundle.
    
    Returns:
        Dictionary with inference results and metrics
    """
    pipeline = bundle['pipeline']
    target_column = bundle['target_column']
    feature_columns = bundle['feature_columns']
    model_name = bundle['model_name']

    # Extract features (exclude target if present)
    if target_column in synthetic_df.columns:
        X_synthetic = synthetic_df.drop(columns=[target_column])
    else:
        X_synthetic = synthetic_df.copy()

    # Align features to what the model expects
    missing_features = set(feature_columns) - set(X_synthetic.columns)
    if missing_features:
        print(f"    ⚠️  Missing features: {missing_features}. Skipping inference.")
        return None

    X_synthetic_aligned = X_synthetic[feature_columns]

    # Run predictions
    predictions = pipeline.predict(X_synthetic_aligned)

    # Prepare results
    results = {
        'bundle_name': bundle_name,
        'target_column': target_column,
        'model_name': model_name,
        'problem_type': problem_type,
        'n_predictions': len(predictions),
        'predictions_sample': predictions[:5].tolist(),  # First 5 predictions
    }

    # Add statistics
    if problem_type == 'regression':
        results['prediction_stats'] = {
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
        }
        # Get original target range from bundle if available
        if 'train_prediction_stats' in bundle:
            results['train_target_stats'] = bundle['train_prediction_stats']

    elif problem_type == 'classification':
        unique_preds = np.unique(predictions)
        pred_counts = {str(cls): int(np.sum(predictions == cls)) for cls in unique_preds}
        results['prediction_distribution'] = pred_counts

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic data and run inference with pretrained models'
    )
    parser.add_argument(
        '--n-rows',
        type=int,
        default=1000,
        help='Number of synthetic rows to generate (default: 1000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for synthetic data (default: artifacts/synthetic_data)'
    )
    args = parser.parse_args()

    n_rows = args.n_rows
    seed = args.seed

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = PROJECT_DIR / 'artifacts' / 'synthetic_data'

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n" + "=" * 70)
    print(f"Generating Synthetic Data & Running Inference")
    print(f"=" * 70)
    print(f"N rows: {n_rows}")
    print(f"Seed: {seed}")
    print(f"Output dir: {output_dir}")

    # Initialize generator
    generator = SyntheticDataGenerator(seed=seed)

    # ========== REGRESSION (Smart Manufacturing Dataset) ==========
    print(f"\n" + "-" * 70)
    print(f"REGRESSION: Smart Manufacturing Dataset")
    print(f"-" * 70)

    print(f"Loading real regression data...")
    reg_df = pd.read_csv(REGRESSION_DATA_PATH)
    print(f"  Shape: {reg_df.shape}")

    print(f"Generating synthetic regression data with {n_rows} rows...")
    synthetic_reg_df = generator.generate_synthetic_data(
        real_df=reg_df,
        n_rows=n_rows,
        target_column='Production_Efficiency',
        exclude_columns=['Agent_ID'],  # Exclude ID columns
    )
    print(f"  Generated shape: {synthetic_reg_df.shape}")

    # Save synthetic regression data
    synthetic_reg_path = output_dir / f'synthetic_regression_{n_rows}_rows.csv'
    synthetic_reg_df.to_csv(synthetic_reg_path, index=False)
    print(f"✓ Saved synthetic regression data to {synthetic_reg_path}")

    # Run inference on synthetic regression data
    print(f"\nRunning regression inference...")
    regression_inferences = []

    reg_bundle_file = ARTIFACT_DIR / 'regression__Production_Efficiency__RandomForestRegressor.joblib'
    if reg_bundle_file.exists():
        try:
            bundle = load_pretrained_bundle(reg_bundle_file)
            result = run_inference_on_synthetic_data(
                synthetic_reg_df,
                bundle,
                bundle_name=reg_bundle_file.name,
                problem_type='regression'
            )
            if result:
                regression_inferences.append(result)
                print(f"✓ {result['model_name']} inference complete")
                print(f"  Predictions sample: {result['predictions_sample']}")
                if 'prediction_stats' in result:
                    print(f"  Prediction stats: {result['prediction_stats']}")
        except Exception as e:
            print(f"✗ Error during inference: {str(e)[:100]}")
    else:
        print(f"✗ Bundle not found: {reg_bundle_file}")

    # ========== CLASSIFICATION (Smart Maintenance Dataset) ==========
    print(f"\n" + "-" * 70)
    print(f"CLASSIFICATION: Smart Manufacturing Maintenance Dataset")
    print(f"-" * 70)

    print(f"Loading real classification data...")
    cls_df = pd.read_csv(CLASSIFICATION_DATA_PATH)
    print(f"  Shape: {cls_df.shape}")

    print(f"Generating synthetic classification data with {n_rows} rows...")
    synthetic_cls_df = generator.generate_synthetic_data(
        real_df=cls_df,
        n_rows=n_rows,
        target_column='Maintenance_Priority',
        exclude_columns=['Machine_ID'],  # Exclude ID columns
    )
    print(f"  Generated shape: {synthetic_cls_df.shape}")

    # Save synthetic classification data
    synthetic_cls_path = output_dir / f'synthetic_classification_{n_rows}_rows.csv'
    synthetic_cls_df.to_csv(synthetic_cls_path, index=False)
    print(f"✓ Saved synthetic classification data to {synthetic_cls_path}")

    # Run inference on synthetic classification data
    print(f"\nRunning classification inference...")
    classification_inferences = []

    cls_bundle_file = ARTIFACT_DIR / 'classification__Maintenance_Priority__RandomForestClassifier.joblib'
    if cls_bundle_file.exists():
        try:
            bundle = load_pretrained_bundle(cls_bundle_file)
            result = run_inference_on_synthetic_data(
                synthetic_cls_df,
                bundle,
                bundle_name=cls_bundle_file.name,
                problem_type='classification'
            )
            if result:
                classification_inferences.append(result)
                print(f"✓ {result['model_name']} inference complete")
                print(f"  Predictions sample: {result['predictions_sample']}")
                if 'prediction_distribution' in result:
                    print(f"  Prediction distribution: {result['prediction_distribution']}")
        except Exception as e:
            print(f"✗ Error during inference: {str(e)[:100]}")
    else:
        print(f"✗ Bundle not found: {cls_bundle_file}")

    # ========== SAVE RESULTS ==========
    print(f"\n" + "-" * 70)
    print(f"Saving Results")
    print(f"-" * 70)

    all_results = {
        'metadata': {
            'n_synthetic_rows': n_rows,
            'seed': seed,
            'timestamp': pd.Timestamp.now().isoformat(),
        },
        'regression': {
            'synthetic_data_file': str(synthetic_reg_path.relative_to(PROJECT_DIR)),
            'inferences': regression_inferences,
        },
        'classification': {
            'synthetic_data_file': str(synthetic_cls_path.relative_to(PROJECT_DIR)),
            'inferences': classification_inferences,
        },
    }

    results_path = output_dir / f'inference_results_{n_rows}_rows.json'
    with results_path.open('w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Saved inference results to {results_path}")

    # ========== SUMMARY ==========
    print(f"\n" + "=" * 70)
    print(f"SUMMARY")
    print(f"=" * 70)
    print(f"Synthetic regression data: {synthetic_reg_df.shape[0]} rows")
    print(f"Synthetic classification data: {synthetic_cls_df.shape[0]} rows")
    print(f"Regression models tested: {len(regression_inferences)}")
    print(f"Classification models tested: {len(classification_inferences)}")
    print(f"\nAll artifacts saved to: {output_dir}")
    print(f"=" * 70)


if __name__ == '__main__':
    main()
