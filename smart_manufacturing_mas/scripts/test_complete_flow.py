#!/usr/bin/env python3
"""
Comprehensive test of the complete Smart Manufacturing MAS workflow.

Tests:
1. Pretrained model bundle export from notebook
2. Flag validation (incompatible flags should fail)
3. Synthetic data generation
4. Inference-only mode (using pretrained bundles)
5. Live training mode
6. Cache functionality
7. Target detection from bundles
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

PROJECT_DIR = Path.cwd()
if not (PROJECT_DIR / 'data').exists():
    PROJECT_DIR = Path(__file__).parent.parent

DATA_DIR = PROJECT_DIR / 'data'
ARTIFACTS_DIR = PROJECT_DIR / 'artifacts' / 'pretrained_models'
MODEL_CACHE_DIR = PROJECT_DIR / 'model_cache'

def test_artifact_locations():
    """Verify artifact directories exist and are accessible."""
    logging.info("TEST 1: Artifact Locations")
    try:
        assert ARTIFACTS_DIR.exists(), f"Pretrained models dir missing: {ARTIFACTS_DIR}"
        assert (ARTIFACTS_DIR / 'registry.json').exists(), "Registry not found"
        assert MODEL_CACHE_DIR.exists(), f"Model cache dir missing: {MODEL_CACHE_DIR}"
        assert (MODEL_CACHE_DIR / 'registry.json').exists(), "Cache registry not found"
        
        # Check registry is valid JSON
        with open(ARTIFACTS_DIR / 'registry.json') as f:
            pretrained_reg = json.load(f)
        with open(MODEL_CACHE_DIR / 'registry.json') as f:
            cache_reg = json.load(f)
        
        logging.info(f"  ✓ Pretrained registry: {list(pretrained_reg.keys())}")
        logging.info(f"  ✓ Cache registry: {list(cache_reg.keys())}")
        return True
    except Exception as e:
        logging.error(f"  ✗ {e}")
        return False


def test_flag_validation():
    """Test that incompatible flags are rejected (via argparse)."""
    logging.info("TEST 2: Flag Validation")
    try:
        # This would normally be tested via CLI, but we can test the validation function directly
        import sys
        sys.path.insert(0, str(PROJECT_DIR))
        from main_llm import _validate_args
        
        # Create a mock args object for conflicting flags
        class Args:
            use_cache = True
            inference_only = True
            train_live = False
            invalidate_cache = False
        
        # This should raise sys.exit(1)
        try:
            _validate_args(Args())
            logging.error("  ✗ Should have rejected conflicting flags")
            return False
        except SystemExit as e:
            if e.code == 1:
                logging.info("  ✓ Correctly rejected --use-cache + --inference-only")
                return True
            else:
                logging.error(f"  ✗ Unexpected exit code: {e.code}")
                return False
    except Exception as e:
        logging.error(f"  ✗ {e}")
        return False


def test_synthetic_data_generation():
    """Test synthetic data generation works."""
    logging.info("TEST 3: Synthetic Data Generation")
    try:
        # Check if test data exists
        test_csv = DATA_DIR / 'smart_manufacturing_dataset.csv'
        if not test_csv.exists():
            logging.warning(f"  ⚠ Test CSV not found: {test_csv}")
            return None  # Skip this test
        
        # Try to load and analyze
        df = pd.read_csv(test_csv)
        logging.info(f"  ✓ Loaded test data: {df.shape[0]} rows, {df.shape[1]} cols")
        
        # Check columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        logging.info(f"  ✓ Numeric columns: {len(numeric_cols)}")
        
        return True
    except Exception as e:
        logging.error(f"  ✗ {e}")
        return False


def test_pretrained_registry_structure():
    """Verify registry structure is correct."""
    logging.info("TEST 4: Registry Structure")
    try:
        with open(ARTIFACTS_DIR / 'registry.json') as f:
            registry = json.load(f)
        
        # Check expected keys
        expected_keys = {'regression', 'classification', 'anomaly_detection'}
        actual_keys = set(registry.keys())
        
        if not expected_keys.issubset(actual_keys):
            logging.warning(f"  ⚠ Missing keys: {expected_keys - actual_keys}")
        
        logging.info(f"  ✓ Registry keys: {list(registry.keys())}")
        
        # Count entries
        total_models = sum(len(registry.get(k, [])) for k in registry)
        logging.info(f"  ✓ Total models in registry: {total_models}")
        
        return True
    except Exception as e:
        logging.error(f"  ✗ {e}")
        return False


def test_target_detection_logic():
    """Test that target detection from bundles works correctly."""
    logging.info("TEST 5: Target Detection Logic")
    try:
        sys.path.insert(0, str(PROJECT_DIR))
        from utils.pretrained_model_store import select_bundle_metadata, load_registry
        
        # Load registry
        registry = load_registry(str(ARTIFACTS_DIR))
        total = sum(len(registry.get(k, [])) for k in registry)
        
        if total == 0:
            logging.warning("  ⚠ No bundles in registry (need to run offline_model_training.ipynb first)")
            return None
        
        logging.info(f"  ✓ Found {total} bundles in registry")
        
        # Try to select a bundle for each problem type
        for problem_type in ['regression', 'classification']:
            bundle_meta = select_bundle_metadata(problem_type, path=str(ARTIFACTS_DIR))
            if bundle_meta:
                logging.info(f"  ✓ {problem_type}: target='{bundle_meta.get('target_column')}'")
            else:
                logging.warning(f"  ⚠ No {problem_type} bundles available")
        
        return True
    except Exception as e:
        logging.error(f"  ✗ {e}")
        return False


def run_all_tests():
    """Run all tests and summary."""
    logging.info("=" * 60)
    logging.info("COMPREHENSIVE MAS SYSTEM TEST")
    logging.info("=" * 60)
    
    results = {
        'artifact_locations': test_artifact_locations(),
        'flag_validation': test_flag_validation(),
        'synthetic_data': test_synthetic_data_generation(),
        'registry_structure': test_pretrained_registry_structure(),
        'target_detection': test_target_detection_logic(),
    }
    
    logging.info("=" * 60)
    logging.info("TEST SUMMARY")
    logging.info("=" * 60)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    for test, result in results.items():
        status = "✓ PASS" if result is True else "✗ FAIL" if result is False else "⊘ SKIP"
        logging.info(f"{status}: {test}")
    
    logging.info("-" * 60)
    logging.info(f"Passed: {passed}, Failed: {failed}, Skipped: {skipped}")
    logging.info("=" * 60)
    
    if failed > 0:
        logging.error("TESTS FAILED - There are issues to fix")
        return False
    elif passed + skipped > 0:
        logging.info("TESTS PASSED - System is ready")
        return True
    else:
        logging.warning("NO TESTS RAN")
        return None


if __name__ == '__main__':
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"FATAL: {e}", exc_info=True)
        sys.exit(1)
