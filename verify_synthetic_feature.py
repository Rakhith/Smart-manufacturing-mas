#!/usr/bin/env python3
"""
Verification script for Synthetic Data Analysis Feature
Tests imports, basic functionality, and data structures
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent / "smart_manufacturing_mas"
sys.path.insert(0, str(project_root))

def check_imports():
    """Verify all required imports are available"""
    print("🔍 Checking imports...")
    
    required_modules = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'scipy': 'SciPy',
        'scipy.stats': 'SciPy Stats',
        'sklearn': 'Scikit-learn',
        'sklearn.metrics': 'Scikit-learn Metrics',
    }
    
    all_ok = True
    for module, name in required_modules.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            all_ok = False
    
    return all_ok

def check_new_modules():
    """Verify new modules can be imported"""
    print("\n🔍 Checking new modules...")
    
    try:
        from utils.synthetic_quality_analyzer import SyntheticQualityAnalyzer
        print("  ✓ SyntheticQualityAnalyzer")
    except Exception as e:
        print(f"  ✗ SyntheticQualityAnalyzer: {e}")
        return False
    
    try:
        from utils.prediction_analyzer import PredictionAnalyzer
        print("  ✓ PredictionAnalyzer")
    except Exception as e:
        print(f"  ✗ PredictionAnalyzer: {e}")
        return False
    
    return True

def test_synthetic_quality_analyzer():
    """Test SyntheticQualityAnalyzer functionality"""
    print("\n🧪 Testing SyntheticQualityAnalyzer...")
    
    try:
        import numpy as np
        import pandas as pd
        from utils.synthetic_quality_analyzer import SyntheticQualityAnalyzer
        
        # Create sample data
        np.random.seed(42)
        original_df = pd.DataFrame({
            'feature1': np.random.normal(50, 10, 100),
            'feature2': np.random.normal(100, 20, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
        })
        
        # Create similar synthetic data
        synthetic_df = pd.DataFrame({
            'feature1': np.random.normal(48, 12, 100),
            'feature2': np.random.normal(102, 18, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
        })
        
        # Test analyzer
        analyzer = SyntheticQualityAnalyzer(original_df, synthetic_df)
        
        # Test methods
        numeric_comp = analyzer.compare_numeric_distributions()
        print(f"  ✓ Numeric comparison: {len(numeric_comp)} columns analyzed")
        
        categorical_comp = analyzer.compare_categorical_distributions()
        print(f"  ✓ Categorical comparison: {len(categorical_comp)} columns analyzed")
        
        missing_comp = analyzer.compare_missing_values()
        print(f"  ✓ Missing value comparison: {len(missing_comp)} columns checked")
        
        outlier_comp = analyzer.detect_outliers_comparison()
        print(f"  ✓ Outlier comparison: {len(outlier_comp)} columns checked")
        
        quality_score = analyzer.calculate_overall_quality_score()
        print(f"  ✓ Overall quality score: {quality_score.get('overall_quality_score'):.1f}/100")
        
        summary = analyzer.get_summary_for_display()
        print(f"  ✓ Display summary quality: {summary['quality_level']}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prediction_analyzer():
    """Test PredictionAnalyzer functionality"""
    print("\n🧪 Testing PredictionAnalyzer...")
    
    try:
        import numpy as np
        from utils.prediction_analyzer import PredictionAnalyzer
        
        # Test classification
        predictions = np.random.choice([0, 1, 2], 100)
        actuals = np.random.choice([0, 1, 2], 100)
        
        analyzer = PredictionAnalyzer(
            predictions=predictions,
            actual_values=actuals,
            problem_type="classification"
        )
        
        analysis = analyzer.analyze_classification_predictions()
        print(f"  ✓ Classification analysis: {analysis['total_predictions']} predictions")
        
        recommendations = analyzer.generate_recommendations()
        print(f"  ✓ Recommendations generated: {len(recommendations)} items")
        
        summary = analyzer.get_summary()
        print(f"  ✓ Summary created with accuracy: {summary['analysis'].get('accuracy', 'N/A')}")
        
        # Test regression
        predictions_reg = np.random.normal(100, 20, 100)
        actuals_reg = predictions_reg + np.random.normal(0, 5, 100)
        
        analyzer_reg = PredictionAnalyzer(
            predictions=predictions_reg,
            actual_values=actuals_reg,
            problem_type="regression"
        )
        
        analysis_reg = analyzer_reg.analyze_regression_predictions()
        print(f"  ✓ Regression analysis: RMSE = {analysis_reg.get('rmse', 'N/A'):.2f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_run_manager_integration():
    """Check if run_manager imports are correct"""
    print("\n🔍 Checking run_manager integration...")
    
    try:
        from webapp import run_manager
        
        # Check if imports are in the module
        import inspect
        source = inspect.getsource(run_manager)
        
        if 'SyntheticQualityAnalyzer' in source:
            print("  ✓ SyntheticQualityAnalyzer imported in run_manager")
        else:
            print("  ✗ SyntheticQualityAnalyzer not imported in run_manager")
            return False
        
        if 'PredictionAnalyzer' in source:
            print("  ✓ PredictionAnalyzer imported in run_manager")
        else:
            print("  ✗ PredictionAnalyzer not imported in run_manager")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ Error checking run_manager: {e}")
        return False

def check_frontend_functions():
    """Check if frontend functions are present"""
    print("\n🔍 Checking frontend functions...")
    
    try:
        app_js_path = project_root / "webapp" / "static" / "app.js"
        with open(app_js_path) as f:
            content = f.read()
        
        functions = [
            'renderQualityScoreBadge',
            'renderDataQualitySection',
            'renderPredictionAnalysisSection',
            'renderSyntheticDashboard',
        ]
        
        all_present = True
        for func in functions:
            if f'function {func}' in content:
                print(f"  ✓ {func} defined")
            else:
                print(f"  ✗ {func} not found")
                all_present = False
        
        return all_present
    except Exception as e:
        print(f"  ✗ Error checking frontend: {e}")
        return False

def check_documentation():
    """Check if documentation files exist"""
    print("\n📚 Checking documentation...")
    
    docs = [
        'IMPLEMENTATION_SUMMARY.md',
        'SYNTHETIC_DATA_ANALYSIS_GUIDE.md',
        'SYNTHETIC_DATA_QUICK_START.md',
    ]
    
    project_path = project_root.parent
    all_present = True
    for doc in docs:
        doc_path = project_path / doc
        if doc_path.exists():
            size = doc_path.stat().st_size
            print(f"  ✓ {doc} ({size} bytes)")
        else:
            print(f"  ✗ {doc} not found")
            all_present = False
    
    return all_present

def main():
    """Run all checks"""
    print("=" * 60)
    print("Synthetic Data Analysis Feature - Verification")
    print("=" * 60)
    
    results = {
        "Imports": check_imports(),
        "New Modules": check_new_modules(),
        "SyntheticQualityAnalyzer": test_synthetic_quality_analyzer(),
        "PredictionAnalyzer": test_prediction_analyzer(),
        "Run Manager Integration": check_run_manager_integration(),
        "Frontend Functions": check_frontend_functions(),
        "Documentation": check_documentation(),
    }
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All checks passed! Feature is ready to use.")
        return 0
    else:
        print("\n⚠️ Some checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
