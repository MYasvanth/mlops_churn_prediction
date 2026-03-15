# run_statistical_analysis.py
"""
Run statistical analysis on existing MLflow experiments.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.model_evaluator import ModelEvaluator
from src.utils.statistical_utils import StatisticalAnalyzer
import mlflow
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_statistical_analysis():
    """Run statistical analysis on existing MLflow experiments"""
    
    print("Running Statistical Analysis on Existing Models...")
    print("=" * 60)
    
    try:
        # Initialize components
        evaluator = ModelEvaluator("configs/model/unified_model_config.yaml")
        analyzer = StatisticalAnalyzer()
        
        # Set MLflow tracking
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Try to set experiment, create if doesn't exist
        try:
            mlflow.set_experiment("churn_prediction_unified")
        except:
            mlflow.create_experiment("churn_prediction_unified")
            mlflow.set_experiment("churn_prediction_unified")
        
        # Get model performance data from MLflow
        model_types = ['xgboost', 'lightgbm', 'random_forest', 'logistic_regression', 'svm']
        model_scores = {}
        model_run_counts = {}
        
        print("Collecting model performance data from MLflow...")
        
        for model_type in model_types:
            try:
                runs = mlflow.search_runs(
                    filter_string=f"params.model_type = '{model_type}'",
                    order_by=["start_time DESC"],
                    max_results=1  # Limit to 1 most recent run
                )
                
                if not runs.empty and 'metrics.accuracy' in runs.columns:
                    scores = runs['metrics.accuracy'].dropna().tolist()[:1]  # Ensure max 1
                    if scores:
                        model_scores[model_type] = scores
                        model_run_counts[model_type] = len(scores)
                        print(f"Found {len(scores)} most recent run for {model_type}")
                    else:
                        print(f"No accuracy scores found for {model_type}")
                else:
                    print(f"No runs found for {model_type}")
                    
            except Exception as e:
                print(f"Error collecting data for {model_type}: {str(e)}")
        
        if not model_scores:
            print("No model data found in MLflow. Please run some experiments first.")
            return
        
        print(f"\nFound data for {len(model_scores)} model types")
        print("Model run counts:", model_run_counts)
        
        # Statistical comparison
        print("\nPerforming statistical comparisons...")
        comparison_results = {}
        model_names = list(model_scores.keys())
        
        for i, model_a in enumerate(model_names):
            for model_b in model_names[i+1:]:
                if model_a in model_scores and model_b in model_scores:
                    print(f"Comparing {model_a} vs {model_b}...")
                    
                    scores_a = model_scores[model_a]
                    scores_b = model_scores[model_b]
                    
                    # Check if we have enough data for statistical tests
                    min_len = min(len(scores_a), len(scores_b))
                    
                    # With only 1 run per model, use descriptive comparison instead of statistical tests
                    if min_len < 2:
                        print(f"Using descriptive comparison for {model_a} vs {model_b} (1 run each)")
                        
                        # Simple descriptive comparison
                        score_a = scores_a[0]
                        score_b = scores_b[0]
                        
                        comparison_key = f"{model_a}_vs_{model_b}"
                        comparison_results[comparison_key] = {
                            'model_a': model_a,
                            'model_b': model_b,
                            'score_a': score_a,
                            'score_b': score_b,
                            'difference': score_a - score_b,
                            'better_model': model_a if score_a > score_b else model_b,
                            'improvement_pct': abs(score_a - score_b) / max(score_a, score_b) * 100,
                            'practically_significant': abs(score_a - score_b) >= 0.01,
                            'note': 'Descriptive comparison only - insufficient data for statistical tests'
                        }
                        continue
                    
                    # If we have multiple runs, proceed with statistical tests
                    scores_a = scores_a[:min_len]
                    scores_b = scores_b[:min_len]
                    
                    comparison_key = f"{model_a}_vs_{model_b}"
                    
                    # Perform comprehensive comparison
                    comparison_results[comparison_key] = analyzer.comprehensive_comparison(
                        scores_a, scores_b, model_a, model_b
                    )
        
        # Apply multiple comparison correction if we have comparisons
        if comparison_results:
            print("\nApplying multiple comparison correction...")
            
            # Extract p-values for correction
            p_values = []
            comparison_keys = []
            
            for comp_key, results in comparison_results.items():
                if 'statistical_tests' in results and 'paired_ttest' in results['statistical_tests']:
                    ttest_result = results['statistical_tests']['paired_ttest']
                    if 'p_value' in ttest_result:
                        p_values.append(ttest_result['p_value'])
                        comparison_keys.append(comp_key)
            
            if p_values:
                correction_results = analyzer.bonferroni_correction(p_values)
                
                # Add correction results to each comparison
                for i, comp_key in enumerate(comparison_keys):
                    comparison_results[comp_key]['multiple_comparison_correction'] = {
                        'original_p_value': p_values[i],
                        'corrected_p_value': correction_results['corrected_p_values'][i],
                        'significant_after_correction': correction_results['significant_after_correction'][i],
                        'corrected_alpha': correction_results['corrected_alpha']
                    }
        
        # Save results
        results_dir = Path("reports/statistical_analysis")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_results = {
            'analysis_timestamp': timestamp,
            'model_run_counts': model_run_counts,
            'statistical_comparisons': comparison_results,
            'analysis_summary': generate_analysis_summary(comparison_results)
        }
        
        results_file = results_dir / f"statistical_analysis_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, cls=NumpyEncoder)
        
        # Save summary report
        summary_report = generate_summary_report(detailed_results)
        summary_file = results_dir / f"statistical_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(summary_report)
        
        # Print summary
        print("\n" + "=" * 60)
        print("STATISTICAL ANALYSIS SUMMARY")
        print("=" * 60)
        print(summary_report)
        
        print(f"\nDetailed results saved to:")
        print(f"   {results_file}")
        print(f"   {summary_file}")
        print("\nStatistical analysis complete!")
        
    except Exception as e:
        logger.error(f"Statistical analysis failed: {str(e)}")
        print(f"Analysis failed: {str(e)}")

def generate_analysis_summary(comparison_results):
    """Generate analysis summary from comparison results"""
    summary = {
        'total_comparisons': len(comparison_results),
        'practically_significant': 0,
        'best_performing_models': [],
        'recommendations': []
    }
    
    model_scores = {}
    
    for comp_key, results in comparison_results.items():
        if 'note' in results:  # Descriptive comparison
            if results.get('practically_significant', False):
                summary['practically_significant'] += 1
            
            # Track model scores
            model_a = results.get('model_a')
            model_b = results.get('model_b')
            score_a = results.get('score_a')
            score_b = results.get('score_b')
            
            if model_a and score_a is not None:
                model_scores[model_a] = score_a
            if model_b and score_b is not None:
                model_scores[model_b] = score_b
            
            # Add recommendation
            better_model = results.get('better_model')
            if better_model:
                summary['recommendations'].append({
                    'comparison': comp_key,
                    'recommendation': f"{better_model} performs better"
                })
    
    # Find best performing model
    if model_scores:
        best_model = max(model_scores, key=model_scores.get)
        summary['best_performing_models'] = [best_model]
    
    return summary

def generate_summary_report(detailed_results):
    """Generate human-readable summary report"""
    summary = detailed_results['analysis_summary']
    comparisons = detailed_results['statistical_comparisons']
    
    report_lines = [
        f"Analysis Date: {detailed_results['analysis_timestamp']}",
        f"Models Analyzed: {len(detailed_results['model_run_counts'])}",
        f"Total Comparisons: {summary['total_comparisons']}",
        "",
        "MODEL RUN COUNTS:",
    ]
    
    for model, count in detailed_results['model_run_counts'].items():
        report_lines.append(f"  - {model}: {count} run(s)")
    
    report_lines.extend([
        "",
        "COMPARISON RESULTS (1 run per model - descriptive analysis):",
        "",
        "DETAILED COMPARISONS:"
    ])
    
    for comp_key, results in comparisons.items():
        if 'note' in results:  # Descriptive comparison
            model_a = results.get('model_a', 'Model A')
            model_b = results.get('model_b', 'Model B')
            score_a = results.get('score_a', 0)
            score_b = results.get('score_b', 0)
            difference = results.get('difference', 0)
            better_model = results.get('better_model', 'Unknown')
            improvement_pct = results.get('improvement_pct', 0)
            practically_significant = results.get('practically_significant', False)
            
            report_lines.extend([
                f"\n{model_a} vs {model_b}:",
                f"   {model_a} Performance: {score_a:.4f}",
                f"   {model_b} Performance: {score_b:.4f}",
                f"   Difference: {difference:.4f} ({improvement_pct:.2f}% improvement)",
                f"   Better Model: {better_model}",
                f"   Practically Significant (>1%): {'Yes' if practically_significant else 'No'}",
                f"   Note: {results.get('note', '')}"
            ])
        else:  # Statistical comparison (if available)
            model_a = results.get('model_a', 'Model A')
            model_b = results.get('model_b', 'Model B')
            
            # Get key statistics
            desc_stats = results.get('descriptive_stats', {})
            mean_a = desc_stats.get('model_a', {}).get('mean', 0)
            mean_b = desc_stats.get('model_b', {}).get('mean', 0)
            
            ttest = results.get('statistical_tests', {}).get('paired_ttest', {})
            p_value = ttest.get('p_value', 1.0)
            significant = ttest.get('significant', False)
            
            effect_size = results.get('effect_size', {})
            cohens_d = effect_size.get('cohens_d', 0)
            effect_interp = effect_size.get('effect_size_interpretation', 'unknown')
            
            report_lines.extend([
                f"\n{model_a} vs {model_b}:",
                f"   Mean Performance: {mean_a:.4f} vs {mean_b:.4f}",
                f"   P-value: {p_value:.4f} ({'Significant' if significant else 'Not significant'})",
                f"   Effect Size: {cohens_d:.3f} ({effect_interp})",
                f"   Recommendation: {results.get('recommendation', 'No recommendation')}"
            ])
            
            # Add multiple comparison correction if available
            if 'multiple_comparison_correction' in results:
                mcc = results['multiple_comparison_correction']
                report_lines.append(f"   After Bonferroni correction: {'Significant' if mcc['significant_after_correction'] else 'Not significant'}")
    
    return "\n".join(report_lines)

if __name__ == "__main__":
    run_statistical_analysis()