# statistical_utils.py
"""
Statistical analysis utilities for ML model comparison.
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """Statistical analysis utilities for ML model comparison"""
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize StatisticalAnalyzer.
        
        Args:
            alpha (float): Significance level for statistical tests
        """
        self.alpha = alpha
    
    def paired_ttest(self, scores_a: List[float], scores_b: List[float]) -> Dict[str, Any]:
        """
        Perform paired t-test between two model performances.
        
        Args:
            scores_a (List[float]): Performance scores for model A
            scores_b (List[float]): Performance scores for model B
            
        Returns:
            Dict[str, Any]: T-test results
        """
        try:
            if len(scores_a) != len(scores_b):
                raise ValueError("Score arrays must have same length")
            
            if len(scores_a) < 2:
                raise ValueError("Need at least 2 samples for t-test")
            
            t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
            
            return {
                'test_type': 'paired_ttest',
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'alpha': self.alpha,
                'degrees_of_freedom': len(scores_a) - 1,
                'interpretation': self._interpret_ttest(t_stat, p_value)
            }
        except Exception as e:
            logger.error(f"Paired t-test failed: {str(e)}")
            return {'error': str(e)}
    
    def wilcoxon_test(self, scores_a: List[float], scores_b: List[float]) -> Dict[str, Any]:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to t-test).
        
        Args:
            scores_a (List[float]): Performance scores for model A
            scores_b (List[float]): Performance scores for model B
            
        Returns:
            Dict[str, Any]: Wilcoxon test results
        """
        try:
            if len(scores_a) != len(scores_b):
                raise ValueError("Score arrays must have same length")
            
            if len(scores_a) < 6:
                logger.warning("Wilcoxon test may be unreliable with < 6 samples")
            
            statistic, p_value = stats.wilcoxon(scores_a, scores_b, alternative='two-sided')
            
            return {
                'test_type': 'wilcoxon_signed_rank',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'alpha': self.alpha,
                'interpretation': self._interpret_wilcoxon(statistic, p_value)
            }
        except Exception as e:
            logger.error(f"Wilcoxon test failed: {str(e)}")
            return {'error': str(e)}
    
    def cohens_d(self, scores_a: List[float], scores_b: List[float]) -> Dict[str, Any]:
        """
        Calculate Cohen's d effect size.
        
        Args:
            scores_a (List[float]): Performance scores for model A
            scores_b (List[float]): Performance scores for model B
            
        Returns:
            Dict[str, Any]: Effect size results
        """
        try:
            mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
            std_a, std_b = np.std(scores_a, ddof=1), np.std(scores_b, ddof=1)
            
            # Pooled standard deviation
            n_a, n_b = len(scores_a), len(scores_b)
            pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
            
            if pooled_std == 0:
                cohens_d = 0.0
            else:
                cohens_d = (mean_a - mean_b) / pooled_std
            
            return {
                'cohens_d': float(cohens_d),
                'effect_size_interpretation': self._interpret_cohens_d(cohens_d),
                'mean_difference': float(mean_a - mean_b),
                'pooled_std': float(pooled_std),
                'mean_a': float(mean_a),
                'mean_b': float(mean_b)
            }
        except Exception as e:
            logger.error(f"Cohen's d calculation failed: {str(e)}")
            return {'error': str(e)}
    
    def bonferroni_correction(self, p_values: List[float]) -> Dict[str, Any]:
        """
        Apply Bonferroni correction for multiple comparisons.
        
        Args:
            p_values (List[float]): List of p-values to correct
            
        Returns:
            Dict[str, Any]: Correction results
        """
        try:
            if not p_values:
                return {'error': 'No p-values provided'}
            
            corrected_alpha = self.alpha / len(p_values)
            corrected_p_values = [min(p * len(p_values), 1.0) for p in p_values]
            
            return {
                'method': 'bonferroni',
                'original_alpha': self.alpha,
                'corrected_alpha': corrected_alpha,
                'n_comparisons': len(p_values),
                'original_p_values': p_values,
                'corrected_p_values': corrected_p_values,
                'significant_before_correction': [p < self.alpha for p in p_values],
                'significant_after_correction': [p < corrected_alpha for p in p_values]
            }
        except Exception as e:
            logger.error(f"Bonferroni correction failed: {str(e)}")
            return {'error': str(e)}
    
    def practical_significance(self, scores_a: List[float], scores_b: List[float], 
                             min_difference: float = 0.01) -> Dict[str, Any]:
        """
        Assess practical significance beyond statistical significance.
        
        Args:
            scores_a (List[float]): Performance scores for model A
            scores_b (List[float]): Performance scores for model B
            min_difference (float): Minimum difference considered practically significant
            
        Returns:
            Dict[str, Any]: Practical significance results
        """
        try:
            mean_a = np.mean(scores_a)
            mean_b = np.mean(scores_b)
            mean_diff = abs(mean_a - mean_b)
            
            return {
                'mean_difference': float(mean_diff),
                'min_practical_difference': min_difference,
                'practically_significant': mean_diff >= min_difference,
                'relative_improvement': float(mean_diff / max(mean_a, mean_b)) if max(mean_a, mean_b) > 0 else 0.0,
                'interpretation': f"{'Practically significant' if mean_diff >= min_difference else 'Not practically significant'}"
            }
        except Exception as e:
            logger.error(f"Practical significance calculation failed: {str(e)}")
            return {'error': str(e)}
    
    def _interpret_ttest(self, t_stat: float, p_value: float) -> str:
        """Interpret t-test results."""
        if p_value < 0.001:
            significance = "highly significant"
        elif p_value < 0.01:
            significance = "very significant"
        elif p_value < 0.05:
            significance = "significant"
        else:
            significance = "not significant"
        
        direction = "Model A performs better" if t_stat > 0 else "Model B performs better"
        return f"Difference is {significance} (p={p_value:.4f}). {direction}."
    
    def _interpret_wilcoxon(self, statistic: float, p_value: float) -> str:
        """Interpret Wilcoxon test results."""
        significance = "significant" if p_value < self.alpha else "not significant"
        return f"Non-parametric test shows {significance} difference (p={p_value:.4f})"
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible effect"
        elif abs_d < 0.5:
            return "small effect"
        elif abs_d < 0.8:
            return "medium effect"
        else:
            return "large effect"
    
    def comprehensive_comparison(self, scores_a: List[float], scores_b: List[float], 
                               model_a_name: str = "Model A", model_b_name: str = "Model B") -> Dict[str, Any]:
        """
        Perform comprehensive statistical comparison between two models.
        
        Args:
            scores_a (List[float]): Performance scores for model A
            scores_b (List[float]): Performance scores for model B
            model_a_name (str): Name of model A
            model_b_name (str): Name of model B
            
        Returns:
            Dict[str, Any]: Comprehensive comparison results
        """
        try:
            results = {
                'model_a': model_a_name,
                'model_b': model_b_name,
                'sample_sizes': {'model_a': len(scores_a), 'model_b': len(scores_b)},
                'descriptive_stats': {
                    'model_a': {
                        'mean': float(np.mean(scores_a)),
                        'std': float(np.std(scores_a, ddof=1)),
                        'median': float(np.median(scores_a)),
                        'min': float(np.min(scores_a)),
                        'max': float(np.max(scores_a))
                    },
                    'model_b': {
                        'mean': float(np.mean(scores_b)),
                        'std': float(np.std(scores_b, ddof=1)),
                        'median': float(np.median(scores_b)),
                        'min': float(np.min(scores_b)),
                        'max': float(np.max(scores_b))
                    }
                },
                'statistical_tests': {
                    'paired_ttest': self.paired_ttest(scores_a, scores_b),
                    'wilcoxon_test': self.wilcoxon_test(scores_a, scores_b)
                },
                'effect_size': self.cohens_d(scores_a, scores_b),
                'practical_significance': self.practical_significance(scores_a, scores_b)
            }
            
            # Overall recommendation
            ttest_significant = results['statistical_tests']['paired_ttest'].get('significant', False)
            practically_significant = results['practical_significance'].get('practically_significant', False)
            
            if ttest_significant and practically_significant:
                recommendation = f"{model_a_name} is statistically and practically better than {model_b_name}"
            elif ttest_significant:
                recommendation = f"{model_a_name} is statistically better than {model_b_name} but difference may not be practically significant"
            elif practically_significant:
                recommendation = f"{model_a_name} shows practically significant improvement over {model_b_name} but not statistically significant"
            else:
                recommendation = f"No significant difference between {model_a_name} and {model_b_name}"
            
            results['recommendation'] = recommendation
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive comparison failed: {str(e)}")
            return {'error': str(e)}