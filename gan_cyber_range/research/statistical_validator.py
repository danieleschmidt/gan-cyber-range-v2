"""
Statistical validation module for cybersecurity research experiments.

This module provides comprehensive statistical testing and validation
to ensure research results meet academic publication standards.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, kruskal
from scipy.stats import shapiro, levene, bartlett
import warnings

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTest:
    """Results from a statistical test"""
    test_name: str
    test_statistic: float
    p_value: float
    critical_value: Optional[float]
    effect_size: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    interpretation: str
    assumptions_met: bool
    warnings: List[str]


@dataclass
class PowerAnalysis:
    """Power analysis results"""
    observed_power: float
    required_sample_size: int
    effect_size: float
    alpha: float
    beta: float
    interpretation: str


class StatisticalValidator:
    """Comprehensive statistical validation for research experiments"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.results_cache: Dict[str, Any] = {}
        
        logger.info(f"Initialized StatisticalValidator with α = {significance_level}")
    
    def validate_results(
        self,
        results: List[Any],  # ExperimentResult objects
        config: Any  # ExperimentConfig object
    ) -> Dict[str, Any]:
        """Perform comprehensive statistical validation of experimental results"""
        
        logger.info("Starting comprehensive statistical validation")
        
        validation_results = {
            'descriptive_statistics': self._calculate_descriptive_stats(results),
            'normality_tests': self._test_normality(results),
            'homogeneity_tests': self._test_homogeneity(results),
            'significance_tests': self._perform_significance_tests(results, config),
            'effect_sizes': self._calculate_effect_sizes(results, config),
            'power_analysis': self._perform_power_analysis(results, config),
            'multiple_comparisons': self._adjust_multiple_comparisons(results, config),
            'non_parametric_tests': self._perform_non_parametric_tests(results),
            'bootstrap_confidence_intervals': self._calculate_bootstrap_ci(results),
            'outlier_analysis': self._detect_outliers(results),
            'validation_summary': {}
        }
        
        # Generate summary
        validation_results['validation_summary'] = self._generate_validation_summary(
            validation_results, config
        )
        
        logger.info("Statistical validation completed")
        return validation_results
    
    def _calculate_descriptive_stats(self, results: List[Any]) -> Dict[str, Any]:
        """Calculate descriptive statistics for all metrics"""
        
        # Group results by method
        method_data = self._group_results_by_method(results)
        
        descriptive_stats = {}
        
        for method_name, method_results in method_data.items():
            method_stats = {}
            
            # Get all metrics for this method
            all_metrics = set()
            for result in method_results:
                all_metrics.update(result.metrics.keys())
            
            for metric in all_metrics:
                values = [r.metrics.get(metric, np.nan) for r in method_results 
                         if metric in r.metrics]
                
                if values:
                    method_stats[metric] = {
                        'count': len(values),
                        'mean': np.mean(values),
                        'std': np.std(values, ddof=1),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'q25': np.percentile(values, 25),
                        'q75': np.percentile(values, 75),
                        'iqr': np.percentile(values, 75) - np.percentile(values, 25),
                        'skewness': stats.skew(values),
                        'kurtosis': stats.kurtosis(values),
                        'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf
                    }
            
            descriptive_stats[method_name] = method_stats
        
        return descriptive_stats
    
    def _test_normality(self, results: List[Any]) -> Dict[str, Any]:
        """Test normality assumptions for all metrics"""
        
        method_data = self._group_results_by_method(results)
        normality_results = {}
        
        for method_name, method_results in method_data.items():
            method_normality = {}
            
            all_metrics = set()
            for result in method_results:
                all_metrics.update(result.metrics.keys())
            
            for metric in all_metrics:
                values = [r.metrics.get(metric, np.nan) for r in method_results 
                         if metric in r.metrics]
                
                if len(values) >= 3:  # Minimum for Shapiro-Wilk
                    stat, p_value = shapiro(values)
                    
                    method_normality[metric] = StatisticalTest(
                        test_name="Shapiro-Wilk",
                        test_statistic=stat,
                        p_value=p_value,
                        critical_value=None,
                        effect_size=None,
                        confidence_interval=None,
                        interpretation="Normal" if p_value > self.significance_level else "Non-normal",
                        assumptions_met=p_value > self.significance_level,
                        warnings=[] if len(values) <= 5000 else ["Large sample size - consider other tests"]
                    )
            
            normality_results[method_name] = method_normality
        
        return normality_results
    
    def _test_homogeneity(self, results: List[Any]) -> Dict[str, Any]:
        """Test homogeneity of variance assumptions"""
        
        method_data = self._group_results_by_method(results)
        homogeneity_results = {}
        
        # Get all metrics
        all_metrics = set()
        for method_results in method_data.values():
            for result in method_results:
                all_metrics.update(result.metrics.keys())
        
        for metric in all_metrics:
            # Collect data for this metric across all methods
            metric_groups = []
            method_names = []
            
            for method_name, method_results in method_data.items():
                values = [r.metrics.get(metric, np.nan) for r in method_results 
                         if metric in r.metrics]
                if len(values) >= 2:  # Need at least 2 values per group
                    metric_groups.append(values)
                    method_names.append(method_name)
            
            if len(metric_groups) >= 2:  # Need at least 2 groups
                # Levene's test (robust to non-normality)
                stat, p_value = levene(*metric_groups)
                
                homogeneity_results[metric] = StatisticalTest(
                    test_name="Levene",
                    test_statistic=stat,
                    p_value=p_value,
                    critical_value=None,
                    effect_size=None,
                    confidence_interval=None,
                    interpretation="Homogeneous" if p_value > self.significance_level else "Heterogeneous",
                    assumptions_met=p_value > self.significance_level,
                    warnings=[]
                )
        
        return homogeneity_results
    
    def _perform_significance_tests(
        self,
        results: List[Any],
        config: Any
    ) -> Dict[str, StatisticalTest]:
        """Perform appropriate significance tests"""
        
        method_data = self._group_results_by_method(results)
        significance_tests = {}
        
        # Get all metrics
        all_metrics = set()
        for method_results in method_data.values():
            for result in method_results:
                all_metrics.update(result.metrics.keys())
        
        method_names = list(method_data.keys())
        
        for metric in all_metrics:
            # Collect data for this metric
            metric_groups = []
            group_names = []
            
            for method_name in method_names:
                values = [r.metrics.get(metric, np.nan) for r in method_data[method_name] 
                         if metric in r.metrics]
                if len(values) >= 2:
                    metric_groups.append(values)
                    group_names.append(method_name)
            
            if len(metric_groups) == 2:
                # Two-group comparison
                significance_tests[f"{metric}_{group_names[0]}_vs_{group_names[1]}"] = \
                    self._two_group_test(metric_groups[0], metric_groups[1], metric)
                    
            elif len(metric_groups) > 2:
                # Multi-group comparison
                significance_tests[f"{metric}_multi_group"] = \
                    self._multi_group_test(metric_groups, group_names, metric)
        
        return significance_tests
    
    def _two_group_test(
        self,
        group1: List[float],
        group2: List[float],
        metric_name: str
    ) -> StatisticalTest:
        """Perform two-group statistical test"""
        
        # Check normality
        _, p1 = shapiro(group1) if len(group1) >= 3 else (0, 0)
        _, p2 = shapiro(group2) if len(group2) >= 3 else (0, 0)
        normal = p1 > self.significance_level and p2 > self.significance_level
        
        # Check equal variances
        if len(group1) >= 2 and len(group2) >= 2:
            _, p_var = levene(group1, group2)
            equal_var = p_var > self.significance_level
        else:
            equal_var = True
        
        warnings_list = []
        
        if normal and equal_var:
            # Parametric t-test
            stat, p_value = ttest_ind(group1, group2, equal_var=True)
            test_name = "Independent t-test"
        elif normal and not equal_var:
            # Welch's t-test
            stat, p_value = ttest_ind(group1, group2, equal_var=False)
            test_name = "Welch's t-test"
            warnings_list.append("Unequal variances - using Welch's correction")
        else:
            # Non-parametric Mann-Whitney U test
            stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            test_name = "Mann-Whitney U"
            warnings_list.append("Non-normal data - using non-parametric test")
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        
        if pooled_std > 0:
            cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        else:
            cohens_d = 0.0
        
        # Confidence interval for difference in means
        se_diff = pooled_std * np.sqrt(1/len(group1) + 1/len(group2))
        df = len(group1) + len(group2) - 2
        t_critical = stats.t.ppf(1 - self.significance_level/2, df)
        mean_diff = np.mean(group1) - np.mean(group2)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        return StatisticalTest(
            test_name=test_name,
            test_statistic=stat,
            p_value=p_value,
            critical_value=t_critical if 't-test' in test_name else None,
            effect_size=cohens_d,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=self._interpret_significance(p_value, cohens_d),
            assumptions_met=normal and equal_var,
            warnings=warnings_list
        )
    
    def _multi_group_test(
        self,
        groups: List[List[float]],
        group_names: List[str],
        metric_name: str
    ) -> StatisticalTest:
        """Perform multi-group statistical test"""
        
        # Check normality for all groups
        normal_groups = []
        for group in groups:
            if len(group) >= 3:
                _, p = shapiro(group)
                normal_groups.append(p > self.significance_level)
            else:
                normal_groups.append(True)  # Assume normal for small groups
        
        all_normal = all(normal_groups)
        
        # Check homogeneity of variance
        if all(len(group) >= 2 for group in groups):
            _, p_var = levene(*groups)
            equal_var = p_var > self.significance_level
        else:
            equal_var = True
        
        warnings_list = []
        
        if all_normal and equal_var:
            # One-way ANOVA
            stat, p_value = stats.f_oneway(*groups)
            test_name = "One-way ANOVA"
        else:
            # Non-parametric Kruskal-Wallis test
            stat, p_value = kruskal(*groups)
            test_name = "Kruskal-Wallis"
            if not all_normal:
                warnings_list.append("Non-normal data - using non-parametric test")
            if not equal_var:
                warnings_list.append("Unequal variances detected")
        
        # Calculate eta-squared (effect size for ANOVA)
        all_values = np.concatenate(groups)
        group_labels = np.concatenate([np.full(len(group), i) for i, group in enumerate(groups)])
        
        ss_between = sum(len(group) * (np.mean(group) - np.mean(all_values))**2 
                        for group in groups)
        ss_total = sum((x - np.mean(all_values))**2 for x in all_values)
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
        
        return StatisticalTest(
            test_name=test_name,
            test_statistic=stat,
            p_value=p_value,
            critical_value=None,
            effect_size=eta_squared,
            confidence_interval=None,
            interpretation=self._interpret_significance(p_value, eta_squared),
            assumptions_met=all_normal and equal_var,
            warnings=warnings_list
        )
    
    def _calculate_effect_sizes(
        self,
        results: List[Any],
        config: Any
    ) -> Dict[str, float]:
        """Calculate effect sizes for all comparisons"""
        
        method_data = self._group_results_by_method(results)
        effect_sizes = {}
        
        # Get all metrics
        all_metrics = set()
        for method_results in method_data.values():
            for result in method_results:
                all_metrics.update(result.metrics.keys())
        
        method_names = list(method_data.keys())
        
        # Calculate Cohen's d for all pairwise comparisons
        for metric in all_metrics:
            for i, method1 in enumerate(method_names):
                for j, method2 in enumerate(method_names[i+1:], i+1):
                    
                    values1 = [r.metrics.get(metric, np.nan) for r in method_data[method1] 
                              if metric in r.metrics]
                    values2 = [r.metrics.get(metric, np.nan) for r in method_data[method2] 
                              if metric in r.metrics]
                    
                    if len(values1) >= 2 and len(values2) >= 2:
                        # Calculate Cohen's d
                        pooled_std = np.sqrt(((len(values1) - 1) * np.var(values1, ddof=1) + 
                                             (len(values2) - 1) * np.var(values2, ddof=1)) / 
                                            (len(values1) + len(values2) - 2))
                        
                        if pooled_std > 0:
                            cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std
                            effect_sizes[f"{metric}_{method1}_vs_{method2}"] = abs(cohens_d)
        
        return effect_sizes
    
    def _perform_power_analysis(
        self,
        results: List[Any],
        config: Any
    ) -> Dict[str, PowerAnalysis]:
        """Perform statistical power analysis"""
        
        method_data = self._group_results_by_method(results)
        power_results = {}
        
        # Basic power analysis for two-group comparisons
        method_names = list(method_data.keys())
        
        if len(method_names) >= 2:
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:  # Common metrics
                # Find two methods with this metric
                methods_with_metric = []
                for method_name in method_names:
                    values = [r.metrics.get(metric, np.nan) for r in method_data[method_name] 
                             if metric in r.metrics]
                    if len(values) >= 2:
                        methods_with_metric.append((method_name, values))
                
                if len(methods_with_metric) >= 2:
                    method1, values1 = methods_with_metric[0]
                    method2, values2 = methods_with_metric[1]
                    
                    # Calculate observed effect size
                    pooled_std = np.sqrt(((len(values1) - 1) * np.var(values1, ddof=1) + 
                                         (len(values2) - 1) * np.var(values2, ddof=1)) / 
                                        (len(values1) + len(values2) - 2))
                    
                    if pooled_std > 0:
                        effect_size = abs(np.mean(values1) - np.mean(values2)) / pooled_std
                        
                        # Estimate power (simplified calculation)
                        n = min(len(values1), len(values2))
                        delta = effect_size * np.sqrt(n / 2)
                        
                        # Approximate power using normal distribution
                        z_alpha = stats.norm.ppf(1 - self.significance_level / 2)
                        z_beta = delta - z_alpha
                        power = stats.norm.cdf(z_beta)
                        
                        # Required sample size for 80% power
                        z_80 = stats.norm.ppf(0.8)
                        required_n = 2 * ((z_alpha + z_80) / effect_size) ** 2 if effect_size > 0 else float('inf')
                        
                        power_results[f"{metric}_{method1}_vs_{method2}"] = PowerAnalysis(
                            observed_power=max(0, min(1, power)),
                            required_sample_size=int(required_n) if required_n != float('inf') else 1000,
                            effect_size=effect_size,
                            alpha=self.significance_level,
                            beta=1 - power,
                            interpretation=self._interpret_power(power)
                        )
        
        return power_results
    
    def _adjust_multiple_comparisons(
        self,
        results: List[Any],
        config: Any
    ) -> Dict[str, Any]:
        """Adjust for multiple comparisons using various methods"""
        
        # Collect all p-values from significance tests
        significance_tests = self._perform_significance_tests(results, config)
        p_values = [test.p_value for test in significance_tests.values()]
        test_names = list(significance_tests.keys())
        
        if not p_values:
            return {'adjusted_tests': {}, 'correction_method': 'none'}
        
        # Bonferroni correction
        bonferroni_alpha = self.significance_level / len(p_values)
        bonferroni_significant = [p < bonferroni_alpha for p in p_values]
        
        # Benjamini-Hochberg (FDR) correction
        fdr_significant = self._benjamini_hochberg_correction(p_values, self.significance_level)
        
        adjusted_tests = {}
        for i, test_name in enumerate(test_names):
            adjusted_tests[test_name] = {
                'original_p': p_values[i],
                'bonferroni_significant': bonferroni_significant[i],
                'fdr_significant': fdr_significant[i],
                'bonferroni_alpha': bonferroni_alpha
            }
        
        return {
            'adjusted_tests': adjusted_tests,
            'bonferroni_alpha': bonferroni_alpha,
            'num_comparisons': len(p_values),
            'correction_method': 'bonferroni_and_fdr'
        }
    
    def _benjamini_hochberg_correction(
        self,
        p_values: List[float],
        alpha: float
    ) -> List[bool]:
        """Apply Benjamini-Hochberg FDR correction"""
        
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]
        
        # Find largest k such that P(k) <= (k/n) * alpha
        significant = np.zeros(n, dtype=bool)
        
        for i in range(n-1, -1, -1):
            if sorted_p[i] <= ((i + 1) / n) * alpha:
                significant[sorted_indices[:i+1]] = True
                break
        
        return significant.tolist()
    
    def _perform_non_parametric_tests(self, results: List[Any]) -> Dict[str, StatisticalTest]:
        """Perform non-parametric alternatives for robustness"""
        
        method_data = self._group_results_by_method(results)
        non_parametric_tests = {}
        
        # Get all metrics
        all_metrics = set()
        for method_results in method_data.values():
            for result in method_results:
                all_metrics.update(result.metrics.keys())
        
        method_names = list(method_data.keys())
        
        for metric in all_metrics:
            # Collect data for this metric
            metric_groups = []
            group_names = []
            
            for method_name in method_names:
                values = [r.metrics.get(metric, np.nan) for r in method_data[method_name] 
                         if metric in r.metrics]
                if len(values) >= 2:
                    metric_groups.append(values)
                    group_names.append(method_name)
            
            if len(metric_groups) == 2:
                # Mann-Whitney U test
                stat, p_value = mannwhitneyu(
                    metric_groups[0], metric_groups[1], alternative='two-sided'
                )
                
                non_parametric_tests[f"{metric}_{group_names[0]}_vs_{group_names[1]}_mw"] = \
                    StatisticalTest(
                        test_name="Mann-Whitney U",
                        test_statistic=stat,
                        p_value=p_value,
                        critical_value=None,
                        effect_size=None,
                        confidence_interval=None,
                        interpretation=self._interpret_significance(p_value, None),
                        assumptions_met=True,  # Non-parametric test
                        warnings=[]
                    )
                    
            elif len(metric_groups) > 2:
                # Kruskal-Wallis test
                stat, p_value = kruskal(*metric_groups)
                
                non_parametric_tests[f"{metric}_multi_group_kw"] = StatisticalTest(
                    test_name="Kruskal-Wallis",
                    test_statistic=stat,
                    p_value=p_value,
                    critical_value=None,
                    effect_size=None,
                    confidence_interval=None,
                    interpretation=self._interpret_significance(p_value, None),
                    assumptions_met=True,  # Non-parametric test
                    warnings=[]
                )
        
        return non_parametric_tests
    
    def _calculate_bootstrap_ci(
        self,
        results: List[Any],
        n_bootstrap: int = 1000
    ) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Calculate bootstrap confidence intervals"""
        
        method_data = self._group_results_by_method(results)
        bootstrap_cis = {}
        
        for method_name, method_results in method_data.items():
            method_cis = {}
            
            all_metrics = set()
            for result in method_results:
                all_metrics.update(result.metrics.keys())
            
            for metric in all_metrics:
                values = [r.metrics.get(metric, np.nan) for r in method_results 
                         if metric in r.metrics]
                
                if len(values) >= 5:  # Minimum for bootstrap
                    bootstrap_means = []
                    
                    for _ in range(n_bootstrap):
                        bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
                        bootstrap_means.append(np.mean(bootstrap_sample))
                    
                    # Calculate 95% confidence interval
                    ci_lower = np.percentile(bootstrap_means, 2.5)
                    ci_upper = np.percentile(bootstrap_means, 97.5)
                    
                    method_cis[metric] = (ci_lower, ci_upper)
            
            bootstrap_cis[method_name] = method_cis
        
        return bootstrap_cis
    
    def _detect_outliers(self, results: List[Any]) -> Dict[str, Dict[str, List[int]]]:
        """Detect outliers using multiple methods"""
        
        method_data = self._group_results_by_method(results)
        outlier_results = {}
        
        for method_name, method_results in method_data.items():
            method_outliers = {}
            
            all_metrics = set()
            for result in method_results:
                all_metrics.update(result.metrics.keys())
            
            for metric in all_metrics:
                values = [r.metrics.get(metric, np.nan) for r in method_results 
                         if metric in r.metrics]
                
                if len(values) >= 4:
                    # IQR method
                    q1, q3 = np.percentile(values, [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    iqr_outliers = [i for i, v in enumerate(values) 
                                   if v < lower_bound or v > upper_bound]
                    
                    # Z-score method
                    z_scores = np.abs(stats.zscore(values))
                    z_outliers = [i for i, z in enumerate(z_scores) if z > 3]
                    
                    method_outliers[metric] = {
                        'iqr_outliers': iqr_outliers,
                        'z_score_outliers': z_outliers,
                        'outlier_values': [values[i] for i in set(iqr_outliers + z_outliers)]
                    }
            
            outlier_results[method_name] = method_outliers
        
        return outlier_results
    
    def _generate_validation_summary(
        self,
        validation_results: Dict[str, Any],
        config: Any
    ) -> Dict[str, Any]:
        """Generate a summary of validation results"""
        
        summary = {
            'overall_validity': 'valid',
            'concerns': [],
            'recommendations': [],
            'statistical_power': 'adequate',
            'effect_sizes': 'meaningful'
        }
        
        # Check for normality violations
        normality_violations = 0
        normality_tests = validation_results.get('normality_tests', {})
        for method_tests in normality_tests.values():
            for test in method_tests.values():
                if not test.assumptions_met:
                    normality_violations += 1
        
        if normality_violations > 0:
            summary['concerns'].append(f"Normality violations detected ({normality_violations} cases)")
            summary['recommendations'].append("Consider non-parametric alternatives")
        
        # Check power analysis
        power_analyses = validation_results.get('power_analysis', {})
        low_power_tests = [name for name, analysis in power_analyses.items() 
                          if analysis.observed_power < 0.8]
        
        if low_power_tests:
            summary['statistical_power'] = 'low'
            summary['concerns'].append(f"Low statistical power detected in {len(low_power_tests)} tests")
            summary['recommendations'].append("Increase sample size for better power")
        
        # Check effect sizes
        effect_sizes = validation_results.get('effect_sizes', {})
        small_effects = [name for name, size in effect_sizes.items() if size < 0.3]
        
        if len(small_effects) > len(effect_sizes) * 0.5:  # More than half are small
            summary['effect_sizes'] = 'small'
            summary['concerns'].append("Many effect sizes are small")
            summary['recommendations'].append("Consider practical significance")
        
        # Overall assessment
        if len(summary['concerns']) == 0:
            summary['overall_validity'] = 'highly_valid'
        elif len(summary['concerns']) <= 2:
            summary['overall_validity'] = 'valid_with_minor_concerns'
        else:
            summary['overall_validity'] = 'valid_with_major_concerns'
        
        return summary
    
    def _group_results_by_method(self, results: List[Any]) -> Dict[str, List[Any]]:
        """Group results by method name"""
        method_data = {}
        for result in results:
            method_name = result.method_name
            if method_name not in method_data:
                method_data[method_name] = []
            method_data[method_name].append(result)
        return method_data
    
    def _interpret_significance(self, p_value: float, effect_size: Optional[float]) -> str:
        """Interpret statistical significance and effect size"""
        
        if p_value < 0.001:
            significance = "Highly significant"
        elif p_value < 0.01:
            significance = "Very significant"
        elif p_value < self.significance_level:
            significance = "Significant"
        else:
            significance = "Not significant"
        
        if effect_size is not None:
            if abs(effect_size) < 0.2:
                effect = "small effect"
            elif abs(effect_size) < 0.5:
                effect = "medium effect"
            elif abs(effect_size) < 0.8:
                effect = "large effect"
            else:
                effect = "very large effect"
            
            return f"{significance} with {effect}"
        else:
            return significance
    
    def _interpret_power(self, power: float) -> str:
        """Interpret statistical power"""
        if power >= 0.8:
            return "Adequate power"
        elif power >= 0.6:
            return "Moderate power"
        else:
            return "Low power - increase sample size"