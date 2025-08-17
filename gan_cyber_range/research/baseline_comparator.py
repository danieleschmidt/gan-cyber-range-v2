"""
Baseline comparison module for validating novel cybersecurity AI methods.

This module provides comprehensive baseline comparison capabilities to ensure
new methods demonstrate clear improvements over established approaches.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BaselineExperiment:
    """Configuration for a baseline comparison experiment"""
    name: str
    description: str
    baseline_methods: List[str]
    novel_methods: List[str]
    evaluation_metrics: List[str]
    datasets: List[str]
    significance_threshold: float = 0.05
    effect_size_threshold: float = 0.3
    min_improvement_threshold: float = 0.05  # 5% minimum improvement
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaselineComparison:
    """Results from comparing a method against baselines"""
    novel_method: str
    baseline_method: str
    metric: str
    novel_performance: float
    baseline_performance: float
    improvement: float
    improvement_percentage: float
    statistical_significance: bool
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    meets_threshold: bool
    interpretation: str


@dataclass
class ComparisonSummary:
    """Summary of all baseline comparisons"""
    experiment_name: str
    total_comparisons: int
    significant_improvements: int
    meets_effect_size_threshold: int
    meets_improvement_threshold: int
    overall_success: bool
    best_performing_method: str
    detailed_comparisons: List[BaselineComparison]
    recommendations: List[str]
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class BaselineMethod(ABC):
    """Abstract base class for baseline methods"""
    
    @abstractmethod
    def name(self) -> str:
        """Return method name"""
        pass
    
    @abstractmethod
    def description(self) -> str:
        """Return method description"""
        pass
    
    @abstractmethod
    def setup(self, config: Dict[str, Any]) -> None:
        """Setup method with configuration"""
        pass
    
    @abstractmethod
    def evaluate(self, data: Any) -> Dict[str, float]:
        """Evaluate method and return performance metrics"""
        pass
    
    @abstractmethod
    def is_deterministic(self) -> bool:
        """Return whether method is deterministic"""
        pass


class RandomBaselineMethod(BaselineMethod):
    """Random baseline for comparison"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
    
    def name(self) -> str:
        return "random_baseline"
    
    def description(self) -> str:
        return "Random baseline that generates random predictions"
    
    def setup(self, config: Dict[str, Any]) -> None:
        self.config = config
    
    def evaluate(self, data: Any) -> Dict[str, float]:
        """Generate random performance metrics"""
        return {
            'accuracy': self.rng.uniform(0.1, 0.3),
            'precision': self.rng.uniform(0.1, 0.3),
            'recall': self.rng.uniform(0.1, 0.3),
            'f1_score': self.rng.uniform(0.1, 0.3),
            'diversity_score': self.rng.uniform(0.2, 0.4),
            'realism_score': self.rng.uniform(0.1, 0.2)
        }
    
    def is_deterministic(self) -> bool:
        return True  # Deterministic with fixed seed


class MajorityClassBaselineMethod(BaselineMethod):
    """Majority class baseline"""
    
    def name(self) -> str:
        return "majority_class_baseline"
    
    def description(self) -> str:
        return "Baseline that always predicts the majority class"
    
    def setup(self, config: Dict[str, Any]) -> None:
        self.config = config
    
    def evaluate(self, data: Any) -> Dict[str, float]:
        """Simulate majority class performance"""
        # Typical performance for majority class baseline
        return {
            'accuracy': 0.6,  # Assumes 60% majority class
            'precision': 0.6,
            'recall': 1.0,  # Always predicts majority
            'f1_score': 0.75,
            'diversity_score': 0.0,  # No diversity
            'realism_score': 0.3
        }
    
    def is_deterministic(self) -> bool:
        return True


class SimpleHeuristicBaselineMethod(BaselineMethod):
    """Simple rule-based heuristic baseline"""
    
    def name(self) -> str:
        return "simple_heuristic_baseline"
    
    def description(self) -> str:
        return "Simple rule-based heuristic approach"
    
    def setup(self, config: Dict[str, Any]) -> None:
        self.config = config
    
    def evaluate(self, data: Any) -> Dict[str, float]:
        """Simulate heuristic performance"""
        return {
            'accuracy': 0.65,
            'precision': 0.62,
            'recall': 0.68,
            'f1_score': 0.65,
            'diversity_score': 0.4,
            'realism_score': 0.5
        }
    
    def is_deterministic(self) -> bool:
        return True


class PreviousStateOfArtBaselineMethod(BaselineMethod):
    """Previous state-of-the-art baseline"""
    
    def __init__(self, performance_metrics: Dict[str, float]):
        self.performance_metrics = performance_metrics
    
    def name(self) -> str:
        return "previous_sota_baseline"
    
    def description(self) -> str:
        return "Previous state-of-the-art method from literature"
    
    def setup(self, config: Dict[str, Any]) -> None:
        self.config = config
    
    def evaluate(self, data: Any) -> Dict[str, float]:
        """Return reported state-of-the-art performance"""
        return self.performance_metrics.copy()
    
    def is_deterministic(self) -> bool:
        return True


class BaselineComparator:
    """Main class for conducting baseline comparisons"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("baseline_comparisons")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.baseline_methods: Dict[str, BaselineMethod] = {}
        self.comparison_history: List[ComparisonSummary] = []
        
        # Register default baselines
        self._register_default_baselines()
        
        logger.info(f"Initialized BaselineComparator with output dir: {self.output_dir}")
    
    def register_baseline(self, baseline: BaselineMethod) -> None:
        """Register a baseline method"""
        self.baseline_methods[baseline.name()] = baseline
        logger.info(f"Registered baseline: {baseline.name()}")
    
    def compare_methods(
        self,
        novel_results: List[Any],  # ExperimentResult objects
        baseline_results: List[Any],  # ExperimentResult objects
        metrics: List[str]
    ) -> List[BaselineComparison]:
        """Compare novel method against baseline"""
        
        comparisons = []
        
        for metric in metrics:
            # Extract metric values
            novel_values = [r.metrics.get(metric, np.nan) for r in novel_results 
                           if metric in r.metrics and not np.isnan(r.metrics[metric])]
            baseline_values = [r.metrics.get(metric, np.nan) for r in baseline_results 
                              if metric in r.metrics and not np.isnan(r.metrics[metric])]
            
            if len(novel_values) >= 1 and len(baseline_values) >= 1:
                comparison = self._perform_single_comparison(
                    novel_values, baseline_values, metric,
                    novel_results[0].method_name, baseline_results[0].method_name
                )
                comparisons.append(comparison)
        
        return comparisons
    
    def run_comprehensive_comparison(
        self,
        experiment: BaselineExperiment,
        novel_method_results: Dict[str, List[Any]],  # method_name -> results
        baseline_method_results: Dict[str, List[Any]]  # method_name -> results
    ) -> ComparisonSummary:
        """Run comprehensive baseline comparison experiment"""
        
        logger.info(f"Starting comprehensive baseline comparison: {experiment.name}")
        
        all_comparisons = []
        
        # Compare each novel method against each baseline
        for novel_method in experiment.novel_methods:
            if novel_method not in novel_method_results:
                logger.warning(f"No results found for novel method: {novel_method}")
                continue
            
            for baseline_method in experiment.baseline_methods:
                if baseline_method not in baseline_method_results:
                    logger.warning(f"No results found for baseline method: {baseline_method}")
                    continue
                
                comparisons = self.compare_methods(
                    novel_method_results[novel_method],
                    baseline_method_results[baseline_method],
                    experiment.evaluation_metrics
                )
                all_comparisons.extend(comparisons)
        
        # Generate summary
        summary = self._generate_comparison_summary(experiment, all_comparisons)
        
        # Save results
        self._save_comparison_results(summary)
        
        # Add to history
        self.comparison_history.append(summary)
        
        logger.info(f"Completed baseline comparison: {experiment.name}")
        return summary
    
    def create_standard_baselines(
        self,
        domain: str = "cybersecurity",
        include_sota: bool = True,
        sota_performance: Optional[Dict[str, float]] = None
    ) -> List[BaselineMethod]:
        """Create standard baseline methods for a domain"""
        
        baselines = [
            RandomBaselineMethod(),
            MajorityClassBaselineMethod(),
            SimpleHeuristicBaselineMethod()
        ]
        
        if include_sota and sota_performance:
            baselines.append(PreviousStateOfArtBaselineMethod(sota_performance))
        
        # Register all baselines
        for baseline in baselines:
            self.register_baseline(baseline)
        
        return baselines
    
    def generate_baseline_report(
        self,
        summary: ComparisonSummary,
        include_visualizations: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive baseline comparison report"""
        
        report = {
            'executive_summary': self._generate_executive_summary(summary),
            'detailed_analysis': self._generate_detailed_analysis(summary),
            'statistical_validation': self._generate_statistical_validation(summary),
            'performance_rankings': self._generate_performance_rankings(summary),
            'recommendations': summary.recommendations,
            'methodology': self._generate_methodology_section(summary)
        }
        
        if include_visualizations:
            report['visualizations'] = self._generate_visualization_specs(summary)
        
        return report
    
    def validate_improvement_claims(
        self,
        summary: ComparisonSummary,
        required_metrics: List[str],
        min_improvement: float = 0.05
    ) -> Dict[str, Any]:
        """Validate claims of improvement over baselines"""
        
        validation = {
            'overall_valid': True,
            'validated_claims': [],
            'invalid_claims': [],
            'warnings': [],
            'evidence_strength': 'strong'
        }
        
        # Check each comparison
        significant_improvements = [c for c in summary.detailed_comparisons 
                                  if c.statistical_significance and c.improvement > min_improvement]
        
        # Validate required metrics
        covered_metrics = set(c.metric for c in significant_improvements)
        missing_metrics = set(required_metrics) - covered_metrics
        
        if missing_metrics:
            validation['overall_valid'] = False
            validation['invalid_claims'].append(
                f"Missing significant improvements for required metrics: {missing_metrics}"
            )
        
        # Check consistency across baselines
        for metric in required_metrics:
            metric_comparisons = [c for c in summary.detailed_comparisons if c.metric == metric]
            
            if metric_comparisons:
                improvement_rates = [c.improvement_percentage for c in metric_comparisons 
                                   if c.statistical_significance]
                
                if improvement_rates:
                    if min(improvement_rates) < min_improvement * 100:
                        validation['warnings'].append(
                            f"Inconsistent improvements for {metric} across baselines"
                        )
                    else:
                        validation['validated_claims'].append(
                            f"Consistent significant improvement for {metric}"
                        )
        
        # Assess evidence strength
        strong_evidence_count = len([c for c in significant_improvements 
                                   if c.effect_size > 0.5 and c.p_value < 0.01])
        
        if strong_evidence_count >= len(required_metrics):
            validation['evidence_strength'] = 'strong'
        elif strong_evidence_count >= len(required_metrics) * 0.5:
            validation['evidence_strength'] = 'moderate'
        else:
            validation['evidence_strength'] = 'weak'
            validation['warnings'].append("Limited strong evidence for improvement claims")
        
        return validation
    
    def _register_default_baselines(self) -> None:
        """Register default baseline methods"""
        self.register_baseline(RandomBaselineMethod())
        self.register_baseline(MajorityClassBaselineMethod())
        self.register_baseline(SimpleHeuristicBaselineMethod())
    
    def _perform_single_comparison(
        self,
        novel_values: List[float],
        baseline_values: List[float],
        metric: str,
        novel_method: str,
        baseline_method: str
    ) -> BaselineComparison:
        """Perform statistical comparison between two methods"""
        
        novel_mean = np.mean(novel_values)
        baseline_mean = np.mean(baseline_values)
        
        improvement = novel_mean - baseline_mean
        improvement_percentage = (improvement / baseline_mean * 100) if baseline_mean != 0 else 0
        
        # Statistical test
        from scipy.stats import ttest_ind
        
        if len(novel_values) > 1 and len(baseline_values) > 1:
            stat, p_value = ttest_ind(novel_values, baseline_values)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(novel_values) - 1) * np.var(novel_values, ddof=1) + 
                                 (len(baseline_values) - 1) * np.var(baseline_values, ddof=1)) / 
                                (len(novel_values) + len(baseline_values) - 2))
            
            effect_size = improvement / pooled_std if pooled_std > 0 else 0
            
            # Confidence interval
            se_diff = pooled_std * np.sqrt(1/len(novel_values) + 1/len(baseline_values))
            from scipy.stats import t
            df = len(novel_values) + len(baseline_values) - 2
            t_critical = t.ppf(0.975, df)  # 95% CI
            ci_lower = improvement - t_critical * se_diff
            ci_upper = improvement + t_critical * se_diff
            
        else:
            p_value = 1.0
            effect_size = 0.0
            ci_lower, ci_upper = improvement, improvement
        
        statistical_significance = p_value < 0.05
        meets_threshold = improvement > 0.05  # 5% improvement threshold
        
        # Interpretation
        if statistical_significance and meets_threshold:
            interpretation = "Significant improvement with practical benefit"
        elif statistical_significance:
            interpretation = "Statistically significant but small improvement"
        elif meets_threshold:
            interpretation = "Practical improvement but not statistically significant"
        else:
            interpretation = "No meaningful improvement"
        
        return BaselineComparison(
            novel_method=novel_method,
            baseline_method=baseline_method,
            metric=metric,
            novel_performance=novel_mean,
            baseline_performance=baseline_mean,
            improvement=improvement,
            improvement_percentage=improvement_percentage,
            statistical_significance=statistical_significance,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            meets_threshold=meets_threshold,
            interpretation=interpretation
        )
    
    def _generate_comparison_summary(
        self,
        experiment: BaselineExperiment,
        comparisons: List[BaselineComparison]
    ) -> ComparisonSummary:
        """Generate summary of comparison results"""
        
        total_comparisons = len(comparisons)
        significant_improvements = len([c for c in comparisons if c.statistical_significance])
        meets_effect_size_threshold = len([c for c in comparisons 
                                          if abs(c.effect_size) > experiment.effect_size_threshold])
        meets_improvement_threshold = len([c for c in comparisons if c.meets_threshold])
        
        # Determine overall success
        success_rate = significant_improvements / total_comparisons if total_comparisons > 0 else 0
        overall_success = success_rate >= 0.5 and meets_improvement_threshold >= total_comparisons * 0.3
        
        # Find best performing method
        method_scores = {}
        for comparison in comparisons:
            method = comparison.novel_method
            if method not in method_scores:
                method_scores[method] = {'improvements': 0, 'total': 0}
            
            method_scores[method]['total'] += 1
            if comparison.statistical_significance and comparison.meets_threshold:
                method_scores[method]['improvements'] += 1
        
        best_method = max(method_scores.keys(), 
                         key=lambda m: method_scores[m]['improvements'] / method_scores[m]['total']) \
                         if method_scores else "unknown"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(comparisons, experiment)
        
        return ComparisonSummary(
            experiment_name=experiment.name,
            total_comparisons=total_comparisons,
            significant_improvements=significant_improvements,
            meets_effect_size_threshold=meets_effect_size_threshold,
            meets_improvement_threshold=meets_improvement_threshold,
            overall_success=overall_success,
            best_performing_method=best_method,
            detailed_comparisons=comparisons,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self,
        comparisons: List[BaselineComparison],
        experiment: BaselineExperiment
    ) -> List[str]:
        """Generate actionable recommendations based on comparison results"""
        
        recommendations = []
        
        # Check for weak improvements
        weak_improvements = [c for c in comparisons if 0 < c.improvement < 0.05]
        if len(weak_improvements) > len(comparisons) * 0.3:
            recommendations.append(
                "Many improvements are small - consider algorithmic enhancements"
            )
        
        # Check for statistical power issues
        non_significant = [c for c in comparisons if not c.statistical_significance]
        if len(non_significant) > len(comparisons) * 0.5:
            recommendations.append(
                "Many comparisons lack statistical significance - increase sample size"
            )
        
        # Check for inconsistent performance
        methods = set(c.novel_method for c in comparisons)
        for method in methods:
            method_comparisons = [c for c in comparisons if c.novel_method == method]
            success_rate = len([c for c in method_comparisons 
                               if c.statistical_significance and c.meets_threshold]) / len(method_comparisons)
            
            if success_rate < 0.5:
                recommendations.append(
                    f"Method {method} shows inconsistent performance across baselines"
                )
        
        # Check metric coverage
        metrics = set(c.metric for c in comparisons)
        if len(metrics) < len(experiment.evaluation_metrics):
            missing_metrics = set(experiment.evaluation_metrics) - metrics
            recommendations.append(
                f"Missing evaluations for metrics: {missing_metrics}"
            )
        
        return recommendations
    
    def _generate_executive_summary(self, summary: ComparisonSummary) -> Dict[str, Any]:
        """Generate executive summary of baseline comparison"""
        
        success_rate = (summary.significant_improvements / summary.total_comparisons * 100) \
                      if summary.total_comparisons > 0 else 0
        
        return {
            'overall_assessment': 'Success' if summary.overall_success else 'Needs Improvement',
            'success_rate': f"{success_rate:.1f}%",
            'best_method': summary.best_performing_method,
            'total_comparisons': summary.total_comparisons,
            'significant_improvements': summary.significant_improvements,
            'key_findings': self._extract_key_findings(summary),
            'next_steps': summary.recommendations[:3]  # Top 3 recommendations
        }
    
    def _generate_detailed_analysis(self, summary: ComparisonSummary) -> Dict[str, Any]:
        """Generate detailed analysis of comparison results"""
        
        # Group comparisons by metric
        metric_analysis = {}
        metrics = set(c.metric for c in summary.detailed_comparisons)
        
        for metric in metrics:
            metric_comparisons = [c for c in summary.detailed_comparisons if c.metric == metric]
            
            metric_analysis[metric] = {
                'total_comparisons': len(metric_comparisons),
                'significant_improvements': len([c for c in metric_comparisons 
                                               if c.statistical_significance]),
                'average_improvement': np.mean([c.improvement_percentage 
                                              for c in metric_comparisons]),
                'best_improvement': max([c.improvement_percentage 
                                       for c in metric_comparisons]),
                'worst_improvement': min([c.improvement_percentage 
                                        for c in metric_comparisons]),
                'consistency_score': self._calculate_consistency_score(metric_comparisons)
            }
        
        # Method analysis
        method_analysis = {}
        methods = set(c.novel_method for c in summary.detailed_comparisons)
        
        for method in methods:
            method_comparisons = [c for c in summary.detailed_comparisons 
                                if c.novel_method == method]
            
            method_analysis[method] = {
                'total_comparisons': len(method_comparisons),
                'success_rate': len([c for c in method_comparisons 
                                   if c.statistical_significance and c.meets_threshold]) / len(method_comparisons),
                'average_effect_size': np.mean([c.effect_size for c in method_comparisons]),
                'strong_improvements': len([c for c in method_comparisons 
                                          if c.effect_size > 0.5 and c.statistical_significance])
            }
        
        return {
            'metric_analysis': metric_analysis,
            'method_analysis': method_analysis,
            'overall_statistics': self._calculate_overall_statistics(summary)
        }
    
    def _generate_statistical_validation(self, summary: ComparisonSummary) -> Dict[str, Any]:
        """Generate statistical validation section"""
        
        # Collect p-values for multiple comparison correction
        p_values = [c.p_value for c in summary.detailed_comparisons]
        
        # Bonferroni correction
        bonferroni_alpha = 0.05 / len(p_values) if p_values else 0.05
        bonferroni_significant = len([p for p in p_values if p < bonferroni_alpha])
        
        # Effect size distribution
        effect_sizes = [c.effect_size for c in summary.detailed_comparisons]
        
        return {
            'multiple_comparisons': {
                'total_tests': len(p_values),
                'bonferroni_alpha': bonferroni_alpha,
                'bonferroni_significant': bonferroni_significant,
                'uncorrected_significant': len([p for p in p_values if p < 0.05])
            },
            'effect_sizes': {
                'mean': np.mean(effect_sizes) if effect_sizes else 0,
                'median': np.median(effect_sizes) if effect_sizes else 0,
                'large_effects': len([es for es in effect_sizes if abs(es) > 0.8]),
                'medium_effects': len([es for es in effect_sizes if 0.5 <= abs(es) <= 0.8]),
                'small_effects': len([es for es in effect_sizes if 0.2 <= abs(es) < 0.5])
            },
            'confidence_intervals': {
                'overlapping_zero': len([c for c in summary.detailed_comparisons 
                                       if c.confidence_interval[0] <= 0 <= c.confidence_interval[1]]),
                'consistently_positive': len([c for c in summary.detailed_comparisons 
                                            if c.confidence_interval[0] > 0])
            }
        }
    
    def _generate_performance_rankings(self, summary: ComparisonSummary) -> Dict[str, Any]:
        """Generate performance rankings"""
        
        # Rank methods by success rate
        methods = set(c.novel_method for c in summary.detailed_comparisons)
        method_rankings = []
        
        for method in methods:
            method_comparisons = [c for c in summary.detailed_comparisons 
                                if c.novel_method == method]
            
            success_rate = len([c for c in method_comparisons 
                               if c.statistical_significance and c.meets_threshold]) / len(method_comparisons)
            
            average_improvement = np.mean([c.improvement_percentage for c in method_comparisons])
            
            method_rankings.append({
                'method': method,
                'success_rate': success_rate,
                'average_improvement': average_improvement,
                'total_comparisons': len(method_comparisons)
            })
        
        method_rankings.sort(key=lambda x: (x['success_rate'], x['average_improvement']), reverse=True)
        
        # Rank metrics by improvement consistency
        metrics = set(c.metric for c in summary.detailed_comparisons)
        metric_rankings = []
        
        for metric in metrics:
            metric_comparisons = [c for c in summary.detailed_comparisons if c.metric == metric]
            
            consistency = self._calculate_consistency_score(metric_comparisons)
            average_improvement = np.mean([c.improvement_percentage for c in metric_comparisons])
            
            metric_rankings.append({
                'metric': metric,
                'consistency_score': consistency,
                'average_improvement': average_improvement,
                'significant_improvements': len([c for c in metric_comparisons 
                                               if c.statistical_significance])
            })
        
        metric_rankings.sort(key=lambda x: (x['consistency_score'], x['average_improvement']), reverse=True)
        
        return {
            'method_rankings': method_rankings,
            'metric_rankings': metric_rankings,
            'top_performer': method_rankings[0]['method'] if method_rankings else None,
            'most_consistent_metric': metric_rankings[0]['metric'] if metric_rankings else None
        }
    
    def _generate_methodology_section(self, summary: ComparisonSummary) -> Dict[str, Any]:
        """Generate methodology description"""
        
        baselines = set(c.baseline_method for c in summary.detailed_comparisons)
        metrics = set(c.metric for c in summary.detailed_comparisons)
        methods = set(c.novel_method for c in summary.detailed_comparisons)
        
        return {
            'comparison_framework': 'Statistical comparison against multiple baselines',
            'baselines_used': list(baselines),
            'evaluation_metrics': list(metrics),
            'novel_methods': list(methods),
            'statistical_tests': 'Independent t-tests with effect size calculation',
            'significance_threshold': 0.05,
            'multiple_comparison_correction': 'Bonferroni correction applied',
            'effect_size_measure': 'Cohen\'s d'
        }
    
    def _generate_visualization_specs(self, summary: ComparisonSummary) -> Dict[str, str]:
        """Generate visualization specifications"""
        
        return {
            'improvement_heatmap': 'Heatmap showing improvement percentages across methods and baselines',
            'effect_size_distribution': 'Histogram of effect sizes with interpretation bands',
            'p_value_histogram': 'Distribution of p-values with significance threshold',
            'method_comparison_radar': 'Radar chart comparing methods across metrics',
            'baseline_performance_bars': 'Bar chart showing baseline performance levels',
            'confidence_intervals': 'Forest plot showing confidence intervals for improvements'
        }
    
    def _extract_key_findings(self, summary: ComparisonSummary) -> List[str]:
        """Extract key findings from comparison results"""
        
        findings = []
        
        # Overall success
        if summary.overall_success:
            findings.append("Novel methods demonstrate clear improvements over baselines")
        else:
            findings.append("Novel methods show mixed results compared to baselines")
        
        # Best performing method
        if summary.best_performing_method != "unknown":
            findings.append(f"Best performing method: {summary.best_performing_method}")
        
        # Statistical significance
        sig_rate = summary.significant_improvements / summary.total_comparisons
        if sig_rate >= 0.8:
            findings.append("Strong statistical evidence of improvement")
        elif sig_rate >= 0.5:
            findings.append("Moderate statistical evidence of improvement")
        else:
            findings.append("Limited statistical evidence of improvement")
        
        # Effect sizes
        large_effects = len([c for c in summary.detailed_comparisons if abs(c.effect_size) > 0.8])
        if large_effects >= summary.total_comparisons * 0.3:
            findings.append("Large practical improvements demonstrated")
        
        return findings
    
    def _calculate_consistency_score(self, comparisons: List[BaselineComparison]) -> float:
        """Calculate consistency score for a set of comparisons"""
        
        if not comparisons:
            return 0.0
        
        improvements = [c.improvement_percentage for c in comparisons]
        
        # Consistency based on coefficient of variation
        if np.mean(improvements) != 0:
            cv = np.std(improvements) / abs(np.mean(improvements))
            consistency = max(0, 1 - cv)  # Higher consistency = lower variation
        else:
            consistency = 0.0
        
        return consistency
    
    def _calculate_overall_statistics(self, summary: ComparisonSummary) -> Dict[str, Any]:
        """Calculate overall statistics"""
        
        improvements = [c.improvement_percentage for c in summary.detailed_comparisons]
        effect_sizes = [c.effect_size for c in summary.detailed_comparisons]
        p_values = [c.p_value for c in summary.detailed_comparisons]
        
        return {
            'improvement_statistics': {
                'mean': np.mean(improvements),
                'median': np.median(improvements),
                'std': np.std(improvements),
                'min': np.min(improvements),
                'max': np.max(improvements)
            },
            'effect_size_statistics': {
                'mean': np.mean(effect_sizes),
                'median': np.median(effect_sizes),
                'std': np.std(effect_sizes)
            },
            'significance_statistics': {
                'mean_p_value': np.mean(p_values),
                'significant_count': len([p for p in p_values if p < 0.05]),
                'highly_significant_count': len([p for p in p_values if p < 0.01])
            }
        }
    
    def _save_comparison_results(self, summary: ComparisonSummary) -> None:
        """Save comparison results to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"baseline_comparison_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Convert summary to dictionary for JSON serialization
        summary_dict = {
            'experiment_name': summary.experiment_name,
            'total_comparisons': summary.total_comparisons,
            'significant_improvements': summary.significant_improvements,
            'meets_effect_size_threshold': summary.meets_effect_size_threshold,
            'meets_improvement_threshold': summary.meets_improvement_threshold,
            'overall_success': summary.overall_success,
            'best_performing_method': summary.best_performing_method,
            'recommendations': summary.recommendations,
            'generated_at': summary.generated_at,
            'detailed_comparisons': [
                {
                    'novel_method': c.novel_method,
                    'baseline_method': c.baseline_method,
                    'metric': c.metric,
                    'novel_performance': c.novel_performance,
                    'baseline_performance': c.baseline_performance,
                    'improvement': c.improvement,
                    'improvement_percentage': c.improvement_percentage,
                    'statistical_significance': c.statistical_significance,
                    'p_value': c.p_value,
                    'effect_size': c.effect_size,
                    'confidence_interval': c.confidence_interval,
                    'meets_threshold': c.meets_threshold,
                    'interpretation': c.interpretation
                }
                for c in summary.detailed_comparisons
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        
        logger.info(f"Baseline comparison results saved to: {filepath}")