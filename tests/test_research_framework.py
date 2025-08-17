"""
Comprehensive test suite for the research framework module.

This module tests all research capabilities including experiment framework,
statistical validation, baseline comparison, and reproducibility features.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import List, Dict, Any

from gan_cyber_range.research.experiment_framework import (
    ExperimentFramework, ExperimentConfig, ExperimentMethod,
    ExperimentResult, Experiment
)
from gan_cyber_range.research.statistical_validator import (
    StatisticalValidator, StatisticalTest
)
from gan_cyber_range.research.baseline_comparator import (
    BaselineComparator, BaselineExperiment, BaselineMethod,
    RandomBaselineMethod, MajorityClassBaselineMethod
)


class MockExperimentMethod(ExperimentMethod):
    """Mock experiment method for testing"""
    
    def __init__(self, method_name: str, performance_metrics: Dict[str, float]):
        self.method_name = method_name
        self.performance_metrics = performance_metrics
        self.setup_called = False
        self.cleanup_called = False
    
    def name(self) -> str:
        return self.method_name
    
    def setup(self, config: ExperimentConfig) -> None:
        self.setup_called = True
    
    def execute(self, data: Any) -> Dict[str, float]:
        # Add some noise to make results realistic
        noisy_metrics = {}
        for metric, value in self.performance_metrics.items():
            noise = np.random.normal(0, value * 0.05)  # 5% noise
            noisy_metrics[metric] = max(0, value + noise)
        return noisy_metrics
    
    def cleanup(self) -> None:
        self.cleanup_called = True


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for experiment outputs"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def experiment_framework(temp_output_dir):
    """Create experiment framework instance"""
    return ExperimentFramework(output_dir=temp_output_dir)


@pytest.fixture
def sample_experiment_config():
    """Create sample experiment configuration"""
    return ExperimentConfig(
        name="Test Experiment",
        description="Testing experiment framework",
        hypothesis="Method A performs better than baselines",
        success_criteria={
            'accuracy': 0.8,
            'precision': 0.75,
            'recall': 0.7
        },
        baseline_methods=['baseline_random', 'baseline_majority'],
        evaluation_metrics=['accuracy', 'precision', 'recall'],
        dataset_size=1000,
        num_trials=5,
        random_seed=42
    )


@pytest.fixture
def mock_methods():
    """Create mock experiment methods"""
    return [
        MockExperimentMethod("novel_method_a", {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.78
        }),
        MockExperimentMethod("novel_method_b", {
            'accuracy': 0.78,
            'precision': 0.75,
            'recall': 0.81
        }),
        MockExperimentMethod("baseline_random", {
            'accuracy': 0.25,
            'precision': 0.22,
            'recall': 0.28
        }),
        MockExperimentMethod("baseline_majority", {
            'accuracy': 0.60,
            'precision': 0.58,
            'recall': 0.65
        })
    ]


@pytest.fixture
def sample_data_generator():
    """Create sample data generator function"""
    def generate_data(size: int):
        # Generate mock training data
        return np.random.randn(size, 10)  # 10 features
    
    return generate_data


class TestExperimentFramework:
    """Test cases for ExperimentFramework"""
    
    def test_initialization(self, temp_output_dir):
        """Test experiment framework initialization"""
        framework = ExperimentFramework(output_dir=temp_output_dir)
        
        assert framework.output_dir == temp_output_dir
        assert framework.output_dir.exists()
        assert len(framework.methods) == 0
        assert len(framework.data_generators) == 0
        assert framework.current_experiment is None
    
    def test_register_method(self, experiment_framework, mock_methods):
        """Test registering experiment methods"""
        method = mock_methods[0]
        experiment_framework.register_method(method)
        
        assert method.name() in experiment_framework.methods
        assert experiment_framework.methods[method.name()] == method
    
    def test_register_data_generator(self, experiment_framework, sample_data_generator):
        """Test registering data generator"""
        experiment_framework.register_data_generator("test_data", sample_data_generator)
        
        assert "test_data" in experiment_framework.data_generators
        assert experiment_framework.data_generators["test_data"] == sample_data_generator
    
    def test_design_experiment(self, experiment_framework, sample_experiment_config, 
                              mock_methods, sample_data_generator):
        """Test experiment design"""
        # Register components
        for method in mock_methods:
            experiment_framework.register_method(method)
        experiment_framework.register_data_generator("test_data", sample_data_generator)
        
        # Design experiment
        method_names = [m.name() for m in mock_methods]
        experiment_id = experiment_framework.design_experiment(
            sample_experiment_config, method_names, "test_data"
        )
        
        assert experiment_id is not None
        assert len(experiment_id) > 0
        
        # Check that config file was created
        config_file = experiment_framework.output_dir / f"{experiment_id}_config.json"
        assert config_file.exists()
    
    def test_design_experiment_invalid_method(self, experiment_framework, 
                                            sample_experiment_config, sample_data_generator):
        """Test experiment design with invalid method"""
        experiment_framework.register_data_generator("test_data", sample_data_generator)
        
        with pytest.raises(ValueError, match="Method invalid_method not registered"):
            experiment_framework.design_experiment(
                sample_experiment_config, ["invalid_method"], "test_data"
            )
    
    def test_design_experiment_invalid_data_generator(self, experiment_framework,
                                                    sample_experiment_config, mock_methods):
        """Test experiment design with invalid data generator"""
        for method in mock_methods:
            experiment_framework.register_method(method)
        
        method_names = [m.name() for m in mock_methods]
        
        with pytest.raises(ValueError, match="Data generator invalid_generator not registered"):
            experiment_framework.design_experiment(
                sample_experiment_config, method_names, "invalid_generator"
            )
    
    def test_run_experiment(self, experiment_framework, sample_experiment_config,
                           mock_methods, sample_data_generator):
        """Test running an experiment"""
        # Setup
        for method in mock_methods:
            experiment_framework.register_method(method)
        experiment_framework.register_data_generator("test_data", sample_data_generator)
        
        method_names = [m.name() for m in mock_methods]
        experiment_id = experiment_framework.design_experiment(
            sample_experiment_config, method_names, "test_data"
        )
        
        # Run experiment
        summary = experiment_framework.run_experiment(experiment_id)
        
        # Validate results
        assert summary is not None
        assert summary.config.name == sample_experiment_config.name
        assert len(summary.results) > 0
        assert summary.statistical_tests is not None
        assert summary.reproducibility_score >= 0.0
        assert summary.reproducibility_score <= 1.0
        
        # Check that all methods were executed
        method_names_in_results = set(r.method_name for r in summary.results)
        assert all(name in method_names_in_results for name in method_names)
        
        # Check that setup and cleanup were called
        for method in mock_methods:
            assert method.setup_called
            assert method.cleanup_called
    
    def test_compare_experiments(self, experiment_framework, sample_experiment_config,
                               mock_methods, sample_data_generator):
        """Test comparing multiple experiments"""
        # Setup and run first experiment
        for method in mock_methods:
            experiment_framework.register_method(method)
        experiment_framework.register_data_generator("test_data", sample_data_generator)
        
        method_names = [m.name() for m in mock_methods]
        
        # Run two experiments
        exp1_id = experiment_framework.design_experiment(
            sample_experiment_config, method_names, "test_data"
        )
        summary1 = experiment_framework.run_experiment(exp1_id)
        
        # Modify config for second experiment
        config2 = sample_experiment_config
        config2.name = "Test Experiment 2"
        config2.random_seed = 123
        
        exp2_id = experiment_framework.design_experiment(config2, method_names, "test_data")
        summary2 = experiment_framework.run_experiment(exp2_id)
        
        # Compare experiments
        comparison = experiment_framework.compare_experiments(
            [exp1_id, exp2_id], ['accuracy', 'precision']
        )
        
        assert 'experiments' in comparison
        assert 'comparisons' in comparison
        assert 'rankings' in comparison
        assert len(comparison['experiments']) == 2
        assert 'accuracy' in comparison['comparisons']
        assert 'precision' in comparison['comparisons']
    
    def test_generate_publication_report(self, experiment_framework, sample_experiment_config,
                                       mock_methods, sample_data_generator):
        """Test generating publication report"""
        # Setup and run experiment
        for method in mock_methods:
            experiment_framework.register_method(method)
        experiment_framework.register_data_generator("test_data", sample_data_generator)
        
        method_names = [m.name() for m in mock_methods]
        experiment_id = experiment_framework.design_experiment(
            sample_experiment_config, method_names, "test_data"
        )
        summary = experiment_framework.run_experiment(experiment_id)
        
        # Generate report
        report = experiment_framework.generate_publication_report(experiment_id)
        
        assert 'experiment_id' in report
        assert 'title' in report
        assert 'abstract' in report
        assert 'methodology' in report
        assert 'results' in report
        assert 'discussion' in report
        assert 'statistical_analysis' in report
        assert 'reproducibility' in report
        
        # Check reproducibility section
        repro = report['reproducibility']
        assert 'score' in repro
        assert 'seed' in repro
        assert 'trials' in repro
        assert repro['score'] >= 0.0
        assert repro['score'] <= 1.0


class TestStatisticalValidator:
    """Test cases for StatisticalValidator"""
    
    def test_initialization(self):
        """Test statistical validator initialization"""
        validator = StatisticalValidator(significance_level=0.01)
        
        assert validator.significance_level == 0.01
        assert len(validator.results_cache) == 0
    
    def test_validate_results_with_mock_data(self, sample_experiment_config):
        """Test statistical validation with mock data"""
        validator = StatisticalValidator()
        
        # Create mock experiment results
        results = []
        for i in range(10):  # 10 trials
            for method_name in ["method_a", "method_b", "baseline"]:
                # Simulate different performance levels
                if method_name == "method_a":
                    accuracy = 0.85 + np.random.normal(0, 0.02)
                elif method_name == "method_b":
                    accuracy = 0.78 + np.random.normal(0, 0.03)
                else:  # baseline
                    accuracy = 0.60 + np.random.normal(0, 0.05)
                
                result = ExperimentResult(
                    trial_id=i,
                    method_name=method_name,
                    metrics={'accuracy': max(0, min(1, accuracy))},
                    execution_time=1.0,
                    memory_usage=100.0,
                    success=True
                )
                results.append(result)
        
        # Validate results
        validation = validator.validate_results(results, sample_experiment_config)
        
        assert 'descriptive_statistics' in validation
        assert 'normality_tests' in validation
        assert 'significance_tests' in validation
        assert 'effect_sizes' in validation
        assert 'validation_summary' in validation
        
        # Check descriptive statistics
        desc_stats = validation['descriptive_statistics']
        assert 'method_a' in desc_stats
        assert 'method_b' in desc_stats
        assert 'baseline' in desc_stats
        
        # Check that statistics are reasonable
        for method_stats in desc_stats.values():
            if 'accuracy' in method_stats:
                stats = method_stats['accuracy']
                assert 'mean' in stats
                assert 'std' in stats
                assert 'count' in stats
                assert stats['count'] == 10  # 10 trials
    
    def test_two_group_comparison(self):
        """Test two-group statistical comparison"""
        validator = StatisticalValidator()
        
        # Create clearly different groups
        group1 = [0.8, 0.82, 0.85, 0.83, 0.81]  # High performance
        group2 = [0.6, 0.58, 0.62, 0.59, 0.61]  # Low performance
        
        test_result = validator._two_group_test(group1, group2, "accuracy")
        
        assert isinstance(test_result, StatisticalTest)
        assert test_result.test_name in ["Independent t-test", "Welch's t-test", "Mann-Whitney U"]
        assert test_result.p_value >= 0.0
        assert test_result.p_value <= 1.0
        assert test_result.effect_size is not None
        assert test_result.confidence_interval is not None
    
    def test_multiple_group_comparison(self):
        """Test multi-group statistical comparison"""
        validator = StatisticalValidator()
        
        # Create three clearly different groups
        groups = [
            [0.85, 0.87, 0.83, 0.86, 0.84],  # High performance
            [0.75, 0.73, 0.77, 0.74, 0.76],  # Medium performance
            [0.60, 0.58, 0.62, 0.59, 0.61]   # Low performance
        ]
        
        test_result = validator._multi_group_test(groups, ["method_a", "method_b", "baseline"], "accuracy")
        
        assert isinstance(test_result, StatisticalTest)
        assert test_result.test_name in ["One-way ANOVA", "Kruskal-Wallis"]
        assert test_result.p_value >= 0.0
        assert test_result.p_value <= 1.0
        assert test_result.effect_size is not None


class TestBaselineComparator:
    """Test cases for BaselineComparator"""
    
    def test_initialization(self, temp_output_dir):
        """Test baseline comparator initialization"""
        comparator = BaselineComparator(output_dir=temp_output_dir)
        
        assert comparator.output_dir == temp_output_dir
        assert len(comparator.baseline_methods) > 0  # Should have default baselines
        assert len(comparator.comparison_history) == 0
    
    def test_register_baseline(self, temp_output_dir):
        """Test registering baseline method"""
        comparator = BaselineComparator(output_dir=temp_output_dir)
        baseline = RandomBaselineMethod()
        
        comparator.register_baseline(baseline)
        
        assert baseline.name() in comparator.baseline_methods
        assert comparator.baseline_methods[baseline.name()] == baseline
    
    def test_random_baseline_method(self):
        """Test random baseline method"""
        baseline = RandomBaselineMethod(random_seed=42)
        
        assert baseline.name() == "random_baseline"
        assert baseline.is_deterministic() == True
        
        baseline.setup({})
        metrics = baseline.evaluate(None)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert all(0 <= v <= 1 for v in metrics.values())
    
    def test_majority_class_baseline_method(self):
        """Test majority class baseline method"""
        baseline = MajorityClassBaselineMethod()
        
        assert baseline.name() == "majority_class_baseline"
        assert baseline.is_deterministic() == True
        
        baseline.setup({})
        metrics = baseline.evaluate(None)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert metrics['accuracy'] == 0.6  # Fixed majority class accuracy
        assert metrics['recall'] == 1.0     # Always predicts majority
    
    def test_compare_methods(self, temp_output_dir):
        """Test comparing methods against baselines"""
        comparator = BaselineComparator(output_dir=temp_output_dir)
        
        # Create mock results
        novel_results = []
        baseline_results = []
        
        # Novel method performs better
        for i in range(5):
            novel_results.append(ExperimentResult(
                trial_id=i,
                method_name="novel_method",
                metrics={'accuracy': 0.85 + np.random.normal(0, 0.02)},
                execution_time=1.0,
                memory_usage=100.0,
                success=True
            ))
            
            baseline_results.append(ExperimentResult(
                trial_id=i,
                method_name="baseline_method",
                metrics={'accuracy': 0.60 + np.random.normal(0, 0.05)},
                execution_time=1.0,
                memory_usage=100.0,
                success=True
            ))
        
        # Compare methods
        comparisons = comparator.compare_methods(novel_results, baseline_results, ['accuracy'])
        
        assert len(comparisons) == 1
        comparison = comparisons[0]
        
        assert comparison.novel_method == "novel_method"
        assert comparison.baseline_method == "baseline_method"
        assert comparison.metric == "accuracy"
        assert comparison.novel_performance > comparison.baseline_performance
        assert comparison.improvement > 0
        assert comparison.improvement_percentage > 0
    
    def test_comprehensive_comparison(self, temp_output_dir):
        """Test comprehensive baseline comparison"""
        comparator = BaselineComparator(output_dir=temp_output_dir)
        
        # Create experiment configuration
        experiment = BaselineExperiment(
            name="Test Baseline Comparison",
            description="Testing baseline comparison functionality",
            baseline_methods=["random_baseline", "majority_baseline"],
            novel_methods=["novel_method"],
            evaluation_metrics=["accuracy", "precision"]
        )
        
        # Create mock results
        novel_method_results = {
            "novel_method": [
                ExperimentResult(
                    trial_id=i,
                    method_name="novel_method",
                    metrics={
                        'accuracy': 0.85 + np.random.normal(0, 0.02),
                        'precision': 0.82 + np.random.normal(0, 0.03)
                    },
                    execution_time=1.0,
                    memory_usage=100.0,
                    success=True
                )
                for i in range(5)
            ]
        }
        
        baseline_method_results = {
            "random_baseline": [
                ExperimentResult(
                    trial_id=i,
                    method_name="random_baseline",
                    metrics={
                        'accuracy': 0.25 + np.random.normal(0, 0.05),
                        'precision': 0.22 + np.random.normal(0, 0.04)
                    },
                    execution_time=1.0,
                    memory_usage=100.0,
                    success=True
                )
                for i in range(5)
            ],
            "majority_baseline": [
                ExperimentResult(
                    trial_id=i,
                    method_name="majority_baseline",
                    metrics={
                        'accuracy': 0.60 + np.random.normal(0, 0.03),
                        'precision': 0.58 + np.random.normal(0, 0.04)
                    },
                    execution_time=1.0,
                    memory_usage=100.0,
                    success=True
                )
                for i in range(5)
            ]
        }
        
        # Run comprehensive comparison
        summary = comparator.run_comprehensive_comparison(
            experiment, novel_method_results, baseline_method_results
        )
        
        assert summary is not None
        assert summary.experiment_name == experiment.name
        assert summary.total_comparisons > 0
        assert summary.best_performing_method == "novel_method"
        assert len(summary.detailed_comparisons) > 0
        assert len(summary.recommendations) >= 0
    
    def test_validation_of_improvement_claims(self, temp_output_dir):
        """Test validation of improvement claims"""
        comparator = BaselineComparator(output_dir=temp_output_dir)
        
        # Create a summary with strong improvements
        from gan_cyber_range.research.baseline_comparator import ComparisonSummary, BaselineComparison
        
        strong_comparisons = [
            BaselineComparison(
                novel_method="novel_method",
                baseline_method="baseline",
                metric="accuracy",
                novel_performance=0.85,
                baseline_performance=0.60,
                improvement=0.25,
                improvement_percentage=41.7,
                statistical_significance=True,
                p_value=0.001,
                effect_size=1.2,
                confidence_interval=(0.20, 0.30),
                meets_threshold=True,
                interpretation="Significant improvement with large effect"
            )
        ]
        
        summary = ComparisonSummary(
            experiment_name="Test Validation",
            total_comparisons=1,
            significant_improvements=1,
            meets_effect_size_threshold=1,
            meets_improvement_threshold=1,
            overall_success=True,
            best_performing_method="novel_method",
            detailed_comparisons=strong_comparisons,
            recommendations=[]
        )
        
        # Validate improvement claims
        validation = comparator.validate_improvement_claims(
            summary, required_metrics=['accuracy'], min_improvement=0.05
        )
        
        assert validation['overall_valid'] == True
        assert validation['evidence_strength'] == 'strong'
        assert len(validation['validated_claims']) > 0


class TestExperimentIntegration:
    """Integration tests for the complete research framework"""
    
    def test_end_to_end_experiment(self, temp_output_dir):
        """Test complete end-to-end experiment workflow"""
        # Initialize components
        framework = ExperimentFramework(output_dir=temp_output_dir)
        experiment = Experiment(framework)
        
        # Create mock GAN methods and baselines
        gan_methods = [
            MockExperimentMethod("wasserstein_gan", {
                'diversity_score': 0.78,
                'realism_score': 0.82,
                'generation_speed': 120.0
            }),
            MockExperimentMethod("vanilla_gan", {
                'diversity_score': 0.72,
                'realism_score': 0.75,
                'generation_speed': 150.0
            })
        ]
        
        baseline_methods = [
            MockExperimentMethod("random_generation", {
                'diversity_score': 0.35,
                'realism_score': 0.20,
                'generation_speed': 200.0
            }),
            MockExperimentMethod("template_based", {
                'diversity_score': 0.45,
                'realism_score': 0.60,
                'generation_speed': 80.0
            })
        ]
        
        def mock_dataset_generator(size: int):
            return np.random.randn(size, 50)  # 50-dimensional feature space
        
        # Create GAN comparison experiment
        experiment_id = experiment.create_gan_comparison_experiment(
            gan_methods, baseline_methods, mock_dataset_generator
        )
        
        assert experiment_id is not None
        
        # Run the experiment
        summary = framework.run_experiment(experiment_id, data_size_override=500)
        
        # Validate results
        assert summary is not None
        assert summary.publication_ready in [True, False]  # Should have assessment
        assert len(summary.results) > 0
        assert summary.reproducibility_score >= 0.0
        
        # Generate publication report
        report = framework.generate_publication_report(experiment_id)
        
        assert 'title' in report
        assert 'methodology' in report
        assert 'results' in report
        
        # Test baseline comparison
        comparator = BaselineComparator(output_dir=temp_output_dir)
        
        # Extract results by method type
        gan_results = [r for r in summary.results if r.method_name in ['wasserstein_gan', 'vanilla_gan']]
        baseline_results = [r for r in summary.results if r.method_name in ['random_generation', 'template_based']]
        
        if gan_results and baseline_results:
            comparisons = comparator.compare_methods(
                gan_results, baseline_results, ['diversity_score', 'realism_score']
            )
            
            assert len(comparisons) > 0
            
            # Should show GAN methods outperforming baselines
            for comparison in comparisons:
                if comparison.novel_method in ['wasserstein_gan', 'vanilla_gan']:
                    assert comparison.improvement >= 0  # GANs should be better


if __name__ == "__main__":
    pytest.main([__file__, "-v"])