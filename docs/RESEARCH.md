# Research Methodology and Academic Framework

## Overview

GAN-Cyber-Range-v2 is designed to support rigorous academic research in cybersecurity education, AI-driven security testing, and defensive capability assessment. This document outlines the research framework, methodologies, and best practices for conducting reproducible research.

## Research Architecture

### Hypothesis-Driven Development Framework

The platform implements a systematic approach to cybersecurity research:

1. **Hypothesis Formation**: Clear, testable hypotheses about cybersecurity phenomena
2. **Experimental Design**: Controlled experiments with proper baselines
3. **Data Collection**: Systematic gathering of training and evaluation data
4. **Statistical Analysis**: Rigorous statistical validation of results
5. **Reproducibility**: Complete documentation and artifact preservation

### Core Research Questions

#### 1. GAN-Based Attack Generation Effectiveness

**Research Question**: How effective are GAN-generated synthetic attacks compared to real-world attack data for training defensive systems?

**Methodology**:
```python
from gan_cyber_range import AttackGAN, DefenseEvaluator

# Experimental setup
def evaluate_synthetic_vs_real_attacks():
    # Train GAN on real attack data
    attack_gan = AttackGAN(architecture="wasserstein")
    attack_gan.train(real_attacks="datasets/mitre_attack_samples/")
    
    # Generate synthetic attacks
    synthetic_attacks = attack_gan.generate(num_samples=10000)
    
    # Evaluate defensive performance
    evaluator = DefenseEvaluator()
    
    # Test 1: Real attacks only
    real_performance = evaluator.test_defenses(
        attacks=real_attacks,
        defense_systems=["snort", "suricata", "zeek"]
    )
    
    # Test 2: Synthetic attacks only  
    synthetic_performance = evaluator.test_defenses(
        attacks=synthetic_attacks,
        defense_systems=["snort", "suricata", "zeek"]
    )
    
    # Test 3: Combined dataset
    combined_attacks = real_attacks + synthetic_attacks
    combined_performance = evaluator.test_defenses(
        attacks=combined_attacks,
        defense_systems=["snort", "suricata", "zeek"]
    )
    
    return {
        'real_only': real_performance,
        'synthetic_only': synthetic_performance,
        'combined': combined_performance
    }
```

**Statistical Validation**:
```python
import scipy.stats as stats
from statsmodels.stats.power import ttest_power

def statistical_analysis(results):
    # Power analysis
    effect_size = 0.5  # Medium effect size
    alpha = 0.05
    power = 0.8
    
    required_sample_size = ttest_power(
        effect_size, power, alpha, alternative='two-sided'
    )
    
    # Paired t-test for detection rates
    real_detection_rates = results['real_only']['detection_rates']
    synthetic_detection_rates = results['synthetic_only']['detection_rates']
    
    statistic, p_value = stats.ttest_rel(
        real_detection_rates, synthetic_detection_rates
    )
    
    # Effect size calculation (Cohen's d)
    mean_diff = np.mean(real_detection_rates) - np.mean(synthetic_detection_rates)
    pooled_std = np.sqrt(
        (np.var(real_detection_rates) + np.var(synthetic_detection_rates)) / 2
    )
    cohens_d = mean_diff / pooled_std
    
    return {
        'required_sample_size': required_sample_size,
        't_statistic': statistic,
        'p_value': p_value,
        'effect_size': cohens_d,
        'significant': p_value < alpha
    }
```

#### 2. LLM-Driven Red Team Effectiveness

**Research Question**: How do LLM-generated attack scenarios compare to human-designed scenarios in terms of realism, coverage, and training effectiveness?

**Methodology**:
```python
from gan_cyber_range import RedTeamLLM, ScenarioEvaluator

def compare_llm_vs_human_scenarios():
    # LLM-generated scenarios
    red_team_llm = RedTeamLLM(
        model="llama2-70b-security",
        creativity=0.8,
        risk_tolerance=0.6
    )
    
    llm_scenarios = []
    for i in range(100):
        scenario = red_team_llm.generate_attack_plan(
            target_profile=generate_random_target(),
            constraints={"stealth_level": "high"}
        )
        llm_scenarios.append(scenario)
    
    # Human-designed scenarios (baseline)
    human_scenarios = load_human_designed_scenarios("datasets/expert_scenarios/")
    
    # Evaluation metrics
    evaluator = ScenarioEvaluator()
    
    results = {
        'llm_realism': evaluator.assess_realism(llm_scenarios),
        'human_realism': evaluator.assess_realism(human_scenarios),
        'llm_coverage': evaluator.assess_mitre_coverage(llm_scenarios),
        'human_coverage': evaluator.assess_mitre_coverage(human_scenarios),
        'llm_training_effectiveness': evaluator.assess_training_value(llm_scenarios),
        'human_training_effectiveness': evaluator.assess_training_value(human_scenarios)
    }
    
    return results
```

#### 3. Adaptive Attack Evolution

**Research Question**: How effectively can AI-driven attacks adapt to evolving defensive measures in real-time?

**Methodology**:
```python
from gan_cyber_range import AdaptiveAttacker, CyberRange

def study_attack_adaptation():
    # Set up controlled environment
    cyber_range = CyberRange(topology=create_standard_topology())
    cyber_range.deploy()
    
    # Initialize adaptive attacker
    attacker = AdaptiveAttacker(
        base_gan=AttackGAN(),
        adaptation_strategy="reinforcement_learning"
    )
    
    # Initialize defender with learning capability
    defender = AdaptiveDefender(
        detection_systems=["ml_ids", "behavior_analysis"],
        learning_rate=0.1
    )
    
    adaptation_results = []
    
    for round in range(50):  # 50 rounds of adaptation
        # Attacker generates attack
        attack = attacker.generate_attack(
            target_info=cyber_range.get_target_info(),
            previous_results=adaptation_results
        )
        
        # Execute attack
        attack_result = cyber_range.execute_attack(attack)
        
        # Defender updates based on attack
        defender.learn_from_attack(attack, attack_result)
        
        # Attacker adapts based on detection
        attacker.adapt_to_detection(attack_result.detection_events)
        
        adaptation_results.append({
            'round': round,
            'attack_success': attack_result.success,
            'detection_confidence': attack_result.detection_confidence,
            'adaptation_score': attacker.get_adaptation_score()
        })
    
    return adaptation_results
```

### Research Data Management

#### Dataset Preparation

```python
class ResearchDataset:
    """Standardized dataset for cybersecurity research"""
    
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.metadata = {
            'creation_date': datetime.now(),
            'creator': 'research_team',
            'license': 'MIT',
            'citation': None
        }
        self.data = {}
        self.labels = {}
        self.validation_split = 0.2
        self.test_split = 0.1
    
    def add_attack_samples(self, samples: List[str], labels: List[str]):
        """Add labeled attack samples"""
        assert len(samples) == len(labels)
        
        for sample, label in zip(samples, labels):
            sample_id = hashlib.md5(sample.encode()).hexdigest()
            self.data[sample_id] = {
                'content': sample,
                'type': 'attack_sample',
                'preprocessing': 'raw'
            }
            self.labels[sample_id] = label
    
    def add_network_traces(self, traces: List[bytes], labels: List[str]):
        """Add network traffic traces"""
        for trace, label in zip(traces, labels):
            trace_id = hashlib.md5(trace).hexdigest()
            self.data[trace_id] = {
                'content': base64.b64encode(trace).decode(),
                'type': 'network_trace',
                'preprocessing': 'pcap'
            }
            self.labels[trace_id] = label
    
    def create_splits(self, random_seed: int = 42):
        """Create train/validation/test splits"""
        np.random.seed(random_seed)
        
        sample_ids = list(self.data.keys())
        np.random.shuffle(sample_ids)
        
        n_samples = len(sample_ids)
        n_test = int(n_samples * self.test_split)
        n_val = int(n_samples * self.validation_split)
        
        self.splits = {
            'test': sample_ids[:n_test],
            'validation': sample_ids[n_test:n_test + n_val],
            'train': sample_ids[n_test + n_val:]
        }
    
    def export_for_reproduction(self, output_dir: str):
        """Export dataset with complete reproducibility info"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export data
        with open(output_path / 'data.json', 'w') as f:
            json.dump(self.data, f, indent=2)
        
        # Export labels
        with open(output_path / 'labels.json', 'w') as f:
            json.dump(self.labels, f, indent=2)
        
        # Export splits
        with open(output_path / 'splits.json', 'w') as f:
            json.dump(self.splits, f, indent=2)
        
        # Export metadata
        metadata = self.metadata.copy()
        metadata['creation_date'] = metadata['creation_date'].isoformat()
        metadata['sample_count'] = len(self.data)
        metadata['label_distribution'] = self._get_label_distribution()
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create citation file
        with open(output_path / 'CITATION.cff', 'w') as f:
            f.write(self._generate_citation())
    
    def _get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels"""
        from collections import Counter
        return dict(Counter(self.labels.values()))
    
    def _generate_citation(self) -> str:
        """Generate citation in Citation File Format"""
        return f"""cff-version: 1.2.0
message: "If you use this dataset, please cite it as below."
type: dataset
title: "{self.name}"
version: "{self.version}"
date-released: "{self.metadata['creation_date'].date()}"
authors:
  - family-names: "Research Team"
    given-names: "Cyber Range"
license: MIT
repository-code: "https://github.com/terragonlabs/gan-cyber-range-v2"
"""
```

#### Experimental Reproducibility

```python
class ExperimentTracker:
    """Track experiments for full reproducibility"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.experiment_id = f"{experiment_name}_{int(time.time())}"
        self.config = {}
        self.results = {}
        self.artifacts = {}
        self.environment = self._capture_environment()
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration"""
        self.config = config
        
        # Save git commit hash for code version
        try:
            import subprocess
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                text=True
            ).strip()
            self.config['git_commit'] = git_hash
        except:
            self.config['git_commit'] = 'unknown'
    
    def log_result(self, metric_name: str, value: float, step: int = None):
        """Log experimental results"""
        if metric_name not in self.results:
            self.results[metric_name] = []
        
        self.results[metric_name].append({
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat()
        })
    
    def save_artifact(self, name: str, artifact: Any):
        """Save experimental artifacts"""
        self.artifacts[name] = artifact
    
    def export_experiment(self, output_dir: str):
        """Export complete experiment for reproduction"""
        output_path = Path(output_dir) / self.experiment_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(output_path / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save results
        with open(output_path / 'results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save environment info
        with open(output_path / 'environment.json', 'w') as f:
            json.dump(self.environment, f, indent=2)
        
        # Save artifacts
        artifacts_dir = output_path / 'artifacts'
        artifacts_dir.mkdir(exist_ok=True)
        
        for name, artifact in self.artifacts.items():
            if hasattr(artifact, 'save'):
                # PyTorch model
                artifact.save(artifacts_dir / f'{name}.pth')
            else:
                # Generic pickle
                with open(artifacts_dir / f'{name}.pkl', 'wb') as f:
                    pickle.dump(artifact, f)
        
        # Create reproduction script
        self._create_reproduction_script(output_path)
    
    def _capture_environment(self) -> Dict[str, Any]:
        """Capture environment information"""
        import platform
        import sys
        
        env = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'hostname': platform.node(),
            'packages': {}
        }
        
        # Capture package versions
        try:
            import pkg_resources
            for package in pkg_resources.working_set:
                env['packages'][package.project_name] = package.version
        except:
            pass
        
        return env
    
    def _create_reproduction_script(self, output_path: Path):
        """Create script to reproduce experiment"""
        script_content = f"""#!/usr/bin/env python3
'''
Reproduction script for experiment: {self.experiment_name}
Generated automatically by GAN-Cyber-Range-v2
'''

import json
from pathlib import Path

def reproduce_experiment():
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    print(f"Reproducing experiment: {self.experiment_name}")
    print(f"Configuration: {{config}}")
    
    # TODO: Add specific reproduction code based on experiment type
    # This would be customized based on the experiment configuration
    
    print("Experiment reproduction completed")

if __name__ == "__main__":
    reproduce_experiment()
"""
        
        with open(output_path / 'reproduce.py', 'w') as f:
            f.write(script_content)
        
        # Make executable
        (output_path / 'reproduce.py').chmod(0o755)
```

### Benchmarking Framework

```python
class CybersecurityBenchmark:
    """Standardized benchmark for cybersecurity research"""
    
    def __init__(self, name: str):
        self.name = name
        self.tasks = {}
        self.baselines = {}
        self.metrics = {}
    
    def add_task(self, task_name: str, task_config: Dict[str, Any]):
        """Add benchmark task"""
        self.tasks[task_name] = {
            'config': task_config,
            'dataset': task_config['dataset'],
            'evaluation_metric': task_config['metric'],
            'baseline_models': task_config.get('baselines', [])
        }
    
    def register_baseline(self, task_name: str, model_name: str, performance: float):
        """Register baseline performance"""
        if task_name not in self.baselines:
            self.baselines[task_name] = {}
        
        self.baselines[task_name][model_name] = performance
    
    def evaluate_model(self, task_name: str, model, test_data):
        """Evaluate model on benchmark task"""
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} not found")
        
        task = self.tasks[task_name]
        metric_func = self._get_metric_function(task['evaluation_metric'])
        
        # Run evaluation
        predictions = model.predict(test_data)
        performance = metric_func(test_data.labels, predictions)
        
        # Compare to baselines
        baselines = self.baselines.get(task_name, {})
        
        results = {
            'performance': performance,
            'baselines': baselines,
            'improvement': {}
        }
        
        for baseline_name, baseline_perf in baselines.items():
            improvement = (performance - baseline_perf) / baseline_perf * 100
            results['improvement'][baseline_name] = improvement
        
        return results
    
    def _get_metric_function(self, metric_name: str):
        """Get evaluation metric function"""
        metrics = {
            'accuracy': lambda y_true, y_pred: np.mean(y_true == y_pred),
            'f1_score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
            'auc_roc': lambda y_true, y_pred: roc_auc_score(y_true, y_pred),
            'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
            'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted')
        }
        
        return metrics[metric_name]

# Standard cybersecurity benchmarks
MALWARE_DETECTION_BENCHMARK = CybersecurityBenchmark("malware_detection")
MALWARE_DETECTION_BENCHMARK.add_task("binary_classification", {
    'dataset': 'malware_dataset_v1',
    'metric': 'f1_score',
    'baselines': ['random_forest', 'svm', 'neural_network']
})

NETWORK_INTRUSION_BENCHMARK = CybersecurityBenchmark("network_intrusion")
NETWORK_INTRUSION_BENCHMARK.add_task("multiclass_classification", {
    'dataset': 'nsl_kdd',
    'metric': 'accuracy',
    'baselines': ['traditional_ids', 'ml_ids']
})

ATTACK_GENERATION_BENCHMARK = CybersecurityBenchmark("attack_generation")
ATTACK_GENERATION_BENCHMARK.add_task("diversity_assessment", {
    'dataset': 'mitre_attack_patterns',
    'metric': 'diversity_score',
    'baselines': ['template_based', 'rule_based']
})
```

### Research Publication Framework

```python
class ResearchPaper:
    """Framework for academic paper generation"""
    
    def __init__(self, title: str, authors: List[str]):
        self.title = title
        self.authors = authors
        self.abstract = ""
        self.sections = {}
        self.experiments = []
        self.references = []
        self.figures = {}
        self.tables = {}
    
    def add_experiment(self, experiment_tracker: ExperimentTracker):
        """Add experiment results to paper"""
        self.experiments.append(experiment_tracker)
    
    def generate_results_section(self) -> str:
        """Auto-generate results section from experiments"""
        results_text = "## Results\n\n"
        
        for i, experiment in enumerate(self.experiments):
            results_text += f"### Experiment {i+1}: {experiment.experiment_name}\n\n"
            
            # Add configuration
            results_text += "**Configuration:**\n"
            for key, value in experiment.config.items():
                results_text += f"- {key}: {value}\n"
            results_text += "\n"
            
            # Add key results
            results_text += "**Results:**\n"
            for metric, values in experiment.results.items():
                if values:
                    final_value = values[-1]['value']
                    results_text += f"- {metric}: {final_value:.4f}\n"
            results_text += "\n"
        
        return results_text
    
    def generate_latex(self) -> str:
        """Generate LaTeX paper template"""
        latex_template = f"""\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}

\\title{{{self.title}}}
\\author{{{' \\and '.join(self.authors)}}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{self.abstract}
\\end{{abstract}}

\\section{{Introduction}}
{self.sections.get('introduction', 'TODO: Add introduction')}

\\section{{Methodology}}
{self.sections.get('methodology', 'TODO: Add methodology')}

{self.generate_results_section()}

\\section{{Discussion}}
{self.sections.get('discussion', 'TODO: Add discussion')}

\\section{{Conclusion}}
{self.sections.get('conclusion', 'TODO: Add conclusion')}

\\bibliographystyle{{plain}}
\\bibliography{{references}}

\\end{{document}}
"""
        return latex_template
    
    def export_paper_package(self, output_dir: str):
        """Export complete paper package with data and code"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export LaTeX
        with open(output_path / 'paper.tex', 'w') as f:
            f.write(self.generate_latex())
        
        # Export experiment data
        experiments_dir = output_path / 'experiments'
        experiments_dir.mkdir(exist_ok=True)
        
        for experiment in self.experiments:
            experiment.export_experiment(str(experiments_dir))
        
        # Create reproduction instructions
        self._create_reproduction_readme(output_path)
    
    def _create_reproduction_readme(self, output_path: Path):
        """Create README for paper reproduction"""
        readme_content = f"""# Reproduction Package for: {self.title}

## Authors
{chr(10).join(f'- {author}' for author in self.authors)}

## Overview
This package contains all code, data, and configuration needed to reproduce the results in this paper.

## Structure
- `paper.tex`: Main paper LaTeX source
- `experiments/`: Individual experiment packages
- `requirements.txt`: Python dependencies
- `reproduce_all.py`: Script to reproduce all experiments

## Reproduction Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run all experiments:
   ```bash
   python reproduce_all.py
   ```

3. Individual experiments can be run:
   ```bash
   cd experiments/[experiment_id]
   python reproduce.py
   ```

## Expected Runtime
Total reproduction time: Approximately X hours on recommended hardware.

## Hardware Requirements
- Minimum: 8 CPU cores, 16GB RAM
- Recommended: 16 CPU cores, 32GB RAM, GPU with 8GB+ VRAM

## Citation
If you use this work, please cite:

```bibtex
@article{{{self.title.lower().replace(' ', '_')}_2024,
  title={{{self.title}}},
  author={{{' and '.join(self.authors)}}},
  journal={{Computer Security Research}},
  year={{2024}}
}}
```
"""
        
        with open(output_path / 'README.md', 'w') as f:
            f.write(readme_content)
```

### Statistical Analysis Framework

```python
class StatisticalAnalyzer:
    """Comprehensive statistical analysis for cybersecurity research"""
    
    @staticmethod
    def power_analysis(effect_size: float, alpha: float = 0.05, power: float = 0.8):
        """Calculate required sample size for statistical power"""
        from statsmodels.stats.power import ttest_power
        
        sample_size = ttest_power(effect_size, power, alpha, alternative='two-sided')
        return int(np.ceil(sample_size))
    
    @staticmethod
    def effect_size_analysis(group1: List[float], group2: List[float]):
        """Calculate effect size (Cohen's d)"""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        
        cohens_d = (mean1 - mean2) / pooled_std
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            interpretation = "small"
        elif abs(cohens_d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return {
            'cohens_d': cohens_d,
            'interpretation': interpretation,
            'magnitude': abs(cohens_d)
        }
    
    @staticmethod
    def multiple_comparison_correction(p_values: List[float], method: str = 'bonferroni'):
        """Apply multiple comparison correction"""
        from statsmodels.stats.multitest import multipletests
        
        rejected, corrected_p, _, _ = multipletests(
            p_values, alpha=0.05, method=method
        )
        
        return {
            'rejected': rejected,
            'corrected_p_values': corrected_p,
            'method': method
        }
    
    @staticmethod
    def bootstrap_confidence_interval(data: List[float], confidence: float = 0.95, n_bootstrap: int = 10000):
        """Calculate bootstrap confidence interval"""
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return {
            'mean': np.mean(data),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence': confidence
        }
```

## Research Ethics and Compliance

### Institutional Review Board (IRB) Framework

```python
class ResearchEthicsFramework:
    """Ensure research compliance with ethical guidelines"""
    
    def __init__(self):
        self.approved_studies = set()
        self.ethical_guidelines = self._load_guidelines()
    
    def submit_for_review(self, study_protocol: Dict[str, Any]) -> str:
        """Submit study for ethical review"""
        review_id = f"IRB_{int(time.time())}"
        
        # Automated pre-screening
        issues = self._prescreen_study(study_protocol)
        
        if not issues:
            self.approved_studies.add(review_id)
            return review_id
        else:
            raise ValueError(f"Ethical issues identified: {issues}")
    
    def _prescreen_study(self, protocol: Dict[str, Any]) -> List[str]:
        """Pre-screen for obvious ethical issues"""
        issues = []
        
        # Check for human subjects
        if protocol.get('involves_human_subjects', False):
            if not protocol.get('informed_consent_process'):
                issues.append("Missing informed consent process")
            
            if not protocol.get('data_anonymization'):
                issues.append("Missing data anonymization plan")
        
        # Check for harmful activities
        if protocol.get('involves_real_attacks', False):
            if not protocol.get('target_consent'):
                issues.append("Real attacks require explicit target consent")
        
        # Check data handling
        if not protocol.get('data_retention_policy'):
            issues.append("Missing data retention policy")
        
        return issues
    
    def _load_guidelines(self) -> Dict[str, Any]:
        """Load ethical guidelines"""
        return {
            'principle_beneficence': "Research should benefit society",
            'principle_non_maleficence': "Research should not cause harm",
            'principle_autonomy': "Respect participant autonomy",
            'principle_justice': "Fair distribution of research benefits/burdens"
        }
```

## Research Collaboration Framework

### Multi-Institutional Studies

```python
class MultiInstitutionalStudy:
    """Framework for collaborative research across institutions"""
    
    def __init__(self, study_name: str, lead_institution: str):
        self.study_name = study_name
        self.lead_institution = lead_institution
        self.participating_institutions = []
        self.data_sharing_agreements = {}
        self.federated_experiments = []
    
    def add_institution(self, institution: str, data_sharing_agreement: Dict[str, Any]):
        """Add participating institution"""
        self.participating_institutions.append(institution)
        self.data_sharing_agreements[institution] = data_sharing_agreement
    
    def create_federated_experiment(self, experiment_config: Dict[str, Any]) -> str:
        """Create experiment that runs across institutions"""
        experiment_id = f"federated_{int(time.time())}"
        
        federated_config = {
            'experiment_id': experiment_id,
            'config': experiment_config,
            'institutions': self.participating_institutions,
            'aggregation_method': experiment_config.get('aggregation', 'weighted_average'),
            'privacy_budget': experiment_config.get('privacy_budget', 10.0)
        }
        
        self.federated_experiments.append(federated_config)
        return experiment_id
    
    def aggregate_results(self, experiment_id: str, institution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from multiple institutions"""
        experiment = next(
            (exp for exp in self.federated_experiments if exp['experiment_id'] == experiment_id),
            None
        )
        
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Weighted average aggregation
        total_samples = sum(results['sample_count'] for results in institution_results.values())
        
        aggregated_metrics = {}
        for metric in institution_results[list(institution_results.keys())[0]]['metrics']:
            weighted_sum = sum(
                results['metrics'][metric] * results['sample_count']
                for results in institution_results.values()
            )
            aggregated_metrics[metric] = weighted_sum / total_samples
        
        return {
            'aggregated_metrics': aggregated_metrics,
            'total_samples': total_samples,
            'participating_institutions': len(institution_results),
            'experiment_id': experiment_id
        }
```

This research methodology framework provides the foundation for conducting rigorous, reproducible, and ethically sound cybersecurity research using GAN-Cyber-Range-v2.