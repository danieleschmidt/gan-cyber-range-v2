"""
Attack quality evaluation and metrics for GAN-generated attacks.

This module provides comprehensive evaluation of synthetic attack quality,
including realism, diversity, sophistication, and detectability assessments.
"""

import logging
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from pathlib import Path

from ..core.attack_gan import AttackVector

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Attack quality evaluation report"""
    overall_score: float
    realism_score: float
    diversity_score: float
    sophistication_score: float
    detectability_score: float
    impact_score: float
    num_attacks_evaluated: int
    evaluation_timestamp: str
    detailed_metrics: Dict[str, Any]


class RealismScorer:
    """Evaluates how realistic generated attacks are compared to real attack data"""
    
    def __init__(self, reference_dataset: str = "mitre_attack"):
        self.reference_dataset = reference_dataset
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.reference_vectors = None
        self.reference_attacks = []
        
        # Load reference attack patterns
        self._load_reference_data()
    
    def score(self, attacks: List[AttackVector]) -> float:
        """Calculate realism score for attacks"""
        if not self.reference_vectors:
            logger.warning("No reference data loaded, using fallback scoring")
            return self._fallback_realism_score(attacks)
        
        # Convert attacks to text representations
        attack_texts = [self._attack_to_text(attack) for attack in attacks]
        
        # Vectorize generated attacks
        attack_vectors = self.vectorizer.transform(attack_texts)
        
        # Calculate similarity to reference attacks
        similarities = cosine_similarity(attack_vectors, self.reference_vectors)
        
        # Score is average maximum similarity for each generated attack
        max_similarities = np.max(similarities, axis=1)
        realism_score = np.mean(max_similarities)
        
        logger.info(f"Realism score: {realism_score:.3f}")
        return float(realism_score)
    
    def _load_reference_data(self) -> None:
        """Load reference attack data"""
        # This would load real MITRE ATT&CK data or other reference datasets
        # For now, using synthetic reference data
        reference_attacks = [
            "Initial access through spear phishing email with malicious attachment",
            "Lateral movement using stolen credentials via RDP",
            "Privilege escalation through unpatched Windows vulnerability",
            "Data exfiltration over encrypted C2 channel",
            "Persistence through registry modification and scheduled tasks",
            "Defense evasion using process hollowing technique",
            "Discovery through network service scanning",
            "Collection of sensitive files from file shares",
            "Command and control via DNS tunneling",
            "Impact through data encryption ransomware deployment"
        ]
        
        self.reference_attacks = reference_attacks
        
        try:
            # Fit vectorizer on reference data
            self.reference_vectors = self.vectorizer.fit_transform(reference_attacks)
            logger.info(f"Loaded {len(reference_attacks)} reference attack patterns")
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")
    
    def _attack_to_text(self, attack: AttackVector) -> str:
        """Convert attack vector to text representation"""
        text_parts = [
            attack.attack_type,
            str(attack.payload) if isinstance(attack.payload, str) else "",
            " ".join(attack.techniques),
            " ".join(attack.target_systems)
        ]
        return " ".join(filter(None, text_parts))
    
    def _fallback_realism_score(self, attacks: List[AttackVector]) -> float:
        """Fallback realism scoring when no reference data available"""
        # Simple heuristic-based scoring
        scores = []
        
        for attack in attacks:
            score = 0.5  # Base score
            
            # Bonus for having MITRE ATT&CK techniques
            if attack.techniques and any(t.startswith('T') for t in attack.techniques):
                score += 0.2
            
            # Bonus for realistic attack types
            realistic_types = ['malware', 'network', 'web', 'social_engineering']
            if attack.attack_type in realistic_types:
                score += 0.1
            
            # Bonus for reasonable severity/stealth levels
            if 0.1 <= attack.severity <= 1.0 and 0.1 <= attack.stealth_level <= 1.0:
                score += 0.1
            
            # Penalty for unrealistic combinations
            if attack.stealth_level > 0.8 and attack.severity > 0.9:
                score -= 0.1  # High stealth + high impact is rare
            
            scores.append(min(1.0, max(0.0, score)))
        
        return np.mean(scores) if scores else 0.0


class DiversityScorer:
    """Evaluates diversity of generated attacks"""
    
    def __init__(self, method: str = "embedding_distance"):
        self.method = method
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    
    def score(self, attacks: List[AttackVector]) -> float:
        """Calculate diversity score for attacks"""
        if len(attacks) < 2:
            return 0.0
        
        if self.method == "embedding_distance":
            return self._embedding_diversity(attacks)
        elif self.method == "categorical_diversity":
            return self._categorical_diversity(attacks)
        else:
            return self._combined_diversity(attacks)
    
    def _embedding_diversity(self, attacks: List[AttackVector]) -> float:
        """Calculate diversity based on text embedding distances"""
        # Convert attacks to text
        attack_texts = [self._attack_to_text(attack) for attack in attacks]
        
        # Vectorize attacks
        try:
            vectors = self.vectorizer.fit_transform(attack_texts)
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(vectors)
            
            # Remove self-similarities (diagonal)
            n = len(attacks)
            mask = ~np.eye(n, dtype=bool)
            similarities = similarities[mask]
            
            # Diversity is 1 - average similarity
            diversity = 1.0 - np.mean(similarities)
            return max(0.0, min(1.0, diversity))
            
        except Exception as e:
            logger.warning(f"Failed to calculate embedding diversity: {e}")
            return self._categorical_diversity(attacks)
    
    def _categorical_diversity(self, attacks: List[AttackVector]) -> float:
        """Calculate diversity based on categorical features"""
        # Count unique values in different categories
        attack_types = set(attack.attack_type for attack in attacks)
        techniques = set()
        for attack in attacks:
            techniques.update(attack.techniques)
        target_systems = set()
        for attack in attacks:
            target_systems.update(attack.target_systems)
        
        # Calculate diversity metrics
        type_diversity = len(attack_types) / max(1, len(attacks))
        technique_diversity = len(techniques) / max(1, len(attacks) * 3)  # Assume max 3 techniques per attack
        target_diversity = len(target_systems) / max(1, len(attacks) * 2)  # Assume max 2 targets per attack
        
        # Combined diversity score
        diversity = (type_diversity + technique_diversity + target_diversity) / 3
        return min(1.0, diversity)
    
    def _combined_diversity(self, attacks: List[AttackVector]) -> float:
        """Combine embedding and categorical diversity"""
        embedding_div = self._embedding_diversity(attacks)
        categorical_div = self._categorical_diversity(attacks)
        return (embedding_div + categorical_div) / 2
    
    def _attack_to_text(self, attack: AttackVector) -> str:
        """Convert attack vector to text representation"""
        text_parts = [
            attack.attack_type,
            str(attack.payload) if isinstance(attack.payload, str) else "",
            " ".join(attack.techniques),
            " ".join(attack.target_systems)
        ]
        return " ".join(filter(None, text_parts))


class SophisticationScorer:
    """Evaluates sophistication level of generated attacks"""
    
    def __init__(self, complexity_model: str = "rule_based"):
        self.complexity_model = complexity_model
        
        # Sophistication weights for different attack characteristics
        self.technique_weights = {
            'T1003': 0.9,  # Credential Dumping
            'T1055': 0.8,  # Process Injection  
            'T1070': 0.7,  # Indicator Removal
            'T1041': 0.7,  # Exfiltration Over C2
            'T1486': 0.6,  # Data Encrypted for Impact
            'T1078': 0.5,  # Valid Accounts
            'T1190': 0.6,  # Exploit Public-Facing Application
            'T1059': 0.4,  # Command Line Interface
            'T1046': 0.3,  # Network Service Scanning
        }
    
    def score(self, attacks: List[AttackVector]) -> float:
        """Calculate sophistication score for attacks"""
        if not attacks:
            return 0.0
        
        scores = []
        for attack in attacks:
            score = self._score_single_attack(attack)
            scores.append(score)
        
        return np.mean(scores)
    
    def _score_single_attack(self, attack: AttackVector) -> float:
        """Score sophistication of a single attack"""
        score = 0.0
        
        # Base score from techniques used
        technique_score = 0.0
        if attack.techniques:
            for technique in attack.techniques:
                technique_score += self.technique_weights.get(technique, 0.2)
            technique_score /= len(attack.techniques)
        
        # Multi-technique bonus
        multi_technique_bonus = min(0.2, len(attack.techniques) * 0.05)
        
        # Stealth level contribution
        stealth_contribution = attack.stealth_level * 0.3
        
        # Severity contribution (sophisticated attacks often have high impact)
        severity_contribution = attack.severity * 0.2
        
        # Complex payload bonus
        payload_complexity = self._assess_payload_complexity(attack.payload)
        
        score = (technique_score + multi_technique_bonus + 
                stealth_contribution + severity_contribution + payload_complexity)
        
        return min(1.0, max(0.0, score))
    
    def _assess_payload_complexity(self, payload: Union[str, bytes, Dict]) -> float:
        """Assess complexity of attack payload"""
        if isinstance(payload, dict):
            # Complex payloads have multiple parameters
            return min(0.2, len(payload) * 0.02)
        elif isinstance(payload, str):
            # Longer, more complex strings indicate sophistication
            complexity_indicators = ['base64', 'powershell', 'encoded', 'obfuscated', 'shell']
            indicator_count = sum(1 for indicator in complexity_indicators if indicator in payload.lower())
            return min(0.2, indicator_count * 0.04)
        else:
            return 0.1  # Default for bytes or unknown types


class DetectabilityScorer:
    """Evaluates how detectable generated attacks are"""
    
    def __init__(self, detection_models: List[str] = None):
        self.detection_models = detection_models or ["signature_based", "behavioral", "ml_based"]
        
        # Detection probability by technique (based on common security tools)
        self.detection_probabilities = {
            'T1059': 0.7,  # Command Line - easily detected
            'T1046': 0.8,  # Network Scanning - very detectable
            'T1190': 0.6,  # Web Exploits - moderately detectable
            'T1078': 0.4,  # Valid Accounts - hard to detect
            'T1055': 0.5,  # Process Injection - moderate
            'T1003': 0.8,  # Credential Dumping - highly detectable
            'T1070': 0.3,  # Indicator Removal - designed to evade
            'T1041': 0.5,  # Exfiltration - depends on method
            'T1486': 0.9,  # Ransomware - very detectable impact
        }
    
    def score(self, attacks: List[AttackVector]) -> float:
        """Calculate detectability score (higher = more detectable)"""
        if not attacks:
            return 0.0
        
        scores = []
        for attack in attacks:
            score = self._score_single_attack(attack)
            scores.append(score)
        
        return np.mean(scores)
    
    def _score_single_attack(self, attack: AttackVector) -> float:
        """Score detectability of a single attack"""
        base_detection = 0.5  # Default detection probability
        
        # Technique-based detection probability
        technique_detection = 0.0
        if attack.techniques:
            for technique in attack.techniques:
                technique_detection += self.detection_probabilities.get(technique, 0.5)
            technique_detection /= len(attack.techniques)
        else:
            technique_detection = base_detection
        
        # Stealth level reduces detectability
        stealth_modifier = 1.0 - (attack.stealth_level * 0.4)
        
        # High severity attacks are often more detectable
        severity_modifier = 1.0 + (attack.severity * 0.2)
        
        # Multiple techniques increase detectability
        multi_technique_penalty = min(0.2, len(attack.techniques) * 0.03)
        
        detectability = (technique_detection * stealth_modifier * severity_modifier + 
                        multi_technique_penalty)
        
        return min(1.0, max(0.0, detectability))


class ImpactScorer:
    """Evaluates potential impact of generated attacks"""
    
    def __init__(self, damage_model: str = "cvss_based"):
        self.damage_model = damage_model
        
        # Impact weights by attack type
        self.impact_weights = {
            'malware': 0.8,
            'ransomware': 0.9,
            'data_breach': 0.85,
            'network': 0.6,
            'web': 0.7,
            'social_engineering': 0.75,
            'insider_threat': 0.8
        }
    
    def score(self, attacks: List[AttackVector]) -> float:
        """Calculate impact score for attacks"""
        if not attacks:
            return 0.0
        
        scores = []
        for attack in attacks:
            score = self._score_single_attack(attack)
            scores.append(score)
        
        return np.mean(scores)
    
    def _score_single_attack(self, attack: AttackVector) -> float:
        """Score impact of a single attack"""
        # Base impact from attack type
        base_impact = self.impact_weights.get(attack.attack_type, 0.5)
        
        # Severity directly contributes to impact
        severity_impact = attack.severity
        
        # Techniques that cause direct damage
        high_impact_techniques = ['T1486', 'T1485', 'T1489', 'T1490']  # Data destruction/encryption
        technique_impact = 0.0
        if attack.techniques:
            for technique in attack.techniques:
                if technique in high_impact_techniques:
                    technique_impact += 0.2
        
        # Crown jewel targets increase impact
        crown_jewel_bonus = 0.0
        if 'database' in attack.target_systems or 'financial' in attack.target_systems:
            crown_jewel_bonus = 0.2
        
        total_impact = (base_impact + severity_impact + technique_impact + crown_jewel_bonus) / 4
        return min(1.0, max(0.0, total_impact))


class AttackQualityEvaluator:
    """Main attack quality evaluator"""
    
    def __init__(self):
        self.realism_scorer = RealismScorer()
        self.diversity_scorer = DiversityScorer()
        self.sophistication_scorer = SophisticationScorer()
        self.detectability_scorer = DetectabilityScorer()
        self.impact_scorer = ImpactScorer()
    
    def evaluate(
        self,
        attacks: List[AttackVector],
        metrics: Dict[str, Any] = None
    ) -> QualityReport:
        """Comprehensive evaluation of attack quality"""
        
        if not attacks:
            return QualityReport(
                overall_score=0.0,
                realism_score=0.0,
                diversity_score=0.0,
                sophistication_score=0.0,
                detectability_score=0.0,
                impact_score=0.0,
                num_attacks_evaluated=0,
                evaluation_timestamp=str(np.datetime64('now')),
                detailed_metrics={}
            )
        
        logger.info(f"Evaluating {len(attacks)} attacks")
        
        # Calculate individual scores
        realism_score = self.realism_scorer.score(attacks)
        diversity_score = self.diversity_scorer.score(attacks)
        sophistication_score = self.sophistication_scorer.score(attacks)
        detectability_score = self.detectability_scorer.score(attacks)
        impact_score = self.impact_scorer.score(attacks)
        
        # Calculate overall score (weighted average)
        weights = {
            'realism': 0.25,
            'diversity': 0.20,
            'sophistication': 0.20,
            'detectability': 0.15,  # Lower weight as high detectability isn't always good
            'impact': 0.20
        }
        
        overall_score = (
            realism_score * weights['realism'] +
            diversity_score * weights['diversity'] +
            sophistication_score * weights['sophistication'] +
            (1.0 - detectability_score) * weights['detectability'] +  # Invert detectability
            impact_score * weights['impact']
        )
        
        # Detailed metrics
        detailed_metrics = {
            'weights_used': weights,
            'attack_type_distribution': self._analyze_attack_types(attacks),
            'technique_distribution': self._analyze_techniques(attacks),
            'severity_statistics': self._analyze_severity(attacks),
            'stealth_statistics': self._analyze_stealth(attacks)
        }
        
        report = QualityReport(
            overall_score=overall_score,
            realism_score=realism_score,
            diversity_score=diversity_score,
            sophistication_score=sophistication_score,
            detectability_score=detectability_score,
            impact_score=impact_score,
            num_attacks_evaluated=len(attacks),
            evaluation_timestamp=str(np.datetime64('now')),
            detailed_metrics=detailed_metrics
        )
        
        logger.info(f"Attack evaluation completed - Overall score: {overall_score:.3f}")
        return report
    
    def generate_report(
        self,
        quality_report: QualityReport,
        output_format: str = "json",
        save_path: Optional[str] = None
    ) -> str:
        """Generate formatted quality report"""
        
        if output_format == "json":
            report_data = {
                'overall_score': quality_report.overall_score,
                'component_scores': {
                    'realism': quality_report.realism_score,
                    'diversity': quality_report.diversity_score,
                    'sophistication': quality_report.sophistication_score,
                    'detectability': quality_report.detectability_score,
                    'impact': quality_report.impact_score
                },
                'metadata': {
                    'num_attacks_evaluated': quality_report.num_attacks_evaluated,
                    'evaluation_timestamp': quality_report.evaluation_timestamp
                },
                'detailed_metrics': quality_report.detailed_metrics
            }
            
            report_text = json.dumps(report_data, indent=2)
            
        elif output_format == "latex":
            report_text = self._generate_latex_report(quality_report)
        else:
            report_text = self._generate_text_report(quality_report)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {save_path}")
        
        return report_text
    
    def _analyze_attack_types(self, attacks: List[AttackVector]) -> Dict[str, int]:
        """Analyze distribution of attack types"""
        type_counts = {}
        for attack in attacks:
            type_counts[attack.attack_type] = type_counts.get(attack.attack_type, 0) + 1
        return type_counts
    
    def _analyze_techniques(self, attacks: List[AttackVector]) -> Dict[str, int]:
        """Analyze distribution of MITRE ATT&CK techniques"""
        technique_counts = {}
        for attack in attacks:
            for technique in attack.techniques:
                technique_counts[technique] = technique_counts.get(technique, 0) + 1
        return technique_counts
    
    def _analyze_severity(self, attacks: List[AttackVector]) -> Dict[str, float]:
        """Analyze severity statistics"""
        severities = [attack.severity for attack in attacks]
        return {
            'mean': float(np.mean(severities)),
            'std': float(np.std(severities)),
            'min': float(np.min(severities)),
            'max': float(np.max(severities))
        }
    
    def _analyze_stealth(self, attacks: List[AttackVector]) -> Dict[str, float]:
        """Analyze stealth level statistics"""
        stealth_levels = [attack.stealth_level for attack in attacks]
        return {
            'mean': float(np.mean(stealth_levels)),
            'std': float(np.std(stealth_levels)),
            'min': float(np.min(stealth_levels)),
            'max': float(np.max(stealth_levels))
        }
    
    def _generate_latex_report(self, quality_report: QualityReport) -> str:
        """Generate LaTeX formatted report"""
        return f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\title{{Attack Quality Evaluation Report}}
\\date{{{quality_report.evaluation_timestamp}}}

\\begin{{document}}
\\maketitle

\\section{{Executive Summary}}
Overall Quality Score: {quality_report.overall_score:.3f}/1.000

\\section{{Component Scores}}
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lr}}
\\toprule
Metric & Score \\\\
\\midrule
Realism & {quality_report.realism_score:.3f} \\\\
Diversity & {quality_report.diversity_score:.3f} \\\\
Sophistication & {quality_report.sophistication_score:.3f} \\\\
Detectability & {quality_report.detectability_score:.3f} \\\\
Impact & {quality_report.impact_score:.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Attack Quality Component Scores}}
\\end{{table}}

\\section{{Evaluation Details}}
Number of attacks evaluated: {quality_report.num_attacks_evaluated}

\\end{{document}}"""
    
    def _generate_text_report(self, quality_report: QualityReport) -> str:
        """Generate plain text report"""
        return f"""Attack Quality Evaluation Report
Generated: {quality_report.evaluation_timestamp}
Attacks Evaluated: {quality_report.num_attacks_evaluated}

OVERALL SCORE: {quality_report.overall_score:.3f}/1.000

COMPONENT SCORES:
- Realism:        {quality_report.realism_score:.3f}
- Diversity:      {quality_report.diversity_score:.3f}
- Sophistication: {quality_report.sophistication_score:.3f}
- Detectability:  {quality_report.detectability_score:.3f}
- Impact:         {quality_report.impact_score:.3f}

DETAILED METRICS:
{json.dumps(quality_report.detailed_metrics, indent=2)}
"""