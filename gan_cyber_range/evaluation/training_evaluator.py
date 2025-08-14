"""
Training effectiveness evaluation and blue team performance measurement.

This module provides comprehensive evaluation of cybersecurity training programs,
measuring skill improvement, knowledge retention, and real-world applicability.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Individual performance metrics for a team member"""
    team_member_id: str
    detection_accuracy: float
    response_time_avg: float  # minutes
    false_positive_rate: float
    incident_handling_score: float
    technical_skill_score: float
    knowledge_retention_score: float
    improvement_rate: float
    scenarios_completed: int
    total_training_time: float  # hours


@dataclass
class TeamPerformance:
    """Team-level performance metrics"""
    team_id: str
    overall_score: float
    detection_rate: float
    mean_time_to_detect: float  # minutes
    mean_time_to_respond: float  # minutes
    false_positive_rate: float
    coverage_completeness: float
    collaboration_score: float
    individual_performances: List[PerformanceMetrics]
    evaluation_timestamp: str


@dataclass
class TrainingProgram:
    """Training program definition and tracking"""
    program_id: str
    name: str
    duration: timedelta
    scenarios: List[Dict[str, Any]]
    learning_objectives: List[str]
    difficulty_progression: str
    target_skills: List[str]
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ImprovementAnalysis:
    """Analysis of training effectiveness and improvement"""
    overall_improvement: float
    detection_improvement: float
    response_time_improvement: float
    skill_improvements: Dict[str, float]
    knowledge_retention: float
    confidence_interval: Tuple[float, float]
    statistical_significance: bool
    p_value: float


class TrainingEffectiveness:
    """Main training effectiveness evaluator"""
    
    def __init__(self):
        self.baseline_scenarios = [
            'ransomware_detection',
            'apt_lateral_movement', 
            'insider_threat',
            'phishing_response',
            'malware_analysis',
            'network_intrusion',
            'data_exfiltration',
            'privilege_escalation'
        ]
        
        # Standard performance benchmarks
        self.performance_benchmarks = {
            'detection_rate': {'novice': 0.6, 'intermediate': 0.75, 'expert': 0.9},
            'false_positive_rate': {'novice': 0.3, 'intermediate': 0.15, 'expert': 0.05},
            'response_time': {'novice': 30, 'intermediate': 15, 'expert': 5},  # minutes
            'technical_skill': {'novice': 0.5, 'intermediate': 0.7, 'expert': 0.9}
        }
    
    def assess_team(
        self,
        team_id: str,
        scenarios: List[str] = None,
        duration_hours: float = 8.0
    ) -> TeamPerformance:
        """Assess team performance across scenarios"""
        
        if scenarios is None:
            scenarios = self.baseline_scenarios
        
        logger.info(f"Assessing team {team_id} across {len(scenarios)} scenarios")
        
        # Simulate team assessment (in real implementation, this would run actual scenarios)
        team_performance = self._simulate_team_assessment(team_id, scenarios, duration_hours)
        
        return team_performance
    
    def calculate_improvement(
        self,
        pre_scores: TeamPerformance,
        post_scores: TeamPerformance
    ) -> ImprovementAnalysis:
        """Calculate improvement between pre and post training assessments"""
        
        logger.info(f"Calculating improvement for team {pre_scores.team_id}")
        
        # Overall improvement
        overall_improvement = post_scores.overall_score - pre_scores.overall_score
        
        # Specific metric improvements
        detection_improvement = post_scores.detection_rate - pre_scores.detection_rate
        
        # Response time improvement (negative means faster response)
        response_time_improvement = (pre_scores.mean_time_to_respond - post_scores.mean_time_to_respond) / pre_scores.mean_time_to_respond
        
        # Individual skill improvements
        skill_improvements = self._calculate_skill_improvements(pre_scores, post_scores)
        
        # Knowledge retention (based on consistency of performance)
        knowledge_retention = self._calculate_knowledge_retention(post_scores)
        
        # Statistical analysis
        statistical_significance, p_value, confidence_interval = self._statistical_analysis(
            pre_scores, post_scores
        )
        
        improvement = ImprovementAnalysis(
            overall_improvement=overall_improvement,
            detection_improvement=detection_improvement,
            response_time_improvement=response_time_improvement,
            skill_improvements=skill_improvements,
            knowledge_retention=knowledge_retention,
            confidence_interval=confidence_interval,
            statistical_significance=statistical_significance,
            p_value=p_value
        )
        
        logger.info(f"Overall improvement: {overall_improvement:.1%}")
        return improvement
    
    def generate_training_recommendations(
        self,
        team_performance: TeamPerformance,
        improvement_analysis: Optional[ImprovementAnalysis] = None
    ) -> Dict[str, Any]:
        """Generate personalized training recommendations"""
        
        recommendations = {
            'team_recommendations': [],
            'individual_recommendations': {},
            'priority_areas': [],
            'suggested_scenarios': [],
            'estimated_training_time': 0
        }
        
        # Analyze team weaknesses
        weak_areas = []
        if team_performance.detection_rate < 0.7:
            weak_areas.append('detection_skills')
        if team_performance.mean_time_to_respond > 20:
            weak_areas.append('response_speed')
        if team_performance.false_positive_rate > 0.2:
            weak_areas.append('alert_analysis')
        if team_performance.collaboration_score < 0.7:
            weak_areas.append('team_coordination')
        
        # Generate team-level recommendations
        for area in weak_areas:
            recommendation = self._generate_area_recommendation(area)
            recommendations['team_recommendations'].append(recommendation)
        
        # Individual recommendations
        for individual in team_performance.individual_performances:
            individual_recs = self._generate_individual_recommendations(individual)
            recommendations['individual_recommendations'][individual.team_member_id] = individual_recs
        
        # Priority areas
        recommendations['priority_areas'] = weak_areas[:3]  # Top 3 priorities
        
        # Suggested scenarios
        recommendations['suggested_scenarios'] = self._suggest_scenarios(weak_areas)
        
        # Estimated training time
        recommendations['estimated_training_time'] = len(weak_areas) * 4  # 4 hours per weak area
        
        return recommendations
    
    def track_progress(
        self,
        team_id: str,
        training_sessions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Track team progress over multiple training sessions"""
        
        progress_data = {
            'team_id': team_id,
            'sessions_completed': len(training_sessions),
            'total_training_time': sum(session.get('duration', 0) for session in training_sessions),
            'skill_progression': {},
            'performance_trend': [],
            'learning_velocity': 0.0,
            'plateau_detection': False
        }
        
        # Analyze performance trend
        scores = [session.get('overall_score', 0) for session in training_sessions]
        if len(scores) >= 2:
            # Calculate learning velocity (improvement rate)
            progress_data['learning_velocity'] = (scores[-1] - scores[0]) / len(scores)
            
            # Detect learning plateau
            recent_scores = scores[-3:] if len(scores) >= 3 else scores
            if len(recent_scores) >= 3:
                variance = np.var(recent_scores)
                progress_data['plateau_detection'] = variance < 0.01  # Low variance indicates plateau
        
        progress_data['performance_trend'] = scores
        
        return progress_data
    
    def _simulate_team_assessment(
        self,
        team_id: str,
        scenarios: List[str],
        duration_hours: float
    ) -> TeamPerformance:
        """Simulate team assessment (placeholder for real assessment)"""
        
        # Generate realistic team performance data
        np.random.seed(hash(team_id) % 2**32)  # Consistent results for same team
        
        # Team-level metrics
        base_skill_level = np.random.uniform(0.4, 0.8)
        detection_rate = min(1.0, base_skill_level + np.random.normal(0, 0.1))
        
        mean_time_to_detect = max(1.0, 20 * (1.5 - base_skill_level) + np.random.normal(0, 5))
        mean_time_to_respond = max(1.0, 15 * (1.5 - base_skill_level) + np.random.normal(0, 3))
        
        false_positive_rate = max(0.0, min(0.5, 0.3 * (1.2 - base_skill_level) + np.random.normal(0, 0.05)))
        
        coverage_completeness = min(1.0, base_skill_level + np.random.uniform(-0.1, 0.1))
        collaboration_score = min(1.0, base_skill_level + np.random.uniform(-0.2, 0.2))
        
        # Generate individual performances
        team_size = np.random.randint(3, 8)
        individual_performances = []
        
        for i in range(team_size):
            individual_skill = base_skill_level + np.random.normal(0, 0.15)
            individual_performance = PerformanceMetrics(
                team_member_id=f"{team_id}_member_{i+1}",
                detection_accuracy=min(1.0, max(0.0, individual_skill + np.random.normal(0, 0.1))),
                response_time_avg=max(1.0, 20 * (1.5 - individual_skill) + np.random.normal(0, 3)),
                false_positive_rate=max(0.0, min(0.5, 0.3 * (1.2 - individual_skill) + np.random.normal(0, 0.05))),
                incident_handling_score=min(1.0, max(0.0, individual_skill + np.random.normal(0, 0.1))),
                technical_skill_score=min(1.0, max(0.0, individual_skill + np.random.normal(0, 0.15))),
                knowledge_retention_score=min(1.0, max(0.0, individual_skill + np.random.normal(0, 0.1))),
                improvement_rate=np.random.uniform(0.05, 0.25),
                scenarios_completed=len(scenarios),
                total_training_time=duration_hours
            )
            individual_performances.append(individual_performance)
        
        # Calculate overall score
        overall_score = (
            detection_rate * 0.25 +
            (1.0 - false_positive_rate) * 0.2 +
            max(0.0, 1.0 - mean_time_to_respond / 30.0) * 0.2 +
            coverage_completeness * 0.15 +
            collaboration_score * 0.1 +
            np.mean([p.technical_skill_score for p in individual_performances]) * 0.1
        )
        
        return TeamPerformance(
            team_id=team_id,
            overall_score=overall_score,
            detection_rate=detection_rate,
            mean_time_to_detect=mean_time_to_detect,
            mean_time_to_respond=mean_time_to_respond,
            false_positive_rate=false_positive_rate,
            coverage_completeness=coverage_completeness,
            collaboration_score=collaboration_score,
            individual_performances=individual_performances,
            evaluation_timestamp=datetime.now().isoformat()
        )
    
    def _calculate_skill_improvements(
        self,
        pre_scores: TeamPerformance,
        post_scores: TeamPerformance
    ) -> Dict[str, float]:
        """Calculate improvements in specific skill areas"""
        
        improvements = {}
        
        # Team-level skill improvements
        improvements['detection_skills'] = post_scores.detection_rate - pre_scores.detection_rate
        improvements['response_efficiency'] = (pre_scores.mean_time_to_respond - post_scores.mean_time_to_respond) / pre_scores.mean_time_to_respond
        improvements['alert_analysis'] = pre_scores.false_positive_rate - post_scores.false_positive_rate
        improvements['collaboration'] = post_scores.collaboration_score - pre_scores.collaboration_score
        
        # Individual skill improvements (averaged)
        pre_individuals = {p.team_member_id: p for p in pre_scores.individual_performances}
        post_individuals = {p.team_member_id: p for p in post_scores.individual_performances}
        
        individual_improvements = []
        for member_id in pre_individuals:
            if member_id in post_individuals:
                pre_perf = pre_individuals[member_id]
                post_perf = post_individuals[member_id]
                
                technical_improvement = post_perf.technical_skill_score - pre_perf.technical_skill_score
                individual_improvements.append(technical_improvement)
        
        if individual_improvements:
            improvements['technical_skills'] = np.mean(individual_improvements)
        else:
            improvements['technical_skills'] = 0.0
        
        return improvements
    
    def _calculate_knowledge_retention(self, team_performance: TeamPerformance) -> float:
        """Calculate knowledge retention based on performance consistency"""
        
        # Knowledge retention based on individual performance variance and retention scores
        retention_scores = [p.knowledge_retention_score for p in team_performance.individual_performances]
        
        if retention_scores:
            # High retention = high mean, low variance
            mean_retention = np.mean(retention_scores)
            variance_penalty = np.var(retention_scores)
            retention = mean_retention - (variance_penalty * 0.5)
            return max(0.0, min(1.0, retention))
        
        return 0.5  # Default moderate retention
    
    def _statistical_analysis(
        self,
        pre_scores: TeamPerformance,
        post_scores: TeamPerformance
    ) -> Tuple[bool, float, Tuple[float, float]]:
        """Perform statistical analysis of improvement significance"""
        
        # Collect pre and post individual scores
        pre_individuals = pre_scores.individual_performances
        post_individuals = post_scores.individual_performances
        
        # Match individuals by ID
        matched_pairs = []
        pre_dict = {p.team_member_id: p for p in pre_individuals}
        post_dict = {p.team_member_id: p for p in post_individuals}
        
        for member_id in pre_dict:
            if member_id in post_dict:
                pre_score = pre_dict[member_id].technical_skill_score
                post_score = post_dict[member_id].technical_skill_score
                matched_pairs.append((pre_score, post_score))
        
        if len(matched_pairs) < 3:
            # Not enough data for statistical analysis
            return False, 1.0, (0.0, 0.0)
        
        # Paired t-test simulation (simplified)
        differences = [post - pre for pre, post in matched_pairs]
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        n = len(differences)
        
        # Calculate t-statistic
        if std_diff > 0:
            t_stat = mean_diff / (std_diff / np.sqrt(n))
            
            # Simplified p-value calculation (assuming normal distribution)
            # In real implementation, would use scipy.stats.t.cdf
            p_value = max(0.01, min(0.99, abs(t_stat) / 3))  # Rough approximation
            
            # 95% confidence interval
            margin_error = 1.96 * (std_diff / np.sqrt(n))  # Approximate
            confidence_interval = (mean_diff - margin_error, mean_diff + margin_error)
            
            significant = p_value < 0.05
        else:
            p_value = 1.0
            confidence_interval = (0.0, 0.0)
            significant = False
        
        return significant, p_value, confidence_interval
    
    def _generate_area_recommendation(self, area: str) -> Dict[str, Any]:
        """Generate recommendation for specific skill area"""
        
        recommendations = {
            'detection_skills': {
                'area': 'Detection Skills',
                'description': 'Improve threat detection accuracy and reduce false negatives',
                'suggested_activities': [
                    'Practice with diverse attack scenarios',
                    'Study threat intelligence feeds',
                    'Enhance log analysis skills'
                ],
                'estimated_time': 8,
                'priority': 'high'
            },
            'response_speed': {
                'area': 'Response Speed',
                'description': 'Reduce mean time to respond to security incidents',
                'suggested_activities': [
                    'Practice incident response procedures',
                    'Automate common response actions',
                    'Improve tool familiarity'
                ],
                'estimated_time': 6,
                'priority': 'high'
            },
            'alert_analysis': {
                'area': 'Alert Analysis',
                'description': 'Reduce false positive rate and improve alert triage',
                'suggested_activities': [
                    'Practice alert correlation techniques',
                    'Study baseline network behavior',
                    'Improve context analysis skills'
                ],
                'estimated_time': 4,
                'priority': 'medium'
            },
            'team_coordination': {
                'area': 'Team Coordination',
                'description': 'Improve collaboration and communication during incidents',
                'suggested_activities': [
                    'Practice team-based scenarios',
                    'Improve communication protocols',
                    'Cross-train on different roles'
                ],
                'estimated_time': 6,
                'priority': 'medium'
            }
        }
        
        return recommendations.get(area, {
            'area': area,
            'description': 'General skill improvement needed',
            'suggested_activities': ['Additional training recommended'],
            'estimated_time': 4,
            'priority': 'low'
        })
    
    def _generate_individual_recommendations(self, performance: PerformanceMetrics) -> List[Dict[str, Any]]:
        """Generate personalized recommendations for individual"""
        
        recommendations = []
        
        # Detection accuracy
        if performance.detection_accuracy < 0.7:
            recommendations.append({
                'skill': 'Detection Accuracy',
                'current_level': performance.detection_accuracy,
                'target_level': 0.8,
                'recommendation': 'Focus on pattern recognition and threat hunting exercises'
            })
        
        # Response time
        if performance.response_time_avg > 15:
            recommendations.append({
                'skill': 'Response Time',
                'current_level': performance.response_time_avg,
                'target_level': 10,
                'recommendation': 'Practice with time-constrained incident response scenarios'
            })
        
        # Technical skills
        if performance.technical_skill_score < 0.7:
            recommendations.append({
                'skill': 'Technical Skills',
                'current_level': performance.technical_skill_score,
                'target_level': 0.8,
                'recommendation': 'Strengthen technical knowledge in forensics and malware analysis'
            })
        
        # Knowledge retention
        if performance.knowledge_retention_score < 0.7:
            recommendations.append({
                'skill': 'Knowledge Retention',
                'current_level': performance.knowledge_retention_score,
                'target_level': 0.8,
                'recommendation': 'Regular review sessions and spaced repetition training'
            })
        
        return recommendations
    
    def _suggest_scenarios(self, weak_areas: List[str]) -> List[str]:
        """Suggest training scenarios based on weak areas"""
        
        scenario_mapping = {
            'detection_skills': ['advanced_persistent_threat', 'steganography_detection', 'zero_day_analysis'],
            'response_speed': ['rapid_response_drill', 'time_critical_incident', 'automated_response_practice'],
            'alert_analysis': ['false_positive_training', 'alert_correlation_exercise', 'baseline_analysis_training'],
            'team_coordination': ['multi_team_incident', 'communication_drill', 'role_switching_exercise']
        }
        
        suggested = []
        for area in weak_areas:
            if area in scenario_mapping:
                suggested.extend(scenario_mapping[area])
        
        # Remove duplicates and limit to top 5
        return list(set(suggested))[:5]