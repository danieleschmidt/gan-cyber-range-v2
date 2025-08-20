"""
Adaptive Curriculum Management System for Cybersecurity Education

This module provides intelligent curriculum management that adapts to individual learning
patterns and ensures comprehensive coverage of defensive cybersecurity concepts.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
from pathlib import Path
import math

logger = logging.getLogger(__name__)


class LearningStyle(Enum):
    """Different learning style preferences"""
    VISUAL = "visual"
    HANDS_ON = "hands_on"
    THEORETICAL = "theoretical"
    COLLABORATIVE = "collaborative"


class KnowledgeLevel(Enum):
    """Knowledge proficiency levels"""
    NOVICE = "novice"
    BEGINNER = "beginner" 
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ContentType(Enum):
    """Types of educational content"""
    LECTURE = "lecture"
    LAB_EXERCISE = "lab_exercise"
    CASE_STUDY = "case_study"
    SIMULATION = "simulation"
    ASSESSMENT = "assessment"
    RESEARCH_PROJECT = "research_project"


@dataclass
class LearningObjective:
    """Represents a specific learning objective"""
    objective_id: str
    title: str
    description: str
    category: str  # e.g., "detection", "analysis", "response", "prevention"
    knowledge_level: KnowledgeLevel
    prerequisites: List[str] = field(default_factory=list)
    estimated_hours: float = 1.0
    assessment_criteria: Dict[str, Any] = field(default_factory=dict)
    industry_relevance: float = 1.0  # 0.0 to 1.0 scale


@dataclass
class EducationalContent:
    """Educational content item"""
    content_id: str
    title: str
    description: str
    content_type: ContentType
    objectives: List[str]  # Learning objective IDs
    difficulty_level: KnowledgeLevel
    estimated_duration: int  # minutes
    content_url: Optional[str] = None
    content_data: Optional[Dict] = None
    interactive_elements: List[str] = field(default_factory=list)
    assessment_questions: List[Dict] = field(default_factory=list)


@dataclass
class LearnerProfile:
    """Individual learner profile and preferences"""
    learner_id: str
    name: str
    role: str  # e.g., "SOC Analyst", "Security Engineer"
    experience_years: float
    learning_style: LearningStyle
    current_knowledge: Dict[str, float] = field(default_factory=dict)  # objective_id -> mastery level
    learning_pace: float = 1.0  # Relative learning speed
    preferred_content_types: List[ContentType] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    availability_hours_per_week: int = 10
    last_activity: Optional[datetime] = None


@dataclass
class CurriculumPath:
    """Structured curriculum path for specific roles or goals"""
    path_id: str
    name: str
    description: str
    target_role: str
    prerequisite_knowledge: Dict[str, float] = field(default_factory=dict)
    learning_objectives: List[str] = field(default_factory=list)  # Ordered sequence
    estimated_total_hours: float = 0.0
    success_criteria: Dict[str, float] = field(default_factory=dict)
    industry_certifications: List[str] = field(default_factory=list)


@dataclass
class LearningSession:
    """Individual learning session record"""
    session_id: str
    learner_id: str
    content_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    completion_percentage: float = 0.0
    engagement_score: float = 0.0  # Derived from interaction patterns
    assessment_scores: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    questions_asked: List[str] = field(default_factory=list)


class CurriculumManager:
    """Adaptive curriculum management system"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.learning_objectives: Dict[str, LearningObjective] = {}
        self.educational_content: Dict[str, EducationalContent] = {}
        self.curriculum_paths: Dict[str, CurriculumPath] = {}
        self.learner_profiles: Dict[str, LearnerProfile] = {}
        self.learning_sessions: Dict[str, LearningSession] = {}
        
        # Analytics and adaptation
        self.content_effectiveness: Dict[str, Dict] = {}
        self.learning_analytics: Dict[str, Any] = {}
        
        # Initialize with default curriculum content
        self._initialize_default_curriculum()
    
    def _initialize_default_curriculum(self):
        """Initialize default cybersecurity curriculum content"""
        
        # Core learning objectives for AI-powered cybersecurity
        objectives = [
            LearningObjective(
                objective_id="ai_threat_landscape",
                title="Understanding AI-Powered Threat Landscape",
                description="Comprehensive understanding of how AI is used in modern cyber attacks",
                category="knowledge", 
                knowledge_level=KnowledgeLevel.BEGINNER,
                estimated_hours=2.0,
                assessment_criteria={
                    "theoretical_knowledge": 0.8,
                    "practical_application": 0.7
                },
                industry_relevance=0.95
            ),
            LearningObjective(
                objective_id="gan_attack_detection",
                title="GAN-Generated Attack Detection",
                description="Identify and analyze attacks generated by Generative Adversarial Networks",
                category="detection",
                knowledge_level=KnowledgeLevel.INTERMEDIATE,
                prerequisites=["ai_threat_landscape"],
                estimated_hours=4.0,
                assessment_criteria={
                    "detection_accuracy": 0.85,
                    "false_positive_rate": 0.15,
                    "analysis_quality": 0.8
                },
                industry_relevance=0.9
            ),
            LearningObjective(
                objective_id="adaptive_defense_strategies",
                title="Adaptive Defense Against AI Attacks", 
                description="Develop and implement dynamic defense strategies against adaptive AI threats",
                category="response",
                knowledge_level=KnowledgeLevel.ADVANCED,
                prerequisites=["gan_attack_detection"],
                estimated_hours=6.0,
                assessment_criteria={
                    "strategy_effectiveness": 0.85,
                    "adaptation_speed": 0.8,
                    "implementation_quality": 0.85
                },
                industry_relevance=0.95
            ),
            LearningObjective(
                objective_id="ai_forensics",
                title="AI-Powered Digital Forensics",
                description="Use AI tools for enhanced digital forensics and incident analysis",
                category="analysis",
                knowledge_level=KnowledgeLevel.ADVANCED,
                prerequisites=["gan_attack_detection"],
                estimated_hours=5.0,
                assessment_criteria={
                    "forensics_accuracy": 0.9,
                    "tool_proficiency": 0.85,
                    "evidence_quality": 0.8
                },
                industry_relevance=0.85
            ),
            LearningObjective(
                objective_id="ethical_ai_security",
                title="Ethical AI in Cybersecurity",
                description="Understanding ethical implications and responsible use of AI in security",
                category="ethics",
                knowledge_level=KnowledgeLevel.INTERMEDIATE,
                estimated_hours=3.0,
                assessment_criteria={
                    "ethical_understanding": 0.9,
                    "decision_making": 0.85,
                    "compliance_knowledge": 0.8
                },
                industry_relevance=0.8
            )
        ]
        
        for obj in objectives:
            self.learning_objectives[obj.objective_id] = obj
        
        # Educational content items
        content_items = [
            EducationalContent(
                content_id="ai_threats_overview",
                title="AI-Powered Cyber Threats: A Comprehensive Overview",
                description="Introduction to how artificial intelligence is transforming the threat landscape",
                content_type=ContentType.LECTURE,
                objectives=["ai_threat_landscape"],
                difficulty_level=KnowledgeLevel.BEGINNER,
                estimated_duration=90,  # 90 minutes
                interactive_elements=["interactive_timeline", "threat_categorization_exercise"],
                assessment_questions=[
                    {
                        "question": "What are the key characteristics that distinguish AI-generated attacks from traditional attacks?",
                        "type": "essay",
                        "points": 10,
                        "key_concepts": ["adaptability", "scale", "stealth", "automation"]
                    },
                    {
                        "question": "Which of the following is NOT a common characteristic of GAN-generated malware?",
                        "type": "multiple_choice",
                        "options": [
                            "High entropy in binary structure",
                            "Consistent opcode patterns", 
                            "Statistical anomalies in metadata",
                            "Polymorphic behavior"
                        ],
                        "correct_answer": 1,
                        "points": 5
                    }
                ]
            ),
            EducationalContent(
                content_id="gan_detection_lab",
                title="Hands-On: GAN Attack Detection Laboratory",
                description="Practical laboratory exercise for detecting GAN-generated attack patterns",
                content_type=ContentType.LAB_EXERCISE,
                objectives=["gan_attack_detection"],
                difficulty_level=KnowledgeLevel.INTERMEDIATE,
                estimated_duration=240,  # 4 hours
                interactive_elements=[
                    "virtual_environment",
                    "analysis_tools",
                    "sample_datasets",
                    "detection_algorithms"
                ],
                assessment_questions=[
                    {
                        "question": "Analyze the provided malware samples and identify which ones were generated by GANs",
                        "type": "practical",
                        "points": 25,
                        "success_criteria": {
                            "correct_identification_rate": 0.85,
                            "false_positive_rate": 0.15,
                            "analysis_methodology": "documented"
                        }
                    }
                ]
            ),
            EducationalContent(
                content_id="adaptive_defense_simulation",
                title="Adaptive Defense Strategy Simulation",
                description="Advanced simulation environment for developing adaptive defense strategies",
                content_type=ContentType.SIMULATION,
                objectives=["adaptive_defense_strategies"],
                difficulty_level=KnowledgeLevel.ADVANCED,
                estimated_duration=360,  # 6 hours
                interactive_elements=[
                    "adversarial_ai_simulation",
                    "defense_strategy_builder",
                    "real_time_adaptation",
                    "effectiveness_metrics"
                ],
                assessment_questions=[
                    {
                        "question": "Design and implement an adaptive defense strategy for a given threat scenario",
                        "type": "project",
                        "points": 40,
                        "deliverables": [
                            "strategy_document",
                            "implementation_code",
                            "effectiveness_analysis",
                            "adaptation_mechanisms"
                        ]
                    }
                ]
            ),
            EducationalContent(
                content_id="ai_ethics_case_studies",
                title="Ethical AI in Cybersecurity: Real-World Case Studies",
                description="Analysis of ethical dilemmas and best practices in AI-powered cybersecurity",
                content_type=ContentType.CASE_STUDY,
                objectives=["ethical_ai_security"],
                difficulty_level=KnowledgeLevel.INTERMEDIATE,
                estimated_duration=180,  # 3 hours
                interactive_elements=[
                    "case_study_analysis",
                    "ethical_decision_tree",
                    "stakeholder_perspective_exercise",
                    "policy_development"
                ],
                assessment_questions=[
                    {
                        "question": "Analyze an ethical dilemma case study and propose a responsible solution",
                        "type": "case_analysis",
                        "points": 20,
                        "evaluation_criteria": [
                            "stakeholder_consideration",
                            "ethical_framework_application",
                            "practical_feasibility",
                            "long_term_implications"
                        ]
                    }
                ]
            )
        ]
        
        for content in content_items:
            self.educational_content[content.content_id] = content
        
        # Curriculum paths for different roles
        soc_analyst_path = CurriculumPath(
            path_id="soc_analyst_ai_specialist",
            name="SOC Analyst - AI Threat Specialist",
            description="Specialized curriculum for SOC analysts focusing on AI-powered threat detection",
            target_role="SOC Analyst",
            prerequisite_knowledge={
                "basic_security_concepts": 0.7,
                "log_analysis": 0.6,
                "incident_response": 0.5
            },
            learning_objectives=[
                "ai_threat_landscape",
                "gan_attack_detection", 
                "ethical_ai_security",
                "ai_forensics"
            ],
            estimated_total_hours=14.0,
            success_criteria={
                "overall_mastery": 0.85,
                "practical_skills": 0.8,
                "theoretical_knowledge": 0.8
            },
            industry_certifications=["GCTI", "GCFA", "CISSP"]
        )
        
        security_engineer_path = CurriculumPath(
            path_id="security_engineer_ai_defense",
            name="Security Engineer - AI Defense Architect", 
            description="Advanced curriculum for security engineers building AI-resilient defenses",
            target_role="Security Engineer",
            prerequisite_knowledge={
                "security_architecture": 0.7,
                "programming": 0.6,
                "machine_learning_basics": 0.5
            },
            learning_objectives=[
                "ai_threat_landscape",
                "gan_attack_detection",
                "adaptive_defense_strategies",
                "ai_forensics",
                "ethical_ai_security"
            ],
            estimated_total_hours=20.0,
            success_criteria={
                "overall_mastery": 0.9,
                "implementation_skills": 0.85,
                "innovation_capability": 0.8
            },
            industry_certifications=["CISSP", "SABSA", "TOGAF"]
        )
        
        self.curriculum_paths[soc_analyst_path.path_id] = soc_analyst_path
        self.curriculum_paths[security_engineer_path.path_id] = security_engineer_path
    
    def create_learner_profile(
        self,
        learner_id: str,
        name: str,
        role: str,
        experience_years: float,
        learning_style: LearningStyle,
        goals: Optional[List[str]] = None,
        availability_hours_per_week: int = 10
    ) -> LearnerProfile:
        """Create a new learner profile"""
        
        profile = LearnerProfile(
            learner_id=learner_id,
            name=name,
            role=role,
            experience_years=experience_years,
            learning_style=learning_style,
            goals=goals or [],
            availability_hours_per_week=availability_hours_per_week,
            last_activity=datetime.now()
        )
        
        # Set default preferred content types based on learning style
        if learning_style == LearningStyle.VISUAL:
            profile.preferred_content_types = [ContentType.SIMULATION, ContentType.CASE_STUDY]
        elif learning_style == LearningStyle.HANDS_ON:
            profile.preferred_content_types = [ContentType.LAB_EXERCISE, ContentType.SIMULATION]
        elif learning_style == LearningStyle.THEORETICAL:
            profile.preferred_content_types = [ContentType.LECTURE, ContentType.RESEARCH_PROJECT]
        elif learning_style == LearningStyle.COLLABORATIVE:
            profile.preferred_content_types = [ContentType.CASE_STUDY, ContentType.LAB_EXERCISE]
        
        # Initialize current knowledge based on experience
        self._initialize_knowledge_baseline(profile)
        
        self.learner_profiles[learner_id] = profile
        
        logger.info(f"Created learner profile for {name} ({role})")
        return profile
    
    def _initialize_knowledge_baseline(self, profile: LearnerProfile):
        """Initialize baseline knowledge levels based on role and experience"""
        
        base_knowledge = {}
        
        # Base knowledge by years of experience (0.0 to 1.0 scale)
        experience_factor = min(profile.experience_years / 10.0, 1.0)
        
        # Role-specific knowledge initialization
        if "analyst" in profile.role.lower():
            base_knowledge.update({
                "ai_threat_landscape": experience_factor * 0.3,
                "gan_attack_detection": experience_factor * 0.2,
                "ethical_ai_security": experience_factor * 0.4
            })
        elif "engineer" in profile.role.lower():
            base_knowledge.update({
                "ai_threat_landscape": experience_factor * 0.4,
                "adaptive_defense_strategies": experience_factor * 0.3,
                "ai_forensics": experience_factor * 0.2,
                "ethical_ai_security": experience_factor * 0.5
            })
        else:
            # General baseline
            base_knowledge.update({
                "ai_threat_landscape": experience_factor * 0.2,
                "ethical_ai_security": experience_factor * 0.3
            })
        
        profile.current_knowledge = base_knowledge
    
    def recommend_learning_path(self, learner_id: str) -> Optional[CurriculumPath]:
        """Recommend an appropriate learning path for a learner"""
        
        if learner_id not in self.learner_profiles:
            return None
        
        profile = self.learner_profiles[learner_id]
        
        # Find matching curriculum paths
        suitable_paths = []
        
        for path_id, path in self.curriculum_paths.items():
            # Check role match
            if profile.role.lower() in path.target_role.lower():
                # Check prerequisite knowledge
                meets_prerequisites = all(
                    profile.current_knowledge.get(obj_id, 0) >= required_level
                    for obj_id, required_level in path.prerequisite_knowledge.items()
                )
                
                if meets_prerequisites:
                    suitable_paths.append(path)
        
        # Return the most suitable path (could be enhanced with more sophisticated matching)
        if suitable_paths:
            return suitable_paths[0]  # For now, return the first match
        
        return None
    
    def generate_personalized_curriculum(
        self,
        learner_id: str,
        target_objectives: Optional[List[str]] = None,
        time_constraint: Optional[int] = None  # weeks
    ) -> Dict[str, Any]:
        """Generate a personalized curriculum for a learner"""
        
        if learner_id not in self.learner_profiles:
            raise ValueError(f"Unknown learner: {learner_id}")
        
        profile = self.learner_profiles[learner_id]
        
        # Determine objectives to include
        if target_objectives:
            objectives = [self.learning_objectives[obj_id] for obj_id in target_objectives
                         if obj_id in self.learning_objectives]
        else:
            # Use recommended learning path
            recommended_path = self.recommend_learning_path(learner_id)
            if recommended_path:
                objectives = [self.learning_objectives[obj_id] for obj_id in recommended_path.learning_objectives
                             if obj_id in self.learning_objectives]
            else:
                # Default to all available objectives
                objectives = list(self.learning_objectives.values())
        
        # Filter objectives based on current knowledge
        suitable_objectives = []
        for obj in objectives:
            current_mastery = profile.current_knowledge.get(obj.objective_id, 0)
            
            # Include if not mastered yet and prerequisites are met
            if current_mastery < 0.8:  # Not yet mastered
                prerequisites_met = all(
                    profile.current_knowledge.get(prereq_id, 0) >= 0.6
                    for prereq_id in obj.prerequisites
                )
                if prerequisites_met:
                    suitable_objectives.append(obj)
        
        # Sort by difficulty and prerequisites
        suitable_objectives.sort(key=lambda x: (x.knowledge_level.value, len(x.prerequisites)))
        
        # Select content for each objective
        curriculum_content = []
        total_estimated_hours = 0
        
        for objective in suitable_objectives:
            # Find suitable content for this objective
            matching_content = [
                content for content in self.educational_content.values()
                if objective.objective_id in content.objectives
            ]
            
            # Filter by preferred content types
            preferred_content = [
                content for content in matching_content
                if content.content_type in profile.preferred_content_types
            ]
            
            if not preferred_content:
                preferred_content = matching_content  # Fallback to any matching content
            
            # Select best content (could be enhanced with more sophisticated selection)
            if preferred_content:
                selected_content = preferred_content[0]  # For now, select first match
                curriculum_content.append({
                    "objective": objective,
                    "content": selected_content,
                    "estimated_hours": objective.estimated_hours,
                    "priority": self._calculate_learning_priority(objective, profile)
                })
                total_estimated_hours += objective.estimated_hours
        
        # Apply time constraints if specified
        if time_constraint:
            max_hours = time_constraint * profile.availability_hours_per_week
            if total_estimated_hours > max_hours:
                # Prioritize content and trim if necessary
                curriculum_content.sort(key=lambda x: x["priority"], reverse=True)
                
                adjusted_content = []
                accumulated_hours = 0
                
                for item in curriculum_content:
                    if accumulated_hours + item["estimated_hours"] <= max_hours:
                        adjusted_content.append(item)
                        accumulated_hours += item["estimated_hours"]
                    else:
                        break
                
                curriculum_content = adjusted_content
                total_estimated_hours = accumulated_hours
        
        # Generate schedule
        schedule = self._generate_learning_schedule(
            curriculum_content, 
            profile.availability_hours_per_week
        )
        
        personalized_curriculum = {
            "learner_id": learner_id,
            "generated_at": datetime.now().isoformat(),
            "total_objectives": len(curriculum_content),
            "estimated_total_hours": total_estimated_hours,
            "estimated_weeks": math.ceil(total_estimated_hours / profile.availability_hours_per_week),
            "curriculum_items": curriculum_content,
            "schedule": schedule,
            "adaptation_recommendations": self._generate_adaptation_recommendations(profile)
        }
        
        return personalized_curriculum
    
    def _calculate_learning_priority(self, objective: LearningObjective, profile: LearnerProfile) -> float:
        """Calculate learning priority score for an objective"""
        
        priority_score = 0.0
        
        # Industry relevance weight
        priority_score += objective.industry_relevance * 0.3
        
        # Current knowledge gap (higher gap = higher priority)
        current_knowledge = profile.current_knowledge.get(objective.objective_id, 0)
        knowledge_gap = 1.0 - current_knowledge
        priority_score += knowledge_gap * 0.4
        
        # Goal alignment
        if any(goal.lower() in objective.category.lower() for goal in profile.goals):
            priority_score += 0.2
        
        # Experience level alignment
        if objective.knowledge_level == KnowledgeLevel.BEGINNER and profile.experience_years < 2:
            priority_score += 0.1
        elif objective.knowledge_level == KnowledgeLevel.ADVANCED and profile.experience_years > 5:
            priority_score += 0.1
        
        return priority_score
    
    def _generate_learning_schedule(
        self, 
        curriculum_content: List[Dict],
        hours_per_week: int
    ) -> List[Dict[str, Any]]:
        """Generate a learning schedule based on curriculum content"""
        
        schedule = []
        current_week = 1
        current_week_hours = 0
        
        for item in curriculum_content:
            objective = item["objective"]
            content = item["content"]
            hours_needed = item["estimated_hours"]
            
            # Calculate how many weeks this item will span
            if current_week_hours + hours_needed <= hours_per_week:
                # Fits in current week
                schedule.append({
                    "week": current_week,
                    "objective_id": objective.objective_id,
                    "objective_title": objective.title,
                    "content_id": content.content_id,
                    "content_title": content.title,
                    "estimated_hours": hours_needed,
                    "content_type": content.content_type.value
                })
                current_week_hours += hours_needed
            else:
                # Move to next week
                current_week += 1
                current_week_hours = hours_needed
                schedule.append({
                    "week": current_week,
                    "objective_id": objective.objective_id,
                    "objective_title": objective.title,
                    "content_id": content.content_id,
                    "content_title": content.title,
                    "estimated_hours": hours_needed,
                    "content_type": content.content_type.value
                })
        
        return schedule
    
    def _generate_adaptation_recommendations(self, profile: LearnerProfile) -> List[str]:
        """Generate personalized adaptation recommendations"""
        
        recommendations = []
        
        # Learning style adaptations
        if profile.learning_style == LearningStyle.VISUAL:
            recommendations.append("Include visual diagrams and flowcharts for complex concepts")
        elif profile.learning_style == LearningStyle.HANDS_ON:
            recommendations.append("Prioritize practical exercises and real-world simulations")
        elif profile.learning_style == LearningStyle.THEORETICAL:
            recommendations.append("Provide in-depth theoretical background and research papers")
        elif profile.learning_style == LearningStyle.COLLABORATIVE:
            recommendations.append("Consider group learning opportunities and peer discussions")
        
        # Experience-based adaptations
        if profile.experience_years < 2:
            recommendations.append("Include additional foundational concepts and glossary")
        elif profile.experience_years > 8:
            recommendations.append("Focus on advanced techniques and cutting-edge research")
        
        # Schedule adaptations
        if profile.availability_hours_per_week < 5:
            recommendations.append("Break complex topics into smaller, digestible modules")
        elif profile.availability_hours_per_week > 15:
            recommendations.append("Include additional enrichment activities and deep-dives")
        
        return recommendations
    
    def start_learning_session(
        self,
        learner_id: str,
        content_id: str
    ) -> str:
        """Start a new learning session"""
        
        if learner_id not in self.learner_profiles:
            raise ValueError(f"Unknown learner: {learner_id}")
        
        if content_id not in self.educational_content:
            raise ValueError(f"Unknown content: {content_id}")
        
        session_id = str(uuid.uuid4())
        
        session = LearningSession(
            session_id=session_id,
            learner_id=learner_id,
            content_id=content_id,
            start_time=datetime.now()
        )
        
        self.learning_sessions[session_id] = session
        
        # Update learner's last activity
        self.learner_profiles[learner_id].last_activity = datetime.now()
        
        logger.info(f"Started learning session {session_id} for learner {learner_id}")
        
        return session_id
    
    def complete_learning_session(
        self,
        session_id: str,
        completion_percentage: float,
        assessment_scores: Optional[Dict[str, float]] = None,
        notes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Complete a learning session and update learner progress"""
        
        if session_id not in self.learning_sessions:
            raise ValueError(f"Unknown session: {session_id}")
        
        session = self.learning_sessions[session_id]
        session.end_time = datetime.now()
        session.completion_percentage = completion_percentage
        
        if assessment_scores:
            session.assessment_scores = assessment_scores
        
        if notes:
            session.notes = notes
        
        # Calculate engagement score based on session data
        session.engagement_score = self._calculate_engagement_score(session)
        
        # Update learner knowledge based on session completion
        self._update_learner_knowledge(session)
        
        # Update content effectiveness analytics
        self._update_content_effectiveness(session)
        
        # Generate session summary
        session_summary = {
            "session_id": session_id,
            "learner_id": session.learner_id,
            "content_id": session.content_id,
            "duration_minutes": (session.end_time - session.start_time).total_seconds() / 60,
            "completion_percentage": completion_percentage,
            "engagement_score": session.engagement_score,
            "assessment_scores": session.assessment_scores,
            "knowledge_gained": self._calculate_knowledge_gained(session),
            "next_recommendations": self._generate_next_content_recommendations(session.learner_id)
        }
        
        logger.info(f"Completed learning session {session_id}: {completion_percentage:.1%} complete")
        
        return session_summary
    
    def _calculate_engagement_score(self, session: LearningSession) -> float:
        """Calculate engagement score based on session activities"""
        
        duration_minutes = (session.end_time - session.start_time).total_seconds() / 60
        content = self.educational_content[session.content_id]
        expected_duration = content.estimated_duration
        
        engagement_score = 0.0
        
        # Duration factor (optimal is close to expected duration)
        duration_ratio = duration_minutes / expected_duration
        if 0.8 <= duration_ratio <= 1.2:  # Within 20% of expected
            engagement_score += 0.3
        elif 0.5 <= duration_ratio < 0.8:  # Rushed
            engagement_score += 0.1
        elif duration_ratio > 1.2:  # Took longer (potentially more engaged)
            engagement_score += 0.2
        
        # Completion factor
        engagement_score += session.completion_percentage * 0.4
        
        # Assessment performance
        if session.assessment_scores:
            avg_assessment_score = sum(session.assessment_scores.values()) / len(session.assessment_scores)
            engagement_score += (avg_assessment_score / 100) * 0.2
        
        # Notes and questions (indicates active learning)
        if session.notes:
            engagement_score += min(len(session.notes) * 0.02, 0.1)
        
        if session.questions_asked:
            engagement_score += min(len(session.questions_asked) * 0.03, 0.1)
        
        return min(engagement_score, 1.0)
    
    def _update_learner_knowledge(self, session: LearningSession):
        """Update learner's knowledge levels based on session completion"""
        
        profile = self.learner_profiles[session.learner_id]
        content = self.educational_content[session.content_id]
        
        # Calculate knowledge gain based on completion and performance
        base_gain = session.completion_percentage * 0.2  # Up to 20% gain from completion
        
        # Assessment bonus
        if session.assessment_scores:
            avg_assessment_score = sum(session.assessment_scores.values()) / len(session.assessment_scores)
            assessment_bonus = (avg_assessment_score / 100) * 0.1  # Up to 10% bonus
        else:
            assessment_bonus = 0
        
        # Engagement bonus
        engagement_bonus = session.engagement_score * 0.05  # Up to 5% bonus
        
        total_gain = base_gain + assessment_bonus + engagement_bonus
        
        # Apply knowledge gain to relevant objectives
        for objective_id in content.objectives:
            if objective_id in self.learning_objectives:
                current_knowledge = profile.current_knowledge.get(objective_id, 0)
                
                # Apply diminishing returns (harder to gain knowledge at higher levels)
                diminishing_factor = 1.0 - (current_knowledge * 0.5)
                adjusted_gain = total_gain * diminishing_factor
                
                new_knowledge = min(current_knowledge + adjusted_gain, 1.0)
                profile.current_knowledge[objective_id] = new_knowledge
                
                logger.debug(f"Updated knowledge for {objective_id}: {current_knowledge:.2f} -> {new_knowledge:.2f}")
    
    def _calculate_knowledge_gained(self, session: LearningSession) -> Dict[str, float]:
        """Calculate how much knowledge was gained in each objective"""
        
        # This is a simplified version - in reality, you'd compare before/after states
        content = self.educational_content[session.content_id]
        base_gain = session.completion_percentage * 0.2
        
        if session.assessment_scores:
            avg_assessment_score = sum(session.assessment_scores.values()) / len(session.assessment_scores)
            assessment_bonus = (avg_assessment_score / 100) * 0.1
        else:
            assessment_bonus = 0
        
        total_gain = base_gain + assessment_bonus
        
        knowledge_gained = {}
        for objective_id in content.objectives:
            knowledge_gained[objective_id] = total_gain
        
        return knowledge_gained
    
    def _update_content_effectiveness(self, session: LearningSession):
        """Update analytics on content effectiveness"""
        
        content_id = session.content_id
        
        if content_id not in self.content_effectiveness:
            self.content_effectiveness[content_id] = {
                "total_sessions": 0,
                "total_completion": 0.0,
                "total_engagement": 0.0,
                "assessment_scores": [],
                "avg_duration_minutes": 0.0
            }
        
        effectiveness = self.content_effectiveness[content_id]
        
        effectiveness["total_sessions"] += 1
        effectiveness["total_completion"] += session.completion_percentage
        effectiveness["total_engagement"] += session.engagement_score
        
        if session.assessment_scores:
            avg_score = sum(session.assessment_scores.values()) / len(session.assessment_scores)
            effectiveness["assessment_scores"].append(avg_score)
        
        duration_minutes = (session.end_time - session.start_time).total_seconds() / 60
        effectiveness["avg_duration_minutes"] = (
            (effectiveness["avg_duration_minutes"] * (effectiveness["total_sessions"] - 1) + duration_minutes) /
            effectiveness["total_sessions"]
        )
    
    def _generate_next_content_recommendations(self, learner_id: str) -> List[str]:
        """Generate recommendations for next content to study"""
        
        profile = self.learner_profiles[learner_id]
        recommendations = []
        
        # Find objectives that are ready to be studied (prerequisites met, not mastered)
        ready_objectives = []
        for objective_id, objective in self.learning_objectives.items():
            current_mastery = profile.current_knowledge.get(objective_id, 0)
            
            if current_mastery < 0.8:  # Not mastered yet
                prerequisites_met = all(
                    profile.current_knowledge.get(prereq_id, 0) >= 0.6
                    for prereq_id in objective.prerequisites
                )
                
                if prerequisites_met:
                    ready_objectives.append((objective_id, objective, current_mastery))
        
        # Sort by priority and current progress
        ready_objectives.sort(key=lambda x: (
            self._calculate_learning_priority(x[1], profile),
            x[2]  # Current mastery (continue partially learned topics)
        ), reverse=True)
        
        # Generate recommendations for top objectives
        for objective_id, objective, current_mastery in ready_objectives[:3]:
            # Find suitable content
            matching_content = [
                content for content in self.educational_content.values()
                if objective_id in content.objectives and
                content.content_type in profile.preferred_content_types
            ]
            
            if matching_content:
                content = matching_content[0]
                recommendations.append(f"{content.title} (Focus: {objective.title})")
        
        return recommendations
    
    def get_learner_progress_report(self, learner_id: str) -> Dict[str, Any]:
        """Generate comprehensive progress report for a learner"""
        
        if learner_id not in self.learner_profiles:
            raise ValueError(f"Unknown learner: {learner_id}")
        
        profile = self.learner_profiles[learner_id]
        
        # Get all sessions for this learner
        learner_sessions = [
            session for session in self.learning_sessions.values()
            if session.learner_id == learner_id
        ]
        
        # Calculate progress metrics
        total_sessions = len(learner_sessions)
        completed_sessions = [s for s in learner_sessions if s.completion_percentage >= 0.8]
        total_study_hours = sum(
            (s.end_time - s.start_time).total_seconds() / 3600
            for s in learner_sessions if s.end_time
        )
        
        avg_engagement = (
            sum(s.engagement_score for s in learner_sessions) / total_sessions
            if total_sessions > 0 else 0
        )
        
        # Knowledge mastery by category
        knowledge_by_category = {}
        for objective_id, mastery_level in profile.current_knowledge.items():
            if objective_id in self.learning_objectives:
                objective = self.learning_objectives[objective_id]
                category = objective.category
                
                if category not in knowledge_by_category:
                    knowledge_by_category[category] = []
                
                knowledge_by_category[category].append(mastery_level)
        
        # Calculate category averages
        category_mastery = {
            category: sum(levels) / len(levels)
            for category, levels in knowledge_by_category.items()
        }
        
        # Overall mastery
        overall_mastery = (
            sum(profile.current_knowledge.values()) / len(profile.current_knowledge)
            if profile.current_knowledge else 0
        )
        
        progress_report = {
            "learner_id": learner_id,
            "name": profile.name,
            "role": profile.role,
            "report_generated_at": datetime.now().isoformat(),
            
            # Session metrics
            "total_sessions": total_sessions,
            "completed_sessions": len(completed_sessions),
            "completion_rate": len(completed_sessions) / total_sessions if total_sessions > 0 else 0,
            "total_study_hours": round(total_study_hours, 1),
            "average_engagement_score": round(avg_engagement, 2),
            
            # Knowledge metrics
            "overall_mastery": round(overall_mastery, 2),
            "category_mastery": {k: round(v, 2) for k, v in category_mastery.items()},
            "objectives_mastered": len([m for m in profile.current_knowledge.values() if m >= 0.8]),
            "objectives_in_progress": len([m for m in profile.current_knowledge.values() if 0.3 <= m < 0.8]),
            
            # Learning path progress
            "recommended_path": None,
            "path_completion": 0.0,
            
            # Next steps
            "next_recommendations": self._generate_next_content_recommendations(learner_id),
            "estimated_completion_time": self._estimate_remaining_time(learner_id)
        }
        
        # Add learning path progress if applicable
        recommended_path = self.recommend_learning_path(learner_id)
        if recommended_path:
            progress_report["recommended_path"] = recommended_path.name
            
            path_objectives = recommended_path.learning_objectives
            completed_objectives = sum(
                1 for obj_id in path_objectives
                if profile.current_knowledge.get(obj_id, 0) >= 0.8
            )
            
            progress_report["path_completion"] = completed_objectives / len(path_objectives)
        
        return progress_report
    
    def _estimate_remaining_time(self, learner_id: str) -> float:
        """Estimate remaining time to complete current learning goals"""
        
        profile = self.learner_profiles[learner_id]
        
        # Find objectives that still need work
        remaining_objectives = []
        for objective_id, objective in self.learning_objectives.items():
            current_mastery = profile.current_knowledge.get(objective_id, 0)
            if current_mastery < 0.8:
                remaining_work = (0.8 - current_mastery) / 0.8  # Fraction of work remaining
                estimated_hours = objective.estimated_hours * remaining_work
                remaining_objectives.append(estimated_hours)
        
        total_remaining_hours = sum(remaining_objectives)
        
        # Adjust for learning pace and availability
        adjusted_hours = total_remaining_hours / profile.learning_pace
        estimated_weeks = adjusted_hours / profile.availability_hours_per_week
        
        return round(estimated_weeks, 1)
    
    def export_curriculum_analytics(self) -> Dict[str, Any]:
        """Export comprehensive analytics for curriculum effectiveness"""
        
        analytics = {
            "export_timestamp": datetime.now().isoformat(),
            "total_learners": len(self.learner_profiles),
            "total_sessions": len(self.learning_sessions),
            "content_effectiveness": {},
            "learning_objective_analytics": {},
            "curriculum_path_analytics": {}
        }
        
        # Content effectiveness
        for content_id, effectiveness in self.content_effectiveness.items():
            if content_id in self.educational_content:
                content = self.educational_content[content_id]
                
                avg_completion = effectiveness["total_completion"] / effectiveness["total_sessions"]
                avg_engagement = effectiveness["total_engagement"] / effectiveness["total_sessions"] 
                avg_assessment = (
                    sum(effectiveness["assessment_scores"]) / len(effectiveness["assessment_scores"])
                    if effectiveness["assessment_scores"] else 0
                )
                
                analytics["content_effectiveness"][content_id] = {
                    "title": content.title,
                    "content_type": content.content_type.value,
                    "total_sessions": effectiveness["total_sessions"],
                    "average_completion": round(avg_completion, 2),
                    "average_engagement": round(avg_engagement, 2),
                    "average_assessment_score": round(avg_assessment, 1),
                    "average_duration_minutes": round(effectiveness["avg_duration_minutes"], 1)
                }
        
        # Learning objective analytics
        for objective_id, objective in self.learning_objectives.items():
            mastery_levels = [
                profile.current_knowledge.get(objective_id, 0)
                for profile in self.learner_profiles.values()
                if objective_id in profile.current_knowledge
            ]
            
            if mastery_levels:
                analytics["learning_objective_analytics"][objective_id] = {
                    "title": objective.title,
                    "category": objective.category,
                    "difficulty_level": objective.knowledge_level.value,
                    "learners_engaged": len(mastery_levels),
                    "average_mastery": round(sum(mastery_levels) / len(mastery_levels), 2),
                    "mastery_distribution": {
                        "novice": len([m for m in mastery_levels if m < 0.3]),
                        "developing": len([m for m in mastery_levels if 0.3 <= m < 0.6]),
                        "proficient": len([m for m in mastery_levels if 0.6 <= m < 0.8]),
                        "mastered": len([m for m in mastery_levels if m >= 0.8])
                    }
                }
        
        return analytics


def create_curriculum_manager(config_path: Optional[Path] = None) -> CurriculumManager:
    """Factory function to create a CurriculumManager instance"""
    return CurriculumManager(config_path)


# Example usage and demonstration
if __name__ == "__main__":
    # Initialize curriculum manager
    manager = create_curriculum_manager()
    
    # Create a sample learner
    learner_profile = manager.create_learner_profile(
        learner_id="analyst_001",
        name="Alice Security",
        role="SOC Analyst",
        experience_years=3.5,
        learning_style=LearningStyle.HANDS_ON,
        goals=["ai_threat_detection", "incident_response"],
        availability_hours_per_week=8
    )
    
    print(f"Created learner profile for {learner_profile.name}")
    print(f"Learning style: {learner_profile.learning_style.value}")
    
    # Generate personalized curriculum
    curriculum = manager.generate_personalized_curriculum("analyst_001")
    
    print(f"\nGenerated personalized curriculum:")
    print(f"Total objectives: {curriculum['total_objectives']}")
    print(f"Estimated time: {curriculum['estimated_total_hours']} hours ({curriculum['estimated_weeks']} weeks)")
    
    print(f"\nLearning schedule:")
    for item in curriculum['schedule']:
        print(f"Week {item['week']}: {item['content_title']} ({item['estimated_hours']} hours)")
    
    # Simulate a learning session
    session_id = manager.start_learning_session("analyst_001", "ai_threats_overview")
    
    # Complete the session with sample data
    session_summary = manager.complete_learning_session(
        session_id=session_id,
        completion_percentage=0.95,
        assessment_scores={"knowledge_check": 85, "practical_exercise": 92},
        notes=["Very informative overview", "Need more examples of GAN detection"]
    )
    
    print(f"\nCompleted learning session:")
    print(f"Duration: {session_summary['duration_minutes']:.1f} minutes")
    print(f"Engagement score: {session_summary['engagement_score']:.2f}")
    print(f"Next recommendations: {session_summary['next_recommendations']}")
    
    # Generate progress report
    progress = manager.get_learner_progress_report("analyst_001")
    
    print(f"\nProgress report for {progress['name']}:")
    print(f"Overall mastery: {progress['overall_mastery']:.2f}")
    print(f"Sessions completed: {progress['completed_sessions']}/{progress['total_sessions']}")
    print(f"Study time: {progress['total_study_hours']} hours")
    print(f"Category mastery:")
    for category, mastery in progress['category_mastery'].items():
        print(f"  {category}: {mastery:.2f}")