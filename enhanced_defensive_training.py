#!/usr/bin/env python3
"""
Enhanced Defensive Training System - Advanced Skills Development
Comprehensive defensive cybersecurity training with AI-driven personalization
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import random
from pathlib import Path

# Core imports
from gan_cyber_range.training import DefensiveTrainingEnhancer, DefensiveSkill, TrainingDifficulty
from gan_cyber_range.security import ThreatDetector, SecurityScanner
from gan_cyber_range.orchestration import WorkflowEngine, Workflow
from gan_cyber_range.evaluation import TrainingEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingModality(Enum):
    """Training delivery methods"""
    HANDS_ON_LAB = "hands_on_lab"
    SIMULATION = "simulation" 
    THEORY = "theory"
    CAPTURE_FLAG = "capture_flag"
    RED_TEAM_EXERCISE = "red_team_exercise"


class LearningObjective(Enum):
    """Specific learning objectives for defensive training"""
    DETECT_MALWARE = "detect_malware"
    ANALYZE_NETWORK_TRAFFIC = "analyze_network_traffic"
    INCIDENT_CONTAINMENT = "incident_containment"
    FORENSIC_EVIDENCE = "forensic_evidence"
    THREAT_INTELLIGENCE = "threat_intelligence"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    SECURITY_AUTOMATION = "security_automation"
    COMPLIANCE_MANAGEMENT = "compliance_management"


@dataclass
class LearningPath:
    """Structured learning progression"""
    name: str
    description: str
    prerequisites: List[DefensiveSkill]
    modules: List['TrainingModule']
    estimated_hours: int
    difficulty_level: TrainingDifficulty
    certification_available: bool = False


@dataclass
class TrainingModule:
    """Individual training module"""
    id: str
    name: str
    description: str
    objectives: List[LearningObjective]
    modality: TrainingModality
    duration_minutes: int
    difficulty: TrainingDifficulty
    prerequisites: List[str] = field(default_factory=list)
    hands_on_component: bool = True
    assessment_included: bool = True


@dataclass
class TrainingResult:
    """Results from training module completion"""
    module_id: str
    student_id: str
    completion_time: datetime
    score: float
    skill_improvement: float
    time_spent_minutes: int
    mistakes_made: List[str]
    strengths_identified: List[str]
    recommendations: List[str]


class EnhancedDefensiveTraining:
    """Advanced defensive cybersecurity training system"""
    
    def __init__(self):
        self.training_enhancer = DefensiveTrainingEnhancer()
        self.threat_detector = ThreatDetector()
        self.security_scanner = SecurityScanner()
        self.workflow_engine = WorkflowEngine()
        self.evaluator = TrainingEvaluator()
        self.learning_paths = {}
        self.training_modules = {}
        self.student_progress = {}
        
        # Initialize training catalog
        self.initialize_training_catalog()
    
    def initialize_training_catalog(self):
        """Initialize comprehensive training catalog"""
        logger.info("Initializing defensive training catalog")
        
        # Create specialized learning paths
        self.learning_paths = {
            "incident_responder": LearningPath(
                name="Incident Response Specialist",
                description="Comprehensive training for cybersecurity incident response",
                prerequisites=[DefensiveSkill.NETWORK_ANALYSIS],
                modules=[],
                estimated_hours=40,
                difficulty_level=TrainingDifficulty.ADVANCED,
                certification_available=True
            ),
            "threat_hunter": LearningPath(
                name="Advanced Threat Hunter", 
                description="Proactive threat hunting and detection techniques",
                prerequisites=[DefensiveSkill.LOG_ANALYSIS, DefensiveSkill.NETWORK_ANALYSIS],
                modules=[],
                estimated_hours=32,
                difficulty_level=TrainingDifficulty.EXPERT,
                certification_available=True
            ),
            "forensics_analyst": LearningPath(
                name="Digital Forensics Analyst",
                description="Digital forensics and evidence analysis",
                prerequisites=[DefensiveSkill.SYSTEM_ANALYSIS],
                modules=[],
                estimated_hours=48,
                difficulty_level=TrainingDifficulty.ADVANCED,
                certification_available=True
            ),
            "security_architect": LearningPath(
                name="Security Architecture Specialist", 
                description="Design and implementation of security architectures",
                prerequisites=[DefensiveSkill.RISK_ASSESSMENT, DefensiveSkill.COMPLIANCE],
                modules=[],
                estimated_hours=60,
                difficulty_level=TrainingDifficulty.EXPERT,
                certification_available=True
            )
        }
        
        # Create specialized training modules
        self.create_training_modules()
        
        # Assign modules to learning paths
        self.assign_modules_to_paths()
    
    def create_training_modules(self):
        """Create comprehensive training modules"""
        
        # Malware Analysis Modules
        self.training_modules["malware_static_analysis"] = TrainingModule(
            id="malware_static_analysis",
            name="Static Malware Analysis",
            description="Learn to analyze malware without execution",
            objectives=[LearningObjective.DETECT_MALWARE, LearningObjective.THREAT_INTELLIGENCE],
            modality=TrainingModality.HANDS_ON_LAB,
            duration_minutes=120,
            difficulty=TrainingDifficulty.INTERMEDIATE
        )
        
        self.training_modules["malware_dynamic_analysis"] = TrainingModule(
            id="malware_dynamic_analysis", 
            name="Dynamic Malware Analysis",
            description="Behavioral analysis of malware in controlled environments",
            objectives=[LearningObjective.DETECT_MALWARE, LearningObjective.FORENSIC_EVIDENCE],
            modality=TrainingModality.SIMULATION,
            duration_minutes=150,
            difficulty=TrainingDifficulty.ADVANCED,
            prerequisites=["malware_static_analysis"]
        )
        
        # Network Analysis Modules
        self.training_modules["network_traffic_analysis"] = TrainingModule(
            id="network_traffic_analysis",
            name="Network Traffic Analysis",
            description="Deep packet inspection and traffic analysis techniques",
            objectives=[LearningObjective.ANALYZE_NETWORK_TRAFFIC, LearningObjective.THREAT_INTELLIGENCE],
            modality=TrainingModality.HANDS_ON_LAB,
            duration_minutes=180,
            difficulty=TrainingDifficulty.INTERMEDIATE
        )
        
        # Incident Response Modules  
        self.training_modules["incident_classification"] = TrainingModule(
            id="incident_classification",
            name="Security Incident Classification",
            description="Proper classification and prioritization of security incidents",
            objectives=[LearningObjective.INCIDENT_CONTAINMENT],
            modality=TrainingModality.THEORY,
            duration_minutes=90,
            difficulty=TrainingDifficulty.BEGINNER
        )
        
        self.training_modules["incident_containment"] = TrainingModule(
            id="incident_containment",
            name="Advanced Incident Containment",
            description="Techniques for containing and eradicating threats",
            objectives=[LearningObjective.INCIDENT_CONTAINMENT, LearningObjective.SECURITY_AUTOMATION],
            modality=TrainingModality.RED_TEAM_EXERCISE,
            duration_minutes=240,
            difficulty=TrainingDifficulty.ADVANCED,
            prerequisites=["incident_classification"]
        )
        
        # Digital Forensics Modules
        self.training_modules["disk_forensics"] = TrainingModule(
            id="disk_forensics",
            name="Disk Forensics and Recovery",
            description="Forensic analysis of storage devices and data recovery",
            objectives=[LearningObjective.FORENSIC_EVIDENCE],
            modality=TrainingModality.HANDS_ON_LAB,
            duration_minutes=200,
            difficulty=TrainingDifficulty.ADVANCED
        )
        
        self.training_modules["memory_forensics"] = TrainingModule(
            id="memory_forensics",
            name="Memory Forensics Analysis",
            description="Volatile memory analysis and artifact extraction",
            objectives=[LearningObjective.FORENSIC_EVIDENCE, LearningObjective.DETECT_MALWARE],
            modality=TrainingModality.HANDS_ON_LAB,
            duration_minutes=180,
            difficulty=TrainingDifficulty.ADVANCED
        )
        
        # Threat Intelligence Modules
        self.training_modules["threat_intel_collection"] = TrainingModule(
            id="threat_intel_collection",
            name="Threat Intelligence Collection",
            description="Methods for gathering and validating threat intelligence",
            objectives=[LearningObjective.THREAT_INTELLIGENCE],
            modality=TrainingModality.THEORY,
            duration_minutes=120,
            difficulty=TrainingDifficulty.INTERMEDIATE
        )
        
        # Security Automation Modules
        self.training_modules["soar_implementation"] = TrainingModule(
            id="soar_implementation", 
            name="SOAR Platform Implementation",
            description="Security orchestration, automation, and response platforms",
            objectives=[LearningObjective.SECURITY_AUTOMATION, LearningObjective.INCIDENT_CONTAINMENT],
            modality=TrainingModality.HANDS_ON_LAB,
            duration_minutes=240,
            difficulty=TrainingDifficulty.EXPERT
        )
    
    def assign_modules_to_paths(self):
        """Assign training modules to appropriate learning paths"""
        
        # Incident Response path
        self.learning_paths["incident_responder"].modules = [
            self.training_modules["incident_classification"],
            self.training_modules["incident_containment"],
            self.training_modules["network_traffic_analysis"],
            self.training_modules["malware_dynamic_analysis"],
            self.training_modules["soar_implementation"]
        ]
        
        # Threat Hunter path
        self.learning_paths["threat_hunter"].modules = [
            self.training_modules["threat_intel_collection"],
            self.training_modules["network_traffic_analysis"], 
            self.training_modules["malware_static_analysis"],
            self.training_modules["malware_dynamic_analysis"],
            self.training_modules["memory_forensics"]
        ]
        
        # Digital Forensics path
        self.learning_paths["forensics_analyst"].modules = [
            self.training_modules["disk_forensics"],
            self.training_modules["memory_forensics"],
            self.training_modules["network_traffic_analysis"],
            self.training_modules["malware_static_analysis"],
            self.training_modules["threat_intel_collection"]
        ]
        
        # Security Architecture path  
        self.learning_paths["security_architect"].modules = [
            self.training_modules["soar_implementation"],
            self.training_modules["threat_intel_collection"],
            self.training_modules["incident_containment"],
            self.training_modules["network_traffic_analysis"]
        ]
    
    async def assess_student_readiness(self, student_id: str, target_path: str) -> Dict:
        """Assess student readiness for a learning path"""
        logger.info(f"Assessing readiness for student {student_id} - path: {target_path}")
        
        if target_path not in self.learning_paths:
            raise ValueError(f"Unknown learning path: {target_path}")
        
        path = self.learning_paths[target_path]
        
        # Simulate skill assessment
        current_skills = {}
        for skill in DefensiveSkill:
            # Simulate current skill level (0.0-1.0)
            current_skills[skill.name] = random.uniform(0.3, 0.9)
        
        # Check prerequisites
        prerequisites_met = True
        missing_prerequisites = []
        
        for prereq in path.prerequisites:
            if current_skills.get(prereq.name, 0.0) < 0.6:
                prerequisites_met = False
                missing_prerequisites.append(prereq.name)
        
        # Calculate readiness score
        readiness_score = sum(current_skills.values()) / len(current_skills)
        
        assessment = {
            "student_id": student_id,
            "target_path": target_path, 
            "readiness_score": readiness_score,
            "prerequisites_met": prerequisites_met,
            "missing_prerequisites": missing_prerequisites,
            "recommended_start_level": self.get_recommended_start_level(readiness_score),
            "current_skills": current_skills,
            "estimated_completion_time": self.estimate_completion_time(path, readiness_score)
        }
        
        return assessment
    
    def get_recommended_start_level(self, readiness_score: float) -> TrainingDifficulty:
        """Recommend appropriate starting difficulty level"""
        if readiness_score >= 0.8:
            return TrainingDifficulty.EXPERT
        elif readiness_score >= 0.6:
            return TrainingDifficulty.ADVANCED
        elif readiness_score >= 0.4:
            return TrainingDifficulty.INTERMEDIATE
        else:
            return TrainingDifficulty.BEGINNER
    
    def estimate_completion_time(self, path: LearningPath, readiness_score: float) -> int:
        """Estimate completion time based on student readiness"""
        base_hours = path.estimated_hours
        
        # Adjust based on readiness
        if readiness_score >= 0.8:
            multiplier = 0.8  # Experienced students finish faster
        elif readiness_score >= 0.6:
            multiplier = 1.0  # Average time
        elif readiness_score >= 0.4:
            multiplier = 1.3  # Need more time
        else:
            multiplier = 1.6  # Beginners need significantly more time
        
        return int(base_hours * multiplier)
    
    async def execute_training_module(self, module: TrainingModule, student_id: str) -> TrainingResult:
        """Execute a training module for a student"""
        logger.info(f"Executing module '{module.name}' for student {student_id}")
        
        start_time = datetime.now()
        
        # Simulate training execution based on modality
        if module.modality == TrainingModality.HANDS_ON_LAB:
            result = await self.execute_hands_on_lab(module, student_id)
        elif module.modality == TrainingModality.SIMULATION:
            result = await self.execute_simulation(module, student_id)
        elif module.modality == TrainingModality.RED_TEAM_EXERCISE:
            result = await self.execute_red_team_exercise(module, student_id)
        else:
            result = await self.execute_standard_training(module, student_id)
        
        completion_time = datetime.now()
        time_spent = int((completion_time - start_time).total_seconds() / 60)
        
        # Create comprehensive training result
        training_result = TrainingResult(
            module_id=module.id,
            student_id=student_id,
            completion_time=completion_time,
            score=result["score"],
            skill_improvement=result["improvement"],
            time_spent_minutes=time_spent,
            mistakes_made=result.get("mistakes", []),
            strengths_identified=result.get("strengths", []),
            recommendations=result.get("recommendations", [])
        )
        
        # Update student progress
        if student_id not in self.student_progress:
            self.student_progress[student_id] = {}
        
        self.student_progress[student_id][module.id] = training_result
        
        logger.info(f"Module completed: {training_result.score:.1f}% score, {training_result.skill_improvement:.1f}% improvement")
        
        return training_result
    
    async def execute_hands_on_lab(self, module: TrainingModule, student_id: str) -> Dict:
        """Execute hands-on laboratory training"""
        # Simulate hands-on training with realistic performance variations
        await asyncio.sleep(1)  # Simulate training time
        
        base_score = random.uniform(70, 95)
        improvement = random.uniform(10, 25)
        
        return {
            "score": base_score,
            "improvement": improvement,
            "mistakes": self.generate_common_mistakes(module),
            "strengths": self.identify_strengths(module),
            "recommendations": self.generate_recommendations(module, base_score)
        }
    
    async def execute_simulation(self, module: TrainingModule, student_id: str) -> Dict:
        """Execute simulation-based training"""
        await asyncio.sleep(1)  # Simulate training time
        
        # Simulations tend to have slightly lower scores but higher improvement
        base_score = random.uniform(65, 90)
        improvement = random.uniform(15, 30)
        
        return {
            "score": base_score,
            "improvement": improvement,
            "mistakes": self.generate_simulation_mistakes(module),
            "strengths": self.identify_simulation_strengths(module),
            "recommendations": self.generate_simulation_recommendations(module, base_score)
        }
    
    async def execute_red_team_exercise(self, module: TrainingModule, student_id: str) -> Dict:
        """Execute red team exercise training"""
        await asyncio.sleep(2)  # Simulate longer exercise time
        
        # Red team exercises are challenging but provide high improvement
        base_score = random.uniform(60, 85)
        improvement = random.uniform(20, 35)
        
        return {
            "score": base_score, 
            "improvement": improvement,
            "mistakes": self.generate_red_team_mistakes(module),
            "strengths": self.identify_red_team_strengths(module),
            "recommendations": self.generate_red_team_recommendations(module, base_score)
        }
    
    async def execute_standard_training(self, module: TrainingModule, student_id: str) -> Dict:
        """Execute standard theoretical training"""
        await asyncio.sleep(0.5)  # Simulate training time
        
        base_score = random.uniform(75, 95)
        improvement = random.uniform(8, 20)
        
        return {
            "score": base_score,
            "improvement": improvement,
            "mistakes": [],
            "strengths": ["theoretical_knowledge"],
            "recommendations": ["Apply knowledge in practical scenarios"]
        }
    
    def generate_common_mistakes(self, module: TrainingModule) -> List[str]:
        """Generate realistic common mistakes for hands-on labs"""
        mistake_db = {
            "malware_static_analysis": [
                "Missed packed executable detection",
                "Incomplete string analysis", 
                "Failed to identify obfuscation techniques"
            ],
            "network_traffic_analysis": [
                "Overlooked encrypted channel indicators",
                "Missed lateral movement patterns",
                "Incomplete protocol analysis"
            ],
            "incident_containment": [
                "Incomplete asset isolation",
                "Missing communication documentation",
                "Inadequate evidence preservation"
            ]
        }
        
        return random.sample(mistake_db.get(module.id, ["Minor analytical oversight"]), k=min(2, len(mistake_db.get(module.id, []))))
    
    def identify_strengths(self, module: TrainingModule) -> List[str]:
        """Identify student strengths during training"""
        strength_db = {
            "malware_static_analysis": [
                "Strong pattern recognition",
                "Thorough documentation practices",
                "Effective tool utilization"
            ],
            "network_traffic_analysis": [
                "Excellent protocol knowledge",
                "Strong anomaly detection",
                "Comprehensive baseline analysis"
            ],
            "incident_containment": [
                "Clear communication skills",
                "Methodical approach",
                "Strong decision-making under pressure"
            ]
        }
        
        return random.sample(strength_db.get(module.id, ["Good analytical skills"]), k=2)
    
    def generate_recommendations(self, module: TrainingModule, score: float) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if score < 70:
            recommendations.append("Review fundamental concepts before advancing")
            recommendations.append("Practice with additional lab exercises")
        elif score < 85:
            recommendations.append("Focus on advanced techniques in next module")
            recommendations.append("Consider specialized training in weak areas")
        else:
            recommendations.append("Ready for expert-level challenges")
            recommendations.append("Consider mentoring other students")
        
        # Module-specific recommendations
        if "malware" in module.id:
            recommendations.append("Stay updated on latest malware trends")
        elif "network" in module.id:
            recommendations.append("Practice with diverse network architectures")
        elif "incident" in module.id:
            recommendations.append("Develop crisis communication skills")
        
        return recommendations[:3]  # Limit to top 3 recommendations
    
    def generate_simulation_mistakes(self, module: TrainingModule) -> List[str]:
        """Generate simulation-specific mistakes"""
        return [
            "Simulation environment limitations not considered",
            "Real-world variables underestimated",
            "Time pressure management issues"
        ][:random.randint(1, 2)]
    
    def identify_simulation_strengths(self, module: TrainingModule) -> List[str]:
        """Identify simulation-specific strengths"""
        return [
            "Adaptability to changing scenarios",
            "Strong problem-solving under constraints",
            "Effective use of available tools"
        ][:2]
    
    def generate_simulation_recommendations(self, module: TrainingModule, score: float) -> List[str]:
        """Generate simulation-specific recommendations"""
        return [
            "Practice in production-like environments",
            "Develop scenario-specific playbooks",
            "Enhance real-time decision making"
        ][:2]
    
    def generate_red_team_mistakes(self, module: TrainingModule) -> List[str]:
        """Generate red team exercise-specific mistakes"""
        return [
            "Insufficient reconnaissance phase",
            "Overlooked defensive countermeasures",
            "Communication breakdown during exercise"
        ][:random.randint(1, 3)]
    
    def identify_red_team_strengths(self, module: TrainingModule) -> List[str]:
        """Identify red team exercise strengths"""
        return [
            "Strong tactical thinking",
            "Excellent team coordination",
            "Creative problem solving"
        ][:2]
    
    def generate_red_team_recommendations(self, module: TrainingModule, score: float) -> List[str]:
        """Generate red team exercise recommendations"""
        return [
            "Participate in additional red team exercises",
            "Study advanced adversary techniques",
            "Develop blue team perspective"
        ][:2]
    
    async def execute_learning_path(self, student_id: str, path_name: str) -> Dict:
        """Execute complete learning path for student"""
        logger.info(f"Starting learning path '{path_name}' for student {student_id}")
        
        if path_name not in self.learning_paths:
            raise ValueError(f"Unknown learning path: {path_name}")
        
        path = self.learning_paths[path_name]
        
        # Assess readiness first
        readiness = await self.assess_student_readiness(student_id, path_name)
        
        if not readiness["prerequisites_met"]:
            logger.warning(f"Student {student_id} missing prerequisites: {readiness['missing_prerequisites']}")
        
        # Execute all modules in path
        path_results = []
        total_score = 0
        total_improvement = 0
        
        for module in path.modules:
            try:
                result = await self.execute_training_module(module, student_id)
                path_results.append(result)
                total_score += result.score
                total_improvement += result.skill_improvement
                
                # Simulate delay between modules
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to execute module {module.id}: {e}")
                continue
        
        # Calculate path completion results
        avg_score = total_score / len(path_results) if path_results else 0
        avg_improvement = total_improvement / len(path_results) if path_results else 0
        
        path_completion = {
            "student_id": student_id,
            "path_name": path_name,
            "modules_completed": len(path_results),
            "modules_total": len(path.modules),
            "completion_rate": len(path_results) / len(path.modules),
            "average_score": avg_score,
            "average_improvement": avg_improvement,
            "total_time_hours": sum(r.time_spent_minutes for r in path_results) / 60,
            "certification_earned": avg_score >= 80 and len(path_results) == len(path.modules),
            "detailed_results": path_results,
            "readiness_assessment": readiness
        }
        
        logger.info(f"Learning path completed: {avg_score:.1f}% average score")
        
        return path_completion
    
    async def generate_training_report(self, student_id: str) -> Dict:
        """Generate comprehensive training report for student"""
        
        if student_id not in self.student_progress:
            return {"error": f"No training data found for student {student_id}"}
        
        progress = self.student_progress[student_id]
        
        # Calculate overall statistics
        all_results = list(progress.values())
        total_modules = len(all_results)
        avg_score = sum(r.score for r in all_results) / total_modules
        avg_improvement = sum(r.skill_improvement for r in all_results) / total_modules
        total_time_hours = sum(r.time_spent_minutes for r in all_results) / 60
        
        # Identify strongest and weakest areas
        module_scores = {r.module_id: r.score for r in all_results}
        strongest_module = max(module_scores.keys(), key=lambda k: module_scores[k])
        weakest_module = min(module_scores.keys(), key=lambda k: module_scores[k])
        
        # Generate comprehensive report
        report = {
            "report_id": f"TRAINING-{student_id}-{datetime.now().strftime('%Y%m%d')}",
            "student_id": student_id,
            "report_date": datetime.now().isoformat(),
            "training_summary": {
                "modules_completed": total_modules,
                "total_training_hours": round(total_time_hours, 1),
                "average_score": round(avg_score, 1),
                "average_improvement": round(avg_improvement, 1),
                "strongest_area": strongest_module,
                "development_area": weakest_module
            },
            "detailed_progress": all_results,
            "recommendations": self.generate_student_recommendations(all_results),
            "next_steps": self.suggest_next_steps(all_results),
            "certification_status": self.check_certification_eligibility(all_results)
        }
        
        return report
    
    def generate_student_recommendations(self, results: List[TrainingResult]) -> List[str]:
        """Generate personalized recommendations based on training results"""
        recommendations = []
        
        avg_score = sum(r.score for r in results) / len(results)
        
        if avg_score >= 90:
            recommendations.append("Excellent performance! Consider advanced specialization training")
            recommendations.append("Explore instructor or mentor opportunities")
        elif avg_score >= 80:
            recommendations.append("Strong performance! Ready for expert-level challenges")
            recommendations.append("Consider pursuing professional certifications")
        elif avg_score >= 70:
            recommendations.append("Good progress! Focus on strengthening weak areas")
            recommendations.append("Additional hands-on practice recommended")
        else:
            recommendations.append("Review fundamental concepts before advancing")
            recommendations.append("Consider additional mentoring or tutoring")
        
        # Add specific recommendations based on common mistakes
        all_mistakes = []
        for result in results:
            all_mistakes.extend(result.mistakes_made)
        
        mistake_patterns = {}
        for mistake in all_mistakes:
            for category in ["analysis", "documentation", "communication", "technical"]:
                if category in mistake.lower():
                    mistake_patterns[category] = mistake_patterns.get(category, 0) + 1
        
        if mistake_patterns:
            top_issue = max(mistake_patterns.keys(), key=lambda k: mistake_patterns[k])
            recommendations.append(f"Focus on improving {top_issue} skills")
        
        return recommendations[:4]  # Limit to top 4 recommendations
    
    def suggest_next_steps(self, results: List[TrainingResult]) -> List[str]:
        """Suggest next steps based on training completion"""
        next_steps = []
        
        completed_modules = {r.module_id for r in results}
        avg_score = sum(r.score for r in results) / len(results)
        
        # Suggest advanced modules based on what's been completed
        if "malware_static_analysis" in completed_modules and avg_score >= 75:
            next_steps.append("Advance to malware_dynamic_analysis module")
        
        if "incident_classification" in completed_modules and avg_score >= 80:
            next_steps.append("Ready for incident_containment advanced training")
        
        if "network_traffic_analysis" in completed_modules and avg_score >= 85:
            next_steps.append("Consider specialized network forensics training")
        
        # General next steps
        if avg_score >= 85:
            next_steps.append("Explore leadership and team management training")
            next_steps.append("Consider contributing to training material development")
        
        if not next_steps:
            next_steps.append("Continue building fundamental skills")
            next_steps.append("Seek additional mentoring and guidance")
        
        return next_steps
    
    def check_certification_eligibility(self, results: List[TrainingResult]) -> Dict:
        """Check eligibility for various certifications"""
        completed_modules = {r.module_id for r in results}
        avg_score = sum(r.score for r in results) / len(results)
        
        certifications = {
            "Defensive Security Analyst": {
                "eligible": avg_score >= 80 and len(completed_modules) >= 3,
                "requirements_met": avg_score >= 80,
                "modules_required": 3,
                "modules_completed": len(completed_modules)
            },
            "Advanced Incident Responder": {
                "eligible": "incident_containment" in completed_modules and avg_score >= 85,
                "requirements_met": avg_score >= 85,
                "modules_required": ["incident_classification", "incident_containment"],
                "modules_completed": list(completed_modules)
            },
            "Digital Forensics Specialist": {
                "eligible": any("forensics" in module for module in completed_modules) and avg_score >= 80,
                "requirements_met": avg_score >= 80,
                "modules_required": ["disk_forensics", "memory_forensics"],
                "modules_completed": list(completed_modules)
            }
        }
        
        return certifications


async def main():
    """Main demonstration function"""
    logger.info("🎓 Starting Enhanced Defensive Training System")
    
    training_system = EnhancedDefensiveTraining()
    
    # Simulate training for multiple students
    students = ["student_001", "student_002", "student_003"]
    learning_paths = ["incident_responder", "threat_hunter", "forensics_analyst"]
    
    all_reports = []
    
    for i, student_id in enumerate(students):
        path = learning_paths[i % len(learning_paths)]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Student {student_id} - Path: {path}")
        logger.info('='*60)
        
        try:
            # Execute learning path
            completion_result = await training_system.execute_learning_path(student_id, path)
            
            # Generate training report
            training_report = await training_system.generate_training_report(student_id)
            
            all_reports.append({
                "student": student_id,
                "path": path,
                "completion": completion_result,
                "report": training_report
            })
            
            # Print summary
            print(f"\n🎯 {student_id} Results:")
            print(f"   Path: {path}")
            print(f"   Modules Completed: {completion_result['modules_completed']}/{completion_result['modules_total']}")
            print(f"   Average Score: {completion_result['average_score']:.1f}%")
            print(f"   Skill Improvement: {completion_result['average_improvement']:.1f}%")
            print(f"   Certification Earned: {'✅' if completion_result['certification_earned'] else '❌'}")
            
        except Exception as e:
            logger.error(f"Training failed for {student_id}: {e}")
    
    # Save comprehensive training report
    final_report = {
        "training_session_id": f"ENHANCED-TRAINING-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "session_date": datetime.now().isoformat(),
        "students_trained": len(students),
        "learning_paths_executed": len(set(learning_paths)),
        "total_training_hours": sum(
            report["completion"]["total_time_hours"] 
            for report in all_reports 
            if "completion" in report
        ),
        "average_completion_rate": sum(
            report["completion"]["completion_rate"] 
            for report in all_reports 
            if "completion" in report
        ) / len(all_reports),
        "student_reports": all_reports,
        "system_performance": {
            "modules_available": len(training_system.training_modules),
            "learning_paths_available": len(training_system.learning_paths),
            "training_modalities": len(TrainingModality),
            "assessment_objectives": len(LearningObjective)
        }
    }
    
    # Save report to file
    report_path = Path(f"enhanced_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("🎓 ENHANCED DEFENSIVE TRAINING COMPLETE")
    print('='*80)
    print(f"📊 Training Report: {report_path}")
    print(f"👥 Students Trained: {final_report['students_trained']}")
    print(f"⏱️  Total Training Hours: {final_report['total_training_hours']:.1f}")
    print(f"📈 Average Completion Rate: {final_report['average_completion_rate']:.1%}")
    print(f"🏆 Certifications Available: {sum(1 for r in all_reports if r.get('completion', {}).get('certification_earned', False))}")


if __name__ == "__main__":
    asyncio.run(main())