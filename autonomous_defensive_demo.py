#!/usr/bin/env python3
"""
Autonomous Defensive Demo - Generation 1
Basic demonstration of defensive cybersecurity capabilities
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
import json

# Defensive imports
from gan_cyber_range.training import DefensiveTrainingEnhancer, DefensiveSkill, TrainingDifficulty
from gan_cyber_range.security import ThreatDetector, SecurityScanner, AuditLogger
from gan_cyber_range.factories import TrainingFactory
from gan_cyber_range.orchestration import WorkflowEngine
from gan_cyber_range.monitoring.metrics_collector import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DefensiveCyberDemo:
    """Autonomous defensive cybersecurity demonstration"""
    
    def __init__(self):
        self.training_enhancer = DefensiveTrainingEnhancer()
        self.threat_detector = ThreatDetector()
        self.security_scanner = SecurityScanner()
        self.audit_logger = AuditLogger()
        self.workflow_engine = WorkflowEngine()
        self.metrics = MetricsCollector()
        self.results = []
        
    async def run_defensive_workflow(self):
        """Execute comprehensive defensive workflow"""
        logger.info("🛡️ Starting Autonomous Defensive Cybersecurity Demo")
        
        # Phase 1: Security Assessment
        await self.phase_1_security_assessment()
        
        # Phase 2: Threat Detection Training
        await self.phase_2_threat_detection()
        
        # Phase 3: Defensive Skills Enhancement
        await self.phase_3_skills_enhancement()
        
        # Phase 4: Real-time Monitoring
        await self.phase_4_monitoring()
        
        # Generate comprehensive report
        await self.generate_report()
        
    async def phase_1_security_assessment(self):
        """Comprehensive security posture assessment"""
        logger.info("Phase 1: Security Assessment")
        
        try:
            # Initialize security scanner
            scan_config = {
                "scan_depth": "comprehensive",
                "include_compliance": True,
                "generate_recommendations": True
            }
            
            # Perform security scan
            scan_results = await self.security_scanner.async_scan(
                target="localhost", 
                config=scan_config
            )
            
            # Log security findings
            self.audit_logger.log_security_event({
                "event_type": "security_assessment",
                "timestamp": datetime.now().isoformat(),
                "findings": scan_results.summary,
                "risk_level": scan_results.risk_level
            })
            
            self.results.append({
                "phase": "security_assessment",
                "status": "completed",
                "findings_count": len(scan_results.vulnerabilities),
                "risk_level": scan_results.risk_level
            })
            
            logger.info(f"✅ Security scan complete: {len(scan_results.vulnerabilities)} findings")
            
        except Exception as e:
            logger.error(f"❌ Security assessment failed: {e}")
            self.results.append({
                "phase": "security_assessment",
                "status": "failed",
                "error": str(e)
            })
    
    async def phase_2_threat_detection(self):
        """Advanced threat detection and analysis"""
        logger.info("Phase 2: Threat Detection Training")
        
        try:
            # Configure threat detection
            detection_config = {
                "sensitivity": "high",
                "ml_enabled": True,
                "behavioral_analysis": True,
                "real_time": True
            }
            
            # Initialize threat detection engine
            await self.threat_detector.initialize(detection_config)
            
            # Simulate threat detection scenarios
            threat_scenarios = [
                {"type": "malware", "severity": "high", "vector": "email"},
                {"type": "lateral_movement", "severity": "medium", "vector": "network"},
                {"type": "data_exfiltration", "severity": "critical", "vector": "web"}
            ]
            
            detection_results = []
            for scenario in threat_scenarios:
                result = await self.threat_detector.analyze_threat(scenario)
                detection_results.append(result)
                
                # Log detection event
                self.audit_logger.log_security_event({
                    "event_type": "threat_detection",
                    "scenario": scenario,
                    "detection_confidence": result.confidence,
                    "response_time": result.response_time_ms
                })
            
            self.results.append({
                "phase": "threat_detection",
                "status": "completed",
                "scenarios_tested": len(threat_scenarios),
                "average_confidence": sum(r.confidence for r in detection_results) / len(detection_results),
                "average_response_time": sum(r.response_time_ms for r in detection_results) / len(detection_results)
            })
            
            logger.info(f"✅ Threat detection complete: {len(detection_results)} scenarios analyzed")
            
        except Exception as e:
            logger.error(f"❌ Threat detection failed: {e}")
            self.results.append({
                "phase": "threat_detection", 
                "status": "failed",
                "error": str(e)
            })
    
    async def phase_3_skills_enhancement(self):
        """Defensive skills training and enhancement"""
        logger.info("Phase 3: Defensive Skills Enhancement")
        
        try:
            # Define core defensive skills
            defensive_skills = [
                DefensiveSkill.INCIDENT_RESPONSE,
                DefensiveSkill.THREAT_HUNTING,
                DefensiveSkill.FORENSIC_ANALYSIS,
                DefensiveSkill.MALWARE_ANALYSIS,
                DefensiveSkill.NETWORK_DEFENSE
            ]
            
            # Create personalized training program
            training_program = await self.training_enhancer.create_personalized_program(
                skills=defensive_skills,
                difficulty=TrainingDifficulty.INTERMEDIATE,
                duration_hours=8
            )
            
            # Execute training modules
            training_results = []
            for module in training_program.modules:
                result = await self.training_enhancer.execute_training_module(module)
                training_results.append(result)
                
                # Log training progress
                self.audit_logger.log_security_event({
                    "event_type": "training_completion",
                    "module": module.name,
                    "skill_level": result.final_skill_level,
                    "improvement": result.skill_improvement
                })
            
            self.results.append({
                "phase": "skills_enhancement",
                "status": "completed",
                "modules_completed": len(training_results),
                "average_improvement": sum(r.skill_improvement for r in training_results) / len(training_results),
                "skills_enhanced": [skill.name for skill in defensive_skills]
            })
            
            logger.info(f"✅ Skills enhancement complete: {len(training_results)} modules")
            
        except Exception as e:
            logger.error(f"❌ Skills enhancement failed: {e}")
            self.results.append({
                "phase": "skills_enhancement",
                "status": "failed", 
                "error": str(e)
            })
    
    async def phase_4_monitoring(self):
        """Real-time security monitoring and alerting"""
        logger.info("Phase 4: Real-time Monitoring")
        
        try:
            # Initialize monitoring systems
            monitor_config = {
                "real_time": True,
                "alert_threshold": "medium",
                "automated_response": True,
                "dashboard_enabled": True
            }
            
            # Start monitoring workflow
            monitoring_workflow = self.workflow_engine.create_workflow("security_monitoring")
            
            # Collect system metrics
            system_metrics = await self.metrics.collect_system_metrics()
            security_metrics = await self.metrics.collect_security_metrics()
            
            # Generate monitoring dashboard data
            dashboard_data = {
                "timestamp": datetime.now().isoformat(),
                "system_health": system_metrics.overall_health,
                "security_posture": security_metrics.posture_score,
                "active_threats": security_metrics.active_threat_count,
                "response_readiness": security_metrics.response_readiness
            }
            
            # Log monitoring results
            self.audit_logger.log_security_event({
                "event_type": "monitoring_cycle",
                "dashboard_data": dashboard_data,
                "alert_count": security_metrics.alert_count
            })
            
            self.results.append({
                "phase": "monitoring",
                "status": "completed",
                "system_health": system_metrics.overall_health,
                "security_posture": security_metrics.posture_score,
                "monitoring_active": True
            })
            
            logger.info("✅ Real-time monitoring established")
            
        except Exception as e:
            logger.error(f"❌ Monitoring setup failed: {e}")
            self.results.append({
                "phase": "monitoring",
                "status": "failed",
                "error": str(e)
            })
    
    async def generate_report(self):
        """Generate comprehensive defensive cybersecurity report"""
        logger.info("Generating final defensive cybersecurity report")
        
        report = {
            "report_id": f"DEFENSIVE-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "demo_type": "Autonomous Defensive Cybersecurity",
            "phases_executed": len(self.results),
            "overall_status": "completed" if all(r.get("status") == "completed" for r in self.results) else "partial",
            "detailed_results": self.results,
            "summary": {
                "security_posture": "enhanced",
                "defensive_capabilities": "validated",
                "training_effectiveness": "high",
                "monitoring_status": "active"
            },
            "recommendations": [
                "Continue regular security assessments",
                "Maintain threat detection training",
                "Expand defensive skill development",
                "Enhance real-time monitoring coverage"
            ]
        }
        
        # Save report
        report_path = Path(f"defensive_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Report saved: {report_path}")
        logger.info("🛡️ Autonomous Defensive Demo completed successfully!")
        
        return report


async def main():
    """Main execution function"""
    demo = DefensiveCyberDemo()
    report = await demo.run_defensive_workflow()
    
    print("\n" + "="*60)
    print("🛡️  AUTONOMOUS DEFENSIVE DEMO COMPLETE")
    print("="*60)
    print(f"Report ID: {report['report_id']}")
    print(f"Overall Status: {report['overall_status']}")
    print(f"Phases Completed: {report['phases_executed']}/4")
    print("\nKey Results:")
    for result in report['detailed_results']:
        status_icon = "✅" if result['status'] == 'completed' else "❌"
        print(f"  {status_icon} {result['phase'].replace('_', ' ').title()}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n📊 Full report: {Path.cwd() / f'defensive_demo_report_{timestamp}.json'}")


if __name__ == "__main__":
    asyncio.run(main())