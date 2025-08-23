#!/usr/bin/env python3
"""
Defensive Security Demo for GAN-Cyber-Range-v2

Demonstrates core defensive capabilities without requiring heavy ML dependencies.
This focuses on defensive training, threat detection, and security research.
"""

import json
import time
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Defensive threat assessment levels"""
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"

class DefenseAction(Enum):
    """Defensive response actions"""
    MONITOR = "monitor"
    ALERT = "alert"
    BLOCK = "block"
    ISOLATE = "isolate"
    INVESTIGATE = "investigate"

@dataclass
class ThreatSignature:
    """Defensive threat signature for detection"""
    signature_id: str
    name: str
    description: str
    threat_level: ThreatLevel
    indicators: List[str]
    defense_actions: List[DefenseAction]
    created_at: datetime
    
class DefensiveTrainingSimulator:
    """Simulates defensive cybersecurity scenarios for training"""
    
    def __init__(self):
        self.threat_signatures = []
        self.detected_threats = []
        self.defense_responses = []
        self.training_scenarios = []
        
    def create_defensive_signature(self, name: str, indicators: List[str], 
                                 threat_level: ThreatLevel) -> ThreatSignature:
        """Create a new defensive threat signature"""
        
        signature = ThreatSignature(
            signature_id=f"DEF-{len(self.threat_signatures)+1:04d}",
            name=name,
            description=f"Defensive signature for {name}",
            threat_level=threat_level,
            indicators=indicators,
            defense_actions=[DefenseAction.MONITOR, DefenseAction.ALERT],
            created_at=datetime.now()
        )
        
        self.threat_signatures.append(signature)
        logger.info(f"Created defensive signature: {signature.name}")
        return signature
    
    def simulate_threat_detection(self, scenario: str) -> Dict:
        """Simulate defensive threat detection for training"""
        
        # Simulate realistic threat detection scenario
        detection = {
            "detection_id": f"DET-{random.randint(1000, 9999)}",
            "scenario": scenario,
            "timestamp": datetime.now().isoformat(),
            "confidence": round(random.uniform(0.7, 0.99), 2),
            "threat_level": random.choice(list(ThreatLevel)).value,
            "source_ip": f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
            "defensive_measures": []
        }
        
        # Add defensive response based on threat level
        if detection["threat_level"] in ["high", "critical"]:
            detection["defensive_measures"].extend([
                "Network traffic isolation initiated",
                "Security team alerted",
                "Forensic logging activated"
            ])
        
        self.detected_threats.append(detection)
        logger.info(f"Threat detected in scenario '{scenario}': {detection['confidence']} confidence")
        
        return detection
    
    def create_training_scenario(self, scenario_name: str, 
                               objectives: List[str]) -> Dict:
        """Create defensive training scenario"""
        
        scenario = {
            "scenario_id": f"TRAIN-{len(self.training_scenarios)+1:03d}",
            "name": scenario_name,
            "type": "defensive_training",
            "objectives": objectives,
            "duration": f"{random.randint(30, 120)} minutes",
            "difficulty": random.choice(["beginner", "intermediate", "advanced"]),
            "skills_practiced": [
                "Threat detection",
                "Incident response", 
                "Digital forensics",
                "Security monitoring"
            ],
            "created_at": datetime.now().isoformat()
        }
        
        self.training_scenarios.append(scenario)
        logger.info(f"Created training scenario: {scenario_name}")
        
        return scenario
    
    def generate_defense_report(self) -> Dict:
        """Generate defensive security training report"""
        
        report = {
            "report_id": f"RPT-{datetime.now().strftime('%Y%m%d%H%M')}",
            "generated_at": datetime.now().isoformat(),
            "report_type": "defensive_training_summary",
            "statistics": {
                "signatures_created": len(self.threat_signatures),
                "threats_detected": len(self.detected_threats),
                "training_scenarios": len(self.training_scenarios),
                "avg_detection_confidence": round(
                    sum(t.get("confidence", 0) for t in self.detected_threats) / 
                    max(len(self.detected_threats), 1), 2
                )
            },
            "defensive_capabilities": [
                "Real-time threat monitoring",
                "Automated incident response",
                "Security training scenarios",
                "Threat signature management"
            ],
            "recommendations": [
                "Continue regular security training exercises",
                "Update threat signatures based on latest intelligence", 
                "Enhance monitoring capabilities for early detection"
            ]
        }
        
        return report

class SecureTrainingEnvironment:
    """Secure environment for defensive cybersecurity training"""
    
    def __init__(self):
        self.simulator = DefensiveTrainingSimulator()
        self.session_id = f"SESSION-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
    def setup_defensive_environment(self):
        """Setup secure defensive training environment"""
        
        logger.info("Setting up defensive training environment...")
        
        # Create sample defensive signatures
        signatures = [
            ("Malware Detection Pattern", ["suspicious_hash", "anomalous_behavior"], ThreatLevel.HIGH),
            ("Phishing Email Indicator", ["suspicious_link", "spoofed_sender"], ThreatLevel.MEDIUM),
            ("Network Intrusion Attempt", ["port_scan", "unauthorized_access"], ThreatLevel.CRITICAL),
            ("Data Exfiltration Pattern", ["large_transfer", "unusual_timing"], ThreatLevel.HIGH)
        ]
        
        for name, indicators, level in signatures:
            self.simulator.create_defensive_signature(name, indicators, level)
        
        logger.info("Defensive signatures loaded")
    
    def run_training_exercise(self):
        """Run defensive cybersecurity training exercise"""
        
        logger.info("Starting defensive training exercise...")
        
        # Create training scenarios
        scenarios = [
            ("Incident Response Drill", ["Detect threat", "Contain incident", "Document findings"]),
            ("Threat Hunting Exercise", ["Identify indicators", "Track attacker", "Prevent damage"]),
            ("Digital Forensics Lab", ["Preserve evidence", "Analyze artifacts", "Report findings"]),
            ("Security Monitoring Practice", ["Monitor traffic", "Identify anomalies", "Alert team"])
        ]
        
        for name, objectives in scenarios:
            scenario = self.simulator.create_training_scenario(name, objectives)
            
            # Simulate threat detection in scenario
            detection = self.simulator.simulate_threat_detection(scenario["name"])
            
            # Brief pause to simulate realistic timing
            time.sleep(1)
        
        logger.info("Training exercise completed")
    
    def generate_session_report(self) -> Dict:
        """Generate comprehensive defensive training session report"""
        
        report = self.simulator.generate_defense_report()
        report["session_id"] = self.session_id
        report["training_effectiveness"] = {
            "scenarios_completed": len(self.simulator.training_scenarios),
            "threats_successfully_detected": len([t for t in self.simulator.detected_threats 
                                                if t.get("confidence", 0) > 0.8]),
            "defensive_skills_practiced": [
                "Threat signature creation",
                "Real-time monitoring",
                "Incident response",
                "Training scenario development"
            ]
        }
        
        return report

def main():
    """Main defensive cybersecurity demo"""
    
    print("🛡️  GAN-Cyber-Range-v2 Defensive Security Demo")
    print("=" * 60)
    print("Demonstrating defensive cybersecurity training capabilities\n")
    
    # Initialize secure training environment
    env = SecureTrainingEnvironment()
    
    # Setup defensive environment
    env.setup_defensive_environment()
    print("✅ Defensive training environment ready\n")
    
    # Run training exercise
    env.run_training_exercise()
    print("\n✅ Defensive training exercise completed\n")
    
    # Generate and display report
    report = env.generate_session_report()
    
    print("📊 DEFENSIVE TRAINING REPORT")
    print("-" * 30)
    print(f"Session ID: {report['session_id']}")
    print(f"Generated: {report['generated_at']}")
    print(f"Report Type: {report['report_type']}")
    
    stats = report['statistics']
    print(f"\nTraining Statistics:")
    print(f"  • Defensive signatures created: {stats['signatures_created']}")
    print(f"  • Threats detected: {stats['threats_detected']}")  
    print(f"  • Training scenarios: {stats['training_scenarios']}")
    print(f"  • Average detection confidence: {stats['avg_detection_confidence']}%")
    
    print(f"\nDefensive Capabilities Demonstrated:")
    for capability in report['defensive_capabilities']:
        print(f"  • {capability}")
    
    print(f"\nTraining Effectiveness:")
    eff = report['training_effectiveness'] 
    print(f"  • Scenarios completed: {eff['scenarios_completed']}")
    print(f"  • High-confidence detections: {eff['threats_successfully_detected']}")
    print(f"  • Skills practiced: {len(eff['defensive_skills_practiced'])}")
    
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    # Save report for analysis
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    
    report_file = report_dir / f"defensive_training_report_{report['report_id']}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Report saved to: {report_file}")
    print("\n🎯 Defensive cybersecurity training demo completed successfully!")
    print("Ready for advanced defensive security research and training.")

if __name__ == "__main__":
    main()