"""
Minimal GAN implementation without heavy dependencies for basic functionality validation.

This module provides lightweight versions of core GAN functionality that can run
without PyTorch or other heavy ML dependencies for basic testing and development.
"""

import numpy as np
import logging
import json
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
from datetime import datetime
import re

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


class MinimalAttackGenerator:
    """Minimal attack generator using rule-based generation instead of neural networks"""
    
    def __init__(self, attack_types: List[str] = None):
        self.attack_types = attack_types or ["malware", "network", "web", "social_engineering"]
        self.attack_patterns = self._initialize_patterns()
        self.generation_history = []
        
    def _initialize_patterns(self) -> Dict[str, List[Dict]]:
        """Initialize rule-based attack patterns"""
        return {
            "malware": [
                {"template": "powershell -enc {encoded_payload}", "severity": 0.8, "stealth": 0.6},
                {"template": "cmd.exe /c {command}", "severity": 0.7, "stealth": 0.4},
                {"template": "rundll32.exe {dll_path},{function}", "severity": 0.9, "stealth": 0.8},
            ],
            "network": [
                {"template": "nmap -sS -T4 {target_ip}", "severity": 0.3, "stealth": 0.7},
                {"template": "nmap -sV -p- {target_ip}", "severity": 0.5, "stealth": 0.5},
                {"template": "masscan -p1-65535 {target_ip} --rate=1000", "severity": 0.6, "stealth": 0.3},
            ],
            "web": [
                {"template": "' OR '1'='1' --", "severity": 0.8, "stealth": 0.4},
                {"template": "<script>alert('xss')</script>", "severity": 0.6, "stealth": 0.3},
                {"template": "../../../etc/passwd", "severity": 0.7, "stealth": 0.5},
            ],
            "social_engineering": [
                {"template": "Your account has been compromised. Click here to verify: {phish_url}", "severity": 0.9, "stealth": 0.8},
                {"template": "IT Support: Please provide your password for system maintenance", "severity": 0.8, "stealth": 0.6},
                {"template": "Urgent: Update required. Download from: {malware_url}", "severity": 0.7, "stealth": 0.7},
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
        
        generated_attacks = []
        
        for _ in range(num_samples):
            # Select attack type
            if attack_type and attack_type in self.attack_patterns:
                selected_type = attack_type
            else:
                selected_type = np.random.choice(self.attack_types)
            
            # Select pattern
            patterns = self.attack_patterns[selected_type]
            pattern = np.random.choice(patterns)
            
            # Generate payload
            payload = self._generate_payload(pattern, selected_type)
            
            # Create attack vector
            attack = AttackVector(
                attack_type=selected_type,
                payload=payload,
                techniques=self._get_techniques(selected_type, payload),
                severity=pattern["severity"] + np.random.normal(0, 0.1),
                stealth_level=pattern["stealth"] + np.random.normal(0, 0.1),
                target_systems=self._get_target_systems(selected_type),
                metadata={
                    "generation_method": "rule_based",
                    "pattern_used": pattern["template"],
                    "generator_version": "minimal_v1.0"
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
        
        # Replace placeholders
        replacements = {
            "encoded_payload": self._generate_encoded_payload(),
            "command": self._generate_command(),
            "dll_path": self._generate_dll_path(),
            "function": self._generate_function_name(),
            "target_ip": self._generate_target_ip(),
            "phish_url": self._generate_phish_url(),
            "malware_url": self._generate_malware_url()
        }
        
        for placeholder, value in replacements.items():
            template = template.replace(f"{{{placeholder}}}", value)
        
        return template
    
    def _generate_encoded_payload(self) -> str:
        """Generate base64-like encoded payload"""
        import base64
        payload = "whoami; net user"
        return base64.b64encode(payload.encode()).decode()[:20] + "..."
    
    def _generate_command(self) -> str:
        commands = ["whoami", "ipconfig", "net user", "dir c:\\", "systeminfo"]
        return np.random.choice(commands)
    
    def _generate_dll_path(self) -> str:
        dlls = ["shell32.dll", "kernel32.dll", "user32.dll", "advapi32.dll"]
        return f"c:\\windows\\system32\\{np.random.choice(dlls)}"
    
    def _generate_function_name(self) -> str:
        functions = ["ShellExecuteA", "WinExec", "CreateProcessA", "LoadLibraryA"]
        return np.random.choice(functions)
    
    def _generate_target_ip(self) -> str:
        return f"192.168.{np.random.randint(1,254)}.{np.random.randint(1,254)}"
    
    def _generate_phish_url(self) -> str:
        domains = ["secure-bank.com", "microsoft-update.net", "gmail-security.org"]
        return f"https://{np.random.choice(domains)}/verify"
    
    def _generate_malware_url(self) -> str:
        domains = ["download-center.net", "software-updates.com", "security-patch.org"]
        files = ["update.exe", "patch.zip", "security_fix.msi"]
        return f"https://{np.random.choice(domains)}/{np.random.choice(files)}"
    
    def _get_techniques(self, attack_type: str, payload: str) -> List[str]:
        """Map attack to MITRE ATT&CK techniques"""
        technique_mapping = {
            "malware": {
                "powershell": ["T1059.001"],  # PowerShell
                "cmd": ["T1059.003"],         # Windows Command Shell
                "rundll32": ["T1055.001"],    # DLL Side-Loading
            },
            "network": {
                "nmap": ["T1046"],            # Network Service Scanning
                "masscan": ["T1046"],         # Network Service Scanning
                "scan": ["T1046"]             # Network Service Scanning
            },
            "web": {
                "sql": ["T1190"],             # Exploit Public-Facing Application
                "xss": ["T1190"],             # Exploit Public-Facing Application
                "lfi": ["T1190"]              # Exploit Public-Facing Application
            },
            "social_engineering": {
                "phish": ["T1566.002"],       # Spearphishing Link
                "password": ["T1110"],        # Brute Force
                "download": ["T1204.002"]     # Malicious File
            }
        }
        
        if attack_type in technique_mapping:
            for keyword, techniques in technique_mapping[attack_type].items():
                if keyword.lower() in payload.lower():
                    return techniques
        
        # Default techniques
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
            "malware": ["windows_workstation", "windows_server"],
            "network": ["router", "firewall", "server"],
            "web": ["web_server", "application_server", "database"],
            "social_engineering": ["email_client", "web_browser", "user_workstation"]
        }
        
        return targets.get(attack_type, ["generic_target"])
    
    def _is_valid_attack(self, attack: AttackVector) -> bool:
        """Validate generated attack vector"""
        # Check required fields
        if not attack.payload or len(str(attack.payload)) < 5:
            return False
        
        if not attack.techniques:
            return False
        
        # Severity and stealth bounds
        if not (0.0 <= attack.severity <= 1.0):
            attack.severity = max(0.0, min(1.0, attack.severity))
        
        if not (0.0 <= attack.stealth_level <= 1.0):
            attack.stealth_level = max(0.0, min(1.0, attack.stealth_level))
        
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
        technique_diversity = min(1.0, unique_techniques / 10)  # Normalize to common techniques
        
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
                "generator": "MinimalAttackGenerator",
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


class MockCyberRange:
    """Minimal cyber range for testing basic functionality"""
    
    def __init__(self):
        self.range_id = str(uuid.uuid4())
        self.status = "initializing"
        self.attack_generator = MinimalAttackGenerator()
        self.event_log = []
        
    def deploy(self, **kwargs) -> str:
        """Mock deployment"""
        self.status = "deployed"
        self.event_log.append({
            "timestamp": datetime.now().isoformat(),
            "event": "range_deployed",
            "range_id": self.range_id
        })
        logger.info(f"Mock cyber range {self.range_id} deployed")
        return self.range_id
    
    def start(self) -> None:
        """Start the mock range"""
        self.status = "running"
        self.event_log.append({
            "timestamp": datetime.now().isoformat(),
            "event": "range_started"
        })
        logger.info("Mock cyber range started")
    
    def generate_attacks(self, num_attacks: int = 5) -> List[AttackVector]:
        """Generate attacks using minimal generator"""
        return self.attack_generator.generate(num_attacks)
    
    def execute_attack(self, attack: AttackVector) -> Dict[str, Any]:
        """Mock attack execution"""
        result = {
            "attack_id": attack.attack_id,
            "status": "completed",
            "execution_time": np.random.uniform(1, 30),
            "success": np.random.choice([True, False], p=[0.7, 0.3]),
            "detected": np.random.choice([True, False], p=[0.4, 0.6])
        }
        
        self.event_log.append({
            "timestamp": datetime.now().isoformat(),
            "event": "attack_executed",
            "attack_id": attack.attack_id,
            "result": result
        })
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get mock metrics"""
        return {
            "range_id": self.range_id,
            "status": self.status,
            "uptime": "00:15:30",
            "attacks_executed": len([e for e in self.event_log if e["event"] == "attack_executed"]),
            "detection_rate": 0.4,
            "resource_usage": {
                "cpu": 25.5,
                "memory": 512.8,
                "network": 150.2
            }
        }


def validate_minimal_functionality() -> bool:
    """Validate that minimal functionality works"""
    try:
        # Test attack generation
        generator = MinimalAttackGenerator()
        attacks = generator.generate(num_samples=5)
        
        if len(attacks) != 5:
            logger.error(f"Expected 5 attacks, got {len(attacks)}")
            return False
        
        # Test diversity calculation
        diversity = generator.diversity_score(attacks)
        if not (0.0 <= diversity <= 1.0):
            logger.error(f"Invalid diversity score: {diversity}")
            return False
        
        # Test cyber range
        cyber_range = MockCyberRange()
        cyber_range.deploy()
        cyber_range.start()
        
        # Test attack execution
        attack = attacks[0]
        result = cyber_range.execute_attack(attack)
        
        if not result or "attack_id" not in result:
            logger.error("Attack execution failed")
            return False
        
        logger.info("✓ All minimal functionality tests passed")
        return True
        
    except Exception as e:
        logger.error(f"Minimal functionality validation failed: {e}")
        return False


if __name__ == "__main__":
    # Run validation
    logging.basicConfig(level=logging.INFO)
    success = validate_minimal_functionality()
    exit(0 if success else 1)