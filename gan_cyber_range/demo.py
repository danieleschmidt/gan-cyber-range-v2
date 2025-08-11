"""
GAN-Cyber-Range-v2 Lightweight Demo System

A minimal demonstration system that showcases core functionality 
without requiring heavy ML dependencies like PyTorch.
"""

import json
import random
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimpleAttackVector:
    """Lightweight attack vector for demo purposes"""
    attack_id: str
    attack_type: str
    payload: str
    severity: float
    stealth_level: float
    target_systems: List[str]
    techniques: List[str]
    timestamp: str
    success_probability: float


class LightweightAttackGenerator:
    """Simple attack generator using templates and randomization"""
    
    def __init__(self):
        self.attack_templates = {
            "malware": [
                "trojan.{variant}.exe --install --stealth",
                "ransomware.{variant} --encrypt /data --key {key}",
                "backdoor.{variant} --listen {port} --callback {domain}",
            ],
            "network": [
                "nmap -sS -O {target_ip} -p {ports}",
                "hydra -l admin -P wordlist.txt {protocol}://{target_ip}",
                "arp-spoof -i {interface} -t {target_ip} {gateway_ip}",
            ],
            "web": [
                "sqlmap -u {url} --batch --level 3",
                "XSS payload: <script>document.cookie</script>",
                "CSRF attack on {endpoint} with token bypass",
            ],
            "social_engineering": [
                "phishing email targeting {department} about {topic}",
                "vishing call impersonating {organization} IT support",
                "USB drop attack using {payload_type} at {location}",
            ]
        }
        
        self.mitre_techniques = {
            "malware": ["T1566.001", "T1204.002", "T1027"],
            "network": ["T1046", "T1110", "T1557.002"],
            "web": ["T1190", "T1059.007", "T1566.002"],
            "social_engineering": ["T1566.001", "T1659", "T1091"]
        }
        
        self.target_systems = [
            "windows_workstation", "linux_server", "web_server", 
            "database_server", "domain_controller", "firewall"
        ]
    
    def generate_attack(self, attack_type: Optional[str] = None) -> SimpleAttackVector:
        """Generate a single synthetic attack"""
        if attack_type is None:
            attack_type = random.choice(list(self.attack_templates.keys()))
        
        template = random.choice(self.attack_templates[attack_type])
        
        # Fill template with random values
        payload = self._fill_template(template, attack_type)
        
        return SimpleAttackVector(
            attack_id=str(uuid.uuid4()),
            attack_type=attack_type,
            payload=payload,
            severity=round(random.uniform(0.3, 1.0), 2),
            stealth_level=round(random.uniform(0.1, 0.9), 2),
            target_systems=random.sample(self.target_systems, random.randint(1, 3)),
            techniques=self.mitre_techniques[attack_type],
            timestamp=datetime.now().isoformat(),
            success_probability=round(random.uniform(0.2, 0.8), 2)
        )
    
    def generate_batch(self, count: int = 100, attack_type: Optional[str] = None) -> List[SimpleAttackVector]:
        """Generate multiple attacks"""
        logger.info(f"Generating {count} synthetic attacks")
        
        attacks = []
        for _ in range(count):
            attack = self.generate_attack(attack_type)
            attacks.append(attack)
        
        return attacks
    
    def _fill_template(self, template: str, attack_type: str) -> str:
        """Fill template placeholders with realistic values"""
        replacements = {
            "variant": random.choice(["alpha", "beta", "gamma", "delta"]),
            "key": f"key_{random.randint(1000, 9999)}",
            "port": str(random.choice([22, 80, 443, 3389, 5432, 3306])),
            "domain": f"c2-{random.randint(100, 999)}.malicious.com",
            "target_ip": f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}",
            "ports": "80,443,22,3389",
            "protocol": random.choice(["ssh", "ftp", "http"]),
            "interface": "eth0",
            "gateway_ip": "192.168.1.1",
            "url": f"http://vulnerable.com/{random.choice(['login', 'admin', 'api'])}",
            "endpoint": "/admin/users",
            "department": random.choice(["HR", "Finance", "IT", "Sales"]),
            "topic": random.choice(["security update", "policy change", "urgent notice"]),
            "organization": random.choice(["Microsoft", "Google", "IT Department"]),
            "payload_type": random.choice(["keylogger", "credential stealer", "backdoor"]),
            "location": random.choice(["parking lot", "lobby", "break room"])
        }
        
        for placeholder, value in replacements.items():
            template = template.replace(f"{{{placeholder}}}", value)
        
        return template


class SimpleCyberRange:
    """Lightweight cyber range for demonstration"""
    
    def __init__(self, name: str = "demo-range"):
        self.range_id = str(uuid.uuid4())
        self.name = name
        self.status = "initializing"
        self.start_time = None
        self.attack_generator = LightweightAttackGenerator()
        
        # Simulated infrastructure
        self.hosts = self._create_demo_hosts()
        self.networks = ["dmz", "internal", "management"]
        self.services = ["web", "database", "email", "file_share"]
        
        # Attack tracking
        self.active_attacks = {}
        self.completed_attacks = []
        self.detection_events = []
        
        logger.info(f"Initialized SimpleCyberRange: {self.name}")
    
    def deploy(self) -> str:
        """Deploy the demo cyber range"""
        logger.info(f"Deploying cyber range: {self.name}")
        
        # Simulate deployment time
        time.sleep(2)
        
        self.status = "deployed"
        self.start_time = datetime.now()
        
        # Create dashboard URL
        self.dashboard_url = f"http://localhost:8080/range/{self.range_id}"
        
        logger.info(f"Cyber range deployed successfully")
        logger.info(f"Dashboard: {self.dashboard_url}")
        
        return self.range_id
    
    def start_simulation(self, duration_minutes: int = 30) -> None:
        """Start attack simulation"""
        if self.status != "deployed":
            raise RuntimeError("Range must be deployed first")
        
        logger.info(f"Starting {duration_minutes}-minute attack simulation")
        self.status = "running"
        
        # Generate and execute attacks
        attacks = self.attack_generator.generate_batch(10)
        
        for attack in attacks:
            self.execute_attack(attack)
            time.sleep(random.uniform(0.5, 2.0))  # Stagger attacks
    
    def execute_attack(self, attack: SimpleAttackVector) -> Dict[str, Any]:
        """Execute a single attack"""
        logger.info(f"Executing {attack.attack_type} attack: {attack.attack_id[:8]}")
        
        # Simulate attack execution
        execution_time = random.uniform(1, 5)
        time.sleep(execution_time)
        
        # Determine success based on probability
        success = random.random() < attack.success_probability
        
        # Simulate detection probability (inverse of stealth level)
        detection_probability = 1.0 - attack.stealth_level
        detected = random.random() < detection_probability
        
        result = {
            "attack_id": attack.attack_id,
            "success": success,
            "detected": detected,
            "execution_time": execution_time,
            "target_hosts": attack.target_systems,
            "timestamp": datetime.now().isoformat()
        }
        
        self.completed_attacks.append(result)
        
        if detected:
            self._trigger_detection_event(attack, result)
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cyber range metrics"""
        total_attacks = len(self.completed_attacks)
        successful_attacks = sum(1 for a in self.completed_attacks if a["success"])
        detected_attacks = sum(1 for a in self.completed_attacks if a["detected"])
        
        detection_rate = detected_attacks / max(1, total_attacks)
        success_rate = successful_attacks / max(1, total_attacks)
        
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            "range_id": self.range_id,
            "status": self.status,
            "uptime_seconds": uptime,
            "total_attacks": total_attacks,
            "successful_attacks": successful_attacks,
            "detected_attacks": detected_attacks,
            "detection_rate": round(detection_rate, 3),
            "success_rate": round(success_rate, 3),
            "active_hosts": len(self.hosts),
            "networks": len(self.networks),
            "services": len(self.services)
        }
    
    def get_attack_summary(self) -> Dict[str, Any]:
        """Get attack summary statistics"""
        if not self.completed_attacks:
            return {"message": "No attacks executed yet"}
        
        attack_types = {}
        for attack in self.completed_attacks:
            # Extract attack type from active_attacks if available
            attack_type = "unknown"
            for active_attack in self.active_attacks.values():
                if hasattr(active_attack, 'attack_type'):
                    attack_type = active_attack.attack_type
                    break
            
            if attack_type not in attack_types:
                attack_types[attack_type] = {"count": 0, "success": 0, "detected": 0}
            
            attack_types[attack_type]["count"] += 1
            if attack["success"]:
                attack_types[attack_type]["success"] += 1
            if attack["detected"]:
                attack_types[attack_type]["detected"] += 1
        
        return {
            "attack_breakdown": attack_types,
            "latest_attacks": self.completed_attacks[-5:],  # Last 5 attacks
            "total_detections": len(self.detection_events)
        }
    
    def _create_demo_hosts(self) -> List[Dict[str, Any]]:
        """Create demo host infrastructure"""
        return [
            {"name": "web-server-01", "ip": "192.168.1.10", "os": "linux", "services": ["http", "https"]},
            {"name": "db-server-01", "ip": "192.168.1.20", "os": "linux", "services": ["mysql"]},
            {"name": "mail-server", "ip": "192.168.1.30", "os": "linux", "services": ["smtp", "imap"]},
            {"name": "workstation-01", "ip": "192.168.1.100", "os": "windows", "services": ["rdp"]},
            {"name": "workstation-02", "ip": "192.168.1.101", "os": "windows", "services": ["rdp"]},
            {"name": "firewall", "ip": "192.168.1.1", "os": "pfsense", "services": ["firewall"]},
        ]
    
    def _trigger_detection_event(self, attack: SimpleAttackVector, result: Dict[str, Any]) -> None:
        """Trigger a detection event"""
        detection_event = {
            "event_id": str(uuid.uuid4()),
            "attack_id": attack.attack_id,
            "attack_type": attack.attack_type,
            "severity": attack.severity,
            "timestamp": datetime.now().isoformat(),
            "detection_method": random.choice(["signature", "anomaly", "behavior"]),
            "confidence": round(random.uniform(0.6, 0.95), 2)
        }
        
        self.detection_events.append(detection_event)
        logger.info(f"🚨 Attack detected: {attack.attack_type} on {result['target_hosts']}")


class DemoAPI:
    """Simple REST API simulation for the demo system"""
    
    def __init__(self):
        self.ranges = {}
    
    def create_range(self, name: str = None) -> Dict[str, Any]:
        """Create a new cyber range"""
        cyber_range = SimpleCyberRange(name or f"demo-range-{len(self.ranges) + 1}")
        range_id = cyber_range.deploy()
        self.ranges[range_id] = cyber_range
        
        return {
            "range_id": range_id,
            "name": cyber_range.name,
            "status": cyber_range.status,
            "dashboard_url": cyber_range.dashboard_url
        }
    
    def get_range_info(self, range_id: str) -> Dict[str, Any]:
        """Get cyber range information"""
        if range_id not in self.ranges:
            return {"error": "Range not found"}
        
        cyber_range = self.ranges[range_id]
        return {
            "range_info": {
                "range_id": range_id,
                "name": cyber_range.name,
                "status": cyber_range.status,
                "hosts": len(cyber_range.hosts),
                "networks": cyber_range.networks
            },
            "metrics": cyber_range.get_metrics(),
            "attacks": cyber_range.get_attack_summary()
        }
    
    def generate_attacks(self, range_id: str, count: int = 10, attack_type: str = None) -> Dict[str, Any]:
        """Generate and execute attacks"""
        if range_id not in self.ranges:
            return {"error": "Range not found"}
        
        cyber_range = self.ranges[range_id]
        attacks = cyber_range.attack_generator.generate_batch(count, attack_type)
        
        # Execute attacks
        results = []
        for attack in attacks:
            result = cyber_range.execute_attack(attack)
            results.append(result)
        
        return {
            "generated_attacks": count,
            "attack_type": attack_type or "mixed",
            "results": results,
            "summary": cyber_range.get_attack_summary()
        }


def demo_basic_usage():
    """Demonstrate basic usage of the system"""
    print("=" * 60)
    print("🛡️  GAN-Cyber-Range-v2 Lightweight Demo")
    print("=" * 60)
    
    # Create demo API
    api = DemoAPI()
    
    print("\n1. Creating cyber range...")
    range_response = api.create_range("healthcare-demo")
    range_id = range_response["range_id"]
    print(f"   Range created: {range_id[:8]}...")
    print(f"   Dashboard: {range_response['dashboard_url']}")
    
    print("\n2. Generating synthetic attacks...")
    attack_response = api.generate_attacks(range_id, count=5, attack_type="malware")
    print(f"   Generated {attack_response['generated_attacks']} attacks")
    print(f"   Attack type: {attack_response['attack_type']}")
    
    print("\n3. Range metrics and summary...")
    info_response = api.get_range_info(range_id)
    metrics = info_response["metrics"]
    
    print(f"   Total attacks: {metrics['total_attacks']}")
    print(f"   Detection rate: {metrics['detection_rate']:.1%}")
    print(f"   Success rate: {metrics['success_rate']:.1%}")
    print(f"   Active hosts: {metrics['active_hosts']}")
    
    print("\n4. Recent attack details:")
    attacks_info = info_response["attacks"]
    if "latest_attacks" in attacks_info:
        for i, attack in enumerate(attacks_info["latest_attacks"][-3:], 1):
            status = "✅ Success" if attack["success"] else "❌ Failed"
            detection = "🚨 Detected" if attack["detected"] else "👻 Undetected"
            print(f"   Attack {i}: {status} | {detection}")
    
    print("\n5. Generating different attack types...")
    for attack_type in ["network", "web", "social_engineering"]:
        print(f"   Generating {attack_type} attacks...")
        result = api.generate_attacks(range_id, count=3, attack_type=attack_type)
        
    # Final metrics
    final_info = api.get_range_info(range_id)
    final_metrics = final_info["metrics"]
    
    print(f"\n📊 Final Metrics:")
    print(f"   Total attacks executed: {final_metrics['total_attacks']}")
    print(f"   Overall detection rate: {final_metrics['detection_rate']:.1%}")
    print(f"   Range uptime: {final_metrics['uptime_seconds']:.1f} seconds")
    
    print("\n✅ Demo completed successfully!")
    print("=" * 60)
    
    return api, range_id


if __name__ == "__main__":
    demo_basic_usage()