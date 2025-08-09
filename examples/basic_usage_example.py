#!/usr/bin/env python3
"""
GAN-Cyber-Range-v2 Basic Usage Example

This example demonstrates the core functionality of the cyber range platform,
including GAN-based attack generation, network simulation, and security validation.

For educational and research purposes only.
"""

import logging
from pathlib import Path

from gan_cyber_range.core.attack_gan import AttackGAN, AttackVector
from gan_cyber_range.core.cyber_range import CyberRange
from gan_cyber_range.core.network_sim import NetworkTopology
from gan_cyber_range.red_team.llm_adversary import RedTeamLLM
from gan_cyber_range.utils.security import EthicalFramework
from gan_cyber_range.utils.monitoring import MetricsCollector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Demonstrate basic cyber range capabilities"""
    
    logger.info("🚀 Starting GAN-Cyber-Range-v2 Basic Demo")
    
    # Step 1: Validate ethical use
    logger.info("Step 1: Validating ethical use case")
    ethics = EthicalFramework()
    use_case = {
        'purpose': 'security_training',
        'target_type': 'sandbox',
        'user_role': 'researcher'
    }
    
    if not ethics.validate_use_case(use_case):
        logger.error("❌ Use case validation failed. Exiting.")
        return
    
    logger.info("✅ Ethical use case validated")
    
    # Step 2: Initialize monitoring
    logger.info("Step 2: Initializing monitoring system")
    metrics = MetricsCollector()
    logger.info("✅ Monitoring system ready")
    
    # Step 3: Create network topology
    logger.info("Step 3: Generating network topology")
    topology = NetworkTopology.generate(
        template="enterprise",
        subnets=["dmz", "internal", "management"],
        hosts_per_subnet={"dmz": 3, "internal": 8, "management": 2}
    )
    logger.info(f"✅ Network topology created with {len(topology.hosts)} hosts")
    
    # Step 4: Initialize cyber range
    logger.info("Step 4: Setting up cyber range environment")
    cyber_range = CyberRange(
        topology=topology,
        hypervisor="docker",
        network_emulation="bridge"
    )
    logger.info(f"✅ Cyber range initialized (ID: {cyber_range.range_id})")
    
    # Step 5: Initialize GAN for attack generation
    logger.info("Step 5: Setting up GAN-based attack generation")
    attack_gan = AttackGAN(
        architecture="wasserstein",
        attack_types=["malware", "network", "web"],
        noise_dim=100
    )
    logger.info("✅ AttackGAN initialized and ready for synthetic attack generation")
    
    # Step 6: Generate synthetic attacks
    logger.info("Step 6: Generating synthetic attack vectors")
    import torch
    
    # Generate 5 synthetic attack patterns
    noise = torch.randn(5, 100)
    with torch.no_grad():
        synthetic_patterns = attack_gan.generator(noise)
    
    logger.info(f"✅ Generated {synthetic_patterns.shape[0]} synthetic attack patterns")
    
    # Create structured attack vectors
    attack_vectors = []
    for i in range(3):
        vector = AttackVector(
            attack_type="malware",
            payload=f"synthetic_payload_{i}",
            techniques=[f"T1059.00{i+1}", f"T1055.00{i+1}"],
            severity=7.0 + i * 0.5,
            stealth_level=0.7 + i * 0.1,
            target_systems=["windows", "linux"]
        )
        attack_vectors.append(vector)
    
    logger.info(f"✅ Created {len(attack_vectors)} structured attack vectors")
    
    # Step 7: Initialize Red Team LLM
    logger.info("Step 7: Setting up LLM-based red team adversary")
    red_team = RedTeamLLM(
        model="gpt-3.5-turbo",
        creativity=0.8,
        risk_tolerance=0.6
    )
    
    # Generate attack plan
    target_info = {
        'ip_range': '192.168.1.0/24',
        'os_types': ['windows', 'linux'],
        'services': ['http', 'ssh', 'rdp'],
        'security_tools': ['antivirus', 'firewall']
    }
    
    attack_plan = red_team.generate_attack_plan(target_info, 'lateral_movement')
    logger.info(f"✅ Generated attack plan with {len(attack_plan['phases'])} phases")
    
    # Step 8: Record metrics
    logger.info("Step 8: Recording performance metrics")
    metrics.record_metric('attacks_generated', 'synthetic', len(attack_vectors))
    metrics.record_metric('network_hosts', 'topology', len(topology.hosts))
    metrics.record_metric('plan_phases', 'red_team', len(attack_plan['phases']))
    
    logger.info("✅ Metrics recorded successfully")
    
    # Step 9: Display summary
    logger.info("Step 9: Demo Summary")
    logger.info("=" * 50)
    logger.info(f"🎯 Cyber Range ID: {cyber_range.range_id}")
    logger.info(f"🌐 Network Hosts: {len(topology.hosts)}")
    logger.info(f"⚔️  Attack Vectors: {len(attack_vectors)}")
    logger.info(f"🧠 Attack Plan Phases: {len(attack_plan['phases'])}")
    logger.info(f"📊 Stealth Score: {attack_plan.get('stealth_score', 'N/A')}")
    logger.info(f"📈 Success Probability: {attack_plan.get('success_probability', 'N/A')}")
    logger.info("=" * 50)
    
    logger.info("🎉 GAN-Cyber-Range-v2 basic demo completed successfully!")
    logger.info("💡 This platform is ready for:")
    logger.info("   - Advanced cybersecurity training")
    logger.info("   - AI-driven red team exercises")
    logger.info("   - Defensive capability assessment")
    logger.info("   - Academic security research")

if __name__ == "__main__":
    main()