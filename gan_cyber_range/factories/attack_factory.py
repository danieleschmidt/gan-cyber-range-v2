"""
Attack Factory for creating and configuring attack components.
"""

import logging
from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass
from pathlib import Path

from ..core.attack_gan import AttackGAN, Generator, Discriminator
from ..core.attack_engine import AttackSimulator, AttackEngine
from ..red_team.llm_adversary import RedTeamLLM
from ..utils.security import SecurityManager
from ..utils.error_handling import CyberRangeError

logger = logging.getLogger(__name__)


@dataclass
class AttackConfig:
    """Configuration for attack components"""
    gan_architecture: str = "wasserstein"
    noise_dim: int = 100
    output_dim: int = 512
    attack_types: List[str] = None
    llm_model: str = "gpt-3.5-turbo"
    max_attack_complexity: float = 0.8
    enable_mutation: bool = True
    differential_privacy: bool = True
    privacy_budget: float = 10.0
    
    def __post_init__(self):
        if self.attack_types is None:
            self.attack_types = ["network", "web", "malware", "social_engineering"]


class AttackFactory:
    """Factory for creating and configuring attack-related components"""
    
    def __init__(self, security_manager: Optional[SecurityManager] = None):
        self.security_manager = security_manager or SecurityManager()
        self._gan_cache: Dict[str, AttackGAN] = {}
        self._llm_cache: Dict[str, RedTeamLLM] = {}
        
    def create_attack_gan(self, config: Optional[AttackConfig] = None) -> AttackGAN:
        """Create a configured AttackGAN instance"""
        config = config or AttackConfig()
        
        # Validate security requirements
        if not self.security_manager.validate_use_case("research", "attack_generation"):
            raise CyberRangeError("Attack GAN creation not authorized for current use case")
            
        # Check cache for existing instance
        cache_key = f"{config.gan_architecture}_{config.noise_dim}_{config.output_dim}"
        if cache_key in self._gan_cache:
            logger.info(f"Returning cached AttackGAN instance: {cache_key}")
            return self._gan_cache[cache_key]
            
        # Create new instance
        logger.info(f"Creating new AttackGAN with architecture: {config.gan_architecture}")
        
        gan = AttackGAN(
            architecture=config.gan_architecture,
            noise_dim=config.noise_dim,
            output_dim=config.output_dim,
            attack_types=config.attack_types,
            differential_privacy=config.differential_privacy,
            privacy_budget=config.privacy_budget
        )
        
        # Apply security wrapping
        gan = self._apply_security_wrapper(gan)
        
        # Cache for reuse
        self._gan_cache[cache_key] = gan
        
        return gan
        
    def create_red_team_llm(self, config: Optional[AttackConfig] = None) -> RedTeamLLM:
        """Create a configured RedTeamLLM instance"""
        config = config or AttackConfig()
        
        # Validate security requirements
        if not self.security_manager.validate_use_case("research", "red_team_simulation"):
            raise CyberRangeError("Red Team LLM creation not authorized")
            
        # Check cache
        cache_key = f"llm_{config.llm_model}_{config.max_attack_complexity}"
        if cache_key in self._llm_cache:
            logger.info(f"Returning cached RedTeamLLM instance: {cache_key}")
            return self._llm_cache[cache_key]
            
        logger.info(f"Creating RedTeamLLM with model: {config.llm_model}")
        
        llm = RedTeamLLM(
            model=config.llm_model,
            creativity=min(config.max_attack_complexity, 0.8),
            enable_mutation=config.enable_mutation
        )
        
        # Apply security constraints
        llm = self._apply_llm_constraints(llm, config)
        
        # Cache for reuse
        self._llm_cache[cache_key] = llm
        
        return llm
        
    def create_attack_simulator(self, 
                              gan: Optional[AttackGAN] = None,
                              llm: Optional[RedTeamLLM] = None,
                              config: Optional[AttackConfig] = None) -> AttackSimulator:
        """Create a complete attack simulation environment"""
        config = config or AttackConfig()
        
        # Create components if not provided
        if gan is None:
            gan = self.create_attack_gan(config)
        if llm is None:
            llm = self.create_red_team_llm(config)
            
        # Create attack engine
        engine = AttackEngine()
        
        # Create simulator
        simulator = AttackSimulator(
            attack_gan=gan,
            red_team_llm=llm,
            attack_engine=engine
        )
        
        # Apply security monitoring
        simulator = self._apply_simulation_monitoring(simulator)
        
        return simulator
        
    def create_training_scenario(self,
                               scenario_type: str,
                               difficulty: str = "medium",
                               custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a training scenario configuration"""
        
        base_scenarios = {
            "apt_simulation": {
                "attack_types": ["reconnaissance", "lateral_movement", "data_exfiltration"],
                "duration": "2_hours",
                "complexity": 0.7,
                "stealth_requirements": True
            },
            "ransomware_incident": {
                "attack_types": ["malware", "encryption", "payment_demand"],
                "duration": "45_minutes", 
                "complexity": 0.6,
                "time_pressure": True
            },
            "insider_threat": {
                "attack_types": ["privilege_escalation", "data_access", "exfiltration"],
                "duration": "1_hour",
                "complexity": 0.5,
                "social_engineering": True
            },
            "zero_day_exploit": {
                "attack_types": ["vulnerability_research", "exploit_development", "deployment"],
                "duration": "3_hours",
                "complexity": 0.9,
                "advanced_techniques": True
            }
        }
        
        if scenario_type not in base_scenarios:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
            
        scenario = base_scenarios[scenario_type].copy()
        
        # Adjust for difficulty
        difficulty_multipliers = {"easy": 0.7, "medium": 1.0, "hard": 1.3, "expert": 1.6}
        multiplier = difficulty_multipliers.get(difficulty, 1.0)
        scenario["complexity"] *= multiplier
        
        # Apply custom parameters
        if custom_params:
            scenario.update(custom_params)
            
        return scenario
        
    def _apply_security_wrapper(self, gan: AttackGAN) -> AttackGAN:
        """Apply security constraints to AttackGAN"""
        # Add monitoring hooks
        original_generate = gan.generate
        
        def monitored_generate(*args, **kwargs):
            # Log generation request
            self.security_manager.log_activity("attack_generation", {
                "timestamp": logger.handlers[0].format(logging.LogRecord("", 0, "", 0, "", (), None)).split()[0],
                "args_count": len(args),
                "kwargs": list(kwargs.keys())
            })
            
            # Execute with monitoring
            result = original_generate(*args, **kwargs)
            
            # Validate generated content
            self.security_manager.validate_generated_content(result)
            
            return result
            
        gan.generate = monitored_generate
        return gan
        
    def _apply_llm_constraints(self, llm: RedTeamLLM, config: AttackConfig) -> RedTeamLLM:
        """Apply security constraints to RedTeamLLM"""
        # Set ethical boundaries
        llm.set_ethical_boundaries([
            "no_real_world_targeting",
            "research_purposes_only", 
            "defensive_training_focus",
            "no_illegal_content"
        ])
        
        # Limit complexity if needed
        if hasattr(llm, 'creativity'):
            llm.creativity = min(llm.creativity, config.max_attack_complexity)
            
        return llm
        
    def _apply_simulation_monitoring(self, simulator: AttackSimulator) -> AttackSimulator:
        """Apply monitoring to attack simulation"""
        # Add telemetry
        original_execute = simulator.execute_campaign
        
        def monitored_execute(*args, **kwargs):
            self.security_manager.start_simulation_monitoring()
            try:
                result = original_execute(*args, **kwargs)
                self.security_manager.log_simulation_success()
                return result
            except Exception as e:
                self.security_manager.log_simulation_error(str(e))
                raise
            finally:
                self.security_manager.stop_simulation_monitoring()
                
        simulator.execute_campaign = monitored_execute
        return simulator
        
    def clear_cache(self):
        """Clear all cached instances"""
        self._gan_cache.clear()
        self._llm_cache.clear()
        logger.info("Attack factory cache cleared")
        
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "gan_cache_size": len(self._gan_cache),
            "llm_cache_size": len(self._llm_cache),
            "total_cached_objects": len(self._gan_cache) + len(self._llm_cache)
        }