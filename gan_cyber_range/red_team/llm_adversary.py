"""
LLM-based red team adversary for adaptive attack scenario generation.

This module implements an AI-driven red team that uses large language models
to generate realistic, adaptive attack scenarios and tactics.
"""

import logging
import json
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import random

logger = logging.getLogger(__name__)


@dataclass
class AdversaryProfile:
    """Profile of an AI adversary with specific characteristics"""
    name: str
    sophistication_level: str  # "low", "medium", "high", "nation_state"
    motivation: str  # "financial", "espionage", "sabotage", "hacktivism"
    capabilities: List[str]
    risk_tolerance: float  # 0.0 to 1.0
    creativity: float  # 0.0 to 1.0
    stealth_preference: float  # 0.0 to 1.0
    target_preferences: List[str] = field(default_factory=list)
    ttp_history: List[str] = field(default_factory=list)


@dataclass
class AttackObjective:
    """Defines an attack objective for the red team"""
    primary_goal: str
    success_criteria: List[str]
    time_constraint: Optional[str] = None
    stealth_requirement: bool = False
    data_targets: List[str] = field(default_factory=list)
    infrastructure_targets: List[str] = field(default_factory=list)


class RedTeamLLM:
    """LLM-based red team adversary"""
    
    def __init__(
        self,
        model: str = "gpt-4",
        creativity: float = 0.8,
        risk_tolerance: float = 0.6,
        objective: str = "data_exfiltration"
    ):
        self.model = model
        self.creativity = creativity
        self.risk_tolerance = risk_tolerance
        self.objective = objective
        
        # Initialize adversary profile
        self.profile = self._create_adversary_profile()
        
        # Attack knowledge base
        self.technique_knowledge = self._load_technique_knowledge()
        self.target_intelligence = {}
        self.attack_history = []
        
        logger.info(f"Initialized RedTeamLLM with model: {model}")
    
    def generate_attack_plan(
        self,
        target_profile: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive attack plan for the target"""
        
        logger.info(f"Generating attack plan for target: {target_profile.get('name', 'unknown')}")
        
        # Analyze target
        target_analysis = self._analyze_target(target_profile)
        
        # Generate attack phases
        attack_phases = self._generate_attack_phases(target_analysis, constraints)
        
        # Create attack plan
        attack_plan = {
            'plan_id': self._generate_plan_id(),
            'target_profile': target_profile,
            'target_analysis': target_analysis,
            'adversary_profile': self.profile.name,
            'phases': attack_phases,
            'estimated_duration': self._estimate_duration(attack_phases),
            'success_probability': self._estimate_success_probability(target_analysis, attack_phases),
            'stealth_score': self._calculate_stealth_score(attack_phases),
            'created_at': datetime.now().isoformat()
        }
        
        return attack_plan
    
    def adapt_tactics(
        self,
        current_plan: Dict[str, Any],
        detection_events: List[Dict[str, Any]],
        blue_team_response: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Adapt attack tactics based on blue team responses"""
        
        logger.info("Adapting tactics based on defensive responses")
        
        # Analyze what was detected
        detection_analysis = self._analyze_detections(detection_events)
        
        # Analyze blue team response
        response_analysis = self._analyze_blue_team_response(blue_team_response)
        
        # Generate adaptive measures
        adaptations = self._generate_adaptations(
            current_plan, detection_analysis, response_analysis
        )
        
        # Update attack plan
        adapted_plan = self._apply_adaptations(current_plan, adaptations)
        
        return adapted_plan
    
    def generate_social_engineering_campaign(
        self,
        target_employees: List[Dict[str, Any]],
        campaign_type: str = "spear_phishing"
    ) -> Dict[str, Any]:
        """Generate social engineering campaign"""
        
        logger.info(f"Generating {campaign_type} campaign for {len(target_employees)} targets")
        
        # Analyze target employees
        employee_profiles = [self._analyze_employee(emp) for emp in target_employees]
        
        # Generate campaign strategy
        campaign_strategy = self._generate_se_strategy(employee_profiles, campaign_type)
        
        # Create campaign materials
        campaign_materials = self._generate_se_materials(campaign_strategy)
        
        campaign = {
            'campaign_id': self._generate_plan_id(),
            'type': campaign_type,
            'target_employees': employee_profiles,
            'strategy': campaign_strategy,
            'materials': campaign_materials,
            'timeline': self._generate_se_timeline(campaign_strategy),
            'success_metrics': self._define_se_success_criteria(campaign_type)
        }
        
        return campaign
    
    def generate_payload_variants(
        self,
        base_payload: Dict[str, Any],
        evasion_requirements: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate payload variants for evasion"""
        
        logger.info(f"Generating payload variants for evasion: {evasion_requirements}")
        
        variants = []
        
        for requirement in evasion_requirements:
            variant = self._create_payload_variant(base_payload, requirement)
            variants.append(variant)
        
        # Generate additional creative variants
        creative_variants = self._generate_creative_variants(base_payload)
        variants.extend(creative_variants)
        
        return variants
    
    def simulate_threat_intelligence(self, actor_name: str) -> Dict[str, Any]:
        """Simulate threat intelligence report for an APT actor"""
        
        logger.info(f"Simulating threat intelligence for: {actor_name}")
        
        # Generate actor profile
        actor_profile = self._generate_apt_profile(actor_name)
        
        # Generate campaign history
        campaign_history = self._generate_campaign_history(actor_profile)
        
        # Generate IOCs
        iocs = self._generate_iocs(actor_profile, campaign_history)
        
        # Generate mitigation recommendations
        mitigations = self._generate_mitigations(actor_profile)
        
        threat_intel = {
            'actor_name': actor_name,
            'profile': actor_profile,
            'campaign_history': campaign_history,
            'iocs': iocs,
            'ttps': self._extract_ttps(campaign_history),
            'mitigations': mitigations,
            'confidence_level': 'medium',
            'report_date': datetime.now().isoformat()
        }
        
        return threat_intel
    
    def _create_adversary_profile(self) -> AdversaryProfile:
        """Create adversary profile based on initialization parameters"""
        
        sophistication_mapping = {
            (0.0, 0.3): "low",
            (0.3, 0.6): "medium", 
            (0.6, 0.8): "high",
            (0.8, 1.0): "nation_state"
        }
        
        sophistication = "medium"
        for (low, high), level in sophistication_mapping.items():
            if low <= self.risk_tolerance < high:
                sophistication = level
                break
        
        capabilities = self._determine_capabilities(sophistication)
        
        return AdversaryProfile(
            name=f"RedTeam-LLM-{random.randint(1000, 9999)}",
            sophistication_level=sophistication,
            motivation=self.objective,
            capabilities=capabilities,
            risk_tolerance=self.risk_tolerance,
            creativity=self.creativity,
            stealth_preference=1.0 - self.risk_tolerance
        )
    
    def _determine_capabilities(self, sophistication: str) -> List[str]:
        """Determine adversary capabilities based on sophistication level"""
        
        capability_sets = {
            "low": ["basic_malware", "social_engineering", "public_exploits"],
            "medium": ["custom_malware", "zero_days", "supply_chain", "advanced_persistence"],
            "high": ["custom_frameworks", "infrastructure", "insider_threats", "industrial_espionage"],
            "nation_state": ["zero_day_stockpile", "intelligence_operations", "critical_infrastructure", "attribution_manipulation"]
        }
        
        base_capabilities = capability_sets.get(sophistication, capability_sets["medium"])
        
        # Add random additional capabilities based on creativity
        all_capabilities = [cap for caps in capability_sets.values() for cap in caps]
        additional_count = int(self.creativity * 3)
        additional_caps = random.sample(
            [cap for cap in all_capabilities if cap not in base_capabilities],
            min(additional_count, len(all_capabilities) - len(base_capabilities))
        )
        
        return base_capabilities + additional_caps
    
    def _load_technique_knowledge(self) -> Dict[str, Any]:
        """Load MITRE ATT&CK technique knowledge base"""
        
        # Simplified technique knowledge - in real implementation would load from MITRE data
        techniques = {
            "T1078": {
                "name": "Valid Accounts",
                "tactic": ["Defense Evasion", "Persistence", "Privilege Escalation", "Initial Access"],
                "difficulty": "low",
                "detection_difficulty": "medium",
                "prerequisites": ["credential_access"]
            },
            "T1190": {
                "name": "Exploit Public-Facing Application", 
                "tactic": ["Initial Access"],
                "difficulty": "medium",
                "detection_difficulty": "low",
                "prerequisites": ["reconnaissance"]
            },
            "T1059": {
                "name": "Command and Scripting Interpreter",
                "tactic": ["Execution"],
                "difficulty": "low",
                "detection_difficulty": "medium",
                "prerequisites": ["initial_access"]
            },
            "T1021": {
                "name": "Remote Services",
                "tactic": ["Lateral Movement"],
                "difficulty": "medium",
                "detection_difficulty": "medium", 
                "prerequisites": ["valid_accounts"]
            }
        }
        
        return techniques
    
    def _analyze_target(self, target_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze target profile to identify attack vectors"""
        
        analysis = {
            'organization_type': target_profile.get('industry', 'unknown'),
            'size': target_profile.get('size', 'medium'),
            'security_maturity': target_profile.get('security_maturity', 'medium'),
            'crown_jewels': target_profile.get('crown_jewels', []),
            'attack_surface': self._assess_attack_surface(target_profile),
            'security_controls': self._assess_security_controls(target_profile),
            'high_value_targets': self._identify_high_value_targets(target_profile),
            'likely_vulnerabilities': self._predict_vulnerabilities(target_profile)
        }
        
        return analysis
    
    def _assess_attack_surface(self, target_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the attack surface of the target"""
        
        surface = {
            'web_applications': target_profile.get('public_services', []),
            'email_security': 'medium',  # Default assessment
            'remote_access': target_profile.get('remote_access', True),
            'cloud_services': target_profile.get('cloud_adoption', 'medium'),
            'mobile_devices': target_profile.get('byod_policy', False),
            'iot_devices': target_profile.get('iot_presence', 'low')
        }
        
        return surface
    
    def _assess_security_controls(self, target_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Assess security controls in place"""
        
        maturity = target_profile.get('security_maturity', 'medium')
        
        control_levels = {
            'low': {'firewall': 'basic', 'ids': 'none', 'siem': 'none', 'edr': 'none'},
            'medium': {'firewall': 'ngfw', 'ids': 'signature', 'siem': 'basic', 'edr': 'basic'},
            'high': {'firewall': 'ngfw', 'ids': 'behavior', 'siem': 'advanced', 'edr': 'advanced'}
        }
        
        return control_levels.get(maturity, control_levels['medium'])
    
    def _identify_high_value_targets(self, target_profile: Dict[str, Any]) -> List[str]:
        """Identify high-value targets within the organization"""
        
        crown_jewels = target_profile.get('crown_jewels', [])
        
        # Add standard high-value targets
        standard_hvts = ['domain_controllers', 'backup_servers', 'database_servers', 'email_servers']
        
        return list(set(crown_jewels + standard_hvts))
    
    def _predict_vulnerabilities(self, target_profile: Dict[str, Any]) -> List[str]:
        """Predict likely vulnerabilities based on target profile"""
        
        vulnerabilities = []
        
        # Industry-specific vulnerabilities
        industry = target_profile.get('industry', 'generic')
        industry_vulns = {
            'healthcare': ['medical_devices', 'legacy_systems', 'hipaa_compliance_gaps'],
            'finance': ['trading_systems', 'regulatory_gaps', 'high_value_data'],
            'manufacturing': ['scada_systems', 'iot_devices', 'operational_technology'],
            'education': ['student_data', 'research_data', 'limited_budgets']
        }
        
        vulnerabilities.extend(industry_vulns.get(industry, ['generic_web_apps', 'email_security']))
        
        # Size-based vulnerabilities
        size = target_profile.get('size', 'medium')
        if size == 'small':
            vulnerabilities.extend(['limited_security_staff', 'basic_controls', 'patch_management'])
        elif size == 'large':
            vulnerabilities.extend(['complex_infrastructure', 'shadow_it', 'insider_threats'])
        
        return vulnerabilities
    
    def _generate_attack_phases(self, target_analysis: Dict[str, Any], constraints: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate attack phases based on target analysis"""
        
        phases = []
        
        # Phase 1: Reconnaissance
        recon_phase = {
            'phase': 'reconnaissance',
            'duration': 'days',
            'techniques': ['T1595', 'T1596', 'T1598'],  # Active Scanning, Search Victim-Owned Websites, Phishing for Information
            'objectives': ['identify_infrastructure', 'gather_employee_info', 'map_technologies'],
            'stealth_level': 'high'
        }
        phases.append(recon_phase)
        
        # Phase 2: Initial Access
        initial_access_techniques = self._select_initial_access_techniques(target_analysis)
        access_phase = {
            'phase': 'initial_access',
            'duration': 'hours',
            'techniques': initial_access_techniques,
            'objectives': ['establish_foothold', 'deploy_backdoor'],
            'stealth_level': 'high'
        }
        phases.append(access_phase)
        
        # Phase 3: Persistence & Privilege Escalation
        persistence_phase = {
            'phase': 'persistence',
            'duration': 'hours',
            'techniques': ['T1078', 'T1547', 'T1053'],  # Valid Accounts, Boot/Logon Autostart, Scheduled Task
            'objectives': ['maintain_access', 'escalate_privileges'],
            'stealth_level': 'medium'
        }
        phases.append(persistence_phase)
        
        # Phase 4: Discovery & Lateral Movement
        movement_phase = {
            'phase': 'lateral_movement',
            'duration': 'days',
            'techniques': ['T1083', 'T1021', 'T1046'],  # File Discovery, Remote Services, Network Service Scanning
            'objectives': ['map_network', 'identify_targets', 'move_laterally'],
            'stealth_level': 'medium'
        }
        phases.append(movement_phase)
        
        # Phase 5: Objective Completion
        objective_techniques = self._select_objective_techniques(target_analysis)
        objective_phase = {
            'phase': 'objectives',
            'duration': 'hours',
            'techniques': objective_techniques,
            'objectives': self._define_final_objectives(target_analysis),
            'stealth_level': 'low'
        }
        phases.append(objective_phase)
        
        return phases
    
    def _select_initial_access_techniques(self, target_analysis: Dict[str, Any]) -> List[str]:
        """Select appropriate initial access techniques"""
        
        techniques = []
        
        # Check attack surface
        surface = target_analysis.get('attack_surface', {})
        
        if surface.get('web_applications'):
            techniques.append('T1190')  # Exploit Public-Facing Application
            
        if surface.get('email_security') in ['low', 'medium']:
            techniques.append('T1566')  # Phishing
            
        if surface.get('remote_access'):
            techniques.append('T1078')  # Valid Accounts
            
        # Always include supply chain as option for sophisticated adversaries
        if self.profile.sophistication_level in ['high', 'nation_state']:
            techniques.append('T1195')  # Supply Chain Compromise
        
        return techniques[:3]  # Limit to top 3 techniques
    
    def _select_objective_techniques(self, target_analysis: Dict[str, Any]) -> List[str]:
        """Select techniques for final objectives"""
        
        techniques = []
        
        # Data exfiltration
        if 'data_exfiltration' in self.objective:
            techniques.extend(['T1041', 'T1048'])  # Exfiltration Over C2, Exfiltration Over Alternative Protocol
        
        # Ransomware/destruction
        if 'sabotage' in self.objective:
            techniques.extend(['T1486', 'T1490'])  # Data Encrypted for Impact, Inhibit System Recovery
        
        # Credential harvesting
        if 'credential' in self.objective:
            techniques.extend(['T1003', 'T1555'])  # OS Credential Dumping, Credentials from Password Stores
        
        return techniques
    
    def _define_final_objectives(self, target_analysis: Dict[str, Any]) -> List[str]:
        """Define final attack objectives"""
        
        objectives = []
        
        crown_jewels = target_analysis.get('crown_jewels', [])
        
        if 'data_exfiltration' in self.objective:
            objectives.extend([f'exfiltrate_{target}' for target in crown_jewels])
        
        if 'credential' in self.objective:
            objectives.append('harvest_credentials')
        
        if 'sabotage' in self.objective:
            objectives.append('disrupt_operations')
        
        return objectives or ['establish_persistence']
    
    def _estimate_duration(self, attack_phases: List[Dict[str, Any]]) -> str:
        """Estimate total attack duration"""
        
        phase_durations = {'hours': 1, 'days': 24, 'weeks': 168}
        
        total_hours = sum(phase_durations.get(phase.get('duration', 'hours'), 1) for phase in attack_phases)
        
        if total_hours < 24:
            return f"{total_hours} hours"
        elif total_hours < 168:
            return f"{total_hours // 24} days"
        else:
            return f"{total_hours // 168} weeks"
    
    def _estimate_success_probability(self, target_analysis: Dict[str, Any], attack_phases: List[Dict[str, Any]]) -> float:
        """Estimate overall attack success probability"""
        
        # Base probability based on adversary sophistication
        base_prob = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'nation_state': 0.9
        }.get(self.profile.sophistication_level, 0.6)
        
        # Adjust based on target security maturity
        security_modifier = {
            'low': 1.3,
            'medium': 1.0,
            'high': 0.7
        }.get(target_analysis.get('security_maturity', 'medium'), 1.0)
        
        # Adjust based on attack complexity
        complexity_modifier = max(0.5, 1.0 - (len(attack_phases) * 0.1))
        
        final_prob = min(1.0, base_prob * security_modifier * complexity_modifier)
        return round(final_prob, 2)
    
    def _calculate_stealth_score(self, attack_phases: List[Dict[str, Any]]) -> float:
        """Calculate overall stealth score of the attack plan"""
        
        stealth_levels = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
        
        total_stealth = sum(stealth_levels.get(phase.get('stealth_level', 'medium'), 0.6) for phase in attack_phases)
        average_stealth = total_stealth / len(attack_phases) if attack_phases else 0.6
        
        return round(average_stealth, 2)
    
    def _generate_plan_id(self) -> str:
        """Generate unique plan ID"""
        return f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(100, 999)}"
    
    def _analyze_detections(self, detection_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze what was detected by blue team"""
        
        analysis = {
            'detected_techniques': [],
            'detection_methods': [],
            'timing_patterns': [],
            'defensive_gaps': []
        }
        
        for event in detection_events:
            technique = event.get('technique_id')
            if technique:
                analysis['detected_techniques'].append(technique)
            
            detection_type = event.get('detection_type')
            if detection_type:
                analysis['detection_methods'].append(detection_type)
        
        # Identify gaps (techniques that weren't detected)
        all_techniques = set(self.technique_knowledge.keys())
        detected = set(analysis['detected_techniques'])
        analysis['defensive_gaps'] = list(all_techniques - detected)
        
        return analysis
    
    def _analyze_blue_team_response(self, blue_team_response: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze blue team response capabilities"""
        
        if not blue_team_response:
            return {'response_time': 'unknown', 'effectiveness': 'unknown', 'coverage': 'unknown'}
        
        analysis = {
            'response_time': blue_team_response.get('response_time', 'unknown'),
            'effectiveness': blue_team_response.get('containment_success', False),
            'tools_used': blue_team_response.get('tools', []),
            'procedures_followed': blue_team_response.get('procedures', [])
        }
        
        return analysis
    
    def _generate_adaptations(
        self,
        current_plan: Dict[str, Any],
        detection_analysis: Dict[str, Any],
        response_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate adaptive measures based on blue team activity"""
        
        adaptations = {
            'technique_substitutions': {},
            'timing_modifications': {},
            'stealth_enhancements': [],
            'new_vectors': []
        }
        
        # Substitute detected techniques
        for detected_technique in detection_analysis['detected_techniques']:
            alternatives = self._find_alternative_techniques(detected_technique)
            if alternatives:
                adaptations['technique_substitutions'][detected_technique] = alternatives[0]
        
        # Exploit defensive gaps
        for gap_technique in detection_analysis['defensive_gaps']:
            adaptations['new_vectors'].append(gap_technique)
        
        # Enhance stealth if quick response detected
        if response_analysis.get('response_time') == 'fast':
            adaptations['stealth_enhancements'].extend(['delay_execution', 'living_off_land', 'encryption'])
        
        return adaptations
    
    def _find_alternative_techniques(self, detected_technique: str) -> List[str]:
        """Find alternative techniques that achieve similar objectives"""
        
        alternatives_map = {
            'T1190': ['T1566', 'T1078'],  # If web exploit detected, try phishing or valid accounts
            'T1566': ['T1195', 'T1078'],  # If phishing detected, try supply chain or valid accounts
            'T1078': ['T1190', 'T1021'],  # If valid accounts detected, try exploits or remote services
        }
        
        return alternatives_map.get(detected_technique, [])
    
    def _apply_adaptations(self, current_plan: Dict[str, Any], adaptations: Dict[str, Any]) -> Dict[str, Any]:
        """Apply adaptations to the current attack plan"""
        
        adapted_plan = current_plan.copy()
        
        # Apply technique substitutions
        for phase in adapted_plan.get('phases', []):
            techniques = phase.get('techniques', [])
            for i, technique in enumerate(techniques):
                if technique in adaptations['technique_substitutions']:
                    techniques[i] = adaptations['technique_substitutions'][technique]
        
        # Add new attack vectors
        if adaptations['new_vectors']:
            new_phase = {
                'phase': 'adaptive_exploitation',
                'duration': 'hours',
                'techniques': adaptations['new_vectors'][:3],
                'objectives': ['exploit_defensive_gaps'],
                'stealth_level': 'high'
            }
            adapted_plan['phases'].append(new_phase)
        
        # Enhance stealth
        if adaptations['stealth_enhancements']:
            for phase in adapted_plan.get('phases', []):
                if phase.get('stealth_level') != 'high':
                    phase['stealth_level'] = 'high'
        
        adapted_plan['adaptation_timestamp'] = datetime.now().isoformat()
        adapted_plan['adaptations_applied'] = adaptations
        
        return adapted_plan
    
    def _analyze_employee(self, employee: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze employee for social engineering targeting"""
        
        profile = {
            'name': employee.get('name', 'Unknown'),
            'role': employee.get('role', 'Employee'),
            'department': employee.get('department', 'Unknown'),
            'seniority': employee.get('seniority', 'junior'),
            'access_level': employee.get('access_level', 'standard'),
            'social_media_presence': employee.get('social_media', 'medium'),
            'susceptibility_score': self._calculate_susceptibility(employee)
        }
        
        return profile
    
    def _calculate_susceptibility(self, employee: Dict[str, Any]) -> float:
        """Calculate employee susceptibility to social engineering"""
        
        base_score = 0.5
        
        # Role-based adjustments
        role_modifiers = {
            'executive': 0.3,  # Higher value target but more security aware
            'it_admin': 0.2,   # Security aware
            'hr': 0.7,         # Often target of business email compromise
            'finance': 0.8,    # High value target
            'intern': 0.9      # Less security awareness
        }
        
        role = employee.get('role', '').lower()
        for role_key, modifier in role_modifiers.items():
            if role_key in role:
                base_score = modifier
                break
        
        # Seniority adjustments
        seniority_modifiers = {
            'executive': -0.1,
            'senior': -0.05,
            'junior': 0.1,
            'intern': 0.2
        }
        
        seniority = employee.get('seniority', 'junior')
        base_score += seniority_modifiers.get(seniority, 0)
        
        return max(0.1, min(1.0, base_score))
    
    def _generate_se_strategy(self, employee_profiles: List[Dict[str, Any]], campaign_type: str) -> Dict[str, Any]:
        """Generate social engineering strategy"""
        
        strategy = {
            'primary_vector': campaign_type,
            'target_selection': self._select_se_targets(employee_profiles),
            'pretext': self._generate_pretext(employee_profiles, campaign_type),
            'timing': self._determine_se_timing(),
            'success_criteria': self._define_se_success_criteria(campaign_type)
        }
        
        return strategy
    
    def _select_se_targets(self, employee_profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select best targets for social engineering"""
        
        # Sort by susceptibility score
        sorted_profiles = sorted(employee_profiles, key=lambda x: x['susceptibility_score'], reverse=True)
        
        # Select top targets
        num_targets = min(5, len(sorted_profiles))
        return sorted_profiles[:num_targets]
    
    def _generate_pretext(self, employee_profiles: List[Dict[str, Any]], campaign_type: str) -> str:
        """Generate pretext for social engineering campaign"""
        
        pretexts = {
            'spear_phishing': 'IT Security Update Required - Action Needed',
            'business_email_compromise': 'Urgent: CEO Request for Financial Information',
            'watering_hole': 'Industry News and Updates Portal',
            'social_media': 'Professional Network Connection Request'
        }
        
        return pretexts.get(campaign_type, 'Important Security Notice')
    
    def _determine_se_timing(self) -> Dict[str, Any]:
        """Determine optimal timing for social engineering"""
        
        return {
            'day_of_week': 'tuesday',  # Mid-week for business pretexts
            'time_of_day': '10:00 AM', # During busy work hours
            'duration': '2 weeks',     # Campaign duration
            'follow_up_interval': '3 days'
        }
    
    def _define_se_success_criteria(self, campaign_type: str) -> List[str]:
        """Define success criteria for social engineering campaign"""
        
        criteria_map = {
            'spear_phishing': ['credential_harvest', 'malware_installation', 'initial_access'],
            'business_email_compromise': ['financial_fraud', 'data_access', 'wire_transfer'],
            'watering_hole': ['traffic_redirection', 'malware_distribution', 'reconnaissance'],
            'social_media': ['information_gathering', 'trust_establishment', 'further_targeting']
        }
        
        return criteria_map.get(campaign_type, ['information_gathering'])
    
    def _generate_se_materials(self, campaign_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Generate social engineering campaign materials"""
        
        materials = {
            'email_templates': self._generate_email_templates(campaign_strategy),
            'landing_pages': self._generate_landing_pages(campaign_strategy),
            'payload_links': self._generate_payload_links(campaign_strategy),
            'social_media_content': self._generate_social_content(campaign_strategy)
        }
        
        return materials
    
    def _generate_email_templates(self, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate email templates for phishing"""
        
        templates = [
            {
                'subject': strategy['pretext'],
                'body': 'This is a simulated phishing email for security training purposes.',
                'sender': 'IT Security <security@company.com>',
                'urgency_level': 'high'
            }
        ]
        
        return templates
    
    def _generate_landing_pages(self, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate landing pages for credential harvesting"""
        
        pages = [
            {
                'page_type': 'credential_harvest',
                'template': 'office365_login',
                'url': 'https://simulated-phishing-page.local',
                'description': 'Simulated credential harvesting page for training'
            }
        ]
        
        return pages
    
    def _generate_payload_links(self, strategy: Dict[str, Any]) -> List[str]:
        """Generate payload download links"""
        
        return [
            'https://simulated-payload.local/training-file.pdf',
            'https://simulated-payload.local/security-update.exe'
        ]
    
    def _generate_social_content(self, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate social media content for campaigns"""
        
        content = [
            {
                'platform': 'linkedin',
                'content_type': 'connection_request',
                'message': 'Hi, I saw your profile and would like to connect for professional networking.',
                'purpose': 'information_gathering'
            }
        ]
        
        return content
    
    def _generate_se_timeline(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Generate timeline for social engineering campaign"""
        
        return {
            'preparation': '3 days',
            'execution': '1 week', 
            'follow_up': '1 week',
            'analysis': '2 days'
        }
    
    def _create_payload_variant(self, base_payload: Dict[str, Any], evasion_requirement: str) -> Dict[str, Any]:
        """Create payload variant for specific evasion requirement"""
        
        variant = base_payload.copy()
        variant['evasion_technique'] = evasion_requirement
        
        evasion_modifications = {
            'antivirus_evasion': {'obfuscation': 'enabled', 'packing': 'upx'},
            'sandbox_evasion': {'sleep_timer': '300', 'environment_checks': 'enabled'},
            'network_detection_evasion': {'encryption': 'aes256', 'domain_fronting': 'enabled'},
            'behavioral_detection_evasion': {'living_off_land': 'enabled', 'legitimate_tools': 'powershell'}
        }
        
        modifications = evasion_modifications.get(evasion_requirement, {})
        variant.update(modifications)
        
        return variant
    
    def _generate_creative_variants(self, base_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate creative payload variants using LLM creativity"""
        
        creative_techniques = [
            'steganography',
            'dll_side_loading',
            'process_hollowing',
            'reflective_dll_loading',
            'fileless_execution'
        ]
        
        variants = []
        num_variants = int(self.creativity * len(creative_techniques))
        
        selected_techniques = random.sample(creative_techniques, min(num_variants, len(creative_techniques)))
        
        for technique in selected_techniques:
            variant = base_payload.copy()
            variant['creative_technique'] = technique
            variant['creativity_score'] = self.creativity
            variants.append(variant)
        
        return variants
    
    def _generate_apt_profile(self, actor_name: str) -> Dict[str, Any]:
        """Generate APT actor profile"""
        
        profile = {
            'name': actor_name,
            'aliases': [f"{actor_name}_variant_{i}" for i in range(1, 3)],
            'sophistication': random.choice(['medium', 'high', 'nation_state']),
            'motivation': random.choice(['espionage', 'financial', 'sabotage']),
            'target_industries': random.sample(['healthcare', 'finance', 'government', 'defense', 'technology'], 2),
            'target_regions': random.sample(['north_america', 'europe', 'asia_pacific'], 2),
            'first_observed': '2020-01-01',
            'last_activity': datetime.now().strftime('%Y-%m-%d')
        }
        
        return profile
    
    def _generate_campaign_history(self, actor_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate historical campaigns for APT actor"""
        
        campaigns = []
        
        for i in range(3):
            campaign = {
                'campaign_name': f"{actor_profile['name']}_Campaign_{i+1}",
                'start_date': f"202{i}-06-01",
                'end_date': f"202{i}-12-31", 
                'targets': random.sample(actor_profile['target_industries'], 1),
                'initial_vector': random.choice(['spear_phishing', 'watering_hole', 'supply_chain']),
                'malware_families': [f"Custom_Malware_{i+1}", f"RAT_{i+1}"],
                'objectives': [actor_profile['motivation'], 'data_exfiltration']
            }
            campaigns.append(campaign)
        
        return campaigns
    
    def _generate_iocs(self, actor_profile: Dict[str, Any], campaign_history: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Generate indicators of compromise"""
        
        iocs = {
            'domains': [f"{actor_profile['name'].lower()}-c2-{i}.com" for i in range(1, 4)],
            'ip_addresses': [f"192.168.{i}.{j}" for i in range(100, 103) for j in range(1, 2)],
            'file_hashes': [f"abcdef{i:06d}" + "0" * 26 for i in range(1, 6)],
            'mutex_names': [f"{actor_profile['name']}_mutex_{i}" for i in range(1, 3)],
            'registry_keys': [f"HKLM\\Software\\{actor_profile['name']}\\Config_{i}" for i in range(1, 3)]
        }
        
        return iocs
    
    def _extract_ttps(self, campaign_history: List[Dict[str, Any]]) -> List[str]:
        """Extract TTPs from campaign history"""
        
        common_ttps = [
            'T1566.001',  # Spearphishing Attachment
            'T1059.001',  # PowerShell
            'T1055',      # Process Injection
            'T1083',      # File and Directory Discovery
            'T1041'       # Exfiltration Over C2 Channel
        ]
        
        return random.sample(common_ttps, 4)
    
    def _generate_mitigations(self, actor_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate mitigation recommendations"""
        
        mitigations = [
            {
                'mitigation_id': 'M1031',
                'name': 'Network Intrusion Prevention',
                'description': 'Deploy network-based intrusion detection/prevention systems'
            },
            {
                'mitigation_id': 'M1049',
                'name': 'Antivirus/Antimalware',
                'description': 'Deploy endpoint protection with behavioral analysis'
            },
            {
                'mitigation_id': 'M1017',
                'name': 'User Training',
                'description': 'Conduct regular security awareness training focusing on phishing'
            }
        ]
        
        return mitigations