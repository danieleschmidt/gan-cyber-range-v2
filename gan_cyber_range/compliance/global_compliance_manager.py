"""
Global compliance management system for international cyber range deployment.

This module orchestrates compliance across multiple jurisdictions and provides
automated compliance monitoring, reporting, and enforcement.
"""

import logging
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ComplianceRegion(Enum):
    """Global compliance regions"""
    EUROPEAN_UNION = "EU"
    UNITED_STATES = "US"
    CALIFORNIA = "CA"
    SINGAPORE = "SG"
    CANADA = "CA"
    AUSTRALIA = "AU"
    UNITED_KINGDOM = "UK"
    BRAZIL = "BR"
    JAPAN = "JP"
    CHINA = "CN"


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    PDPA = "pdpa"
    PIPEDA = "pipeda"
    ISO27001 = "iso27001"
    SOC2 = "soc2"
    NIST = "nist"
    PCI_DSS = "pci_dss"


class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    PENDING_REVIEW = "pending_review"
    EXEMPTED = "exempted"


@dataclass
class ComplianceRequirement:
    """Individual compliance requirement"""
    id: str
    framework: ComplianceFramework
    region: ComplianceRegion
    title: str
    description: str
    mandatory: bool = True
    implementation_deadline: Optional[datetime] = None
    current_status: ComplianceStatus = ComplianceStatus.PENDING_REVIEW
    evidence_required: List[str] = field(default_factory=list)
    remediation_steps: List[str] = field(default_factory=list)
    responsible_team: Optional[str] = None
    last_assessment: Optional[datetime] = None
    next_review: Optional[datetime] = None
    risk_level: str = "medium"
    automation_available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceAssessment:
    """Compliance assessment results"""
    assessment_id: str
    framework: ComplianceFramework
    region: ComplianceRegion
    assessment_date: datetime
    overall_status: ComplianceStatus
    total_requirements: int
    compliant_requirements: int
    non_compliant_requirements: int
    compliance_percentage: float
    critical_findings: List[str]
    recommendations: List[str]
    remediation_timeline: Optional[datetime] = None
    assessor: Optional[str] = None
    evidence_collected: List[str] = field(default_factory=list)
    next_assessment_due: Optional[datetime] = None


@dataclass
class DataProcessingActivity:
    """Data processing activity for compliance tracking"""
    activity_id: str
    name: str
    description: str
    data_categories: List[str]
    processing_purposes: List[str]
    legal_basis: str
    data_subjects: List[str]
    recipients: List[str]
    retention_period: str
    security_measures: List[str]
    cross_border_transfers: List[str] = field(default_factory=list)
    automated_decision_making: bool = False
    consent_required: bool = False
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


class GlobalComplianceManager:
    """Comprehensive global compliance management system"""
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        default_regions: Optional[List[ComplianceRegion]] = None
    ):
        self.config_path = config_path or Path("compliance_config.json")
        self.default_regions = default_regions or [
            ComplianceRegion.EUROPEAN_UNION,
            ComplianceRegion.UNITED_STATES,
            ComplianceRegion.CALIFORNIA
        ]
        
        # Compliance state
        self.requirements: Dict[str, ComplianceRequirement] = {}
        self.assessments: Dict[str, ComplianceAssessment] = {}
        self.data_activities: Dict[str, DataProcessingActivity] = {}
        self.compliance_policies: Dict[ComplianceFramework, Dict[str, Any]] = {}
        
        # Regional configurations
        self.region_configs: Dict[ComplianceRegion, Dict[str, Any]] = {}
        self.framework_mappings: Dict[ComplianceRegion, List[ComplianceFramework]] = {}
        
        # Monitoring and alerting
        self.compliance_alerts: List[Dict[str, Any]] = []
        self.monitoring_enabled = True
        self.alert_thresholds = {
            'compliance_percentage_min': 85.0,
            'critical_findings_max': 5,
            'overdue_assessments_max': 3
        }
        
        # Initialize frameworks
        self._initialize_framework_mappings()
        self._initialize_requirements()
        self._load_configuration()
        
        logger.info("GlobalComplianceManager initialized")
    
    def add_compliance_requirement(self, requirement: ComplianceRequirement) -> None:
        """Add a compliance requirement"""
        
        self.requirements[requirement.id] = requirement
        logger.info(f"Added compliance requirement: {requirement.id}")
    
    def update_requirement_status(
        self,
        requirement_id: str,
        status: ComplianceStatus,
        evidence: Optional[List[str]] = None,
        notes: Optional[str] = None
    ) -> bool:
        """Update the status of a compliance requirement"""
        
        if requirement_id not in self.requirements:
            logger.error(f"Requirement {requirement_id} not found")
            return False
        
        requirement = self.requirements[requirement_id]
        requirement.current_status = status
        requirement.last_assessment = datetime.now()
        
        if evidence:
            requirement.evidence_required.extend(evidence)
        
        if notes:
            requirement.metadata['last_update_notes'] = notes
        
        logger.info(f"Updated requirement {requirement_id} status to {status.value}")
        return True
    
    def conduct_compliance_assessment(
        self,
        framework: ComplianceFramework,
        region: ComplianceRegion,
        assessor: Optional[str] = None
    ) -> ComplianceAssessment:
        """Conduct comprehensive compliance assessment"""
        
        logger.info(f"Starting compliance assessment for {framework.value} in {region.value}")
        
        # Filter requirements for this framework and region
        relevant_requirements = [
            req for req in self.requirements.values()
            if req.framework == framework and req.region == region
        ]
        
        total_requirements = len(relevant_requirements)
        compliant_count = len([
            req for req in relevant_requirements
            if req.current_status == ComplianceStatus.COMPLIANT
        ])
        non_compliant_count = len([
            req for req in relevant_requirements
            if req.current_status == ComplianceStatus.NON_COMPLIANT
        ])
        
        compliance_percentage = (compliant_count / total_requirements * 100) if total_requirements > 0 else 0
        
        # Determine overall status
        if compliance_percentage >= 95:
            overall_status = ComplianceStatus.COMPLIANT
        elif compliance_percentage >= 70:
            overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            overall_status = ComplianceStatus.NON_COMPLIANT
        
        # Identify critical findings
        critical_findings = []
        for req in relevant_requirements:
            if (req.current_status == ComplianceStatus.NON_COMPLIANT and 
                req.mandatory and req.risk_level == "high"):
                critical_findings.append(f"{req.id}: {req.title}")
        
        # Generate recommendations
        recommendations = self._generate_compliance_recommendations(
            relevant_requirements, framework, region
        )
        
        # Create assessment
        assessment = ComplianceAssessment(
            assessment_id=f"{framework.value}_{region.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            framework=framework,
            region=region,
            assessment_date=datetime.now(),
            overall_status=overall_status,
            total_requirements=total_requirements,
            compliant_requirements=compliant_count,
            non_compliant_requirements=non_compliant_count,
            compliance_percentage=compliance_percentage,
            critical_findings=critical_findings,
            recommendations=recommendations,
            assessor=assessor,
            next_assessment_due=datetime.now() + timedelta(days=90)  # Quarterly reviews
        )
        
        # Store assessment
        self.assessments[assessment.assessment_id] = assessment
        
        # Check for alerts
        self._check_compliance_alerts(assessment)
        
        logger.info(f"Compliance assessment completed: {assessment.assessment_id}")
        logger.info(f"Overall status: {overall_status.value}, Compliance: {compliance_percentage:.1f}%")
        
        return assessment
    
    def register_data_processing_activity(self, activity: DataProcessingActivity) -> None:
        """Register a data processing activity"""
        
        self.data_activities[activity.activity_id] = activity
        
        # Check compliance implications
        self._assess_data_activity_compliance(activity)
        
        logger.info(f"Registered data processing activity: {activity.activity_id}")
    
    def check_cross_border_transfer_compliance(
        self,
        source_region: ComplianceRegion,
        target_region: ComplianceRegion,
        data_categories: List[str]
    ) -> Dict[str, Any]:
        """Check compliance for cross-border data transfers"""
        
        transfer_check = {
            'source_region': source_region.value,
            'target_region': target_region.value,
            'data_categories': data_categories,
            'compliant': True,
            'restrictions': [],
            'requirements': [],
            'safeguards_needed': []
        }
        
        # GDPR restrictions
        if source_region == ComplianceRegion.EUROPEAN_UNION:
            if target_region not in [ComplianceRegion.UNITED_KINGDOM, ComplianceRegion.CANADA]:
                transfer_check['requirements'].extend([
                    'Adequacy decision or appropriate safeguards required',
                    'Data subject consent or legitimate interest basis needed'
                ])
                transfer_check['safeguards_needed'].extend([
                    'Standard Contractual Clauses (SCCs)',
                    'Binding Corporate Rules (BCRs)',
                    'Certification schemes'
                ])
        
        # CCPA restrictions
        if source_region == ComplianceRegion.CALIFORNIA:
            if 'personal_information' in data_categories:
                transfer_check['requirements'].append(
                    'Consumer right to know about third-party sharing'
                )
        
        # Check for sensitive data
        sensitive_categories = ['biometric', 'health', 'financial', 'genetic']
        if any(cat in data_categories for cat in sensitive_categories):
            transfer_check['restrictions'].append(
                'Enhanced protection required for sensitive personal data'
            )
            transfer_check['safeguards_needed'].append('Encryption in transit and at rest')
        
        # Determine overall compliance
        if transfer_check['restrictions'] or transfer_check['requirements']:
            transfer_check['compliant'] = False
        
        return transfer_check
    
    def generate_compliance_report(
        self,
        frameworks: Optional[List[ComplianceFramework]] = None,
        regions: Optional[List[ComplianceRegion]] = None,
        include_remediation_plan: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        
        frameworks = frameworks or list(ComplianceFramework)
        regions = regions or self.default_regions
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'scope': {
                'frameworks': [f.value for f in frameworks],
                'regions': [r.value for r in regions]
            },
            'executive_summary': {},
            'detailed_assessments': {},
            'compliance_matrix': {},
            'risk_assessment': {},
            'data_processing_summary': {},
            'alerts_and_issues': self.compliance_alerts[-10:],  # Last 10 alerts
            'recommendations': []
        }
        
        # Executive summary
        total_assessments = len(self.assessments)
        compliant_assessments = len([
            a for a in self.assessments.values()
            if a.overall_status == ComplianceStatus.COMPLIANT
        ])
        
        report['executive_summary'] = {
            'total_frameworks_assessed': len(frameworks),
            'total_regions_covered': len(regions),
            'overall_compliance_rate': (compliant_assessments / total_assessments * 100) if total_assessments > 0 else 0,
            'critical_issues': len([
                alert for alert in self.compliance_alerts
                if alert.get('severity') == 'critical'
            ]),
            'data_processing_activities': len(self.data_activities)
        }
        
        # Detailed assessments
        for framework in frameworks:
            for region in regions:
                # Find latest assessment
                relevant_assessments = [
                    a for a in self.assessments.values()
                    if a.framework == framework and a.region == region
                ]
                
                if relevant_assessments:
                    latest_assessment = max(relevant_assessments, key=lambda x: x.assessment_date)
                    
                    key = f"{framework.value}_{region.value}"
                    report['detailed_assessments'][key] = {
                        'assessment_id': latest_assessment.assessment_id,
                        'status': latest_assessment.overall_status.value,
                        'compliance_percentage': latest_assessment.compliance_percentage,
                        'last_assessment': latest_assessment.assessment_date.isoformat(),
                        'critical_findings': latest_assessment.critical_findings,
                        'next_review': latest_assessment.next_assessment_due.isoformat() if latest_assessment.next_assessment_due else None
                    }
        
        # Compliance matrix
        report['compliance_matrix'] = self._generate_compliance_matrix(frameworks, regions)
        
        # Risk assessment
        report['risk_assessment'] = self._generate_risk_assessment()
        
        # Data processing summary
        report['data_processing_summary'] = self._generate_data_processing_summary()
        
        # Overall recommendations
        if include_remediation_plan:
            report['remediation_plan'] = self._generate_remediation_plan(frameworks, regions)
        
        return report
    
    def get_compliance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for compliance dashboard"""
        
        current_time = datetime.now()
        
        # Calculate key metrics
        total_requirements = len(self.requirements)
        compliant_requirements = len([
            req for req in self.requirements.values()
            if req.current_status == ComplianceStatus.COMPLIANT
        ])
        
        overdue_assessments = len([
            req for req in self.requirements.values()
            if req.next_review and req.next_review < current_time
        ])
        
        recent_alerts = [
            alert for alert in self.compliance_alerts
            if alert.get('timestamp', datetime.min) > current_time - timedelta(days=7)
        ]
        
        dashboard_data = {
            'overview': {
                'total_frameworks': len(set(req.framework for req in self.requirements.values())),
                'total_regions': len(set(req.region for req in self.requirements.values())),
                'overall_compliance_rate': (compliant_requirements / total_requirements * 100) if total_requirements > 0 else 0,
                'total_requirements': total_requirements,
                'compliant_requirements': compliant_requirements
            },
            'alerts': {
                'total_active': len(self.compliance_alerts),
                'recent_count': len(recent_alerts),
                'critical_count': len([a for a in recent_alerts if a.get('severity') == 'critical']),
                'overdue_assessments': overdue_assessments
            },
            'by_framework': self._get_compliance_by_framework(),
            'by_region': self._get_compliance_by_region(),
            'trending': self._get_compliance_trends(),
            'upcoming_deadlines': self._get_upcoming_deadlines(),
            'recent_activities': self._get_recent_activities()
        }
        
        return dashboard_data
    
    def auto_remediate_compliance_issues(self) -> Dict[str, Any]:
        """Automatically remediate compliance issues where possible"""
        
        remediation_results = {
            'attempted': 0,
            'successful': 0,
            'failed': 0,
            'actions_taken': [],
            'manual_intervention_required': []
        }
        
        # Find requirements that can be auto-remediated
        auto_remediable = [
            req for req in self.requirements.values()
            if (req.current_status == ComplianceStatus.NON_COMPLIANT and 
                req.automation_available)
        ]
        
        for requirement in auto_remediable:
            remediation_results['attempted'] += 1
            
            try:
                success = self._execute_auto_remediation(requirement)
                
                if success:
                    requirement.current_status = ComplianceStatus.COMPLIANT
                    requirement.last_assessment = datetime.now()
                    remediation_results['successful'] += 1
                    remediation_results['actions_taken'].append({
                        'requirement_id': requirement.id,
                        'action': 'Auto-remediated',
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    remediation_results['failed'] += 1
                    remediation_results['manual_intervention_required'].append({
                        'requirement_id': requirement.id,
                        'reason': 'Auto-remediation failed',
                        'recommended_action': requirement.remediation_steps[0] if requirement.remediation_steps else 'Manual review required'
                    })
                    
            except Exception as e:
                remediation_results['failed'] += 1
                remediation_results['manual_intervention_required'].append({
                    'requirement_id': requirement.id,
                    'reason': f'Auto-remediation error: {str(e)}',
                    'recommended_action': 'Manual review and remediation required'
                })
                logger.error(f"Auto-remediation failed for {requirement.id}: {e}")
        
        logger.info(f"Auto-remediation completed: {remediation_results['successful']}/{remediation_results['attempted']} successful")
        
        return remediation_results
    
    def _initialize_framework_mappings(self) -> None:
        """Initialize framework to region mappings"""
        
        self.framework_mappings = {
            ComplianceRegion.EUROPEAN_UNION: [ComplianceFramework.GDPR, ComplianceFramework.ISO27001],
            ComplianceRegion.UNITED_STATES: [ComplianceFramework.NIST, ComplianceFramework.SOC2],
            ComplianceRegion.CALIFORNIA: [ComplianceFramework.CCPA, ComplianceFramework.SOC2],
            ComplianceRegion.SINGAPORE: [ComplianceFramework.PDPA, ComplianceFramework.ISO27001],
            ComplianceRegion.CANADA: [ComplianceFramework.PIPEDA, ComplianceFramework.ISO27001],
            ComplianceRegion.AUSTRALIA: [ComplianceFramework.ISO27001],
            ComplianceRegion.UNITED_KINGDOM: [ComplianceFramework.GDPR, ComplianceFramework.ISO27001],
            ComplianceRegion.BRAZIL: [ComplianceFramework.ISO27001],
            ComplianceRegion.JAPAN: [ComplianceFramework.ISO27001],
            ComplianceRegion.CHINA: [ComplianceFramework.ISO27001]
        }
    
    def _initialize_requirements(self) -> None:
        """Initialize standard compliance requirements"""
        
        # GDPR requirements
        gdpr_requirements = [
            ComplianceRequirement(
                id="GDPR_ART_5",
                framework=ComplianceFramework.GDPR,
                region=ComplianceRegion.EUROPEAN_UNION,
                title="Principles of Processing",
                description="Personal data must be processed lawfully, fairly and transparently",
                mandatory=True,
                risk_level="high",
                automation_available=True
            ),
            ComplianceRequirement(
                id="GDPR_ART_6",
                framework=ComplianceFramework.GDPR,
                region=ComplianceRegion.EUROPEAN_UNION,
                title="Lawfulness of Processing",
                description="Processing must have a lawful basis",
                mandatory=True,
                risk_level="high",
                automation_available=False
            ),
            ComplianceRequirement(
                id="GDPR_ART_32",
                framework=ComplianceFramework.GDPR,
                region=ComplianceRegion.EUROPEAN_UNION,
                title="Security of Processing",
                description="Implement appropriate technical and organizational measures",
                mandatory=True,
                risk_level="high",
                automation_available=True
            )
        ]
        
        # CCPA requirements
        ccpa_requirements = [
            ComplianceRequirement(
                id="CCPA_1798_100",
                framework=ComplianceFramework.CCPA,
                region=ComplianceRegion.CALIFORNIA,
                title="Consumer Right to Know",
                description="Consumers have the right to know what personal information is collected",
                mandatory=True,
                risk_level="medium",
                automation_available=True
            ),
            ComplianceRequirement(
                id="CCPA_1798_105",
                framework=ComplianceFramework.CCPA,
                region=ComplianceRegion.CALIFORNIA,
                title="Consumer Right to Delete",
                description="Consumers have the right to delete personal information",
                mandatory=True,
                risk_level="high",
                automation_available=True
            )
        ]
        
        # Add all requirements
        all_requirements = gdpr_requirements + ccpa_requirements
        for req in all_requirements:
            self.requirements[req.id] = req
    
    def _generate_compliance_recommendations(
        self,
        requirements: List[ComplianceRequirement],
        framework: ComplianceFramework,
        region: ComplianceRegion
    ) -> List[str]:
        """Generate compliance recommendations"""
        
        recommendations = []
        
        non_compliant = [req for req in requirements if req.current_status == ComplianceStatus.NON_COMPLIANT]
        
        if non_compliant:
            recommendations.append(f"Address {len(non_compliant)} non-compliant requirements")
            
            # High-priority recommendations
            high_risk = [req for req in non_compliant if req.risk_level == "high"]
            if high_risk:
                recommendations.append(f"Prioritize {len(high_risk)} high-risk requirements")
        
        # Framework-specific recommendations
        if framework == ComplianceFramework.GDPR:
            recommendations.extend([
                "Implement data protection by design and by default",
                "Ensure lawful basis documentation is complete",
                "Review and update privacy notices"
            ])
        elif framework == ComplianceFramework.CCPA:
            recommendations.extend([
                "Implement consumer request handling procedures",
                "Update privacy policy with CCPA disclosures",
                "Establish data mapping for consumer requests"
            ])
        
        return recommendations
    
    def _check_compliance_alerts(self, assessment: ComplianceAssessment) -> None:
        """Check for compliance alerts based on assessment"""
        
        current_time = datetime.now()
        
        # Check compliance percentage threshold
        if assessment.compliance_percentage < self.alert_thresholds['compliance_percentage_min']:
            alert = {
                'type': 'compliance_threshold',
                'severity': 'high',
                'message': f"Compliance percentage ({assessment.compliance_percentage:.1f}%) below threshold",
                'framework': assessment.framework.value,
                'region': assessment.region.value,
                'timestamp': current_time
            }
            self.compliance_alerts.append(alert)
        
        # Check critical findings threshold
        if len(assessment.critical_findings) > self.alert_thresholds['critical_findings_max']:
            alert = {
                'type': 'critical_findings',
                'severity': 'critical',
                'message': f"Too many critical findings: {len(assessment.critical_findings)}",
                'framework': assessment.framework.value,
                'region': assessment.region.value,
                'timestamp': current_time
            }
            self.compliance_alerts.append(alert)
    
    def _assess_data_activity_compliance(self, activity: DataProcessingActivity) -> None:
        """Assess compliance implications of data processing activity"""
        
        # Check for GDPR implications
        if 'personal_data' in activity.data_categories:
            if not activity.legal_basis:
                alert = {
                    'type': 'missing_legal_basis',
                    'severity': 'high',
                    'message': f"Data activity {activity.activity_id} lacks legal basis",
                    'activity_id': activity.activity_id,
                    'timestamp': datetime.now()
                }
                self.compliance_alerts.append(alert)
        
        # Check for cross-border transfers
        if activity.cross_border_transfers:
            for transfer in activity.cross_border_transfers:
                # Simplified check - would need more sophisticated logic
                if 'non_adequate_country' in transfer:
                    alert = {
                        'type': 'cross_border_risk',
                        'severity': 'medium',
                        'message': f"Cross-border transfer to non-adequate country detected",
                        'activity_id': activity.activity_id,
                        'timestamp': datetime.now()
                    }
                    self.compliance_alerts.append(alert)
    
    def _generate_compliance_matrix(
        self,
        frameworks: List[ComplianceFramework],
        regions: List[ComplianceRegion]
    ) -> Dict[str, Dict[str, str]]:
        """Generate compliance status matrix"""
        
        matrix = {}
        
        for framework in frameworks:
            matrix[framework.value] = {}
            for region in regions:
                # Find latest assessment
                relevant_assessments = [
                    a for a in self.assessments.values()
                    if a.framework == framework and a.region == region
                ]
                
                if relevant_assessments:
                    latest = max(relevant_assessments, key=lambda x: x.assessment_date)
                    matrix[framework.value][region.value] = latest.overall_status.value
                else:
                    matrix[framework.value][region.value] = "not_assessed"
        
        return matrix
    
    def _generate_risk_assessment(self) -> Dict[str, Any]:
        """Generate overall risk assessment"""
        
        high_risk_requirements = [
            req for req in self.requirements.values()
            if (req.risk_level == "high" and 
                req.current_status == ComplianceStatus.NON_COMPLIANT)
        ]
        
        return {
            'overall_risk_level': 'high' if len(high_risk_requirements) > 5 else 'medium',
            'high_risk_count': len(high_risk_requirements),
            'risk_factors': [
                req.title for req in high_risk_requirements[:5]  # Top 5 risks
            ]
        }
    
    def _generate_data_processing_summary(self) -> Dict[str, Any]:
        """Generate data processing activities summary"""
        
        return {
            'total_activities': len(self.data_activities),
            'activities_with_consent': len([
                a for a in self.data_activities.values()
                if a.consent_required
            ]),
            'cross_border_activities': len([
                a for a in self.data_activities.values()
                if a.cross_border_transfers
            ]),
            'automated_decision_making': len([
                a for a in self.data_activities.values()
                if a.automated_decision_making
            ])
        }
    
    def _generate_remediation_plan(
        self,
        frameworks: List[ComplianceFramework],
        regions: List[ComplianceRegion]
    ) -> Dict[str, Any]:
        """Generate remediation plan for non-compliant requirements"""
        
        non_compliant = [
            req for req in self.requirements.values()
            if (req.current_status == ComplianceStatus.NON_COMPLIANT and
                req.framework in frameworks and req.region in regions)
        ]
        
        # Prioritize by risk level and mandatory status
        high_priority = [req for req in non_compliant if req.risk_level == "high" and req.mandatory]
        medium_priority = [req for req in non_compliant if req.risk_level == "medium" and req.mandatory]
        low_priority = [req for req in non_compliant if not req.mandatory]
        
        return {
            'total_issues': len(non_compliant),
            'high_priority': len(high_priority),
            'medium_priority': len(medium_priority),
            'low_priority': len(low_priority),
            'estimated_timeline_days': len(high_priority) * 7 + len(medium_priority) * 3 + len(low_priority) * 1,
            'prioritized_actions': [
                {
                    'requirement_id': req.id,
                    'title': req.title,
                    'priority': 'high' if req in high_priority else 'medium' if req in medium_priority else 'low',
                    'steps': req.remediation_steps
                }
                for req in (high_priority + medium_priority + low_priority)[:10]  # Top 10
            ]
        }
    
    def _get_compliance_by_framework(self) -> Dict[str, Any]:
        """Get compliance status by framework"""
        
        by_framework = {}
        
        for framework in ComplianceFramework:
            framework_reqs = [
                req for req in self.requirements.values()
                if req.framework == framework
            ]
            
            if framework_reqs:
                compliant_count = len([
                    req for req in framework_reqs
                    if req.current_status == ComplianceStatus.COMPLIANT
                ])
                
                by_framework[framework.value] = {
                    'total': len(framework_reqs),
                    'compliant': compliant_count,
                    'percentage': (compliant_count / len(framework_reqs)) * 100
                }
        
        return by_framework
    
    def _get_compliance_by_region(self) -> Dict[str, Any]:
        """Get compliance status by region"""
        
        by_region = {}
        
        for region in ComplianceRegion:
            region_reqs = [
                req for req in self.requirements.values()
                if req.region == region
            ]
            
            if region_reqs:
                compliant_count = len([
                    req for req in region_reqs
                    if req.current_status == ComplianceStatus.COMPLIANT
                ])
                
                by_region[region.value] = {
                    'total': len(region_reqs),
                    'compliant': compliant_count,
                    'percentage': (compliant_count / len(region_reqs)) * 100
                }
        
        return by_region
    
    def _get_compliance_trends(self) -> Dict[str, Any]:
        """Get compliance trends over time"""
        
        # Simplified trend calculation
        recent_assessments = sorted(
            self.assessments.values(),
            key=lambda x: x.assessment_date,
            reverse=True
        )[:5]  # Last 5 assessments
        
        if len(recent_assessments) >= 2:
            latest_avg = sum(a.compliance_percentage for a in recent_assessments[:2]) / 2
            older_avg = sum(a.compliance_percentage for a in recent_assessments[2:]) / max(1, len(recent_assessments[2:]))
            
            trend = "improving" if latest_avg > older_avg else "declining" if latest_avg < older_avg else "stable"
        else:
            trend = "insufficient_data"
        
        return {
            'trend': trend,
            'recent_assessments': len(recent_assessments),
            'latest_percentage': recent_assessments[0].compliance_percentage if recent_assessments else 0
        }
    
    def _get_upcoming_deadlines(self) -> List[Dict[str, Any]]:
        """Get upcoming compliance deadlines"""
        
        upcoming = []
        current_time = datetime.now()
        
        for req in self.requirements.values():
            if req.next_review and req.next_review > current_time:
                days_until = (req.next_review - current_time).days
                if days_until <= 30:  # Next 30 days
                    upcoming.append({
                        'requirement_id': req.id,
                        'title': req.title,
                        'due_date': req.next_review.isoformat(),
                        'days_until': days_until,
                        'priority': req.risk_level
                    })
        
        return sorted(upcoming, key=lambda x: x['days_until'])[:10]  # Next 10 deadlines
    
    def _get_recent_activities(self) -> List[Dict[str, Any]]:
        """Get recent compliance activities"""
        
        activities = []
        current_time = datetime.now()
        
        # Recent requirement updates
        for req in self.requirements.values():
            if req.last_assessment and (current_time - req.last_assessment).days <= 7:
                activities.append({
                    'type': 'requirement_update',
                    'description': f"Updated {req.title}",
                    'timestamp': req.last_assessment.isoformat(),
                    'status': req.current_status.value
                })
        
        # Recent assessments
        for assessment in self.assessments.values():
            if (current_time - assessment.assessment_date).days <= 7:
                activities.append({
                    'type': 'assessment_completed',
                    'description': f"Completed {assessment.framework.value} assessment for {assessment.region.value}",
                    'timestamp': assessment.assessment_date.isoformat(),
                    'status': assessment.overall_status.value
                })
        
        return sorted(activities, key=lambda x: x['timestamp'], reverse=True)[:10]
    
    def _execute_auto_remediation(self, requirement: ComplianceRequirement) -> bool:
        """Execute automatic remediation for a requirement"""
        
        # Placeholder for auto-remediation logic
        # In practice, this would contain specific remediation steps
        
        if requirement.id == "GDPR_ART_32":
            # Example: Enable encryption
            return True
        elif requirement.id == "CCPA_1798_100":
            # Example: Update privacy notice
            return True
        
        return False
    
    def _load_configuration(self) -> None:
        """Load compliance configuration from file"""
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Load alert thresholds
                if 'alert_thresholds' in config:
                    self.alert_thresholds.update(config['alert_thresholds'])
                
                # Load monitoring settings
                self.monitoring_enabled = config.get('monitoring_enabled', True)
                
                logger.info(f"Loaded compliance configuration from {self.config_path}")
                
            except Exception as e:
                logger.error(f"Failed to load compliance config: {e}")
        else:
            # Create default configuration
            self._create_default_config()
    
    def _create_default_config(self) -> None:
        """Create default compliance configuration"""
        
        default_config = {
            'monitoring_enabled': True,
            'alert_thresholds': {
                'compliance_percentage_min': 85.0,
                'critical_findings_max': 5,
                'overdue_assessments_max': 3
            },
            'assessment_frequency_days': 90,
            'auto_remediation_enabled': True
        }
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            logger.info(f"Created default compliance configuration at {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to create default compliance config: {e}")


# Convenience functions for common compliance operations
def quick_gdpr_assessment(manager: GlobalComplianceManager) -> ComplianceAssessment:
    """Quick GDPR compliance assessment for EU region"""
    return manager.conduct_compliance_assessment(
        ComplianceFramework.GDPR,
        ComplianceRegion.EUROPEAN_UNION
    )


def quick_ccpa_assessment(manager: GlobalComplianceManager) -> ComplianceAssessment:
    """Quick CCPA compliance assessment for California"""
    return manager.conduct_compliance_assessment(
        ComplianceFramework.CCPA,
        ComplianceRegion.CALIFORNIA
    )