#!/usr/bin/env python3
"""
Progressive Quality Gates - Generation 1: Foundation
Autonomous SDLC with evolutionary quality validation

This implements the first generation of progressive quality gates with:
- Basic functionality validation
- Security compliance checking
- Performance baseline establishment
- Defensive capability verification
"""

import asyncio
import json
import logging
import time
import sys
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import psutil
import hashlib
import uuid
import tempfile
import importlib
import threading

# Security and monitoring imports
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics data structure"""
    gate_name: str
    generation: int
    success: bool
    score: float
    execution_time_ms: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    security_level: str = "standard"
    performance_tier: str = "baseline"
    defensive_readiness: str = "developing"


@dataclass
class ProgressiveReport:
    """Progressive quality gates report"""
    generation: int
    timestamp: datetime
    overall_success: bool
    overall_score: float
    total_gates: int
    passed_gates: int
    failed_gates: int
    gate_metrics: List[QualityMetrics]
    execution_time_total_ms: float
    system_fingerprint: str
    recommendations: List[str] = field(default_factory=list)
    next_generation_ready: bool = False
    defensive_capabilities_score: float = 0.0


class SecurityValidator:
    """Generation 1 Security Validation Engine"""
    
    def __init__(self):
        self.security_rules = {
            'ethical_compliance': True,
            'defensive_only': True,
            'no_malicious_patterns': True,
            'input_sanitization': True,
            'access_control': True
        }
        
    async def validate_ethical_compliance(self, project_root: Path) -> Dict[str, Any]:
        """Validate ethical compliance for defensive security"""
        logger.info("🛡️ Validating ethical compliance...")
        
        start_time = time.time()
        try:
            # Check for ethical frameworks
            ethical_indicators = 0
            security_frameworks = []
            
            # Scan for ethical compliance markers
            python_files = list(project_root.rglob("*.py"))
            for py_file in python_files[:20]:  # Sample files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                        # Look for ethical compliance indicators
                        ethical_patterns = [
                            'ethical_framework',
                            'defensive',
                            'authorized',
                            'compliance',
                            'permission',
                            'consent'
                        ]
                        
                        for pattern in ethical_patterns:
                            if pattern in content:
                                ethical_indicators += 1
                                if py_file.name not in security_frameworks:
                                    security_frameworks.append(py_file.name)
                                break
                                
                except Exception:
                    continue
            
            # Check for malicious patterns (should be none in defensive tools)
            malicious_patterns = [
                'exploit',
                'backdoor', 
                'rootkit',
                'keylogger',
                'trojan'
            ]
            
            malicious_findings = []
            for py_file in python_files[:10]:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        for pattern in malicious_patterns:
                            if pattern in content:
                                # Context check - allow if in defensive context
                                context_words = ['detect', 'defend', 'prevent', 'block', 'security']
                                if not any(ctx in content for ctx in context_words):
                                    malicious_findings.append(f"{py_file.name}:{pattern}")
                except Exception:
                    continue
            
            # Score calculation
            ethical_score = 100
            ethical_score += min(50, ethical_indicators * 5)  # Bonus for ethical indicators
            ethical_score -= len(malicious_findings) * 25     # Penalty for suspicious patterns
            ethical_score = min(100, max(0, ethical_score))
            
            execution_time = (time.time() - start_time) * 1000
            success = ethical_score >= 80 and len(malicious_findings) == 0
            
            recommendations = []
            if ethical_indicators < 5:
                recommendations.append("Add more ethical compliance documentation")
            if malicious_findings:
                recommendations.append(f"Review suspicious patterns: {malicious_findings}")
            if success:
                recommendations.append("Excellent ethical compliance posture")
            
            return {
                'success': success,
                'score': ethical_score,
                'execution_time_ms': execution_time,
                'ethical_indicators': ethical_indicators,
                'security_frameworks': security_frameworks,
                'malicious_findings': malicious_findings,
                'recommendations': recommendations
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Ethical compliance validation failed: {e}")
            return {
                'success': False,
                'score': 0,
                'execution_time_ms': execution_time,
                'error': str(e),
                'recommendations': ['Fix ethical compliance validation system']
            }
    
    async def validate_defensive_focus(self, project_root: Path) -> Dict[str, Any]:
        """Validate defensive security focus"""
        logger.info("🛡️ Validating defensive security focus...")
        
        start_time = time.time()
        try:
            defensive_indicators = 0
            offensive_indicators = 0
            
            # Check for defensive vs offensive patterns
            defensive_patterns = [
                'blue_team', 'defense', 'detection', 'prevention',
                'monitoring', 'incident_response', 'forensics',
                'threat_hunting', 'security_operations'
            ]
            
            offensive_patterns = [
                'red_team', 'attack', 'exploit', 'penetration',
                'vulnerability', 'payload', 'injection'
            ]
            
            python_files = list(project_root.rglob("*.py"))
            for py_file in python_files[:15]:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                        for pattern in defensive_patterns:
                            if pattern in content:
                                defensive_indicators += 1
                                break
                        
                        for pattern in offensive_patterns:
                            if pattern in content:
                                # Check if it's in defensive context
                                defensive_context = any(d in content for d in ['defend', 'detect', 'prevent'])
                                if not defensive_context:
                                    offensive_indicators += 1
                                break
                                
                except Exception:
                    continue
            
            # Calculate defensive focus score
            total_indicators = defensive_indicators + offensive_indicators
            if total_indicators == 0:
                defensive_ratio = 0.5  # Neutral
            else:
                defensive_ratio = defensive_indicators / total_indicators
            
            defensive_score = defensive_ratio * 100
            execution_time = (time.time() - start_time) * 1000
            success = defensive_ratio >= 0.7  # At least 70% defensive
            
            recommendations = []
            if defensive_ratio < 0.5:
                recommendations.append("Increase focus on defensive security capabilities")
            elif defensive_ratio >= 0.8:
                recommendations.append("Excellent defensive security focus")
            else:
                recommendations.append("Good defensive focus, consider strengthening further")
            
            return {
                'success': success,
                'score': defensive_score,
                'execution_time_ms': execution_time,
                'defensive_indicators': defensive_indicators,
                'offensive_indicators': offensive_indicators,
                'defensive_ratio': defensive_ratio,
                'recommendations': recommendations
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Defensive focus validation failed: {e}")
            return {
                'success': False,
                'score': 0,
                'execution_time_ms': execution_time,
                'error': str(e),
                'recommendations': ['Fix defensive focus validation system']
            }


class PerformanceAnalyzer:
    """Generation 1 Performance Analysis Engine"""
    
    def __init__(self):
        self.baseline_metrics = {
            'import_time_ms': 1000,
            'memory_usage_mb': 100,
            'cpu_utilization': 50
        }
    
    async def analyze_import_performance(self, project_root: Path) -> Dict[str, Any]:
        """Analyze module import performance"""
        logger.info("⚡ Analyzing import performance...")
        
        start_time = time.time()
        try:
            # Test critical module imports
            critical_modules = [
                'gan_cyber_range',
                'autonomous_defensive_demo',
                'enhanced_defensive_training'
            ]
            
            import_results = []
            total_import_time = 0
            
            for module in critical_modules:
                module_start = time.time()
                try:
                    # Dynamic import with timeout
                    importlib.import_module(module)
                    import_time = (time.time() - module_start) * 1000
                    total_import_time += import_time
                    import_results.append({
                        'module': module,
                        'success': True,
                        'import_time_ms': import_time
                    })
                except Exception as e:
                    import_time = (time.time() - module_start) * 1000
                    import_results.append({
                        'module': module,
                        'success': False,
                        'error': str(e),
                        'import_time_ms': import_time
                    })
            
            # Calculate performance score
            avg_import_time = total_import_time / len(critical_modules)
            baseline_time = self.baseline_metrics['import_time_ms']
            
            if avg_import_time <= baseline_time * 0.5:
                performance_score = 100
            elif avg_import_time <= baseline_time:
                performance_score = 80
            elif avg_import_time <= baseline_time * 1.5:
                performance_score = 60
            else:
                performance_score = 40
            
            successful_imports = sum(1 for r in import_results if r['success'])
            success_rate = successful_imports / len(critical_modules)
            
            final_score = (performance_score * 0.6) + (success_rate * 100 * 0.4)
            execution_time = (time.time() - start_time) * 1000
            success = final_score >= 70
            
            recommendations = []
            if avg_import_time > baseline_time:
                recommendations.append("Optimize module imports for better performance")
            if success_rate < 1.0:
                recommendations.append("Fix failing module imports")
            if final_score >= 90:
                recommendations.append("Excellent import performance")
            
            return {
                'success': success,
                'score': final_score,
                'execution_time_ms': execution_time,
                'import_results': import_results,
                'average_import_time_ms': avg_import_time,
                'success_rate': success_rate,
                'recommendations': recommendations
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Import performance analysis failed: {e}")
            return {
                'success': False,
                'score': 0,
                'execution_time_ms': execution_time,
                'error': str(e),
                'recommendations': ['Fix import performance analysis system']
            }
    
    async def analyze_resource_usage(self) -> Dict[str, Any]:
        """Analyze system resource usage"""
        logger.info("💾 Analyzing resource usage...")
        
        start_time = time.time()
        try:
            # Get current resource usage
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent(interval=1)
            
            memory_mb = memory_info.rss / 1024 / 1024
            baseline_memory = self.baseline_metrics['memory_usage_mb']
            baseline_cpu = self.baseline_metrics['cpu_utilization']
            
            # Score memory usage
            if memory_mb <= baseline_memory * 0.5:
                memory_score = 100
            elif memory_mb <= baseline_memory:
                memory_score = 80
            elif memory_mb <= baseline_memory * 1.5:
                memory_score = 60
            else:
                memory_score = 40
            
            # Score CPU usage
            if cpu_percent <= baseline_cpu * 0.5:
                cpu_score = 100
            elif cpu_percent <= baseline_cpu:
                cpu_score = 80
            elif cpu_percent <= baseline_cpu * 1.5:
                cpu_score = 60
            else:
                cpu_score = 40
            
            # Overall resource score
            resource_score = (memory_score + cpu_score) / 2
            execution_time = (time.time() - start_time) * 1000
            success = resource_score >= 70
            
            recommendations = []
            if memory_mb > baseline_memory:
                recommendations.append("Optimize memory usage")
            if cpu_percent > baseline_cpu:
                recommendations.append("Optimize CPU utilization")
            if resource_score >= 90:
                recommendations.append("Excellent resource efficiency")
            
            return {
                'success': success,
                'score': resource_score,
                'execution_time_ms': execution_time,
                'memory_usage_mb': memory_mb,
                'cpu_utilization_percent': cpu_percent,
                'memory_score': memory_score,
                'cpu_score': cpu_score,
                'recommendations': recommendations
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Resource usage analysis failed: {e}")
            return {
                'success': False,
                'score': 0,
                'execution_time_ms': execution_time,
                'error': str(e),
                'recommendations': ['Fix resource usage analysis system']
            }


class DefensiveCapabilityAssessor:
    """Generation 1 Defensive Capability Assessment Engine"""
    
    def __init__(self):
        self.capability_domains = {
            'detection': 0.3,
            'response': 0.25, 
            'training': 0.2,
            'monitoring': 0.15,
            'research': 0.1
        }
    
    async def assess_detection_capabilities(self, project_root: Path) -> Dict[str, Any]:
        """Assess threat detection capabilities"""
        logger.info("🔍 Assessing detection capabilities...")
        
        start_time = time.time()
        try:
            detection_modules = [
                'threat_detector',
                'anomaly_detection', 
                'signature_matching',
                'behavioral_analysis',
                'pattern_recognition'
            ]
            
            found_capabilities = []
            detection_files = []
            
            # Search for detection-related files
            python_files = list(project_root.rglob("*.py"))
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        for capability in detection_modules:
                            if capability in content or capability.replace('_', '') in content:
                                if capability not in found_capabilities:
                                    found_capabilities.append(capability)
                                if py_file.name not in detection_files:
                                    detection_files.append(py_file.name)
                except Exception:
                    continue
            
            # Check for detection frameworks
            detection_frameworks = ['yara', 'snort', 'suricata', 'sigma']
            framework_count = 0
            for framework in detection_frameworks:
                for py_file in python_files[:10]:
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            if framework in f.read().lower():
                                framework_count += 1
                                break
                    except Exception:
                        continue
            
            # Calculate detection capability score
            capability_ratio = len(found_capabilities) / len(detection_modules)
            framework_bonus = min(20, framework_count * 5)
            detection_score = (capability_ratio * 80) + framework_bonus
            
            execution_time = (time.time() - start_time) * 1000
            success = detection_score >= 60
            
            recommendations = []
            if capability_ratio < 0.5:
                recommendations.append("Enhance detection capabilities - add more detection methods")
            if framework_count == 0:
                recommendations.append("Consider integrating detection frameworks (YARA, Sigma, etc.)")
            if detection_score >= 80:
                recommendations.append("Strong detection capabilities present")
            
            return {
                'success': success,
                'score': detection_score,
                'execution_time_ms': execution_time,
                'found_capabilities': found_capabilities,
                'detection_files': detection_files,
                'framework_count': framework_count,
                'capability_ratio': capability_ratio,
                'recommendations': recommendations
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Detection capability assessment failed: {e}")
            return {
                'success': False,
                'score': 0,
                'execution_time_ms': execution_time,
                'error': str(e),
                'recommendations': ['Fix detection capability assessment system']
            }
    
    async def assess_training_effectiveness(self, project_root: Path) -> Dict[str, Any]:
        """Assess defensive training effectiveness"""
        logger.info("📚 Assessing training effectiveness...")
        
        start_time = time.time()
        try:
            training_components = [
                'curriculum',
                'scenario',
                'exercise',
                'simulation',
                'assessment',
                'feedback'
            ]
            
            found_training = []
            training_files = []
            
            # Search for training-related components
            python_files = list(project_root.rglob("*.py"))
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        for component in training_components:
                            if component in content:
                                if component not in found_training:
                                    found_training.append(component)
                                if py_file.name not in training_files:
                                    training_files.append(py_file.name)
                except Exception:
                    continue
            
            # Check for training methodologies
            methodologies = ['gamification', 'adaptive', 'progressive', 'hands_on']
            methodology_count = 0
            for methodology in methodologies:
                for py_file in python_files[:10]:
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            if methodology in f.read().lower():
                                methodology_count += 1
                                break
                    except Exception:
                        continue
            
            # Calculate training effectiveness score
            component_ratio = len(found_training) / len(training_components)
            methodology_bonus = min(15, methodology_count * 5)
            training_score = (component_ratio * 85) + methodology_bonus
            
            execution_time = (time.time() - start_time) * 1000
            success = training_score >= 60
            
            recommendations = []
            if component_ratio < 0.5:
                recommendations.append("Expand training components - add scenarios, assessments")
            if methodology_count == 0:
                recommendations.append("Implement modern training methodologies")
            if training_score >= 80:
                recommendations.append("Comprehensive training framework present")
            
            return {
                'success': success,
                'score': training_score,
                'execution_time_ms': execution_time,
                'found_training': found_training,
                'training_files': training_files,
                'methodology_count': methodology_count,
                'component_ratio': component_ratio,
                'recommendations': recommendations
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Training effectiveness assessment failed: {e}")
            return {
                'success': False,
                'score': 0,
                'execution_time_ms': execution_time,
                'error': str(e),
                'recommendations': ['Fix training effectiveness assessment system']
            }


class ProgressiveQualityGatesGeneration1:
    """Generation 1: Foundation Quality Gates Engine"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.generation = 1
        self.security_validator = SecurityValidator()
        self.performance_analyzer = PerformanceAnalyzer()
        self.defensive_assessor = DefensiveCapabilityAssessor()
        self.metrics_history = []
        self.system_fingerprint = self._generate_system_fingerprint()
        
    def _generate_system_fingerprint(self) -> str:
        """Generate unique system fingerprint"""
        try:
            system_info = f"{sys.platform}_{multiprocessing.cpu_count()}_{psutil.virtual_memory().total}"
            return hashlib.sha256(system_info.encode()).hexdigest()[:16]
        except Exception:
            return hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:16]
    
    async def execute_progressive_gates(self) -> ProgressiveReport:
        """Execute Generation 1 progressive quality gates"""
        logger.info("🚀 Starting Progressive Quality Gates - Generation 1")
        
        start_time = time.time()
        gate_metrics = []
        
        # Define Generation 1 gates with weights
        gates = [
            ("Ethical Compliance", self._gate_ethical_compliance, 0.25),
            ("Defensive Focus", self._gate_defensive_focus, 0.20),
            ("Import Performance", self._gate_import_performance, 0.15),
            ("Resource Efficiency", self._gate_resource_efficiency, 0.15),
            ("Detection Capabilities", self._gate_detection_capabilities, 0.15),
            ("Training Effectiveness", self._gate_training_effectiveness, 0.10)
        ]
        
        # Execute gates with concurrent processing
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_gate = {
                executor.submit(self._execute_single_gate, gate_name, gate_func): (gate_name, weight)
                for gate_name, gate_func, weight in gates
            }
            
            for future in as_completed(future_to_gate):
                gate_name, weight = future_to_gate[future]
                try:
                    metrics = await future.result()
                    metrics.generation = self.generation
                    gate_metrics.append((metrics, weight))
                    
                    status = "✅ PASSED" if metrics.success else "❌ FAILED"
                    logger.info(f"{status} {gate_name}: {metrics.score:.1f}/100 ({metrics.execution_time_ms:.1f}ms)")
                    
                except Exception as e:
                    error_metrics = QualityMetrics(
                        gate_name=gate_name,
                        generation=self.generation,
                        success=False,
                        score=0.0,
                        execution_time_ms=0.0,
                        timestamp=datetime.now(timezone.utc),
                        errors=[str(e)],
                        recommendations=[f"Fix {gate_name} execution error"]
                    )
                    gate_metrics.append((error_metrics, weight))
                    logger.error(f"❌ FAILED {gate_name}: {e}")
        
        # Calculate weighted overall score
        weighted_scores = [metrics.score * weight for metrics, weight in gate_metrics]
        overall_score = sum(weighted_scores)
        
        # Determine success criteria
        passed_gates = sum(1 for metrics, _ in gate_metrics if metrics.success)
        overall_success = passed_gates >= len(gates) * 0.75  # 75% pass rate required
        
        # Calculate defensive capabilities score
        defensive_gates = [m for m, _ in gate_metrics if 'detection' in m.gate_name.lower() or 'training' in m.gate_name.lower()]
        defensive_score = sum(m.score for m in defensive_gates) / len(defensive_gates) if defensive_gates else 0
        
        # Generate recommendations
        recommendations = self._generate_progressive_recommendations(gate_metrics, overall_score)
        
        # Determine readiness for next generation
        next_gen_ready = overall_success and overall_score >= 85 and defensive_score >= 80
        
        total_execution_time = (time.time() - start_time) * 1000
        
        report = ProgressiveReport(
            generation=self.generation,
            timestamp=datetime.now(timezone.utc),
            overall_success=overall_success,
            overall_score=overall_score,
            total_gates=len(gates),
            passed_gates=passed_gates,
            failed_gates=len(gates) - passed_gates,
            gate_metrics=[metrics for metrics, _ in gate_metrics],
            execution_time_total_ms=total_execution_time,
            system_fingerprint=self.system_fingerprint,
            recommendations=recommendations,
            next_generation_ready=next_gen_ready,
            defensive_capabilities_score=defensive_score
        )
        
        return report
    
    async def _execute_single_gate(self, gate_name: str, gate_func) -> QualityMetrics:
        """Execute a single quality gate"""
        gate_start = time.time()
        
        try:
            result = await gate_func()
            execution_time = (time.time() - gate_start) * 1000
            
            # Determine security and performance tiers
            security_level = "high" if result.get('score', 0) >= 90 else "standard" if result.get('score', 0) >= 70 else "needs_improvement"
            performance_tier = "excellent" if execution_time <= 1000 else "good" if execution_time <= 3000 else "needs_optimization"
            defensive_readiness = "ready" if result.get('score', 0) >= 85 else "developing" if result.get('score', 0) >= 60 else "requires_work"
            
            metrics = QualityMetrics(
                gate_name=gate_name,
                generation=self.generation,
                success=result.get('success', False),
                score=result.get('score', 0.0),
                execution_time_ms=execution_time,
                timestamp=datetime.now(timezone.utc),
                details=result,
                warnings=result.get('warnings', []),
                errors=result.get('errors', []),
                recommendations=result.get('recommendations', []),
                security_level=security_level,
                performance_tier=performance_tier,
                defensive_readiness=defensive_readiness
            )
            
            return metrics
            
        except Exception as e:
            execution_time = (time.time() - gate_start) * 1000
            logger.error(f"Gate {gate_name} failed: {e}")
            
            return QualityMetrics(
                gate_name=gate_name,
                generation=self.generation,
                success=False,
                score=0.0,
                execution_time_ms=execution_time,
                timestamp=datetime.now(timezone.utc),
                errors=[str(e)],
                recommendations=[f"Fix {gate_name} execution"]
            )
    
    async def _gate_ethical_compliance(self) -> Dict[str, Any]:
        """Ethical compliance quality gate"""
        return await self.security_validator.validate_ethical_compliance(self.project_root)
    
    async def _gate_defensive_focus(self) -> Dict[str, Any]:
        """Defensive focus quality gate"""
        return await self.security_validator.validate_defensive_focus(self.project_root)
    
    async def _gate_import_performance(self) -> Dict[str, Any]:
        """Import performance quality gate"""
        return await self.performance_analyzer.analyze_import_performance(self.project_root)
    
    async def _gate_resource_efficiency(self) -> Dict[str, Any]:
        """Resource efficiency quality gate"""
        return await self.performance_analyzer.analyze_resource_usage()
    
    async def _gate_detection_capabilities(self) -> Dict[str, Any]:
        """Detection capabilities quality gate"""
        return await self.defensive_assessor.assess_detection_capabilities(self.project_root)
    
    async def _gate_training_effectiveness(self) -> Dict[str, Any]:
        """Training effectiveness quality gate"""
        return await self.defensive_assessor.assess_training_effectiveness(self.project_root)
    
    def _generate_progressive_recommendations(self, gate_metrics: List[Tuple[QualityMetrics, float]], overall_score: float) -> List[str]:
        """Generate progressive enhancement recommendations"""
        recommendations = []
        
        # Analyze failed gates
        failed_gates = [metrics for metrics, _ in gate_metrics if not metrics.success]
        low_score_gates = [metrics for metrics, _ in gate_metrics if metrics.score < 70]
        
        if failed_gates:
            gate_names = [m.gate_name for m in failed_gates]
            recommendations.append(f"🚨 Priority: Fix failing gates - {', '.join(gate_names)}")
        
        if low_score_gates:
            gate_names = [m.gate_name for m in low_score_gates]
            recommendations.append(f"⚠️ Improve low-scoring areas - {', '.join(gate_names)}")
        
        # Generation-specific recommendations
        if overall_score >= 85:
            recommendations.append("🎯 Generation 1 Complete - Ready for Generation 2 (Robust Features)")
        elif overall_score >= 70:
            recommendations.append("📈 Good foundation - Address remaining issues before Generation 2")
        else:
            recommendations.append("🔧 Strengthen foundation before advancing to next generation")
        
        # Defensive capability recommendations
        defensive_metrics = [m for m, _ in gate_metrics if 'detection' in m.gate_name.lower() or 'training' in m.gate_name.lower()]
        if defensive_metrics:
            avg_defensive = sum(m.score for m in defensive_metrics) / len(defensive_metrics)
            if avg_defensive >= 80:
                recommendations.append("🛡️ Strong defensive capabilities foundation")
            else:
                recommendations.append("🛡️ Enhance defensive capabilities for production readiness")
        
        # Performance recommendations
        performance_metrics = [m for m, _ in gate_metrics if 'performance' in m.gate_name.lower() or 'efficiency' in m.gate_name.lower()]
        if performance_metrics:
            slow_gates = [m.gate_name for m in performance_metrics if m.execution_time_ms > 3000]
            if slow_gates:
                recommendations.append(f"⚡ Optimize performance in: {', '.join(slow_gates)}")
        
        return recommendations
    
    def save_progress_report(self, report: ProgressiveReport) -> Path:
        """Save progressive quality gates report"""
        timestamp = report.timestamp.strftime('%Y%m%d_%H%M%S')
        report_file = self.project_root / f"progressive_quality_gates_gen1_{timestamp}.json"
        
        # Convert dataclass to dict for JSON serialization
        report_dict = {
            'generation': report.generation,
            'timestamp': report.timestamp.isoformat(),
            'overall_success': report.overall_success,
            'overall_score': report.overall_score,
            'total_gates': report.total_gates,
            'passed_gates': report.passed_gates,
            'failed_gates': report.failed_gates,
            'execution_time_total_ms': report.execution_time_total_ms,
            'system_fingerprint': report.system_fingerprint,
            'next_generation_ready': report.next_generation_ready,
            'defensive_capabilities_score': report.defensive_capabilities_score,
            'gate_metrics': [
                {
                    'gate_name': m.gate_name,
                    'generation': m.generation,
                    'success': m.success,
                    'score': m.score,
                    'execution_time_ms': m.execution_time_ms,
                    'timestamp': m.timestamp.isoformat(),
                    'details': m.details,
                    'warnings': m.warnings,
                    'errors': m.errors,
                    'recommendations': m.recommendations,
                    'security_level': m.security_level,
                    'performance_tier': m.performance_tier,
                    'defensive_readiness': m.defensive_readiness
                }
                for m in report.gate_metrics
            ],
            'recommendations': report.recommendations
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            logger.info(f"📄 Report saved: {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            return None


async def main():
    """Main execution function for Generation 1"""
    logger.info("🚀 Progressive Quality Gates - Generation 1: Foundation")
    
    project_root = Path.cwd()
    quality_gates = ProgressiveQualityGatesGeneration1(project_root)
    
    try:
        # Execute Generation 1 quality gates
        report = await quality_gates.execute_progressive_gates()
        
        # Display results
        print(f"\n{'='*80}")
        print("🔬 PROGRESSIVE QUALITY GATES - GENERATION 1 REPORT")
        print(f"{'='*80}")
        
        print(f"🎯 Generation: {report.generation}")
        print(f"📊 Overall Score: {report.overall_score:.1f}/100")
        print(f"🏆 Gates Passed: {report.passed_gates}/{report.total_gates}")
        print(f"⏱️  Execution Time: {report.execution_time_total_ms:.1f}ms")
        print(f"🛡️  Defensive Score: {report.defensive_capabilities_score:.1f}/100")
        print(f"✅ Status: {'GENERATION 1 COMPLETE' if report.overall_success else 'NEEDS IMPROVEMENT'}")
        print(f"🔄 Next Gen Ready: {'YES' if report.next_generation_ready else 'NO'}")
        
        print(f"\n📋 GATE RESULTS:")
        for metrics in report.gate_metrics:
            status_icon = "✅" if metrics.success else "❌"
            tier_info = f"[{metrics.security_level.upper()}/{metrics.performance_tier.upper()}/{metrics.defensive_readiness.upper()}]"
            print(f"  {status_icon} {metrics.gate_name}: {metrics.score:.1f}/100 {tier_info}")
            
            # Show top recommendations per gate
            if metrics.recommendations:
                for rec in metrics.recommendations[:2]:
                    print(f"    💡 {rec}")
        
        print(f"\n🎯 PROGRESSIVE RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
        
        # Save report
        report_file = quality_gates.save_progress_report(report)
        if report_file:
            print(f"\n📄 Detailed report saved: {report_file}")
        
        # Exit with status code
        sys.exit(0 if report.overall_success else 1)
        
    except Exception as e:
        logger.error(f"❌ Progressive Quality Gates Generation 1 failed: {e}")
        print(f"\n💥 EXECUTION FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run Generation 1 Progressive Quality Gates
    asyncio.run(main())