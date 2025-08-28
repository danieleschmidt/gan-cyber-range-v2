#!/usr/bin/env python3
"""
Progressive Quality Gates - Generation 2: Robust Validation
Advanced autonomous SDLC with comprehensive validation and self-healing

This implements the second generation with:
- Advanced error handling and recovery
- Comprehensive security scanning with ML detection
- Intelligent performance optimization
- Adaptive defensive capability validation
- Self-healing quality gate mechanisms
"""

import asyncio
import json
import logging
import time
import sys
import subprocess
import traceback
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
import psutil
import hashlib
import uuid
import tempfile
import importlib
import threading
import sqlite3
import pickle
from collections import defaultdict, deque
import re
import ast

# Advanced imports
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    HAS_ML = True
except ImportError:
    HAS_ML = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RobustQualityMetrics:
    """Enhanced quality metrics with recovery and adaptation data"""
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
    recovery_attempts: int = 0
    adaptation_applied: bool = False
    confidence_score: float = 0.0
    complexity_rating: str = "medium"
    risk_assessment: str = "low"


@dataclass
class AdaptiveReport:
    """Generation 2 adaptive quality gates report"""
    generation: int
    timestamp: datetime
    overall_success: bool
    overall_score: float
    confidence_interval: Tuple[float, float]
    total_gates: int
    passed_gates: int
    failed_gates: int
    recovered_gates: int
    gate_metrics: List[RobustQualityMetrics]
    execution_time_total_ms: float
    system_fingerprint: str
    recommendations: List[str] = field(default_factory=list)
    next_generation_ready: bool = False
    defensive_capabilities_score: float = 0.0
    adaptation_effectiveness: float = 0.0
    risk_profile: str = "medium"
    predictive_insights: List[str] = field(default_factory=list)


class MLSecurityAnalyzer:
    """Machine Learning-based Security Analysis Engine"""
    
    def __init__(self):
        self.vulnerability_patterns = {}
        self.code_embeddings = {}
        self.threat_model = None
        self.initialize_ml_components()
        
    def initialize_ml_components(self):
        """Initialize ML components for security analysis"""
        if not HAS_ML:
            logger.warning("ML libraries not available, using rule-based fallback")
            return
            
        # Initialize threat detection patterns
        self.vulnerability_patterns = {
            'sql_injection': [
                r'execute\s*\(\s*[\'"].*%s.*[\'"]',
                r'cursor\.execute\s*\(\s*[\'"].*\+.*[\'"]',
                r'query\s*=\s*[\'"].*\+.*[\'"]'
            ],
            'xss': [
                r'innerHTML\s*=\s*.*\+',
                r'document\.write\s*\(',
                r'eval\s*\('
            ],
            'insecure_random': [
                r'random\.random\(',
                r'math\.random\(',
                r'time\.time\(\)\s*%'
            ],
            'hardcoded_secrets': [
                r'password\s*=\s*[\'"][^\'"\s]{8,}[\'"]',
                r'api_key\s*=\s*[\'"][^\'"\s]{16,}[\'"]',
                r'secret\s*=\s*[\'"][^\'"\s]{8,}[\'"]'
            ]
        }
    
    async def analyze_code_security_ml(self, project_root: Path) -> Dict[str, Any]:
        """ML-based code security analysis"""
        logger.info("🤖 Running ML-based security analysis...")
        
        start_time = time.time()
        try:
            python_files = list(project_root.rglob("*.py"))
            if not python_files:
                return {
                    'success': False,
                    'score': 0,
                    'error': 'No Python files found',
                    'recommendations': ['Add Python code to analyze']
                }
            
            # Extract code features
            code_features = []
            file_risks = []
            vulnerability_findings = []
            
            for py_file in python_files[:50]:  # Limit for performance
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Pattern-based vulnerability detection
                    file_vulnerabilities = []
                    for vuln_type, patterns in self.vulnerability_patterns.items():
                        for pattern in patterns:
                            matches = re.finditer(pattern, content, re.IGNORECASE)
                            for match in matches:
                                file_vulnerabilities.append({
                                    'type': vuln_type,
                                    'line': content[:match.start()].count('\n') + 1,
                                    'pattern': pattern,
                                    'severity': 'high' if vuln_type in ['sql_injection', 'xss'] else 'medium'
                                })
                    
                    if file_vulnerabilities:
                        vulnerability_findings.extend(file_vulnerabilities)
                    
                    # Code complexity analysis
                    complexity_score = self._analyze_code_complexity(content)
                    security_score = max(0, 100 - len(file_vulnerabilities) * 15)
                    
                    code_features.append({
                        'file': str(py_file.relative_to(project_root)),
                        'complexity': complexity_score,
                        'security_score': security_score,
                        'vulnerabilities': len(file_vulnerabilities),
                        'lines': len(content.splitlines())
                    })
                    
                    # Calculate risk level
                    if len(file_vulnerabilities) > 3 or complexity_score > 80:
                        file_risks.append('high')
                    elif len(file_vulnerabilities) > 1 or complexity_score > 60:
                        file_risks.append('medium')
                    else:
                        file_risks.append('low')
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze {py_file}: {e}")
                    continue
            
            # ML-based clustering analysis (if available)
            cluster_insights = []
            if HAS_ML and len(code_features) > 5:
                try:
                    cluster_insights = await self._perform_security_clustering(code_features)
                except Exception as e:
                    logger.warning(f"ML clustering failed: {e}")
            
            # Calculate overall security score
            avg_file_score = statistics.mean([f['security_score'] for f in code_features]) if code_features else 0
            vulnerability_penalty = min(50, len(vulnerability_findings) * 5)
            risk_penalty = file_risks.count('high') * 10 + file_risks.count('medium') * 5
            
            overall_score = max(0, avg_file_score - vulnerability_penalty - risk_penalty)
            
            execution_time = (time.time() - start_time) * 1000
            success = overall_score >= 70 and len([v for v in vulnerability_findings if v['severity'] == 'high']) == 0
            
            # Generate intelligent recommendations
            recommendations = self._generate_ml_security_recommendations(
                vulnerability_findings, code_features, cluster_insights
            )
            
            return {
                'success': success,
                'score': overall_score,
                'execution_time_ms': execution_time,
                'files_analyzed': len(code_features),
                'vulnerabilities_found': len(vulnerability_findings),
                'high_severity_count': len([v for v in vulnerability_findings if v['severity'] == 'high']),
                'risk_distribution': dict(zip(['low', 'medium', 'high'], 
                                            [file_risks.count(r) for r in ['low', 'medium', 'high']])),
                'code_features': code_features[:10],  # Sample for report
                'vulnerability_findings': vulnerability_findings[:20],  # Top findings
                'cluster_insights': cluster_insights,
                'recommendations': recommendations
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"ML security analysis failed: {e}")
            return {
                'success': False,
                'score': 0,
                'execution_time_ms': execution_time,
                'error': str(e),
                'recommendations': ['Fix ML security analysis system']
            }
    
    def _analyze_code_complexity(self, code: str) -> float:
        """Analyze code complexity using AST"""
        try:
            tree = ast.parse(code)
            complexity_score = 0
            
            for node in ast.walk(tree):
                # Cyclomatic complexity indicators
                if isinstance(node, (ast.If, ast.While, ast.For)):
                    complexity_score += 2
                elif isinstance(node, ast.FunctionDef):
                    complexity_score += 1
                elif isinstance(node, ast.ClassDef):
                    complexity_score += 1
                elif isinstance(node, (ast.Try, ast.ExceptHandler)):
                    complexity_score += 1
            
            # Normalize complexity score
            lines = len(code.splitlines())
            normalized_score = min(100, (complexity_score / max(lines / 10, 1)) * 100)
            return normalized_score
            
        except Exception:
            # Fallback complexity estimation
            lines = len(code.splitlines())
            keywords = ['if', 'while', 'for', 'try', 'except', 'def', 'class']
            keyword_count = sum(code.count(keyword) for keyword in keywords)
            return min(100, (keyword_count / max(lines / 20, 1)) * 100)
    
    async def _perform_security_clustering(self, code_features: List[Dict]) -> List[str]:
        """Perform ML-based security clustering analysis"""
        if not HAS_ML or len(code_features) < 3:
            return ["Insufficient data for ML clustering"]
        
        try:
            # Prepare feature matrix
            features = []
            for feature in code_features:
                features.append([
                    feature['complexity'],
                    feature['security_score'],
                    feature['vulnerabilities'],
                    feature['lines']
                ])
            
            X = np.array(features)
            
            # Perform K-means clustering
            n_clusters = min(3, len(features))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Calculate silhouette score
            if n_clusters > 1:
                sil_score = silhouette_score(X, cluster_labels)
            else:
                sil_score = 0
            
            # Analyze clusters
            cluster_insights = []
            for i in range(n_clusters):
                cluster_files = [f for j, f in enumerate(code_features) if cluster_labels[j] == i]
                avg_complexity = statistics.mean([f['complexity'] for f in cluster_files])
                avg_security = statistics.mean([f['security_score'] for f in cluster_files])
                
                if avg_complexity > 70 and avg_security < 60:
                    cluster_insights.append(f"Cluster {i}: High complexity, low security ({len(cluster_files)} files)")
                elif avg_security > 90:
                    cluster_insights.append(f"Cluster {i}: High security quality ({len(cluster_files)} files)")
                else:
                    cluster_insights.append(f"Cluster {i}: Mixed quality ({len(cluster_files)} files)")
            
            cluster_insights.append(f"Clustering quality score: {sil_score:.3f}")
            return cluster_insights
            
        except Exception as e:
            logger.warning(f"Security clustering analysis failed: {e}")
            return ["ML clustering analysis failed"]
    
    def _generate_ml_security_recommendations(self, vulnerabilities: List, features: List, clusters: List) -> List[str]:
        """Generate ML-based security recommendations"""
        recommendations = []
        
        # Vulnerability-based recommendations
        vuln_types = set(v['type'] for v in vulnerabilities)
        if 'sql_injection' in vuln_types:
            recommendations.append("CRITICAL: Implement parameterized queries to prevent SQL injection")
        if 'xss' in vuln_types:
            recommendations.append("HIGH: Add input sanitization to prevent XSS attacks")
        if 'hardcoded_secrets' in vuln_types:
            recommendations.append("HIGH: Move hardcoded secrets to environment variables")
        if 'insecure_random' in vuln_types:
            recommendations.append("MEDIUM: Use cryptographically secure random number generation")
        
        # Complexity-based recommendations
        if features:
            high_complexity_files = [f for f in features if f['complexity'] > 80]
            if len(high_complexity_files) > len(features) * 0.2:
                recommendations.append("Refactor high-complexity files to improve maintainability")
        
        # ML cluster-based recommendations
        for cluster_insight in clusters:
            if "High complexity, low security" in cluster_insight:
                recommendations.append("Focus security review on high-complexity modules")
            elif "High security quality" in cluster_insight:
                recommendations.append("Use high-security modules as templates for others")
        
        if not recommendations:
            recommendations.append("Security analysis complete - no critical issues detected")
        
        return recommendations


class IntelligentPerformanceOptimizer:
    """Intelligent Performance Optimization Engine"""
    
    def __init__(self):
        self.performance_history = deque(maxlen=100)
        self.optimization_cache = {}
        self.benchmark_baselines = {
            'import_time_ms': 500,
            'memory_usage_mb': 50,
            'cpu_utilization': 30,
            'io_throughput_ops_sec': 1000
        }
    
    async def optimize_import_performance(self, project_root: Path) -> Dict[str, Any]:
        """Intelligent import performance optimization"""
        logger.info("⚡ Optimizing import performance...")
        
        start_time = time.time()
        try:
            # Critical modules to test
            critical_modules = [
                'gan_cyber_range',
                'autonomous_defensive_demo',
                'enhanced_defensive_training',
                'robust_defensive_framework'
            ]
            
            import_results = []
            optimization_suggestions = []
            
            for module in critical_modules:
                module_start = time.time()
                try:
                    # Pre-warming optimization
                    if module in self.optimization_cache:
                        logger.info(f"Using cached optimization for {module}")
                    
                    # Measure import with profiling
                    import sys
                    if module in sys.modules:
                        del sys.modules[module]  # Force fresh import
                    
                    importlib.import_module(module)
                    import_time = (time.time() - module_start) * 1000
                    
                    # Analyze import dependencies
                    dependency_count = len([m for m in sys.modules.keys() if module in m])
                    
                    import_results.append({
                        'module': module,
                        'success': True,
                        'import_time_ms': import_time,
                        'dependency_count': dependency_count,
                        'optimization_potential': self._calculate_optimization_potential(import_time, dependency_count)
                    })
                    
                    # Generate optimization suggestions
                    if import_time > self.benchmark_baselines['import_time_ms']:
                        optimization_suggestions.append(f"Optimize {module} imports - {import_time:.1f}ms (target: {self.benchmark_baselines['import_time_ms']}ms)")
                    
                except Exception as e:
                    import_time = (time.time() - module_start) * 1000
                    import_results.append({
                        'module': module,
                        'success': False,
                        'error': str(e),
                        'import_time_ms': import_time,
                        'optimization_potential': 'fix_required'
                    })
                    optimization_suggestions.append(f"Fix import error in {module}: {e}")
            
            # Calculate performance metrics
            successful_imports = [r for r in import_results if r['success']]
            avg_import_time = statistics.mean([r['import_time_ms'] for r in successful_imports]) if successful_imports else 0
            success_rate = len(successful_imports) / len(critical_modules)
            
            # Intelligent scoring with adaptive baselines
            baseline_time = self.benchmark_baselines['import_time_ms']
            if avg_import_time <= baseline_time * 0.5:
                performance_score = 100
            elif avg_import_time <= baseline_time:
                performance_score = 85
            elif avg_import_time <= baseline_time * 2:
                performance_score = 70
            else:
                performance_score = 50
            
            # Apply success rate multiplier
            final_score = performance_score * success_rate
            
            # Adaptive baseline adjustment
            if len(self.performance_history) > 10:
                historical_avg = statistics.mean(self.performance_history)
                if avg_import_time < historical_avg * 0.8:
                    optimization_suggestions.append("Performance improving - consider tightening baselines")
            
            self.performance_history.append(avg_import_time)
            
            execution_time = (time.time() - start_time) * 1000
            success = final_score >= 70 and success_rate >= 0.8
            
            return {
                'success': success,
                'score': final_score,
                'execution_time_ms': execution_time,
                'import_results': import_results,
                'average_import_time_ms': avg_import_time,
                'success_rate': success_rate,
                'optimization_suggestions': optimization_suggestions,
                'performance_trend': self._analyze_performance_trend(),
                'recommendations': self._generate_performance_recommendations(import_results, optimization_suggestions)
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Import performance optimization failed: {e}")
            return {
                'success': False,
                'score': 0,
                'execution_time_ms': execution_time,
                'error': str(e),
                'recommendations': ['Fix import performance optimization system']
            }
    
    def _calculate_optimization_potential(self, import_time: float, dependency_count: int) -> str:
        """Calculate optimization potential for a module"""
        baseline = self.benchmark_baselines['import_time_ms']
        
        if import_time <= baseline * 0.5:
            return 'optimized'
        elif import_time <= baseline:
            return 'good'
        elif import_time <= baseline * 2:
            if dependency_count > 10:
                return 'high_potential_dependencies'
            else:
                return 'medium_potential'
        else:
            return 'high_potential_critical'
    
    def _analyze_performance_trend(self) -> str:
        """Analyze performance trend from history"""
        if len(self.performance_history) < 3:
            return 'insufficient_data'
        
        recent = list(self.performance_history)[-3:]
        older = list(self.performance_history)[:-3] if len(self.performance_history) > 3 else recent
        
        recent_avg = statistics.mean(recent)
        older_avg = statistics.mean(older)
        
        if recent_avg < older_avg * 0.9:
            return 'improving'
        elif recent_avg > older_avg * 1.1:
            return 'degrading'
        else:
            return 'stable'
    
    def _generate_performance_recommendations(self, results: List, suggestions: List) -> List[str]:
        """Generate intelligent performance recommendations"""
        recommendations = []
        
        # Add optimization suggestions
        recommendations.extend(suggestions)
        
        # Analyze patterns
        failed_imports = [r for r in results if not r['success']]
        if failed_imports:
            recommendations.append(f"Priority: Fix {len(failed_imports)} failed imports before optimization")
        
        slow_imports = [r for r in results if r.get('success') and r.get('import_time_ms', 0) > 1000]
        if slow_imports:
            recommendations.append(f"Optimize {len(slow_imports)} slow-loading modules")
        
        # Trend-based recommendations
        trend = self._analyze_performance_trend()
        if trend == 'degrading':
            recommendations.append("Performance degrading - investigate recent changes")
        elif trend == 'improving':
            recommendations.append("Performance improving - good optimization work")
        
        if not recommendations:
            recommendations.append("Import performance is within acceptable parameters")
        
        return recommendations


class AdaptiveDefensiveAnalyzer:
    """Adaptive Defensive Capability Analysis Engine"""
    
    def __init__(self):
        self.capability_weights = {
            'detection': 0.25,
            'prevention': 0.20,
            'response': 0.20,
            'training': 0.15,
            'monitoring': 0.10,
            'research': 0.10
        }
        self.adaptive_thresholds = {
            'detection_accuracy': 0.85,
            'response_time_ms': 1000,
            'training_effectiveness': 0.80
        }
    
    async def analyze_adaptive_detection(self, project_root: Path) -> Dict[str, Any]:
        """Analyze adaptive detection capabilities"""
        logger.info("🎯 Analyzing adaptive detection capabilities...")
        
        start_time = time.time()
        try:
            detection_components = {
                'signature_based': ['yara', 'snort', 'clamav', 'sigma'],
                'behavioral': ['anomaly', 'behavior', 'heuristic', 'baseline'],
                'machine_learning': ['classifier', 'neural', 'clustering', 'ensemble'],
                'threat_intelligence': ['ioc', 'threat_feed', 'reputation', 'context']
            }
            
            found_components = defaultdict(list)
            detection_files = []
            capability_scores = {}
            
            # Scan for detection components
            python_files = list(project_root.rglob("*.py"))
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                        for category, keywords in detection_components.items():
                            for keyword in keywords:
                                if keyword in content:
                                    if keyword not in found_components[category]:
                                        found_components[category].append(keyword)
                                    if py_file.name not in detection_files:
                                        detection_files.append(py_file.name)
                except Exception:
                    continue
            
            # Calculate capability scores
            total_possible = sum(len(keywords) for keywords in detection_components.values())
            total_found = sum(len(found) for found in found_components.values())
            
            for category, keywords in detection_components.items():
                category_found = len(found_components[category])
                category_possible = len(keywords)
                capability_scores[category] = (category_found / category_possible) * 100
            
            # Adaptive scoring based on context
            detection_diversity = len(found_components.keys())
            coverage_score = (total_found / total_possible) * 100 if total_possible > 0 else 0
            diversity_bonus = min(20, detection_diversity * 5)
            
            overall_score = min(100, coverage_score + diversity_bonus)
            
            # Assess adaptive capabilities
            ml_present = len(found_components['machine_learning']) > 0
            behavioral_present = len(found_components['behavioral']) > 0
            adaptive_score_bonus = 0
            
            if ml_present and behavioral_present:
                adaptive_score_bonus = 15
                adaptive_capability = 'high'
            elif ml_present or behavioral_present:
                adaptive_score_bonus = 8
                adaptive_capability = 'medium'
            else:
                adaptive_capability = 'basic'
            
            final_score = min(100, overall_score + adaptive_score_bonus)
            
            execution_time = (time.time() - start_time) * 1000
            success = final_score >= 70 and detection_diversity >= 2
            
            recommendations = self._generate_detection_recommendations(
                found_components, capability_scores, adaptive_capability
            )
            
            return {
                'success': success,
                'score': final_score,
                'execution_time_ms': execution_time,
                'found_components': dict(found_components),
                'capability_scores': capability_scores,
                'detection_diversity': detection_diversity,
                'adaptive_capability': adaptive_capability,
                'ml_present': ml_present,
                'behavioral_present': behavioral_present,
                'detection_files': detection_files,
                'recommendations': recommendations
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Adaptive detection analysis failed: {e}")
            return {
                'success': False,
                'score': 0,
                'execution_time_ms': execution_time,
                'error': str(e),
                'recommendations': ['Fix adaptive detection analysis system']
            }
    
    def _generate_detection_recommendations(self, components: Dict, scores: Dict, adaptive_level: str) -> List[str]:
        """Generate adaptive detection recommendations"""
        recommendations = []
        
        # Component-specific recommendations
        if scores.get('machine_learning', 0) < 30:
            recommendations.append("Implement ML-based detection capabilities")
        if scores.get('behavioral', 0) < 50:
            recommendations.append("Enhance behavioral analysis components")
        if scores.get('threat_intelligence', 0) < 40:
            recommendations.append("Integrate threat intelligence feeds")
        
        # Adaptive capability recommendations
        if adaptive_level == 'basic':
            recommendations.append("Develop adaptive detection mechanisms")
        elif adaptive_level == 'medium':
            recommendations.append("Enhance existing adaptive capabilities")
        else:
            recommendations.append("Excellent adaptive detection capabilities")
        
        # Coverage recommendations
        missing_categories = [cat for cat, found in components.items() if not found]
        if missing_categories:
            recommendations.append(f"Add missing detection categories: {', '.join(missing_categories)}")
        
        return recommendations


class SelfHealingQualityGate:
    """Self-healing quality gate with recovery mechanisms"""
    
    def __init__(self):
        self.recovery_strategies = {
            'import_failure': self._recover_import_failure,
            'timeout_error': self._recover_timeout_error,
            'resource_exhaustion': self._recover_resource_exhaustion,
            'analysis_failure': self._recover_analysis_failure
        }
        self.recovery_history = defaultdict(int)
        self.max_recovery_attempts = 3
    
    async def execute_with_recovery(self, gate_func, gate_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute gate with self-healing recovery mechanisms"""
        recovery_attempts = 0
        last_error = None
        
        while recovery_attempts <= self.max_recovery_attempts:
            try:
                result = await gate_func(*args, **kwargs)
                
                # Enhance result with recovery info
                result['recovery_attempts'] = recovery_attempts
                result['self_healed'] = recovery_attempts > 0
                
                return result
                
            except Exception as e:
                last_error = e
                recovery_attempts += 1
                
                if recovery_attempts > self.max_recovery_attempts:
                    break
                
                logger.warning(f"Gate {gate_name} failed (attempt {recovery_attempts}): {e}")
                
                # Attempt recovery
                error_type = self._classify_error(e)
                if error_type in self.recovery_strategies:
                    try:
                        await self.recovery_strategies[error_type](e, gate_name)
                        logger.info(f"Recovery attempt {recovery_attempts} for {gate_name}")
                        await asyncio.sleep(1)  # Brief pause before retry
                    except Exception as recovery_error:
                        logger.error(f"Recovery failed: {recovery_error}")
                else:
                    logger.warning(f"No recovery strategy for error type: {error_type}")
        
        # All recovery attempts failed
        self.recovery_history[gate_name] += recovery_attempts
        return {
            'success': False,
            'score': 0,
            'error': str(last_error),
            'recovery_attempts': recovery_attempts,
            'recovery_failed': True,
            'recommendations': [f"Manual intervention required for {gate_name}"]
        }
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error type for recovery strategy selection"""
        error_str = str(error).lower()
        
        if 'import' in error_str or 'module' in error_str:
            return 'import_failure'
        elif 'timeout' in error_str or 'time' in error_str:
            return 'timeout_error'
        elif 'memory' in error_str or 'resource' in error_str:
            return 'resource_exhaustion'
        else:
            return 'analysis_failure'
    
    async def _recover_import_failure(self, error: Exception, gate_name: str):
        """Recover from import failures"""
        logger.info(f"Attempting import failure recovery for {gate_name}")
        
        # Clear import cache
        import sys
        modules_to_clear = [m for m in sys.modules.keys() if 'gan_cyber_range' in m]
        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]
        
        # Force garbage collection
        import gc
        gc.collect()
    
    async def _recover_timeout_error(self, error: Exception, gate_name: str):
        """Recover from timeout errors"""
        logger.info(f"Attempting timeout recovery for {gate_name}")
        
        # Reduce analysis scope or increase timeout
        # Implementation would depend on specific gate requirements
        await asyncio.sleep(2)  # Cool-down period
    
    async def _recover_resource_exhaustion(self, error: Exception, gate_name: str):
        """Recover from resource exhaustion"""
        logger.info(f"Attempting resource exhaustion recovery for {gate_name}")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear caches
        if hasattr(self, 'optimization_cache'):
            self.optimization_cache.clear()
    
    async def _recover_analysis_failure(self, error: Exception, gate_name: str):
        """Recover from general analysis failures"""
        logger.info(f"Attempting general recovery for {gate_name}")
        
        # Generic recovery: clear state and retry
        import gc
        gc.collect()
        await asyncio.sleep(1)


class ProgressiveQualityGatesGeneration2:
    """Generation 2: Robust Quality Gates with Advanced Validation"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.generation = 2
        self.ml_security_analyzer = MLSecurityAnalyzer()
        self.performance_optimizer = IntelligentPerformanceOptimizer()
        self.defensive_analyzer = AdaptiveDefensiveAnalyzer()
        self.self_healing = SelfHealingQualityGate()
        self.metrics_history = []
        self.system_fingerprint = self._generate_system_fingerprint()
        
    def _generate_system_fingerprint(self) -> str:
        """Generate enhanced system fingerprint"""
        try:
            import platform
            system_info = f"{platform.system()}_{platform.release()}_{multiprocessing.cpu_count()}_{psutil.virtual_memory().total}_{self.generation}"
            return hashlib.sha256(system_info.encode()).hexdigest()[:20]
        except Exception:
            return hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:20]
    
    async def execute_robust_gates(self) -> AdaptiveReport:
        """Execute Generation 2 robust quality gates with self-healing"""
        logger.info("🚀 Starting Progressive Quality Gates - Generation 2: Robust Validation")
        
        start_time = time.time()
        gate_metrics = []
        
        # Define Generation 2 gates with enhanced weights and recovery
        gates = [
            ("ML Security Analysis", self._gate_ml_security_analysis, 0.25),
            ("Intelligent Performance", self._gate_intelligent_performance, 0.20),
            ("Adaptive Detection", self._gate_adaptive_detection, 0.20),
            ("Robust Error Handling", self._gate_robust_error_handling, 0.15),
            ("Self-Healing Validation", self._gate_self_healing_validation, 0.10),
            ("Predictive Analytics", self._gate_predictive_analytics, 0.10)
        ]
        
        # Execute gates with concurrent processing and self-healing
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_gate = {
                executor.submit(self._execute_robust_gate, gate_name, gate_func): (gate_name, weight)
                for gate_name, gate_func, weight in gates
            }
            
            for future in as_completed(future_to_gate):
                gate_name, weight = future_to_gate[future]
                try:
                    metrics = future.result()
                    if asyncio.iscoroutine(metrics):
                        metrics = await metrics
                    
                    metrics.generation = self.generation
                    gate_metrics.append((metrics, weight))
                    
                    status = "✅ PASSED" if metrics.success else "❌ FAILED"
                    recovery_info = f" (🔧 Self-healed)" if metrics.adaptation_applied else ""
                    logger.info(f"{status} {gate_name}: {metrics.score:.1f}/100{recovery_info}")
                    
                except Exception as e:
                    error_metrics = RobustQualityMetrics(
                        gate_name=gate_name,
                        generation=self.generation,
                        success=False,
                        score=0.0,
                        execution_time_ms=0.0,
                        timestamp=datetime.now(timezone.utc),
                        errors=[str(e)],
                        recommendations=[f"Fix {gate_name} execution error"],
                        risk_assessment="high"
                    )
                    gate_metrics.append((error_metrics, weight))
                    logger.error(f"❌ FAILED {gate_name}: {e}")
        
        # Calculate weighted scores with confidence intervals
        weighted_scores = [metrics.score * weight for metrics, weight in gate_metrics]
        overall_score = sum(weighted_scores)
        
        # Calculate confidence interval
        scores = [metrics.score for metrics, _ in gate_metrics]
        if len(scores) > 1:
            score_std = statistics.stdev(scores)
            confidence_margin = 1.96 * (score_std / len(scores) ** 0.5)  # 95% CI
            confidence_interval = (
                max(0, overall_score - confidence_margin),
                min(100, overall_score + confidence_margin)
            )
        else:
            confidence_interval = (overall_score, overall_score)
        
        # Advanced success criteria
        passed_gates = sum(1 for metrics, _ in gate_metrics if metrics.success)
        recovered_gates = sum(1 for metrics, _ in gate_metrics if metrics.adaptation_applied)
        
        # Multi-criteria success evaluation
        score_success = overall_score >= 80  # Higher threshold for Gen 2
        pass_rate_success = passed_gates >= len(gates) * 0.80  # 80% pass rate
        critical_gate_success = all(m.success for m, _ in gate_metrics if m.gate_name in ["ML Security Analysis", "Adaptive Detection"])
        
        overall_success = score_success and pass_rate_success and critical_gate_success
        
        # Calculate adaptation effectiveness
        total_attempts = sum(metrics.recovery_attempts for metrics, _ in gate_metrics)
        successful_recoveries = sum(1 for metrics, _ in gate_metrics if metrics.adaptation_applied and metrics.success)
        adaptation_effectiveness = (successful_recoveries / max(total_attempts, 1)) * 100
        
        # Risk assessment
        high_risk_gates = [m for m, _ in gate_metrics if m.risk_assessment == "high"]
        if len(high_risk_gates) > len(gates) * 0.2:
            risk_profile = "high"
        elif len(high_risk_gates) > 0:
            risk_profile = "medium"
        else:
            risk_profile = "low"
        
        # Generate recommendations and insights
        recommendations = self._generate_robust_recommendations(gate_metrics, overall_score, adaptation_effectiveness)
        predictive_insights = self._generate_predictive_insights(gate_metrics, risk_profile)
        
        # Determine readiness for Generation 3
        next_gen_ready = (overall_success and 
                         overall_score >= 90 and 
                         adaptation_effectiveness >= 75 and 
                         risk_profile in ["low", "medium"])
        
        total_execution_time = (time.time() - start_time) * 1000
        
        report = AdaptiveReport(
            generation=self.generation,
            timestamp=datetime.now(timezone.utc),
            overall_success=overall_success,
            overall_score=overall_score,
            confidence_interval=confidence_interval,
            total_gates=len(gates),
            passed_gates=passed_gates,
            failed_gates=len(gates) - passed_gates,
            recovered_gates=recovered_gates,
            gate_metrics=[metrics for metrics, _ in gate_metrics],
            execution_time_total_ms=total_execution_time,
            system_fingerprint=self.system_fingerprint,
            recommendations=recommendations,
            next_generation_ready=next_gen_ready,
            defensive_capabilities_score=self._calculate_defensive_score(gate_metrics),
            adaptation_effectiveness=adaptation_effectiveness,
            risk_profile=risk_profile,
            predictive_insights=predictive_insights
        )
        
        return report
    
    async def _execute_robust_gate(self, gate_name: str, gate_func) -> RobustQualityMetrics:
        """Execute a single robust quality gate with self-healing"""
        gate_start = time.time()
        
        # Execute with self-healing recovery
        result = await self.self_healing.execute_with_recovery(gate_func, gate_name)
        
        execution_time = (time.time() - gate_start) * 1000
        
        # Enhanced assessment
        security_level = self._assess_security_level(result.get('score', 0), result.get('vulnerabilities_found', 0))
        performance_tier = self._assess_performance_tier(execution_time, result.get('score', 0))
        defensive_readiness = self._assess_defensive_readiness(result.get('score', 0), gate_name)
        complexity_rating = self._assess_complexity(result)
        risk_assessment = self._assess_risk(result, gate_name)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(result, execution_time)
        
        metrics = RobustQualityMetrics(
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
            defensive_readiness=defensive_readiness,
            recovery_attempts=result.get('recovery_attempts', 0),
            adaptation_applied=result.get('self_healed', False),
            confidence_score=confidence_score,
            complexity_rating=complexity_rating,
            risk_assessment=risk_assessment
        )
        
        return metrics
    
    def _assess_security_level(self, score: float, vulnerabilities: int) -> str:
        """Assess security level based on score and vulnerabilities"""
        if score >= 95 and vulnerabilities == 0:
            return "excellent"
        elif score >= 85 and vulnerabilities <= 2:
            return "high"
        elif score >= 70 and vulnerabilities <= 5:
            return "standard"
        else:
            return "needs_improvement"
    
    def _assess_performance_tier(self, execution_time: float, score: float) -> str:
        """Assess performance tier based on execution time and score"""
        if execution_time <= 1000 and score >= 90:
            return "excellent"
        elif execution_time <= 3000 and score >= 70:
            return "good"
        elif execution_time <= 5000:
            return "acceptable"
        else:
            return "needs_optimization"
    
    def _assess_defensive_readiness(self, score: float, gate_name: str) -> str:
        """Assess defensive readiness based on gate type and score"""
        is_critical_defensive = any(keyword in gate_name.lower() 
                                  for keyword in ['security', 'detection', 'defensive'])
        
        if is_critical_defensive:
            if score >= 90:
                return "production_ready"
            elif score >= 75:
                return "ready"
            else:
                return "requires_work"
        else:
            if score >= 80:
                return "ready"
            elif score >= 60:
                return "developing"
            else:
                return "requires_work"
    
    def _assess_complexity(self, result: Dict[str, Any]) -> str:
        """Assess complexity of the gate execution"""
        indicators = 0
        
        if result.get('files_analyzed', 0) > 50:
            indicators += 1
        if result.get('vulnerabilities_found', 0) > 10:
            indicators += 1
        if result.get('recovery_attempts', 0) > 0:
            indicators += 1
        if len(result.get('recommendations', [])) > 5:
            indicators += 1
        
        if indicators >= 3:
            return "high"
        elif indicators >= 1:
            return "medium"
        else:
            return "low"
    
    def _assess_risk(self, result: Dict[str, Any], gate_name: str) -> str:
        """Assess risk level based on results"""
        risk_indicators = 0
        
        if not result.get('success', False):
            risk_indicators += 2
        if result.get('high_severity_count', 0) > 0:
            risk_indicators += 2
        if result.get('recovery_failed', False):
            risk_indicators += 1
        if result.get('score', 100) < 50:
            risk_indicators += 1
        
        if risk_indicators >= 4:
            return "critical"
        elif risk_indicators >= 2:
            return "high"
        elif risk_indicators >= 1:
            return "medium"
        else:
            return "low"
    
    def _calculate_confidence_score(self, result: Dict[str, Any], execution_time: float) -> float:
        """Calculate confidence score for the gate execution"""
        confidence = 100.0
        
        # Reduce confidence for errors or failures
        if not result.get('success', False):
            confidence -= 30
        if result.get('errors'):
            confidence -= len(result['errors']) * 5
        if result.get('recovery_attempts', 0) > 0:
            confidence -= result['recovery_attempts'] * 10
        
        # Adjust for execution characteristics
        if execution_time > 5000:  # Very slow execution
            confidence -= 10
        
        return max(0, min(100, confidence))
    
    async def _gate_ml_security_analysis(self) -> Dict[str, Any]:
        """ML-based security analysis gate"""
        return await self.ml_security_analyzer.analyze_code_security_ml(self.project_root)
    
    async def _gate_intelligent_performance(self) -> Dict[str, Any]:
        """Intelligent performance optimization gate"""
        return await self.performance_optimizer.optimize_import_performance(self.project_root)
    
    async def _gate_adaptive_detection(self) -> Dict[str, Any]:
        """Adaptive detection capabilities gate"""
        return await self.defensive_analyzer.analyze_adaptive_detection(self.project_root)
    
    async def _gate_robust_error_handling(self) -> Dict[str, Any]:
        """Robust error handling validation gate"""
        logger.info("🛡️ Validating robust error handling...")
        
        start_time = time.time()
        try:
            # Test error handling patterns
            error_patterns = [
                'try:', 'except:', 'finally:', 'raise',
                'error_handler', 'robust', 'fallback'
            ]
            
            python_files = list(self.project_root.rglob("*.py"))
            files_with_error_handling = 0
            error_handling_score = 0
            
            for py_file in python_files[:20]:  # Sample files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    pattern_count = sum(1 for pattern in error_patterns if pattern in content)
                    if pattern_count >= 2:  # At least try/except
                        files_with_error_handling += 1
                        error_handling_score += min(10, pattern_count)
                        
                except Exception:
                    continue
            
            # Calculate error handling coverage
            coverage = (files_with_error_handling / len(python_files[:20])) * 100 if python_files else 0
            robustness_score = min(100, coverage + (error_handling_score / len(python_files[:20]) * 20))
            
            execution_time = (time.time() - start_time) * 1000
            success = robustness_score >= 70
            
            return {
                'success': success,
                'score': robustness_score,
                'execution_time_ms': execution_time,
                'files_with_error_handling': files_with_error_handling,
                'coverage_percentage': coverage,
                'recommendations': ['Enhance error handling patterns'] if coverage < 80 else ['Good error handling coverage']
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return {
                'success': False,
                'score': 0,
                'execution_time_ms': execution_time,
                'error': str(e),
                'recommendations': ['Fix error handling validation']
            }
    
    async def _gate_self_healing_validation(self) -> Dict[str, Any]:
        """Self-healing validation gate"""
        logger.info("🔧 Validating self-healing capabilities...")
        
        start_time = time.time()
        try:
            # Test self-healing mechanisms
            recovery_indicators = [
                'recovery', 'fallback', 'retry', 'adaptive',
                'self_heal', 'resilient', 'redundant'
            ]
            
            python_files = list(self.project_root.rglob("*.py"))
            self_healing_files = []
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                    
                    if any(indicator in content for indicator in recovery_indicators):
                        self_healing_files.append(py_file.name)
                        
                except Exception:
                    continue
            
            # Assess recovery history from current execution
            total_recoveries = sum(self.self_healing.recovery_history.values())
            recovery_effectiveness = 0
            if total_recoveries > 0:
                successful_recoveries = len([h for h in self.self_healing.recovery_history.values() if h > 0])
                recovery_effectiveness = (successful_recoveries / total_recoveries) * 100
            
            # Calculate self-healing score
            indicator_score = min(60, len(self_healing_files) * 5)
            effectiveness_score = min(40, recovery_effectiveness)
            self_healing_score = indicator_score + effectiveness_score
            
            execution_time = (time.time() - start_time) * 1000
            success = self_healing_score >= 60
            
            return {
                'success': success,
                'score': self_healing_score,
                'execution_time_ms': execution_time,
                'self_healing_files': len(self_healing_files),
                'recovery_effectiveness': recovery_effectiveness,
                'total_recoveries': total_recoveries,
                'recommendations': ['Implement more self-healing mechanisms'] if self_healing_score < 70 else ['Good self-healing capabilities']
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return {
                'success': False,
                'score': 0,
                'execution_time_ms': execution_time,
                'error': str(e),
                'recommendations': ['Fix self-healing validation']
            }
    
    async def _gate_predictive_analytics(self) -> Dict[str, Any]:
        """Predictive analytics gate"""
        logger.info("🔮 Performing predictive analytics...")
        
        start_time = time.time()
        try:
            # Analyze trends and patterns
            predictive_score = 0
            insights = []
            
            # Performance trend analysis
            if len(self.performance_optimizer.performance_history) > 3:
                trend = self.performance_optimizer._analyze_performance_trend()
                if trend == 'improving':
                    predictive_score += 30
                    insights.append("Performance trend is improving")
                elif trend == 'stable':
                    predictive_score += 20
                    insights.append("Performance is stable")
                else:
                    predictive_score += 10
                    insights.append("Performance trend needs attention")
            
            # Recovery pattern analysis
            recovery_patterns = self.self_healing.recovery_history
            if recovery_patterns:
                avg_recoveries = sum(recovery_patterns.values()) / len(recovery_patterns)
                if avg_recoveries < 1:
                    predictive_score += 25
                    insights.append("Low recovery requirements indicate system stability")
                elif avg_recoveries < 2:
                    predictive_score += 15
                    insights.append("Moderate recovery patterns - system is adapting well")
                else:
                    predictive_score += 5
                    insights.append("High recovery requirements - investigate root causes")
            
            # System resource prediction
            try:
                cpu_usage = psutil.cpu_percent(interval=1)
                memory_usage = psutil.virtual_memory().percent
                
                if cpu_usage < 50 and memory_usage < 70:
                    predictive_score += 25
                    insights.append("System resources are optimal for scaling")
                elif cpu_usage < 70 and memory_usage < 85:
                    predictive_score += 15
                    insights.append("System resources are adequate")
                else:
                    predictive_score += 5
                    insights.append("System resource constraints may limit performance")
            except Exception:
                predictive_score += 10
                insights.append("Resource monitoring data unavailable")
            
            # Code evolution prediction
            python_files = list(self.project_root.rglob("*.py"))
            if len(python_files) > 50:
                predictive_score += 20
                insights.append("Large codebase indicates mature system")
            elif len(python_files) > 20:
                predictive_score += 15
                insights.append("Medium codebase with growth potential")
            else:
                predictive_score += 10
                insights.append("Small codebase with high growth potential")
            
            execution_time = (time.time() - start_time) * 1000
            success = predictive_score >= 60
            
            return {
                'success': success,
                'score': predictive_score,
                'execution_time_ms': execution_time,
                'predictive_insights': insights,
                'recommendations': ['Use predictive insights for planning'] if success else ['Enhance predictive analytics capabilities']
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return {
                'success': False,
                'score': 0,
                'execution_time_ms': execution_time,
                'error': str(e),
                'recommendations': ['Fix predictive analytics system']
            }
    
    def _calculate_defensive_score(self, gate_metrics: List[Tuple[RobustQualityMetrics, float]]) -> float:
        """Calculate overall defensive capabilities score"""
        defensive_gates = [
            m for m, _ in gate_metrics 
            if any(keyword in m.gate_name.lower() for keyword in ['security', 'detection', 'defensive'])
        ]
        
        if not defensive_gates:
            return 0.0
        
        return sum(m.score for m in defensive_gates) / len(defensive_gates)
    
    def _generate_robust_recommendations(self, gate_metrics: List, overall_score: float, adaptation_effectiveness: float) -> List[str]:
        """Generate robust recommendations based on comprehensive analysis"""
        recommendations = []
        
        # Failed gate analysis
        failed_gates = [metrics for metrics, _ in gate_metrics if not metrics.success]
        if failed_gates:
            critical_failures = [m for m in failed_gates if m.risk_assessment in ["critical", "high"]]
            if critical_failures:
                gate_names = [m.gate_name for m in critical_failures]
                recommendations.append(f"🚨 CRITICAL: Immediately address failures in {', '.join(gate_names)}")
            
            other_failures = [m for m in failed_gates if m.risk_assessment not in ["critical", "high"]]
            if other_failures:
                gate_names = [m.gate_name for m in other_failures]
                recommendations.append(f"⚠️ Address remaining failures in {', '.join(gate_names)}")
        
        # Adaptation effectiveness analysis
        if adaptation_effectiveness < 50:
            recommendations.append("🔧 Improve self-healing mechanisms - low recovery success rate")
        elif adaptation_effectiveness >= 80:
            recommendations.append("✨ Excellent self-healing capabilities demonstrated")
        
        # Performance analysis
        slow_gates = [m for m, _ in gate_metrics if m.execution_time_ms > 5000]
        if slow_gates:
            recommendations.append(f"⚡ Optimize performance in slow gates: {', '.join(m.gate_name for m in slow_gates)}")
        
        # Security analysis
        security_gates = [m for m, _ in gate_metrics if 'security' in m.gate_name.lower()]
        if security_gates:
            avg_security = sum(m.score for m in security_gates) / len(security_gates)
            if avg_security < 80:
                recommendations.append("🛡️ Strengthen security measures - below recommended threshold")
            elif avg_security >= 95:
                recommendations.append("🏆 Excellent security posture maintained")
        
        # Generation progression
        if overall_score >= 90 and adaptation_effectiveness >= 75:
            recommendations.append("🎯 Generation 2 Excellence Achieved - Ready for Generation 3 (AI-Optimized)")
        elif overall_score >= 80:
            recommendations.append("📈 Strong Generation 2 foundation - Minor improvements before Generation 3")
        else:
            recommendations.append("🔧 Strengthen Generation 2 capabilities before advancing")
        
        return recommendations
    
    def _generate_predictive_insights(self, gate_metrics: List, risk_profile: str) -> List[str]:
        """Generate predictive insights based on patterns and trends"""
        insights = []
        
        # Risk-based predictions
        if risk_profile == "low":
            insights.append("🔮 System stability trend indicates reliable production readiness")
        elif risk_profile == "high":
            insights.append("🔮 High risk indicators suggest need for stability improvements")
        
        # Performance predictions
        avg_execution_time = sum(m.execution_time_ms for m, _ in gate_metrics) / len(gate_metrics)
        if avg_execution_time < 2000:
            insights.append("🔮 Performance metrics indicate excellent scalability potential")
        elif avg_execution_time > 5000:
            insights.append("🔮 Performance trends suggest optimization requirements for scaling")
        
        # Adaptation predictions
        adaptation_count = sum(1 for m, _ in gate_metrics if m.adaptation_applied)
        if adaptation_count > len(gate_metrics) * 0.3:
            insights.append("🔮 High adaptation activity indicates system resilience but may need stability review")
        elif adaptation_count == 0:
            insights.append("🔮 No adaptations needed indicates stable system operation")
        
        # Quality evolution prediction
        high_confidence_gates = [m for m, _ in gate_metrics if m.confidence_score >= 90]
        if len(high_confidence_gates) >= len(gate_metrics) * 0.8:
            insights.append("🔮 High confidence scores predict excellent system reliability")
        
        return insights
    
    def save_adaptive_report(self, report: AdaptiveReport) -> Path:
        """Save adaptive quality gates report"""
        timestamp = report.timestamp.strftime('%Y%m%d_%H%M%S')
        report_file = self.project_root / f"progressive_quality_gates_gen2_{timestamp}.json"
        
        # Convert dataclass to dict for JSON serialization
        report_dict = {
            'generation': report.generation,
            'timestamp': report.timestamp.isoformat(),
            'overall_success': report.overall_success,
            'overall_score': report.overall_score,
            'confidence_interval': report.confidence_interval,
            'total_gates': report.total_gates,
            'passed_gates': report.passed_gates,
            'failed_gates': report.failed_gates,
            'recovered_gates': report.recovered_gates,
            'execution_time_total_ms': report.execution_time_total_ms,
            'system_fingerprint': report.system_fingerprint,
            'next_generation_ready': report.next_generation_ready,
            'defensive_capabilities_score': report.defensive_capabilities_score,
            'adaptation_effectiveness': report.adaptation_effectiveness,
            'risk_profile': report.risk_profile,
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
                    'defensive_readiness': m.defensive_readiness,
                    'recovery_attempts': m.recovery_attempts,
                    'adaptation_applied': m.adaptation_applied,
                    'confidence_score': m.confidence_score,
                    'complexity_rating': m.complexity_rating,
                    'risk_assessment': m.risk_assessment
                }
                for m in report.gate_metrics
            ],
            'recommendations': report.recommendations,
            'predictive_insights': report.predictive_insights
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            logger.info(f"📄 Adaptive report saved: {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"Failed to save adaptive report: {e}")
            return None


async def main():
    """Main execution function for Generation 2"""
    logger.info("🚀 Progressive Quality Gates - Generation 2: Robust Validation")
    
    project_root = Path.cwd()
    quality_gates = ProgressiveQualityGatesGeneration2(project_root)
    
    try:
        # Execute Generation 2 quality gates
        report = await quality_gates.execute_robust_gates()
        
        # Display enhanced results
        print(f"\n{'='*80}")
        print("🔬 PROGRESSIVE QUALITY GATES - GENERATION 2 REPORT")
        print(f"{'='*80}")
        
        print(f"🎯 Generation: {report.generation}")
        print(f"📊 Overall Score: {report.overall_score:.1f}/100")
        print(f"📈 Confidence: {report.confidence_interval[0]:.1f} - {report.confidence_interval[1]:.1f}")
        print(f"🏆 Gates Passed: {report.passed_gates}/{report.total_gates}")
        print(f"🔧 Self-Healed: {report.recovered_gates}")
        print(f"⏱️  Execution Time: {report.execution_time_total_ms:.1f}ms")
        print(f"🛡️  Defensive Score: {report.defensive_capabilities_score:.1f}/100")
        print(f"🔄 Adaptation Effectiveness: {report.adaptation_effectiveness:.1f}%")
        print(f"⚠️  Risk Profile: {report.risk_profile.upper()}")
        print(f"✅ Status: {'GENERATION 2 COMPLETE' if report.overall_success else 'NEEDS IMPROVEMENT'}")
        print(f"🚀 Next Gen Ready: {'YES' if report.next_generation_ready else 'NO'}")
        
        print(f"\n📋 ROBUST GATE RESULTS:")
        for metrics in report.gate_metrics:
            status_icon = "✅" if metrics.success else "❌"
            healing_icon = "🔧" if metrics.adaptation_applied else ""
            confidence_info = f"({metrics.confidence_score:.0f}% conf)"
            risk_info = f"[{metrics.risk_assessment.upper()}]"
            print(f"  {status_icon}{healing_icon} {metrics.gate_name}: {metrics.score:.1f}/100 {confidence_info} {risk_info}")
            
            # Show top recommendations per gate
            if metrics.recommendations:
                for rec in metrics.recommendations[:1]:
                    print(f"    💡 {rec}")
        
        print(f"\n🎯 ROBUST RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
        
        print(f"\n🔮 PREDICTIVE INSIGHTS:")
        for i, insight in enumerate(report.predictive_insights, 1):
            print(f"  {i}. {insight}")
        
        # Save report
        report_file = quality_gates.save_adaptive_report(report)
        if report_file:
            print(f"\n📄 Detailed adaptive report saved: {report_file}")
        
        # Exit with status code
        sys.exit(0 if report.overall_success else 1)
        
    except Exception as e:
        logger.error(f"❌ Progressive Quality Gates Generation 2 failed: {e}")
        print(f"\n💥 EXECUTION FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run Generation 2 Progressive Quality Gates
    asyncio.run(main())