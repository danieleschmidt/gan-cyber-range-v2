"""
Comprehensive testing framework for GAN-Cyber-Range-v2.
Implements unit tests, integration tests, performance tests, and security tests.
"""

import logging
import unittest
import asyncio
import time
import threading
import json
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import traceback
import sys
import os

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of tests"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    LOAD = "load"
    CHAOS = "chaos"


class TestResult(Enum):
    """Test result status"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    """Individual test case"""
    test_id: str
    name: str
    description: str
    test_type: TestType
    test_function: Callable
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    timeout_seconds: int = 300
    expected_result: Any = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class TestExecution:
    """Test execution result"""
    test_case: TestCase
    result: TestResult
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    error_message: Optional[str] = None
    output: Optional[str] = None
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class TestRunner:
    """Advanced test runner with parallel execution"""
    
    def __init__(self, max_parallel_tests: int = 4):
        self.max_parallel_tests = max_parallel_tests
        self.test_cases = {}
        self.test_suites = {}
        self.execution_history = []
        self.setup_functions = []
        self.teardown_functions = []
    
    def register_test(self, test_case: TestCase):
        """Register a test case"""
        self.test_cases[test_case.test_id] = test_case
        logger.info(f"Registered test: {test_case.test_id}")
    
    def register_test_suite(self, suite_name: str, test_ids: List[str]):
        """Register a test suite"""
        self.test_suites[suite_name] = test_ids
        logger.info(f"Registered test suite: {suite_name} with {len(test_ids)} tests")
    
    def add_global_setup(self, setup_function: Callable):
        """Add global setup function"""
        self.setup_functions.append(setup_function)
    
    def add_global_teardown(self, teardown_function: Callable):
        """Add global teardown function"""
        self.teardown_functions.append(teardown_function)
    
    async def run_test(self, test_case: TestCase) -> TestExecution:
        """Run individual test case"""
        execution = TestExecution(
            test_case=test_case,
            result=TestResult.ERROR,
            start_time=datetime.now()
        )
        
        try:
            # Setup
            if test_case.setup_function:
                await self._run_with_timeout(test_case.setup_function, 60)
            
            # Execute test
            if asyncio.iscoroutinefunction(test_case.test_function):
                result = await self._run_with_timeout(
                    test_case.test_function, 
                    test_case.timeout_seconds
                )
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    test_case.test_function
                )
            
            # Validate result
            if test_case.expected_result is not None:
                if result == test_case.expected_result:
                    execution.result = TestResult.PASSED
                else:
                    execution.result = TestResult.FAILED
                    execution.error_message = f"Expected {test_case.expected_result}, got {result}"
            else:
                execution.result = TestResult.PASSED
            
        except asyncio.TimeoutError:
            execution.result = TestResult.FAILED
            execution.error_message = f"Test timed out after {test_case.timeout_seconds}s"
        except AssertionError as e:
            execution.result = TestResult.FAILED
            execution.error_message = str(e)
        except Exception as e:
            execution.result = TestResult.ERROR
            execution.error_message = f"{type(e).__name__}: {str(e)}"
            execution.output = traceback.format_exc()
        
        finally:
            # Teardown
            try:
                if test_case.teardown_function:
                    await self._run_with_timeout(test_case.teardown_function, 60)
            except Exception as e:
                logger.warning(f"Teardown failed for {test_case.test_id}: {e}")
            
            execution.end_time = datetime.now()
            execution.duration_ms = (execution.end_time - execution.start_time).total_seconds() * 1000
        
        return execution
    
    async def run_tests(self, test_ids: List[str] = None, parallel: bool = True) -> List[TestExecution]:
        """Run multiple tests"""
        if test_ids is None:
            test_ids = list(self.test_cases.keys())
        
        # Run global setup
        for setup_func in self.setup_functions:
            try:
                await self._run_with_timeout(setup_func, 120)
            except Exception as e:
                logger.error(f"Global setup failed: {e}")
                return []
        
        try:
            if parallel and len(test_ids) > 1:
                executions = await self._run_tests_parallel(test_ids)
            else:
                executions = await self._run_tests_sequential(test_ids)
        finally:
            # Run global teardown
            for teardown_func in self.teardown_functions:
                try:
                    await self._run_with_timeout(teardown_func, 120)
                except Exception as e:
                    logger.warning(f"Global teardown failed: {e}")
        
        self.execution_history.extend(executions)
        return executions
    
    async def _run_tests_parallel(self, test_ids: List[str]) -> List[TestExecution]:
        """Run tests in parallel"""
        semaphore = asyncio.Semaphore(self.max_parallel_tests)
        
        async def run_with_semaphore(test_id):
            async with semaphore:
                test_case = self.test_cases[test_id]
                return await self.run_test(test_case)
        
        tasks = [run_with_semaphore(test_id) for test_id in test_ids if test_id in self.test_cases]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    async def _run_tests_sequential(self, test_ids: List[str]) -> List[TestExecution]:
        """Run tests sequentially"""
        executions = []
        for test_id in test_ids:
            if test_id in self.test_cases:
                test_case = self.test_cases[test_id]
                execution = await self.run_test(test_case)
                executions.append(execution)
        return executions
    
    async def _run_with_timeout(self, func: Callable, timeout_seconds: int):
        """Run function with timeout"""
        if asyncio.iscoroutinefunction(func):
            return await asyncio.wait_for(func(), timeout=timeout_seconds)
        else:
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, func),
                timeout=timeout_seconds
            )
    
    def run_test_suite(self, suite_name: str, parallel: bool = True) -> List[TestExecution]:
        """Run a test suite"""
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite {suite_name} not found")
        
        test_ids = self.test_suites[suite_name]
        return asyncio.run(self.run_tests(test_ids, parallel))
    
    def generate_report(self, executions: List[TestExecution]) -> Dict[str, Any]:
        """Generate test report"""
        passed = sum(1 for e in executions if e.result == TestResult.PASSED)
        failed = sum(1 for e in executions if e.result == TestResult.FAILED)
        errors = sum(1 for e in executions if e.result == TestResult.ERROR)
        skipped = sum(1 for e in executions if e.result == TestResult.SKIPPED)
        
        total_duration = sum(e.duration_ms or 0 for e in executions)
        avg_duration = total_duration / len(executions) if executions else 0
        
        # Group by test type
        by_type = {}
        for execution in executions:
            test_type = execution.test_case.test_type.value
            if test_type not in by_type:
                by_type[test_type] = {'passed': 0, 'failed': 0, 'errors': 0, 'skipped': 0}
            
            by_type[test_type][execution.result.value] += 1
        
        report = {
            'summary': {
                'total_tests': len(executions),
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'skipped': skipped,
                'success_rate': passed / len(executions) if executions else 0,
                'total_duration_ms': total_duration,
                'average_duration_ms': avg_duration
            },
            'by_type': by_type,
            'failed_tests': [
                {
                    'test_id': e.test_case.test_id,
                    'name': e.test_case.name,
                    'error': e.error_message,
                    'duration_ms': e.duration_ms
                }
                for e in executions 
                if e.result in [TestResult.FAILED, TestResult.ERROR]
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        return report


class PerformanceTestFramework:
    """Framework for performance testing"""
    
    def __init__(self):
        self.benchmarks = {}
        self.performance_history = []
    
    def register_benchmark(self, name: str, function: Callable, **kwargs):
        """Register performance benchmark"""
        self.benchmarks[name] = {
            'function': function,
            'config': kwargs
        }
    
    async def run_performance_test(
        self,
        function: Callable,
        iterations: int = 100,
        concurrent_users: int = 1,
        ramp_up_seconds: int = 0
    ) -> Dict[str, Any]:
        """Run performance test"""
        results = {
            'function_name': function.__name__,
            'iterations': iterations,
            'concurrent_users': concurrent_users,
            'response_times': [],
            'errors': [],
            'start_time': datetime.now()
        }
        
        if concurrent_users == 1:
            # Sequential execution
            for i in range(iterations):
                start_time = time.time()
                try:
                    if asyncio.iscoroutinefunction(function):
                        await function()
                    else:
                        function()
                    response_time = (time.time() - start_time) * 1000
                    results['response_times'].append(response_time)
                except Exception as e:
                    results['errors'].append(str(e))
        else:
            # Concurrent execution
            semaphore = asyncio.Semaphore(concurrent_users)
            
            async def execute_with_timing():
                async with semaphore:
                    start_time = time.time()
                    try:
                        if asyncio.iscoroutinefunction(function):
                            await function()
                        else:
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, function)
                        return (time.time() - start_time) * 1000
                    except Exception as e:
                        return e
            
            # Ramp up gradually
            tasks = []
            for i in range(iterations):
                if ramp_up_seconds > 0:
                    await asyncio.sleep(ramp_up_seconds / iterations)
                tasks.append(execute_with_timing())
            
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in task_results:
                if isinstance(result, Exception):
                    results['errors'].append(str(result))
                else:
                    results['response_times'].append(result)
        
        results['end_time'] = datetime.now()
        
        # Calculate statistics
        if results['response_times']:
            response_times = sorted(results['response_times'])
            results['statistics'] = {
                'min_ms': min(response_times),
                'max_ms': max(response_times),
                'avg_ms': sum(response_times) / len(response_times),
                'median_ms': response_times[len(response_times) // 2],
                'p95_ms': response_times[int(len(response_times) * 0.95)],
                'p99_ms': response_times[int(len(response_times) * 0.99)],
                'throughput_req_per_sec': len(response_times) / (sum(response_times) / 1000) if sum(response_times) > 0 else 0
            }
        
        results['error_rate'] = len(results['errors']) / iterations
        
        self.performance_history.append(results)
        return results
    
    async def run_load_test(
        self,
        function: Callable,
        duration_seconds: int = 60,
        max_concurrent_users: int = 100,
        ramp_up_seconds: int = 10
    ) -> Dict[str, Any]:
        """Run load test with gradual ramp-up"""
        results = {
            'function_name': function.__name__,
            'duration_seconds': duration_seconds,
            'max_concurrent_users': max_concurrent_users,
            'ramp_up_seconds': ramp_up_seconds,
            'start_time': datetime.now(),
            'response_times_by_second': [],
            'error_counts_by_second': [],
            'active_users_by_second': []
        }
        
        # Track metrics per second
        metrics_lock = threading.Lock()
        response_times_current_second = []
        errors_current_second = 0
        active_users = 0
        
        def record_response_time(response_time_ms):
            nonlocal response_times_current_second
            with metrics_lock:
                response_times_current_second.append(response_time_ms)
        
        def record_error():
            nonlocal errors_current_second
            with metrics_lock:
                errors_current_second += 1
        
        def increment_active_users():
            nonlocal active_users
            with metrics_lock:
                active_users += 1
        
        def decrement_active_users():
            nonlocal active_users
            with metrics_lock:
                active_users -= 1
        
        # Metrics collection task
        async def collect_metrics():
            nonlocal response_times_current_second, errors_current_second
            
            for second in range(duration_seconds):
                await asyncio.sleep(1)
                
                with metrics_lock:
                    if response_times_current_second:
                        avg_response_time = sum(response_times_current_second) / len(response_times_current_second)
                    else:
                        avg_response_time = 0
                    
                    results['response_times_by_second'].append(avg_response_time)
                    results['error_counts_by_second'].append(errors_current_second)
                    results['active_users_by_second'].append(active_users)
                    
                    # Reset for next second
                    response_times_current_second = []
                    errors_current_second = 0
        
        # User simulation task
        async def simulate_user():
            increment_active_users()
            try:
                while True:
                    start_time = time.time()
                    try:
                        if asyncio.iscoroutinefunction(function):
                            await function()
                        else:
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, function)
                        
                        response_time = (time.time() - start_time) * 1000
                        record_response_time(response_time)
                        
                    except Exception as e:
                        record_error()
                    
                    # Small delay between requests
                    await asyncio.sleep(0.1)
            finally:
                decrement_active_users()
        
        # Start metrics collection
        metrics_task = asyncio.create_task(collect_metrics())
        
        # Gradually ramp up users
        user_tasks = []
        users_per_second = max_concurrent_users / ramp_up_seconds if ramp_up_seconds > 0 else max_concurrent_users
        
        for second in range(ramp_up_seconds):
            users_to_add = int(users_per_second * (second + 1)) - len(user_tasks)
            for _ in range(users_to_add):
                user_task = asyncio.create_task(simulate_user())
                user_tasks.append(user_task)
            
            await asyncio.sleep(1)
        
        # Wait for test duration
        await asyncio.sleep(duration_seconds - ramp_up_seconds)
        
        # Stop all user tasks
        for task in user_tasks:
            task.cancel()
        
        # Wait for metrics collection to complete
        await metrics_task
        
        results['end_time'] = datetime.now()
        
        # Calculate overall statistics
        all_response_times = [rt for rt in results['response_times_by_second'] if rt > 0]
        if all_response_times:
            results['overall_statistics'] = {
                'avg_response_time_ms': sum(all_response_times) / len(all_response_times),
                'min_response_time_ms': min(all_response_times),
                'max_response_time_ms': max(all_response_times),
                'total_errors': sum(results['error_counts_by_second']),
                'error_rate': sum(results['error_counts_by_second']) / (len(results['response_times_by_second']) * max_concurrent_users),
                'peak_concurrent_users': max(results['active_users_by_second']) if results['active_users_by_second'] else 0
            }
        
        return results


class SecurityTestFramework:
    """Framework for security testing"""
    
    def __init__(self):
        self.security_tests = {}
        self.vulnerability_scanner = self._create_vulnerability_scanner()
    
    def _create_vulnerability_scanner(self):
        """Create basic vulnerability scanner"""
        return {
            'sql_injection_patterns': [
                "' OR '1'='1",
                "'; DROP TABLE users; --",
                "' UNION SELECT * FROM users --"
            ],
            'xss_patterns': [
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "<img src=x onerror=alert('xss')>"
            ],
            'command_injection_patterns': [
                "; rm -rf /",
                "| cat /etc/passwd",
                "&& whoami"
            ]
        }
    
    def register_security_test(self, name: str, test_function: Callable):
        """Register security test"""
        self.security_tests[name] = test_function
    
    async def run_input_validation_test(self, validation_function: Callable) -> Dict[str, Any]:
        """Test input validation against common attacks"""
        results = {
            'test_name': 'input_validation',
            'function_name': validation_function.__name__,
            'vulnerabilities_found': [],
            'tests_passed': 0,
            'tests_failed': 0
        }
        
        # Test SQL injection patterns
        for pattern in self.vulnerability_scanner['sql_injection_patterns']:
            try:
                is_safe = validation_function(pattern)
                if is_safe:
                    results['vulnerabilities_found'].append({
                        'type': 'sql_injection',
                        'pattern': pattern,
                        'description': 'Function incorrectly validated malicious SQL injection pattern'
                    })
                    results['tests_failed'] += 1
                else:
                    results['tests_passed'] += 1
            except Exception as e:
                results['tests_passed'] += 1  # Exception is good for malicious input
        
        # Test XSS patterns
        for pattern in self.vulnerability_scanner['xss_patterns']:
            try:
                is_safe = validation_function(pattern)
                if is_safe:
                    results['vulnerabilities_found'].append({
                        'type': 'xss',
                        'pattern': pattern,
                        'description': 'Function incorrectly validated malicious XSS pattern'
                    })
                    results['tests_failed'] += 1
                else:
                    results['tests_passed'] += 1
            except Exception as e:
                results['tests_passed'] += 1
        
        # Test command injection patterns
        for pattern in self.vulnerability_scanner['command_injection_patterns']:
            try:
                is_safe = validation_function(pattern)
                if is_safe:
                    results['vulnerabilities_found'].append({
                        'type': 'command_injection',
                        'pattern': pattern,
                        'description': 'Function incorrectly validated malicious command injection pattern'
                    })
                    results['tests_failed'] += 1
                else:
                    results['tests_passed'] += 1
            except Exception as e:
                results['tests_passed'] += 1
        
        results['security_score'] = results['tests_passed'] / (results['tests_passed'] + results['tests_failed']) if (results['tests_passed'] + results['tests_failed']) > 0 else 0
        
        return results
    
    async def run_authentication_test(self, auth_function: Callable) -> Dict[str, Any]:
        """Test authentication mechanisms"""
        results = {
            'test_name': 'authentication',
            'function_name': auth_function.__name__,
            'vulnerabilities_found': [],
            'tests_passed': 0,
            'tests_failed': 0
        }
        
        # Test weak passwords
        weak_passwords = ['password', '123456', 'admin', '', 'password123']
        for password in weak_passwords:
            try:
                is_accepted = auth_function('test_user', password)
                if is_accepted:
                    results['vulnerabilities_found'].append({
                        'type': 'weak_password',
                        'password': password,
                        'description': f'Weak password "{password}" was accepted'
                    })
                    results['tests_failed'] += 1
                else:
                    results['tests_passed'] += 1
            except Exception as e:
                results['tests_passed'] += 1
        
        # Test SQL injection in authentication
        malicious_usernames = ["admin'--", "' OR '1'='1'--", "admin'; DROP TABLE users; --"]
        for username in malicious_usernames:
            try:
                is_accepted = auth_function(username, 'anypassword')
                if is_accepted:
                    results['vulnerabilities_found'].append({
                        'type': 'sql_injection_auth',
                        'username': username,
                        'description': f'SQL injection in username was successful: {username}'
                    })
                    results['tests_failed'] += 1
                else:
                    results['tests_passed'] += 1
            except Exception as e:
                results['tests_passed'] += 1
        
        results['security_score'] = results['tests_passed'] / (results['tests_passed'] + results['tests_failed']) if (results['tests_passed'] + results['tests_failed']) > 0 else 0
        
        return results


# Global test framework instances
test_runner = TestRunner()
performance_framework = PerformanceTestFramework()
security_framework = SecurityTestFramework()


# Test decorators
def test_case(test_id: str, name: str, test_type: TestType = TestType.UNIT, **kwargs):
    """Decorator to register test case"""
    def decorator(func):
        test_case_obj = TestCase(
            test_id=test_id,
            name=name,
            test_type=test_type,
            test_function=func,
            **kwargs
        )
        test_runner.register_test(test_case_obj)
        return func
    return decorator


def performance_test(iterations: int = 100, concurrent_users: int = 1):
    """Decorator for performance testing"""
    def decorator(func):
        async def wrapper():
            return await performance_framework.run_performance_test(
                func, iterations=iterations, concurrent_users=concurrent_users
            )
        return wrapper
    return decorator


def security_test(test_type: str = "input_validation"):
    """Decorator for security testing"""
    def decorator(func):
        async def wrapper():
            if test_type == "input_validation":
                return await security_framework.run_input_validation_test(func)
            elif test_type == "authentication":
                return await security_framework.run_authentication_test(func)
            else:
                raise ValueError(f"Unknown security test type: {test_type}")
        return wrapper
    return decorator