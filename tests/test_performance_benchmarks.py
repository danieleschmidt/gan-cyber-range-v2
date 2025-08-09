"""
Performance benchmarking tests for GAN-Cyber-Range-v2 components.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
import statistics

from gan_cyber_range.factories.attack_factory import AttackFactory, AttackConfig
from gan_cyber_range.factories.range_factory import CyberRangeFactory
from gan_cyber_range.factories.network_factory import NetworkFactory
from gan_cyber_range.orchestration.workflow_engine import WorkflowEngine, Workflow, WorkflowStep
from gan_cyber_range.orchestration.pipeline_manager import PipelineManager, PipelineType
from gan_cyber_range.utils.security import SecurityManager


class TestPerformanceBenchmarks:
    """Performance benchmark test suite"""
    
    def setup_method(self):
        """Set up performance test environment"""
        self.security_manager = Mock(spec=SecurityManager)
        self.security_manager.validate_use_case.return_value = True
        self.security_manager.check_clearance_level.return_value = True
        
    def test_attack_factory_creation_performance(self):
        """Benchmark attack factory object creation performance"""
        attack_factory = AttackFactory(self.security_manager)
        
        # Benchmark GAN creation
        gan_creation_times = []
        
        with patch('gan_cyber_range.factories.attack_factory.AttackGAN') as mock_gan:
            mock_gan_instance = Mock()
            mock_gan.return_value = mock_gan_instance
            
            for _ in range(100):
                start_time = time.perf_counter()
                attack_factory.create_attack_gan()
                end_time = time.perf_counter()
                gan_creation_times.append(end_time - start_time)
                
        # Performance assertions
        avg_creation_time = statistics.mean(gan_creation_times)
        max_creation_time = max(gan_creation_times)
        
        assert avg_creation_time < 0.01  # Less than 10ms average
        assert max_creation_time < 0.05  # Less than 50ms maximum
        assert len(gan_creation_times) == 100
        
        # Test caching performance improvement
        cache_hit_times = []
        
        for _ in range(50):
            start_time = time.perf_counter()
            attack_factory.create_attack_gan()  # Should hit cache
            end_time = time.perf_counter()
            cache_hit_times.append(end_time - start_time)
            
        avg_cache_hit_time = statistics.mean(cache_hit_times)
        
        # Cache hits should be significantly faster
        assert avg_cache_hit_time < avg_creation_time / 2
        assert mock_gan.call_count == 1  # Only called once due to caching
        
    def test_network_factory_scaling_performance(self):
        """Benchmark network topology creation at different scales"""
        network_factory = NetworkFactory()
        
        scale_factors = [0.5, 1.0, 2.0, 5.0, 10.0]
        creation_times = {}
        
        with patch('gan_cyber_range.factories.network_factory.NetworkTopology') as mock_topology:
            mock_topology.generate.return_value = Mock()
            
            for scale in scale_factors:
                times = []
                for _ in range(10):  # 10 runs per scale
                    start_time = time.perf_counter()
                    network_factory.create_from_template("enterprise", scale_factor=scale)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                    
                creation_times[scale] = {
                    "avg": statistics.mean(times),
                    "max": max(times),
                    "min": min(times)
                }
                
        # Performance scaling analysis
        for scale in scale_factors:
            avg_time = creation_times[scale]["avg"]
            max_time = creation_times[scale]["max"]
            
            # Base performance requirements
            assert avg_time < 0.1  # Less than 100ms average
            assert max_time < 0.5  # Less than 500ms maximum
            
        # Verify reasonable scaling (not exponential)
        small_scale_time = creation_times[1.0]["avg"]
        large_scale_time = creation_times[10.0]["avg"]
        
        # 10x scale should not be more than 5x slower
        assert large_scale_time < small_scale_time * 5
        
    @pytest.mark.asyncio
    async def test_workflow_engine_concurrent_execution_performance(self):
        """Benchmark concurrent workflow execution performance"""
        workflow_engine = WorkflowEngine()
        
        # Create simple test workflow
        async def fast_handler(context):
            await asyncio.sleep(0.001)  # Simulate minimal work
            return {"result": "success"}
            
        workflow = Workflow("Performance Test", "Test workflow for performance")
        workflow.add_step(WorkflowStep("fast_step", "Fast Step", "Fast step", fast_handler))
        
        workflow_id = workflow_engine.register_workflow(workflow)
        
        # Benchmark sequential execution
        sequential_start = time.perf_counter()
        sequential_results = []
        
        for _ in range(10):
            result = await workflow_engine.execute_workflow(workflow_id)
            sequential_results.append(result)
            
        sequential_end = time.perf_counter()
        sequential_time = sequential_end - sequential_start
        
        # Benchmark concurrent execution
        concurrent_start = time.perf_counter()
        
        concurrent_tasks = []
        for _ in range(10):
            task = workflow_engine.execute_workflow(workflow_id)
            concurrent_tasks.append(task)
            
        concurrent_results = await asyncio.gather(*concurrent_tasks)
        
        concurrent_end = time.perf_counter()
        concurrent_time = concurrent_end - concurrent_start
        
        # Performance analysis
        assert len(sequential_results) == 10
        assert len(concurrent_results) == 10
        assert all(r.status.value == "completed" for r in sequential_results)
        assert all(r.status.value == "completed" for r in concurrent_results)
        
        # Concurrent execution should be significantly faster
        assert concurrent_time < sequential_time * 0.5  # At least 50% faster
        assert concurrent_time < 1.0  # Less than 1 second total
        
    def test_cyber_range_factory_memory_efficiency(self):
        """Test memory efficiency of cyber range factory"""
        range_factory = CyberRangeFactory(self.security_manager)
        
        # Monitor memory usage during range creation
        initial_range_count = len(range_factory.get_active_ranges())
        
        with patch('gan_cyber_range.factories.range_factory.CyberRange') as mock_range:
            mock_range_instances = []
            
            def create_mock_range(config):
                mock_instance = Mock()
                mock_instance.config = config
                mock_instances.append(mock_instance)
                return mock_instance
                
            mock_range.side_effect = create_mock_range
            
            # Create multiple ranges
            range_ids = []
            for i in range(20):
                range_id = f"test_range_{i}"
                range_factory.create_from_template(
                    "educational_basic",
                    {"name_suffix": str(i)}
                )
                range_ids.append(range_id)
                
            # Verify ranges are tracked efficiently
            active_ranges = range_factory.get_active_ranges()
            assert len(active_ranges) == 20
            
            # Test cleanup efficiency
            cleanup_start = time.perf_counter()
            range_factory.shutdown_all_ranges()
            cleanup_end = time.perf_counter()
            
            cleanup_time = cleanup_end - cleanup_start
            assert cleanup_time < 1.0  # Less than 1 second for cleanup
            assert len(range_factory.get_active_ranges()) == 0
            
    @pytest.mark.asyncio
    async def test_pipeline_manager_throughput_performance(self):
        """Benchmark pipeline manager throughput"""
        workflow_engine = Mock()
        pipeline_manager = PipelineManager(workflow_engine)
        
        # Mock workflow execution
        workflow_engine.register_workflow.return_value = "test_workflow"
        
        mock_workflow_result = Mock()
        mock_workflow_result.status.value = "completed"
        mock_workflow_result.results = {"stage_01": {"result": "success"}}
        
        workflow_engine.execute_workflow = Mock(return_value=mock_workflow_result)
        
        # Mock resource availability
        pipeline_manager._check_resource_availability = Mock(return_value=True)
        pipeline_manager._allocate_pipeline_resources = Mock()
        pipeline_manager._release_pipeline_resources = Mock()
        
        # Create pipeline
        pipeline_id = pipeline_manager.create_pipeline(
            PipelineType.DATA_PROCESSING,
            "Throughput Test Pipeline"
        )
        
        # Benchmark pipeline creation throughput
        creation_start = time.perf_counter()
        
        pipeline_ids = []
        for i in range(50):
            pid = pipeline_manager.create_pipeline(
                PipelineType.THREAT_ANALYSIS,
                f"Pipeline {i}"
            )
            pipeline_ids.append(pid)
            
        creation_end = time.perf_counter()
        creation_time = creation_end - creation_start
        
        # Performance requirements
        assert creation_time < 2.0  # Less than 2 seconds for 50 pipelines
        assert len(pipeline_ids) == 50
        
        creation_rate = 50 / creation_time
        assert creation_rate > 25  # At least 25 pipelines per second
        
    def test_training_factory_program_generation_performance(self):
        """Benchmark training program generation performance"""
        from gan_cyber_range.factories.training_factory import TrainingFactory, SkillLevel, TrainingDomain
        
        training_factory = TrainingFactory()
        
        # Benchmark comprehensive program creation
        program_creation_times = []
        
        for i in range(20):
            start_time = time.perf_counter()
            
            program = training_factory.create_comprehensive_program(
                program_name=f"Test Program {i}",
                target_audience="professionals",
                domains=[TrainingDomain.INCIDENT_RESPONSE, TrainingDomain.THREAT_HUNTING],
                skill_level=SkillLevel.INTERMEDIATE
            )
            
            end_time = time.perf_counter()
            program_creation_times.append(end_time - start_time)
            
        # Performance analysis
        avg_creation_time = statistics.mean(program_creation_times)
        max_creation_time = max(program_creation_times)
        
        assert avg_creation_time < 0.1  # Less than 100ms average
        assert max_creation_time < 0.5  # Less than 500ms maximum
        
        # Benchmark adaptive learning path creation
        learner_profile = {
            "id": "test_learner",
            "skill_level": "intermediate",
            "domains": [TrainingDomain.INCIDENT_RESPONSE],
            "weekly_hours": 10,
            "learning_style": "hands_on"
        }
        
        path_creation_times = []
        
        for i in range(15):
            start_time = time.perf_counter()
            
            path = training_factory.create_adaptive_learning_path(
                learner_profile,
                ["improve_skills", "certification_prep"]
            )
            
            end_time = time.perf_counter()
            path_creation_times.append(end_time - start_time)
            
        avg_path_time = statistics.mean(path_creation_times)
        assert avg_path_time < 0.05  # Less than 50ms average
        
    def test_scenario_orchestrator_event_processing_performance(self):
        """Benchmark scenario event processing performance"""
        from gan_cyber_range.orchestration.scenario_orchestrator import ScenarioOrchestrator, ScenarioType
        
        workflow_engine = Mock()
        attack_factory = Mock()
        range_factory = Mock()
        
        orchestrator = ScenarioOrchestrator(workflow_engine, attack_factory, range_factory)
        
        # Create scenario
        scenario = orchestrator.create_scenario(
            ScenarioType.INCIDENT_RESPONSE,
            "Performance Test Scenario"
        )
        
        # Create mock execution
        execution_id = "perf_test_execution"
        execution = Mock()
        execution.events = []
        execution.real_time_feedback = {}
        
        orchestrator.active_executions[execution_id] = execution
        orchestrator._process_injected_event = Mock()
        
        # Benchmark event injection performance
        event_injection_times = []
        
        for i in range(100):
            start_time = time.perf_counter()
            
            # Simulate async call with mock
            asyncio.create_task(orchestrator.inject_scenario_event(
                execution_id,
                f"test_event_{i}",
                {"data": f"test_data_{i}"}
            ))
            
            end_time = time.perf_counter()
            event_injection_times.append(end_time - start_time)
            
        avg_injection_time = statistics.mean(event_injection_times)
        max_injection_time = max(event_injection_times)
        
        # Performance requirements for real-time event processing
        assert avg_injection_time < 0.01  # Less than 10ms average
        assert max_injection_time < 0.05  # Less than 50ms maximum
        
        # Verify all events were queued
        assert len(execution.events) == 100
        
    def test_security_validation_performance(self):
        """Benchmark security validation performance"""
        security_manager = SecurityManager()
        
        # Benchmark use case validation
        validation_times = []
        
        test_cases = [
            ("research", "attack_generation"),
            ("training", "scenario_execution"), 
            ("education", "cyber_range_deployment"),
            ("assessment", "threat_hunting"),
            ("development", "security_testing")
        ]
        
        for _ in range(50):  # 50 iterations
            for use_case, activity in test_cases:
                start_time = time.perf_counter()
                
                try:
                    result = security_manager.validate_use_case(use_case, activity)
                except:
                    result = False  # Handle any validation errors
                    
                end_time = time.perf_counter()
                validation_times.append(end_time - start_time)
                
        # Performance analysis
        avg_validation_time = statistics.mean(validation_times)
        max_validation_time = max(validation_times)
        
        # Security validation should be very fast to not impact performance
        assert avg_validation_time < 0.001  # Less than 1ms average
        assert max_validation_time < 0.01   # Less than 10ms maximum
        
        # Benchmark batch validation
        batch_start = time.perf_counter()
        
        batch_results = []
        for use_case, activity in test_cases * 20:  # 100 total validations
            try:
                result = security_manager.validate_use_case(use_case, activity)
                batch_results.append(result)
            except:
                batch_results.append(False)
                
        batch_end = time.perf_counter()
        batch_time = batch_end - batch_start
        
        assert batch_time < 0.1  # Less than 100ms for 100 validations
        assert len(batch_results) == 100
        
    def test_component_integration_performance(self):
        """Benchmark integrated component performance"""
        # Setup integrated environment
        security_manager = Mock(spec=SecurityManager)
        security_manager.validate_use_case.return_value = True
        security_manager.check_clearance_level.return_value = True
        
        attack_factory = AttackFactory(security_manager)
        range_factory = CyberRangeFactory(security_manager)
        network_factory = NetworkFactory()
        
        # Benchmark end-to-end scenario setup
        scenario_setup_times = []
        
        with patch('gan_cyber_range.factories.attack_factory.AttackGAN'), \
             patch('gan_cyber_range.factories.attack_factory.RedTeamLLM'), \
             patch('gan_cyber_range.factories.range_factory.CyberRange'), \
             patch('gan_cyber_range.factories.network_factory.NetworkTopology'):
            
            for i in range(10):
                start_time = time.perf_counter()
                
                # Simulate complete scenario setup
                # 1. Create network topology
                topology = network_factory.create_from_template("enterprise")
                
                # 2. Create attack components
                attack_config = AttackConfig(attack_types=["network", "web"])
                attack_gan = attack_factory.create_attack_gan(attack_config)
                red_team_llm = attack_factory.create_red_team_llm(attack_config)
                
                # 3. Create cyber range
                cyber_range = range_factory.create_from_template("professional_training")
                
                end_time = time.perf_counter()
                scenario_setup_times.append(end_time - start_time)
                
        # Performance requirements for integrated setup
        avg_setup_time = statistics.mean(scenario_setup_times)
        max_setup_time = max(scenario_setup_times)
        
        assert avg_setup_time < 0.5  # Less than 500ms average
        assert max_setup_time < 1.0  # Less than 1 second maximum
        
        # Test teardown performance
        teardown_start = time.perf_counter()
        
        # Simulate cleanup
        attack_factory.clear_cache()
        range_factory.shutdown_all_ranges()
        
        teardown_end = time.perf_counter()
        teardown_time = teardown_end - teardown_start
        
        assert teardown_time < 0.1  # Less than 100ms for cleanup
        
    def test_memory_usage_patterns(self):
        """Test memory usage patterns under load"""
        import sys
        
        # Get initial memory baseline
        initial_objects = len(gc.get_objects()) if 'gc' in sys.modules else 0
        
        # Create components and monitor memory
        security_manager = Mock(spec=SecurityManager)
        security_manager.validate_use_case.return_value = True
        
        components = []
        
        with patch('gan_cyber_range.factories.attack_factory.AttackGAN'), \
             patch('gan_cyber_range.factories.range_factory.CyberRange'):
            
            # Create many components
            for i in range(50):
                attack_factory = AttackFactory(security_manager)
                range_factory = CyberRangeFactory(security_manager)
                
                # Create objects
                attack_factory.create_attack_gan()
                range_factory.create_from_template("educational_basic")
                
                components.extend([attack_factory, range_factory])
                
        # Check memory growth
        if 'gc' in sys.modules:
            import gc
            current_objects = len(gc.get_objects())
            memory_growth = current_objects - initial_objects
            
            # Memory growth should be reasonable
            assert memory_growth < 10000  # Less than 10k new objects
            
        # Test cleanup
        components.clear()
        
        if 'gc' in sys.modules:
            import gc
            gc.collect()  # Force garbage collection
            
        # Memory should be manageable after cleanup
        assert len(components) == 0


# Import gc module at the top level for memory testing
try:
    import gc
except ImportError:
    gc = None


class TestConcurrencyPerformance:
    """Test performance under concurrent load"""
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution_scaling(self):
        """Test workflow engine performance under concurrent load"""
        workflow_engine = WorkflowEngine()
        
        # Create lightweight test workflow
        async def concurrent_handler(context):
            await asyncio.sleep(0.01)  # 10ms simulated work
            return {"worker_id": context.get("worker_id", "unknown")}
            
        workflow = Workflow("Concurrent Test", "Concurrent execution test")
        workflow.add_step(WorkflowStep(
            "concurrent_step", 
            "Concurrent Step", 
            "Concurrent test step", 
            concurrent_handler
        ))
        
        workflow_id = workflow_engine.register_workflow(workflow)
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        performance_results = {}
        
        for concurrency in concurrency_levels:
            start_time = time.perf_counter()
            
            # Create concurrent tasks
            tasks = []
            for i in range(concurrency):
                task = workflow_engine.execute_workflow(
                    workflow_id,
                    initial_variables={"worker_id": f"worker_{i}"}
                )
                tasks.append(task)
                
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            performance_results[concurrency] = {
                "execution_time": execution_time,
                "throughput": concurrency / execution_time,
                "success_count": sum(1 for r in results if r.status.value == "completed")
            }
            
        # Performance analysis
        for concurrency in concurrency_levels:
            result = performance_results[concurrency]
            
            # All executions should succeed
            assert result["success_count"] == concurrency
            
            # Execution time should scale reasonably
            assert result["execution_time"] < concurrency * 0.05  # Better than sequential
            
            # Throughput should be reasonable
            assert result["throughput"] > 10  # At least 10 workflows per second
            
    @pytest.mark.asyncio
    async def test_concurrent_scenario_monitoring(self):
        """Test scenario monitoring under concurrent access"""
        from gan_cyber_range.orchestration.scenario_orchestrator import ScenarioOrchestrator, ScenarioType
        
        workflow_engine = Mock()
        attack_factory = Mock()
        range_factory = Mock()
        
        orchestrator = ScenarioOrchestrator(workflow_engine, attack_factory, range_factory)
        
        # Create multiple active executions
        execution_ids = []
        for i in range(10):
            scenario = orchestrator.create_scenario(
                ScenarioType.INCIDENT_RESPONSE,
                f"Concurrent Scenario {i}"
            )
            
            execution_id = f"concurrent_execution_{i}"
            execution = Mock()
            execution.scenario_id = scenario.id
            execution.current_phase = Mock()
            execution.current_phase.value = "execution"
            execution.phase_start_time = time.time()
            execution.start_time = time.time()
            execution.participants = [{"id": f"user_{j}"} for j in range(3)]
            execution.events = []
            
            orchestrator.active_executions[execution_id] = execution
            execution_ids.append(execution_id)
            
        # Mock monitoring methods
        orchestrator._collect_performance_metrics = AsyncMock(return_value={"metrics": True})
        orchestrator._assess_learning_progress = AsyncMock(return_value={"progress": True})
        orchestrator._get_environment_status = AsyncMock(return_value={"status": "healthy"})
        
        # Test concurrent monitoring
        monitoring_start = time.perf_counter()
        
        monitoring_tasks = []
        for execution_id in execution_ids:
            task = orchestrator.monitor_scenario_execution(execution_id)
            monitoring_tasks.append(task)
            
        monitoring_results = await asyncio.gather(*monitoring_tasks)
        
        monitoring_end = time.perf_counter()
        monitoring_time = monitoring_end - monitoring_start
        
        # Performance requirements
        assert len(monitoring_results) == 10
        assert monitoring_time < 1.0  # Less than 1 second for 10 concurrent monitors
        
        # All monitoring should succeed
        assert all("execution_id" in result for result in monitoring_results)
        
        monitoring_throughput = 10 / monitoring_time
        assert monitoring_throughput > 10  # At least 10 monitors per second