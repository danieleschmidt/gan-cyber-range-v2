"""
Performance Benchmark Suite for GAN-Cyber-Range-v2

Comprehensive performance testing and benchmarking system that measures:
- Attack generation performance
- API response times
- Memory usage patterns
- Concurrent operation handling
- Scalability metrics
"""

import sys
import time
import threading
import multiprocessing as mp
import psutil
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Import our demo system
sys.path.append(str(Path(__file__).parent / "gan_cyber_range"))
from demo import LightweightAttackGenerator, SimpleCyberRange, DemoAPI


@dataclass
class BenchmarkResult:
    test_name: str
    duration: float
    operations_per_second: float
    memory_usage_mb: float
    cpu_percent: float
    success_rate: float
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    results: List[BenchmarkResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = None
    total_duration: float = 0.0


class PerformanceBenchmark:
    """Comprehensive performance benchmarking system"""
    
    def __init__(self):
        self.suite = BenchmarkSuite()
        self.process = psutil.Process()
        
    def run_all_benchmarks(self) -> BenchmarkSuite:
        """Run complete benchmark suite"""
        print("🚀 Starting Performance Benchmark Suite")
        print("="*60)
        
        benchmarks = [
            ("Attack Generation Speed", self.benchmark_attack_generation),
            ("Attack Generation Scale", self.benchmark_attack_scale),
            ("Cyber Range Operations", self.benchmark_cyber_range),
            ("API Performance", self.benchmark_api_performance),
            ("Memory Usage", self.benchmark_memory_usage),
            ("Concurrent Operations", self.benchmark_concurrent_operations),
            ("Threading Performance", self.benchmark_threading),
            ("Process Pool Performance", self.benchmark_process_pool),
            ("Stress Test", self.benchmark_stress_test)
        ]
        
        for test_name, benchmark_func in benchmarks:
            print(f"\n🔬 Running {test_name}...")
            try:
                result = benchmark_func()
                self.suite.results.append(result)
                print(f"✅ {test_name}: {result.operations_per_second:.1f} ops/sec")
            except Exception as e:
                print(f"❌ {test_name} failed: {e}")
        
        self.suite.end_time = datetime.now()
        self.suite.total_duration = (self.suite.end_time - self.suite.start_time).total_seconds()
        
        return self.suite
    
    def benchmark_attack_generation(self) -> BenchmarkResult:
        """Benchmark attack generation speed"""
        generator = LightweightAttackGenerator()
        
        # Warm-up
        generator.generate_batch(10)
        
        # Measure generation speed
        start_memory = self.process.memory_info().rss / 1024 / 1024
        start_cpu = self.process.cpu_percent()
        
        start_time = time.time()
        attacks = generator.generate_batch(1000)
        end_time = time.time()
        
        end_memory = self.process.memory_info().rss / 1024 / 1024
        end_cpu = self.process.cpu_percent()
        
        duration = end_time - start_time
        ops_per_sec = len(attacks) / duration
        
        # Validate attack quality
        attack_types = set(attack.attack_type for attack in attacks)
        unique_payloads = len(set(attack.payload for attack in attacks))
        
        return BenchmarkResult(
            test_name="Attack Generation Speed",
            duration=duration,
            operations_per_second=ops_per_sec,
            memory_usage_mb=end_memory - start_memory,
            cpu_percent=(start_cpu + end_cpu) / 2,
            success_rate=1.0,
            additional_metrics={
                "attack_types": len(attack_types),
                "unique_payloads": unique_payloads,
                "diversity_ratio": unique_payloads / len(attacks)
            }
        )
    
    def benchmark_attack_scale(self) -> BenchmarkResult:
        """Benchmark attack generation scalability"""
        generator = LightweightAttackGenerator()
        
        batch_sizes = [10, 50, 100, 500, 1000, 5000]
        results = []
        
        for batch_size in batch_sizes:
            start_time = time.time()
            attacks = generator.generate_batch(batch_size)
            duration = time.time() - start_time
            
            ops_per_sec = len(attacks) / duration
            results.append((batch_size, ops_per_sec, duration))
        
        # Calculate average performance
        avg_ops_per_sec = statistics.mean(result[1] for result in results)
        total_duration = sum(result[2] for result in results)
        
        return BenchmarkResult(
            test_name="Attack Generation Scale",
            duration=total_duration,
            operations_per_second=avg_ops_per_sec,
            memory_usage_mb=self.process.memory_info().rss / 1024 / 1024,
            cpu_percent=self.process.cpu_percent(),
            success_rate=1.0,
            additional_metrics={
                "batch_results": results,
                "max_ops_per_sec": max(result[1] for result in results),
                "min_ops_per_sec": min(result[1] for result in results)
            }
        )
    
    def benchmark_cyber_range(self) -> BenchmarkResult:
        """Benchmark cyber range operations"""
        operations = []
        successful_ops = 0
        
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Test multiple range operations
        for i in range(10):
            try:
                # Create range
                op_start = time.time()
                cyber_range = SimpleCyberRange(f"benchmark-{i}")
                range_id = cyber_range.deploy()
                
                # Execute attacks
                for _ in range(5):
                    attack = cyber_range.attack_generator.generate_attack()
                    result = cyber_range.execute_attack(attack)
                
                # Get metrics
                metrics = cyber_range.get_metrics()
                
                op_duration = time.time() - op_start
                operations.append(op_duration)
                successful_ops += 1
                
            except Exception:
                operations.append(0)
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        
        total_duration = end_time - start_time
        ops_per_sec = successful_ops / total_duration
        success_rate = successful_ops / len(operations)
        
        return BenchmarkResult(
            test_name="Cyber Range Operations",
            duration=total_duration,
            operations_per_second=ops_per_sec,
            memory_usage_mb=end_memory - start_memory,
            cpu_percent=self.process.cpu_percent(),
            success_rate=success_rate,
            additional_metrics={
                "avg_operation_time": statistics.mean(op for op in operations if op > 0),
                "successful_operations": successful_ops,
                "total_operations": len(operations)
            }
        )
    
    def benchmark_api_performance(self) -> BenchmarkResult:
        """Benchmark API performance"""
        api = DemoAPI()
        response_times = []
        successful_requests = 0
        
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Test API operations
        for i in range(50):
            try:
                # Create range
                req_start = time.time()
                response = api.create_range(f"api-test-{i}")
                req_time = time.time() - req_start
                response_times.append(req_time)
                
                range_id = response['range_id']
                
                # Generate attacks
                req_start = time.time()
                api.generate_attacks(range_id, count=5)
                req_time = time.time() - req_start
                response_times.append(req_time)
                
                # Get range info
                req_start = time.time()
                api.get_range_info(range_id)
                req_time = time.time() - req_start
                response_times.append(req_time)
                
                successful_requests += 3
                
            except Exception:
                pass
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        
        total_duration = end_time - start_time
        requests_per_sec = successful_requests / total_duration
        success_rate = successful_requests / (50 * 3)  # 3 requests per iteration
        
        return BenchmarkResult(
            test_name="API Performance",
            duration=total_duration,
            operations_per_second=requests_per_sec,
            memory_usage_mb=end_memory - start_memory,
            cpu_percent=self.process.cpu_percent(),
            success_rate=success_rate,
            additional_metrics={
                "avg_response_time": statistics.mean(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0,
                "min_response_time": min(response_times) if response_times else 0,
                "p95_response_time": self._percentile(response_times, 95) if response_times else 0
            }
        )
    
    def benchmark_memory_usage(self) -> BenchmarkResult:
        """Benchmark memory usage patterns"""
        generator = LightweightAttackGenerator()
        api = DemoAPI()
        
        memory_samples = []
        start_time = time.time()
        
        # Sample memory usage over time
        for i in range(20):
            # Create some load
            attacks = generator.generate_batch(100)
            range_response = api.create_range(f"memory-test-{i}")
            api.generate_attacks(range_response['range_id'], count=10)
            
            # Sample memory
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            memory_samples.append(memory_mb)
            
            time.sleep(0.1)  # Small delay
        
        duration = time.time() - start_time
        
        return BenchmarkResult(
            test_name="Memory Usage",
            duration=duration,
            operations_per_second=20 / duration,
            memory_usage_mb=max(memory_samples) - min(memory_samples),
            cpu_percent=self.process.cpu_percent(),
            success_rate=1.0,
            additional_metrics={
                "memory_samples": memory_samples,
                "avg_memory": statistics.mean(memory_samples),
                "max_memory": max(memory_samples),
                "min_memory": min(memory_samples),
                "memory_growth": memory_samples[-1] - memory_samples[0]
            }
        )
    
    def benchmark_concurrent_operations(self) -> BenchmarkResult:
        """Benchmark concurrent operations"""
        api = DemoAPI()
        results = []
        
        def worker(worker_id):
            try:
                start_time = time.time()
                
                # Create range
                response = api.create_range(f"concurrent-{worker_id}")
                range_id = response['range_id']
                
                # Generate attacks
                api.generate_attacks(range_id, count=10)
                
                # Get metrics
                api.get_range_info(range_id)
                
                duration = time.time() - start_time
                results.append((worker_id, duration, True))
                
            except Exception as e:
                results.append((worker_id, 0, False))
        
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(20)]
            for future in futures:
                future.result()  # Wait for completion
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        
        successful_workers = sum(1 for _, _, success in results if success)
        total_duration = end_time - start_time
        
        return BenchmarkResult(
            test_name="Concurrent Operations",
            duration=total_duration,
            operations_per_second=successful_workers / total_duration,
            memory_usage_mb=end_memory - start_memory,
            cpu_percent=self.process.cpu_percent(),
            success_rate=successful_workers / len(results),
            additional_metrics={
                "concurrent_workers": 20,
                "successful_workers": successful_workers,
                "avg_worker_time": statistics.mean(duration for _, duration, success in results if success) if successful_workers > 0 else 0
            }
        )
    
    def benchmark_threading(self) -> BenchmarkResult:
        """Benchmark threading performance"""
        generator = LightweightAttackGenerator()
        
        def generate_attacks(count):
            return generator.generate_batch(count)
        
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Test with different thread counts
        thread_counts = [1, 2, 4, 8]
        results = []
        
        for thread_count in thread_counts:
            threads_start = time.time()
            
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(generate_attacks, 50) for _ in range(thread_count)]
                thread_results = [future.result() for future in futures]
            
            threads_duration = time.time() - threads_start
            total_attacks = sum(len(result) for result in thread_results)
            threads_ops_per_sec = total_attacks / threads_duration
            
            results.append((thread_count, threads_ops_per_sec))
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        
        total_duration = end_time - start_time
        best_performance = max(result[1] for result in results)
        
        return BenchmarkResult(
            test_name="Threading Performance",
            duration=total_duration,
            operations_per_second=best_performance,
            memory_usage_mb=end_memory - start_memory,
            cpu_percent=self.process.cpu_percent(),
            success_rate=1.0,
            additional_metrics={
                "thread_results": results,
                "optimal_threads": max(results, key=lambda x: x[1])[0],
                "scaling_factor": best_performance / results[0][1] if results[0][1] > 0 else 0
            }
        )
    
    def benchmark_process_pool(self) -> BenchmarkResult:
        """Benchmark process pool performance"""
        
        def generate_attacks_process(count):
            # Import inside function for multiprocessing
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent / "gan_cyber_range"))
            from demo import LightweightAttackGenerator
            
            generator = LightweightAttackGenerator()
            return generator.generate_batch(count)
        
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        
        cpu_count = mp.cpu_count()
        process_counts = [1, min(2, cpu_count), min(4, cpu_count)]
        results = []
        
        for process_count in process_counts:
            if process_count <= cpu_count:
                proc_start = time.time()
                
                with ProcessPoolExecutor(max_workers=process_count) as executor:
                    futures = [executor.submit(generate_attacks_process, 25) for _ in range(process_count)]
                    proc_results = [future.result() for future in futures]
                
                proc_duration = time.time() - proc_start
                total_attacks = sum(len(result) for result in proc_results)
                proc_ops_per_sec = total_attacks / proc_duration
                
                results.append((process_count, proc_ops_per_sec))
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        
        total_duration = end_time - start_time
        best_performance = max(result[1] for result in results) if results else 0
        
        return BenchmarkResult(
            test_name="Process Pool Performance",
            duration=total_duration,
            operations_per_second=best_performance,
            memory_usage_mb=end_memory - start_memory,
            cpu_percent=self.process.cpu_percent(),
            success_rate=1.0 if results else 0.0,
            additional_metrics={
                "process_results": results,
                "cpu_count": cpu_count,
                "optimal_processes": max(results, key=lambda x: x[1])[0] if results else 0
            }
        )
    
    def benchmark_stress_test(self) -> BenchmarkResult:
        """Stress test with high load"""
        api = DemoAPI()
        generator = LightweightAttackGenerator()
        
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        start_cpu = self.process.cpu_percent()
        
        operations_completed = 0
        operations_failed = 0
        
        # High intensity operations
        try:
            # Create many ranges rapidly
            ranges = []
            for i in range(50):
                try:
                    response = api.create_range(f"stress-{i}")
                    ranges.append(response['range_id'])
                    operations_completed += 1
                except Exception:
                    operations_failed += 1
            
            # Generate many attacks
            for range_id in ranges[:10]:  # Limit to first 10 to avoid excessive load
                try:
                    api.generate_attacks(range_id, count=20)
                    operations_completed += 1
                except Exception:
                    operations_failed += 1
            
            # Generate large batches
            for _ in range(5):
                try:
                    attacks = generator.generate_batch(1000)
                    operations_completed += 1
                except Exception:
                    operations_failed += 1
        
        except Exception:
            operations_failed += 1
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        end_cpu = self.process.cpu_percent()
        
        total_operations = operations_completed + operations_failed
        duration = end_time - start_time
        
        return BenchmarkResult(
            test_name="Stress Test",
            duration=duration,
            operations_per_second=operations_completed / duration,
            memory_usage_mb=end_memory - start_memory,
            cpu_percent=(start_cpu + end_cpu) / 2,
            success_rate=operations_completed / max(1, total_operations),
            additional_metrics={
                "operations_completed": operations_completed,
                "operations_failed": operations_failed,
                "memory_peak": end_memory,
                "ranges_created": len(ranges) if 'ranges' in locals() else 0
            }
        )
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index == int(index):
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def generate_report(self, output_file: str = "performance_report.json") -> None:
        """Generate performance report"""
        import json
        
        report = {
            "benchmark_summary": {
                "start_time": self.suite.start_time.isoformat(),
                "end_time": self.suite.end_time.isoformat() if self.suite.end_time else None,
                "total_duration": self.suite.total_duration,
                "tests_completed": len(self.suite.results),
                "system_info": {
                    "cpu_count": mp.cpu_count(),
                    "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                    "python_version": sys.version
                }
            },
            "test_results": []
        }
        
        for result in self.suite.results:
            report["test_results"].append({
                "test_name": result.test_name,
                "duration": result.duration,
                "operations_per_second": result.operations_per_second,
                "memory_usage_mb": result.memory_usage_mb,
                "cpu_percent": result.cpu_percent,
                "success_rate": result.success_rate,
                "additional_metrics": result.additional_metrics
            })
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Performance report saved to: {output_file}")
    
    def print_summary(self) -> None:
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("🚀 PERFORMANCE BENCHMARK RESULTS")
        print("="*60)
        print(f"⏱️ Total Duration: {self.suite.total_duration:.2f}s")
        print(f"🧪 Tests Completed: {len(self.suite.results)}")
        print(f"💾 System Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB")
        print(f"🖥️ CPU Cores: {mp.cpu_count()}")
        print()
        
        print("Performance Results:")
        print("-" * 60)
        
        for result in self.suite.results:
            status = "✅" if result.success_rate >= 0.9 else "⚠️" if result.success_rate >= 0.7 else "❌"
            print(f"{status} {result.test_name:<25} | {result.operations_per_second:>8.1f} ops/sec | {result.memory_usage_mb:>6.1f}MB")
        
        # Overall performance score
        avg_ops_per_sec = statistics.mean(r.operations_per_second for r in self.suite.results)
        avg_success_rate = statistics.mean(r.success_rate for r in self.suite.results)
        
        print()
        print(f"📊 Overall Performance Score:")
        print(f"   Average Operations/sec: {avg_ops_per_sec:.1f}")
        print(f"   Average Success Rate: {avg_success_rate:.1%}")
        
        # Performance classification
        if avg_ops_per_sec > 1000 and avg_success_rate > 0.95:
            print("   Grade: 🏆 EXCELLENT")
        elif avg_ops_per_sec > 500 and avg_success_rate > 0.90:
            print("   Grade: ✅ GOOD")
        elif avg_ops_per_sec > 100 and avg_success_rate > 0.80:
            print("   Grade: ⚠️ ADEQUATE")
        else:
            print("   Grade: ❌ NEEDS IMPROVEMENT")
        
        print("="*60)


def main():
    """Main benchmark execution"""
    benchmark = PerformanceBenchmark()
    
    print("🚀 GAN-Cyber-Range-v2 Performance Benchmark Suite")
    print("="*60)
    print("This benchmark will test system performance across multiple dimensions.")
    print("Please ensure no other intensive processes are running.\n")
    
    # Run benchmarks
    suite = benchmark.run_all_benchmarks()
    
    # Print results
    benchmark.print_summary()
    
    # Generate report
    benchmark.generate_report("performance_benchmark_report.json")
    
    # Determine exit code based on performance
    avg_success_rate = statistics.mean(r.success_rate for r in suite.results) if suite.results else 0
    
    if avg_success_rate >= 0.9:
        print("\n🎉 Performance benchmark passed!")
        return 0
    elif avg_success_rate >= 0.7:
        print("\n⚠️ Performance benchmark completed with warnings")
        return 0
    else:
        print("\n❌ Performance benchmark failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)