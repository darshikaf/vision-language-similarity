"""CLAUDE generated: Model performance benchmarks for inference time and throughput."""

import asyncio
import time
import statistics
from dataclasses import dataclass
from typing import List, Dict, Any
import pytest

from service.core.ml.engines.openclip_evaluator import OpenCLIPEvaluator
from .test_memory_usage import MemoryTracker, memory_tracking


@dataclass
class PerformanceBenchmark:
    """Performance benchmark results."""
    test_name: str
    num_images: int
    num_prompts: int
    model_config: str
    total_time_seconds: float
    avg_time_per_item_ms: float
    throughput_items_per_second: float
    memory_peak_mb: float
    memory_growth_mb: float
    successful_evaluations: int
    failed_evaluations: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


class PerformanceBenchmarker:
    """Utility class for running performance benchmarks."""
    
    def __init__(self):
        self.results: List[PerformanceBenchmark] = []
    
    async def benchmark_single_evaluations(
        self, 
        evaluator: OpenCLIPEvaluator,
        images: List,
        prompts: List[str],
        test_name: str
    ) -> PerformanceBenchmark:
        """Benchmark multiple single evaluations."""
        
        with memory_tracking(f"benchmark_{test_name}") as tracker:
            tracker.take_snapshot("benchmark_start")
            
            latencies = []
            successful = 0
            failed = 0
            
            start_time = time.time()
            
            # Run single evaluations
            for i, (image, prompt) in enumerate(zip(images, prompts)):
                eval_start = time.time()
                try:
                    result = await evaluator.evaluate_single(image, prompt)
                    eval_time = (time.time() - eval_start) * 1000
                    latencies.append(eval_time)
                    
                    if result.error is None:
                        successful += 1
                    else:
                        failed += 1
                        
                except Exception:
                    failed += 1
                    eval_time = (time.time() - eval_start) * 1000
                    latencies.append(eval_time)
                
                # Take periodic memory snapshots
                if i % max(1, len(images) // 5) == 0:
                    tracker.take_snapshot(f"after_{i}_evaluations")
            
            total_time = time.time() - start_time
            tracker.take_snapshot("benchmark_end")
        
        # Calculate statistics
        num_items = len(images)
        avg_time_per_item = (total_time * 1000) / num_items if num_items > 0 else 0
        throughput = num_items / total_time if total_time > 0 else 0
        
        # Memory statistics
        memory_diff = tracker.get_memory_diff(f"benchmark_{test_name}_start", "benchmark_end")
        peak_snapshot = max(tracker.snapshots, key=lambda s: s.rss_mb)
        
        # Latency percentiles
        latencies.sort()
        p50 = statistics.median(latencies) if latencies else 0
        p95 = latencies[int(0.95 * len(latencies))] if latencies else 0
        p99 = latencies[int(0.99 * len(latencies))] if latencies else 0
        
        benchmark = PerformanceBenchmark(
            test_name=test_name,
            num_images=len(images),
            num_prompts=len(prompts),
            model_config=evaluator.model_config_name,
            total_time_seconds=total_time,
            avg_time_per_item_ms=avg_time_per_item,
            throughput_items_per_second=throughput,
            memory_peak_mb=peak_snapshot.rss_mb,
            memory_growth_mb=memory_diff["rss_diff_mb"],
            successful_evaluations=successful,
            failed_evaluations=failed,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99
        )
        
        self.results.append(benchmark)
        return benchmark
    
    def print_benchmark_summary(self):
        """Print summary of all benchmarks."""
        if not self.results:
            print("No benchmark results available")
            return
        
        print("\n" + "="*100)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*100)
        print(f"{'Test Name':<25} {'Model':<8} {'Items':<6} {'Total(s)':<8} {'Avg(ms)':<8} {'Throughput':<10} {'P95(ms)':<8} {'Memory(MB)':<10}")
        print("-"*100)
        
        for result in self.results:
            print(f"{result.test_name:<25} {result.model_config:<8} {result.num_images:<6} "
                  f"{result.total_time_seconds:<8.2f} {result.avg_time_per_item_ms:<8.1f} "
                  f"{result.throughput_items_per_second:<10.2f} {result.p95_latency_ms:<8.1f} "
                  f"{result.memory_peak_mb:<10.1f}")
        
        print("="*100)


class TestModelPerformanceBaselines:
    """Establish baseline performance metrics for current system."""
    
    @pytest.mark.asyncio
    async def test_single_image_performance_baseline(self, fast_evaluator, performance_image):
        """Baseline performance for single image evaluation."""
        benchmarker = PerformanceBenchmarker()
        
        # Single image, single prompt
        images = [performance_image]
        prompts = ["A blue colored test image"]
        
        result = await benchmarker.benchmark_single_evaluations(
            fast_evaluator, images, prompts, "single_image_fast"
        )
        
        benchmarker.print_benchmark_summary()
        
        # Assertions for reasonable performance
        assert result.successful_evaluations == 1
        assert result.failed_evaluations == 0
        assert result.avg_time_per_item_ms < 10000  # Should be less than 10 seconds
        assert result.throughput_items_per_second > 0.05  # At least 0.05 items per second
    
    @pytest.mark.asyncio
    async def test_small_batch_performance_baseline(self, fast_evaluator, performance_images_small, performance_prompts_small):
        """Baseline performance for small batch evaluation."""
        benchmarker = PerformanceBenchmarker()
        
        result = await benchmarker.benchmark_single_evaluations(
            fast_evaluator, performance_images_small, performance_prompts_small, "small_batch"
        )
        
        benchmarker.print_benchmark_summary()
        
        # Performance assertions
        assert result.successful_evaluations >= len(performance_images_small) * 0.9  # At least 90% success rate
        assert result.avg_time_per_item_ms < 8000  # Should be less than 8 seconds per item
    
    @pytest.mark.asyncio
    async def test_multiple_evaluators_memory_baseline(self, performance_images_small, performance_prompts_small):
        """Test memory usage with multiple evaluators - THIS IS THE KEY BASELINE."""
        # Create multiple evaluators with same model config
        evaluators = [
            OpenCLIPEvaluator(model_config_name="fast"),
            OpenCLIPEvaluator(model_config_name="fast"),
            OpenCLIPEvaluator(model_config_name="fast"),
        ]
        
        with memory_tracking("multiple_evaluators_baseline") as tracker:
            tracker.take_snapshot("evaluators_created")
            
            # Run one evaluation with each to ensure models are loaded
            for i, evaluator in enumerate(evaluators):
                await evaluator.evaluate_single(performance_images_small[0], performance_prompts_small[0])
                tracker.take_snapshot(f"evaluator_{i}_used")
        
        benchmarker = PerformanceBenchmarker()
        benchmarker.print_benchmark_summary() 
        
        # This baseline will show high memory usage - to be optimized in later phases
        memory_diff = tracker.get_memory_diff("multiple_evaluators_baseline_start", "multiple_evaluators_baseline_end")
        print(f"BASELINE: Multiple evaluators memory usage: {memory_diff['rss_diff_mb']:.1f}MB")
        print("This will be significantly reduced in Phase 3 with shared models!")
        
        # Cleanup
        for evaluator in evaluators:
            del evaluator


# Performance regression thresholds for Phase 1 baseline
PERFORMANCE_THRESHOLDS = {
    "max_single_evaluation_ms": 10000,  # Single evaluation should not exceed 10 seconds (conservative baseline)
    "min_throughput_per_second": 0.05,  # Minimum throughput (conservative)
    "max_memory_growth_mb": 1000,  # Maximum memory growth (will be high in baseline)
    "min_success_rate": 0.9,  # Minimum 90% success rate
}
