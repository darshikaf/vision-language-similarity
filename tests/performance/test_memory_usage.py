"""CLAUDE generated: Memory usage tracking utilities and tests."""

import gc
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import pytest
import psutil
from contextlib import contextmanager

from service.core.ml.engines.openclip_evaluator import OpenCLIPEvaluator


@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a point in time."""
    timestamp: float
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    available_mb: float  # Available system memory in MB
    cpu_percent: float  # CPU usage percentage
    label: str = ""


class MemoryTracker:
    """Track memory usage during tests."""
    
    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []
        self.process = psutil.Process()
    
    def take_snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a memory usage snapshot."""
        gc.collect()  # Force garbage collection for accurate measurement
        time.sleep(0.1)  # Allow cleanup to complete
        
        memory_info = self.process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            available_mb=virtual_memory.available / 1024 / 1024,
            cpu_percent=self.process.cpu_percent(),
            label=label
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_memory_diff(self, start_label: str, end_label: str) -> Dict[str, float]:
        """Get memory difference between two snapshots."""
        start_snapshot = None
        end_snapshot = None
        
        for snapshot in self.snapshots:
            if snapshot.label == start_label:
                start_snapshot = snapshot
            elif snapshot.label == end_label:
                end_snapshot = snapshot
        
        if not start_snapshot or not end_snapshot:
            raise ValueError(f"Could not find snapshots with labels: {start_label}, {end_label}")
        
        return {
            "rss_diff_mb": end_snapshot.rss_mb - start_snapshot.rss_mb,
            "vms_diff_mb": end_snapshot.vms_mb - start_snapshot.vms_mb,
            "available_diff_mb": end_snapshot.available_mb - start_snapshot.available_mb,
            "time_diff_seconds": end_snapshot.timestamp - start_snapshot.timestamp
        }
    
    def print_summary(self):
        """Print memory usage summary."""
        if not self.snapshots:
            print("No memory snapshots taken")
            return
            
        print("\n" + "="*60)
        print("MEMORY USAGE SUMMARY")
        print("="*60)
        
        for i, snapshot in enumerate(self.snapshots):
            print(f"{i+1:2d}. {snapshot.label:25s} | "
                  f"RSS: {snapshot.rss_mb:6.1f}MB | "
                  f"VMS: {snapshot.vms_mb:6.1f}MB | "
                  f"Available: {snapshot.available_mb:6.0f}MB")
        
        if len(self.snapshots) >= 2:
            baseline = self.snapshots[0]
            peak = max(self.snapshots, key=lambda s: s.rss_mb)
            final = self.snapshots[-1]
            
            print("\n" + "-"*60)
            print(f"Baseline RSS:        {baseline.rss_mb:6.1f}MB")
            print(f"Peak RSS:           {peak.rss_mb:6.1f}MB  (+{peak.rss_mb - baseline.rss_mb:5.1f}MB)")
            print(f"Final RSS:          {final.rss_mb:6.1f}MB  (+{final.rss_mb - baseline.rss_mb:5.1f}MB)")
            print(f"Memory Growth:      {final.rss_mb - baseline.rss_mb:6.1f}MB")
        print("="*60)
    
    def assert_memory_leak_threshold(self, start_label: str, end_label: str, threshold_mb: float = 50):
        """Assert that memory growth is within threshold."""
        diff = self.get_memory_diff(start_label, end_label)
        memory_growth = diff["rss_diff_mb"]
        
        assert memory_growth <= threshold_mb, (
            f"Memory leak detected! Memory grew by {memory_growth:.1f}MB "
            f"(threshold: {threshold_mb}MB) between '{start_label}' and '{end_label}'"
        )


@contextmanager
def memory_tracking(label_prefix: str = "test"):
    """Context manager for automatic memory tracking."""
    tracker = MemoryTracker()
    tracker.take_snapshot(f"{label_prefix}_start")
    
    try:
        yield tracker
    finally:
        tracker.take_snapshot(f"{label_prefix}_end")
        tracker.print_summary()


class TestMemoryTracking:
    """Test memory tracking utilities."""
    
    def test_memory_tracker_basic_functionality(self):
        """Test that memory tracker works correctly."""
        tracker = MemoryTracker()
        
        # Take baseline snapshot
        snapshot1 = tracker.take_snapshot("baseline")
        assert snapshot1.rss_mb > 0
        assert snapshot1.vms_mb > 0
        
        # Allocate some memory
        large_list = [i for i in range(100000)]
        snapshot2 = tracker.take_snapshot("after_allocation")
        
        # Clean up
        del large_list
        gc.collect()
        snapshot3 = tracker.take_snapshot("after_cleanup")
        
        # Verify snapshots are reasonable
        assert snapshot2.rss_mb > snapshot1.rss_mb  # Memory should increase
        assert len(tracker.snapshots) == 3
        
        # Test memory diff calculation
        diff = tracker.get_memory_diff("baseline", "after_allocation")
        assert diff["rss_diff_mb"] > 0
        assert diff["time_diff_seconds"] >= 0
    
    def test_memory_context_manager(self):
        """Test memory tracking context manager."""
        with memory_tracking("context_test") as tracker:
            # Allocate some memory
            data = [i * 2 for i in range(50000)]
            tracker.take_snapshot("mid_test")
        
        # Should have start, mid, and end snapshots
        assert len(tracker.snapshots) == 3
        assert tracker.snapshots[0].label == "context_test_start"
        assert tracker.snapshots[1].label == "mid_test"
        assert tracker.snapshots[2].label == "context_test_end"


class TestBaselineMemoryUsage:
    """Establish baseline memory usage for current system."""
    
    def test_single_evaluator_memory_baseline(self, fast_evaluator):
        """Measure memory usage of single evaluator."""
        with memory_tracking("single_evaluator") as tracker:
            # Evaluator is already created by fixture
            tracker.take_snapshot("evaluator_created")
            
            # Verify evaluator works
            assert fast_evaluator.model_config_name == "fast"
            assert fast_evaluator.device is not None
        
        # Record the memory usage (will be printed by context manager)
        # This establishes our baseline for future comparisons
    
    def test_multiple_same_evaluators_memory_baseline(self, multiple_fast_evaluators):
        """Measure memory usage with multiple evaluators using same model."""
        with memory_tracking("multiple_same_evaluators") as tracker:
            tracker.take_snapshot("all_evaluators_created")
            
            # Verify all evaluators work
            for i, evaluator in enumerate(multiple_fast_evaluators):
                assert evaluator.model_config_name == "fast"
                tracker.take_snapshot(f"evaluator_{i}_verified")
        
        # This should show significant memory usage (3x model weights)
        # Will be optimized in later phases
    
    def test_multiple_different_evaluators_memory_baseline(self, mixed_evaluators):
        """Measure memory usage with evaluators using different models."""
        with memory_tracking("multiple_different_evaluators") as tracker:
            tracker.take_snapshot("all_evaluators_created")
            
            # Verify evaluators
            assert mixed_evaluators[0].model_config_name == "fast"
            assert mixed_evaluators[1].model_config_name == "accurate"
            assert mixed_evaluators[2].model_config_name == "fast"
            
            tracker.take_snapshot("all_evaluators_verified")
        
        # This should show even higher memory usage (different model weights)
    
    def test_evaluator_creation_memory_pattern(self):
        """Test memory pattern during evaluator creation and destruction."""
        with memory_tracking("evaluator_lifecycle") as tracker:
            # Create evaluator
            evaluator = OpenCLIPEvaluator(model_config_name="fast")
            tracker.take_snapshot("evaluator_created")
            
            # Use evaluator
            assert evaluator.device is not None
            tracker.take_snapshot("evaluator_used")
            
            # Destroy evaluator
            del evaluator
            gc.collect()
            tracker.take_snapshot("evaluator_destroyed")
        
        # Check for memory leaks (using 150MB threshold based on baseline findings)
        tracker.assert_memory_leak_threshold("evaluator_lifecycle_start", "evaluator_destroyed", 150)


# Performance thresholds for regression detection
MEMORY_THRESHOLDS = {
    "single_evaluator_max_mb": 1000,  # Single evaluator should not exceed 1GB
    "multiple_evaluator_growth_mb": 500,  # Each additional evaluator should not add more than 500MB
    "memory_leak_threshold_mb": 50,  # Memory leaks should not exceed 50MB
}
