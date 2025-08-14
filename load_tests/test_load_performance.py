"""
Performance tests for CI/CD integration

This module provides automated performance testing that can be integrated
into CI/CD pipelines. It runs lightweight load tests and validates
performance against defined thresholds.

Usage:
    # Run performance tests
    python -m pytest load_tests/test_load_performance.py -v

    # Run with custom service URL
    LOAD_TEST_HOST=http://localhost:8000 python -m pytest load_tests/test_load_performance.py
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

import pytest
import requests
from locustfile import PERFORMANCE_THRESHOLDS, test_data


class TestServicePerformance:
    """Performance tests using Locust for automated CI/CD validation"""
    
    @pytest.fixture(scope="class")
    def service_host(self):
        """Get service host from environment or default"""
        return os.getenv("LOAD_TEST_HOST", "http://localhost:8000")
    
    @pytest.fixture(scope="class")
    def service_ready(self, service_host):
        """Ensure service is ready before running load tests"""
        max_retries = 30
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{service_host}/evaluator/health", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "available":
                        return True
                
                print(f"Service not ready (attempt {attempt + 1}/{max_retries}): {response.status_code}")
                time.sleep(retry_delay)
                
            except requests.RequestException as e:
                print(f"Connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
        
        pytest.fail("Service is not available after waiting")
    
    def run_locust_test(self, service_host: str, users: int = 5, spawn_rate: int = 2, 
                       run_time: str = "30s", user_class: str = "LightLoadUser") -> Dict[str, Any]:
        """Run a Locust test and return performance metrics"""
        
        # Create temporary file for results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            results_file = temp_file.name
        
        try:
            # Build Locust command
            locust_file = Path(__file__).parent / "locustfile.py"
            cmd = [
                "locust",
                "-f", str(locust_file),
                "--host", service_host,
                "--users", str(users),
                "--spawn-rate", str(spawn_rate),
                "--run-time", run_time,
                "--headless",
                "--html", f"{results_file}.html",
                "--csv", results_file.replace('.json', ''),
                "--user-class", user_class,
                "--only-summary"
            ]
            
            print(f"Running load test: {' '.join(cmd)}")
            
            # Run Locust test
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=Path(__file__).parent.parent
            )
            
            if result.returncode != 0:
                print(f"Locust stderr: {result.stderr}")
                print(f"Locust stdout: {result.stdout}")
                raise Exception(f"Locust test failed with return code {result.returncode}")
            
            # Parse results from stdout
            return self._parse_locust_output(result.stdout)
            
        finally:
            # Cleanup temporary files
            for ext in ['.json', '.html', '_stats.csv', '_failures.csv', '_exceptions.csv']:
                temp_path = Path(results_file.replace('.json', '') + ext)
                if temp_path.exists():
                    temp_path.unlink()
    
    def _parse_locust_output(self, output: str) -> Dict[str, Any]:
        """Parse Locust output to extract performance metrics"""
        metrics = {}
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            
            # Parse summary statistics
            if "Total requests" in line:
                metrics['total_requests'] = int(line.split()[-1])
            elif "Failures" in line:
                failures = line.split()[-1]
                metrics['total_failures'] = int(failures) if failures.isdigit() else 0
            elif "req/s" in line and "Average" in line:
                # Extract average response time and RPS
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "ms" and i > 0:
                        metrics['avg_response_time_ms'] = float(parts[i-1])
                    elif part == "req/s" and i > 0:
                        metrics['requests_per_second'] = float(parts[i-1])
            elif "95%" in line:
                # Extract 95th percentile
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "ms" and "95%" in line:
                        metrics['95th_percentile_ms'] = float(parts[i-1])
            elif "Max" in line and "ms" in line:
                # Extract max response time
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "ms" and i > 0:
                        metrics['max_response_time_ms'] = float(parts[i-1])
        
        # Calculate failure rate
        if 'total_requests' in metrics and 'total_failures' in metrics:
            if metrics['total_requests'] > 0:
                metrics['failure_rate_percent'] = (metrics['total_failures'] / metrics['total_requests']) * 100
            else:
                metrics['failure_rate_percent'] = 0
        
        return metrics
    
    def test_light_load_performance(self, service_host, service_ready):
        """Test performance under light load conditions"""
        
        # Verify test data is available
        data_stats = test_data.get_stats()
        assert data_stats['total_pairs'] > 0, "No test data available"
        
        # Run light load test
        metrics = self.run_locust_test(
            service_host=service_host,
            users=3,
            spawn_rate=1,
            run_time="30s",
            user_class="LightLoadUser"
        )
        
        print(f"Light load test results: {json.dumps(metrics, indent=2)}")
        
        # Validate against thresholds
        self._validate_performance_thresholds(metrics, "light_load")
    
    def test_moderate_load_performance(self, service_host, service_ready):
        """Test performance under moderate load conditions"""
        
        metrics = self.run_locust_test(
            service_host=service_host,
            users=8,
            spawn_rate=2,
            run_time="45s",
            user_class="VisionLanguageUser"
        )
        
        print(f"Moderate load test results: {json.dumps(metrics, indent=2)}")
        
        # Relaxed thresholds for moderate load
        moderate_thresholds = PERFORMANCE_THRESHOLDS.copy()
        moderate_thresholds['avg_response_time_ms'] = 3000
        moderate_thresholds['95th_percentile_ms'] = 8000
        moderate_thresholds['failure_rate_percent'] = 2.0
        
        self._validate_performance_thresholds(metrics, "moderate_load", moderate_thresholds)
    
    def test_health_endpoint_performance(self, service_host, service_ready):
        """Test health endpoint performance specifically"""
        
        # Simple direct test for health endpoint
        start_time = time.time()
        response = requests.get(f"{service_host}/evaluator/health")
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        assert response_time_ms < 500, f"Health endpoint too slow: {response_time_ms}ms"
        
        data = response.json()
        assert data.get("status") == "available", f"Service not available: {data}"
        
        print(f"Health endpoint response time: {response_time_ms:.2f}ms")
    
    def test_single_evaluation_performance(self, service_host, service_ready):
        """Test single evaluation endpoint performance"""
        
        # Get a test pair with local file if available
        pair = test_data.get_random_pair(prefer_local=True)
        assert pair, "No test data available"
        
        # Use local file if available, otherwise URL
        image_input = str(pair['local_path']) if pair.get('has_local_file') else pair['url']
        
        payload = {
            "image_input": image_input,
            "text_prompt": pair["caption"][:100],  # Truncate for faster processing
            "model_config_name": "fast"
        }
        
        # Measure response time
        start_time = time.time()
        response = requests.post(f"{service_host}/evaluator/v1/evaluation/single", 
                               json=payload, timeout=30)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200, f"Evaluation failed: {response.status_code}"
        
        data = response.json()
        assert "clip_score" in data, f"Invalid response: {data}"
        assert data["clip_score"] >= 0, f"Invalid CLIP score: {data['clip_score']}"
        
        # Performance assertion
        assert response_time_ms < 5000, f"Single evaluation too slow: {response_time_ms}ms"
        
        print(f"Single evaluation response time: {response_time_ms:.2f}ms, CLIP score: {data['clip_score']}")
    
    def _validate_performance_thresholds(self, metrics: Dict[str, Any], test_name: str, 
                                       thresholds: Dict[str, float] = None):
        """Validate performance metrics against thresholds"""
        
        if thresholds is None:
            thresholds = PERFORMANCE_THRESHOLDS
        
        failures = []
        
        for metric_name, threshold in thresholds.items():
            if metric_name in metrics:
                actual_value = metrics[metric_name]
                
                if metric_name == "requests_per_second":
                    # For RPS, actual should be >= threshold
                    if actual_value < threshold:
                        failures.append(f"{metric_name}: {actual_value} < {threshold}")
                else:
                    # For other metrics, actual should be <= threshold
                    if actual_value > threshold:
                        failures.append(f"{metric_name}: {actual_value} > {threshold}")
        
        if failures:
            failure_msg = f"Performance test '{test_name}' failed thresholds:\n" + "\n".join(failures)
            failure_msg += f"\nActual metrics: {json.dumps(metrics, indent=2)}"
            failure_msg += f"\nThresholds: {json.dumps(thresholds, indent=2)}"
            pytest.fail(failure_msg)
        
        print(f"✓ Performance test '{test_name}' passed all thresholds")


class TestDataAvailability:
    """Test that required test data is available"""
    
    def test_test_data_loaded(self):
        """Ensure test data is properly loaded"""
        stats = test_data.get_stats()
        
        assert stats['total_pairs'] > 0, "No test data pairs loaded"
        assert stats['local_files'] > 0, "No local image files found"
        assert stats['matched_pairs'] > 0, "No matched pairs found"
        
        print(f"Test data validation passed: {stats}")
    
    def test_local_file_coverage(self):
        """Test that we have good local file coverage"""
        stats = test_data.get_stats()
        
        coverage_percent = stats['local_coverage_percent']
        min_coverage = 50  # At least 50% of URLs should have local files
        
        assert coverage_percent >= min_coverage, (
            f"Local file coverage too low: {coverage_percent}% < {min_coverage}%"
        )
        
        print(f"Local file coverage: {coverage_percent}%")


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v", "-s"])