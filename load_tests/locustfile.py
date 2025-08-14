"""
Load tests for Vision-Language Similarity Evaluation Service

This module provides comprehensive load testing scenarios for the FastAPI service
using Locust. It tests both single and batch evaluation endpoints with real data
from the test dataset, prioritizing local images when available.

Usage:
    # Run load test with default settings (10 users, 2 ramp-up rate)
    locust -f load_tests/locustfile.py --host=http://localhost:8000

    # Run headless with specific parameters
    locust -f load_tests/locustfile.py --host=http://localhost:8000 \
           --users 50 --spawn-rate 5 --run-time 5m --headless

    # CI/CD integration
    python -m pytest load_tests/test_load_performance.py
"""

import base64
import csv
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from locust import HttpUser, TaskSet, between, task
from locust.exception import StopUser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load test data
TEST_DATA_PATH = Path(__file__).parent.parent / "tests" / "data" / "samples"
CHALLENGE_SET_PATH = TEST_DATA_PATH / "challenge_set.csv"

def extract_user_id_from_url(url: str) -> Optional[str]:
    """Extract the user ID (first UUID) from Leonardo AI URL"""
    uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    matches = re.findall(uuid_pattern, url, re.IGNORECASE)
    return matches[0] if matches else None

class TestDataLoader:
    """Loads and manages test data for load testing"""
    
    def __init__(self):
        self.image_text_pairs: List[Dict[str, Any]] = []
        self.local_image_files: Dict[str, Path] = {}
        self.matched_pairs: List[Dict[str, Any]] = []
        self._load_data()
    
    def _load_data(self):
        """Load test data from CSV and local files"""
        # Get all available local image files mapped by UUID (filename)
        for image_file in TEST_DATA_PATH.glob("*.png"):
            uuid = image_file.stem
            self.local_image_files[uuid] = image_file
        
        logger.info(f"Found {len(self.local_image_files)} local image files")
        
        # Load image-text pairs from CSV
        if CHALLENGE_SET_PATH.exists():
            with open(CHALLENGE_SET_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    url = row['url']
                    caption = row['caption']
                    user_id = extract_user_id_from_url(url)
                    
                    # Check if we have a local file for this UUID
                    local_path = None
                    if user_id and user_id in self.local_image_files:
                        local_path = self.local_image_files[user_id]
                    
                    pair_data = {
                        'url': url,
                        'caption': caption,
                        'user_id': user_id,
                        'local_path': local_path,
                        'has_local_file': local_path is not None
                    }
                    
                    self.image_text_pairs.append(pair_data)
                    
                    # If we have a local file, add to matched pairs for priority usage
                    if local_path:
                        self.matched_pairs.append(pair_data)
        
        logger.info(f"Loaded {len(self.image_text_pairs)} total image-text pairs")
        logger.info(f"Found {len(self.matched_pairs)} pairs with local image files")
    
    def get_random_pair(self, prefer_local: bool = True) -> Dict[str, Any]:
        """Get a random image-text pair, preferring local files if available"""
        if prefer_local and self.matched_pairs:
            return random.choice(self.matched_pairs)
        return random.choice(self.image_text_pairs)
    
    def get_local_image_as_base64(self, pair: Dict[str, Any]) -> Optional[str]:
        """Convert local image to base64 if available"""
        if not pair.get('local_path'):
            return None
        
        try:
            with open(pair['local_path'], 'rb') as f:
                image_data = f.read()
                return f"data:image/png;base64,{base64.b64encode(image_data).decode()}"
        except Exception as e:
            logger.warning(f"Failed to load local image {pair['local_path']}: {e}")
            return None
    
    def get_batch_data(self, batch_size: int = 5, prefer_local: bool = True) -> List[Dict[str, Any]]:
        """Get a batch of random image-text pairs"""
        source_data = self.matched_pairs if prefer_local and self.matched_pairs else self.image_text_pairs
        sample_size = min(batch_size, len(source_data))
        return random.sample(source_data, sample_size)
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about loaded data"""
        return {
            'total_pairs': len(self.image_text_pairs),
            'local_files': len(self.local_image_files),
            'matched_pairs': len(self.matched_pairs),
            'local_coverage_percent': round((len(self.matched_pairs) / len(self.image_text_pairs)) * 100, 1) if self.image_text_pairs else 0
        }

# Global test data loader
test_data = TestDataLoader()

# Log data stats on startup
data_stats = test_data.get_stats()
logger.info(f"Test data loaded: {data_stats}")

class HealthCheckTasks(TaskSet):
    """Health check and basic endpoint validation tasks"""
    
    @task(3)
    def health_check(self):
        """Test health endpoint"""
        with self.client.get("/evaluator/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "available":
                    response.success()
                else:
                    response.failure(f"Health check failed: {data}")
            else:
                response.failure(f"Health check returned {response.status_code}")
    
    @task(1)
    def get_models(self):
        """Test models endpoint"""
        with self.client.get("/evaluator/v1/evaluation/models", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "available_configs" in data:
                    response.success()
                else:
                    response.failure("Models endpoint missing expected data")
            else:
                response.failure(f"Models endpoint returned {response.status_code}")

class SingleEvaluationTasks(TaskSet):
    """Single evaluation endpoint load testing"""
    
    @task(7)
    def evaluate_single_local_file(self):
        """Test single evaluation with local image file (highest priority)"""
        pair = test_data.get_random_pair(prefer_local=True)
        
        if pair.get('has_local_file'):
            # Use local file path directly
            payload = {
                "image_input": str(pair['local_path']),
                "text_prompt": pair["caption"],
                "model_config_name": random.choice(["fast", "accurate"])
            }
            
            with self.client.post(
                "/evaluator/v1/evaluation/single",
                json=payload,
                catch_response=True,
                timeout=30
            ) as response:
                if response.status_code == 200:
                    data = response.json()
                    if "clip_score" in data and data["clip_score"] >= 0:
                        response.success()
                        # Log successful local file usage
                        logger.debug(f"Local file evaluation successful: {pair['user_id']}")
                    else:
                        response.failure(f"Invalid response data: {data}")
                else:
                    response.failure(f"Local file evaluation failed: {response.status_code}")
        else:
            # Fallback to URL if no local file
            self.evaluate_single_url()
    
    @task(3)
    def evaluate_single_base64(self):
        """Test single evaluation with base64 image from local file"""
        pair = test_data.get_random_pair(prefer_local=True)
        base64_image = test_data.get_local_image_as_base64(pair)
        
        if base64_image:
            payload = {
                "image_input": base64_image,
                "text_prompt": pair["caption"][:200],  # Truncate long captions for faster processing
                "model_config_name": "fast"  # Use fast model for base64 to reduce load
            }
            
            with self.client.post(
                "/evaluator/v1/evaluation/single",
                json=payload,
                catch_response=True,
                timeout=30
            ) as response:
                if response.status_code == 200:
                    data = response.json()
                    if "clip_score" in data:
                        response.success()
                    else:
                        response.failure(f"Invalid response data: {data}")
                else:
                    response.failure(f"Base64 evaluation failed: {response.status_code}")
        else:
            # Fallback to URL evaluation
            self.evaluate_single_url()
    
    @task(2)
    def evaluate_single_url(self):
        """Test single evaluation with image URL (fallback)"""
        pair = test_data.get_random_pair(prefer_local=False)
        
        payload = {
            "image_input": pair["url"],
            "text_prompt": pair["caption"],
            "model_config_name": random.choice(["fast", "accurate"])
        }
        
        with self.client.post(
            "/evaluator/v1/evaluation/single",
            json=payload,
            catch_response=True,
            timeout=30
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "clip_score" in data and data["clip_score"] >= 0:
                    response.success()
                else:
                    response.failure(f"Invalid response data: {data}")
            else:
                response.failure(f"URL evaluation failed: {response.status_code}")

class BatchEvaluationTasks(TaskSet):
    """Batch evaluation endpoint load testing"""
    
    @task(4)
    def evaluate_batch_local_files(self):
        """Test batch evaluation with local image files"""
        batch_data = test_data.get_batch_data(batch_size=4, prefer_local=True)
        evaluations = []
        
        for pair in batch_data:
            if pair.get('has_local_file'):
                # Use local file path
                evaluations.append({
                    "image_input": str(pair['local_path']),
                    "text_prompt": pair["caption"],
                    "model_config_name": "fast"
                })
            else:
                # Fallback to URL
                evaluations.append({
                    "image_input": pair["url"],
                    "text_prompt": pair["caption"],
                    "model_config_name": "fast"
                })
        
        payload = {
            "evaluations": evaluations,
            "batch_size": 8,
            "show_progress": False
        }
        
        with self.client.post(
            "/evaluator/v1/evaluation/batch",
            json=payload,
            catch_response=True,
            timeout=60
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if ("total_processed" in data and 
                    data["total_processed"] == len(evaluations)):
                    response.success()
                    logger.debug(f"Batch evaluation successful: {data['total_successful']}/{data['total_processed']}")
                else:
                    response.failure(f"Batch processing incomplete: {data}")
            else:
                response.failure(f"Batch evaluation failed: {response.status_code}")
    
    @task(2)
    def evaluate_batch_mixed(self):
        """Test batch evaluation with mixed local files and URLs"""
        batch_data = test_data.get_batch_data(batch_size=6, prefer_local=False)
        evaluations = []
        
        for pair in batch_data:
            # Mix of local files, base64, and URLs
            choice = random.choice(['local', 'base64', 'url'])
            
            if choice == 'local' and pair.get('has_local_file'):
                evaluations.append({
                    "image_input": str(pair['local_path']),
                    "text_prompt": pair["caption"],
                    "model_config_name": "fast"
                })
            elif choice == 'base64' and pair.get('has_local_file'):
                base64_image = test_data.get_local_image_as_base64(pair)
                if base64_image:
                    evaluations.append({
                        "image_input": base64_image,
                        "text_prompt": pair["caption"][:150],
                        "model_config_name": "fast"
                    })
                else:
                    # Fallback to URL
                    evaluations.append({
                        "image_input": pair["url"],
                        "text_prompt": pair["caption"],
                        "model_config_name": "fast"
                    })
            else:
                # Use URL
                evaluations.append({
                    "image_input": pair["url"],
                    "text_prompt": pair["caption"],
                    "model_config_name": random.choice(["fast", "accurate"])
                })
        
        payload = {
            "evaluations": evaluations,
            "batch_size": 16,
            "show_progress": False
        }
        
        with self.client.post(
            "/evaluator/v1/evaluation/batch",
            json=payload,
            catch_response=True,
            timeout=120
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "total_processed" in data:
                    response.success()
                else:
                    response.failure(f"Invalid batch response: {data}")
            else:
                response.failure(f"Mixed batch evaluation failed: {response.status_code}")

class VisionLanguageUser(HttpUser):
    """Main user class that simulates different usage patterns"""
    
    # Wait time between tasks (1-3 seconds)
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts"""
        logger.info(f"User started - Data stats: {test_data.get_stats()}")
        
        # Verify service is available
        try:
            response = self.client.get("/evaluator/health", timeout=10)
            if response.status_code != 200:
                logger.error("Service health check failed, stopping user")
                raise StopUser()
        except Exception as e:
            logger.error(f"Failed to connect to service: {e}")
            raise StopUser()
    
    # Define task weights for different user behaviors
    tasks = {
        HealthCheckTasks: 1,      # 10% - monitoring/health checks
        SingleEvaluationTasks: 6, # 60% - most common usage
        BatchEvaluationTasks: 3,  # 30% - power users
    }

class LightLoadUser(HttpUser):
    """Light load user for CI/CD testing with local files priority"""
    
    wait_time = between(2, 5)
    
    @task(3)
    def health_check(self):
        self.client.get("/evaluator/health")
    
    @task(7)
    def single_evaluation_local(self):
        pair = test_data.get_random_pair(prefer_local=True)
        if pair.get('has_local_file'):
            payload = {
                "image_input": str(pair['local_path']),
                "text_prompt": pair["caption"][:100],
                "model_config_name": "fast"
            }
        else:
            payload = {
                "image_input": pair["url"],
                "text_prompt": pair["caption"][:100],
                "model_config_name": "fast"
            }
        
        self.client.post("/evaluator/v1/evaluation/single", json=payload, timeout=30)

class StressTestUser(HttpUser):
    """High-intensity user for stress testing with local files"""
    
    wait_time = between(0.1, 0.5)
    
    tasks = {
        SingleEvaluationTasks: 8,
        BatchEvaluationTasks: 2,
    }

# Performance thresholds for automated testing
PERFORMANCE_THRESHOLDS = {
    "avg_response_time_ms": 2000,      # Average response time under 2s
    "95th_percentile_ms": 5000,        # 95th percentile under 5s
    "max_response_time_ms": 10000,     # Max response time under 10s
    "failure_rate_percent": 1.0,       # Failure rate under 1%
    "requests_per_second": 10,         # Minimum throughput
}