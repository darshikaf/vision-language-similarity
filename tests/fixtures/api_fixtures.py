import pytest
from fastapi.testclient import TestClient
from service.main import app
from service.evaluation.handler import EvaluationHandler
from unittest.mock import Mock, AsyncMock


@pytest.fixture(scope="session")
def test_client():
    """FastAPI test client for API testing"""
    return TestClient(app)


@pytest.fixture
def mock_evaluation_handler():
    """Mock evaluation handler for isolated API tests"""
    handler = Mock(spec=EvaluationHandler)
    handler.evaluate_single = AsyncMock()
    handler.evaluate_batch = AsyncMock() 
    handler.health_check = AsyncMock()
    return handler


@pytest.fixture
def evaluation_request_payloads(sample_image_path, sample_prompts):
    """Common evaluation request payloads for API testing using test data"""
    return {
        "valid_single": {
            "image_input": sample_image_path,
            "text_prompt": sample_prompts[0],
            "model_config_name": "fast"
        },
        "valid_batch": {
            "evaluations": [
                {
                    "image_input": sample_image_path,
                    "text_prompt": sample_prompts[0],
                    "model_config_name": "fast"
                },
                {
                    "image_input": sample_image_path, 
                    "text_prompt": sample_prompts[1] if len(sample_prompts) > 1 else sample_prompts[0],
                    "model_config_name": "accurate"
                }
            ]
        },
        "invalid_single": {
            "image_input": "nonexistent_file.jpg",
            "text_prompt": "test prompt",
            "model_config_name": "invalid_config"
        },
        "empty_batch": {
            "evaluations": []
        }
    }


@pytest.fixture 
def expected_response_schemas():
    """Expected response schemas for validation"""
    return {
        "health_response": {
            "status": str,
            "model_loaded": bool,
            "available_configs": list
        },
        "evaluation_response": {
            "image_input": str,
            "text_prompt": str,
            "clip_score": float,
            "processing_time_ms": float,
            "model_used": str
        },
        "batch_response": {
            "results": list,
            "total_processed": int,
            "total_successful": int,
            "total_failed": int,
            "total_processing_time_ms": float
        }
    }
