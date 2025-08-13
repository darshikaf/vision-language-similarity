import pytest
from fastapi import status


class TestEvaluationAPI:
    """Simplified API tests focusing on core functionality"""

    def test_health_endpoints(self, test_client):
        response = test_client.get("/evaluator/health")
        assert response.status_code == status.HTTP_200_OK
        
        response = test_client.get("/evaluator/v1/evaluation/health")
        assert response.status_code == status.HTTP_200_OK

    def test_models_endpoint(self, test_client):
        response = test_client.get("/evaluator/v1/evaluation/models")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "available_configs" in data
        assert "fast" in data["available_configs"]
        assert "accurate" in data["available_configs"]

    def test_single_evaluation_success(self, test_client, evaluation_request_payloads):
        payload = evaluation_request_payloads["valid_single"]
        
        response = test_client.post("/evaluator/v1/evaluation/single", json=payload)
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "clip_score" in data
        assert "processing_time_ms" in data
        assert "model_used" in data

    def test_batch_evaluation_success(self, test_client, evaluation_request_payloads):
        payload = evaluation_request_payloads["valid_batch"]
        
        response = test_client.post("/evaluator/v1/evaluation/batch", json=payload)
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "results" in data
        assert "total_processed" in data
        assert len(data["results"]) == len(payload["evaluations"])

    def test_batch_evaluation_empty_request(self, test_client):
        payload = {"evaluations": []}
        
        response = test_client.post("/evaluator/v1/evaluation/batch", json=payload)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
