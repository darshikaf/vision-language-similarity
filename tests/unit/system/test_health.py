import pytest
from fastapi.testclient import TestClient
from service.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data
    assert "health" in data


def test_health_endpoint():
    """Test basic health endpoint"""
    response = client.get("/evaluator/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "available"
    assert "service" in data


def test_evaluation_health_endpoint():
    """Test evaluation service health endpoint"""
    response = client.get("/evaluator/v1/evaluation/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "available_configs" in data
