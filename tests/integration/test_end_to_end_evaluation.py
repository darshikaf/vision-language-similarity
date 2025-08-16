import pytest
import time
from pathlib import Path
from fastapi import status


class TestEndToEndEvaluation:
    @pytest.mark.integration
    def test_complete_single_evaluation_workflow(self, test_client, matched_image_caption_pairs):
        if not matched_image_caption_pairs:
            pytest.skip("No matched image-caption pairs available for integration testing")
        
        image_path, caption = matched_image_caption_pairs[0]
        
        payload = {
            "image_input": image_path,
            "text_prompt": caption,
            "model_config_name": "fast"
        }
        
        start_time = time.time()
        response = test_client.post("/evaluator/v1/evaluation/single", json=payload)
        end_time = time.time()
        
        # Validate response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Validate response structure
        required_fields = {"image_input", "text_prompt", "clip_score", "processing_time_ms", "model_used"}
        assert all(field in data for field in required_fields)
        
        # Validate response values
        assert data["image_input"] == image_path
        assert data["text_prompt"] == caption
        assert data["clip_score"] > 0.0
        assert data["processing_time_ms"] > 0.0
        assert data["model_used"] == "fast"
        assert data.get("error") is None
        
        # Validate reasonable processing time
        total_time_seconds = end_time - start_time
        assert total_time_seconds < 30.0
        
        # Validate that API time aligns with reported processing time
        api_processing_time_ms = data["processing_time_ms"]
        assert api_processing_time_ms < total_time_seconds * 1000

    @pytest.mark.integration
    @pytest.mark.parametrize("model_config", ["fast", "accurate"])
    def test_model_config_differences(self, test_client, matched_image_caption_pairs, model_config):
        if not matched_image_caption_pairs:
            pytest.skip("No matched image-caption pairs available for integration testing")
        
        image_path, caption = matched_image_caption_pairs[0]
        
        payload = {
            "image_input": image_path,
            "text_prompt": caption,
            "model_config_name": model_config
        }
        
        response = test_client.post("/evaluator/v1/evaluation/single", json=payload)
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["model_used"] == model_config
        assert data["clip_score"] > 0.0
        assert data.get("error") is None

    @pytest.mark.integration
    def test_batch_evaluation_real_data(self, test_client, matched_image_caption_pairs):
        if len(matched_image_caption_pairs) < 2:
            pytest.skip("Need at least 2 matched pairs for batch testing")
        
        # Use first 3 pairs or all if less than 3
        pairs_to_test = matched_image_caption_pairs[:3]
        
        evaluations = []
        for image_path, caption in pairs_to_test:
            evaluations.append({
                "image_input": image_path,
                "text_prompt": caption,
                "model_config_name": "fast"
            })
        
        payload = {"evaluations": evaluations}
        
        start_time = time.time()
        response = test_client.post("/evaluator/v1/evaluation/batch", json=payload)
        end_time = time.time()
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Validate batch response structure
        assert data["total_processed"] == len(pairs_to_test)
        assert data["total_successful"] == len(pairs_to_test)
        assert data["total_failed"] == 0
        assert len(data["results"]) == len(pairs_to_test)
        assert data["total_processing_time_ms"] > 0.0
        
        # Validate individual results
        for i, result in enumerate(data["results"]):
            expected_image, expected_caption = pairs_to_test[i]
            assert result["image_input"] == expected_image
            assert result["text_prompt"] == expected_caption
            assert result["clip_score"] > 0.0
            assert result.get("error") is None
            assert result["model_used"] == "fast"
        
        # Validate performance - batch should be more efficient than sequential
        total_time_seconds = end_time - start_time
        assert total_time_seconds < len(pairs_to_test) * 10

    @pytest.mark.integration
    def test_mixed_batch_success_and_failure(self, test_client, matched_image_caption_pairs):
        """Test batch evaluation with mix of valid and invalid inputs"""
        if not matched_image_caption_pairs:
            pytest.skip("No matched image-caption pairs available for integration testing")
        
        image_path, caption = matched_image_caption_pairs[0]
        
        evaluations = [
            {
                "image_input": image_path,
                "text_prompt": caption,
                "model_config_name": "fast"
            },
            {
                "image_input": "nonexistent_file.jpg",
                "text_prompt": "This should fail",
                "model_config_name": "fast"
            },
            {
                "image_input": image_path,
                "text_prompt": caption,
                "model_config_name": "fast"
            }
        ]
        
        payload = {"evaluations": evaluations}
        response = test_client.post("/evaluator/v1/evaluation/batch", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Should process all 3, with 2 successful and 1 failed
        assert data["total_processed"] == 3
        assert data["total_successful"] == 2
        assert data["total_failed"] == 1
        
        # Validate results
        results = data["results"]
        assert len(results) == 3
        
        # First and third should succeed
        assert results[0].get("error") is None  # error field may be omitted for successful evaluations
        assert results[0]["clip_score"] > 0.0
        assert results[2].get("error") is None  # error field may be omitted for successful evaluations
        assert results[2]["clip_score"] > 0.0
        
        # Second should fail
        assert results[1]["error"] is not None
        assert results[1]["clip_score"] == 0.0

    @pytest.mark.integration
    def test_health_check_with_real_models(self, test_client):
        """Test health check loads real models successfully"""
        response = test_client.get("/evaluator/v1/evaluation/health")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert isinstance(data["model_loaded"], bool)
        assert "fast" in data["available_configs"]
        assert "accurate" in data["available_configs"]
        
        # If health check passes, model should be loaded
        if data["status"] == "healthy":
            assert data["model_loaded"] is True

    @pytest.mark.integration
    def test_model_performance_comparison(self, test_client, matched_image_caption_pairs):
        """Compare performance between fast and accurate models"""
        if not matched_image_caption_pairs:
            pytest.skip("No matched image-caption pairs available for integration testing")
        
        image_path, caption = matched_image_caption_pairs[0]
        
        results = {}
        
        for model_config in ["fast", "accurate"]:
            payload = {
                "image_input": image_path,
                "text_prompt": caption,
                "model_config_name": model_config
            }
            
            start_time = time.time()
            response = test_client.post("/evaluator/v1/evaluation/single", json=payload)
            end_time = time.time()
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            results[model_config] = {
                "clip_score": data["clip_score"],
                "processing_time_ms": data["processing_time_ms"],
                "total_time_seconds": end_time - start_time
            }
        
        assert results["fast"]["clip_score"] > 0.0
        assert results["accurate"]["clip_score"] > 0.0
        
        if results["fast"]["total_time_seconds"] > 2.0:
            assert results["fast"]["total_time_seconds"] <= results["accurate"]["total_time_seconds"] * 1.5

    @pytest.mark.integration
    def test_stress_batch_evaluation(self, test_client, matched_image_caption_pairs):
        if len(matched_image_caption_pairs) < 3:
            pytest.skip("Need at least 3 matched pairs for stress testing")
        
        evaluations = []
        target_batch_size = min(10, len(matched_image_caption_pairs) * 3)
        
        for i in range(target_batch_size):
            image_path, caption = matched_image_caption_pairs[i % len(matched_image_caption_pairs)]
            evaluations.append({
                "image_input": image_path,
                "text_prompt": caption,
                "model_config_name": "fast"
            })
        
        payload = {"evaluations": evaluations}
        
        start_time = time.time()
        response = test_client.post("/evaluator/v1/evaluation/batch", json=payload)
        end_time = time.time()
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # All evaluations should succeed with real data
        assert data["total_processed"] == target_batch_size
        assert data["total_successful"] == target_batch_size
        assert data["total_failed"] == 0
        
        # Reasonable performance for larger batch
        total_time_seconds = end_time - start_time
        assert total_time_seconds < target_batch_size * 5
        
        # Validate all results
        for result in data["results"]:
            assert result["clip_score"] > 0.0
            assert result.get("error") is None

    @pytest.mark.integration
    def test_api_documentation_accessibility(self, test_client):
        docs_response = test_client.get("/evaluator/docs")
        assert docs_response.status_code == status.HTTP_200_OK
        
        openapi_response = test_client.get("/evaluator/openapi.json")
        assert openapi_response.status_code == status.HTTP_200_OK
        
        # Validate OpenAPI schema contains our endpoints
        openapi_data = openapi_response.json()
        assert "paths" in openapi_data
        assert "/evaluator/v1/evaluation/single" in openapi_data["paths"]
        assert "/evaluator/v1/evaluation/batch" in openapi_data["paths"]
