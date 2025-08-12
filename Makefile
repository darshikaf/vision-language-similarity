export PROJECT_NAME := vision_language_similarity_service
export PROJECT_ALIAS := evaluator

include scripts/Makefile

.PHONY: run-local-otel stop-otel clean-otel test-otel

run-local-otel:
	@echo "Starting observability stack..."
	@echo "Grafana: http://localhost:3000 (admin/grafana)"
	@echo "Prometheus: http://localhost:9090" 
	@echo "Jaeger: http://localhost:16686"
	@echo "Loki: http://localhost:3100"
	docker-compose -f docker/observability/docker-compose.otel.yml up --build --force-recreate

stop-otel:
	docker-compose -f docker/observability/docker-compose.otel.yml down

clean-otel:
	docker-compose -f docker/observability/docker-compose.otel.yml down -v
	docker system prune -f

test-otel:
	curl -s http://localhost:8000/evaluator/health | jq .
	curl -s -X POST http://localhost:8000/evaluator/v1/evaluation/single \
	  -H "Content-Type: application/json" \
	  -d '{"image_input": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==", "text_prompt": "test", "model_config_name": "fast"}' | jq .