export PROJECT_NAME := vision_language_similarity_service
export PROJECT_ALIAS := evaluator

include scripts/Makefile

.PHONY: run-local-otel stop-otel clean-otel test-otel build-ray-base build-ray run-local-ray stop-ray clean-ray test-ray

run-local-otel:
	@echo "Starting observability stack..."
	@echo "Grafana: http://localhost:3000 (admin/grafana)"
	@echo "Prometheus: http://localhost:9090"
	@echo "Jaeger: http://localhost:16686"
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

build-ray-base:
	scripts/build.sh build-ray-base

build-ray: build-ray-base
	scripts/build.sh build-ray

run-local-ray: build-ray
	@echo "Starting Ray Serve stack..."
	@echo "Service: http://localhost:8000/evaluator/docs"
	@echo "Ray Dashboard: http://localhost:8265"
	@echo "Health Check: http://localhost:8000/evaluator/health"
	@echo ""
	@echo "Starting Ray Serve container..."
	docker run --rm -p 8000:8000 -p 8265:8265 --shm-size=4gb \
		-e RAY_SERVE_ENABLE_SCALING=1 \
		-e RAY_DISABLE_DOCKER_CPU_WARNING=1 \
		-e RAY_DEDUP_LOGS=0 \
		-e RAY_OBJECT_STORE_ALLOW_SLOW=1 \
		-e RAY_memory_monitor_refresh_ms=0 \
		-e RAY_task_queue_timeout_ms=100 \
		local/vision_language_similarity_service:ray.latest \
		python service/ray_main.py

stop-ray:
	@echo "Stopping Ray Serve containers..."
	docker ps -q --filter "ancestor=local/vision_language_similarity_service:ray.latest" | xargs -r docker stop
	@echo "Ray Serve containers stopped."

clean-ray:
	@echo "Cleaning Ray Serve containers and images..."
	docker ps -q --filter "ancestor=local/vision_language_similarity_service:ray.latest" | xargs -r docker stop
	docker system prune -f
	@echo "Ray Serve cleanup complete."

test-ray:
	curl -s http://localhost:8000/evaluator/health | jq .
	curl -s -X POST http://localhost:8000/evaluator/v1/evaluation/single \
	  -H "Content-Type: application/json" \
	  -d '{"image_input": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==", "text_prompt": "test", "model_config_name": "fast"}' | jq .