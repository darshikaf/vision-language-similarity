import contextlib

import ray
from ray import serve

from service.log import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


# Import FastAPI app at module level to avoid circular imports
from service.main import app  # noqa: E402


@serve.deployment(
    name="vision-similarity-service",
    num_replicas="auto",
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 0,
        "memory": 4294967296,  # 4GB
    },
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_num_ongoing_requests_per_replica": 10,
    },
)
@serve.ingress(app)
class VisionSimilarityService:
    """Ray Serve deployment for vision-language similarity evaluation"""

    def __init__(self):
        """Initialize the similarity service deployment"""
        self.logger = get_logger(__name__)
        self.logger.info("Initializing Ray Serve VisionSimilarityService")


def main():
    """Main entry point for Ray Serve deployment"""

    # Ray configuration for optimal performance
    ray_config = {
        "dashboard_host": "0.0.0.0",
        "ignore_reinit_error": True,
        "_metrics_export_port": 8080,
    }

    logger.info("Initializing Ray cluster...")
    ray.init(**ray_config)

    # Shutdown existing Ray Serve to ensure clean restart
    logger.info("Cleaning up any existing Ray Serve deployments...")
    with contextlib.suppress(Exception):
        serve.shutdown()

    # Start Ray Serve with HTTP configuration
    logger.info("Starting Ray Serve...")
    serve.start(detached=False, http_options={"host": "0.0.0.0", "port": 8000, "root_path": ""})

    # Create and deploy the service
    logger.info("Deploying vision-similarity-service...")
    deployment = VisionSimilarityService.bind()
    serve.run(deployment, blocking=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error in Ray Serve deployment: {e}")
        raise
    finally:
        logger.info("Ray Serve deployment finished.")
