import pytest
from unittest.mock import Mock, patch


@pytest.fixture
def disable_metrics():
    """Disable metrics recording for unit tests"""
    with patch('service.evaluation.handler.get_metrics_middleware') as mock_middleware:
        mock_middleware.side_effect = RuntimeError("Metrics not available")
        yield
