import pytest
from service.core import MinimalOpenCLIPEvaluator


@pytest.fixture(scope="session")
def fast_evaluator():
    """Session-scoped fixture for fast evaluator to avoid reloading model"""
    return MinimalOpenCLIPEvaluator.create_fast_evaluator()


@pytest.fixture(scope="session") 
def accurate_evaluator():
    """Session-scoped fixture for accurate evaluator"""
    return MinimalOpenCLIPEvaluator.create_accurate_evaluator()


@pytest.fixture
def evaluator_configs():
    """Available evaluator configurations"""
    return {
        "fast": {"model_name": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"},
        "accurate": {"model_name": "ViT-L-14", "pretrained": "laion2b_s32b_b82k"},
    }
