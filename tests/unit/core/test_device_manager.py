import pytest
import torch

from service.core.ml.utils.device_manager import DeviceManager


@pytest.fixture
def cpu_device():
    """CPU device for testing"""
    return torch.device("cpu")


@pytest.fixture  
def cuda_device():
    """CUDA device for testing"""
    return torch.device("cuda")


class TestDeviceManager:
    """Test device manager functionality"""

    def test_get_optimal_device_returns_valid_device(self):
        """Test device selection returns a valid torch device"""
        device = DeviceManager.get_optimal_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]

    def test_get_optimal_device_explicit_cuda(self):
        """Test explicit CUDA device selection"""
        selected_device = DeviceManager.get_optimal_device("cuda")
        assert selected_device.type == "cuda"

    def test_get_optimal_device_auto_selection(self):
        """Test automatic device selection logic"""
        device = DeviceManager.get_optimal_device()
        # Should select best available device
        assert isinstance(device, torch.device)
        
        # Verify device is actually usable
        test_tensor = torch.tensor([1.0], device=device)
        assert test_tensor.device.type == device.type

    @pytest.mark.parametrize("device_type", ["cpu", "cuda"])
    def test_get_optimal_precision_by_device_type(self, device_type):
        """Test precision selection for different device types"""
        device = torch.device(device_type)
        precision = DeviceManager.get_optimal_precision(device)
        
        if device_type == "cpu":
            assert precision == "fp32"
        elif device_type == "cuda":
            assert precision == "fp16"
        
        assert precision in ["fp16", "fp32"]





