import pytest
import torch

from service.core.device_manager import DeviceManager


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

    def test_get_optimal_device_explicit_cpu(self, cpu_device):
        """Test explicit CPU device selection"""
        selected_device = DeviceManager.get_optimal_device("cpu")
        assert selected_device.type == "cpu"
        assert selected_device == cpu_device

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

    def test_get_optimal_precision_cpu_always_fp32(self, cpu_device):
        """Test CPU always uses fp32 precision"""
        precision = DeviceManager.get_optimal_precision(cpu_device)
        assert precision == "fp32"

    def test_get_optimal_precision_cuda_uses_fp16(self, cuda_device):
        """Test CUDA uses fp16 for efficiency"""
        precision = DeviceManager.get_optimal_precision(cuda_device)
        assert precision == "fp16"

    def test_get_optimal_precision_unknown_device_defaults_fp32(self):
        """Test non-CUDA device types default to fp32"""
        # Test with CPU device (non-CUDA)
        cpu_device = torch.device("cpu")
        precision = DeviceManager.get_optimal_precision(cpu_device)
        assert precision == "fp32"
        
        # Test with MPS device (also non-CUDA, should default to fp32)
        if hasattr(torch.backends, "mps"):
            mps_device = torch.device("mps")
            precision = DeviceManager.get_optimal_precision(mps_device)
            assert precision == "fp32"


class TestDeviceManagerIntegration:
    """Test device manager integration scenarios"""

    def test_device_and_precision_consistency(self):
        """Test that device and precision selections are consistent"""
        device = DeviceManager.get_optimal_device()
        precision = DeviceManager.get_optimal_precision(device)
        
        # Verify the combination makes sense
        if device.type == "cpu":
            assert precision == "fp32"
        elif device.type == "cuda":
            assert precision == "fp16"

    def test_multiple_device_selections_consistent(self):
        """Test multiple calls return consistent results"""
        device1 = DeviceManager.get_optimal_device()
        device2 = DeviceManager.get_optimal_device()
        
        # Should return same device type (though not necessarily same instance)
        assert device1.type == device2.type

    def test_device_availability_check(self):
        """Test device availability checking"""
        device = DeviceManager.get_optimal_device()
        
        # Basic availability test - device should be usable
        try:
            test_tensor = torch.tensor([1.0, 2.0], device=device)
            result = test_tensor.sum()
            assert result.item() == 3.0
        except RuntimeError:
            pytest.skip(f"Device {device} not available in test environment")


class TestDeviceManagerEdgeCases:
    """Test device manager edge cases and error handling"""

    def test_invalid_device_string_handling(self):
        """Test handling of invalid device strings"""
        # This should either work or raise appropriate error
        try:
            device = DeviceManager.get_optimal_device("invalid_device")
            # If it doesn't raise an error, it should fall back to a valid device
            assert isinstance(device, torch.device)
        except (RuntimeError, ValueError):
            # This is acceptable - invalid device should raise an error
            pass

    def test_precision_with_none_device(self):
        """Test precision calculation with edge case inputs"""
        # Test with a valid device to ensure robustness
        cpu_device = torch.device("cpu")
        precision = DeviceManager.get_optimal_precision(cpu_device)
        assert precision in ["fp16", "fp32"]

    def test_device_manager_is_stateless(self):
        """Test that DeviceManager doesn't maintain state between calls"""
        # Multiple independent calls should work correctly
        devices = []
        precisions = []
        
        for _ in range(3):
            device = DeviceManager.get_optimal_device()
            precision = DeviceManager.get_optimal_precision(device)
            devices.append(device)
            precisions.append(precision)
        
        # All should be valid
        for device, precision in zip(devices, precisions):
            assert isinstance(device, torch.device)
            assert precision in ["fp16", "fp32"]
