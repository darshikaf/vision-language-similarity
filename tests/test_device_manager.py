import torch
from service.core import DeviceManager

def test_get_optimal_device():
    """Test device selection"""
    device = DeviceManager.get_optimal_device()
    assert isinstance(device, torch.device)

def test_get_optimal_device_explicit():
    """Test explicit device selection"""
    cpu_device = DeviceManager.get_optimal_device("cpu")
    assert cpu_device.type == "cpu"

def test_get_optimal_precision():
    """Test precision selection based on device"""
    cpu_device = torch.device("cpu")
    assert DeviceManager.get_optimal_precision(cpu_device) == "fp32"
    
    cuda_device = torch.device("cuda")
    assert DeviceManager.get_optimal_precision(cuda_device) == "fp16"
