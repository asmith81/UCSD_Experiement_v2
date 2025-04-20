"""
Tests for the performance optimization module.
"""

import pytest
from datetime import datetime, timedelta
import torch
from src.performance_optimizer import PerformanceOptimizer
from src.config_protocol import ConfigManager
from src.validation import ValidationManager

@pytest.fixture
def config_manager():
    """Create a ConfigManager instance."""
    return ConfigManager()

@pytest.fixture
def validation_manager(config_manager):
    """Create a ValidationManager instance."""
    return ValidationManager(config_manager)

@pytest.fixture
def optimizer(config_manager, validation_manager):
    """Create a PerformanceOptimizer instance."""
    return PerformanceOptimizer(config_manager, validation_manager)

def test_optimal_device_detection(optimizer):
    """Test optimal device detection."""
    device = optimizer._get_optimal_device()
    assert device in ["cuda", "cpu"]
    
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb >= optimizer.max_memory_gb:
            assert device == "cuda"
        else:
            assert device == "cpu"
    else:
        assert device == "cpu"

def test_model_loading_optimization(optimizer):
    """Test model loading optimization."""
    loading_params = optimizer.optimize_model_loading("llama_vision")
    
    assert "device_map" in loading_params
    assert "torch_dtype" in loading_params
    assert "low_cpu_mem_usage" in loading_params
    
    if optimizer.device == "cuda":
        assert loading_params["torch_dtype"] == torch.float16
    else:
        assert loading_params["torch_dtype"] == torch.float32

def test_batch_processing_optimization(optimizer):
    """Test batch processing optimization."""
    # Create test inputs
    inputs = [f"input_{i}" for i in range(10)]
    
    # Test batch optimization
    batches = optimizer.optimize_batch_processing("llama_vision", inputs)
    
    assert len(batches) > 0
    assert all(len(batch) > 0 for batch in batches)
    assert sum(len(batch) for batch in batches) == len(inputs)

def test_memory_optimization(optimizer):
    """Test memory optimization."""
    # Store initial memory usage
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated()
        
        # Run memory optimization
        optimizer.optimize_memory_usage()
        
        # Verify memory was cleared
        assert torch.cuda.memory_allocated() <= initial_memory

def test_performance_tracking(optimizer):
    """Test performance tracking."""
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1)
    
    metrics = optimizer.track_performance(
        start_time=start_time,
        end_time=end_time,
        gpu_utilization=0.5,
        memory_usage=0.3
    )
    
    assert "processing_time" in metrics
    assert "gpu_utilization" in metrics
    assert "memory_usage" in metrics
    assert "throughput" in metrics
    
    assert metrics["processing_time"] == 1.0
    assert metrics["gpu_utilization"] == 0.5
    assert metrics["memory_usage"] == 0.3
    assert metrics["throughput"] == 1.0

def test_performance_recommendations(optimizer):
    """Test performance recommendations."""
    recommendations = optimizer.get_performance_recommendations("llama_vision")
    
    assert "device" in recommendations
    assert "batch_size" in recommendations
    assert "quantization" in recommendations
    assert "memory_optimization" in recommendations
    assert "suggested_improvements" in recommendations
    
    assert isinstance(recommendations["suggested_improvements"], list)

def test_error_handling(optimizer):
    """Test error handling."""
    # Test invalid model name
    with pytest.raises(RuntimeError):
        optimizer.optimize_model_loading("invalid_model")
        
    # Test invalid batch processing
    with pytest.raises(RuntimeError):
        optimizer.optimize_batch_processing("invalid_model", [])
        
    # Test invalid recommendations
    with pytest.raises(RuntimeError):
        optimizer.get_performance_recommendations("invalid_model") 