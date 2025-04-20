"""
Integration tests for LMM Invoice Data Extraction Comparison.
Tests the interaction between configuration, validation, and storage protocols.
"""

import pytest
from pathlib import Path
import json
import yaml
from datetime import datetime
from src.config_protocol import ConfigManager, ConfigSection
from src.validation import ValidationManager
from src.result_storage import ResultStorage

@pytest.fixture
def config_manager():
    """Create a ConfigManager instance."""
    return ConfigManager()

@pytest.fixture
def validation_manager(config_manager):
    """Create a ValidationManager instance."""
    return ValidationManager(config_manager)

@pytest.fixture
def result_storage(config_manager, validation_manager):
    """Create a ResultStorage instance."""
    return ResultStorage(config_manager, validation_manager)

def test_config_loading(config_manager):
    """Test configuration loading and validation."""
    # Test model configurations
    llama_config = config_manager.get_model_config("llama_vision")
    doctr_config = config_manager.get_model_config("doctr")
    
    # Verify required sections
    assert "architecture" in llama_config
    assert "hardware" in llama_config
    assert "inference" in llama_config
    assert "performance" in llama_config
    assert "error_handling" in llama_config
    assert "model_params" in llama_config
    
    # Verify specific values
    assert llama_config["architecture"]["name"] == "llama_vision"
    assert doctr_config["architecture"]["name"] == "doctr"
    assert llama_config["hardware"]["min_gpu_memory_gb"] == 16
    assert doctr_config["hardware"]["min_gpu_memory_gb"] == 8

def test_validation_flow(validation_manager):
    """Test the validation flow for models and results."""
    # Test model validation
    llama_config = validation_manager.config_manager.get_model_config("llama_vision")
    validation_manager.validate_model("llama_vision", llama_config)
    
    # Test result validation
    test_result = {
        "work_order_number": "WO12345",
        "total_cost": "100.00",
        "confidence": 0.85,
        "processing_time": 2.5,
        "timestamp": datetime.now().isoformat(),
        "gpu_utilization": 0.6,
        "inference_time": 2.0,
        "accuracy": 0.85
    }
    validation_manager.validate_result(test_result, "llama_vision")

def test_storage_operations(result_storage):
    """Test result storage operations."""
    # Test storing a result
    test_result = {
        "work_order_number": "WO12345",
        "total_cost": "100.00",
        "confidence": 0.85,
        "processing_time": 2.5,
        "timestamp": datetime.now().isoformat(),
        "gpu_utilization": 0.6,
        "inference_time": 2.0,
        "accuracy": 0.85
    }
    
    # Store result
    result_storage.store_result(test_result, "llama_vision")
    
    # Test retrieving results
    results = result_storage.get_model_results("llama_vision")
    assert len(results) > 0
    assert results[0]["work_order_number"] == "WO12345"
    
    # Test performance metrics
    metrics = result_storage.get_performance_metrics("llama_vision")
    assert "total_results" in metrics
    assert "average_confidence" in metrics
    assert "average_processing_time" in metrics
    assert "success_rate" in metrics

def test_error_handling(validation_manager, result_storage):
    """Test error handling in validation and storage."""
    # Test invalid model configuration
    with pytest.raises(ValidationError):
        validation_manager.validate_model("invalid_model", {})
    
    # Test invalid result
    invalid_result = {
        "work_order_number": "WO12345",
        "confidence": 1.5,  # Invalid confidence
        "processing_time": -1.0  # Invalid processing time
    }
    with pytest.raises(ResultValidationError):
        validation_manager.validate_result(invalid_result, "llama_vision")
    
    # Test storage errors
    with pytest.raises(StorageError):
        result_storage.get_result("nonexistent_model", "20230101_000000")

def test_performance_tracking(result_storage):
    """Test performance tracking functionality."""
    # Store multiple results
    for i in range(3):
        result = {
            "work_order_number": f"WO{i}",
            "total_cost": "100.00",
            "confidence": 0.7 + (i * 0.1),
            "processing_time": 2.0 + (i * 0.5),
            "timestamp": datetime.now().isoformat(),
            "gpu_utilization": 0.6,
            "inference_time": 2.0,
            "accuracy": 0.7 + (i * 0.1)
        }
        result_storage.store_result(result, "llama_vision")
    
    # Verify performance metrics
    metrics = result_storage.get_performance_metrics("llama_vision")
    assert metrics["total_results"] == 3
    assert 0.7 <= metrics["average_confidence"] <= 0.9
    assert 2.0 <= metrics["average_processing_time"] <= 3.0
    assert metrics["success_rate"] > 0

def test_export_functionality(result_storage):
    """Test result export functionality."""
    # Store a test result
    test_result = {
        "work_order_number": "WO12345",
        "total_cost": "100.00",
        "confidence": 0.85,
        "processing_time": 2.5,
        "timestamp": datetime.now().isoformat(),
        "gpu_utilization": 0.6,
        "inference_time": 2.0,
        "accuracy": 0.85
    }
    result_storage.store_result(test_result, "llama_vision")
    
    # Test JSON export
    result_storage.export_results("llama_vision", "json")
    export_file = result_storage.results_dir / "llama_vision_export.json"
    assert export_file.exists()
    
    # Test YAML export
    result_storage.export_results("llama_vision", "yaml")
    export_file = result_storage.results_dir / "llama_vision_export.yaml"
    assert export_file.exists()
    
    # Test invalid format
    with pytest.raises(StorageError):
        result_storage.export_results("llama_vision", "invalid") 