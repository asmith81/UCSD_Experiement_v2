"""
Validation module for LMM Invoice Data Extraction Comparison.
Implements validation across all protocols.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import yaml
from dataclasses import dataclass
from enum import Enum
from .config_protocol import ConfigManager, ConfigSection, ConfigValidationError

class ValidationError(Exception):
    """Base class for validation errors."""
    pass

class ModelValidationError(ValidationError):
    """Raised when model validation fails."""
    pass

class ImageValidationError(ValidationError):
    """Raised when image validation fails."""
    pass

class ResultValidationError(ValidationError):
    """Raised when result validation fails."""
    pass

class ValidationManager:
    """Manages validation across all protocols."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize with configuration manager."""
        self.config_manager = config_manager
        
    def validate_model(self, model_name: str, model_config: Dict[str, Any]) -> None:
        """Validate model configuration and requirements."""
        try:
            # Get model configuration
            config = self.config_manager.get_model_config(model_name)
            
            # Validate hardware requirements
            self._validate_hardware_requirements(config)
            
            # Validate inference parameters
            self._validate_inference_parameters(config)
            
            # Validate model-specific parameters
            self._validate_model_parameters(config)
            
        except ConfigValidationError as e:
            raise ModelValidationError(f"Model configuration validation failed: {str(e)}")
            
    def validate_image(self, image_path: Union[str, Path], model_name: str) -> None:
        """Validate image for processing."""
        try:
            # Get model configuration
            config = self.config_manager.get_model_config(model_name)
            
            # Validate image format
            self._validate_image_format(image_path, config)
            
            # Validate image size
            self._validate_image_size(image_path, config)
            
            # Validate image content
            self._validate_image_content(image_path)
            
        except ConfigValidationError as e:
            raise ImageValidationError(f"Image validation failed: {str(e)}")
            
    def validate_result(self, result: Dict[str, Any], model_name: str) -> None:
        """Validate processing result."""
        try:
            # Get model configuration
            config = self.config_manager.get_model_config(model_name)
            
            # Validate result structure
            self._validate_result_structure(result)
            
            # Validate result content
            self._validate_result_content(result, config)
            
            # Validate performance metrics
            self._validate_performance_metrics(result)
            
        except ConfigValidationError as e:
            raise ResultValidationError(f"Result validation failed: {str(e)}")
            
    def _validate_hardware_requirements(self, config: Dict[str, Any]) -> None:
        """Validate hardware requirements."""
        required_fields = ["min_gpu_memory_gb", "recommended_gpu_memory_gb", "supported_devices"]
        for field in required_fields:
            if field not in config["hardware"]:
                raise ConfigValidationError(f"Missing hardware requirement: {field}")
                
    def _validate_inference_parameters(self, config: Dict[str, Any]) -> None:
        """Validate inference parameters."""
        required_fields = ["max_new_tokens", "do_sample", "temperature", "batch_size"]
        for field in required_fields:
            if field not in config["inference"]:
                raise ConfigValidationError(f"Missing inference parameter: {field}")
                
    def _validate_model_parameters(self, config: Dict[str, Any]) -> None:
        """Validate model-specific parameters."""
        required_fields = ["image_processor", "max_image_size", "image_format"]
        for field in required_fields:
            if field not in config["model_params"]:
                raise ConfigValidationError(f"Missing model parameter: {field}")
                
    def _validate_image_format(self, image_path: Path, config: Dict[str, Any]) -> None:
        """Validate image format."""
        # Implementation would check image format against config["model_params"]["image_format"]
        pass
        
    def _validate_image_size(self, image_path: Path, config: Dict[str, Any]) -> None:
        """Validate image size."""
        # Implementation would check image dimensions against config["model_params"]["max_image_size"]
        pass
        
    def _validate_image_content(self, image_path: Path) -> None:
        """Validate image content."""
        # Implementation would check image content (e.g., not corrupted)
        pass
        
    def _validate_result_structure(self, result: Dict[str, Any]) -> None:
        """Validate result structure."""
        required_fields = ["work_order_number", "total_cost", "confidence", "processing_time"]
        for field in required_fields:
            if field not in result:
                raise ConfigValidationError(f"Missing result field: {field}")
                
    def _validate_result_content(self, result: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Validate result content."""
        # Validate confidence score
        if not 0 <= result["confidence"] <= 1:
            raise ConfigValidationError("Confidence score must be between 0 and 1")
            
        # Validate processing time
        if result["processing_time"] > config["performance"]["inference_timeout_seconds"]:
            raise ConfigValidationError("Processing time exceeds timeout")
            
    def _validate_performance_metrics(self, result: Dict[str, Any]) -> None:
        """Validate performance metrics."""
        required_metrics = ["gpu_utilization", "inference_time", "accuracy"]
        for metric in required_metrics:
            if metric not in result:
                raise ConfigValidationError(f"Missing performance metric: {metric}") 