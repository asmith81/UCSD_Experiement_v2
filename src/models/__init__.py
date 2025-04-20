"""
Model registry for LMM invoice data extraction.

This module provides a centralized registry for model types and a factory function
for creating model instances. It follows a functional programming approach with
clear type definitions and simple data structures.
"""

from typing import Dict, Type, Callable, Any, List, Union, TypedDict, Optional
from dataclasses import dataclass
import yaml
import os
from pathlib import Path
import logging
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Export specific functions and classes
__all__ = [
    'load_model',
    'run_inference',
    'process_image',
    'ModelConfig',
    'ModelRegistry',
    'ImageData',
    'ModelResponse',
    'EvaluationResult',
    'ResultStructure'
]

class ImageData(TypedDict):
    """Structure for image data."""
    path: Path
    data: np.ndarray
    format: str
    size: tuple[int, int]

class ModelResponse(TypedDict):
    """Structure for model response data."""
    raw_value: str
    normalized_value: str
    confidence: float
    processing_time: float

class EvaluationResult(TypedDict):
    """Structure for evaluation results."""
    normalized_match: bool
    cer: float
    error_category: str

class ResultStructure(TypedDict):
    """Structure for complete result file."""
    test_parameters: Dict[str, Any]  # Contains model, quantization, prompt_strategy
    model_response: ModelResponse
    evaluation: Dict[str, Dict[str, Any]]  # Contains work_order_number and total_cost metrics

@dataclass
class ModelConfig:
    """Configuration for a model instance."""
    name: str
    quantization_levels: List[int]
    architecture: Dict[str, Any]
    framework: Dict[str, Any]
    hardware: Dict[str, Any]
    config_path: Path
    prompt_strategy: str

class ModelRegistry:
    """Registry for model types and their factory functions."""
    
    def __init__(self):
        self._models: Dict[str, Type] = {}
        self._factories: Dict[str, Callable] = {}
    
    def register(self, name: str, model_class: Type, factory: Callable) -> None:
        """Register a model type and its factory function.
        
        Args:
            name: Unique identifier for the model type
            model_class: The model class to register
            factory: Function that creates model instances
            
        Raises:
            ValueError: If model is already registered
        """
        if name in self._models:
            raise ValueError(f"Model {name} is already registered")
            
        self._models[name] = model_class
        self._factories[name] = factory
        logger.info(f"Registered model: {name}")
    
    def create_model(self, config: ModelConfig, quantization: int) -> Any:
        """Create a model instance using the registered factory.
        
        Args:
            config: Model configuration including name and parameters
            quantization: Quantization level to use
            
        Returns:
            An instance of the requested model type
            
        Raises:
            ValueError: If model type is not registered or quantization is invalid
        """
        if config.name not in self._factories:
            raise ValueError(f"Model type {config.name} not registered")
            
        if str(quantization) not in config.quantization_levels:
            raise ValueError(f"Invalid quantization level {quantization} for model {config.name}")
        
        logger.info(f"Creating model {config.name} with quantization {quantization}")
        return self._factories[config.name](config, quantization)

# Create global registry instance
registry = ModelRegistry()

def load_model_config(config_path: Union[str, Path]) -> ModelConfig:
    """Load model configuration from YAML file.
    
    Args:
        config_path: Path to the model configuration file
        
    Returns:
        ModelConfig instance with loaded configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If config structure is invalid
    """
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Model config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        # Validate required fields
        required_fields = ['name', 'quantization_levels', 'architecture', 
                         'framework', 'hardware', 'prompt_strategy']
        missing_fields = [field for field in required_fields if field not in config_data]
        if missing_fields:
            raise ValueError(f"Model config missing required fields: {missing_fields}")
        
        return ModelConfig(
            name=config_data['name'],
            quantization_levels=[int(q) for q in config_data['quantization_levels']],
            architecture=config_data['architecture'],
            framework=config_data['framework'],
            hardware=config_data['hardware'],
            config_path=config_path,
            prompt_strategy=config_data['prompt_strategy']
        )
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing model config {config_path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading model config {config_path}: {str(e)}")
        raise

def load_model(model_name: str, quantization: int) -> Any:
    """
    Load a model with specific quantization.
    
    Args:
        model_name: Name of the model to load
        quantization: Quantization level to use
        
    Returns:
        Loaded model instance
        
    Raises:
        ImportError: If model module not found
        ValueError: If quantization level not supported
    """
    try:
        # Import model-specific module
        module = __import__(f"src.models.{model_name}", fromlist=['load_model'])
        return module.load_model(model_name, quantization)
    except ImportError as e:
        logger.error(f"Failed to import model {model_name}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise

def process_image(model: Any, image: Union[ImageData, Image.Image], prompt: str) -> ModelResponse:
    """
    Process an image with the model using a specific prompt.
    
    Args:
        model: Loaded model instance
        image: Image to process (either ImageData or PIL Image)
        prompt: Prompt template to use
        
    Returns:
        ModelResponse containing extraction results
        
    Raises:
        ImportError: If model module not found
        ValueError: If processing fails
    """
    try:
        # Get model name from instance
        model_name = getattr(model, 'name', None)
        if not model_name:
            raise ValueError("Model instance missing 'name' attribute")
            
        # Import model-specific module
        module = __import__(f"src.models.{model_name}", fromlist=['process_image'])
        return module.process_image(model, image, prompt)
    except ImportError as e:
        logger.error(f"Failed to import model module: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to process image: {str(e)}")
        raise

def run_inference(model: Any, prompt: str) -> Dict[str, Any]:
    """
    Run inference with a model using a prompt.
    
    Args:
        model: Loaded model instance
        prompt: Prompt to use for inference
        
    Returns:
        Dictionary containing inference results
        
    Raises:
        ValueError: If model is invalid or inference fails
    """
    try:
        # Get model name from instance
        model_name = getattr(model, 'name', None)
        if not model_name:
            raise ValueError("Model instance missing 'name' attribute")
            
        # Import model-specific module
        module = __import__(f"src.models.{model_name}", fromlist=['process_image'])
        
        # Create mock image for testing
        from PIL import Image
        import numpy as np
        mock_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        
        # Run inference
        result = module.process_image(model, mock_image, prompt)
        
        # Format response
        response = {
            "raw_value": result.get("raw_text", ""),
            "normalized_value": result.get("normalized_value", ""),
            "confidence": result.get("confidence", 0.0),
            "processing_time": result.get("processing_time", 0.0)
        }
        
        return response
        
    except ImportError as e:
        logger.error(f"Failed to import model module: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to run inference: {str(e)}")
        raise 