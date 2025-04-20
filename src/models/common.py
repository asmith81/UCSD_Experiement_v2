"""
Common utilities for model implementations.

This module provides shared functionality for all model implementations:
- Image preprocessing
- Model loading utilities
- Output parsing common code
"""

from typing import Dict, Any, Tuple, Optional, TypedDict, Union
import torch
import numpy as np
from PIL import Image
import logging
from pathlib import Path
from ..data_utils import DataConfig
from ..results_logging import ModelOutput, ModelResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_image(image_path: Union[str, Path], config: DataConfig) -> Image.Image:
    """Preprocess an image for model input.
    
    Args:
        image_path: Path to the image file
        config: Data configuration containing image processing settings
        
    Returns:
        Preprocessed PIL Image
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image processing fails
    """
    try:
        # Convert to Path object if string
        image_path = Path(image_path)
        
        # If path is relative, join with image_dir
        if not image_path.is_absolute():
            # Remove any leading 'data/images' from the path to avoid duplication
            path_str = str(image_path)
            if path_str.startswith('data/images/'):
                path_str = path_str[12:]  # Remove 'data/images/'
            image_path = config.image_dir / path_str
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Load and process image
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode not in config.supported_formats:
            image = image.convert('RGB')
            
        # Resize if needed
        if config.max_image_size:
            # Calculate new size while maintaining aspect ratio
            ratio = min(config.max_image_size / max(image.size), 1.0)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            if new_size != image.size:
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                
        return image
        
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {str(e)}")

def load_model_weights(
    model: torch.nn.Module,
    weights_path: Path,
    strict: bool = True
) -> None:
    """Load model weights from checkpoint.
    
    Args:
        model: Model to load weights into
        weights_path: Path to weights file
        strict: Whether to strictly enforce matching keys
        
    Raises:
        FileNotFoundError: If weights file doesn't exist
        RuntimeError: If loading fails
    """
    try:
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=strict)
        logger.info(f"Loaded weights from {weights_path}")
        
    except Exception as e:
        logger.error(f"Error loading weights from {weights_path}: {str(e)}")
        raise

def parse_model_output(
    output: Any,
    field_type: str,
    confidence_threshold: float = 0.5
) -> ModelOutput:
    """Parse model output into standardized format.
    
    Args:
        output: Raw model output
        field_type: Type of field being extracted
        confidence_threshold: Minimum confidence for accepting prediction
        
    Returns:
        ModelOutput dictionary with parsed values
        
    Raises:
        ValueError: If parsing fails
    """
    try:
        result: ModelOutput = {
            'raw_text': str(output),
            'parsed_value': None,
            'normalized_value': None
        }
        
        # Extract numeric values for total_cost
        if field_type == 'total_cost':
            # Remove currency symbols and whitespace
            clean_text = ''.join(c for c in str(output) if c.isdigit() or c == '.')
            if clean_text:
                try:
                    result['parsed_value'] = clean_text
                    result['normalized_value'] = float(clean_text)
                except ValueError:
                    result['parsed_value'] = str(output)
                    result['normalized_value'] = 0.0
                    
        # Extract alphanumeric values for work_order_number
        elif field_type == 'work_order_number':
            # Remove non-alphanumeric characters
            clean_text = ''.join(c for c in str(output) if c.isalnum())
            if clean_text:
                result['parsed_value'] = clean_text
                result['normalized_value'] = clean_text
            else:
                result['parsed_value'] = str(output)
                result['normalized_value'] = ""
                
        return result
        
    except Exception as e:
        logger.error(f"Error parsing model output: {str(e)}")
        raise

def validate_model_output(
    parsed_output: ModelOutput,
    field_type: str
) -> bool:
    """Validate parsed model output.
    
    Args:
        parsed_output: Parsed output dictionary
        field_type: Type of field being extracted
        
    Returns:
        Whether the output is valid
        
    Raises:
        ValueError: If validation fails
    """
    try:
        # Check required fields
        required_fields = ['raw_text', 'parsed_value', 'normalized_value']
        if not all(field in parsed_output for field in required_fields):
            return False
            
        # Field-specific validation
        if field_type == 'total_cost':
            if not isinstance(parsed_output['normalized_value'], (int, float)):
                return False
            if parsed_output['normalized_value'] < 0:
                return False
                
        elif field_type == 'work_order_number':
            if not isinstance(parsed_output['normalized_value'], str):
                return False
            if not parsed_output['normalized_value'].isalnum():
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Error validating model output: {str(e)}")
        raise 