"""
Doctr model implementation with performance optimization.

This module implements the Doctr model for extracting structured data
from invoice images. It includes model loading with quantization support,
inference functions, and output parsing.
"""

from typing import Dict, Any, List, Optional, Union
import torch
from doctr.models import ocr_predictor
from PIL import Image
import logging
from pathlib import Path
import time
from transformers import BitsAndBytesConfig, AutoProcessor
import numpy as np

from .common import (
    preprocess_image,
    parse_model_output,
    validate_model_output
)
from ..data_utils import DataConfig
from ..results_logging import ModelResponse
from .base_model import BaseModel
from .config_protocol import ConfigManager, ConfigSection
from .validation import ValidationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DoctrModel(BaseModel):
    """Doctr model implementation with performance optimization."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config_manager: ConfigManager,
        validation_manager: ValidationManager,
        quantization: int = 32
    ):
        """Initialize Doctr model with performance optimization.
        
        Args:
            model_path: Path to model weights
            config_manager: Configuration manager instance
            validation_manager: Validation manager instance
            quantization: Bit width for quantization (4, 8, 16, 32)
        """
        super().__init__(
            model_path=model_path,
            config_manager=config_manager,
            validation_manager=validation_manager,
            quantization=quantization
        )
        
    @property
    def name(self) -> str:
        """Get model name."""
        return "doctr"
        
    def _load_model_weights(self, loading_params: Dict[str, Any]) -> Any:
        """Load model weights with specified parameters."""
        return ocr_predictor(
            det_arch='db_resnet50',
            reco_arch='crnn_vgg16_bn',
            pretrained=True,
            **loading_params
        )
        
    def _load_processor(self) -> Any:
        """Load model processor."""
        return None  # Doctr handles processing internally
        
    def _process_batch(
        self,
        batch: List[Union[str, Path, Image.Image]],
        prompt: str,
        field_type: str
    ) -> List[Dict[str, Any]]:
        """Process a single batch of inputs."""
        try:
            # Convert paths to images if needed
            images = []
            for item in batch:
                if isinstance(item, (str, Path)):
                    image = Image.open(item)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    images.append(image)
                else:
                    images.append(item)
                    
            # Process images
            results = []
            for image in images:
                # Run inference
                with torch.no_grad():
                    doc = self.model([image])
                    
                # Extract text
                output_text = " ".join([word for page in doc.pages for word in page.words])
                
                # Parse output
                parsed_value = self._parse_output(output_text, field_type)
                
                results.append({
                    'test_parameters': {
                        'model': self.name,
                        'quantization': self.quantization
                    },
                    'model_response': {
                        'output': output_text,
                        'parsed_value': parsed_value
                    }
                })
                
            return results
            
        except Exception as e:
            raise RuntimeError(f"Failed to process batch: {str(e)}")
            
    def _parse_output(self, output_text: str, field_type: str) -> Any:
        """Parse model output based on field type."""
        # Implementation would depend on specific field types
        return output_text

def download_doctr_model(model_path: Path, repo_id: str) -> bool:
    """Download Doctr model from HuggingFace.
    
    Args:
        model_path: Path where model should be downloaded
        repo_id: HuggingFace repository ID
        
    Returns:
        bool: True if download successful, False otherwise
        
    Raises:
        RuntimeError: If download fails
    """
    try:
        from huggingface_hub import snapshot_download
        
        if model_path.exists():
            logger.info(f"Model already exists at {model_path}")
            return True
            
        logger.info(f"Downloading Doctr model from {repo_id}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_path),
            local_dir_use_symlinks=False
        )
        return True
        
    except Exception as e:
        logger.error(f"Failed to download Doctr model: {str(e)}")
        return False

def load_model(model_name: str, quantization: int, models_dir: Path, config: dict) -> DoctrModel:
    """Load Doctr model with specified quantization.
    
    Args:
        model_name: Name of the model (must be 'doctr')
        quantization: Bit width for quantization (4, 8, 16, 32)
        models_dir: Path to models directory
        config: Model configuration from YAML
        
    Returns:
        Loaded DoctrModel instance
        
    Raises:
        ValueError: If model_name is not 'doctr'
        FileNotFoundError: If model directory doesn't exist
    """
    if model_name != "doctr":
        raise ValueError(f"Invalid model name for Doctr loader: {model_name}")
        
    model_path = models_dir / "doctr"
    
    # Download model if needed
    if not model_path.exists():
        if not download_doctr_model(model_path, config['repo_id']):
            raise RuntimeError(f"Failed to download Doctr model to {model_path}")
        
    return DoctrModel(
        model_path=model_path,
        quantization=quantization
    )

def process_image_wrapper(
    model: DoctrModel,
    prompt_template: str,
    image_path: Union[str, Path],
    field_type: str,
    config: DataConfig
) -> Dict[str, Any]:
    """Wrapper function to process image with Doctr model.
    
    Args:
        model: Loaded DoctrModel instance
        prompt_template: Prompt template to use (not used for Doctr)
        image_path: Path to input image
        field_type: Type of field to extract
        config: Data configuration
        
    Returns:
        Dictionary with test parameters and model response
    """
    # Get model response
    response = model.process_image(
        image_path=image_path,
        prompt=prompt_template,
        field_type=field_type,
        config=config
    )
    
    # Update test parameters with prompt strategy
    response['test_parameters']['prompt_strategy'] = prompt_template
    
    return response 