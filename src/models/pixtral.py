"""
Pixtral-12B model implementation for invoice data extraction.

This module implements the Pixtral-12B model for extracting structured data
from invoice images. It includes model loading with quantization support,
inference functions, and output parsing.
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import logging
from pathlib import Path
import time
from transformers import BitsAndBytesConfig
import json
from .base_model import BaseModel
from src.config_protocol import ConfigManager, ConfigSection
from src.validation import ValidationManager

from .common import (
    preprocess_image,
    parse_model_output,
    validate_model_output
)
from ..data_utils import DataConfig
from ..results_logging import ModelResponse
from .loader import get_model_path, get_hf_token

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_cer(pred: str, true: str) -> float:
    """
    Calculate Character Error Rate between predicted and true values.
    
    Args:
        pred: Predicted string
        true: True string
        
    Returns:
        Character Error Rate (0.0 to 1.0)
    """
    if not true:
        return 1.0  # Maximum error if true value is empty
        
    # Levenshtein distance calculation
    if len(pred) < len(true):
        return calculate_cer(true, pred)
        
    if len(true) == 0:
        return len(pred)
        
    previous_row = range(len(true) + 1)
    for i, c1 in enumerate(pred):
        current_row = [i + 1]
        for j, c2 in enumerate(true):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1] / len(true)

def categorize_error(pred: str, true: str, field_type: str) -> str:
    """
    Categorize error type based on field and difference.
    
    Args:
        pred: Predicted value
        true: True value
        field_type: Type of field ('work_order' or 'total_cost')
        
    Returns:
        Error category string
    """
    if not pred or not true:
        return "missing_field"
        
    if field_type == "work_order":
        # Check for transposition
        if sorted(pred) == sorted(true) and pred != true:
            return "transposition"
            
        # Check for length differences
        if len(pred) < len(true):
            return "missing_character"
        if len(pred) > len(true):
            return "extra_character"
            
        return "wrong_character"
        
    elif field_type == "total_cost":
        # Check for currency symbol error
        if ('$' in pred) != ('$' in true):
            return "currency_error"
            
        # Check for decimal point errors
        if ('.' in pred) != ('.' in true):
            return "decimal_error"
            
        # Check for formatting errors
        if any(c in pred for c in ',.') != any(c in true for c in ',.'):
            return "formatting_error"
            
        return "digit_error"
        
    return "unknown_error"

class PixtralModel(BaseModel):
    """Pixtral-12B model implementation with performance optimization."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config_manager: ConfigManager,
        validation_manager: ValidationManager,
        quantization: int = 32
    ):
        """Initialize Pixtral model with performance optimization.
        
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
        
        # Pre-process chat template format
        self.chat_template = self.processor.tokenizer.chat_template
        
    @property
    def name(self) -> str:
        """Get model name."""
        return "pixtral"
        
    def _load_model_weights(self, loading_params: Dict[str, Any]) -> Any:
        """Load model weights with specified parameters."""
        return LlavaForConditionalGeneration.from_pretrained(
            str(self.model_path),
            **loading_params
        )
        
    def _load_processor(self) -> Any:
        """Load model processor."""
        return AutoProcessor.from_pretrained(
            str(self.model_path),
            use_fast=True
        )
        
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
                    
            # Format chat-style input
            chat = [{
                "role": "user",
                "content": [
                    {"type": "text", "content": prompt},
                    *[{"type": "image", "image": img} for img in images]
                ]
            }]
            
            # Process inputs
            inputs = self.processor(
                images=images,
                text=prompt,
                return_tensors="pt",
                padding=True
            ).to(self.optimizer.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=False
                )
                
            # Decode outputs
            output_texts = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            
            # Parse outputs
            results = []
            for output_text in output_texts:
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

def download_pixtral_model(model_path: Path, repo_id: str) -> bool:
    """Download Pixtral model from HuggingFace.
    
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
            
        logger.info(f"Downloading Pixtral model from {repo_id}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_path),
            local_dir_use_symlinks=False
        )
        return True
        
    except Exception as e:
        logger.error(f"Failed to download Pixtral model: {str(e)}")
        return False

def load_model(model_name: str, quantization: int, models_dir: Path, config: dict) -> PixtralModel:
    """Load Pixtral model with specified quantization.
    
    Args:
        model_name: Name of the model (must be 'pixtral')
        quantization: Bit width for quantization (4, 8, 16, 32)
        models_dir: Path to models directory
        config: Model configuration from YAML
        
    Returns:
        Loaded PixtralModel instance
        
    Raises:
        ValueError: If model_name is not 'pixtral'
        FileNotFoundError: If model directory doesn't exist
    """
    if model_name != "pixtral":
        raise ValueError(f"Invalid model name for Pixtral loader: {model_name}")
        
    model_path = models_dir / "pixtral-12b"
    
    # Download model if needed
    if not model_path.exists():
        if not download_pixtral_model(model_path, config['repo_id']):
            raise RuntimeError(f"Failed to download Pixtral model to {model_path}")
        
    return PixtralModel(
        model_path=model_path,
        quantization=quantization
    )

def process_image_wrapper(
    model: PixtralModel,
    prompt_template: str,
    image_path: Union[str, Path],
    field_type: str,
    config: DataConfig
) -> Dict[str, Any]:
    """Wrapper function to process image with Pixtral model.
    
    Args:
        model: Loaded PixtralModel instance
        prompt_template: Prompt template to use
        image_path: Path to input image
        field_type: Type of field to extract
        config: Data configuration
        
    Returns:
        Dictionary with test parameters and model response
    """
    # Get model response
    response = model.process_image(
        image_paths=image_path,
        prompt=prompt_template,
        field_type=field_type,
        config=config
    )
    
    # Update test parameters with prompt strategy
    response['test_parameters']['prompt_strategy'] = prompt_template
    
    # Ensure field type is set in model response
    response['model_response']['field_type'] = field_type
    
    return response

def load_pixtral_model(model_name: str, config: Dict[str, Any], models_dir: Optional[Path] = None) -> Tuple[Any, Any]:
    """Load Pixtral model with proper configuration."""
    try:
        model_path = get_model_path(model_name, models_dir)
        hf_token = get_hf_token()
        
        # Get quantization level from config
        quant_level = config.get("quant_level", 16)
        if quant_level not in [4, 8, 16, 32]:
            raise ValueError(f"Invalid quantization level: {quant_level}. Must be one of [4, 8, 16, 32]")
            
        # Configure quantization
        quantization_config = {
            "load_in_4bit": quant_level == 4,
            "load_in_8bit": quant_level == 8,
            "torch_dtype": torch.float16 if quant_level == 16 else torch.float32
        }
        
        # Load model
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            token=hf_token,
            device_map="auto",
            **quantization_config
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_path,
            token=hf_token
        )
        
        return model, processor
    
    except Exception as e:
        logger.error(f"Error loading Pixtral model: {str(e)}")
        raise

def process_pixtral_image(model: Any, processor: Any, image: Union[Image.Image, Dict[str, Any]], prompt: str) -> Dict[str, Any]:
    """Process image through Pixtral model."""
    try:
        # Format input
        if isinstance(image, dict) and "images" in image:
            image_tokens = "[IMG]" * len(image["images"])
            formatted_prompt = f"<s>[INST]{prompt}\n{image_tokens}[/INST]"
        else:
            formatted_prompt = f"<s>[INST]{prompt}\n[IMG][/INST]"
            
        # Process input
        inputs = processor(
            image, 
            formatted_prompt, 
            text_kwargs={"add_special_tokens": False}, 
            return_tensors="pt"
        ).to(model.device)
        
        # Generate output
        outputs = model.generate(
            **inputs,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=2048
        )
        
        # Decode output
        result = processor.decode(outputs[0])[len(formatted_prompt):]
        
        return {
            "text": result,
            "model_type": "pixtral",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error processing image with Pixtral: {str(e)}")
        raise 