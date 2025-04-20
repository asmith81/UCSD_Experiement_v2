"""
Llama-3.2-11B-Vision model implementation for invoice data extraction.

This module implements the Llama Vision model for extracting structured data
from invoice images. It includes model loading with quantization support,
inference functions, and output parsing.
"""

from typing import Dict, Any, Optional, Union
import torch
from transformers import MllamaForConditionalGeneration, MllamaProcessor, BitsAndBytesConfig
from PIL import Image
import logging
from pathlib import Path
import time

from .common import (
    preprocess_image,
    parse_model_output,
    validate_model_output
)
from ..data_utils import DataConfig
from ..results_logging import ModelResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LlamaVisionModel:
    """Llama-3.2-11B-Vision model implementation."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        quantization: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize Llama Vision model.
        
        Args:
            model_path: Path to model weights
            quantization: Bit width for quantization (4, 8, 16, 32)
            device: Device to run model on
            
        Raises:
            FileNotFoundError: If model path doesn't exist
            ValueError: If quantization level is invalid
        """
        # Validate model path
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
        # Validate quantization
        if quantization not in [4, 8, 16, 32]:
            raise ValueError(f"Invalid quantization level: {quantization}. Must be one of [4, 8, 16, 32]")
            
        self.quantization = quantization
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()
        
    def _load_model(self) -> None:
        """Load model with specified quantization.
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info(f"Loading Llama Vision model with {self.quantization}-bit quantization")
            
            # Validate model directory structure
            required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
            missing_files = [f for f in required_files if not (self.model_path / f).exists()]
            if missing_files:
                raise FileNotFoundError(f"Missing required model files: {missing_files}")
            
            # Load processor with fast processing enabled
            self.processor = MllamaProcessor.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                use_fast=True  # Enable fast processing
            )
            
            # Set default dtype based on quantization
            if self.quantization in [4, 8, 16]:
                default_dtype = torch.float16
            else:
                default_dtype = torch.float32
            
            # Load model with quantization
            if self.quantization == 32:
                self.model = MllamaForConditionalGeneration.from_pretrained(
                    str(self.model_path),
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    torch_dtype=default_dtype
                )
            elif self.quantization == 16:
                self.model = MllamaForConditionalGeneration.from_pretrained(
                    str(self.model_path),
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    torch_dtype=default_dtype
                )
            elif self.quantization == 8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=default_dtype,
                    bnb_8bit_use_double_quant=True
                )
                self.model = MllamaForConditionalGeneration.from_pretrained(
                    str(self.model_path),
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    quantization_config=quantization_config
                )
            elif self.quantization == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=default_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = MllamaForConditionalGeneration.from_pretrained(
                    str(self.model_path),
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    quantization_config=quantization_config
                )
            else:
                raise ValueError(f"Unsupported quantization level: {self.quantization}")
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Llama Vision model: {str(e)}")
            raise RuntimeError(f"Failed to load Llama Vision model: {str(e)}")
            
    def process_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        field_type: str,
        config: DataConfig
    ) -> Dict[str, Any]:
        """Process image and extract field value.
        
        Args:
            image_path: Path to input image
            prompt: Prompt template for extraction
            field_type: Type of field to extract
            config: Data configuration
            
        Returns:
            Dictionary with test parameters and model response
            
        Raises:
            ValueError: If processing fails
        """
        try:
            start_time = time.time()
            
            # Preprocess image
            image = preprocess_image(Path(image_path), config)
            
            # Format chat-style input
            chat = [{
                "role": "user",
                "content": [
                    {"type": "text", "content": prompt},
                    {"type": "image"}
                ]
            }]
            
            # Apply chat template and process inputs
            inputs = self.processor.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
                
            # Decode output
            output_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Parse output
            parsed_value = parse_model_output(output_text, field_type)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Structure result according to validation requirements
            result = {
                'test_parameters': {
                    'model': 'llama_vision',
                    'quantization': self.quantization
                },
                'model_response': {
                    'output': output_text,
                    'processing_time': processing_time
                },
                'evaluation': {
                    'work_order_number': {
                        'raw_string_match': False,
                        'normalized_match': False,
                        'cer': 0.0,
                        'error_category': 'not_evaluated'
                    },
                    'total_cost': {
                        'raw_string_match': False,
                        'normalized_match': False,
                        'cer': 0.0,
                        'error_category': 'not_evaluated'
                    }
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

def download_llama_vision_model(model_path: Path, repo_id: str) -> bool:
    """Download Llama Vision model from HuggingFace.
    
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
            
        logger.info(f"Downloading Llama Vision model from {repo_id}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_path),
            local_dir_use_symlinks=False
        )
        return True
        
    except Exception as e:
        logger.error(f"Failed to download Llama Vision model: {str(e)}")
        return False

def load_model(model_name: str, quantization: int, models_dir: Path, config: dict) -> LlamaVisionModel:
    """Load Llama Vision model with specified quantization.
    
    Args:
        model_name: Name of the model (must be 'llama_vision')
        quantization: Bit width for quantization (4, 8, 16, 32)
        models_dir: Path to models directory
        config: Model configuration from YAML
        
    Returns:
        Loaded LlamaVisionModel instance
        
    Raises:
        ValueError: If model_name is not 'llama_vision'
        FileNotFoundError: If model directory doesn't exist
    """
    if model_name != "llama_vision":
        raise ValueError(f"Invalid model name for Llama Vision loader: {model_name}")
        
    model_path = models_dir / "llama-vision-11b"
    
    # Download model if needed
    if not model_path.exists():
        if not download_llama_vision_model(model_path, config['repo_id']):
            raise RuntimeError(f"Failed to download Llama Vision model to {model_path}")
        
    return LlamaVisionModel(
        model_path=model_path,
        quantization=quantization
    )

def process_image_wrapper(
    model: LlamaVisionModel,
    prompt_template: str,
    image_path: Union[str, Path],
    field_type: str,
    config: DataConfig
) -> Dict[str, Any]:
    """Wrapper function to process image with Llama Vision model.
    
    Args:
        model: Loaded LlamaVisionModel instance
        prompt_template: Prompt template to use
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