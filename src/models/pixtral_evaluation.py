"""
Pixtral model implementation for invoice data extraction.

This module implements the Pixtral model for extracting structured data
from invoice images. It includes model loading with quantization support,
inference functions, and output parsing.
"""

from typing import Dict, Any, Optional, Union
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
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

class PixtralModel:
    """Pixtral model implementation."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        quantization: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize Pixtral model.
        
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
            logger.info(f"Loading Pixtral model with {self.quantization}-bit quantization")
            
            # Validate model directory structure
            required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
            missing_files = [f for f in required_files if not (self.model_path / f).exists()]
            if missing_files:
                raise FileNotFoundError(f"Missing required model files: {missing_files}")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
            
            # Set default dtype based on quantization
            if self.quantization in [4, 8, 16]:
                default_dtype = torch.float16
            else:
                default_dtype = torch.float32
            
            # Load model with quantization
            if self.quantization == 32:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    torch_dtype=default_dtype
                )
            elif self.quantization == 16:
                self.model = AutoModelForCausalLM.from_pretrained(
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
                self.model = AutoModelForCausalLM.from_pretrained(
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
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    quantization_config=quantization_config
                )
            else:
                raise ValueError(f"Unsupported quantization level: {self.quantization}")
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Pixtral model: {str(e)}")
            raise RuntimeError(f"Failed to load Pixtral model: {str(e)}") 