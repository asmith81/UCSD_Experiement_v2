"""
Model Loading Protocol implementation.
Handles loading of different model types with their specific requirements.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union, Tuple

import torch
from huggingface_hub import HfFolder
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    LlavaForConditionalGeneration,
    MllamaForConditionalGeneration,
    MllamaProcessor,
)
from doctr.models import ocr_predictor

logger = logging.getLogger(__name__)

class ModelLoader(Protocol):
    """Protocol for model loading functions."""
    def __call__(self, model_name: str, config: Dict[str, Any], models_dir: Optional[Path] = None) -> Any:
        ...

def get_hf_token() -> str:
    """Retrieve Hugging Face token from cache or environment."""
    token = HfFolder.get_token()
    if not token:
        raise RuntimeError("Hugging Face token not found. Please login using `huggingface-cli login`.")
    return token

def load_model(model_name: str, config: Dict[str, Any], models_dir: Optional[Path] = None) -> Any:
    """Load model with proper configuration and initialization."""
    try:
        model_type = config.get("model_type", "default")
        logger.info(f"Loading {model_type} model: {model_name}")
        
        if model_type == "pixtral":
            return _load_pixtral(model_name, config, models_dir)
        elif model_type == "llama":
            return _load_llama(model_name, config, models_dir)
        elif model_type == "doctr":
            return _load_doctr(model_name, config, models_dir)
        else:
            return _load_default(model_name, config, models_dir)
    
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise

def _load_pixtral(model_name: str, config: Dict[str, Any], models_dir: Optional[Path] = None) -> Tuple[Any, Any]:
    """Load Pixtral model with proper configuration."""
    try:
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        
        model_path = get_model_path(model_name, models_dir)
        hf_token = get_hf_token()
        
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            token=hf_token,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        processor = AutoProcessor.from_pretrained(
            model_path,
            token=hf_token
        )
        
        return model, processor
    
    except Exception as e:
        logger.error(f"Error loading Pixtral model: {str(e)}")
        raise

def _load_llama(model_name: str, config: Dict[str, Any], models_dir: Optional[Path] = None) -> Tuple[Any, Any]:
    """Load Llama model with proper configuration."""
    try:
        from transformers import MllamaForConditionalGeneration, MllamaProcessor
        
        model_path = get_model_path(model_name, models_dir)
        hf_token = get_hf_token()
        
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            token=hf_token,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        processor = MllamaProcessor.from_pretrained(
            model_path,
            token=hf_token
        )
        
        return model, processor
    
    except Exception as e:
        logger.error(f"Error loading Llama model: {str(e)}")
        raise

def _load_doctr(model_name: str, config: Dict[str, Any], models_dir: Optional[Path] = None) -> Any:
    """Load Doctr model with proper configuration."""
    try:
        from doctr.models import ocr_predictor
        
        model = ocr_predictor(
            det_arch=config.get("detection_arch", "db_resnet50"),
            reco_arch=config.get("recognition_arch", "crnn_vgg16_bn"),
            pretrained=True,
            assume_straight_boxes=config.get("assume_straight_boxes", True),
            export_as_straight_boxes=config.get("export_as_straight_boxes", True)
        )
        
        return model
    
    except Exception as e:
        logger.error(f"Error loading Doctr model: {str(e)}")
        raise

def _load_default(model_name: str, config: Dict[str, Any], models_dir: Optional[Path] = None) -> Tuple[Any, Any]:
    """Load default model with proper configuration."""
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        
        model_path = get_model_path(model_name, models_dir)
        hf_token = get_hf_token()
        
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            token=hf_token,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        processor = AutoProcessor.from_pretrained(
            model_path,
            token=hf_token
        )
        
        return model, processor
    
    except Exception as e:
        logger.error(f"Error loading default model: {str(e)}")
        raise

def get_model_path(model_name: str, quant_level: int, models_dir: Optional[Path] = None) -> Path:
    """
    Get the path to model files based on model name and quantization level.
    
    Args:
        model_name: Name of the model
        quant_level: Quantization level (4, 8, 16, 32)
        models_dir: Optional directory containing model files
        
    Returns:
        Path to model files
    """
    if quant_level not in [4, 8, 16, 32]:
        raise ValueError(f"Invalid quantization level: {quant_level}. Must be one of [4, 8, 16, 32]")
    
    if models_dir is None:
        models_dir = Path.cwd()
    
    model_path = models_dir / f"{model_name}_q{quant_level}"
    if not model_path.exists():
        raise FileNotFoundError(f"Model files not found at {model_path}")
    
    return model_path 