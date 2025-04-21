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

def get_model_path(model_name: str, models_dir: Optional[Path] = None) -> Path:
    """
    Get the path to model files based on model name.
    
    Args:
        model_name: Name of the model
        models_dir: Optional directory containing model files
        
    Returns:
        Path to model files
    """
    if models_dir is None:
        models_dir = Path.cwd()
    
    model_path = models_dir / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model files not found at {model_path}")
    
    return model_path

def load_model(model_name: str, config: Dict[str, Any], models_dir: Optional[Path] = None) -> Tuple[Any, Any]:
    """Load model with proper configuration and initialization."""
    try:
        model_type = config.get("model_type", "default")
        logger.info(f"Loading {model_type} model: {model_name}")
        
        if model_type == "pixtral":
            from .pixtral import load_pixtral_model
            return load_pixtral_model(model_name, config, models_dir)
        elif model_type == "llama":
            from .llama import load_llama_model
            return load_llama_model(model_name, config, models_dir)
        elif model_type == "doctr":
            from .doctr import load_doctr_model
            return load_doctr_model(model_name, config, models_dir)
        else:
            return _load_default(model_name, config, models_dir)
    
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
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