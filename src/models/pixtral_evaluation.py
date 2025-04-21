# %% [markdown]
# Pixtral Model Evaluation Notebook
# 
# This notebook implements the Pixtral model for extracting structured data
# from invoice images. It includes model loading with quantization support,
# inference functions, and output parsing.

# %% [markdown]
# ## Setup and Dependencies
# 
# First, we'll set up the environment and install all required dependencies.

# %%
import os
import sys
import subprocess
from pathlib import Path
import logging
import json

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# %% [markdown]
# ## Root Directory Setup
# 
# Determine the project root directory and add it to the Python path.

# %%
# Determine root directory
try:
    # When running as a script
    ROOT_DIR = Path(__file__).parent.parent
except NameError:
    # When running in a notebook, look for project root markers
    current_dir = Path.cwd()
    while current_dir != current_dir.parent:
        if (current_dir / 'src').exists() and (current_dir / 'notebooks').exists():
            ROOT_DIR = current_dir
            break
        current_dir = current_dir.parent
    else:
        raise RuntimeError("Could not find project root directory. Make sure you're running from within the project structure.")

sys.path.append(str(ROOT_DIR))

# %% [markdown]
# ## Install Dependencies
# 
# Install all required packages including base requirements, PyTorch, and AI-specific dependencies.

# %%
# Install dependencies
print("Installing dependencies...")
try:
    # Install base requirements first
    print("Installing base requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(ROOT_DIR / "requirements.txt")])
    print("Base requirements installed successfully.")
    
    # Install PyTorch dependencies separately
    print("Installing PyTorch dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "torchaudio==2.1.0",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ])
    print("PyTorch dependencies installed successfully.")
    
    # Install AI-specific dependencies
    print("Installing AI-specific dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "transformers==4.50.3",
        "accelerate>=0.26.0",
        "bitsandbytes==0.45.5",
        "huggingface_hub>=0.20.3",
        "flash-attn==2.5.0"
    ])
    print("AI-specific dependencies installed successfully.")
except subprocess.CalledProcessError as e:
    logger.error(f"Error installing dependencies: {e}")
    raise

# %% [markdown]
# ## Import Project Modules
# 
# Import all necessary project modules after dependencies are installed.

# %%
# Import project modules
from src import execution
from src.environment import setup_environment, download_model
from src.config import load_yaml_config
from src.models.llama_vision import load_model, process_image_wrapper, download_llama_vision_model
from src.prompts import load_prompt_template
from src.results_logging import track_execution, log_result, ResultStructure, evaluate_model_output
from src.validation import validate_results
from src.data_utils import DataConfig, setup_data_paths

# %% [markdown]
# ## Environment Setup
# 
# Configure the project environment and validate required paths.

# %%
# Setup environment
try:
    env = setup_environment(
        project_root=ROOT_DIR,
        requirements_path=ROOT_DIR / "requirements.txt"
    )
    
    # Validate paths
    required_paths = ['data_dir', 'models_dir', 'logs_dir', 'results_dir', 'prompts_dir']
    missing_paths = [path for path in required_paths if path not in env]
    if missing_paths:
        raise RuntimeError(f"Missing required paths in environment: {missing_paths}")
        
    # Ensure required directories exist
    for path in required_paths:
        env[path].mkdir(parents=True, exist_ok=True)
    
except Exception as e:
    logger.error(f"Error setting up environment: {str(e)}")
    raise

# %% [markdown]
# ## Configuration Loading
# 
# Load and validate the model configuration.

# %%
# Load configuration
config_path = ROOT_DIR / "config" / "models" / "llama_vision.yaml"
if not config_path.exists():
    raise FileNotFoundError(f"Configuration file not found: {config_path}")

try:
    config = load_yaml_config(str(config_path))
    # Validate required configuration sections
    required_sections = ['name', 'loading', 'quantization', 'prompt', 'inference']
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Configuration missing required sections: {missing_sections}")
except Exception as e:
    logger.error(f"Error loading configuration: {str(e)}")
    raise

# %% [markdown]
# ## Data Configuration
# 
# Set up data paths and validate data configuration.

# %%
# Setup data configuration
try:
    data_config = setup_data_paths(
        env_config=env,
        image_extensions=['.jpg', '.jpeg', '.png'],
        max_image_size=1120,
        supported_formats=['RGB', 'L']
    )
    logger.info("Data configuration setup successfully")
except Exception as e:
    logger.error(f"Error setting up data configuration: {str(e)}")
    raise

# %% [markdown]
# ## Model Configuration
# 
# Load and validate model-specific configuration.

# %%
# Load model configuration
try:
    # The config is already loaded and validated with required sections
    # We can use the config directly as it matches our needs
    model_config = {
        'name': config['name'],
        'path': config['repo_id'],
        'quantization_levels': list(config['quantization']['options'].keys())
    }
    
    prompt_config = {
        'format': config['prompt']['format'],
        'image_placeholder': config['prompt']['image_placeholder'],
        'default_field': config['prompt']['default_field']
    }
    
    # Validate model configuration
    required_model_fields = ['name', 'path', 'quantization_levels']
    missing_fields = [field for field in required_model_fields if field not in model_config]
    if missing_fields:
        raise ValueError(f"Model configuration missing required fields: {missing_fields}")
        
except KeyError as e:
    logger.error(f"Missing required configuration section: {e}")
    raise
except Exception as e:
    logger.error(f"Error loading model configuration: {str(e)}")
    raise

print(f"✓ Model configuration loaded successfully for {MODEL_NAME}")

# %% [markdown]
# ## Test Matrix Setup
# 
# Configure and validate the test matrix for model evaluation.

# %%
# Set model for this notebook
MODEL_NAME = "llama_vision"
TEST_MATRIX_PATH = str(ROOT_DIR / "config" / "test_matrix.json")
EXECUTION_LOG_PATH = env['logs_dir'] / f"{MODEL_NAME}_execution.log"

# Validate test matrix exists and is valid
try:
    if not Path(TEST_MATRIX_PATH).exists():
        raise FileNotFoundError(f"Test matrix file not found: {TEST_MATRIX_PATH}")
        
    # Load and validate test matrix
    with open(TEST_MATRIX_PATH, 'r') as f:
        test_matrix = json.load(f)
        
    # Validate test matrix structure
    if 'test_cases' not in test_matrix:
        raise ValueError("Test matrix must contain 'test_cases' array")
        
    # Validate required fields
    required_fields = ['model_name', 'field_type', 'prompt_type', 'quant_level', 'image_path']
    for test_case in test_matrix['test_cases']:
        missing_fields = [field for field in required_fields if field not in test_case]
        if missing_fields:
            raise ValueError(f"Test case missing required fields: {missing_fields}")
            
    # Validate quantization values
    valid_quantization = [4, 8, 16, 32]
    invalid_quantization = [case['quant_level'] for case in test_matrix['test_cases'] 
                          if case['quant_level'] not in valid_quantization]
    if invalid_quantization:
        raise ValueError(f"Invalid quantization values found: {invalid_quantization}")
            
except Exception as e:
    logger.error(f"Error validating test matrix: {str(e)}")
    raise

# %% [markdown]
# ## Pixtral Model Implementation
# 
# The core Pixtral model implementation for invoice data extraction.

# %%
from typing import Dict, Any, Optional, Union
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import time

from .common import (
    preprocess_image,
    parse_model_output,
    validate_model_output
)
from ..data_utils import DataConfig
from ..results_logging import ModelResponse

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