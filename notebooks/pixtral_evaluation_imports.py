"""
Pixtral Model Evaluation - Import and Environment Setup
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# Import project modules
from src.environment import setup_environment, EnvironmentConfig
from src.config import load_yaml_config
from src.models.loader import load_model
from src.models.pixtral import process_image_wrapper, download_pixtral_model
from src.prompts import load_prompt_template
from src.results_logging import (
    track_execution,
    log_result,
    ResultStructure,
    evaluate_model_output,
    ModelResponse
)
from src.validation import validate_results, ValidationManager
from src.data_utils import DataConfig, setup_data_paths
from src.config_protocol import ConfigManager, ConfigSection
from src.performance_optimizer import PerformanceOptimizer

# Setup environment
env = setup_environment(
    cuda_visible_devices=None,  # Set to specific GPU IDs if needed
    seed=42,
    project_root=ROOT_DIR
)

# Validate environment setup
required_paths = ['data_dir', 'models_dir', 'logs_dir', 'results_dir', 'prompts_dir', 'config_dir']
missing_paths = [path for path in required_paths if path not in env]
if missing_paths:
    raise RuntimeError(f"Missing required paths in environment: {missing_paths}")

# Ensure required directories exist
for path in required_paths:
    env[path].mkdir(parents=True, exist_ok=True)
    logger.info(f"Created/verified directory: {env[path]}")

# Create subdirectories for results
(env['results_dir'] / 'raw_results').mkdir(parents=True, exist_ok=True)
(env['results_dir'] / 'processed_results').mkdir(parents=True, exist_ok=True)

logger.info("Environment setup completed successfully") 