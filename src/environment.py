"""
Environment setup module for consistent execution environment configuration.
Follows the interface control document specifications.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, TypedDict

class EnvironmentConfig(TypedDict):
    """Configuration for environment setup."""
    data_dir: Path
    models_dir: Path
    logs_dir: Path
    results_dir: Path
    prompts_dir: Path
    config_dir: Path
    cuda_visible_devices: Optional[str]
    seed: int

def setup_environment(
    cuda_visible_devices: Optional[str] = None,
    seed: int = 42,
    project_root: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Set up environment for model execution.
    
    Args:
        cuda_visible_devices: GPU devices to use (e.g., "0,1")
        seed: Random seed for reproducibility
        project_root: Optional project root directory
        
    Returns:
        dict: Environment configuration with paths and settings
        
    Raises:
        RuntimeError: If environment setup fails
    """
    try:
        # Set CUDA devices if specified
        if cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            
        # Set random seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        
        # Determine project root
        if project_root is None:
            project_root = _find_project_root()
            
        # Setup paths
        paths = _setup_paths(project_root)
        
        # Setup logging
        logging_config = _setup_logging(paths['logs_dir'])
        
        # Combine all configuration
        env_config = {
            'project_root': project_root,
            'paths': paths,
            'logging': logging_config,
            'cuda_visible_devices': cuda_visible_devices,
            'seed': seed
        }
        
        return env_config
        
    except Exception as e:
        raise RuntimeError(f"Environment setup failed: {str(e)}")

def _find_project_root() -> Path:
    """Find project root directory."""
    current_dir = Path.cwd()
    while current_dir != current_dir.parent:
        if (current_dir / 'src').exists() and (current_dir / 'notebooks').exists():
            return current_dir
        current_dir = current_dir.parent
    raise RuntimeError("Could not find project root directory")

def _setup_paths(project_root: Path) -> Dict[str, Path]:
    """Setup and validate project paths."""
    paths = {
        'data_dir': project_root / 'data',
        'models_dir': project_root / 'models',
        'logs_dir': project_root / 'logs',
        'results_dir': project_root / 'results',
        'config_dir': project_root / 'config'
    }
    
    # Create directories if they don't exist
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
        
    return paths

def _setup_logging(logs_dir: Path) -> Dict[str, Any]:
    """Setup logging configuration."""
    log_file = logs_dir / 'execution.log'
    
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'file': {
                'class': 'logging.FileHandler',
                'filename': str(log_file),
                'formatter': 'standard'
            },
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard'
            }
        },
        'root': {
            'handlers': ['file', 'console'],
            'level': 'INFO'
        }
    }
    
    return logging_config 