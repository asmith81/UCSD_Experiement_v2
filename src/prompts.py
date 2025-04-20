"""
Prompt template management module.

This module provides functionality for loading and managing prompt templates
from YAML configuration files.
"""

import yaml
from pathlib import Path
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_prompt_template(prompt_strategy: str, prompts_dir: Optional[Path] = None) -> str:
    """
    Load a prompt template for a specific strategy.
    
    Args:
        prompt_strategy: Name of the prompt strategy (e.g., 'basic_extraction', 'detailed')
        prompts_dir: Optional path to prompts directory. If not provided, defaults to "config/prompts"
        
    Returns:
        Loaded prompt template string
        
    Raises:
        FileNotFoundError: If prompt template doesn't exist
        ValueError: If prompt template is invalid
    """
    try:
        # Construct path to prompt template
        if prompts_dir is None:
            prompts_dir = Path("config/prompts")
        prompt_path = prompts_dir / f"{prompt_strategy}.yaml"
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
            
        # Load and parse YAML
        with open(prompt_path, 'r') as f:
            template_data = yaml.safe_load(f)
            
        # Validate template structure
        if 'prompts' not in template_data:
            raise ValueError(f"Invalid prompt template: missing 'prompts' array in {prompt_path}")
            
        # Find the matching prompt
        for prompt in template_data['prompts']:
            if prompt['name'] == prompt_strategy:
                if 'text' not in prompt:
                    raise ValueError(f"Invalid prompt template: missing 'text' field in prompt {prompt_strategy}")
                return prompt['text']
                
        raise ValueError(f"Prompt strategy {prompt_strategy} not found in {prompt_path}")
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing prompt template {prompt_strategy}: {str(e)}")
        raise ValueError(f"Invalid YAML in prompt template {prompt_strategy}")
    except Exception as e:
        logger.error(f"Error loading prompt template {prompt_strategy}: {str(e)}")
        raise 