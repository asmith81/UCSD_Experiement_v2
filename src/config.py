"""
Configuration management for LMM invoice data extraction comparison.
Handles YAML config loading, test matrix parsing, and logging setup.
"""

import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, TypedDict
import json

class ModelConfig(TypedDict):
    name: str
    quantization_levels: List[str]
    architecture: Dict[str, Any]
    framework: Dict[str, Any]
    hardware: Dict[str, Any]

class LoggingConfig(TypedDict):
    log_dir: Path
    result_file: str
    execution_log: str
    log_level: str

class PromptConfig(TypedDict):
    name: str
    text: str
    category: str
    field_to_extract: List[str]
    description: str
    version: str
    format_instructions: Dict[str, Any]
    metadata: Dict[str, Any]

class PromptConfigInfo(TypedDict):
    name: str
    description: str
    version: str
    last_updated: str

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML config: {str(e)}")

def parse_test_matrix(json_path: str) -> List[Dict[str, Any]]:
    """
    Parse test matrix JSON into list of test cases.
    
    Args:
        json_path: Path to test matrix JSON file
        
    Returns:
        List of dictionaries containing test case parameters
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON is invalid or missing required fields
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        if 'test_cases' not in data:
            raise ValueError("Test matrix JSON must contain 'test_cases' array")
            
        # Validate each test case
        required_fields = ['model_name', 'field_type', 'prompt_type', 'quant_level']
        for case in data['test_cases']:
            missing_fields = [field for field in required_fields if field not in case]
            if missing_fields:
                raise ValueError(f"Test case missing required fields: {missing_fields}")
                
            # Validate quantization level
            if case['quant_level'] not in [4, 8, 16, 32]:
                raise ValueError(f"Invalid quantization level: {case['quant_level']}")
                
        return data['test_cases']
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Test matrix file not found: {json_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in test matrix file: {json_path}")

def load_model_config(config_path: str, model_name: str) -> ModelConfig:
    """
    Load model-specific configuration.
    
    Args:
        config_path: Path to YAML configuration file
        model_name: Name of the model to load config for
        
    Returns:
        ModelConfig: Model-specific configuration
        
    Raises:
        ValueError: If model config is not found or invalid
    """
    config = load_yaml_config(config_path)
    if model_name not in config['models']:
        raise ValueError(f"Configuration for model {model_name} not found")
    return config['models'][model_name]

def load_prompt_config(config_path: str, prompt_name: str) -> PromptConfig:
    """
    Load prompt-specific configuration.
    
    Args:
        config_path: Path to YAML configuration file
        prompt_name: Name of the prompt to load config for
        
    Returns:
        PromptConfig: Prompt-specific configuration
        
    Raises:
        ValueError: If prompt config is not found or invalid
    """
    config = load_yaml_config(config_path)
    if 'prompts' not in config:
        raise ValueError("No prompts section found in configuration")
    
    for prompt in config['prompts']:
        if prompt['name'] == prompt_name:
            return prompt
    
    raise ValueError(f"Configuration for prompt {prompt_name} not found")

def setup_logging_config(config: Dict[str, Any], log_dir: Path) -> LoggingConfig:
    """
    Setup logging configuration from main config.
    
    Args:
        config: Main configuration dictionary
        log_dir: Path to logging directory
        
    Returns:
        LoggingConfig: Logging configuration
        
    Raises:
        KeyError: If required logging config keys are missing
    """
    try:
        log_config: LoggingConfig = {
            'log_dir': log_dir,
            'result_file': config['logging']['result_file'],
            'execution_log': config['logging']['execution_log'],
            'log_level': config['logging'].get('log_level', 'INFO')
        }
        
        # Ensure log directory exists
        log_config['log_dir'].mkdir(parents=True, exist_ok=True)
        
        return log_config
    except KeyError as e:
        raise KeyError(f"Missing required logging configuration: {str(e)}")

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure and required fields.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ['models', 'data', 'logging']
    required_model_fields = ['name', 'quantization_levels']
    required_data_fields = ['image_dir', 'ground_truth_csv']
    required_logging_fields = ['result_file', 'execution_log']
    
    # Check required sections
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Validate model configuration
    for model in config['models']:
        for field in required_model_fields:
            if field not in model:
                raise ValueError(f"Model missing required field: {field}")
    
    # Validate data configuration
    for field in required_data_fields:
        if field not in config['data']:
            raise ValueError(f"Data section missing required field: {field}")
    
    # Validate logging configuration
    for field in required_logging_fields:
        if field not in config['logging']:
            raise ValueError(f"Logging section missing required field: {field}")
    
    return True

def validate_prompt_config(config: Dict[str, Any]) -> bool:
    """
    Validate prompt configuration structure.
    
    Args:
        config: Prompt configuration dictionary to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If prompt configuration is invalid
    """
    required_config_info_fields = ['name', 'description', 'version', 'last_updated']
    required_prompt_fields = [
        'name', 'text', 'category', 'field_to_extract',
        'description', 'version', 'format_instructions', 'metadata'
    ]
    
    # Validate config_info section
    if 'config_info' not in config:
        raise ValueError("Missing config_info section in prompt configuration")
    
    for field in required_config_info_fields:
        if field not in config['config_info']:
            raise ValueError(f"config_info missing required field: {field}")
    
    # Validate prompts section
    if 'prompts' not in config:
        raise ValueError("Missing prompts section in prompt configuration")
    
    for prompt in config['prompts']:
        for field in required_prompt_fields:
            if field not in prompt:
                raise ValueError(f"Prompt missing required field: {field}")
        
        # Validate format_instructions
        if 'output_format' not in prompt['format_instructions']:
            raise ValueError("Prompt format_instructions missing output_format")
        if 'required_fields' not in prompt['format_instructions']:
            raise ValueError("Prompt format_instructions missing required_fields")
    
    return True 