"""
Unified Configuration Protocol for LMM Invoice Data Extraction Comparison.
This protocol standardizes configuration handling across all components.
"""

from typing import Dict, Any, List, Optional, Union, TypedDict
from pathlib import Path
import yaml
from dataclasses import dataclass
from enum import Enum

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass

class ConfigSection(Enum):
    """Enumeration of configuration sections."""
    MODELS = "models"
    PROMPTS = "prompts"
    TEST = "test"
    LOGGING = "logging"
    DATA = "data"
    PERFORMANCE = "performance"
    ERROR_HANDLING = "error_handling"

@dataclass
class BaseConfig:
    """Base configuration class with common validation methods."""
    
    def validate_required_fields(self, required_fields: List[str], section: str) -> None:
        """Validate that all required fields are present."""
        for field in required_fields:
            if not hasattr(self, field):
                raise ConfigValidationError(f"Missing required field '{field}' in {section}")

    def validate_field_types(self, field_types: Dict[str, type], section: str) -> None:
        """Validate that fields have correct types."""
        for field, expected_type in field_types.items():
            if hasattr(self, field):
                value = getattr(self, field)
                if not isinstance(value, expected_type):
                    raise ConfigValidationError(
                        f"Field '{field}' in {section} must be of type {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )

@dataclass
class ModelConfig(BaseConfig):
    """Model-specific configuration."""
    name: str
    quantization_levels: List[int]
    max_tokens: int
    temperature: float
    config_file: str
    
    def validate(self) -> None:
        """Validate model configuration."""
        self.validate_required_fields(
            ["name", "quantization_levels", "max_tokens", "temperature", "config_file"],
            ConfigSection.MODELS.value
        )
        self.validate_field_types(
            {
                "name": str,
                "quantization_levels": list,
                "max_tokens": int,
                "temperature": float,
                "config_file": str
            },
            ConfigSection.MODELS.value
        )
        
        # Validate quantization levels
        if not all(isinstance(q, int) for q in self.quantization_levels):
            raise ConfigValidationError("All quantization levels must be integers")
        if not all(q in [4, 8, 16, 32] for q in self.quantization_levels):
            raise ConfigValidationError("Quantization levels must be one of [4, 8, 16, 32]")

@dataclass
class PromptConfig(BaseConfig):
    """Prompt-specific configuration."""
    strategies: List[str]
    default_format: str
    input_format: Dict[str, Any]
    
    def validate(self) -> None:
        """Validate prompt configuration."""
        self.validate_required_fields(
            ["strategies", "default_format", "input_format"],
            ConfigSection.PROMPTS.value
        )
        self.validate_field_types(
            {
                "strategies": list,
                "default_format": str,
                "input_format": dict
            },
            ConfigSection.PROMPTS.value
        )

@dataclass
class TestConfig(BaseConfig):
    """Test-specific configuration."""
    batch_size: int
    max_retries: int
    timeout_seconds: int
    performance_thresholds: Dict[str, Any]
    
    def validate(self) -> None:
        """Validate test configuration."""
        self.validate_required_fields(
            ["batch_size", "max_retries", "timeout_seconds", "performance_thresholds"],
            ConfigSection.TEST.value
        )
        self.validate_field_types(
            {
                "batch_size": int,
                "max_retries": int,
                "timeout_seconds": int,
                "performance_thresholds": dict
            },
            ConfigSection.TEST.value
        )

@dataclass
class LoggingConfig(BaseConfig):
    """Logging-specific configuration."""
    level: str
    format: str
    results_dir: str
    performance_metrics: List[str]
    
    def validate(self) -> None:
        """Validate logging configuration."""
        self.validate_required_fields(
            ["level", "format", "results_dir", "performance_metrics"],
            ConfigSection.LOGGING.value
        )
        self.validate_field_types(
            {
                "level": str,
                "format": str,
                "results_dir": str,
                "performance_metrics": list
            },
            ConfigSection.LOGGING.value
        )
        
        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level not in valid_levels:
            raise ConfigValidationError(f"Log level must be one of {valid_levels}")

class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Union[str, Path]):
        """Initialize with path to configuration file."""
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        
    def load_config(self) -> None:
        """Load and validate configuration."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigValidationError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Error parsing YAML config: {str(e)}")
            
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate the entire configuration."""
        # Validate required sections
        required_sections = [section.value for section in ConfigSection]
        for section in required_sections:
            if section not in self.config:
                raise ConfigValidationError(f"Missing required section: {section}")
                
        # Validate each section
        self._validate_models()
        self._validate_prompts()
        self._validate_test()
        self._validate_logging()
        
    def _validate_models(self) -> None:
        """Validate models section."""
        for model_name, model_config in self.config[ConfigSection.MODELS.value].items():
            config = ModelConfig(**model_config)
            config.validate()
            
    def _validate_prompts(self) -> None:
        """Validate prompts section."""
        config = PromptConfig(**self.config[ConfigSection.PROMPTS.value])
        config.validate()
        
    def _validate_test(self) -> None:
        """Validate test section."""
        config = TestConfig(**self.config[ConfigSection.TEST.value])
        config.validate()
        
    def _validate_logging(self) -> None:
        """Validate logging section."""
        config = LoggingConfig(**self.config[ConfigSection.LOGGING.value])
        config.validate()
        
    def get_config(self, section: ConfigSection) -> Dict[str, Any]:
        """Get configuration for a specific section."""
        return self.config.get(section.value, {})
        
    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model."""
        if model_name not in self.config[ConfigSection.MODELS.value]:
            raise ConfigValidationError(f"Model configuration not found: {model_name}")
        return ModelConfig(**self.config[ConfigSection.MODELS.value][model_name]) 