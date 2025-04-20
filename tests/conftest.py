"""
Pytest configuration and fixtures for LMM invoice data extraction tests.
"""

import pytest
import os
from pathlib import Path
from typing import Dict, Any
import torch
from src.environment import detect_environment

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """Get the test data directory."""
    return project_root / "tests" / "data"

@pytest.fixture(scope="session")
def mock_data_dir(project_root: Path) -> Path:
    """Get the mock data directory."""
    return project_root / "tests" / "mock_data"

@pytest.fixture(scope="session")
def environment() -> str:
    """Get the current environment (local or runpod)."""
    return detect_environment()

@pytest.fixture(scope="session")
def cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()

@pytest.fixture(scope="session")
def test_config(project_root: Path, environment: str) -> Dict[str, Any]:
    """Get test configuration based on environment."""
    config = {
        "local": {
            "use_gpu": False,
            "model_size": "small",
            "test_images": 2,
            "quantization_levels": [32],  # Only test full precision locally
            "prompt_strategies": ["basic"]
        },
        "runpod": {
            "use_gpu": True,
            "model_size": "full",
            "test_images": 10,
            "quantization_levels": [4, 8, 16, 32],
            "prompt_strategies": ["basic", "detailed", "few-shot"]
        }
    }
    return config[environment]

@pytest.fixture(scope="session")
def mock_model():
    """Create a mock model for local testing."""
    class MockModel:
        def __init__(self):
            self.name = "mock_model"
            self.quantization = 32
            
        def __call__(self, prompt: str) -> Dict[str, Any]:
            return {
                "raw_value": "MOCK123",
                "normalized_value": "MOCK123",
                "confidence": 0.95,
                "processing_time": 0.1
            }
    return MockModel() 