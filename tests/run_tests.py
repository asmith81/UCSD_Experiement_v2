"""
Test runner script for LMM invoice data extraction tests.
Adapts to local and RunPod environments.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.environment import detect_environment

def main():
    """Run tests based on environment."""
    # Detect environment
    environment = detect_environment()
    print(f"Running tests in {environment} environment")
    
    # Set up test arguments
    args = [
        "-v",  # Verbose output
        "--tb=short",  # Shorter traceback format
    ]
    
    if environment == "local":
        # Local environment: run core tests and mock model tests
        args.extend([
            "tests/test_core_utils.py",
            "tests/test_models.py::test_mock_inference",
            "tests/test_models.py::test_prompt_strategies"
        ])
    else:
        # RunPod environment: run all tests
        args.extend([
            "tests/test_core_utils.py",
            "tests/test_models.py"
        ])
    
    # Add coverage reporting if coverage is installed
    try:
        import coverage
        args.extend([
            "--cov=src",
            "--cov-report=term-missing"
        ])
    except ImportError:
        print("Coverage not installed, skipping coverage reporting")
    
    # Run tests
    sys.exit(pytest.main(args))

if __name__ == "__main__":
    main() 