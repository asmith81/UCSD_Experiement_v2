"""
Test module for validating test matrix data flow.

This module contains tests to ensure test matrix data is correctly
passed to the model and processed.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import pytest
from src.validation import validate_test_matrix
from src.results_logging import validate_ground_truth, normalize_total_cost

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_test_matrix_validation():
    """
    Test that test matrix data is correctly validated and processed.
    
    This test verifies:
    1. Test matrix structure is valid
    2. Required fields are present
    3. Quantization levels are valid
    4. Image paths exist
    5. Ground truth data is properly formatted
    """
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent
        
        # Load test matrix
        test_matrix_path = project_root / "config" / "test_matrix.json"
        if not test_matrix_path.exists():
            raise FileNotFoundError(f"Test matrix not found at {test_matrix_path}")
            
        with open(test_matrix_path, 'r') as f:
            test_matrix = json.load(f)
            
        # Define supported parameters
        supported_quant_levels = [4, 8, 16, 32]
        available_prompt_types = ['basic_extraction']
        data_dir = project_root  # Changed from project_root / "data"
        
        # Validate test matrix structure
        assert validate_test_matrix(
            test_matrix=test_matrix,
            supported_quant_levels=supported_quant_levels,
            available_prompt_types=available_prompt_types,
            data_dir=data_dir
        ), "Test matrix validation failed"
        
        # Test individual test cases
        for i, test_case in enumerate(test_matrix['test_cases']):
            # Check required fields
            required_fields = ['model_name', 'prompt_type', 'quant_level', 
                             'field_type', 'image_path']
            for field in required_fields:
                assert field in test_case, f"Missing required field {field} in test case {i}"
                
            # Validate quantization level
            assert test_case['quant_level'] in supported_quant_levels, \
                f"Invalid quantization level {test_case['quant_level']} in test case {i}"
                
            # Validate prompt type
            assert test_case['prompt_type'] in available_prompt_types, \
                f"Invalid prompt type {test_case['prompt_type']} in test case {i}"
                
            # Validate field type
            assert test_case['field_type'] in ['work_order_number', 'total_cost', 'both'], \
                f"Invalid field type {test_case['field_type']} in test case {i}"
                
            # Validate image path exists
            image_path = data_dir / test_case['image_path']
            assert image_path.exists(), f"Image path does not exist: {image_path}"
            
            # If ground truth is provided, validate it
            if 'ground_truth' in test_case:
                assert validate_ground_truth(test_case['ground_truth']), \
                    f"Invalid ground truth in test case {i}"
                
                # Validate total cost format if present
                if 'total_cost' in test_case['ground_truth']:
                    try:
                        normalize_total_cost(test_case['ground_truth']['total_cost'])
                    except ValueError as e:
                        pytest.fail(f"Invalid total cost format in test case {i}: {str(e)}")
                        
        logger.info("Test matrix validation passed successfully")
        
    except Exception as e:
        logger.error(f"Error in test matrix validation: {str(e)}")
        raise 