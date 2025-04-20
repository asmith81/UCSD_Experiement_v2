"""
Test script to verify logging functionality for Pixtral evaluation results.
This script creates a small test case and verifies the logging works correctly.
"""

import os
import sys
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Determine root directory
try:
    ROOT_DIR = Path(__file__).parent.parent
except NameError:
    current_dir = Path.cwd()
    while current_dir != current_dir.parent:
        if (current_dir / 'src').exists() and (current_dir / 'notebooks').exists():
            ROOT_DIR = current_dir
            break
        current_dir = current_dir.parent
    else:
        raise RuntimeError("Could not find project root directory")

sys.path.append(str(ROOT_DIR))

# Import required modules
from src.results_logging import log_quantization_results

def create_test_result():
    """Create a test result that matches the expected structure."""
    return {
        'test_parameters': {
            'model': 'pixtral',
            'quantization_level': 4,
            'prompt_type': 'basic_extraction',
            'image_path': 'images/1017.jpg'
        },
        'ground_truth': {
            'work_order_number': 'WO12345',
            'total_cost': '123.45'
        },
        'model_response': {
            'output': 'Work Order: WO12345\nTotal: $123.45',
            'processing_time': 1.23
        },
        'evaluation': {
            'work_order_number': {
                'normalized_match': True,
                'cer': 0.0
            },
            'total_cost': {
                'normalized_match': True,
                'cer': 0.0
            }
        }
    }

def test_logging():
    """Test the logging functionality with a single test case."""
    try:
        # Create test results
        test_results = [create_test_result()]
        
        # Set up test output path
        test_output_path = ROOT_DIR / 'logs' / 'test_logging_results.json'
        test_output_path.parent.mkdir(exist_ok=True)
        
        # Test logging
        log_quantization_results(
            result_path=str(test_output_path),
            results=test_results,
            model_name='pixtral',
            quant_level=4
        )
        
        # Verify the file was created and has the correct structure
        if test_output_path.exists():
            with open(test_output_path, 'r') as f:
                logged_data = json.load(f)
                
            # Verify basic structure
            assert 'results_by_prompt' in logged_data
            assert 'basic_extraction' in logged_data['results_by_prompt']
            assert 'results_by_image' in logged_data['results_by_prompt']['basic_extraction']
            
            logger.info("✓ Logging test passed successfully")
            logger.info(f"Test results written to: {test_output_path}")
            
            # Clean up test file
            test_output_path.unlink()
            logger.info("✓ Test file cleaned up")
            
        else:
            raise AssertionError("Test output file was not created")
            
    except Exception as e:
        logger.error(f"Logging test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_logging() 