"""
Test script for evaluation and logging functionality.
"""

import sys
from pathlib import Path
import logging
import json
from typing import Dict, Any

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

from src.results_logging import (
    evaluate_model_output,
    log_result,
    load_result,
    FileSystemStorage
)
from src.data_utils import GroundTruthData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_evaluation_and_logging():
    """Test evaluation and logging functionality."""
    try:
        # Model output from actual run
        model_output = {
            'output': '''Please extract the following information from this invoice:
Work Order Number
Total Cost
Return the information in JSON format with these exact keys: { "work_order_number": "extracted value", "total_cost": "extracted value" }

{
  "work_order_number": "20502",
  "total_cost": "950.00"
}''',
            'field_type': 'both'
        }

        # Ground truth data
        ground_truth: GroundTruthData = {
            'work_order_number': '20502',
            'total_cost': '950.00'
        }

        # Set up test paths
        test_dir = Path(__file__).parent / 'test_results'
        test_dir.mkdir(parents=True, exist_ok=True)
        result_path = test_dir / 'test_result.json'

        # Log result
        print("\nLogging result...")
        log_result(
            result_path=result_path,
            image_id='test_image_001',
            model_output=model_output,
            ground_truth=ground_truth,
            processing_time=4.31,
            model_name='pixtral',
            prompt_type='basic_extraction',
            quant_level=32,
            environment='local_test'
        )

        # Load and verify result
        print("\nLoading result...")
        loaded_result = load_result(result_path)

        # Get evaluation from loaded result
        evaluation = loaded_result['results_by_image']['test_image_001']['evaluation']

        # Print evaluation results
        print("\nEvaluation Results from Log:")
        print(f"Work Order Number:")
        print(f"  Raw String Match: {evaluation['work_order_number']['raw_string_match']}")
        print(f"  Normalized Match: {evaluation['work_order_number']['normalized_match']}")
        print(f"  CER: {evaluation['work_order_number']['cer']}")
        print(f"  Error Category: {evaluation['work_order_number']['error_category']}")
        
        print(f"\nTotal Cost:")
        print(f"  Raw String Match: {evaluation['total_cost']['raw_string_match']}")
        print(f"  Normalized Match: {evaluation['total_cost']['normalized_match']}")
        print(f"  CER: {evaluation['total_cost']['cer']}")
        print(f"  Error Category: {evaluation['total_cost']['error_category']}")

        # Verify expected results
        assert evaluation['work_order_number']['raw_string_match'] == True, "Work order raw string should match"
        assert evaluation['work_order_number']['normalized_match'] == True, "Work order normalized should match"
        assert evaluation['work_order_number']['cer'] == 0.0, "Work order CER should be 0.0"
        assert evaluation['work_order_number']['error_category'] == 'no_error', "Work order should have no error"

        assert evaluation['total_cost']['raw_string_match'] == True, "Total cost raw string should match"
        assert evaluation['total_cost']['normalized_match'] == True, "Total cost normalized should match"
        assert evaluation['total_cost']['cer'] == 0.0, "Total cost CER should be 0.0"
        assert evaluation['total_cost']['error_category'] == 'no_error', "Total cost should have no error"

        print("\nTest completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == '__main__':
    test_evaluation_and_logging() 