"""
Test script for output processing and logging systems.
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
    FileSystemStorage,
    GroundTruthData
)
from src.data_utils import DataConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_output_processing():
    """Test output processing and logging systems."""
    try:
        # Test data
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
        
        ground_truth: GroundTruthData = {
            'work_order_number': '20502',
            'total_cost': '950.00'
        }
        
        # Test evaluation
        print("\nTesting evaluate_model_output...")
        evaluation = evaluate_model_output(
            model_output['output'],
            ground_truth,
            model_output['field_type']
        )
        
        print("\nEvaluation Results:")
        print(json.dumps(evaluation, indent=2))
        
        # Test logging
        print("\nTesting log_result...")
        result_path = Path(__file__).parent / 'test_results' / 'output_processing_test.json'
        result_path.parent.mkdir(parents=True, exist_ok=True)
        
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
        
        print(f"\nResults logged to: {result_path}")
        
        # Verify logged results
        storage = FileSystemStorage()
        logged_result = storage.load_result(result_path)
        
        print("\nLogged Results:")
        print(json.dumps(logged_result, indent=2))
        
        # Verify evaluation matches
        logged_evaluation = logged_result['results_by_image']['test_image_001']['evaluation']
        assert logged_evaluation == evaluation, "Logged evaluation does not match computed evaluation"
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == '__main__':
    test_output_processing() 