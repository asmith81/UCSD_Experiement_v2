"""
Test module for validating results logging functionality.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
import pytest
from src.results_logging import (
    ModelResponse,
    EvaluationResult,
    validate_ground_truth,
    normalize_total_cost,
    ResultStructure,
    ResultEntry
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_results_logging_flow():
    """
    Test the complete results logging flow with a sample model output.
    
    This test:
    1. Creates a sample model output
    2. Creates a dummy ground truth
    3. Validates the ground truth
    4. Normalizes the total cost
    5. Creates evaluation results
    """
    try:
        # Sample model output
        model_output = {
            "work_order_number": "20502",
            "total_cost": "$950.00"
        }
        
        # Create dummy ground truth
        ground_truth = {
            "work_order_number": "20502",
            "total_cost": "$950.00"
        }
        
        # Validate ground truth structure
        assert validate_ground_truth(ground_truth), "Ground truth validation failed"
        
        # Normalize total cost
        normalized_cost = normalize_total_cost(ground_truth["total_cost"])
        assert normalized_cost == 950.00, f"Expected normalized cost 950.00, got {normalized_cost}"
        
        # Create model response
        model_response: ModelResponse = {
            "work_order_number": {
                "raw_text": json.dumps(model_output),
                "parsed_value": model_output["work_order_number"],
                "normalized_value": model_output["work_order_number"]
            },
            "total_cost": {
                "raw_text": json.dumps(model_output),
                "parsed_value": model_output["total_cost"],
                "normalized_value": normalized_cost
            },
            "processing_time": 1.5  # Dummy processing time
        }
        
        # Create evaluation result
        evaluation_result: Dict[str, EvaluationResult] = {
            "work_order_number": {
                "normalized_match": True,
                "cer": 0.0,
                "error_category": "none"
            },
            "total_cost": {
                "normalized_match": True,
                "cer": 0.0,
                "error_category": "none"
            }
        }
        
        # Create result entry
        result_entry: ResultEntry = {
            "ground_truth": ground_truth,
            "model_response": model_response,
            "evaluation": evaluation_result
        }
        
        # Create final result structure
        result: ResultStructure = {
            "meta": {
                "experiment_id": "test_exp_001",
                "timestamp": "2024-03-21T12:00:00",
                "environment": "test_environment"
            },
            "test_parameters": {
                "model_name": "pixtral",
                "prompt_type": "basic_extraction",
                "quantization": 32,
                "field_type": "both"
            },
            "results_by_image": {
                "test_image_001": result_entry
            }
        }
        
        # Print the complete result as JSON
        print("\nTest Result JSON Output:")
        print(json.dumps(result, indent=2))
        
        # Validate the complete result structure
        assert result["test_parameters"]["model_name"] == "pixtral"
        assert result["test_parameters"]["field_type"] == "both"
        assert result["results_by_image"]["test_image_001"]["model_response"]["work_order_number"]["parsed_value"] == "20502"
        assert result["results_by_image"]["test_image_001"]["model_response"]["total_cost"]["parsed_value"] == "$950.00"
        assert result["results_by_image"]["test_image_001"]["ground_truth"]["work_order_number"] == "20502"
        assert result["results_by_image"]["test_image_001"]["ground_truth"]["total_cost"] == "$950.00"
        assert result["results_by_image"]["test_image_001"]["evaluation"]["work_order_number"]["normalized_match"] is True
        assert result["results_by_image"]["test_image_001"]["evaluation"]["total_cost"]["normalized_match"] is True
        assert result["results_by_image"]["test_image_001"]["evaluation"]["work_order_number"]["cer"] == 0.0
        assert result["results_by_image"]["test_image_001"]["evaluation"]["total_cost"]["cer"] == 0.0
        
        logger.info("Results logging flow test passed successfully")
        
    except Exception as e:
        logger.error(f"Error in results logging flow test: {str(e)}")
        raise 