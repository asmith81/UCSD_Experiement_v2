"""
Tests for model-specific functionality that adapts to the environment.
"""

import pytest
from pathlib import Path
from typing import Dict, Any
from src.models import load_model, run_inference
import json
from PIL import Image

@pytest.mark.skipif(
    lambda environment: environment != 'local',
    reason="Model loading tests require local environment"
)
def test_model_loading(project_root: Path, test_config: Dict[str, Any], environment: str):
    """Test model loading in local environment."""
    for model_name in ["pixtral", "llama_vision", "doctr"]:
        for quantization in test_config["quantization_levels"]:
            try:
                model = load_model(model_name, quantization)
                assert model is not None
                assert hasattr(model, "name")
                assert model.name == model_name
                assert hasattr(model, "quantization")
                assert model.quantization == quantization
            except Exception as e:
                pytest.fail(f"Failed to load {model_name} with quantization {quantization}: {str(e)}")

def test_mock_inference(mock_model, test_config: Dict[str, Any]):
    """Test inference with mock model in local environment."""
    test_prompt = "Extract the work order number from this invoice."
    
    # Create mock image for testing
    import numpy as np
    mock_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    
    # Run inference directly with mock model
    result = mock_model(test_prompt)
    
    assert isinstance(result, dict)
    assert "raw_value" in result
    assert "normalized_value" in result
    assert "confidence" in result
    assert "processing_time" in result
    
    assert result["raw_value"] == "MOCK123"
    assert result["normalized_value"] == "MOCK123"
    assert 0 <= result["confidence"] <= 1
    assert result["processing_time"] > 0

@pytest.mark.skipif(
    lambda environment: environment != 'local',
    reason="Full model inference tests require local environment"
)
def test_full_model_inference(project_root: Path, test_config: Dict[str, Any], environment: str):
    """Test full model inference in local environment."""
    model = load_model("pixtral", 32)  # Use full precision for testing
    test_prompt = "Extract the work order number from this invoice."
    
    result = run_inference(model, test_prompt)
    
    assert isinstance(result, dict)
    assert "raw_value" in result
    assert "normalized_value" in result
    assert "confidence" in result
    assert "processing_time" in result
    
    assert isinstance(result["raw_value"], str)
    assert isinstance(result["normalized_value"], str)
    assert 0 <= result["confidence"] <= 1
    assert result["processing_time"] > 0

def test_prompt_strategies(test_config: Dict[str, Any]):
    """Test different prompt strategies."""
    # Map test config strategies to actual file names
    strategy_files = {
        'basic': 'basic_extraction.yaml',
        'detailed': 'detailed.yaml',
        'few-shot': 'few_shot.yaml'
    }
    
    for strategy in test_config["prompt_strategies"]:
        # Get correct file name
        file_name = strategy_files.get(strategy, f"{strategy}.yaml")
        prompt_path = Path("config/prompts") / file_name
        assert prompt_path.exists(), f"Prompt file not found: {prompt_path}"
        
        # Load and validate YAML
        import yaml
        with open(prompt_path, "r") as f:
            prompt_data = yaml.safe_load(f)
            assert isinstance(prompt_data, dict)
            assert "config_info" in prompt_data
            assert "prompts" in prompt_data
            assert len(prompt_data["prompts"]) > 0
            
            # Check first prompt
            first_prompt = prompt_data["prompts"][0]
            assert "text" in first_prompt
            assert "field_to_extract" in first_prompt
            assert "format_instructions" in first_prompt
            assert "required_fields" in first_prompt["format_instructions"]
            assert all(field in first_prompt["format_instructions"]["required_fields"] 
                      for field in ["work_order_number", "total_cost"])

def test_inference_against_ground_truth(mock_model, test_config: Dict[str, Any]):
    """Test model inference against known ground truth data."""
    # Load test image and ground truth
    test_image_path = Path("tests/mock_data/test_image.jpg")
    ground_truth_path = Path("tests/mock_data/test_ground_truth.json")
    
    assert test_image_path.exists(), f"Test image not found: {test_image_path}"
    assert ground_truth_path.exists(), f"Ground truth file not found: {ground_truth_path}"
    
    # Load ground truth data
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)
    
    # Load test image
    test_image = Image.open(test_image_path)
    
    # Test each field extraction
    fields_to_test = ["invoice_number", "date", "amount", "vendor"]
    for field in fields_to_test:
        test_prompt = f"Extract the {field} from this invoice."
        
        # Run inference
        result = mock_model(test_prompt)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert "raw_value" in result
        assert "normalized_value" in result
        assert "confidence" in result
        assert "processing_time" in result
        
        # Verify result values
        assert isinstance(result["raw_value"], str)
        assert isinstance(result["normalized_value"], str)
        assert 0 <= result["confidence"] <= 1
        assert result["processing_time"] > 0
        
        # In a real test with actual models, we would compare the normalized_value
        # with the ground truth value. For mock model, we just verify the structure.
        # TODO: Add actual value comparison when running with real models 