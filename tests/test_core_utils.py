"""
Tests for core utilities that can run in both local and remote environments.
"""

import pytest
from pathlib import Path
from src.environment import setup_paths, check_cuda_availability, get_environment_overrides
from src.data_utils import load_image, validate_ground_truth, DataConfig, DefaultImageProcessor
from src.results_logging import log_result, load_result

def test_setup_paths(project_root: Path):
    """Test directory structure setup."""
    paths = setup_paths(project_root)
    
    # Check all required directories exist
    required_dirs = ['data_dir', 'models_dir', 'results_dir', 'logs_dir', 
                    'config_dir', 'prompts_dir']
    for dir_name in required_dirs:
        assert dir_name in paths
        assert paths[dir_name].exists()
        assert paths[dir_name].is_dir()

def test_environment_overrides(environment: str):
    """Test environment-specific overrides."""
    overrides = get_environment_overrides()
    
    if environment == "local":
        assert overrides['use_gpu'] is False
        assert overrides['log_level'] == 'DEBUG'
    else:
        assert overrides['use_gpu'] is True
        assert overrides['log_level'] == 'INFO'

def test_cuda_availability(cuda_available: bool, environment: str):
    """Test CUDA availability check."""
    cuda_info = check_cuda_availability()
    
    if environment == "local":
        # Local environment may or may not have CUDA
        assert isinstance(cuda_info['cuda_available'], bool)
    else:
        # Remote environment should have CUDA
        assert cuda_info['cuda_available'] is True
        assert cuda_info['cuda_version'] is not None

def test_mock_data_loading(mock_data_dir: Path):
    """Test loading mock data."""
    # Create mock data if it doesn't exist
    mock_image = mock_data_dir / "test_image.jpg"
    mock_gt = mock_data_dir / "test_ground_truth.json"
    
    # Check if files exist
    assert mock_image.exists(), f"Mock image not found: {mock_image}"
    assert mock_gt.exists(), f"Mock ground truth not found: {mock_gt}"
    
    # Load and validate data
    image = load_image(mock_image, DataConfig(
        image_dir=mock_data_dir,
        ground_truth_csv=mock_gt,
        image_extensions=['.jpg'],
        max_image_size=1024,
        supported_formats=['JPEG'],
        image_processor=DefaultImageProcessor()
    ))
    
    assert image is not None
    assert image.size[0] > 0
    assert image.size[1] > 0
    
    # Load and validate ground truth
    gt_data = validate_ground_truth(mock_gt)
    assert gt_data is not None
    assert 'test_image' in gt_data
    assert 'work_order_number' in gt_data['test_image']
    assert 'total_cost' in gt_data['test_image']

def test_result_logging(project_root: Path):
    """Test result logging functionality."""
    # Create test data with new structure
    test_model_output = {
        'output': 'TEST123',
        'field_type': 'both'
    }
    
    test_ground_truth = {
        'work_order_number': {
            'raw_value': 'TEST123',
            'normalized_value': 'TEST123'
        },
        'total_cost': {
            'raw_value': '$100.00',
            'normalized_value': 100.00
        }
    }
    
    # Test logging
    result_path = project_root / "results" / "test_result.json"
    log_result(
        result_path=result_path,
        image_id='test_image',
        model_output=test_model_output,
        ground_truth=test_ground_truth,
        processing_time=0.1,
        model_name='test_model',
        prompt_type='basic',
        quant_level=32
    )
    
    # Test loading
    loaded_result = load_result(result_path)
    assert loaded_result is not None
    assert 'meta' in loaded_result
    assert 'test_parameters' in loaded_result
    assert 'results_by_image' in loaded_result
    assert 'test_image' in loaded_result['results_by_image']
    
    # Cleanup
    if result_path.exists():
        result_path.unlink() 