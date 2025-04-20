"""
Data management utilities for LMM invoice data extraction comparison.
Handles image loading and ground truth data management with type normalization.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, TypedDict, Protocol
import pandas as pd
import numpy as np
from PIL import Image
import logging
from .environment import EnvironmentConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageProcessor(Protocol):
    """Protocol for image processing functions."""
    def process_image(self, image: Image.Image, config: 'DataConfig') -> Image.Image:
        ...

class DataConfig:
    """Configuration for data processing."""
    
    def __init__(
        self,
        image_dir: Path,
        ground_truth_csv: Path,
        image_extensions: List[str],
        max_image_size: int,
        supported_formats: List[str],
        image_processor: Optional[ImageProcessor] = None
    ):
        self.image_dir = image_dir
        self.ground_truth_csv = ground_truth_csv
        self.image_extensions = image_extensions
        self.max_image_size = max_image_size
        self.supported_formats = supported_formats
        self.image_processor = image_processor

class DefaultImageProcessor:
    """Default image processing implementation."""
    def process_image(self, image: Image.Image, config: DataConfig) -> Image.Image:
        """Process image according to configuration."""
        # Convert format if needed
        if image.mode not in config.supported_formats:
            image = image.convert('RGB')
            
        # Resize if needed
        if max(image.size) > config.max_image_size:
            ratio = config.max_image_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
        return image

class ImageData(TypedDict):
    """Structure for image data."""
    image_id: str
    image_path: Path
    image_size: tuple[int, int]
    format: str

class GroundTruthData(TypedDict):
    """Structure for ground truth data."""
    work_order_number: Dict[str, str]
    total_cost: Dict[str, float]

def setup_data_paths(
    env_config: EnvironmentConfig,
    image_processor: Optional[ImageProcessor] = None,
    image_extensions: Optional[List[str]] = None,
    max_image_size: Optional[int] = None,
    supported_formats: Optional[List[str]] = None
) -> DataConfig:
    """Setup data paths using environment configuration.
    
    Args:
        env_config: Environment configuration from environment.py
        image_processor: Optional image processor implementation
        image_extensions: Optional list of supported image extensions
        max_image_size: Optional maximum image size
        supported_formats: Optional list of supported image formats
        
    Returns:
        DataConfig: Data configuration with paths and settings
        
    Raises:
        OSError: If required directories don't exist
    """
    try:
        # Create DataConfig object
        data_config = DataConfig(
            image_dir=env_config['data_dir'] / 'images',
            ground_truth_csv=env_config['data_dir'] / 'ground_truth.csv',
            image_extensions=image_extensions or ['.jpg', '.jpeg', '.png'],
            max_image_size=max_image_size or 1120,
            supported_formats=supported_formats or ['RGB', 'L'],
            image_processor=image_processor or DefaultImageProcessor()
        )
        
        # Validate paths
        if not data_config.image_dir.exists():
            raise OSError(f"Image directory not found: {data_config.image_dir}")
        if not data_config.ground_truth_csv.exists():
            raise OSError(f"Ground truth CSV not found: {data_config.ground_truth_csv}")
            
        return data_config
        
    except Exception as e:
        logger.error(f"Error setting up data paths: {str(e)}")
        raise

def validate_data_config(config: DataConfig) -> bool:
    """Validate data configuration structure.
    
    Args:
        config: Data configuration to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = ['image_dir', 'ground_truth_csv', 'image_extensions', 
                      'max_image_size', 'supported_formats']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in data config: {field}")
            
    if not isinstance(config['max_image_size'], int):
        raise ValueError("max_image_size must be an integer")
        
    if not all(isinstance(ext, str) for ext in config['image_extensions']):
        raise ValueError("image_extensions must be a list of strings")
        
    return True

def load_image(
    image_path: Union[str, Path],
    config: DataConfig,
    processor: Optional[ImageProcessor] = None
) -> Image.Image:
    """Load and preprocess image according to configuration."""
    try:
        # Convert path to Path object
        image_path = Path(image_path)
        
        # If path is relative, join with image directory from config
        if not image_path.is_absolute():
            image_path = config.image_dir / image_path
            
        # Check if file exists
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Load image
        image = Image.open(image_path)
        
        # Convert format if needed
        if image.mode not in config.supported_formats:
            image = image.convert('RGB')
            
        # Resize if needed
        if config.max_image_size:
            ratio = min(config.max_image_size / max(image.size), 1.0)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            if new_size != image.size:
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                
        # Process image if processor is provided
        processor = processor or config.image_processor
        if processor:
            image = processor.process_image(image, config)
            
        return image
        
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        raise

def normalize_work_order(value: str) -> str:
    """
    Normalize work order number to consistent format.
    
    Args:
        value: Raw work order number
        
    Returns:
        Normalized work order number
    """
    if pd.isna(value):
        return ""
        
    # Convert to string and strip whitespace
    return str(value).strip()

def normalize_total_cost(value: Union[str, float]) -> float:
    """
    Normalize total cost to float value.
    
    Args:
        value: Raw total cost value
        
    Returns:
        Normalized float value
        
    Raises:
        ValueError: If value cannot be normalized
    """
    if pd.isna(value):
        return 0.0
        
    # Convert to string for processing
    str_value = str(value)
    
    # Remove currency symbols and whitespace
    clean_value = str_value.replace('$', '').strip()
    
    # Remove commas and convert to float
    try:
        return float(clean_value.replace(',', ''))
    except ValueError:
        logger.error(f"Could not normalize total cost value: {value}")
        return 0.0

def load_ground_truth(csv_path: Union[str, Path]) -> Dict[str, GroundTruthData]:
    """
    Load and normalize ground truth data from CSV.
    
    Args:
        csv_path: Path to ground truth CSV file
        
    Returns:
        Dictionary mapping image IDs to normalized ground truth values
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid
    """
    try:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")
            
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_columns = ['Invoice', 'Work Order Number/Numero de Orden', 'Total']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Ground truth CSV missing required columns: {missing_columns}")
            
        # Initialize results dictionary
        ground_truth: Dict[str, GroundTruthData] = {}
        
        # Process each row
        for _, row in df.iterrows():
            image_id = str(row['Invoice'])
            
            # Normalize values
            work_order = normalize_work_order(row['Work Order Number/Numero de Orden'])
            total_cost = normalize_total_cost(row['Total'])
            
            # Store normalized values
            ground_truth[image_id] = {
                'work_order_number': {
                    'raw_value': str(row['Work Order Number/Numero de Orden']),
                    'normalized_value': work_order
                },
                'total_cost': {
                    'raw_value': str(row['Total']),
                    'normalized_value': total_cost
                }
            }
            
        return ground_truth
        
    except Exception as e:
        logger.error(f"Error loading ground truth: {str(e)}")
        raise

def prepare_data_for_logging(data: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare data in format expected by results logging.
    
    Args:
        data: Raw data to prepare
        
    Returns:
        Data formatted for logging
    """
    return {
        'raw_data': data,
        'normalized_data': {
            'work_order_number': normalize_work_order(data.get('work_order_number', '')),
            'total_cost': normalize_total_cost(data.get('total_cost', 0.0))
        }
    }

def validate_image_directory(image_dir: Union[str, Path], config: DataConfig) -> List[ImageData]:
    """
    Validate image directory and return list of valid image files.
    
    Args:
        image_dir: Path to image directory
        config: Data configuration
        
    Returns:
        List of valid image data
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no valid images found
    """
    try:
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
            
        # Get all image files
        image_files: List[ImageData] = []
        
        for file in image_dir.glob('*'):
            if file.suffix.lower() in config['image_extensions']:
                try:
                    with Image.open(file) as img:
                        image_files.append({
                            'image_id': file.stem,
                            'image_path': file,
                            'image_size': img.size,
                            'format': img.mode
                        })
                except Exception as e:
                    logger.warning(f"Could not process image {file}: {str(e)}")
                    
        if not image_files:
            raise ValueError(f"No valid images found in directory: {image_dir}")
            
        return image_files
        
    except Exception as e:
        logger.error(f"Error validating image directory: {str(e)}")
        raise

def validate_ground_truth(ground_truth_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate ground truth data structure.
    
    Args:
        ground_truth_path: Path to ground truth file
        
    Returns:
        Validated ground truth data
        
    Raises:
        FileNotFoundError: If ground truth file not found
        ValueError: If data is invalid
    """
    try:
        ground_truth_path = Path(ground_truth_path)
        if not ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
            
        # Load and validate data
        if ground_truth_path.suffix == '.csv':
            data = load_ground_truth(ground_truth_path)
        elif ground_truth_path.suffix == '.json':
            import json
            with open(ground_truth_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported ground truth file format: {ground_truth_path.suffix}")
            
        # Validate structure
        for image_id, gt_data in data.items():
            if not isinstance(gt_data, dict):
                raise ValueError(f"Invalid ground truth data for image {image_id}")
                
            if 'work_order_number' not in gt_data or 'total_cost' not in gt_data:
                raise ValueError(f"Missing required fields in ground truth data for image {image_id}")
                
            # Validate work order number
            if not isinstance(gt_data['work_order_number'], dict):
                raise ValueError(f"Invalid work order number data for image {image_id}")
                
            # Validate total cost
            if not isinstance(gt_data['total_cost'], dict):
                raise ValueError(f"Invalid total cost data for image {image_id}")
                
        return data
        
    except Exception as e:
        logger.error(f"Error validating ground truth data: {str(e)}")
        raise 