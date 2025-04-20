"""
Standardized image processing implementation.
"""

from typing import Dict, Any, Union, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .config_protocol import ConfigManager
from .validation import ValidationManager

class ImageProcessor:
    """Standardized image processing implementation."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        validation_manager: ValidationManager
    ):
        """Initialize image processor.
        
        Args:
            config_manager: Configuration manager instance
            validation_manager: Validation manager instance
        """
        self.config_manager = config_manager
        self.validation_manager = validation_manager
        self.config = self.config_manager.get_section("image_processing")
        
    def __call__(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        config: Dict[str, Any] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process image according to configuration.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            config: Optional processing configuration override
            
        Returns:
            Tuple of (processed tensor, metadata)
        """
        try:
            # Use provided config or default
            config = config or self.config
            
            # Load image if path provided
            if isinstance(image, (str, Path)):
                image = Image.open(image)
                
            # Convert to PIL Image if numpy array
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
                
            # Validate image
            self.validation_manager.validate_image(image)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Get target size
            target_size = config.get('target_size', (1024, 1024))
            
            # Resize while maintaining aspect ratio
            image = self._resize_image(image, target_size)
            
            # Convert to tensor
            tensor = self._to_tensor(image)
            
            # Normalize if specified
            if config.get('normalize', True):
                tensor = self._normalize_tensor(tensor)
                
            # Get metadata
            metadata = {
                'original_size': image.size,
                'processed_size': tensor.shape[-2:],
                'normalized': config.get('normalize', True)
            }
            
            return tensor, metadata
            
        except Exception as e:
            raise RuntimeError(f"Failed to process image: {str(e)}")
            
    def _resize_image(
        self,
        image: Image.Image,
        target_size: Tuple[int, int]
    ) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        # Calculate aspect ratio
        width, height = image.size
        target_width, target_height = target_size
        
        # Calculate new dimensions
        ratio = min(target_width/width, target_height/height)
        new_size = (int(width * ratio), int(height * ratio))
        
        # Resize image
        return image.resize(new_size, Image.Resampling.LANCZOS)
        
    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor."""
        # Convert to numpy array
        array = np.array(image)
        
        # Convert to tensor and permute dimensions
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        
        # Add batch dimension
        return tensor.unsqueeze(0)
        
    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to [0, 1] range."""
        return tensor.float() / 255.0 