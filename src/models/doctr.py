"""
Doctr model implementation.
Handles loading and processing for the Doctr model.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

from PIL import Image

from .loader import get_model_path

logger = logging.getLogger(__name__)

def load_doctr_model(model_name: str, config: Dict[str, Any], models_dir: Optional[Path] = None) -> Any:
    """Load Doctr model with proper configuration."""
    try:
        from doctr.models import ocr_predictor
        
        # Load model with configuration
        model = ocr_predictor(
            det_arch=config.get("detection_arch", "db_resnet50"),
            reco_arch=config.get("recognition_arch", "crnn_vgg16_bn"),
            pretrained=True,
            assume_straight_boxes=config.get("assume_straight_boxes", True),
            export_as_straight_boxes=config.get("export_as_straight_boxes", True)
        )
        
        return model
    
    except Exception as e:
        logger.error(f"Error loading Doctr model: {str(e)}")
        raise

def process_doctr_image(model: Any, image: Union[Image.Image, Dict[str, Any]]) -> Dict[str, Any]:
    """Process image through Doctr model."""
    try:
        # Convert image if needed
        if isinstance(image, dict) and 'path' in image:
            image = Image.open(image['path'])
            
        # Process image
        result = model(image)
        
        # Export results
        return {
            "document": result.export(),
            "model_type": "doctr",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error processing image with Doctr: {str(e)}")
        raise 