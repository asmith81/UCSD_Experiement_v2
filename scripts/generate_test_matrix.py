"""
Generate Test Matrix Script

This script generates a complete test matrix with all combinations of:
- Images (20)
- Prompt Types (5)
- Quantization Levels (4)
- Field Types (1)
- Models (1)
"""

import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_test_matrix():
    """Generate a complete test matrix with all combinations."""
    # Configuration
    MODEL_NAME = "pixtral"
    QUANT_LEVELS = [4, 8, 16, 32]
    PROMPT_TYPES = [
        "basic_extraction",
        "detailed",
        "few_shot",
        "locational",
        "step_by_step"
    ]
    FIELD_TYPE = "both"
    
    # Get all image files
    data_dir = Path("data/images")
    image_files = sorted(list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png")))
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {data_dir}")
    
    logger.info(f"Found {len(image_files)} images")
    
    # Generate test cases
    test_cases = []
    for image_path in image_files:
        for prompt_type in PROMPT_TYPES:
            for quant_level in QUANT_LEVELS:
                test_case = {
                    "model_name": MODEL_NAME,
                    "prompt_type": prompt_type,
                    "quant_level": quant_level,
                    "field_type": FIELD_TYPE,
                    "image_path": str(image_path)
                }
                test_cases.append(test_case)
    
    # Create test matrix
    test_matrix = {
        "test_cases": test_cases
    }
    
    # Save to file
    output_path = Path("config/test_matrix.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(test_matrix, f, indent=4)
    
    logger.info(f"Generated {len(test_cases)} test cases")
    logger.info(f"Saved test matrix to {output_path}")

if __name__ == "__main__":
    generate_test_matrix() 