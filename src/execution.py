"""
Standardized test execution module.
Implements the Test Execution interface from the interface control document.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union
from PIL import Image
from datetime import datetime

from .models.loader import load_model
from .environment import setup_environment

logger = logging.getLogger(__name__)

class ModelLoader(Protocol):
    """Protocol for model loading functions."""
    def __call__(
        self,
        model_name: str,
        config: Dict[str, Any],
        models_dir: Optional[Path] = None
    ) -> Any:
        ...

class ModelProcessor(Protocol):
    """Protocol for model processing functions."""
    def __call__(
        self, 
        model: Any, 
        image: Union[Image.Image, Dict[str, Any]], 
        prompt: str
    ) -> Dict[str, Any]:
        ...

class StandardModelProcessor:
    """Standardized model processing implementation."""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        
    def _validate_input(self, image: Union[Image.Image, Dict[str, Any]], prompt: str) -> None:
        """Validate input format and content."""
        if not isinstance(image, (Image.Image, dict)):
            raise ValueError("Image must be PIL Image or dict")
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be string")
            
    def _format_prompt(self, prompt: str, image: Union[Image.Image, Dict[str, Any]]) -> Any:
        """Format prompt based on model type."""
        if self.model_type == "pixtral":
            from .models.pixtral import process_pixtral_image
            return process_pixtral_image(self.model, self.processor, image, prompt)
        elif self.model_type == "llama":
            from .models.llama import process_llama_image
            return process_llama_image(self.model, self.processor, image, prompt)
        elif self.model_type == "doctr":
            from .models.doctr import process_doctr_image
            return process_doctr_image(self.model, image)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
    def _log_raw_output(self, raw_output: Any, test_id: str) -> None:
        """Log raw model output."""
        output_dir = Path("results/raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{test_id}_raw.json"
        with open(output_path, "w") as f:
            json.dump(raw_output, f, indent=2)
            
    def process(self, model: Any, processor: Any, image: Union[Image.Image, Dict[str, Any]], prompt: str, test_id: str) -> Dict[str, Any]:
        """Process input through model and return standardized output."""
        try:
            self.model = model
            self.processor = processor
            self._validate_input(image, prompt)
            result = self._format_prompt(prompt, image)
            self._log_raw_output(result, test_id)
            return result
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            raise

def run_test_suite(
    model_name: str,
    test_matrix_path: Union[str, Path],
    model_loader: Optional[ModelLoader] = None,
    processor: Optional[StandardModelProcessor] = None,
    prompt_loader: Optional[callable] = None,
    result_validator: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """Run test suite for a model."""
    try:
        # Load test matrix
        with open(test_matrix_path, 'r') as f:
            test_matrix = json.load(f)
            
        if 'test_cases' not in test_matrix:
            raise ValueError("Test matrix missing 'test_cases' key")
            
        # Filter test cases for this model
        model_tests = [
            test for test in test_matrix['test_cases']
            if test['model_name'] == model_name
        ]
        
        results = []
        
        # Run each test case
        for test_case in model_tests:
            try:
                # Load model
                if model_loader is None:
                    model, processor = load_model(model_name, test_case)
                else:
                    model = model_loader(model_name, test_case)
                    
                # Load prompt
                if prompt_loader is not None:
                    prompt = prompt_loader(test_case['prompt_type'])
                else:
                    prompt = test_case['prompt']
                    
                # Process image
                if processor is not None:
                    result = processor.process(model, processor, test_case['image'], prompt, test_case.get('test_id', 'unknown'))
                else:
                    result = _default_processor(model, test_case['image'], prompt)
                    
                # Validate result
                if result_validator is not None:
                    result = result_validator(result)
                    
                # Add test parameters to result
                result['test_parameters'] = test_case
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Test case failed: {str(e)}")
                continue
                
        return results
        
    except Exception as e:
        logger.error(f"Test suite execution failed: {str(e)}")
        raise RuntimeError(f"Test execution failed: {str(e)}")

def _default_processor(
    model: Any,
    image: Union[Image.Image, Dict[str, Any]],
    prompt: str
) -> Dict[str, Any]:
    """Default model processor implementation."""
    try:
        # Process image if it's a path
        if isinstance(image, dict) and 'path' in image:
            image = Image.open(image['path'])
            
        # Run inference
        inputs = model.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(model.device)
        
        outputs = model.generate(**inputs)
        result = model.processor.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'raw_value': result,
            'processing_time': 0.0  # TODO: Add timing
        }
        
    except Exception as e:
        logger.error(f"Model processing failed: {str(e)}")
        raise RuntimeError(f"Processing failed: {str(e)}") 