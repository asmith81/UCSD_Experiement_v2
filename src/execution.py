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

class ModelProcessor(Protocol):
    """Protocol for model processing functions."""
    def __call__(
        self, 
        model: Any, 
        image: Union[Image.Image, Dict[str, Any]], 
        prompt: str
    ) -> Dict[str, Any]:
        ...

class ModelProcessor:
    """Standardized model processing protocol implementation."""
    
    def __init__(self, model: Any, processor: Any, model_type: str):
        self.model = model
        self.processor = processor
        self.model_type = model_type
        self._validate_model_components()
        
    def _validate_model_components(self) -> None:
        """Validate model and processor components."""
        if self.model_type == "pixtral":
            if not isinstance(self.model, LlavaForConditionalGeneration):
                raise ValueError("Invalid model type for Pixtral")
            if not isinstance(self.processor, AutoProcessor):
                raise ValueError("Invalid processor type for Pixtral")
        elif self.model_type == "llama":
            if not isinstance(self.model, MllamaForConditionalGeneration):
                raise ValueError("Invalid model type for Llama")
            if not isinstance(self.processor, MllamaProcessor):
                raise ValueError("Invalid processor type for Llama")
        elif self.model_type == "doctr":
            if not isinstance(self.model, ocr_predictor):
                raise ValueError("Invalid model type for Doctr")
    
    def _validate_input(self, image: Union[Image.Image, Dict[str, Any]], prompt: str) -> None:
        """Validate input format based on model type."""
        if self.model_type in ["pixtral", "llama"]:
            if not isinstance(image, (Image.Image, dict)):
                raise ValueError(f"Invalid image type for {self.model_type}")
            if not isinstance(prompt, str):
                raise ValueError(f"Invalid prompt type for {self.model_type}")
        elif self.model_type == "doctr":
            if not isinstance(image, (Image.Image, DocumentFile)):
                raise ValueError("Invalid document type for Doctr")
    
    def _format_prompt(self, prompt: str, image: Union[Image.Image, Dict[str, Any]]) -> Any:
        """Format prompt according to model requirements."""
        if self.model_type == "pixtral":
            return self._format_pixtral_prompt(prompt, image)
        elif self.model_type == "llama":
            return self._format_llama_prompt(prompt, image)
        elif self.model_type == "doctr":
            return self._format_doctr_prompt(image)
    
    def _format_pixtral_prompt(self, prompt: str, image: Union[Image.Image, Dict[str, Any]]) -> str:
        """Format prompt for Pixtral model."""
        if isinstance(image, dict) and "images" in image:
            image_tokens = "[IMG]" * len(image["images"])
            return f"<s>[INST]{prompt}\n{image_tokens}[/INST]"
        return f"<s>[INST]{prompt}\n[IMG][/INST]"
    
    def _format_llama_prompt(self, prompt: str, image: Union[Image.Image, Dict[str, Any]]) -> Dict[str, Any]:
        """Format prompt for Llama model."""
        return {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    
    def _format_doctr_prompt(self, image: Union[Image.Image, DocumentFile]) -> DocumentFile:
        """Format input for Doctr model."""
        if isinstance(image, Image.Image):
            return DocumentFile.from_images(image)
        return image
    
    def _log_raw_output(self, raw_output: Any, test_id: str) -> None:
        """Log raw model output before processing."""
        try:
            raw_output_path = Path("results/raw") / f"{test_id}_raw.json"
            raw_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(raw_output_path, "w") as f:
                json.dump({
                    "raw_output": str(raw_output),
                    "timestamp": datetime.now().isoformat(),
                    "test_id": test_id
                }, f, indent=2)
            
            logger.info(f"Raw output logged to {raw_output_path}")
        except Exception as e:
            logger.error(f"Error logging raw output: {str(e)}")
            # Don't raise - we want to continue processing even if logging fails
    
    def _process_output(self, raw_output: Any, formatted_input: Any) -> Dict[str, Any]:
        """Process model output into standardized format."""
        try:
            if self.model_type in ["pixtral", "llama"]:
                result = self.processor.decode(raw_output[0])[len(formatted_input):]
                return {
                    "text": result,
                    "model_type": self.model_type,
                    "timestamp": datetime.now().isoformat()
                }
            elif self.model_type == "doctr":
                return {
                    "document": raw_output.export(),
                    "model_type": self.model_type,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error processing model output: {str(e)}")
            return {
                "error": str(e),
                "raw_output": str(raw_output),
                "model_type": self.model_type,
                "timestamp": datetime.now().isoformat()
            }
    
    def process(self, image: Union[Image.Image, Dict[str, Any]], prompt: str, test_id: str) -> Dict[str, Any]:
        """Process input according to model requirements."""
        self._validate_input(image, prompt)
        formatted_input = self._format_prompt(prompt, image)
        
        try:
            if self.model_type in ["pixtral", "llama"]:
                inputs = self.processor(
                    image, 
                    formatted_input, 
                    text_kwargs={"add_special_tokens": False}, 
                    return_tensors="pt"
                ).to(self.model.device)
                
                raw_output = self.model.generate(
                    **inputs,
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=2048
                )
                
                # Log raw output before processing
                self._log_raw_output(raw_output, test_id)
                
                # Process the output
                return self._process_output(raw_output, formatted_input)
            
            elif self.model_type == "doctr":
                raw_output = self.model(formatted_input)
                
                # Log raw output before processing
                self._log_raw_output(raw_output, test_id)
                
                # Process the output
                return self._process_output(raw_output, formatted_input)
        
        except Exception as e:
            logger.error(f"Error in model processing: {str(e)}")
            return {
                "error": str(e),
                "model_type": self.model_type,
                "timestamp": datetime.now().isoformat()
            }

def run_test_suite(
    model_name: str,
    test_matrix_path: Union[str, Path],
    model_loader: Optional[ModelLoader] = None,
    processor: Optional[ModelProcessor] = None,
    prompt_loader: Optional[callable] = None,
    result_validator: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Run test suite for a model.
    
    Args:
        model_name: Name of the model to test
        test_matrix_path: Path to test matrix JSON
        model_loader: Optional function to load model
        processor: Optional function to process images
        prompt_loader: Optional function to load prompts
        result_validator: Optional function to validate results
        
    Returns:
        List of test results
        
    Raises:
        FileNotFoundError: If test matrix not found
        ValueError: If test matrix is invalid
        RuntimeError: If test execution fails
    """
    try:
        # Setup environment
        env = setup_environment()
        
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
                    result = processor(model, test_case['image'], prompt)
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