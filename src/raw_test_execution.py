"""
Raw test execution module.

This module handles the execution of tests and logging of raw outputs without
any processing or evaluation. It follows a simple, reliable approach to ensure
test results are captured even if processing fails.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Callable
from datetime import datetime
from src.prompts import load_prompt_template
from src.models.pixtral import process_image_wrapper
from src.data_utils import DataConfig, DefaultImageProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_raw_result(
    output_dir: Path,
    test_id: str,
    test_parameters: Dict[str, Any],
    raw_output: str,
    processing_time: float,
    error: str = None
) -> None:
    """
    Log raw test result to a JSON file.
    
    Args:
        output_dir: Directory to save results
        test_id: Unique identifier for the test
        test_parameters: Test configuration parameters
        raw_output: Raw model output text
        processing_time: Time taken for processing
        error: Optional error message if test failed
    """
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create result structure
        result = {
            "timestamp": datetime.now().isoformat(),
            "test_id": test_id,
            "test_parameters": test_parameters,
            "raw_output": raw_output,
            "processing_time": processing_time
        }
        
        # Add error if present
        if error:
            result["error"] = error
            
        # Save to file
        output_file = output_dir / f"{test_id}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
            
        logger.info(f"Raw result logged to {output_file}")
        
    except Exception as e:
        logger.error(f"Error logging raw result: {str(e)}")
        raise

def run_raw_test(
    model,
    prompt_template: str,
    image_path: str,
    test_parameters: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    Run a single test and log raw output.
    
    Args:
        model: Loaded model instance
        prompt_template: Prompt template to use
        image_path: Path to input image
        test_parameters: Test configuration parameters
        output_dir: Directory to save results
    """
    import time
    
    test_id = f"{test_parameters['model']}_{test_parameters['quantization']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Create data config
        config = DataConfig(
            image_dir=Path(image_path).parent,
            ground_truth_csv=Path(image_path).parent / "ground_truth.csv",  # Default path, can be overridden
            image_extensions=['.jpg', '.png', '.jpeg'],
            max_image_size=1024,
            supported_formats=['RGB'],
            image_processor=DefaultImageProcessor()
        )
        
        # Run inference using process_image_wrapper
        start_time = time.time()
        result = process_image_wrapper(
            model=model,
            prompt_template=prompt_template,
            image_path=image_path,
            field_type=test_parameters['field_type'],
            config=config
        )
        processing_time = time.time() - start_time
        
        # Log raw result
        log_raw_result(
            output_dir=output_dir,
            test_id=test_id,
            test_parameters=test_parameters,
            raw_output=result['raw_text'],
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in test execution: {str(e)}")
        # Log error result
        log_raw_result(
            output_dir=output_dir,
            test_id=test_id,
            test_parameters=test_parameters,
            raw_output="",
            processing_time=0.0,
            error=str(e)
        )
        raise

def run_raw_test_suite(
    model_loader: Callable,
    test_matrix: List[Dict[str, Any]],
    output_dir: Path,
    prompts_dir: Path
) -> None:
    """
    Run a suite of tests and log raw outputs.
    
    Args:
        model_loader: Function to load model with specific quantization
        test_matrix: List of test cases to run
        output_dir: Directory to save results
        prompts_dir: Directory containing prompt templates
    """
    # Group tests by quantization to minimize model reloads
    quantization_groups = {}
    for test_case in test_matrix:
        quant_level = test_case['quantization_level']
        if quant_level not in quantization_groups:
            quantization_groups[quant_level] = []
        quantization_groups[quant_level].append(test_case)
    
    # Run tests for each quantization level
    for quant_level, test_cases in quantization_groups.items():
        logger.info(f"Running tests for {quant_level}-bit quantization")
        
        try:
            # Load model once for this quantization level
            model = model_loader(quant_level)
            
            # Group test cases by prompt type to minimize prompt loading
            prompt_groups = {}
            for test_case in test_cases:
                prompt_type = test_case['prompt_type']
                if prompt_type not in prompt_groups:
                    prompt_groups[prompt_type] = []
                prompt_groups[prompt_type].append(test_case)
            
            # Run tests for each prompt type
            for prompt_type, cases in prompt_groups.items():
                logger.info(f"Loading prompt template for type: {prompt_type}")
                try:
                    # Load prompt template once for this group
                    prompt_template = load_prompt_template(
                        prompt_strategy=prompt_type,
                        prompts_dir=prompts_dir
                    )
                    
                    # Run each test case with this prompt
                    for test_case in cases:
                        test_parameters = {
                            "model": test_case["model"],
                            "quantization": quant_level,
                            "prompt_type": prompt_type,
                            "field_type": test_case["field_type"],
                            "image_path": test_case["image_path"]
                        }
                        
                        run_raw_test(
                            model=model,
                            prompt_template=prompt_template,
                            image_path=test_case["image_path"],
                            test_parameters=test_parameters,
                            output_dir=output_dir
                        )
                        
                except Exception as e:
                    logger.error(f"Error processing prompt type {prompt_type}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in quantization level {quant_level}: {str(e)}")
            continue 