"""
Result logging and evaluation module.

This module provides functionality for logging and evaluating test results
across different models and configurations.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, TypedDict, Protocol
from datetime import datetime
import pandas as pd
from src.data_utils import normalize_work_order, normalize_total_cost, GroundTruthData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResultStorage(Protocol):
    """Protocol for result storage implementations."""
    def save_result(self, result_path: Union[str, Path], result: Dict[str, Any]) -> None:
        ...
    def load_result(self, result_path: Union[str, Path]) -> Dict[str, Any]:
        ...

class FileSystemStorage:
    """Default file system storage implementation."""
    def save_result(self, result_path: Union[str, Path], result: Dict[str, Any]) -> None:
        """Save result to file system."""
        result_path = Path(result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
            
    def load_result(self, result_path: Union[str, Path]) -> Dict[str, Any]:
        """Load result from file system."""
        result_path = Path(result_path)
        with open(result_path, 'r') as f:
            return json.load(f)

class ModelOutput(TypedDict):
    """Structure for model output data."""
    raw_text: str
    parsed_value: str
    normalized_value: Union[str, float]

class ModelResponse(TypedDict):
    """Structure for model response data."""
    raw_text: str
    work_order_number: ModelOutput
    total_cost: ModelOutput
    processing_time: float

class EvaluationResult(TypedDict):
    """Structure for evaluation results."""
    normalized_match: bool
    cer: float
    error_category: str

class ResultEntry(TypedDict):
    """Structure for a single result entry."""
    ground_truth: GroundTruthData
    model_response: ModelResponse
    evaluation: Dict[str, EvaluationResult]

class ResultStructure(TypedDict):
    """Structure for complete result file."""
    meta: Dict[str, str]
    test_parameters: Dict[str, Any]
    results_by_image: Dict[str, ResultEntry]

def validate_ground_truth(ground_truth: Dict[str, Any]) -> bool:
    """
    Validate ground truth data structure.
    
    Args:
        ground_truth: Ground truth data to validate
        
    Returns:
        True if structure is valid
        
    Raises:
        ValueError: If structure is invalid
    """
    required_fields = ['work_order_number', 'total_cost']
    for field in required_fields:
        if field not in ground_truth:
            raise ValueError(f"Missing required ground truth field: {field}")
    return True

def normalize_total_cost(value: Union[str, int, float]) -> str:
    """
    Normalize total cost value to float.
    
    Args:
        value: Value to normalize (string or float)
        
    Returns:
        Normalized float value
    """
    if isinstance(value, (int, float)):
        return str(value)
    return ''.join(c for c in str(value) if c.isdigit() or c == '.')

def calculate_cer(pred: str, true: str) -> float:
    """
    Calculate Character Error Rate between predicted and true values.
    
    Args:
        pred: Predicted string
        true: True string
        
    Returns:
        Character Error Rate (0.0 to 1.0)
    """
    if not true:
        return 1.0  # Maximum error if true value is empty
        
    # Levenshtein distance calculation
    if len(pred) < len(true):
        return calculate_cer(true, pred)
        
    if len(true) == 0:
        return len(pred)
        
    previous_row = range(len(true) + 1)
    for i, c1 in enumerate(pred):
        current_row = [i + 1]
        for j, c2 in enumerate(true):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1] / len(true)

def categorize_error(pred: str, true: str, field_type: str) -> str:
    """
    Categorize error type based on field and difference.
    
    Args:
        pred: Predicted value
        true: True value
        field_type: Type of field ('work_order' or 'total_cost')
        
    Returns:
        Error category string
    """
    if not pred or not true:
        return "missing_field"
        
    # Return no_error if values match exactly
    if pred == true:
        return "no_error"
        
    if field_type == "work_order":
        # Check for transposition
        if sorted(pred) == sorted(true) and pred != true:
            return "transposition"
            
        # Check for length differences
        if len(pred) < len(true):
            return "missing_character"
        if len(pred) > len(true):
            return "extra_character"
            
        return "wrong_character"
        
    elif field_type == "total_cost":
        # Check for currency symbol error
        if ('$' in pred) != ('$' in true):
            return "currency_error"
            
        # Check for decimal point errors
        if ('.' in pred) != ('.' in true):
            return "decimal_error"
            
        # Check for formatting errors
        if any(c in pred for c in ',.') != any(c in true for c in ',.'):
            return "formatting_error"
            
        return "digit_error"
        
    return "unknown_error"

def create_result_structure(
    model_name: str,
    prompt_type: str,
    quant_level: int,
    environment: Optional[str] = None
) -> ResultStructure:
    """
    Create base result structure for a test run.
    
    Args:
        model_name: Name of the model being tested
        prompt_type: Type of prompt used
        quant_level: Quantization level used
        environment: Optional testing environment description
        
    Returns:
        Base result structure dictionary
    """
    return {
        "meta": {
            "experiment_id": f"exp-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "environment": environment or "RunPod T4 GPU"
        },
        "test_parameters": {
            "model_name": model_name,
            "field_type": "both",
            "prompt_type": prompt_type,
            "quant_level": quant_level
        },
        "results_by_image": {}
    }

def validate_model_output(model_output: Dict[str, Any]) -> bool:
    """
    Validate model output structure.
    
    Args:
        model_output: Model output to validate
        
    Returns:
        True if structure is valid
        
    Raises:
        ValueError: If structure is invalid
    """
    required_fields = ['work_order_number', 'total_cost']
    required_subfields = ['raw_text', 'parsed_value', 'normalized_value']
    
    for field in required_fields:
        if field not in model_output:
            raise ValueError(f"Missing required field: {field}")
            
        for subfield in required_subfields:
            if subfield not in model_output[field]:
                raise ValueError(f"Missing required subfield {subfield} in {field}")
                
    return True

def extract_json_from_output(output_text: str) -> Dict[str, Any]:
    """
    Extract JSON object from model output text.
    
    Args:
        output_text: Raw model output text
        
    Returns:
        Extracted JSON object as dictionary
    """
    try:
        # Find the last JSON object in the output
        json_start = output_text.rfind('{')
        json_end = output_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = output_text[json_start:json_end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON: {json_str}")
                return {}
        return {}
    except Exception as e:
        logger.warning(f"Error extracting JSON: {str(e)}")
        return {}

def evaluate_model_output(
    model_output: str,
    ground_truth: GroundTruthData,
    field_type: str
) -> Dict[str, Any]:
    """
    Evaluate model output against ground truth.
    
    Args:
        model_output: Raw model output text
        ground_truth: Ground truth data
        field_type: Type of field being evaluated ('work_order_number', 'total_cost', or 'both')
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Extract values from model output
    parsed_output = extract_json_from_output(model_output)
    work_order_pred = parsed_output.get('work_order_number', '')
    total_cost_pred = parsed_output.get('total_cost', '')
    
    # Get ground truth values
    work_order_truth = ground_truth.get('work_order_number', '')
    total_cost_truth = ground_truth.get('total_cost', '')
    
    # Normalize values for comparison
    def normalize_work_order(value: str) -> str:
        return ''.join(c for c in str(value) if c.isalnum())
        
    def normalize_total_cost(value: Union[str, int, float]) -> str:
        # Convert numeric values to string with 2 decimal places
        if isinstance(value, (int, float)):
            return f"{float(value):.2f}"
        # For strings, remove currency symbols and whitespace, keep only digits and decimal
        return ''.join(c for c in str(value) if c.isdigit() or c == '.')
    
    # Normalize values
    work_order_pred_norm = normalize_work_order(work_order_pred)
    work_order_truth_norm = normalize_work_order(work_order_truth)
    total_cost_pred_norm = normalize_total_cost(total_cost_pred)
    total_cost_truth_norm = normalize_total_cost(total_cost_truth)
    
    # Initialize result dictionary
    result = {}
    
    # Evaluate work order number if requested
    if field_type in ['work_order_number', 'both']:
        work_order_cer = calculate_cer(work_order_pred_norm, work_order_truth_norm)
        result['work_order_number'] = {
            'raw_string_match': work_order_pred == work_order_truth,
            'normalized_match': work_order_pred_norm == work_order_truth_norm,
            'cer': work_order_cer,
            'error_category': categorize_error(work_order_pred_norm, work_order_truth_norm, 'work_order')
        }
    
    # Evaluate total cost if requested
    if field_type in ['total_cost', 'both']:
        total_cost_cer = calculate_cer(total_cost_pred_norm, total_cost_truth_norm)
        result['total_cost'] = {
            'raw_string_match': total_cost_pred == total_cost_truth,
            'normalized_match': total_cost_pred_norm == total_cost_truth_norm,
            'cer': total_cost_cer,
            'error_category': categorize_error(total_cost_pred_norm, total_cost_truth_norm, 'total_cost')
        }
    
    return result

def log_result(
    result_path: Union[str, Path],
    image_id: str,
    model_output: Dict[str, Any],
    ground_truth: GroundTruthData,
    processing_time: float,
    model_name: str,
    prompt_type: str,
    quant_level: int,
    environment: Optional[str] = None,
    storage: Optional[ResultStorage] = None
) -> None:
    """
    Log result for a single image.
    
    Args:
        result_path: Path to result file
        image_id: ID of the image
        model_output: Model output for the image
        ground_truth: Ground truth for the image
        processing_time: Time taken for processing
        model_name: Name of the model being tested
        prompt_type: Type of prompt used
        quant_level: Quantization level used
        environment: Optional testing environment description
        storage: Optional result storage implementation
        
    Raises:
        ValueError: If model output is invalid
    """
    try:
        # Use provided storage or default
        storage = storage or FileSystemStorage()
        
        # Create result structure
        result = create_result_structure(model_name, prompt_type, quant_level, environment)
        
        # Get field type from model output
        field_type = model_output.get('field_type', 'both')
        
        # Extract JSON part from model output
        output_text = model_output['output']
        parsed_output = extract_json_from_output(output_text)
        
        # Evaluate model output
        evaluation = evaluate_model_output(
            output_text,
            ground_truth,
            field_type
        )
        
        # Add result entry
        result['results_by_image'][image_id] = {
            "ground_truth": ground_truth,
            "model_response": {
                "work_order_number": {
                    "raw_text": output_text,
                    "parsed_value": parsed_output.get('work_order_number', ''),
                    "normalized_value": normalize_work_order(
                        parsed_output.get('work_order_number', '')
                    )
                },
                "total_cost": {
                    "raw_text": output_text,
                    "parsed_value": parsed_output.get('total_cost', ''),
                    "normalized_value": normalize_total_cost(
                        parsed_output.get('total_cost', '')
                    )
                },
                "processing_time": processing_time
            },
            "evaluation": evaluation
        }
        
        # Save result
        storage.save_result(result_path, result)
        
    except Exception as e:
        logger.error(f"Error logging result: {str(e)}")
        raise

def track_execution(
    execution_log_path: Union[str, Path],
    model_name: str,
    prompt_type: str,
    quant_level: int,
    status: str,
    error: Optional[str] = None
) -> None:
    """
    Track execution status for a test run.
    
    Args:
        execution_log_path: Path to execution log file
        model_name: Name of the model
        prompt_type: Type of prompt used
        quant_level: Quantization level used
        status: Execution status
        error: Error message if any
    """
    try:
        execution_log_path = Path(execution_log_path)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "prompt_type": prompt_type,
            "quant_level": quant_level,
            "status": status,
            "error": error
        }
        
        # Append to execution log
        if execution_log_path.exists():
            df = pd.read_csv(execution_log_path)
        else:
            df = pd.DataFrame(columns=[
                'timestamp', 'model_name', 'prompt_type',
                'quant_level', 'status', 'error'
            ])
            
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
        df.to_csv(execution_log_path, index=False)
        
    except Exception as e:
        logger.error(f"Error tracking execution: {str(e)}")
        raise

def load_result(
    result_path: Union[str, Path],
    storage: Optional[ResultStorage] = None
) -> Dict[str, Any]:
    """
    Load result from a file.
    
    Args:
        result_path: Path to result file
        storage: Optional result storage implementation
        
    Returns:
        Dictionary containing result data
        
    Raises:
        FileNotFoundError: If result file doesn't exist
        ValueError: If result data is invalid
    """
    try:
        # Use provided storage or default
        storage = storage or FileSystemStorage()
        
        # Load result
        result = storage.load_result(result_path)
        
        # Validate structure
        if not isinstance(result, dict):
            raise ValueError("Invalid result data structure")
            
        required_fields = ['meta', 'test_parameters', 'results_by_image']
        missing_fields = [field for field in required_fields if field not in result]
        if missing_fields:
            raise ValueError(f"Missing required fields in result: {missing_fields}")
            
        return result
        
    except Exception as e:
        logger.error(f"Error loading result from {result_path}: {str(e)}")
        raise

def concatenate_results(
    logs_dir: Union[str, Path],
    model_name: str,
    quant_level: Optional[int] = None,
    storage: Optional[ResultStorage] = None
) -> Dict[str, Any]:
    """
    Concatenate results from multiple files into a single organized structure.
    If quant_level is provided, only concatenate results for that quantization level.
    
    Args:
        logs_dir: Directory containing result files
        model_name: Name of the model to concatenate results for
        quant_level: Optional quantization level to filter by
        storage: Optional result storage implementation
        
    Returns:
        Dictionary containing concatenated results for the specified quantization level
    """
    try:
        # Use provided storage or default
        storage = storage or FileSystemStorage()
        
        # Convert to Path object
        logs_dir = Path(logs_dir)
        
        # Find all result files for this model
        if quant_level is not None:
            result_files = list(logs_dir.glob(f"{model_name}_{quant_level}bit_*.json"))
        else:
            result_files = list(logs_dir.glob(f"{model_name}_*bit_*.json"))
        
        if not result_files:
            logger.warning(f"No result files found for {model_name} with quantization {quant_level or 'any'}")
            return None
            
        # Initialize result structure
        concatenated_results = {
            "meta": {
                "model_name": model_name,
                "quantization_level": quant_level,
                "timestamp": datetime.now().isoformat(),
                "total_files": len(result_files)
            },
            "results": []
        }
        
        # Process each result file
        for result_file in result_files:
            try:
                # Load result
                result = storage.load_result(result_file)
                
                # Add result to list
                concatenated_results["results"].append(result)
                
            except Exception as e:
                logger.error(f"Error processing result file {result_file}: {str(e)}")
                continue
        
        return concatenated_results
        
    except Exception as e:
        logger.error(f"Error concatenating results: {str(e)}")
        raise

def log_quantization_results(
    result_path: str,
    results: List[Dict],
    model_name: str,
    quant_level: int,
    environment: Optional[str] = None,
    storage: Optional[ResultStorage] = None
) -> None:
    """Log results for a quantization level, including results from all prompt types.
    
    Args:
        result_path: Base path for saving results
        results: List of result dictionaries
        model_name: Name of the model being tested
        quant_level: Quantization level being tested
        environment: Optional environment identifier
        storage: Optional storage implementation
        
    Raises:
        ValueError: If any result has an invalid structure
    """
    try:
        # Create base result structure
        result = {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "environment": environment or "local",
                "model": model_name,
                "quantization_level": quant_level
            },
            "test_parameters": {
                "model_name": model_name,
                "quantization_level": quant_level
            },
            "results_by_prompt": {}
        }
        
        # Group results by prompt type first
        for r in results:
            if r.get("error"):
                continue
                
            # Validate response structure before processing
            try:
                validate_model_response_structure(r)
            except ValueError as e:
                logger.error(f"Invalid result structure: {str(e)}")
                logger.error(f"Result data: {json.dumps(r, indent=2)}")
                raise
                
            prompt_type = r["test_parameters"]["prompt_type"]
            image_id = r["test_parameters"]["image_path"].split("/")[-1]
            
            if prompt_type not in result["results_by_prompt"]:
                result["results_by_prompt"][prompt_type] = {
                    "results_by_image": {}
                }
            
            result["results_by_prompt"][prompt_type]["results_by_image"][image_id] = {
                "image_path": r["test_parameters"]["image_path"],
                "ground_truth": {
                    "work_order_number": r["ground_truth"]["work_order_number"],
                    "total_cost": r["ground_truth"]["total_cost"]
                },
                "model_response": {
                    "work_order_number": {
                        "raw_text": r["model_response"]["output"],
                        "parsed_value": r["model_response"].get("work_order_number", {}).get("parsed_value", ""),
                        "normalized_value": r["model_response"].get("work_order_number", {}).get("normalized_value", "")
                    },
                    "total_cost": {
                        "raw_text": r["model_response"]["output"],
                        "parsed_value": r["model_response"].get("total_cost", {}).get("parsed_value", ""),
                        "normalized_value": r["model_response"].get("total_cost", {}).get("normalized_value", "")
                    },
                    "processing_time": r["model_response"]["processing_time"]
                },
                "evaluation": {
                    "work_order_number": r["evaluation"]["work_order_number"],
                    "total_cost": r["evaluation"]["total_cost"]
                }
            }
        
        # Save using specified storage or default to file system
        storage = storage or FileSystemStorage()
        storage.save_result(result_path, result)
        
    except Exception as e:
        logger.error(f"Error logging quantization results: {str(e)}")
        raise 