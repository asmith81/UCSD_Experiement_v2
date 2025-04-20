"""
Result processing module.

This module handles the post-processing of raw test results, including:
- Parsing raw outputs
- Validating results
- Calculating metrics
- Generating reports
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_raw_results(results_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all raw results from a directory.
    
    Args:
        results_dir: Directory containing raw result files
        
    Returns:
        List of raw results
    """
    results = []
    for result_file in results_dir.glob("*.json"):
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            logger.error(f"Error loading result file {result_file}: {str(e)}")
            continue
    return results

def parse_raw_output(raw_output: str) -> Dict[str, Any]:
    """
    Parse raw model output into structured data.
    
    Args:
        raw_output: Raw model output text
        
    Returns:
        Parsed output dictionary
    """
    try:
        # Find JSON in output
        start_idx = raw_output.find('{')
        end_idx = raw_output.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = raw_output[start_idx:end_idx]
            return json.loads(json_str)
        else:
            return {"raw_text": raw_output}
            
    except Exception as e:
        logger.error(f"Error parsing raw output: {str(e)}")
        return {"raw_text": raw_output, "parse_error": str(e)}

def process_results(
    results_dir: Path,
    output_dir: Path,
    ground_truth: Optional[Dict[str, Any]] = None
) -> None:
    """
    Process raw results and generate analysis.
    
    Args:
        results_dir: Directory containing raw results
        output_dir: Directory to save processed results
        ground_truth: Optional ground truth data for evaluation
    """
    # Load raw results
    raw_results = load_raw_results(results_dir)
    if not raw_results:
        logger.warning("No raw results found to process")
        return
        
    # Process each result
    processed_results = []
    for result in raw_results:
        try:
            # Parse raw output
            parsed_output = parse_raw_output(result["raw_output"])
            
            # Create processed result
            processed_result = {
                "test_id": result["test_id"],
                "timestamp": result["timestamp"],
                "test_parameters": result["test_parameters"],
                "parsed_output": parsed_output,
                "processing_time": result["processing_time"]
            }
            
            # Add error if present
            if "error" in result:
                processed_result["error"] = result["error"]
                
            processed_results.append(processed_result)
            
        except Exception as e:
            logger.error(f"Error processing result {result.get('test_id', 'unknown')}: {str(e)}")
            continue
            
    # Save processed results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"processed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(processed_results, f, indent=2)
        
    logger.info(f"Processed results saved to {output_file}")
    
    # Generate analysis if ground truth is provided
    if ground_truth:
        generate_analysis(processed_results, ground_truth, output_dir)

def generate_analysis(
    processed_results: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    Generate analysis and metrics from processed results.
    
    Args:
        processed_results: List of processed results
        ground_truth: Ground truth data
        output_dir: Directory to save analysis
    """
    # Initialize metrics
    metrics = {
        "total_tests": len(processed_results),
        "successful_tests": 0,
        "failed_tests": 0,
        "average_processing_time": 0.0,
        "field_accuracy": {
            "work_order_number": {"correct": 0, "total": 0},
            "total_cost": {"correct": 0, "total": 0}
        }
    }
    
    # Calculate metrics
    total_time = 0.0
    for result in processed_results:
        if "error" not in result:
            metrics["successful_tests"] += 1
            total_time += result["processing_time"]
            
            # Check field accuracy
            parsed = result["parsed_output"]
            if "work_order_number" in parsed:
                metrics["field_accuracy"]["work_order_number"]["total"] += 1
                if parsed["work_order_number"] == ground_truth["work_order_number"]:
                    metrics["field_accuracy"]["work_order_number"]["correct"] += 1
                    
            if "total_cost" in parsed:
                metrics["field_accuracy"]["total_cost"]["total"] += 1
                if parsed["total_cost"] == ground_truth["total_cost"]:
                    metrics["field_accuracy"]["total_cost"]["correct"] += 1
        else:
            metrics["failed_tests"] += 1
            
    # Calculate averages
    if metrics["successful_tests"] > 0:
        metrics["average_processing_time"] = total_time / metrics["successful_tests"]
        
    # Calculate accuracy percentages
    for field in metrics["field_accuracy"]:
        if metrics["field_accuracy"][field]["total"] > 0:
            metrics["field_accuracy"][field]["accuracy"] = (
                metrics["field_accuracy"][field]["correct"] /
                metrics["field_accuracy"][field]["total"]
            )
            
    # Save metrics
    metrics_file = output_dir / f"analysis_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    logger.info(f"Analysis metrics saved to {metrics_file}")
    
    # Generate CSV report
    generate_csv_report(processed_results, output_dir)

def generate_csv_report(
    processed_results: List[Dict[str, Any]],
    output_dir: Path
) -> None:
    """
    Generate CSV report from processed results.
    
    Args:
        processed_results: List of processed results
        output_dir: Directory to save report
    """
    # Prepare data for DataFrame
    report_data = []
    for result in processed_results:
        row = {
            "test_id": result["test_id"],
            "timestamp": result["timestamp"],
            "model": result["test_parameters"]["model"],
            "quantization": result["test_parameters"]["quantization"],
            "prompt_type": result["test_parameters"]["prompt_type"],
            "processing_time": result["processing_time"],
            "status": "success" if "error" not in result else "failed"
        }
        
        # Add parsed fields if available
        if "error" not in result:
            parsed = result["parsed_output"]
            row.update({
                "work_order_number": parsed.get("work_order_number", ""),
                "total_cost": parsed.get("total_cost", "")
            })
            
        report_data.append(row)
        
    # Create DataFrame and save to CSV
    df = pd.DataFrame(report_data)
    report_file = output_dir / f"results_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(report_file, index=False)
    
    logger.info(f"CSV report saved to {report_file}") 