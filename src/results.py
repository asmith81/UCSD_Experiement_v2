"""
Standardized result processing module.
Implements the Result Storage Protocol from the interface control document.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, TypedDict
from datetime import datetime

logger = logging.getLogger(__name__)

class ResultStorage(Protocol):
    """Protocol for result storage implementations."""
    def save_result(self, result: Dict[str, Any], path: Path) -> None:
        ...
    def load_result(self, path: Path) -> Dict[str, Any]:
        ...

class ResultStructure(TypedDict):
    """Structure for complete result file."""
    test_parameters: Dict[str, Any]
    model_response: Dict[str, Any]
    evaluation: Dict[str, Dict[str, Any]]

class FileSystemStorage:
    """Default file system storage implementation."""
    
    def save_result(self, result: Dict[str, Any], path: Path) -> None:
        """
        Save result to file system.
        
        Args:
            result: Result structure to save
            path: Path to save result
            
        Raises:
            OSError: If file cannot be saved
        """
        try:
            # Create directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add timestamp
            result['timestamp'] = datetime.now().isoformat()
            
            # Save as JSON
            with open(path, 'w') as f:
                json.dump(result, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save result: {str(e)}")
            raise OSError(f"Result save failed: {str(e)}")
            
    def load_result(self, path: Path) -> Dict[str, Any]:
        """
        Load result from file system.
        
        Args:
            path: Path to load result from
            
        Returns:
            Loaded result structure
            
        Raises:
            FileNotFoundError: If result file not found
            ValueError: If result file is invalid
        """
        try:
            if not path.exists():
                raise FileNotFoundError(f"Result file not found: {path}")
                
            with open(path, 'r') as f:
                result = json.load(f)
                
            return result
            
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in result file: {path}")
        except Exception as e:
            logger.error(f"Failed to load result: {str(e)}")
            raise RuntimeError(f"Result load failed: {str(e)}")

def process_results(
    results: List[Dict[str, Any]],
    output_dir: Path,
    storage: Optional[ResultStorage] = None
) -> Dict[str, Any]:
    """
    Process and save test results.
    
    Args:
        results: List of test results
        output_dir: Directory to save processed results
        storage: Optional storage implementation
        
    Returns:
        Summary of processed results
        
    Raises:
        RuntimeError: If processing fails
    """
    try:
        if storage is None:
            storage = FileSystemStorage()
            
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each result
        processed_results = []
        for result in results:
            try:
                # Validate result structure
                if not _validate_result_structure(result):
                    logger.warning(f"Invalid result structure: {result}")
                    continue
                    
                # Save individual result
                result_path = output_dir / f"result_{result['test_parameters']['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                storage.save_result(result, result_path)
                
                processed_results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process result: {str(e)}")
                continue
                
        # Generate summary
        summary = _generate_summary(processed_results)
        
        # Save summary
        summary_path = output_dir / "summary.json"
        storage.save_result(summary, summary_path)
        
        return summary
        
    except Exception as e:
        logger.error(f"Result processing failed: {str(e)}")
        raise RuntimeError(f"Processing failed: {str(e)}")

def _validate_result_structure(result: Dict[str, Any]) -> bool:
    """Validate result structure."""
    required_keys = ['test_parameters', 'model_response', 'evaluation']
    return all(key in result for key in required_keys)

def _generate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary of results."""
    summary = {
        'total_tests': len(results),
        'successful_tests': sum(1 for r in results if r['evaluation'].get('success', False)),
        'failed_tests': sum(1 for r in results if not r['evaluation'].get('success', True)),
        'average_processing_time': sum(r['model_response'].get('processing_time', 0) for r in results) / len(results) if results else 0,
        'field_accuracy': {
            'work_order_number': {
                'total': sum(1 for r in results if r['test_parameters']['field_type'] == 'work_order_number'),
                'success': sum(1 for r in results if r['test_parameters']['field_type'] == 'work_order_number' and r['evaluation'].get('success', False))
            },
            'total_cost': {
                'total': sum(1 for r in results if r['test_parameters']['field_type'] == 'total_cost'),
                'success': sum(1 for r in results if r['test_parameters']['field_type'] == 'total_cost' and r['evaluation'].get('success', False))
            }
        }
    }
    
    # Calculate accuracy percentages
    for field in summary['field_accuracy']:
        total = summary['field_accuracy'][field]['total']
        success = summary['field_accuracy'][field]['success']
        summary['field_accuracy'][field]['accuracy'] = success / total if total > 0 else 0
        
    return summary 