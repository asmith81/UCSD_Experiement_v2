"""
Concatenate Results Script

This script concatenates individual test results into separate files for each quantization level.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Determine root directory
try:
    # When running as a script
    ROOT_DIR = Path(__file__).parent.parent
except NameError:
    # When running in a notebook, look for project root markers
    current_dir = Path.cwd()
    while current_dir != current_dir.parent:
        if (current_dir / 'src').exists() and (current_dir / 'notebooks').exists():
            ROOT_DIR = current_dir
            break
        current_dir = current_dir.parent
    else:
        raise RuntimeError("Could not find project root directory. Make sure you're running from within the project structure.")

sys.path.append(str(ROOT_DIR))

from src.results_logging import concatenate_results, FileSystemStorage
from src.config import load_yaml_config

def main():
    """Concatenate results for all models and quantization levels."""
    try:
        # Load configuration
        config_path = ROOT_DIR / "config" / "models" / "pixtral.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        config = load_yaml_config(str(config_path))
        
        # Get models to process
        models = ['pixtral']  # Add other models as needed
        
        # Get quantization levels to process
        quant_levels = [32, 16, 8, 4]  # All quantization levels used in testing
        
        # Process each model
        for model_name in models:
            logger.info(f"Processing results for {model_name}")
            
            # Get logs directory
            logs_dir = ROOT_DIR / "logs"
            if not logs_dir.exists():
                raise FileNotFoundError(f"Logs directory not found: {logs_dir}")
            
            # Process each quantization level
            for quant_level in quant_levels:
                logger.info(f"Processing {quant_level}-bit results")
                
                # Concatenate results for this quantization level
                concatenated_results = concatenate_results(
                    logs_dir=logs_dir,
                    model_name=model_name,
                    quant_level=quant_level
                )
                
                if concatenated_results is None:
                    logger.warning(f"No results found for {model_name} at {quant_level}-bit")
                    continue
                
                # Save concatenated results
                output_path = logs_dir / f"{model_name}_{quant_level}bit_results.json"
                storage = FileSystemStorage()
                storage.save_result(output_path, concatenated_results)
                
                logger.info(f"✓ {quant_level}-bit results saved to: {output_path}")
                
                # Print summary
                print(f"\nSummary for {model_name} at {quant_level}-bit:")
                print(f"Total Tests: {concatenated_results['meta']['total_files']}")
                
                # Calculate success/failure rates
                successful_tests = sum(1 for r in concatenated_results['results'] if 'error' not in r)
                failed_tests = sum(1 for r in concatenated_results['results'] if 'error' in r)
                
                print(f"Successful Tests: {successful_tests}")
                print(f"Failed Tests: {failed_tests}")
                
                if concatenated_results['meta']['total_files'] > 0:
                    success_rate = (successful_tests / concatenated_results['meta']['total_files']) * 100
                    print(f"Success Rate: {success_rate:.1f}%")
            
    except Exception as e:
        logger.error(f"Error concatenating results: {str(e)}")
        raise

if __name__ == "__main__":
    main() 