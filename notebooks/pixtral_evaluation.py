"""
Pixtral Model Evaluation Notebook

This notebook evaluates the Pixtral-12B model's performance on invoice data extraction.
It follows the project's notebook handling rules and functional programming approach.
"""

# %% [markdown]
# # Pixtral Model Evaluation
# 
# This notebook evaluates the Pixtral-12B model's performance across different quantization levels
# and prompt strategies for invoice data extraction.

# %% [markdown]
# ## Environment Setup
# 
# First, we need to set up the environment and import required modules.

# %%
import os
import sys
from pathlib import Path
import logging

# Import project modules
from src.environment import setup_environment
from src.models.loader import load_model
from src.execution import run_test_suite
from src.results import process_results

# Setup environment
env = setup_environment()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set model for this notebook
MODEL_NAME = "pixtral"
TEST_MATRIX_PATH = env['paths']['config_dir'] / "test_matrix.json"
RAW_RESULTS_DIR = env['paths']['results_dir'] / "raw_results"
PROCESSED_RESULTS_DIR = env['paths']['results_dir'] / "processed_results"

# %% [markdown]
# ## Phase 1: Raw Test Execution
# 
# Run the tests and log raw outputs without any processing.

# %%
# Run test suite
try:
    results = run_test_suite(
        model_name=MODEL_NAME,
        test_matrix_path=TEST_MATRIX_PATH,
        model_loader=load_model
    )
    print("✓ Raw test execution completed successfully")
except Exception as e:
    logger.error(f"Error in raw test execution: {str(e)}")
    raise

# %% [markdown]
# ## Phase 2: Result Processing
# 
# Process the raw results and generate analysis.

# %%
# Process results
try:
    summary = process_results(
        results=results,
        output_dir=PROCESSED_RESULTS_DIR
    )
    print("✓ Result processing completed successfully")
except Exception as e:
    logger.error(f"Error in result processing: {str(e)}")
    raise

# %% [markdown]
# ## Analysis and Visualization
# 
# Display the processed results.

# %%
# Display metrics
print("\nPerformance Metrics:")
print(f"Total Tests: {summary['total_tests']}")
print(f"Successful Tests: {summary['successful_tests']}")
print(f"Failed Tests: {summary['failed_tests']}")
print(f"Average Processing Time: {summary['average_processing_time']:.2f}s")

print("\nField Accuracy:")
for field, metrics in summary['field_accuracy'].items():
    print(f"{field}: {metrics['accuracy']:.2%} ({metrics['success']}/{metrics['total']})") 