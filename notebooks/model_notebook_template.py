"""
Model Evaluation Notebook Template

This template provides the structure for model-specific evaluation notebooks.
It follows the project's notebook handling rules and functional programming approach.
"""

# %% [markdown]
# # Model Evaluation Notebook
# 
# This notebook evaluates a specific model's performance across different quantization levels
# and prompt strategies. It uses the execution framework to run the test suite and
# analyze results.

# %% [markdown]
# ## Environment Setup
# 
# First, we need to set up the environment and import required modules.

# %%
import os
import sys
from pathlib import Path

# Determine root directory
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# Import project modules
import execution
from src.environment import setup_environment
from src.config import load_yaml_config
from src.validation import validate_results

# Setup environment
env = setup_environment()
paths = env['paths']

# Load configuration
config = load_yaml_config('config/config.yaml')

# %% [markdown]
# ## Model Configuration
# 
# Set the model name and test matrix path.

# %%
# Set model for this notebook
MODEL_NAME = "pixtral"  # or llama_vision or doctr
TEST_MATRIX_PATH = "config/test_matrix.json"

# %% [markdown]
# ## Run Test Suite
# 
# Execute the test suite for the specified model.

# %%
def main():
    """Run the test suite and handle results."""
    try:
        # Run test suite
        results = execution.run_test_suite(
            model_name=MODEL_NAME,
            test_matrix_path=TEST_MATRIX_PATH
        )
        
        # Log results
        print(f"Completed {len(results)} test cases for {MODEL_NAME}")
        
    except Exception as e:
        print(f"Error running test suite: {str(e)}")
        raise

if __name__ == "__main__":
    main()

# %% [markdown]
# ## Results Analysis
# 
# Analyze and visualize the results.

# %%
# Results analysis will be added here
# This section will be implemented in the model-specific notebooks 