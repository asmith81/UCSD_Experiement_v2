"""
Doctr Model Evaluation Notebook

This notebook evaluates the Doctr model's performance on invoice data extraction.
It follows the project's notebook handling rules and functional programming approach.
"""

# %% [markdown]
# # Doctr Model Evaluation
# 
# This notebook evaluates the Doctr model's performance across different quantization levels
# and prompt strategies for invoice data extraction.

# %% [markdown]
# ## Environment Setup
# 
# First, we need to set up the environment and import required modules.

# %%
import os
import sys
import subprocess
from pathlib import Path
import logging
import json

# Configure logging first
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

# Install dependencies
print("Installing dependencies...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(ROOT_DIR / "requirements.txt")])
    print("Dependencies installed successfully.")
    
    # Install PyTorch dependencies separately
    print("Installing PyTorch dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "torchaudio==2.1.0",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ])
    print("PyTorch dependencies installed successfully.")
except subprocess.CalledProcessError as e:
    logger.error(f"Error installing dependencies: {e}")
    raise

# Import project modules
from src import execution
from src.environment import setup_environment, download_model
from src.config import load_yaml_config
from src.models.doctr import load_model, process_image_wrapper, download_doctr_model
from src.prompts import load_prompt_template
from src.results_logging import track_execution, log_result, ResultStructure, evaluate_model_output
from src.validation import validate_results
from src.data_utils import DataConfig, setup_data_paths

# Setup environment
try:
    env = setup_environment(
        project_root=ROOT_DIR,
        requirements_path=ROOT_DIR / "requirements.txt"
    )
    
    # Validate paths
    required_paths = ['data_dir', 'models_dir', 'logs_dir', 'results_dir', 'prompts_dir']
    missing_paths = [path for path in required_paths if path not in env]
    if missing_paths:
        raise RuntimeError(f"Missing required paths in environment: {missing_paths}")
        
    # Ensure required directories exist
    for path in required_paths:
        env[path].mkdir(parents=True, exist_ok=True)
    
except Exception as e:
    logger.error(f"Error setting up environment: {str(e)}")
    raise

# Load configuration
config_path = ROOT_DIR / "config" / "models" / "doctr.yaml"
if not config_path.exists():
    raise FileNotFoundError(f"Configuration file not found: {config_path}")

try:
    config = load_yaml_config(str(config_path))
    # Validate required configuration sections
    required_sections = ['name', 'loading', 'quantization', 'prompt', 'inference']
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Configuration missing required sections: {missing_sections}")
except Exception as e:
    logger.error(f"Error loading configuration: {str(e)}")
    raise

# Setup data configuration
try:
    data_config = setup_data_paths(
        env_config=env,
        image_extensions=['.jpg', '.jpeg', '.png'],
        max_image_size=1120,
        supported_formats=['RGB', 'L']
    )
    logger.info("Data configuration setup successfully")
except Exception as e:
    logger.error(f"Error setting up data configuration: {str(e)}")
    raise

# %% [markdown]
# ## Model Configuration
# 
# Set the model name and test matrix path.

# %%
# Set model for this notebook
MODEL_NAME = "doctr"
TEST_MATRIX_PATH = str(ROOT_DIR / "config" / "test_matrix.json")
EXECUTION_LOG_PATH = env['logs_dir'] / f"{MODEL_NAME}_execution.log"

# Validate test matrix exists and is valid
try:
    if not Path(TEST_MATRIX_PATH).exists():
        raise FileNotFoundError(f"Test matrix file not found: {TEST_MATRIX_PATH}")
        
    # Load and validate test matrix
    with open(TEST_MATRIX_PATH, 'r') as f:
        test_matrix = json.load(f)
        
    # Validate test matrix structure
    if 'test_cases' not in test_matrix:
        raise ValueError("Test matrix must contain 'test_cases' array")
        
    # Validate required fields
    required_fields = ['model_name', 'field_type', 'prompt_type', 'quant_level', 'image_path']
    for test_case in test_matrix['test_cases']:
        missing_fields = [field for field in required_fields if field not in test_case]
        if missing_fields:
            raise ValueError(f"Test case missing required fields: {missing_fields}")
            
    # Validate quantization values
    valid_quantization = [4, 8, 16, 32]
    invalid_quantization = [case['quant_level'] for case in test_matrix['test_cases'] 
                          if case['quant_level'] not in valid_quantization]
    if invalid_quantization:
        raise ValueError(f"Invalid quantization values found: {invalid_quantization}")
            
except Exception as e:
    logger.error(f"Error validating test matrix: {str(e)}")
    raise

# Load model configuration
try:
    # The config is already loaded and validated with required sections
    # We can use the config directly as it matches our needs
    model_config = {
        'name': config['name'],
        'path': config['repo_id'],
        'quantization_levels': list(config['quantization']['options'].keys())
    }
    
    prompt_config = {
        'format': config['prompt']['format'],
        'image_placeholder': config['prompt']['image_placeholder'],
        'default_field': config['prompt']['default_field']
    }
    
    # Validate model configuration
    required_model_fields = ['name', 'path', 'quantization_levels']
    missing_fields = [field for field in required_model_fields if field not in model_config]
    if missing_fields:
        raise ValueError(f"Model configuration missing required fields: {missing_fields}")
        
except KeyError as e:
    logger.error(f"Missing required configuration section: {e}")
    raise
except Exception as e:
    logger.error(f"Error loading model configuration: {str(e)}")
    raise

print(f"✓ Model configuration loaded successfully for {MODEL_NAME}")

# Model Download
from src.models.doctr import download_doctr_model

# Set up model path
model_path = env['models_dir'] / "doctr"

# Download model if needed
if not model_path.exists():
    logger.info("Model not found locally. Downloading...")
    if download_doctr_model(model_path, config['repo_id']):
        logger.info("Model downloaded successfully")
    else:
        logger.error("Failed to download model")
        raise RuntimeError("Model download failed")
else:
    logger.info("Model already exists locally")

# Model Loader
try:
    # The config is already loaded and validated with required sections
    # We can use the config directly as it matches our needs
    model_loader = lambda name, quant: load_model(
        model_name=name,
        quantization=quant,
        model_path=env['models_dir'] / config['name']
    )
    
    # Validate model loader
    if not callable(model_loader):
        raise ValueError("Model loader must be a callable function")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# %% [markdown]
# ## Run Test Suite
# 
# Execute the test suite for Doctr model.

# %%
def main():
    """Run the test suite and handle results."""
    try:
        # Track execution start
        track_execution(
            EXECUTION_LOG_PATH,
            MODEL_NAME,
            "all",
            0,
            "started"
        )
        
        # Run test suite
        results = execution.run_test_suite(
            model_name=MODEL_NAME,
            test_matrix_path=TEST_MATRIX_PATH,
            model_loader=model_loader,
            processor=lambda model, prompt, test_case: process_image_wrapper(
                model=model,
                prompt_template=prompt,
                image_path=test_case['image_path'],
                field_type=test_case['field_type'],
                config=data_config
            ),
            prompt_loader=lambda strategy: load_prompt_template(
                prompt_strategy=strategy,
                prompts_dir=ROOT_DIR / "config" / "prompts"
            ),
            result_validator=validate_results
        )
        
        # Log results
        logger.info(f"Completed {len(results)} test cases for {MODEL_NAME}")
        
        # Track execution completion
        track_execution(
            EXECUTION_LOG_PATH,
            MODEL_NAME,
            "all",
            0,
            "completed"
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error running test suite: {str(e)}")
        track_execution(
            EXECUTION_LOG_PATH,
            MODEL_NAME,
            "all",
            0,
            "failed",
            str(e)
        )
        raise

if __name__ == "__main__":
    results = main()

# %% [markdown]
# ## Results Analysis
# 
# Analyze and visualize the results for Doctr model.

# %%
def analyze_results(results: list) -> dict:
    """Analyze test results for Doctr model."""
    if not results:
        logger.warning("No results to analyze")
        return {}
        
    # Validate result structure
    required_fields = ['test_parameters', 'evaluation', 'model_response']
    for result in results:
        missing_fields = [field for field in required_fields if field not in result]
        if missing_fields:
            logger.warning(f"Result missing required fields: {missing_fields}")
            continue
            
    # Group results by quantization level
    quant_results = {}
    for result in results:
        try:
            quant_level = result['test_parameters']['quantization']
            if quant_level not in quant_results:
                quant_results[quant_level] = []
            quant_results[quant_level].append(result)
        except KeyError:
            logger.warning("Result missing quantization level")
            continue
    
    # Calculate metrics per quantization level
    metrics = {}
    for quant_level, quant_results in quant_results.items():
        if not quant_results:
            continue
            
        total_tests = len(quant_results)
        correct_work_order = sum(1 for r in quant_results 
                               if r['evaluation']['work_order_number']['normalized_match'])
        correct_total_cost = sum(1 for r in quant_results 
                               if r['evaluation']['total_cost']['normalized_match'])
        
        metrics[quant_level] = {
            'work_order_accuracy': correct_work_order / total_tests if total_tests > 0 else 0,
            'total_cost_accuracy': correct_total_cost / total_tests if total_tests > 0 else 0,
            'avg_processing_time': sum(r['model_response']['processing_time'] 
                                    for r in quant_results) / total_tests if total_tests > 0 else 0
        }
    
    return metrics

# Analyze results
if 'results' in locals():
    metrics = analyze_results(results)
    if metrics:
        print("\nPerformance Metrics by Quantization Level:")
        for quant_level, metric in metrics.items():
            print(f"\n{quant_level}-bit Quantization:")
            print(f"Work Order Accuracy: {metric['work_order_accuracy']:.2%}")
            print(f"Total Cost Accuracy: {metric['total_cost_accuracy']:.2%}")
            print(f"Average Processing Time: {metric['avg_processing_time']:.2f}s")
    else:
        print("No valid results to analyze") 