# LMM Invoice Data Extraction Comparison

## Project Overview
This project evaluates multiple open-source Large Multimodal Models (LMMs) on their ability to extract specific structured data from handwritten invoice images. The project systematically compares different models, prompts, and quantization levels to identify the optimal solution for extracting work order numbers and total costs from invoices.

### Key Features
- Evaluation of 3 open-source LMMs: Pixtral-12B, Llama-3.2-11B-Vision, and Doctr
- Field-specific extraction for work order numbers and total costs
- Multiple quantization levels (4, 8, 16, 32 bit) per model
- Systematic comparison of prompt strategies
- Comprehensive evaluation metrics and error analysis
- Efficient test execution with minimized model reloading
- Standardized image preprocessing and output parsing
- Flexible dependency injection for model loading, processing, and storage
- Protocol-based interfaces for extensibility
- Advanced visualization and analysis tools

## Project Structure
```
UCSD_MJM_Experiment/
├── config/                         # Configuration files
│   ├── models/                     # Model-specific configurations
│   ├── prompts/                    # Field-specific prompt templates
│   ├── config.yaml                 # Main configuration file
│   └── test_matrix.csv            # Test configuration matrix
├── src/                            # Source code modules
│   ├── environment.py              # Environment setup and paths
│   ├── config.py                   # Configuration management
│   ├── execution.py                # Test execution framework
│   ├── data_utils.py               # Data loading and processing
│   ├── results_logging.py          # Result logging and tracking
│   └── models/                     # Model implementations
│       ├── __init__.py            # Model registry
│       └── common.py              # Shared model utilities
├── notebooks/                      # Jupyter notebooks
│   ├── model_notebook_template.py  # Template for model evaluation
│   ├── pixtral_evaluation.py      # Pixtral model evaluation
│   ├── llama_vision_evaluation.py # Llama Vision model evaluation
│   ├── doctr_evaluation.py        # Doctr model evaluation
│   └── results_analysis.py        # Results analysis and comparison
├── docs/                           # Documentation
│   ├── interface_control_document.md
│   ├── project_dir.md
│   └── project_todo.md
├── tests/                          # Test files
├── requirements.txt                # Python dependencies
├── README.md                       # Project overview
└── .gitignore                     # Git ignore rules
```

## Setup Instructions

1. **Environment Setup**
   ```powershell
   # Create and activate virtual environment
   python -m venv venv
   .\venv\Scripts\activate

   # Install dependencies (two options available)

   # Option 1: Using environment.py script
   python src/environment.py

   # Option 2: Manual installation
   pip install -r requirements.txt
   ```

   The `environment.py` script will:
   - Set up project directory structure
   - Install PyTorch with CUDA support
   - Install remaining dependencies
   - Verify CUDA availability
   - Create necessary project directories

2. **Data Preparation**
   - Place invoice images in `data/images/`
   - Ensure ground truth data is in `data/ground_truth.csv`

3. **Configuration**
   - Review and adjust model configurations in `config/models/`
   - Customize prompts in `config/prompts/`
   - Configuration is managed through `src/config.py`

## Usage

1. **Model Evaluation**
   Each model has its own evaluation notebook that follows a consistent structure:
   ```python
   # In notebook
   import execution
   from src.environment import setup_environment
   from src.config import load_yaml_config
   from src.models.<model_name> import load_model, process_image
   from src.results_logging import track_execution, log_result

   # Setup environment and logging
   env = setup_environment()
   paths = env['paths']
   config = load_yaml_config(f'config/{model_name}.yaml')

   # Run evaluation with dependency injection
   results = execution.run_test_suite(
       model_name=MODEL_NAME,
       test_matrix_path=TEST_MATRIX_PATH,
       model_loader=load_model,  # Custom model loader
       processor=process_image,  # Custom image processor
       prompt_loader=load_prompt,  # Custom prompt loader
       result_validator=validate_results  # Custom result validator
   )
   ```

2. **Results Analysis**
   The `results_analysis.py` notebook provides comprehensive analysis tools:

   ```python
   # Load and analyze results
   results = load_all_results(paths['results'])
   
   # Generate visualizations
   plot_model_comparison(metrics)  # Compare models by field type
   plot_prompt_comparison(prompt_metrics)  # Compare prompt strategies
   
   # Create heatmaps
   create_performance_heatmap(prompt_metrics, field_type)  # Model vs prompt performance
   plot_quantization_heatmap(quant_metrics, model, prompt)  # Quantization analysis
   
   # Find optimal configurations
   best_config = find_best_configuration(results, field_type)
   best_prompt = find_best_prompt_strategy(prompt_metrics, field_type)
   ```

   Analysis features include:
   - Model comparison by field type
   - Prompt strategy analysis
   - Quantization level comparison
   - Error distribution analysis
   - Performance heatmaps
   - Optimal configuration identification

## Development Guidelines

- Follow functional programming approach with stateless functions
- Keep functions small and focused (< 25 lines)
- Maximum 3 levels of nesting in any function
- Document all functions with clear docstrings
- Use simple configuration files (YAML/JSON)
- Implement core error handling for critical operations
- Use shared model utilities for common functionality
- Implement protocol-based interfaces for extensibility
- Use dependency injection for model loading, processing, and storage
- Follow type hints and protocol definitions

## Documentation

- `docs/interface_control_document.md`: Component interfaces and data structures
- `docs/project_dir.md`: Detailed directory structure
- `docs/project_todo.md`: Task list and progress tracking

## Results

Results are organized by field type and stored in JSON format, including:
- Raw and normalized values
- Processing times
- Evaluation metrics
- Error categorization
- Execution tracking information
- Storage implementation details
- Performance heatmaps and visualizations

### Data Formats
All data exchange in the system uses JSON format for consistency and type safety:

1. **Ground Truth Data**
```json
{
    "image_id": {
        "work_order_number": {
            "raw_value": "string",
            "normalized_value": "string"
        },
        "total_cost": {
            "raw_value": "string",
            "normalized_value": "float"
        }
    }
}
```

2. **Model Responses**
```json
{
    "work_order_number": {
        "raw_text": "string",
        "parsed_value": "string",
        "normalized_value": "string",
        "confidence": "float",
        "processing_time": "float"
    },
    "total_cost": {
        "raw_text": "string",
        "parsed_value": "string",
        "normalized_value": "float",
        "confidence": "float",
        "processing_time": "float"
    }
}
```

3. **Test Matrix**
```json
{
    "test_cases": [
        {
            "model_name": "string",
            "field_type": "string",
            "prompt_type": "string",
            "quant_level": "integer"
        }
    ]
}
```

## Contributing

1. Follow the project structure and coding guidelines
2. Document all changes
3. Update relevant documentation files
4. Test changes thoroughly
5. Implement protocol-based interfaces for new components
6. Use dependency injection for extensibility

## License

[Add appropriate license information]

## Contact

[Add contact information]

## Model Configurations

### Pixtral-12B
- Uses direct prompt format for simplicity
- Optimized for single-image extraction
- No conversation history needed
- Simple prompt structure
- Image preprocessing handled by AutoProcessor
- Returns standardized result structure with:
  - Test parameters (model, quantization, prompt strategy)
  - Model response (output, error, processing time)
  - Evaluation metrics (normalized match, CER, error category)

### Llama-3.2-11B-Vision
- Direct prompt format with system message
- Optimized for invoice extraction
- Supports LoRA fine-tuning
- Image preprocessing requirements:
  - Max size: 1120x1120
  - RGB conversion
  - Normalization handled by MllamaProcessor
  - Aspect ratio maintenance
- Higher token limit (2048) for detailed responses
- No content safety checks (business documents only)
- Returns standardized result structure with:
  - Test parameters (model, quantization, prompt strategy)
  - Model response (output, error, processing time)
  - Evaluation metrics (normalized match, CER, error category)

### Doctr
- Two-stage architecture:
  - Text detection (DBNet)
  - Text recognition (CRNN)
- KIE support for structured extraction
- PyTorch backend with visualization support
- Document handling:
  - PDF support
  - Single and multi-page images
  - Preprocessed images only
- Output structure:
  - JSON format
  - Confidence scores
  - Geometry information
- Lower hardware requirements (8GB GPU)
- Higher batch processing capability
- Returns standardized result structure with:
  - Test parameters (model, quantization, prompt strategy)
  - Model response (output, error, processing time)
  - Evaluation metrics (normalized match, CER, error category)

## Test Matrix and Execution

The project uses a systematic approach to testing different model configurations:

### Test Matrix Structure
```csv
model,quantization,prompt_strategy
pixtral,4,basic_extraction
pixtral,8,basic_extraction
...
```

### Execution Strategy
1. Each notebook focuses on one model
2. Test cases are grouped by quantization to minimize model reloading
3. Prompt templates define the expected fields and JSON structure
4. Results are validated against the prompt template structure
5. Shared utilities handle common preprocessing and parsing
6. Execution is tracked and logged for each test run
7. Custom implementations can be injected for model loading, processing, and storage

### Notebook Structure
Each model evaluation notebook follows this structure:
1. Environment setup and configuration
2. Model-specific configuration loading
3. Test suite execution with dependency injection
4. Results analysis and visualization
5. Performance metrics by quantization level