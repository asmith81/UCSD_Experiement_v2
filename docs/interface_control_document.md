# Interface Control Document: LMM Invoice Data Extraction Comparison

## Overview
This document defines the interfaces and data structures used in the LMM Invoice Data Extraction Comparison project. It provides a comprehensive guide to the component interactions and data flow within the system.

## Core Protocols

### Model Loading Protocol
```python
class ModelLoader(Protocol):
    """Protocol for model loading functions."""
    def __call__(self, model_name: str, config: Dict[str, Any]) -> Any:
        ...
```

### Model Processing Protocol
```python
class ModelProcessor(Protocol):
    """Protocol for model processing functions."""
    def __call__(
        self, 
        model: Any, 
        image: Union[ImageData, Image.Image], 
        prompt: str
    ) -> ModelResponse:
        """Process an image with a model using chat-style input format.
        
        Args:
            model: Loaded model instance
            image: Input image data
            prompt: Text prompt for the model
            
        Returns:
            ModelResponse containing the processed results
            
        Note:
            All models use a standardized chat-style input format:
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "content": prompt},
                        {"type": "image"}
                    ]
                }
            ]
        """
```

### Result Storage Protocol
```python
class ResultStorage(Protocol):
    """Protocol for result storage implementations."""
    def save_result(self, result: ResultStructure, path: Path) -> None:
        ...
    def load_result(self, path: Path) -> ResultStructure:
        ...
```

### Image Processing Protocol
```python
class ImageProcessor(Protocol):
    """Protocol for image processing functions."""
    def __call__(self, image: ImageData, config: Dict[str, Any]) -> ImageData:
        ...
```

## Data Structures

### Model Output
```python
class ModelOutput(TypedDict):
    """Structure for model output data."""
    raw_value: str
    normalized_value: str
    confidence: float
    processing_time: float
```

### Model Response
```python
class ModelResponse(TypedDict):
    """Structure for model response data."""
    raw_value: str
    normalized_value: str
    confidence: float
    processing_time: float
```

### Evaluation Result
```python
class EvaluationResult(TypedDict):
    """Structure for evaluation results."""
    normalized_match: bool
    cer: float
    error_category: Optional[str]
```

### Result Entry
```python
class ResultEntry(TypedDict):
    """Structure for a single result entry."""
    model_response: ModelResponse
    evaluation: EvaluationResult
```

### Result Structure
```python
class ResultStructure(TypedDict):
    """Structure for complete result file."""
    test_parameters: Dict[str, Any]  # Contains model, quantization, prompt_strategy
    model_response: ModelResponse
    evaluation: Dict[str, Dict[str, Any]]  # Contains work_order_number and total_cost metrics
```

## Configuration Structures

### Data Configuration
```python
class DataConfig(TypedDict):
    """Configuration for data management."""
    image_extensions: List[str]
    max_image_size: Tuple[int, int]
    supported_formats: List[str]
    image_processor: Optional[ImageProcessor]
```

### Image Data
```python
class ImageData(TypedDict):
    """Structure for image data."""
    path: Path
    data: np.ndarray
    format: str
    size: Tuple[int, int]
```

### Ground Truth Data
```python
class GroundTruthData(TypedDict):
    """Structure for ground truth data."""
    image_id: str
    work_order_number: str
    total_cost: str
```

## Analysis Functions

### Performance Analysis
```python
def analyze_prompt_performance(results: List[ResultStructure]) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Analyze performance by prompt strategy across models and field types."""
    ...

def analyze_quantization_performance(results: List[ResultStructure]) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Analyze performance by quantization level across models and prompt strategies."""
    ...

def aggregate_by_model_field(results: List[ResultStructure]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate results by model and field type."""
    ...

def analyze_error_distribution(results: List[ResultStructure], field_type: str) -> Dict[str, int]:
    """Calculate distribution of error categories for specific field type."""
    ...
```

### Visualization Functions
```python
def plot_model_comparison(metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """Plot comparison of models by field type."""
    ...

def plot_prompt_comparison(prompt_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]) -> None:
    """Plot comparison of prompt strategies across models and field types."""
    ...

def create_performance_heatmap(prompt_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], 
                             field_type: str, metric: str = 'accuracy') -> None:
    """Create a heatmap of model vs prompt performance."""
    ...

def plot_quantization_heatmap(quant_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], 
                            model: str, prompt_strategy: str, metric: str = 'accuracy') -> None:
    """Create a heatmap of quantization performance for a specific model and prompt strategy."""
    ...

def plot_error_distribution(error_dist: Dict[str, int], field_type: str) -> None:
    """Plot distribution of error categories."""
    ...
```

### Configuration Functions
```python
def setup_environment() -> Dict[str, Any]:
    """Set up the project environment and return paths and configuration."""
    ...

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    ...

def setup_data_paths() -> Dict[str, Path]:
    """Set up and validate data directory paths."""
    ...
```

## Core Functions

### Test Execution
```python
def run_test_suite(
    model_name: str, 
    test_matrix_path: str,
    model_loader: Optional[ModelLoader] = None,
    processor: Optional[ModelProcessor] = None,
    prompt_loader: Optional[Callable[[str], str]] = None,
    result_validator: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Run test suite for a model.
    
    Args:
        model_name: Name of the model to test
        test_matrix_path: Path to test matrix CSV
        model_loader: Optional function to load model
        processor: Optional function to process images
        prompt_loader: Optional function to load prompts
        result_validator: Optional function to validate results
        
    Returns:
        List of test results
    """
```

### Result Logging
```python
def log_result(
    result_path: Union[str, Path],
    result: ResultStructure,
    storage: Optional[ResultStorage] = None
) -> None:
    """Log a test result.
    
    Args:
        result_path: Path to save result
        result: Result structure to save
        storage: Optional storage implementation
    """
```

### Data Loading
```python
def load_image(
    image_path: Union[str, Path],
    config: DataConfig,
    processor: Optional[ImageProcessor] = None
) -> Image.Image
```

## Default Implementations

### File System Storage
```python
class FileSystemStorage:
    """Default file system storage implementation."""
    def save_result(self, result: ResultStructure, path: Path) -> None:
        ...
    def load_result(self, path: Path) -> ResultStructure:
        ...
```

### Default Image Processor
```python
class DefaultImageProcessor:
    """Default image processing implementation."""
    def __call__(self, image: ImageData, config: Dict[str, Any]) -> ImageData:
        ...
```

## Error Handling

The system uses a consistent error handling approach:
1. All critical operations are wrapped in try-except blocks
2. Errors are logged with appropriate context
3. Error information is included in result structures
4. Custom error types are used for specific failure modes

## Data Flow

1. **Configuration Loading**
   - Environment setup
   - Model configuration
   - Data paths
   - Processing parameters

2. **Model Execution**
   - Model loading with quantization
   - Image processing
   - Inference execution
   - Result validation

3. **Result Processing**
   - Error categorization
   - Performance metrics
   - Storage operations
   - Execution tracking

## Dependencies

- Python 3.8+
- PyTorch with CUDA support
- PIL for image processing
- pandas for data handling
- typing for type hints
- pathlib for path management

## Version Control

This document should be updated whenever:
1. New protocols are added
2. Existing interfaces are modified
3. New data structures are introduced
4. Default implementations are changed

## 1. Core Function Interfaces

### 1.1 Environment Setup

```python
def setup_environment(cuda_visible_devices=None, seed=42):
    """
    Set up environment for model execution.
    
    Args:
        cuda_visible_devices: str, GPU devices to use (e.g., "0,1")
        seed: int, random seed for reproducibility
        
    Returns:
        dict: Environment configuration
    """
    
def get_model_path(model_name, quant_level):
    """
    Get path to model with specific quantization.
    
    Args:
        model_name: str, name of the model
        quant_level: int, quantization level (4, 8, 16, 32)
        
    Returns:
        Path: Path to the model files
    """
```

### 1.2 Data Management

```python
def load_image(image_id, image_dir):
    """
    Load image by ID from directory.
    
    Args:
        image_id: str, image identifier
        image_dir: Path, directory containing images
        
    Returns:
        PIL.Image: Loaded image
        
    Raises:
        FileNotFoundError: If image file not found
        ValueError: If image loading fails
    """
    
def load_ground_truth(csv_path):
    """
    Load ground truth data from CSV with both raw and normalized values.
    
    Args:
        csv_path: Path, path to ground truth CSV
        
    Returns:
        Dict[str, GroundTruthData]: Dictionary mapping image IDs to ground truth data
        
    Raises:
        FileNotFoundError: If CSV file not found
        ValueError: If CSV parsing fails
    """
    
def get_field_ground_truth(ground_truth, image_id, field_type):
    """
    Get ground truth for a specific field type and image.
    
    Args:
        ground_truth: Dict[str, GroundTruthData], complete ground truth dictionary
        image_id: str, image identifier
        field_type: str, type of field to extract ("work_order_number" or "total_cost")
        
    Returns:
        Dict[str, Union[str, float]]: Ground truth for specific field
        
    Raises:
        KeyError: If image_id or field_type not found
    """
    
def normalize_cost(value):
    """
    Convert cost string to float, handling currency formatting.
    
    Args:
        value: str, cost value with potential formatting
        
    Returns:
        float: Normalized cost value
        
    Raises:
        ValueError: If value cannot be normalized
    """
```

### 1.3 Field-Specific Evaluation

```python
def evaluate_field_extraction(field_type, parsed_value, ground_truth):
    """
    Evaluate field extraction against ground truth.
    
    Args:
        field_type: str, type of field ("work_order_number" or "total_cost")
        parsed_value: str, extracted value from model
        ground_truth: Dict[str, Union[str, float]], ground truth with raw and normalized values
        
    Returns:
        EvaluationResult: Evaluation metrics appropriate for field type
        
    Raises:
        ValueError: If field_type is invalid
    """
    
def calculate_cer(pred, true):
    """
    Calculate Character Error Rate between predicted and true values.
    
    Args:
        pred: str, predicted value
        true: str, true value
        
    Returns:
        float: Character Error Rate (0.0 to 1.0)
    """
    
def categorize_error(pred, true, field_type):
    """
    Categorize error type based on field and difference.
    
    Args:
        pred: str, predicted value
        true: str, true value
        field_type: str, type of field ("work_order_number" or "total_cost")
        
    Returns:
        str: Error category (e.g., "missing_character", "currency_error")
        
    Raises:
        ValueError: If field_type is invalid
    """
```

### 1.4 Result Management

```python
def create_result_structure(model_name, prompt_type, quant_level, environment="RunPod T4 GPU"):
    """
    Create a new result structure for test parameters.
    
    Args:
        model_name: str, name of the model
        prompt_type: str, type of prompt used
        quant_level: int, quantization level
        environment: str, testing environment description
        
    Returns:
        ResultStructure: Result structure with metadata and empty results section
    """
    
def track_execution(
    execution_log_path: Union[str, Path],
    model_name: str,
    prompt_type: str,
    quant_level: int,
    status: str,
    error: Optional[str] = None
) -> None:
    """Track execution status.
    
    Args:
        execution_log_path: Path to execution log
        model_name: Name of the model
        prompt_type: Type of prompt used
        quant_level: Quantization level
        status: Execution status
        error: Optional error message
    """
```

### 1.5 Model Interface

```python
def load_model(model_name, quant_level):
    """
    Load model with specific quantization.
    
    Args:
        model_name: str, name of the model
        quant_level: int, quantization level (4, 8, 16, 32)
        
    Returns:
        object: Loaded model
        
    Raises:
        ValueError: If model_name or quant_level is invalid
        FileNotFoundError: If model files not found
    """
    
def run_field_inference(model, image, prompt, field_type):
    """
    Run inference for a specific field type.
    
    Args:
        model: object, loaded model
        image: PIL.Image, input image
        prompt: str, formatted prompt for specific field type
        field_type: str, type of field to extract
        
    Returns:
        ModelOutput: Model output with raw text, parsed value, and normalized value
        
    Raises:
        ValueError: If field_type is invalid
        RuntimeError: If inference fails
    """
```

### 1.6 Prompt Management

```python
def get_field_prompt_template(prompt_type, field_type):
    """
    Get prompt template for specific field type.
    
    Args:
        prompt_type: str, type of prompt (e.g., "simple", "detailed")
        field_type: str, type of field ("work_order_number" or "total_cost")
        
    Returns:
        str: Field-specific prompt template
    """
    
def format_field_prompt(template, model_name, field_type, **kwargs):
    """
    Format field-specific prompt template.
    
    Args:
        template: str, prompt template
        model_name: str, name of the model
        field_type: str, type of field to extract
        **kwargs: Additional keyword arguments for formatting
        
    Returns:
        str: Formatted field-specific prompt
    """
```

## 2. Result Dictionary Structure

```python
{
    "meta": {
        "experiment_id": "exp-20250410-123045",
        "timestamp": "2025-04-10T12:30:45",
        "environment": "RunPod T4 GPU"
    },
    "test_parameters": {
        "model_name": "pixtral",
        "field_type": "total_cost",
        "prompt_type": "simple_total",
        "quant_level": 4
    },
    "results_by_image": {
        "1017": {
            "ground_truth": {
                "raw_value": " 950.00 ",
                "normalized_value": 950.00
            },
            "model_response": {
                "raw_text": "Total Cost: $950.00",
                "parsed_value": "$950.00",
                "normalized_value": 950.00,
                "processing_time": 1.23
            },
            "evaluation": {
                "raw_string_match": false,
                "normalized_match": true,
                "cer": 0.25,
                "error_category": "currency_format"
            }
        },
        "1018": {
            "ground_truth": {
                "raw_value": " 550.00 ",
                "normalized_value": 550.00
            },
            "model_response": {
                "raw_text": "Total Cost: $550.00",
                "parsed_value": "$550.00",
                "normalized_value": 550.00,
                "processing_time": 1.19
            },
            "evaluation": {
                "raw_string_match": false,
                "normalized_match": true,
                "cer": 0.25,
                "error_category": "currency_format"
            }
        },
        // More images...
    }
}
```

## 3. File Formats

### 3.1 Ground Truth JSON Format

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

### 3.2 Model Response JSON Format

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

### 3.3 Test Matrix JSON Format

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

## 4. Analysis Functions

```python
def filter_results(results, **kwargs):
    """
    Filter results by any parameter.
    
    Args:
        results: list, result dictionaries
        **kwargs: Key-value pairs for filtering (e.g., model_name="pixtral")
        
    Returns:
        list: Filtered result dictionaries
    """
    
def aggregate_by_model_field(results):
    """
    Aggregate results by model and field type.
    
    Args:
        results: list, result dictionaries
        
    Returns:
        dict: Metrics by model and field type
             {
               "pixtral": {
                 "work_order_number": {"total": 100, "matches": 80, "avg_cer": 0.1},
                 "total_cost": {"total": 100, "matches": 75, "avg_cer": 0.15}
               },
               "llama": {
                 "work_order_number": {"total": 100, "matches": 85, "avg_cer": 0.05},
                 "total_cost": {"total": 100, "matches": 70, "avg_cer": 0.2}
               }
             }
    """
    
def aggregate_by_prompt_field(results):
    """
    Aggregate results by prompt type and field type.
    
    Args:
        results: list, result dictionaries
        
    Returns:
        dict: Metrics by prompt type and field type
    """
    
def aggregate_by_quant_field(results):
    """
    Aggregate results by quantization level and field type.
    
    Args:
        results: list, result dictionaries
        
    Returns:
        dict: Metrics by quantization level and field type
    """
    
def calculate_error_distribution(results, field_type):
    """
    Calculate distribution of error categories for specific field type.
    
    Args:
        results: list, result dictionaries
        field_type: str, field type to analyze
        
    Returns:
        dict: Error category distribution
             {
               "currency_error": 15,
               "digit_error": 10,
               "decimal_error": 5,
               ...
             }
    """
```

## 5. Notebook Flow

Each model-specific notebook follows this field-specific flow:

```python
# 1. Set up environment
env_config = setup_environment()

# 2. Load test configurations
test_matrix = load_test_matrix("config/test_matrix.csv")
model_tests = filter_test_matrix(test_matrix, model_name="pixtral")
all_ground_truth = load_ground_truth("data/ground_truth.csv")

# 3. For each test configuration
for test_params in model_tests:
    # Create result dictionary for this configuration
    result_dict = create_result_dict(test_params)
    
    # Load model with specified quantization
    model = load_model(test_params["model_name"], test_params["quant_level"])
    
    # Get field-specific prompt template
    field_type = test_params["field_type"]
    prompt_type = test_params["prompt_type"]
    template = get_field_prompt_template(prompt_type, field_type)
    
    # For each test image
    for image_id in get_test_image_ids():
        # Get ground truth for this field and image
        gt = get_field_ground_truth(all_ground_truth, image_id, field_type)
        
        # Load image
        image = load_image(image_id, "data/images")
        
        # Format field-specific prompt
        prompt = format_field_prompt(template, test_params["model_name"], field_type)
        
        # Run inference for specific field
        raw_text, parsed_value, normalized_value, processing_time = (
            run_field_inference(model, image, prompt, field_type)
        )
        
        # Evaluate extraction for this field
        evaluation = evaluate_field_extraction(field_type, parsed_value, gt)
        
        # Add result for this image
        add_image_result(
            result_dict, image_id, gt, 
            raw_text, parsed_value, normalized_value, 
            processing_time, evaluation
        )
    
    # Save results for this configuration
    save_result(result_dict, "results/")
    
    # Unload model to free memory
    del model
```

The analysis notebook follows this field-specific flow:

```python
# 1. Load all results
results = load_results("results/")

# 2. Calculate field-specific metrics
model_field_metrics = aggregate_by_model_field(results)
prompt_field_metrics = aggregate_by_prompt_field(results)
quant_field_metrics = aggregate_by_quant_field(results)

# 3. Analyze error distributions by field type
wo_errors = calculate_error_distribution(results, "work_order_number")
cost_errors = calculate_error_distribution(results, "total_cost")

# 4. Generate field-specific visualizations
plot_model_field_comparison(model_field_metrics)
plot_prompt_field_comparison(prompt_field_metrics)
plot_quant_field_comparison(quant_field_metrics)
plot_error_distributions(wo_errors, cost_errors)

# 5. Identify best configurations for each field type
best_wo_config = find_best_configuration(results, "work_order_number")
best_cost_config = find_best_configuration(results, "total_cost")
```

## 6. Component Integration Flow

1. Config Layer provides:
   - Field-specific test matrix configurations
   - Field-specific prompt templates

2. Data Layer provides:
   - Image loading
   - Ground truth data with raw and normalized values
   - Field-specific ground truth access

3. Model Layer handles:
   - Model loading with quantization
   - Field-specific inference execution
   - Field-specific output parsing

4. Result Layer manages:
   - Creating result dictionaries organized by test parameters
   - Storing results by image
   - Field-appropriate evaluation metrics
   - Saving complete results

5. Analysis Layer provides:
   - Field-specific result aggregation
   - Field-specific metric calculation
   - Field-specific visualization generation
   - Best configuration identification for each field type

This functional approach with field-specific processing and dictionary data structures supports efficient testing and analysis while maintaining simplicity for rapid implementation.

## Configuration Management Interface

### src/config.py

#### load_yaml_config
```python
def load_yaml_config(config_path: str) -> Dict[str, Any]
```
Loads and parses YAML configuration file.

**Parameters:**
- `config_path`: Path to YAML configuration file

**Returns:**
- Dictionary containing configuration parameters

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If YAML parsing fails

#### parse_test_matrix
```python
def parse_test_matrix(csv_path: str) -> List[Dict[str, Any]]
```
Parses test matrix CSV into list of test cases.

**Parameters:**
- `csv_path`: Path to test matrix CSV file

**Returns:**
- List of dictionaries containing test case parameters

**Raises:**
- `FileNotFoundError`: If CSV file doesn't exist
- `pd.errors.EmptyDataError`: If CSV is empty

#### setup_logging_config
```python
def setup_logging_config(config: Dict[str, Any], log_dir: Path) -> Dict[str, Any]
```
Sets up logging configuration from main config.

**Parameters:**
- `config`: Main configuration dictionary
- `log_dir`: Path to logging directory (provided by environment setup)

**Returns:**
- Dictionary containing logging configuration

**Raises:**
- `KeyError`: If required logging config keys are missing

#### validate_config
```python
def validate_config(config: Dict[str, Any]) -> bool
```
Validates configuration structure and required fields.

**Parameters:**
- `config`: Configuration dictionary to validate

**Returns:**
- True if configuration is valid

**Raises:**
- `ValueError`: If configuration is invalid
```

## Execution Framework

### Test Execution Interface
```python
def run_test_suite(model_name: str, test_matrix_path: str) -> List[Dict[str, Any]]:
    """
    Run the test suite for a specific model.
    
    Args:
        model_name: Name of the model to test
        test_matrix_path: Path to the test matrix CSV file
        
    Returns:
        List of test results with metadata
    """
```

### Test Matrix Structure
```csv
model,quantization,prompt_strategy
pixtral,4,basic_extraction
pixtral,8,basic_extraction
...
```

## Prompt Template Structure

### YAML Format
```yaml
config_info:
  name: basic_extraction
  description: Basic prompts for extracting work order number and total cost
  version: 1.0
  last_updated: "2024-04-11"

prompts:
  - name: basic_extraction
    text: |
      Please extract the following information from this invoice:
      1. Work Order Number
      2. Total Cost
      
      Return the information in JSON format with these exact keys:
      {
        "work_order_number": "extracted value",
        "total_cost": "extracted value"
      }
    category: basic
    field_to_extract: [work_order, cost]
    format_instructions: 
      output_format: "JSON"
      required_fields: ["work_order_number", "total_cost"]
```

### Prompt Template Fields
- `config_info`: Metadata about the configuration
- `prompts`: List of prompt definitions
  - `name`: Unique identifier for the prompt
  - `text`: The actual prompt text
  - `category`: Classification of the prompt
  - `field_to_extract`: Fields to be extracted
  - `format_instructions`: Output format requirements

## Model Preprocessing

### Image Preprocessing
```python
def preprocess_image(
    image_path: Path,
    config: DataConfig
) -> Image.Image:
    """Preprocess image for model input.
    
    Args:
        image_path: Path to the image file
        config: Data configuration with preprocessing settings
        
    Returns:
        Preprocessed PIL Image
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image processing fails
    """
```

The `preprocess_image` function handles:
- Image loading and RGB conversion
- Size normalization while maintaining aspect ratio
- No pixel value normalization (handled by model processors)

### Model Output Parsing
```python
def parse_model_output(
    output: Any,
    field_type: str,
    confidence_threshold: float = 0.5
) -> ModelOutput:
    """Parse model output into standardized format.
    
    Args:
        output: Raw model output
        field_type: Type of field being extracted
        confidence_threshold: Minimum confidence for accepting prediction
        
    Returns:
        ModelOutput dictionary with parsed values
        
    Raises:
        ValueError: If parsing fails
    """
```

The `parse_model_output` function handles:
- Field-specific parsing for work_order_number and total_cost
- Normalization of values to consistent formats
- Error handling for invalid formats

### Model Output Validation
```python
def validate_model_output(
    parsed_output: ModelOutput,
    field_type: str
) -> bool:
    """Validate parsed model output.
    
    Args:
        parsed_output: Parsed output dictionary
        field_type: Type of field being extracted
        
    Returns:
        Whether the output is valid
        
    Raises:
        ValueError: If validation fails
    """
```

The `validate_model_output` function handles:
- Required field validation
- Field-specific value validation
- Type checking for normalized values

### Image Processing
```python
def process_image(
    model: Any,
    image: Union[ImageData, Image.Image],
    prompt: str
) -> ModelResponse:
    """Process an image with a model.
    
    Args:
        model: Loaded model instance
        image: Image to process (either ImageData or PIL Image)
        prompt: Prompt template to use
        
    Returns:
        ModelResponse containing extraction results
    """
```

### Chat-Style Input Format
```python
class ChatInput(TypedDict):
    """Structure for chat-style model inputs."""
    role: str  # "user" or "assistant"
    content: List[Dict[str, str]]  # List of content items with type and content

class ContentItem(TypedDict):
    """Structure for individual content items in chat input."""
    type: str  # "text" or "image"
    content: str  # For text, the actual content; for image, placeholder
```

### Model Input Processing
```python
class ModelInputProcessor(Protocol):
    """Protocol for processing model inputs."""
    def apply_chat_template(
        self,
        chat: List[ChatInput],
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        return_dict: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Process chat-style inputs into model-ready tensors.
        
        Args:
            chat: List of chat messages in standardized format
            add_generation_prompt: Whether to add generation prompt
            tokenize: Whether to tokenize the input
            return_dict: Whether to return as dictionary
            return_tensors: Format of returned tensors ("pt" for PyTorch)
            
        Returns:
            Dictionary of processed tensors ready for model input
        """
```