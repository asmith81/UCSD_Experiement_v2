# Project Rules: LMM Invoice Data Extraction Comparison (Rapid Implementation Version)

## Revised Architectural Principles for Rapid Implementation

### 1. Functional Programming Approach
- Use stateless functions that transform data rather than complex classes
- Minimize state management to reduce complexity
- Use simple data structures (dictionaries, lists) for data exchange
- Focus on pipeline-oriented processing for clarity
- Clear input/output contracts for each function

### 2. Code Constraints (Simplified)
- Favor readability over complex abstractions
- Maximum 100 characters per line
- Keep functions small and focused (ideally < 25 lines)
- Maximum 3 levels of nesting in any function
- Prioritize simplicity over extensibility for this phase

### 3. Simplified Structure
- Organize code by function rather than complex inheritance hierarchies
- Use utility modules with logical grouping of related functions
- Create thin notebooks that import functionality from utility modules
- Keep core logic in utility modules, not notebooks
- Minimize interdependencies between components

### 4. Configuration Management
- Use simple configuration files (YAML or JSON)
- Move environment setup, dependencies, and paths to source modules
- Store prompt templates in separate files for reference
- Keep model-specific loading and inference functions in source code
- Allow notebooks to be thin execution environments

### 5. Test Matrix Management
- Use CSV file to define test combinations
- Load test parameters from CSV
- Track execution status for resumability
- Log results to CSV as they occur (crash resilience)

### 6. Error Handling (Simplified)
- Focus on core error handling for critical operations
- Use try/except blocks with meaningful error messages
- Log errors immediately when they occur
- Design for graceful crash recovery with result tracking
- Skip complex error hierarchies in favor of basic error types

### 7. Documentation
- Document all functions with clear docstrings
- Include parameters and return values in docstrings
- Add inline comments for complex operations
- Use markdown cells in notebooks to explain process
- Maintain clear README files in key directories

## File Organization

### Source Files
- Create a `src/` directory with utility modules
- Group functions by purpose (environment, data, models, evaluation)
- Keep related functionality in the same module
- Use clear, descriptive file and function names

### Notebook Organization
- Create one notebook per model for parallel execution
- Import utility functions at the top of each notebook
- Clearly document notebook purpose and parameters
- Use markdown cells to explain each execution step
- Minimize code duplication across notebooks

## Naming Conventions

### Function Names
- Use descriptive snake_case for all function names
- Make function purpose clear from name
- Use verb phrases for action functions
- Use noun phrases for data access functions

### Variable Names
- Use descriptive snake_case for all variables
- Name variables according to their content/purpose
- Use consistent naming patterns across modules
- Prefer explicit names over abbreviations

### File Names
- Use descriptive snake_case for all file names
- Group related functionality in single modules
- Use consistent naming patterns for similar files

## Test Matrix Management

### Configuration Storage
- Store test parameters in CSV format:
  ```
  model_name,prompt_type,quant_level
  pixtral,direct,4
  pixtral,direct,8
  pixtral,detailed,4
  ...
  ```

### Result Logging
- Log results immediately after each test:
  ```
  model_name,prompt_type,quant_level,image_id,exact_match,work_order_match,total_cost_match,timestamp
  pixtral,direct,4,1001,True,True,True,2023-03-01T12:30:45
  ```

### Execution Tracking
- Check existing results before running tests
- Skip already-executed test combinations
- Design for parallel notebook execution
- Handle interrupted execution gracefully

# Project Rules: LMM Invoice Data Extraction Comparison (Rapid Implementation)

## Revised Architectural Principles for Rapid Implementation

### 1. Functional Programming Approach
- Use stateless functions that transform data rather than complex classes
- Minimize state management to reduce complexity
- Use simple data structures (dictionaries, lists) for data exchange
- Focus on pipeline-oriented processing for clarity
- Clear input/output contracts for each function

### 2. Field-Specific Testing Approach
- Test each field type separately with field-specific prompts
- Maintain loose coupling between field types and prompt types
- Field-specific prompts should focus on extracting a single field value
- Field-specific evaluation metrics appropriate to data type
- Field-specific error categorization for detailed analysis

### 3. Results Storage Strategy
- Group results by test parameters to avoid duplication
- Store results for multiple images within each test configuration
- Include complete context for each test:
  - Test parameters (model, field type, prompt, quantization)
  - Metadata (experiment ID, timestamp, environment)
  - Results by image, including:
    - Ground truth (raw and normalized)
    - Model response (raw text, parsed value, normalized value)
    - Evaluation metrics (appropriate to field type)
- Standard result dictionary structure:

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
        "1018": { ... },
        // More images...
    }
}
```

### 4. Field Extraction Rules
- Each field type has specific extraction requirements:
  - **Work Order Number**:
    - Preserve exact string format
    - Handle alphanumeric values
    - Use regex pattern extraction when possible
    - Store raw value and normalized value (trimmed)
    - Evaluate using exact string match
  
  - **Total Cost**:
    - Extract exact value as displayed in model output
    - Store both raw string and normalized float value
    - Handle various formats ($, commas, spaces)
    - Use currency-specific extraction patterns
    - Evaluate using normalized value matching
  
  - **Future Fields**:
    - Date: Format-specific extraction with date parsing
    - Customer: Name extraction with normalization
    - Line Items: Structured extraction with tabular parsing

### 5. Field-Specific Prompt Design
- Work Order prompts should:
  - Emphasize exact preservation of format
  - Specify location hints when useful
  - Request alphanumeric precision
  
- Total Cost prompts should:
  - Specify currency extraction
  - Emphasize numerical precision
  - Allow for formatted or unformatted response

### 6. Code Constraints
- Favor readability over complex abstractions
- Maximum 100 characters per line
- Keep functions small and focused (ideally < 25 lines)
- Maximum 3 levels of nesting in any function
- Prioritize simplicity over extensibility for this phase

### 7. Simplified Structure
- Organize code by function rather than complex inheritance hierarchies
- Use utility modules with logical grouping of related functions
- Create thin notebooks that import functionality from utility modules
- Keep core logic in utility modules, not notebooks
- Minimize interdependencies between components

### 8. Test Matrix Management
- Use CSV file to define test combinations of:
  - Model name
  - Field type
  - Prompt type
  - Quantization level
- Load test parameters from CSV
- Track execution status for resumability
- Log results to JSON as they occur (crash resilience)

### 9. Error Handling
- Focus on core error handling for critical operations
- Use try/except blocks with meaningful error messages
- Log errors immediately when they occur
- Design for graceful crash recovery with result tracking
- Skip complex error hierarchies in favor of basic error types

### 10. Documentation
- Document all functions with clear docstrings
- Include parameters and return values in docstrings
- Add inline comments for complex operations
- Use markdown cells in notebooks to explain process
- Maintain clear README files in key directories

## Field-Specific Data Handling Rules

### Work Order Number Handling
- Store as string to preserve format
- Normalize by trimming whitespace only
- Compare using exact string matching
- Error categories:
  - missing_character: Length shorter than expected
  - extra_character: Length longer than expected
  - transposition: Same characters in different order
  - wrong_character: Different characters at same position

### Total Cost Handling
- Store both raw string and normalized float
- Normalize by:
  - Removing currency symbols ($)
  - Removing commas (,)
  - Trimming whitespace
  - Converting to float
- Compare using normalized value matching (float equality)
- Error categories:
  - currency_error: Difference in currency symbol usage
  - decimal_error: Difference in decimal point usage
  - digit_error: Difference in numerical digits
  - formatting_error: Difference in thousands separators, spacing, etc.

## Implementation Priorities

1. **Field-Specific Testing First**: Implement the field-specific extraction and evaluation
2. **End-to-End Flow**: Create working pipeline before optimizing components
3. **Documentation As You Go**: Document concurrently with implementation
4. **Error Handling**: Focus on critical error paths first
5. **Parallel Execution**: Design for notebook parallelization from the start

## Testing Strategy

1. **Minimal Test Dataset**: Start with 10-20 representative images
2. **Early Validation**: Validate each component as it's implemented
3. **Incremental Testing**: Add features and tests incrementally
4. **Result Validation**: Verify results against ground truth manually for subset

## Quality Standards

Despite the rapid implementation focus, maintain these quality standards:
1. All functions should be documented
2. Critical paths should have error handling
3. Results should be logged consistently
4. Code should be readable and maintainable
5. Processing pipeline should be resumable after crashes

## Model Processing Rules

1. **Input Format Standardization**
   - All models must use the standardized chat-style input format
   - Input processing must use AutoProcessor for consistency
   - Support both text and image inputs in unified format

2. **Model Loading**
   - Use AutoModelForVision2Seq for model loading
   - Support both local and HuggingFace model loading
   - Implement standardized model configuration

3. **Processing Pipeline**
   - Follow the standardized processing pipeline
   - Use consistent tensor handling and processing
   - Implement unified error handling and logging

4. **Result Handling**
   - Use standardized output format for all models
   - Implement consistent result validation
   - Follow unified logging and reporting guidelines