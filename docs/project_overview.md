# Project Overview: LMM Invoice Data Extraction Comparison

## Project Summary
This project evaluates multiple open-source Large Multimodal Models (LMMs) on their ability to extract specific structured data from handwritten invoice images. The project will identify the optimal model and prompt combination through systematic comparison with a focus on rapid implementation.

## Business Context
A local general contractor needs to automate the extraction of data from invoices that contain typed forms filled in by hand. This project will compare different models to identify the most effective solution for this task.

## Problem Statement
Compare multiple open-source LMMs on their ability to extract specific fields from handwritten invoice images:
1. Work Order Number (alphanumeric string)
2. Total Cost (currency value)

## Modified Project Goals (Rapid Implementation Focus)
1. Compare the performance of 3 open-source LMMs on invoice data extraction
2. Test field-specific prompting strategies (separate prompts optimized for each field type)
3. Evaluate each model at multiple quantization levels (4, 8, 16, 32 bit)
4. Identify the best-performing model/prompt/quantization combination for each field type
5. Complete implementation within 2 days (approximately 8 working hours)

## Simplified Project Scope
- Focus on extracting fields one at a time with field-specific prompts
- Begin with a small test set (10-20 images), expand if time permits
- Evaluate 3 specific LMMs:
  - Pixtral-12B
  - Llama-3.2-11B-Vision
  - Doctr
- Test field-specific prompt strategies
- Test 4 quantization levels per model
- Store raw and normalized values
- For work order numbers: evaluate on exact string match
- For total costs: evaluate on normalized value match
- Keep implementation functional and modular, but prioritize rapid completion

## Test Matrix
The project will systematically test combinations of:
- 3 models × 2 fields × ~3 field-specific prompts × 4 quantization levels = ~72 test combinations

## Data Handling Approach
- Use dictionary structures to store test results efficiently
- Group results by test parameters to avoid duplication
- Store both raw values and normalized values
- Apply field-specific evaluation metrics:
  - Normalized value comparison for currency amounts
  - Exact string matching for alphanumeric identifiers
- Calculate Character Error Rate (CER) for error analysis
- Categorize errors for non-matching values

## Testing Environment
- RunPod environment with approximately 94GB RAM and 150GB storage
- Jupyter notebooks with separate kernels for each experiment
- No shared environment between notebooks

## Expected Deliverables
1. A functional, well-documented codebase optimized for rapid completion
2. Comprehensive model comparison results by field type, prompt structure, and quantization
3. Analysis identifying best-performing model/prompt/quantization for each field type
4. Template notebooks for easy reproducibility

## Success Criteria
- Clear identification of best-performing model configuration for each field type
- Quantifiable metrics for extraction accuracy
- Completion within the 2-day timeframe
- Well-documented, maintainable code despite rapid implementation

## Technical Approach

The project will implement a standardized approach to model processing:

1. **Model Loading**
   - Use AutoModelForVision2Seq for consistent model loading
   - Support both local and HuggingFace model loading
   - Implement standardized model configuration

2. **Input Processing**
   - Use AutoProcessor for consistent input handling
   - Implement standardized chat-style input format
   - Support both text and image inputs in unified format

3. **Model Execution**
   - Standardized processing pipeline across all models
   - Consistent tensor handling and processing
   - Unified error handling and logging

4. **Result Processing**
   - Standardized output format for all models
   - Consistent result validation and error handling
   - Unified logging and reporting