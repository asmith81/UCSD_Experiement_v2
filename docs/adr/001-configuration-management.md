# ADR 001: Configuration Management Design

## Status
Accepted

## Context
The project requires configuration management for:
- YAML configuration loading
- Test matrix CSV parsing
- Result logging configuration

Initial implementation created a circular dependency between `config.py` and `environment.py`, where:
- `config.py` imported `environment.py` for path setup
- This caused environment setup to be triggered on every config import
- Could lead to performance issues and race conditions in notebooks

We need a consistent way to manage configurations across different models and environments.

## Decision
Use YAML-based configuration files with clear structure and documentation.

## Consequences
- Easy to read and modify configurations
- Version control friendly
- Clear documentation of decisions

## Implementation Details
- Separate config files for models and prompts
- Environment setup in source modules
- Model-specific loading and inference functions

## Prompt Format Decision
For Pixtral model, we use the direct format instead of chat template because:
1. Our use case is single-image extraction
2. We don't need conversation history
3. Our prompts are relatively simple
4. We're not doing multi-turn interactions

This decision is documented in the model's configuration file and can be revisited if requirements change.

## Llama Vision Configuration Decisions

### Token Limit
- Increased max_new_tokens to 2048 based on documentation
- Original limit of 50 was too restrictive for detailed responses
- Documented in llama_vision.yaml

### Image Preprocessing
- Added specific image preprocessing requirements:
  - Max size: 1120x1120
  - RGB conversion
  - Normalization
  - Aspect ratio maintenance
- Based on documentation recommendations

### LoRA Support
- Added LoRA support for potential fine-tuning
- Configured with invoice-specific parameters
- Not currently required but available if needed

### Content Safety
- Decided against content safety checks because:
  - Processing business documents only
  - Structured, predictable content
  - Minimal risk of harmful content
  - Would add unnecessary complexity

### System Prompt
- Using built-in system prompt in direct format
- No need for dynamic system prompt changes
- Sufficient for structured invoice extraction

## Doctr Configuration Decisions

### Architecture Details
- Added explicit architecture configuration for:
  - Text detection using DBNet (db_resnet50)
  - Text recognition using CRNN (crnn_vgg16_bn)
  - KIE support for structured extraction
- Based on documentation recommendations

### Framework Selection
- Chose PyTorch backend for:
  - Better GPU support
  - More active development
  - Better integration with other tools
- Specified required dependencies

### Document Handling
- Added support for multiple document types:
  - PDF files
  - Single images
  - Multi-page documents
- Preprocessing requirements for each type
- No rotation handling (pre-processed images)

### Output Structure
- Defined clear JSON output structure
- Includes confidence scores
- Maintains geometry information
- Structured for easy parsing

### KIE Support
- Enabled KIE predictor for:
  - Work order number extraction
  - Total cost extraction
- Added confidence threshold
- Structured output format

## Notes
This decision aligns with the project's functional programming approach and code constraints:
- Stateless functions that transform data
- Clear input/output contracts
- Simple data structures for data exchange
- Focus on readability and maintainability 