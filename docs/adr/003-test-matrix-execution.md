# ADR 003: Test Matrix Structure and Execution Strategy

## Status
Accepted

## Context
The project requires a systematic way to test different combinations of:
- Models (pixtral, llama_vision, doctr)
- Quantization levels (4, 8, 16, 32)
- Prompt strategies (basic_extraction, detailed, few_shot, step_by_step, locational)

We needed to design a test matrix that:
1. Supports efficient execution in notebooks
2. Minimizes model reloading
3. Maintains clear separation of concerns
4. Aligns with the single prompt strategy (ADR 002)

## Decision
We will use a simplified test matrix structure with three columns:
- `model`: Specifies which model to use
- `quantization`: The quantization level for the model
- `prompt_strategy`: Which prompt template to use

The expected fields and JSON structure are defined in the prompt templates (YAML files) rather than the test matrix.

## Consequences
### Positive
- Cleaner test matrix structure
- Clear separation between test configuration and prompt content
- More efficient execution by grouping by quantization
- Better alignment with notebook-based execution
- Easier to maintain and extend

### Negative
- Need to ensure prompt templates are properly structured
- Additional complexity in prompt template management
- Requires careful coordination between test matrix and prompt templates

### Mitigations
- Clear documentation of prompt template structure
- Validation of prompt templates during execution
- Consistent naming conventions between test matrix and templates

## Implementation Details
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

### Notebook Structure
```python
# notebook_template.py
import execution

def main():
    # Set model for this notebook
    MODEL_NAME = "pixtral"  # or llama_vision or doctr
    
    # Run test suite
    execution.run_test_suite(
        model_name=MODEL_NAME,
        test_matrix_path="config/test_matrix.csv"
    )
```

## Alternatives Considered
1. Including expected fields in test matrix
   - Pros: Clear field expectations in one place
   - Cons: Redundant with prompt templates, harder to maintain
   - Rejected: Violates DRY principle, complicates maintenance

2. Separate test matrices per model
   - Pros: Simpler per-model management
   - Cons: Duplicates quantization and prompt strategy combinations
   - Rejected: Would lead to maintenance issues and potential inconsistencies

## Notes
This decision aligns with:
- ADR 001: Configuration Management Design
- ADR 002: Single Prompt Strategy
- Project's functional programming approach
- Code constraints for readability and maintainability 