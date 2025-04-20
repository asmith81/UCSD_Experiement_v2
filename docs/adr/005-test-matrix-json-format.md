# ADR 005: Test Matrix JSON Format

## Status
Accepted

## Context
Following ADR 004 (Data Format Standardization), we need to convert the test matrix from CSV to JSON format to maintain consistency across the project. The test matrix currently uses a CSV format with columns for model, quantization, and prompt strategy.

## Decision
Convert the test matrix to JSON format with the following structure:
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

Key changes:
1. Renamed fields for clarity:
   - `model` → `model_name`
   - `quantization` → `quant_level`
   - `prompt_strategy` → `prompt_type`
2. Added `field_type` to specify which field is being tested
3. Structured as a JSON array of test cases
4. Added validation for required fields and quantization levels

## Consequences
### Positive
- Consistent with project-wide JSON format standardization
- Better type safety and validation
- More flexible structure for future extensions
- Clearer field naming
- Better support for nested configurations

### Negative
- Need to update all test matrix references in code
- Migration effort for existing test matrices
- Need to update documentation

### Mitigations
- Provide conversion utility for existing CSV files
- Update all relevant code to handle JSON format
- Update documentation and examples
- Add validation for JSON structure

## Implementation Notes
1. Update `parse_test_matrix` in `src/config.py`
2. Update `load_test_matrix` in `src/execution.py`
3. Update model notebook templates
4. Create JSON version of test matrix
5. Add validation for required fields
6. Update documentation

## Alternatives Considered
1. Keep CSV format
   - Pros: Simple, widely supported
   - Cons: Limited structure, no type safety
   - Rejected: Inconsistent with project standards

2. Use YAML format
   - Pros: Human readable, supports complex structures
   - Cons: Slower parsing, less common in ML
   - Rejected: Performance concerns, inconsistent with other JSON formats

3. Use mixed format
   - Pros: No migration needed
   - Cons: Complex validation, inconsistent handling
   - Rejected: Violates standardization principle 