# LLM Code Verification Prompt Template

This prompt template is designed to help LLMs verify that the codebase matches the specifications in the function mapping JSON.

```markdown
I need to verify that the codebase matches the specification in the function mapping JSON. Please analyze the following aspects:

1. Function Signature Verification:
   - Compare each function's signature in the code with its specification in the JSON
   - Check parameter types, defaults, and return types
   - Verify that all required parameters are present
   - Confirm that optional parameters have correct defaults

2. Error Handling Verification:
   - Check if all specified error patterns (ENV001, MOD001, etc.) are implemented
   - Verify that error recovery strategies are implemented as specified
   - Confirm that error severity levels are handled appropriately
   - Check if error logging matches the specified format

3. Data Flow Verification:
   - Verify that input/output transformations match the specification
   - Check that data types are handled correctly at each step
   - Confirm that validation chains are implemented
   - Verify that error handling is applied at each transformation step

4. Dependency Verification:
   - Check that all specified dependencies are properly imported
   - Verify that class methods are implemented as specified
   - Confirm that external dependencies (transformers, etc.) are used correctly
   - Check that internal dependencies follow the specified paths

5. Logging Strategy Verification:
   - Verify that raw outputs are logged as specified
   - Check that processed results follow the correct format
   - Confirm that error logs are generated appropriately
   - Verify that log paths and naming conventions match

For each discrepancy found, please:
1. Identify the specific function or component
2. Describe the expected behavior from the JSON
3. Describe the actual behavior in the code
4. Suggest a fix to align the code with the specification

Please analyze the codebase and provide a detailed report of any mismatches or missing implementations.
```

## Usage Notes

1. **Context**: This prompt should be used with access to both the function mapping JSON and the actual codebase.

2. **Scope**: The prompt covers five main areas of verification:
   - Function signatures
   - Error handling
   - Data flow
   - Dependencies
   - Logging strategy

3. **Output Format**: The LLM should provide a structured report with:
   - Clear identification of discrepancies
   - Expected vs actual behavior
   - Specific suggestions for fixes

4. **Integration**: This prompt can be used with:
   - Code review tools
   - Continuous integration pipelines
   - Documentation generation
   - Test case generation

## Example Response Format

```markdown
# Verification Report

## 1. Function Signature Mismatches

### Function: setup_environment
- Expected: Parameter 'cuda_visible_devices' with type 'Optional[str]'
- Actual: Parameter missing
- Fix: Add parameter with correct type and default value

## 2. Error Handling Issues

### Error Code: ENV001
- Expected: Recovery strategy "Use default device configuration"
- Actual: No recovery strategy implemented
- Fix: Implement recovery logic in error handler

## 3. Data Flow Problems

### Transformation: find_root
- Expected: Input None -> Output Path
- Actual: Input None -> Output Optional[Path]
- Fix: Update return type annotation and add validation

## 4. Dependency Issues

### Class: ModelProcessor
- Expected: Method '_validate_model_components'
- Actual: Method missing
- Fix: Implement missing method with specified signature

## 5. Logging Inconsistencies

### Raw Output Logging
- Expected: Path format "results/raw/{test_id}_raw.json"
- Actual: Using different path format
- Fix: Update path construction to match specification
```

## Related Documents

- [Function Mapping JSON](pixtral_function_mapping.json)
- [Interface Control Document](interface_control_document.md)
- [Project Implementation Mapping](project_implementation_mapping.json) 