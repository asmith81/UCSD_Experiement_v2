# ADR 004: Data Format Standardization

## Status
Accepted

## Context
The project initially used a mix of data formats:
- CSV for ground truth data
- JSON for model responses and results
- YAML for configuration

This led to inconsistencies in:
- Data validation
- Test execution
- Result processing
- Error handling

## Decision
Standardize on JSON format for all data exchange, including:
1. Ground truth data
2. Model responses
3. Test results
4. Configuration files

### Ground Truth Format
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

### Model Response Format
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

## Consequences
### Positive
- Consistent data handling across the system
- Simplified validation logic
- Better type safety with JSON schema
- Easier debugging and error tracking
- Reduced code complexity

### Negative
- Migration effort for existing CSV data
- Need to update test fixtures
- Potential performance impact for large datasets

### Mitigations
- Provide conversion utilities for legacy data
- Update documentation and examples
- Add validation schemas for JSON structures
- Implement efficient JSON parsing

## Implementation Notes
1. Update data_utils.py to support JSON format
2. Convert existing test data to JSON
3. Update validation functions
4. Add JSON schema validation
5. Update documentation

## Alternatives Considered
1. Keep mixed format approach
   - Pros: No migration needed
   - Cons: Complex validation, inconsistent handling
   - Rejected: Too complex to maintain

2. Use CSV for everything
   - Pros: Simple, widely supported
   - Cons: Limited structure, no type safety
   - Rejected: Not suitable for complex data

3. Use YAML for everything
   - Pros: Human readable, supports complex structures
   - Cons: Slower parsing, less common in ML
   - Rejected: Performance concerns 