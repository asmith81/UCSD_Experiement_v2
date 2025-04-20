# ADR 002: Single Prompt Strategy for Multiple Field Extraction

## Status
Accepted

## Context
When designing the prompt strategy for invoice field extraction, we needed to decide between:
1. Using a single prompt to extract both work order number and total cost simultaneously
2. Using separate prompts for each field, requiring multiple inferences per image

## Decision
We will use a single prompt strategy that extracts both fields in one inference call. This approach:
- Tests the model's ability to handle multiple tasks simultaneously
- More closely resembles real-world usage
- Reduces API costs and processing time
- Tests the model's ability to maintain context across multiple extractions

## Consequences
### Positive
- More efficient (one inference per image)
- Better represents production use case
- Reduces API costs
- Tests model's multitasking capability
- Simpler testing framework

### Negative
- Harder to isolate field-specific performance issues
- More complex prompts might lead to formatting errors
- Potential for one field's extraction to affect the other

### Mitigations
- Clear JSON structure for easy parsing
- Comprehensive error logging
- Detailed response validation
- Field-specific performance analysis in results

## Alternatives Considered
1. Separate prompts for each field
   - Pros: Clear isolation of performance, simpler prompts
   - Cons: 2x API calls, doesn't test multitasking
   - Rejected: Less efficient, doesn't match production use

2. Conditional prompting
   - Pros: Could optimize based on field difficulty
   - Cons: Adds complexity, harder to test
   - Rejected: Overly complex for this phase

## Implementation Notes
- Prompts include clear JSON structure
- Results tracking includes field-specific metrics
- Error handling accounts for partial successes
- Testing framework supports both individual and combined field analysis 