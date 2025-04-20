# ADR 006: Ground Truth Column Names

## Status
Accepted

## Context
The ground truth data loading functionality was previously designed to be flexible with column names, using a mapping system to handle different variations of column names (e.g., 'image_id', 'id', 'image', etc.). This flexibility was intended to make the system more adaptable to different data sources.

## Decision
We have decided to simplify the ground truth data loading by requiring exact column names:
- `image_id`
- `work_order_number`
- `total_cost`

The column name mapping system has been removed in favor of direct column access.

## Consequences
### Positive
- Simpler, more maintainable code
- Clearer error messages when columns are missing
- Reduced complexity in data loading logic
- Better alignment with the principle of explicit over implicit

### Negative
- Less flexibility in handling different column names
- Requires data providers to use exact column names
- May require data preprocessing for external data sources

## Implementation Notes
The `load_ground_truth` function in `data_utils.py` has been simplified to:
1. Validate the presence of exact column names
2. Use direct column access instead of mapping
3. Provide clear error messages for missing columns

## Related ADRs
- [ADR 004: Data Format Standardization](004-data-format-standardization.md) 