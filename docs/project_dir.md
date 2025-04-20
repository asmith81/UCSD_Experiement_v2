# Project Directory Structure

This document describes the project's directory structure and the purpose of each component.

## Root Directory

```
UCSD_MJM_Experiment/
├── config/                         # Configuration files
│   ├── models/                     # Model-specific configurations
│   ├── prompts/                    # Field-specific prompt templates
│   ├── config.yaml                 # Main configuration file
│   └── test_matrix.csv            # Test configuration matrix
├── src/                            # Source code modules
│   ├── environment.py              # Environment setup and paths
│   ├── config.py                   # Configuration management
│   ├── execution.py                # Test execution framework
│   ├── data_utils.py               # Data loading and processing
│   ├── results_logging.py          # Result logging and tracking
│   └── models/                     # Model implementations
├── notebooks/                      # Jupyter notebooks
│   └── model_notebook_template.py  # Template for model evaluation
├── docs/                           # Documentation
│   ├── interface_control_document.md
│   ├── project_dir.md
│   └── project_todo.md
├── tests/                          # Test files
├── requirements.txt                # Python dependencies
├── README.md                       # Project overview
└── .gitignore                     # Git ignore rules
```

## Directory Descriptions

### config/
Contains all configuration files for the project:
- `models/`: Model-specific configurations
- `prompts/`: Prompt templates for different strategies
- `config.yaml`: Main configuration file
- `test_matrix.csv`: Test configuration matrix

### src/
Core source code modules:
- `environment.py`: Environment setup
- `config.py`: Configuration management
- `execution.py`: Test execution framework
- `data_utils.py`: Data loading and processing
- `results_logging.py`: Result logging and tracking
- `models/`: Model implementations

### notebooks/
Jupyter notebooks for evaluation:
- `model_notebook_template.py`: Template for model evaluation

## Implementation Notes

1. **Flat structure**: Minimizes directory nesting for simpler navigation
2. **Functional organization**: Groups code by functionality rather than architecture
3. **Clear separation**: Keeps configuration, code, and data clearly separated
4. **Parallel execution**: Supports running model notebooks in parallel
5. **Simplified imports**: Uses direct imports rather than complex package structure

This directory structure is designed for rapid implementation while maintaining good organization and separation of concerns. The structure follows a functional approach that can be implemented quickly while still producing reliable results.