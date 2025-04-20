# Project To-Do: LMM Invoice Data Extraction Comparison (Rapid Implementation)

This to-do list is organized for the 8-hour rapid implementation timeline, with tasks grouped by functional area. Focus on completing core functionality before moving to additional features.

## Completed Tasks

### Core Framework
- [x] Set up project directory structure
- [x] Create environment setup script
- [x] Implement configuration management
- [x] Create test execution framework
- [x] Implement result logging system
- [x] Add dependency injection support
- [x] Implement protocol-based interfaces
- [x] Add default implementations for storage and processing

### Model Evaluation
- [x] Create model notebook template
- [x] Implement Pixtral evaluation notebook
- [x] Implement Llama Vision evaluation notebook
- [x] Implement Doctr evaluation notebook
- [x] Create results analysis notebook

### Documentation
- [x] Update README with project structure
- [x] Document core interfaces
- [x] Add setup instructions
- [x] Document dependency injection
- [x] Update interface control document

## Pending Tasks

### Testing
- [ ] Create unit tests for core functions
- [ ] Add integration tests for model evaluation
- [ ] Test dependency injection implementations
- [ ] Validate protocol implementations
- [ ] Test error handling scenarios

### Model Implementation
- [x] Implement Pixtral model loading
- [x] Implement Llama Vision model loading
- [x] Implement Doctr model loading
- [x] Add model-specific processors
- [x] Implement prompt loading system

### Results Analysis
- [ ] Add visualization functions
- [ ] Implement performance metrics
- [ ] Create comparison reports
- [ ] Add error analysis tools
- [ ] Implement storage adapters

### Documentation
- [ ] Add API documentation
- [ ] Create development guide
- [ ] Document testing procedures
- [ ] Add troubleshooting guide
- [ ] Create contribution guidelines

### Optimization
- [ ] Profile execution performance
- [ ] Optimize model loading
- [ ] Improve error handling
- [ ] Enhance logging system
- [ ] Add caching mechanisms

## Notes
- Focus on completing testing tasks next
- Prioritize model implementation tasks
- Document all changes as they are made
- Follow dependency injection patterns
- Maintain protocol-based interfaces

## Phase 1: Core Framework (2 hours)

### 1.A: Environment and Configuration (0.5 hour)
- [x] 1.1 Create `src/environment.py` for environment setup
  - Dependencies installation
  - Path configuration
  - CUDA setup for different quantization levels
- [x] 1.2 Create `src/config.py` for configuration management
  - YAML configuration loading
  - Test matrix CSV parsing
  - Result logging configuration
- [x] 1.3 Create `config/` directory structure
  - Model configuration files
  - Prompt template files
  - Test matrix CSV
- [x] 1.4 Create `src/execution.py` for test execution framework
  - Test matrix loading and filtering
  - Model loading with quantization
  - Prompt template loading
  - Result validation and logging
  - Execution tracking

### 1.B: Data Utilities (1.5 hours)
- [x] 1.5 Create `src/data_utils.py` for data management
  - Image loading functions
  - Ground truth loading with type normalization
- [x] 1.6 Create `src/results_logging.py` for result tracking
  - Execute the dictionary-based results capture in accordance with @adr and @ interace_control_document
  - Execution tracking functions
  - Result structure definitions

## Phase 2: Model Framework (3 hours)

### 2.A: Model Infrastructure (1 hour)
- [x] 2.1 Create `src/models/__init__.py` with model registry
  - Model type definitions
  - Factory function for model creation
- [x] 2.2 Create `src/models/common.py` with shared functions
  - Image preprocessing
  - Model loading utilities
  - Output parsing common code

### 2.B: Model Implementations (2 hours)
- [x] 2.3 Create `src/models/pixtral.py` for Pixtral-12B model
  - Model loading with quantization
  - Inference function
  - Output parsing
- [x] 2.4 Create `src/models/llama_vision.py` for Llama-3.2-11B-Vision model
  - Model loading with quantization
  - Inference function
  - Output parsing
- [x] 2.5 Create `src/models/doctr.py` for Doctr model
  - Model loading with quantization
  - Inference function
  - Output parsing

## Phase 3: Prompt Management (1 hour)
- [x] 3.1 Create `src/prompts.py` for prompt management
  - Prompt template loading
  - Prompt formatting for different models
  - Prompt strategy implementations (5-10 strategies)
- [x] 3.2 Create prompt template files
  - Basic extraction prompt
  - Detailed prompt
  - Few-shot example prompt
  - Step-by-step prompt
  - Locational prompt

## Phase 4: Notebook Implementation (1.5 hours)
- [x] 4.1 Create `notebooks/model_notebook_template.py`
  - Environment setup
  - Configuration loading
  - Test matrix parsing
  - Execution loop structure
  - Result logging
- [x] 4.2 Create model-specific notebooks
  - `notebooks/pixtral_evaluation.py`
  - `notebooks/llama_vision_evaluation.py`
  - `notebooks/doctr_evaluation.py`
- [x] 4.3 Create `notebooks/results_analysis.py`
  - Result loading functions
  - Comparative visualization
  - Best model identification

## Phase 5: Testing and Refinement (0.5 hour)
- [ ] 5.1 Test end-to-end pipeline
  - Verify model loading works
  - Confirm inference executes correctly
  - Validate result logging functions
  - Check resumability after interruption
- [ ] 5.2 Refine implementation based on initial tests
  - Fix any identified issues
  - Optimize performance bottlenecks
  - Improve error handling

## Priority Schedule (8 Hours)

| Time | Focus Area | Key Deliverables |
|------|------------|------------------|
| 0:00-0:30 | Environment & Config | Working environment setup |
| 0:30-2:00 | Data Utilities | Data loading and validation |
| 2:00-3:00 | Model Infrastructure | Model factory and registry |
| 3:00-5:00 | Model Implementations | Working model interfaces |
| 5:00-6:00 | Prompt Management | Prompt templates and formatting |
| 6:00-7:30 | Notebook Implementation | Working evaluation notebooks |
| 7:30-8:00 | Testing & Refinement | End-to-end verification |

## Implementation Notes

1. **Start with One Model**: Get one model working end-to-end before adding others
2. **Parallelize Testing**: Design for parallel execution of model notebooks
3. **Log Results Incrementally**: Save results after each test for crash resilience
4. **Focus on Functionality**: Prioritize working code over perfect architecture
5. **Documentation In Code**: Document functions as they're implemented

## Quick Checks for Each Phase

- **Phase 1**: Can we load and validate image and ground truth data?
- **Phase 2**: Can we load models at different quantization levels?
- **Phase 3**: Can we generate properly formatted prompts?
- **Phase 4**: Can we run a single test case end-to-end?
- **Phase 5**: Can we run multiple test cases and analyze results?