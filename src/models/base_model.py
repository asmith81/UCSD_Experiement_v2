"""
Base model class with performance optimization integration.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import torch
from datetime import datetime
from src.config_protocol import ConfigManager
from src.validation import ValidationManager
from .performance_optimizer import PerformanceOptimizer

class BaseModel:
    """Base model class with performance optimization."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config_manager: ConfigManager,
        validation_manager: ValidationManager,
        quantization: int = 32
    ):
        """Initialize base model with performance optimization.
        
        Args:
            model_path: Path to model weights
            config_manager: Configuration manager instance
            validation_manager: Validation manager instance
            quantization: Bit width for quantization (4, 8, 16, 32)
        """
        self.model_path = Path(model_path)
        self.config_manager = config_manager
        self.validation_manager = validation_manager
        self.quantization = quantization
        
        # Initialize performance optimizer
        self.optimizer = PerformanceOptimizer(config_manager, validation_manager)
        
        # Get optimal loading parameters
        self.loading_params = self.optimizer.optimize_model_loading(self.name)
        
        # Initialize model and processor
        self.model = None
        self.processor = None
        self._load_model()
        
    def _load_model(self) -> None:
        """Load model with performance optimization."""
        try:
            # Apply optimized loading parameters
            self.model = self._load_model_weights(self.loading_params)
            self.processor = self._load_processor()
            
            # Move model to optimal device
            self.model = self.model.to(self.optimizer.device)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
            
    def _load_model_weights(self, loading_params: Dict[str, Any]) -> Any:
        """Load model weights with specified parameters."""
        raise NotImplementedError("Subclasses must implement _load_model_weights")
        
    def _load_processor(self) -> Any:
        """Load model processor."""
        raise NotImplementedError("Subclasses must implement _load_processor")
        
    def process_batch(
        self,
        inputs: List[Any],
        prompt: str,
        field_type: str
    ) -> List[Dict[str, Any]]:
        """Process a batch of inputs with performance optimization."""
        try:
            # Optimize batch size
            batches = self.optimizer.optimize_batch_processing(self.name, inputs)
            
            results = []
            for batch in batches:
                # Track performance
                start_time = datetime.now()
                
                # Process batch
                batch_results = self._process_batch(batch, prompt, field_type)
                
                # Get performance metrics
                end_time = datetime.now()
                gpu_utilization = self._get_gpu_utilization()
                memory_usage = self._get_memory_usage()
                
                metrics = self.optimizer.track_performance(
                    start_time=start_time,
                    end_time=end_time,
                    gpu_utilization=gpu_utilization,
                    memory_usage=memory_usage
                )
                
                # Add metrics to results
                for result in batch_results:
                    result.update(metrics)
                    
                results.extend(batch_results)
                
                # Optimize memory usage
                self.optimizer.optimize_memory_usage()
                
            return results
            
        except Exception as e:
            raise RuntimeError(f"Failed to process batch: {str(e)}")
            
    def _process_batch(
        self,
        batch: List[Any],
        prompt: str,
        field_type: str
    ) -> List[Dict[str, Any]]:
        """Process a single batch of inputs."""
        raise NotImplementedError("Subclasses must implement _process_batch")
        
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization."""
        if torch.cuda.is_available():
            return torch.cuda.utilization()
        return 0.0
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        return 0.0
        
    def get_performance_recommendations(self) -> Dict[str, Any]:
        """Get performance optimization recommendations."""
        return self.optimizer.get_performance_recommendations(self.name)
        
    @property
    def name(self) -> str:
        """Get model name."""
        raise NotImplementedError("Subclasses must implement name property") 