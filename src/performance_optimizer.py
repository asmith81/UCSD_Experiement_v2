"""
Performance optimization module for LMM Invoice Data Extraction Comparison.
Handles batch processing and memory optimization.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import torch
import gc
from datetime import datetime
from .config_protocol import ConfigManager, ConfigSection
from .validation import ValidationManager

class PerformanceOptimizer:
    """Manages performance optimization for model processing."""
    
    def __init__(self, config_manager: ConfigManager, validation_manager: ValidationManager):
        """Initialize with configuration and validation managers."""
        self.config_manager = config_manager
        self.validation_manager = validation_manager
        self._setup_optimization()
        
    def _setup_optimization(self) -> None:
        """Setup optimization parameters."""
        self.batch_size = 1
        self.max_memory_gb = 10
        self.device = self._get_optimal_device()
        
    def _get_optimal_device(self) -> str:
        """Determine the optimal device for processing."""
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb >= self.max_memory_gb:
                return "cuda"
        return "cpu"
        
    def optimize_model_loading(self, model_name: str) -> Dict[str, Any]:
        """Optimize model loading based on hardware capabilities."""
        try:
            config = self.config_manager.get_model_config(model_name)
            self.validation_manager.validate_model(model_name, config)
            
            # Get hardware requirements
            min_memory = config["hardware"]["min_gpu_memory_gb"]
            recommended_memory = config["hardware"]["recommended_gpu_memory_gb"]
            
            # Determine optimal loading strategy
            loading_params = {
                "device_map": "auto",
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True
            }
            
            # Apply quantization if supported and beneficial
            if config["hardware"]["quantization_support"]:
                if self.device == "cpu" or min_memory > torch.cuda.get_device_properties(0).total_memory / (1024**3):
                    loading_params.update({
                        "load_in_8bit": True,
                        "bnb_8bit_quant_type": "fp8"
                    })
                    
            return loading_params
            
        except Exception as e:
            raise RuntimeError(f"Failed to optimize model loading: {str(e)}")
            
    def optimize_batch_processing(self, model_name: str, inputs: List[Any]) -> List[List[Any]]:
        """Optimize batch processing based on memory constraints."""
        try:
            config = self.config_manager.get_model_config(model_name)
            
            # Calculate optimal batch size
            max_batch_memory = config["inference"]["max_batch_memory_gb"]
            available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if self.device == "cuda" else 8
            
            # Calculate safe batch size
            safe_batch_size = min(
                config["inference"]["batch_size"],
                int(available_memory / max_batch_memory)
            )
            
            # Split inputs into optimal batches
            batches = []
            for i in range(0, len(inputs), safe_batch_size):
                batches.append(inputs[i:i + safe_batch_size])
                
            return batches
            
        except Exception as e:
            raise RuntimeError(f"Failed to optimize batch processing: {str(e)}")
            
    def optimize_memory_usage(self) -> None:
        """Optimize memory usage by clearing caches and garbage collection."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
    def track_performance(self, start_time: datetime, end_time: datetime, 
                         gpu_utilization: float, memory_usage: float) -> Dict[str, float]:
        """Track and return performance metrics."""
        processing_time = (end_time - start_time).total_seconds()
        
        metrics = {
            "processing_time": processing_time,
            "gpu_utilization": gpu_utilization,
            "memory_usage": memory_usage,
            "throughput": 1.0 / processing_time if processing_time > 0 else 0.0
        }
        
        return metrics
        
    def get_performance_recommendations(self, model_name: str) -> Dict[str, Any]:
        """Get performance optimization recommendations."""
        try:
            config = self.config_manager.get_model_config(model_name)
            
            recommendations = {
                "device": self.device,
                "batch_size": config["inference"]["batch_size"],
                "quantization": config["hardware"]["quantization_support"],
                "memory_optimization": True,
                "suggested_improvements": []
            }
            
            # Add specific recommendations based on configuration
            if self.device == "cpu" and config["hardware"]["quantization_support"]:
                recommendations["suggested_improvements"].append(
                    "Enable quantization for better CPU performance"
                )
                
            if config["inference"]["batch_size"] > 1:
                recommendations["suggested_improvements"].append(
                    "Consider increasing batch size for better throughput"
                )
                
            return recommendations
            
        except Exception as e:
            raise RuntimeError(f"Failed to get performance recommendations: {str(e)}") 