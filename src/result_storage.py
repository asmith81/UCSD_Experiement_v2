"""
Unified Result Storage Protocol for LMM Invoice Data Extraction Comparison.
Consolidates overlapping functionality from different storage implementations.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import yaml
from datetime import datetime
from .config_protocol import ConfigManager, ConfigSection, ConfigValidationError
from .validation import ValidationManager, ResultValidationError

class StorageError(Exception):
    """Base class for storage errors."""
    pass

class ResultStorage:
    """Unified result storage implementation."""
    
    def __init__(self, config_manager: ConfigManager, validation_manager: ValidationManager):
        """Initialize with configuration and validation managers."""
        self.config_manager = config_manager
        self.validation_manager = validation_manager
        self._setup_storage()
        
    def _setup_storage(self) -> None:
        """Setup storage directories and files."""
        config = self.config_manager.get_config(ConfigSection.LOGGING)
        self.results_dir = Path(config["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def store_result(self, result: Dict[str, Any], model_name: str) -> None:
        """Store a processing result."""
        try:
            # Validate result
            self.validation_manager.validate_result(result, model_name)
            
            # Generate result filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}.json"
            filepath = self.results_dir / filename
            
            # Store result
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
                
            # Update index
            self._update_index(model_name, filename, result)
            
        except (ResultValidationError, StorageError) as e:
            raise StorageError(f"Failed to store result: {str(e)}")
            
    def get_result(self, model_name: str, timestamp: str) -> Dict[str, Any]:
        """Retrieve a specific result."""
        try:
            filename = f"{model_name}_{timestamp}.json"
            filepath = self.results_dir / filename
            
            if not filepath.exists():
                raise StorageError(f"Result not found: {filename}")
                
            with open(filepath, 'r') as f:
                return json.load(f)
                
        except (json.JSONDecodeError, StorageError) as e:
            raise StorageError(f"Failed to retrieve result: {str(e)}")
            
    def get_model_results(self, model_name: str) -> List[Dict[str, Any]]:
        """Retrieve all results for a model."""
        try:
            results = []
            for filepath in self.results_dir.glob(f"{model_name}_*.json"):
                with open(filepath, 'r') as f:
                    results.append(json.load(f))
            return results
            
        except (json.JSONDecodeError, StorageError) as e:
            raise StorageError(f"Failed to retrieve model results: {str(e)}")
            
    def _update_index(self, model_name: str, filename: str, result: Dict[str, Any]) -> None:
        """Update the results index."""
        try:
            index_file = self.results_dir / "index.yaml"
            
            # Load existing index
            if index_file.exists():
                with open(index_file, 'r') as f:
                    index = yaml.safe_load(f) or {}
            else:
                index = {}
                
            # Update index
            if model_name not in index:
                index[model_name] = []
                
            index[model_name].append({
                "filename": filename,
                "timestamp": result.get("timestamp", datetime.now().isoformat()),
                "confidence": result.get("confidence", 0.0),
                "processing_time": result.get("processing_time", 0.0)
            })
            
            # Store updated index
            with open(index_file, 'w') as f:
                yaml.dump(index, f)
                
        except (yaml.YAMLError, StorageError) as e:
            raise StorageError(f"Failed to update index: {str(e)}")
            
    def get_performance_metrics(self, model_name: str) -> Dict[str, Any]:
        """Get performance metrics for a model."""
        try:
            results = self.get_model_results(model_name)
            
            metrics = {
                "total_results": len(results),
                "average_confidence": 0.0,
                "average_processing_time": 0.0,
                "success_rate": 0.0
            }
            
            if results:
                total_confidence = sum(r.get("confidence", 0.0) for r in results)
                total_time = sum(r.get("processing_time", 0.0) for r in results)
                success_count = sum(1 for r in results if r.get("confidence", 0.0) >= 0.7)
                
                metrics["average_confidence"] = total_confidence / len(results)
                metrics["average_processing_time"] = total_time / len(results)
                metrics["success_rate"] = success_count / len(results)
                
            return metrics
            
        except StorageError as e:
            raise StorageError(f"Failed to get performance metrics: {str(e)}")
            
    def export_results(self, model_name: str, format: str = "json") -> None:
        """Export results in specified format."""
        try:
            results = self.get_model_results(model_name)
            export_file = self.results_dir / f"{model_name}_export.{format}"
            
            if format == "json":
                with open(export_file, 'w') as f:
                    json.dump(results, f, indent=2)
            elif format == "yaml":
                with open(export_file, 'w') as f:
                    yaml.dump(results, f)
            else:
                raise StorageError(f"Unsupported export format: {format}")
                
        except (json.JSONDecodeError, yaml.YAMLError, StorageError) as e:
            raise StorageError(f"Failed to export results: {str(e)}") 