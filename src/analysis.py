"""
Standardized analysis implementation for model comparison.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .config_protocol import ConfigManager
from .validation import ValidationManager

class ModelAnalyzer:
    """Standardized model analysis implementation."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        validation_manager: ValidationManager
    ):
        """Initialize model analyzer.
        
        Args:
            config_manager: Configuration manager instance
            validation_manager: Validation manager instance
        """
        self.config_manager = config_manager
        self.validation_manager = validation_manager
        self.config = self.config_manager.get_section("analysis")
        
    def analyze_prompt_performance(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
        """Analyze performance by prompt strategy across models and field types."""
        try:
            # Convert results to DataFrame
            df = pd.DataFrame(results)
            
            # Group by model, prompt, and field type
            grouped = df.groupby(['model', 'prompt_strategy', 'field_type'])
            
            # Calculate metrics
            metrics = {}
            for (model, prompt, field), group in grouped:
                if model not in metrics:
                    metrics[model] = {}
                if prompt not in metrics[model]:
                    metrics[model][prompt] = {}
                    
                metrics[model][prompt][field] = {
                    'accuracy': group['normalized_match'].mean(),
                    'cer': group['cer'].mean(),
                    'processing_time': group['processing_time'].mean()
                }
                
            return metrics
            
        except Exception as e:
            raise RuntimeError(f"Failed to analyze prompt performance: {str(e)}")
            
    def analyze_quantization_performance(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
        """Analyze performance by quantization level across models and prompt strategies."""
        try:
            # Convert results to DataFrame
            df = pd.DataFrame(results)
            
            # Group by model, quantization, and prompt strategy
            grouped = df.groupby(['model', 'quantization', 'prompt_strategy'])
            
            # Calculate metrics
            metrics = {}
            for (model, quant, prompt), group in grouped:
                if model not in metrics:
                    metrics[model] = {}
                if quant not in metrics[model]:
                    metrics[model][quant] = {}
                    
                metrics[model][quant][prompt] = {
                    'accuracy': group['normalized_match'].mean(),
                    'cer': group['cer'].mean(),
                    'processing_time': group['processing_time'].mean(),
                    'memory_usage': group['memory_usage'].mean()
                }
                
            return metrics
            
        except Exception as e:
            raise RuntimeError(f"Failed to analyze quantization performance: {str(e)}")
            
    def aggregate_by_model_field(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Aggregate results by model and field type."""
        try:
            # Convert results to DataFrame
            df = pd.DataFrame(results)
            
            # Group by model and field type
            grouped = df.groupby(['model', 'field_type'])
            
            # Calculate metrics
            metrics = {}
            for (model, field), group in grouped:
                if model not in metrics:
                    metrics[model] = {}
                    
                metrics[model][field] = {
                    'total': len(group),
                    'matches': group['normalized_match'].sum(),
                    'avg_cer': group['cer'].mean(),
                    'avg_time': group['processing_time'].mean()
                }
                
            return metrics
            
        except Exception as e:
            raise RuntimeError(f"Failed to aggregate by model field: {str(e)}")
            
    def analyze_error_distribution(
        self,
        results: List[Dict[str, Any]],
        field_type: str
    ) -> Dict[str, int]:
        """Calculate distribution of error categories for specific field type."""
        try:
            # Convert results to DataFrame
            df = pd.DataFrame(results)
            
            # Filter by field type
            df = df[df['field_type'] == field_type]
            
            # Count error categories
            error_dist = df['error_category'].value_counts().to_dict()
            
            return error_dist
            
        except Exception as e:
            raise RuntimeError(f"Failed to analyze error distribution: {str(e)}")
            
    def plot_model_comparison(
        self,
        metrics: Dict[str, Dict[str, Dict[str, float]]],
        output_path: Optional[Path] = None
    ) -> None:
        """Plot comparison of models by field type."""
        try:
            # Convert metrics to DataFrame
            data = []
            for model, fields in metrics.items():
                for field, stats in fields.items():
                    data.append({
                        'model': model,
                        'field': field,
                        'accuracy': stats['matches'] / stats['total'],
                        'cer': stats['avg_cer'],
                        'time': stats['avg_time']
                    })
            df = pd.DataFrame(data)
            
            # Create figure
            fig, axes = plt.subplots(3, 1, figsize=(10, 15))
            
            # Plot accuracy
            sns.barplot(data=df, x='model', y='accuracy', hue='field', ax=axes[0])
            axes[0].set_title('Accuracy by Model and Field')
            axes[0].set_ylim(0, 1)
            
            # Plot CER
            sns.barplot(data=df, x='model', y='cer', hue='field', ax=axes[1])
            axes[1].set_title('Character Error Rate by Model and Field')
            
            # Plot time
            sns.barplot(data=df, x='model', y='time', hue='field', ax=axes[2])
            axes[2].set_title('Processing Time by Model and Field')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show
            if output_path:
                plt.savefig(output_path)
            else:
                plt.show()
                
        except Exception as e:
            raise RuntimeError(f"Failed to plot model comparison: {str(e)}")
            
    def plot_prompt_comparison(
        self,
        prompt_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
        output_path: Optional[Path] = None
    ) -> None:
        """Plot comparison of prompt strategies across models and field types."""
        try:
            # Convert metrics to DataFrame
            data = []
            for model, prompts in prompt_metrics.items():
                for prompt, fields in prompts.items():
                    for field, stats in fields.items():
                        data.append({
                            'model': model,
                            'prompt': prompt,
                            'field': field,
                            'accuracy': stats['accuracy'],
                            'cer': stats['cer'],
                            'time': stats['processing_time']
                        })
            df = pd.DataFrame(data)
            
            # Create figure
            fig, axes = plt.subplots(3, 1, figsize=(15, 15))
            
            # Plot accuracy
            sns.barplot(data=df, x='model', y='accuracy', hue='prompt', ax=axes[0])
            axes[0].set_title('Accuracy by Model and Prompt')
            axes[0].set_ylim(0, 1)
            
            # Plot CER
            sns.barplot(data=df, x='model', y='cer', hue='prompt', ax=axes[1])
            axes[1].set_title('Character Error Rate by Model and Prompt')
            
            # Plot time
            sns.barplot(data=df, x='model', y='time', hue='prompt', ax=axes[2])
            axes[2].set_title('Processing Time by Model and Prompt')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show
            if output_path:
                plt.savefig(output_path)
            else:
                plt.show()
                
        except Exception as e:
            raise RuntimeError(f"Failed to plot prompt comparison: {str(e)}")
            
    def create_performance_heatmap(
        self,
        prompt_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
        field_type: str,
        metric: str = 'accuracy',
        output_path: Optional[Path] = None
    ) -> None:
        """Create a heatmap of model vs prompt performance."""
        try:
            # Convert metrics to DataFrame
            data = []
            for model, prompts in prompt_metrics.items():
                for prompt, fields in prompts.items():
                    if field_type in fields:
                        data.append({
                            'model': model,
                            'prompt': prompt,
                            'value': fields[field_type][metric]
                        })
            df = pd.DataFrame(data)
            
            # Pivot for heatmap
            heatmap_data = df.pivot('model', 'prompt', 'value')
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu')
            plt.title(f'{metric.capitalize()} by Model and Prompt ({field_type})')
            
            # Save or show
            if output_path:
                plt.savefig(output_path)
            else:
                plt.show()
                
        except Exception as e:
            raise RuntimeError(f"Failed to create performance heatmap: {str(e)}")
            
    def plot_error_distribution(
        self,
        error_dist: Dict[str, int],
        field_type: str,
        output_path: Optional[Path] = None
    ) -> None:
        """Plot distribution of error categories."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame({
                'error_category': list(error_dist.keys()),
                'count': list(error_dist.values())
            })
            
            # Create bar plot
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x='error_category', y='count')
            plt.title(f'Error Distribution for {field_type}')
            plt.xticks(rotation=45)
            
            # Save or show
            if output_path:
                plt.savefig(output_path)
            else:
                plt.show()
                
        except Exception as e:
            raise RuntimeError(f"Failed to plot error distribution: {str(e)}") 