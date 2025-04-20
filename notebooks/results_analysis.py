"""
Results Analysis Notebook for LMM Invoice Data Extraction Comparison

This notebook provides tools for analyzing and comparing results across different models,
quantization levels, and prompt strategies.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.environment import setup_environment
from src.config import load_yaml_config
from src.results_logging import load_result

# Setup environment and logging
env = setup_environment()
paths = env['paths']
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure plotting style
plt.style.use('seaborn')
sns.set_palette("husl")

def load_all_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all result files from the results directory."""
    all_results = []
    for result_file in results_dir.glob('*.json'):
        try:
            result = load_result(result_file)
            all_results.append(result)
            logger.info(f"Loaded results from {result_file.name}")
        except Exception as e:
            logger.error(f"Error loading {result_file}: {str(e)}")
    return all_results

def filter_results(results: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """Filter results by any parameter."""
    filtered = []
    for result in results:
        match = True
        for key, value in kwargs.items():
            if result['test_parameters'].get(key) != value:
                match = False
                break
        if match:
            filtered.append(result)
    return filtered

def aggregate_by_model_field(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Aggregate results by model and field type."""
    metrics = {}
    for result in results:
        model = result['test_parameters']['model_name']
        field_type = result['test_parameters']['field_type']
        
        if model not in metrics:
            metrics[model] = {}
        if field_type not in metrics[model]:
            metrics[model][field_type] = {
                'total': 0,
                'matches': 0,
                'avg_cer': 0.0
            }
        
        total = 0
        matches = 0
        cer_sum = 0.0
        
        for image_result in result['results_by_image'].values():
            total += 1
            if image_result['evaluation']['normalized_match']:
                matches += 1
            cer_sum += image_result['evaluation']['cer']
        
        metrics[model][field_type]['total'] += total
        metrics[model][field_type]['matches'] += matches
        metrics[model][field_type]['avg_cer'] = cer_sum / total if total > 0 else 0.0
    
    return metrics

def plot_model_comparison(metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """Plot comparison of models by field type."""
    models = list(metrics.keys())
    field_types = list(metrics[models[0]].keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    for field_type in field_types:
        accuracies = [
            metrics[model][field_type]['matches'] / metrics[model][field_type]['total']
            for model in models
        ]
        axes[0].plot(models, accuracies, marker='o', label=field_type)
    
    axes[0].set_title('Accuracy by Model and Field Type')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot CER
    for field_type in field_types:
        cers = [
            metrics[model][field_type]['avg_cer']
            for model in models
        ]
        axes[1].plot(models, cers, marker='o', label=field_type)
    
    axes[1].set_title('Character Error Rate by Model and Field Type')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('CER')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_error_distribution(results: List[Dict[str, Any]], field_type: str) -> Dict[str, int]:
    """Calculate distribution of error categories for specific field type."""
    error_dist = {}
    for result in results:
        if result['test_parameters']['field_type'] != field_type:
            continue
            
        for image_result in result['results_by_image'].values():
            if not image_result['evaluation']['normalized_match']:
                error_type = image_result['evaluation']['error_category']
                error_dist[error_type] = error_dist.get(error_type, 0) + 1
    
    return error_dist

def plot_error_distribution(error_dist: Dict[str, int], field_type: str) -> None:
    """Plot distribution of error categories."""
    plt.figure(figsize=(10, 6))
    plt.bar(error_dist.keys(), error_dist.values())
    plt.title(f'Error Distribution for {field_type}')
    plt.xlabel('Error Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def find_best_configuration(results: List[Dict[str, Any]], field_type: str) -> Dict[str, Any]:
    """Find the best configuration for a specific field type."""
    best_config = None
    best_accuracy = 0.0
    
    for result in results:
        if result['test_parameters']['field_type'] != field_type:
            continue
            
        total = 0
        matches = 0
        
        for image_result in result['results_by_image'].values():
            total += 1
            if image_result['evaluation']['normalized_match']:
                matches += 1
        
        accuracy = matches / total if total > 0 else 0.0
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_config = result['test_parameters']
    
    return best_config

def analyze_prompt_performance(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Analyze performance by prompt strategy across models and field types."""
    prompt_metrics = {}
    
    for result in results:
        model = result['test_parameters']['model_name']
        field_type = result['test_parameters']['field_type']
        prompt_strategy = result['test_parameters']['prompt_strategy']
        
        if prompt_strategy not in prompt_metrics:
            prompt_metrics[prompt_strategy] = {}
        if model not in prompt_metrics[prompt_strategy]:
            prompt_metrics[prompt_strategy][model] = {}
        if field_type not in prompt_metrics[prompt_strategy][model]:
            prompt_metrics[prompt_strategy][model][field_type] = {
                'total': 0,
                'matches': 0,
                'avg_cer': 0.0,
                'avg_time': 0.0
            }
        
        total = 0
        matches = 0
        cer_sum = 0.0
        time_sum = 0.0
        
        for image_result in result['results_by_image'].values():
            total += 1
            if image_result['evaluation']['normalized_match']:
                matches += 1
            cer_sum += image_result['evaluation']['cer']
            time_sum += image_result['model_response']['processing_time']
        
        metrics = prompt_metrics[prompt_strategy][model][field_type]
        metrics['total'] += total
        metrics['matches'] += matches
        metrics['avg_cer'] = cer_sum / total if total > 0 else 0.0
        metrics['avg_time'] = time_sum / total if total > 0 else 0.0
    
    return prompt_metrics

def plot_prompt_comparison(prompt_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]) -> None:
    """Plot comparison of prompt strategies across models and field types."""
    prompt_strategies = list(prompt_metrics.keys())
    models = list(prompt_metrics[prompt_strategies[0]].keys())
    field_types = list(prompt_metrics[prompt_strategies[0]][models[0]].keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # Plot accuracy by prompt strategy
    for model in models:
        for field_type in field_types:
            accuracies = [
                prompt_metrics[strategy][model][field_type]['matches'] / 
                prompt_metrics[strategy][model][field_type]['total']
                for strategy in prompt_strategies
            ]
            axes[0, 0].plot(prompt_strategies, accuracies, marker='o', 
                          label=f"{model} - {field_type}")
    
    axes[0, 0].set_title('Accuracy by Prompt Strategy')
    axes[0, 0].set_xlabel('Prompt Strategy')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
    
    # Plot CER by prompt strategy
    for model in models:
        for field_type in field_types:
            cers = [
                prompt_metrics[strategy][model][field_type]['avg_cer']
                for strategy in prompt_strategies
            ]
            axes[0, 1].plot(prompt_strategies, cers, marker='o',
                          label=f"{model} - {field_type}")
    
    axes[0, 1].set_title('Character Error Rate by Prompt Strategy')
    axes[0, 1].set_xlabel('Prompt Strategy')
    axes[0, 1].set_ylabel('CER')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
    
    # Plot processing time by prompt strategy
    for model in models:
        for field_type in field_types:
            times = [
                prompt_metrics[strategy][model][field_type]['avg_time']
                for strategy in prompt_strategies
            ]
            axes[1, 0].plot(prompt_strategies, times, marker='o',
                          label=f"{model} - {field_type}")
    
    axes[1, 0].set_title('Processing Time by Prompt Strategy')
    axes[1, 0].set_xlabel('Prompt Strategy')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
    
    # Plot error distribution by prompt strategy
    for model in models:
        for field_type in field_types:
            error_rates = [
                1 - (prompt_metrics[strategy][model][field_type]['matches'] / 
                     prompt_metrics[strategy][model][field_type]['total'])
                for strategy in prompt_strategies
            ]
            axes[1, 1].plot(prompt_strategies, error_rates, marker='o',
                          label=f"{model} - {field_type}")
    
    axes[1, 1].set_title('Error Rate by Prompt Strategy')
    axes[1, 1].set_xlabel('Prompt Strategy')
    axes[1, 1].set_ylabel('Error Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

def find_best_prompt_strategy(prompt_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], 
                            field_type: str) -> Dict[str, Any]:
    """Find the best prompt strategy for a specific field type."""
    best_strategy = None
    best_model = None
    best_accuracy = 0.0
    
    for strategy in prompt_metrics:
        for model in prompt_metrics[strategy]:
            if field_type in prompt_metrics[strategy][model]:
                metrics = prompt_metrics[strategy][model][field_type]
                accuracy = metrics['matches'] / metrics['total'] if metrics['total'] > 0 else 0.0
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_strategy = strategy
                    best_model = model
    
    return {
        'strategy': best_strategy,
        'model': best_model,
        'accuracy': best_accuracy,
        'cer': prompt_metrics[best_strategy][best_model][field_type]['avg_cer'],
        'avg_time': prompt_metrics[best_strategy][best_model][field_type]['avg_time']
    }

def create_performance_heatmap(prompt_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], 
                             field_type: str, metric: str = 'accuracy') -> None:
    """Create a heatmap of model vs prompt performance."""
    models = sorted(list(prompt_metrics[next(iter(prompt_metrics))].keys()))
    strategies = sorted(list(prompt_metrics.keys()))
    
    # Create performance matrix
    performance_matrix = np.zeros((len(models), len(strategies)))
    
    for i, model in enumerate(models):
        for j, strategy in enumerate(strategies):
            if field_type in prompt_metrics[strategy][model]:
                metrics = prompt_metrics[strategy][model][field_type]
                if metric == 'accuracy':
                    value = metrics['matches'] / metrics['total'] if metrics['total'] > 0 else 0.0
                elif metric == 'cer':
                    value = metrics['avg_cer']
                elif metric == 'time':
                    value = metrics['avg_time']
                performance_matrix[i, j] = value
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        performance_matrix,
        annot=True,
        fmt='.2f',
        cmap='YlGnBu',
        xticklabels=strategies,
        yticklabels=models,
        cbar_kws={'label': metric.title()}
    )
    plt.title(f'{metric.title()} by Model and Prompt Strategy\nField Type: {field_type}')
    plt.xlabel('Prompt Strategy')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.show()

def analyze_quantization_performance(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Analyze performance by quantization level across models and prompt strategies."""
    quant_metrics = {}
    
    for result in results:
        model = result['test_parameters']['model_name']
        field_type = result['test_parameters']['field_type']
        prompt_strategy = result['test_parameters']['prompt_strategy']
        quant_level = result['test_parameters']['quantization']
        
        if quant_level not in quant_metrics:
            quant_metrics[quant_level] = {}
        if model not in quant_metrics[quant_level]:
            quant_metrics[quant_level][model] = {}
        if prompt_strategy not in quant_metrics[quant_level][model]:
            quant_metrics[quant_level][model][prompt_strategy] = {
                'total': 0,
                'matches': 0,
                'avg_cer': 0.0,
                'avg_time': 0.0
            }
        
        total = 0
        matches = 0
        cer_sum = 0.0
        time_sum = 0.0
        
        for image_result in result['results_by_image'].values():
            total += 1
            if image_result['evaluation']['normalized_match']:
                matches += 1
            cer_sum += image_result['evaluation']['cer']
            time_sum += image_result['model_response']['processing_time']
        
        metrics = quant_metrics[quant_level][model][prompt_strategy]
        metrics['total'] += total
        metrics['matches'] += matches
        metrics['avg_cer'] = cer_sum / total if total > 0 else 0.0
        metrics['avg_time'] = time_sum / total if total > 0 else 0.0
    
    return quant_metrics

def plot_quantization_heatmap(quant_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], 
                            model: str, prompt_strategy: str, metric: str = 'accuracy') -> None:
    """Create a heatmap of quantization performance for a specific model and prompt strategy."""
    quant_levels = sorted(list(quant_metrics.keys()))
    field_types = ['work_order_number', 'total_cost']
    
    # Create performance matrix
    performance_matrix = np.zeros((len(quant_levels), len(field_types)))
    
    for i, quant_level in enumerate(quant_levels):
        for j, field_type in enumerate(field_types):
            if model in quant_metrics[quant_level] and prompt_strategy in quant_metrics[quant_level][model]:
                metrics = quant_metrics[quant_level][model][prompt_strategy]
                if metric == 'accuracy':
                    value = metrics['matches'] / metrics['total'] if metrics['total'] > 0 else 0.0
                elif metric == 'cer':
                    value = metrics['avg_cer']
                elif metric == 'time':
                    value = metrics['avg_time']
                performance_matrix[i, j] = value
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        performance_matrix,
        annot=True,
        fmt='.2f',
        cmap='YlGnBu',
        xticklabels=field_types,
        yticklabels=quant_levels,
        cbar_kws={'label': metric.title()}
    )
    plt.title(f'{metric.title()} by Quantization Level and Field Type\nModel: {model}, Prompt: {prompt_strategy}')
    plt.xlabel('Field Type')
    plt.ylabel('Quantization Level (bits)')
    plt.tight_layout()
    plt.show()

def load_test_matrix(test_matrix_path: str) -> List[Dict[str, Any]]:
    """
    Load and validate the test matrix JSON file.
    
    Args:
        test_matrix_path: Path to the test matrix JSON file
        
    Returns:
        List of test cases
        
    Raises:
        FileNotFoundError: If test matrix file doesn't exist
        ValueError: If test matrix is invalid
    """
    if not Path(test_matrix_path).exists():
        raise FileNotFoundError(f"Test matrix file not found: {test_matrix_path}")
        
    try:
        with open(test_matrix_path, 'r') as f:
            data = json.load(f)
            
        if 'test_cases' not in data:
            raise ValueError("Test matrix must contain 'test_cases' array")
            
        return data['test_cases']
        
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in test matrix file: {test_matrix_path}")

def analyze_results(results: List[Dict[str, Any]], test_matrix_path: str) -> Dict[str, Any]:
    """
    Analyze test results and generate performance metrics.
    
    Args:
        results: List of test results
        test_matrix_path: Path to test matrix JSON file
        
    Returns:
        Dictionary containing analysis results
    """
    # Load test matrix
    test_cases = load_test_matrix(test_matrix_path)
    
    # Initialize analysis structure
    analysis = {
        'by_model': {},
        'by_field': {},
        'by_quantization': {},
        'overall': {
            'total_tests': len(results),
            'successful_tests': 0,
            'failed_tests': 0,
            'average_processing_time': 0.0
        }
    }
    
    # Process results
    total_time = 0.0
    for result in results:
        # Update overall metrics
        if 'error' not in result:
            analysis['overall']['successful_tests'] += 1
            total_time += result.get('processing_time', 0.0)
        else:
            analysis['overall']['failed_tests'] += 1
            
        # Update model-specific metrics
        model_name = result['model_name']
        if model_name not in analysis['by_model']:
            analysis['by_model'][model_name] = {
                'total_tests': 0,
                'successful_tests': 0,
                'failed_tests': 0,
                'average_processing_time': 0.0
            }
        model_stats = analysis['by_model'][model_name]
        model_stats['total_tests'] += 1
        if 'error' not in result:
            model_stats['successful_tests'] += 1
            model_stats['average_processing_time'] += result.get('processing_time', 0.0)
        else:
            model_stats['failed_tests'] += 1
            
        # Update field-specific metrics
        field_type = result['field_type']
        if field_type not in analysis['by_field']:
            analysis['by_field'][field_type] = {
                'total_tests': 0,
                'successful_tests': 0,
                'failed_tests': 0,
                'average_processing_time': 0.0
            }
        field_stats = analysis['by_field'][field_type]
        field_stats['total_tests'] += 1
        if 'error' not in result:
            field_stats['successful_tests'] += 1
            field_stats['average_processing_time'] += result.get('processing_time', 0.0)
        else:
            field_stats['failed_tests'] += 1
            
        # Update quantization-specific metrics
        quant_level = result['quant_level']
        if quant_level not in analysis['by_quantization']:
            analysis['by_quantization'][quant_level] = {
                'total_tests': 0,
                'successful_tests': 0,
                'failed_tests': 0,
                'average_processing_time': 0.0
            }
        quant_stats = analysis['by_quantization'][quant_level]
        quant_stats['total_tests'] += 1
        if 'error' not in result:
            quant_stats['successful_tests'] += 1
            quant_stats['average_processing_time'] += result.get('processing_time', 0.0)
        else:
            quant_stats['failed_tests'] += 1
            
    # Calculate averages
    if analysis['overall']['successful_tests'] > 0:
        analysis['overall']['average_processing_time'] = total_time / analysis['overall']['successful_tests']
        
    for model_stats in analysis['by_model'].values():
        if model_stats['successful_tests'] > 0:
            model_stats['average_processing_time'] /= model_stats['successful_tests']
            
    for field_stats in analysis['by_field'].values():
        if field_stats['successful_tests'] > 0:
            field_stats['average_processing_time'] /= field_stats['successful_tests']
            
    for quant_stats in analysis['by_quantization'].values():
        if quant_stats['successful_tests'] > 0:
            quant_stats['average_processing_time'] /= quant_stats['successful_tests']
            
    return analysis

def main():
    """Main analysis function."""
    # Load all results
    results = load_all_results(paths['results'])
    if not results:
        logger.error("No results found to analyze")
        return
    
    # Calculate metrics by model and field
    metrics = aggregate_by_model_field(results)
    
    # Plot model comparison
    plot_model_comparison(metrics)
    
    # Analyze error distributions
    for field_type in ['work_order_number', 'total_cost']:
        error_dist = analyze_error_distribution(results, field_type)
        plot_error_distribution(error_dist, field_type)
    
    # Find best configurations
    best_wo_config = find_best_configuration(results, 'work_order_number')
    best_cost_config = find_best_configuration(results, 'total_cost')
    
    logger.info(f"Best configuration for work order numbers: {best_wo_config}")
    logger.info(f"Best configuration for total costs: {best_cost_config}")
    
    # Analyze prompt performance
    prompt_metrics = analyze_prompt_performance(results)
    plot_prompt_comparison(prompt_metrics)
    
    # Create heatmaps for each field type
    for field_type in ['work_order_number', 'total_cost']:
        for metric in ['accuracy', 'cer', 'time']:
            create_performance_heatmap(prompt_metrics, field_type, metric)
    
    # Find best prompt strategies
    best_wo_prompt = find_best_prompt_strategy(prompt_metrics, 'work_order_number')
    best_cost_prompt = find_best_prompt_strategy(prompt_metrics, 'total_cost')
    
    logger.info(f"Best prompt strategy for work order numbers: {best_wo_prompt}")
    logger.info(f"Best prompt strategy for total costs: {best_cost_prompt}")
    
    # Analyze quantization performance
    quant_metrics = analyze_quantization_performance(results)
    
    # Plot quantization heatmaps for best performing configurations
    for field_type, best_config in [('work_order_number', best_wo_prompt), 
                                   ('total_cost', best_cost_prompt)]:
        model = best_config['model']
        prompt = best_config['strategy']
        for metric in ['accuracy', 'cer', 'time']:
            plot_quantization_heatmap(quant_metrics, model, prompt, metric)

if __name__ == "__main__":
    main() 