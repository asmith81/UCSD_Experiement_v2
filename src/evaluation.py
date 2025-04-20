"""
Standardized evaluation implementation for field-specific analysis.
"""

from typing import Dict, Any, List, Optional, Union
import re
from decimal import Decimal
import numpy as np
from .config_protocol import ConfigManager
from .validation import ValidationManager

class FieldEvaluator:
    """Standardized field evaluation implementation."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        validation_manager: ValidationManager
    ):
        """Initialize field evaluator.
        
        Args:
            config_manager: Configuration manager instance
            validation_manager: Validation manager instance
        """
        self.config_manager = config_manager
        self.validation_manager = validation_manager
        self.config = self.config_manager.get_section("evaluation")
        
    def evaluate_field(
        self,
        field_type: str,
        predicted: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """Evaluate field extraction against ground truth.
        
        Args:
            field_type: Type of field ("work_order_number" or "total_cost")
            predicted: Predicted value
            ground_truth: Ground truth value
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Validate inputs
            self.validation_manager.validate_field_type(field_type)
            
            # Get field-specific evaluation
            if field_type == "work_order_number":
                return self._evaluate_work_order(predicted, ground_truth)
            elif field_type == "total_cost":
                return self._evaluate_cost(predicted, ground_truth)
            else:
                raise ValueError(f"Unsupported field type: {field_type}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate field: {str(e)}")
            
    def _evaluate_work_order(
        self,
        predicted: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """Evaluate work order number extraction."""
        # Clean strings
        pred = self._clean_work_order(predicted)
        gt = self._clean_work_order(ground_truth)
        
        # Calculate metrics
        raw_match = pred == gt
        normalized_match = self._normalize_work_order(pred) == self._normalize_work_order(gt)
        cer = self._calculate_cer(pred, gt)
        error_category = self._categorize_work_order_error(pred, gt)
        
        return {
            'raw_string_match': raw_match,
            'normalized_match': normalized_match,
            'cer': cer,
            'error_category': error_category
        }
        
    def _evaluate_cost(
        self,
        predicted: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """Evaluate total cost extraction."""
        # Clean strings
        pred = self._clean_cost(predicted)
        gt = self._clean_cost(ground_truth)
        
        # Calculate metrics
        raw_match = pred == gt
        normalized_match = self._normalize_cost(pred) == self._normalize_cost(gt)
        cer = self._calculate_cer(pred, gt)
        error_category = self._categorize_cost_error(pred, gt)
        
        return {
            'raw_string_match': raw_match,
            'normalized_match': normalized_match,
            'cer': cer,
            'error_category': error_category
        }
        
    def _clean_work_order(self, value: str) -> str:
        """Clean work order number string."""
        # Remove all non-alphanumeric characters
        return re.sub(r'[^a-zA-Z0-9]', '', value)
        
    def _normalize_work_order(self, value: str) -> str:
        """Normalize work order number string."""
        # Convert to uppercase and remove spaces
        return value.upper().strip()
        
    def _clean_cost(self, value: str) -> str:
        """Clean cost string."""
        # Remove currency symbols and extra spaces
        return re.sub(r'[^\d.]', '', value).strip()
        
    def _normalize_cost(self, value: str) -> float:
        """Normalize cost to float."""
        try:
            return float(Decimal(value))
        except:
            return 0.0
            
    def _calculate_cer(self, pred: str, gt: str) -> float:
        """Calculate Character Error Rate."""
        if not gt:
            return 1.0
            
        # Calculate Levenshtein distance
        if len(pred) < len(gt):
            return self._calculate_cer(gt, pred)
            
        if len(gt) == 0:
            return len(pred)
            
        previous_row = range(len(gt) + 1)
        for i, c1 in enumerate(pred):
            current_row = [i + 1]
            for j, c2 in enumerate(gt):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1] / len(gt)
        
    def _categorize_work_order_error(
        self,
        pred: str,
        gt: str
    ) -> str:
        """Categorize work order number error."""
        if not pred:
            return "missing_field"
            
        if not gt:
            return "invalid_ground_truth"
            
        if len(pred) < len(gt):
            return "missing_characters"
            
        if len(pred) > len(gt):
            return "extra_characters"
            
        if sorted(pred) == sorted(gt):
            return "transposition"
            
        return "wrong_characters"
        
    def _categorize_cost_error(
        self,
        pred: str,
        gt: str
    ) -> str:
        """Categorize cost error."""
        if not pred:
            return "missing_field"
            
        if not gt:
            return "invalid_ground_truth"
            
        try:
            pred_num = float(Decimal(pred))
            gt_num = float(Decimal(gt))
            
            if pred_num == gt_num:
                return "correct"
                
            if pred_num > gt_num:
                return "overestimation"
                
            return "underestimation"
            
        except:
            return "format_error" 