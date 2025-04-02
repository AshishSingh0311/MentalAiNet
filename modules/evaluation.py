import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import time
from modules.visualization import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

class ModelEvaluator:
    """
    Evaluator for MH-Net models.
    
    This class handles the evaluation of MH-Net models, including various
    metrics, cross-validation, and performance visualization.
    """
    
    def __init__(
        self,
        model,
        metrics: Dict[str, bool] = None,
        threshold: float = 0.5,
        n_classes: Optional[int] = None
    ):
        """
        Initialize the ModelEvaluator.
        
        Args:
            model: The MH-Net model to evaluate
            metrics: Dictionary specifying which metrics to calculate
            threshold: Classification threshold for binary problems
            n_classes: Number of classes (for multi-class problems)
        """
        self.model = model
        self.threshold = threshold
        self.n_classes = n_classes
        
        # Default metrics to calculate
        self.metrics = metrics or {
            "accuracy": True,
            "precision": True,
            "recall": True,
            "f1_score": True,
            "roc_auc": True,
            "confusion_matrix": True
        }
    
    def _prepare_data(
        self,
        data: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Prepare data for evaluation.
        
        Args:
            data: Dictionary containing the multimodal data
            
        Returns:
            Tuple of (input_data, true_labels)
        """
        # Prepare inputs
        input_data = {}
        
        # Include each available modality
        if "text" in data:
            input_data["text_input"] = data["text"]
        
        if "audio" in data:
            input_data["audio_input"] = data["audio"]
        
        if "physiological" in data:
            input_data["physiological_input"] = data["physiological"]
        
        if "imaging" in data:
            input_data["imaging_input"] = data["imaging"]
        
        # Get true labels
        if "labels" in data:
            true_labels = data["labels"]
        else:
            raise ValueError("Labels are required for evaluation")
        
        return input_data, true_labels
    
    def _get_predictions(
        self,
        inputs: Dict[str, np.ndarray],
        true_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get model predictions.
        
        Args:
            inputs: Dictionary of input data
            true_labels: True class labels
            
        Returns:
            Tuple of (predicted_probabilities, predicted_labels)
        """
        # Get predicted probabilities
        pred_probs = self.model.predict(inputs)
        
        # Get predicted labels
        if pred_probs.shape[1] > 1:  # Multi-class
            pred_labels = np.argmax(pred_probs, axis=1)
        else:  # Binary
            pred_labels = (pred_probs > self.threshold).astype(int)
        
        return pred_probs, pred_labels
    
    def _calculate_metrics(
        self,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        pred_probs: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate evaluation metrics.
        
        Args:
            true_labels: True class labels
            pred_labels: Predicted class labels
            pred_probs: Predicted class probabilities
            
        Returns:
            Dictionary of calculated metrics
        """
        results = {}
        
        # Determine if this is a binary or multi-class problem
        n_classes = self.n_classes or len(np.unique(true_labels))
        is_binary = n_classes == 2
        
        # Basic metrics
        if self.metrics.get("accuracy", False):
            results["accuracy"] = accuracy_score(true_labels, pred_labels)
        
        if self.metrics.get("precision", False):
            if is_binary:
                results["precision"] = precision_score(true_labels, pred_labels, average="binary")
            else:
                results["precision"] = precision_score(true_labels, pred_labels, average="weighted")
        
        if self.metrics.get("recall", False):
            if is_binary:
                results["recall"] = recall_score(true_labels, pred_labels, average="binary")
            else:
                results["recall"] = recall_score(true_labels, pred_labels, average="weighted")
        
        if self.metrics.get("f1_score", False):
            if is_binary:
                results["f1_score"] = f1_score(true_labels, pred_labels, average="binary")
            else:
                results["f1_score"] = f1_score(true_labels, pred_labels, average="weighted")
        
        # ROC AUC (for binary or one-vs-rest for multi-class)
        if self.metrics.get("roc_auc", False):
            try:
                if is_binary:
                    results["roc_auc"] = roc_auc_score(true_labels, pred_probs[:, 1] if pred_probs.shape[1] > 1 else pred_probs)
                else:
                    # One-vs-rest ROC AUC
                    results["roc_auc"] = roc_auc_score(
                        tf.keras.utils.to_categorical(true_labels, num_classes=n_classes),
                        pred_probs,
                        average="weighted",
                        multi_class="ovr"
                    )
            except Exception as e:
                print(f"Warning: Could not calculate ROC AUC: {e}")
                results["roc_auc"] = np.nan
        
        # Confusion matrix
        if self.metrics.get("confusion_matrix", False):
            results["confusion_matrix"] = confusion_matrix(true_labels, pred_labels)
        
        # Store predictions and true labels for later visualization
        results["predictions"] = pred_probs
        results["true_labels"] = true_labels
        
        return results
    
    def _calculate_per_class_metrics(
        self,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        pred_probs: np.ndarray
    ) -> Dict[str, List[float]]:
        """
        Calculate per-class evaluation metrics.
        
        Args:
            true_labels: True class labels
            pred_labels: Predicted class labels
            pred_probs: Predicted class probabilities
            
        Returns:
            Dictionary of per-class metrics
        """
        n_classes = self.n_classes or len(np.unique(true_labels))
        
        # Convert true_labels to one-hot encoding for per-class metrics
        true_labels_one_hot = tf.keras.utils.to_categorical(true_labels, num_classes=n_classes)
        
        # Calculate per-class metrics
        per_class_metrics = {
            "precision": [],
            "recall": [],
            "f1_score": [],
            "roc_auc": []
        }
        
        for class_idx in range(n_classes):
            # Binary metrics for this class vs. rest
            true_binary = (true_labels == class_idx).astype(int)
            pred_binary = (pred_labels == class_idx).astype(int)
            prob_binary = pred_probs[:, class_idx] if pred_probs.shape[1] > 1 else pred_probs
            
            # Calculate metrics
            per_class_metrics["precision"].append(
                precision_score(true_binary, pred_binary, average="binary", zero_division=0)
            )
            
            per_class_metrics["recall"].append(
                recall_score(true_binary, pred_binary, average="binary", zero_division=0)
            )
            
            per_class_metrics["f1_score"].append(
                f1_score(true_binary, pred_binary, average="binary", zero_division=0)
            )
            
            try:
                per_class_metrics["roc_auc"].append(
                    roc_auc_score(true_binary, prob_binary)
                )
            except Exception:
                per_class_metrics["roc_auc"].append(np.nan)
        
        return per_class_metrics
    
    def evaluate(
        self,
        data: Dict[str, Any],
        return_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the model on the provided data.
        
        Args:
            data: Dictionary containing the evaluation data
            return_predictions: Whether to include predictions in the results
            
        Returns:
            Dictionary of evaluation results
        """
        # Prepare data
        inputs, true_labels = self._prepare_data(data)
        
        # Get predictions
        pred_probs, pred_labels = self._get_predictions(inputs, true_labels)
        
        # Calculate metrics
        results = self._calculate_metrics(true_labels, pred_labels, pred_probs)
        
        # Calculate per-class metrics
        class_metrics = self._calculate_per_class_metrics(true_labels, pred_labels, pred_probs)
        results["class_metrics"] = class_metrics
        
        # Classification report
        report = classification_report(true_labels, pred_labels, output_dict=True)
        results["classification_report"] = report
        
        # Remove predictions if not requested
        if not return_predictions:
            results.pop("predictions", None)
            results.pop("true_labels", None)
        
        return results
    
    def cross_validate(
        self,
        data: Dict[str, Any],
        n_folds: int = 5,
        stratified: bool = True,
        random_state: int = 42
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation on the provided data.
        
        Args:
            data: Dictionary containing the data for cross-validation
            n_folds: Number of folds
            stratified: Whether to use stratified folds
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary of cross-validation results
        """
        # Prepare data
        inputs, true_labels = self._prepare_data(data)
        
        # Determine number of classes
        n_classes = len(np.unique(true_labels))
        self.n_classes = n_classes
        
        # Create cross-validation splitter
        if stratified:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        # Initialize results dictionary
        cv_results = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "roc_auc": []
        }
        
        # Store per-class metrics across folds
        per_class_metrics = {
            "precision": [[] for _ in range(n_classes)],
            "recall": [[] for _ in range(n_classes)],
            "f1_score": [[] for _ in range(n_classes)],
            "roc_auc": [[] for _ in range(n_classes)]
        }
        
        # Split data into folds
        fold_idx = 1
        for train_idx, val_idx in cv.split(true_labels, true_labels):
            print(f"Evaluating fold {fold_idx}/{n_folds}...")
            
            # Prepare fold data
            fold_inputs = {k: v[val_idx] for k, v in inputs.items()}
            fold_true_labels = true_labels[val_idx]
            
            # Get predictions
            fold_pred_probs, fold_pred_labels = self._get_predictions(fold_inputs, fold_true_labels)
            
            # Calculate metrics
            fold_metrics = self._calculate_metrics(fold_true_labels, fold_pred_labels, fold_pred_probs)
            
            # Add to results
            for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
                if metric in fold_metrics:
                    cv_results[metric].append(fold_metrics[metric])
            
            # Calculate per-class metrics for this fold
            fold_class_metrics = self._calculate_per_class_metrics(
                fold_true_labels, fold_pred_labels, fold_pred_probs
            )
            
            # Add to per-class results
            for metric in ["precision", "recall", "f1_score", "roc_auc"]:
                for class_idx in range(n_classes):
                    per_class_metrics[metric][class_idx].append(fold_class_metrics[metric][class_idx])
            
            fold_idx += 1
        
        # Calculate average per-class metrics
        avg_per_class_metrics = {
            "precision": [np.mean(values) for values in per_class_metrics["precision"]],
            "recall": [np.mean(values) for values in per_class_metrics["recall"]],
            "f1_score": [np.mean(values) for values in per_class_metrics["f1_score"]],
            "roc_auc": [np.mean(values) for values in per_class_metrics["roc_auc"]]
        }
        
        # Add per-class metrics to results
        cv_results["class_metrics"] = avg_per_class_metrics
        
        # Print summary
        print("Cross-validation results:")
        for metric, values in cv_results.items():
            if metric != "class_metrics":
                print(f"  {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
        return cv_results
    
    def evaluate_fairness(
        self,
        data: Dict[str, Any],
        protected_attribute: str,
        protected_values: List[Any]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate fairness metrics across different protected attribute values.
        
        Args:
            data: Dictionary containing the evaluation data
            protected_attribute: Name of the protected attribute in the data
            protected_values: List of values of the protected attribute to evaluate
            
        Returns:
            Dictionary of fairness evaluation results
        """
        fairness_results = {}
        
        # Check if protected attribute exists in data
        if protected_attribute not in data:
            raise ValueError(f"Protected attribute '{protected_attribute}' not found in data")
        
        # Prepare overall data
        inputs, true_labels = self._prepare_data(data)
        
        # Evaluate for each protected value
        for value in protected_values:
            # Get indices for this protected value
            indices = np.where(data[protected_attribute] == value)[0]
            
            if len(indices) == 0:
                print(f"Warning: No samples found for protected value {value}")
                continue
            
            # Prepare subset of data
            value_inputs = {k: v[indices] for k, v in inputs.items()}
            value_true_labels = true_labels[indices]
            
            # Get predictions
            value_pred_probs, value_pred_labels = self._get_predictions(value_inputs, value_true_labels)
            
            # Calculate metrics
            value_results = self._calculate_metrics(value_true_labels, value_pred_labels, value_pred_probs)
            
            # Remove predictions and true labels to save memory
            value_results.pop("predictions", None)
            value_results.pop("true_labels", None)
            value_results.pop("confusion_matrix", None)
            
            # Store results
            fairness_results[str(value)] = value_results
        
        return fairness_results
    
    def feature_importance(
        self,
        data: Dict[str, Any],
        feature_names: Optional[Dict[str, List[str]]] = None,
        n_permutations: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Calculate feature importance using permutation importance.
        
        Args:
            data: Dictionary containing the evaluation data
            feature_names: Dictionary mapping modality names to feature names
            n_permutations: Number of permutations per feature
            
        Returns:
            Dictionary of feature importance scores
        """
        # Prepare data
        inputs, true_labels = self._prepare_data(data)
        
        # Get baseline performance
        baseline_pred_probs, baseline_pred_labels = self._get_predictions(inputs, true_labels)
        baseline_metrics = self._calculate_metrics(true_labels, baseline_pred_labels, baseline_pred_probs)
        baseline_performance = baseline_metrics.get("accuracy", 0.0)
        
        feature_importance_results = {}
        
        # Calculate importance for each modality
        for modality, X in inputs.items():
            print(f"Calculating feature importance for {modality}...")
            
            # Skip if feature is not a numpy array or has no features
            if not isinstance(X, np.ndarray) or X.size == 0:
                continue
            
            # Determine feature dimension
            if X.ndim == 2:
                n_features = X.shape[1]
            elif X.ndim == 3:
                n_features = X.shape[2]
            else:
                print(f"Skipping {modality}: Unsupported shape {X.shape}")
                continue
            
            # Get feature names if provided
            feat_names = feature_names.get(modality, [f"Feature_{i}" for i in range(n_features)]) if feature_names else [f"Feature_{i}" for i in range(n_features)]
            
            # Initialize importance scores
            importance_scores = np.zeros(n_features)
            
            # Calculate importance for each feature
            for feat_idx in range(n_features):
                # Initialize performance drop for this feature
                feat_performance_drop = 0.0
                
                # Permute the feature multiple times
                for _ in range(n_permutations):
                    # Create a copy of the inputs
                    permuted_inputs = {k: v.copy() for k, v in inputs.items()}
                    
                    # Permute the feature
                    if X.ndim == 2:
                        permuted_inputs[modality][:, feat_idx] = np.random.permutation(permuted_inputs[modality][:, feat_idx])
                    elif X.ndim == 3:
                        permuted_inputs[modality][:, :, feat_idx] = np.random.permutation(permuted_inputs[modality][:, :, feat_idx])
                    
                    # Get predictions with permuted feature
                    permuted_pred_probs, permuted_pred_labels = self._get_predictions(permuted_inputs, true_labels)
                    permuted_metrics = self._calculate_metrics(true_labels, permuted_pred_labels, permuted_pred_probs)
                    permuted_performance = permuted_metrics.get("accuracy", 0.0)
                    
                    # Calculate performance drop
                    feat_performance_drop += baseline_performance - permuted_performance
                
                # Average performance drop across permutations
                importance_scores[feat_idx] = feat_performance_drop / n_permutations
            
            # Normalize importance scores
            if np.sum(importance_scores) > 0:
                importance_scores = importance_scores / np.sum(importance_scores)
            
            # Store results
            feature_importance_results[modality] = {
                "feature_names": feat_names[:n_features],
                "importance_scores": importance_scores
            }
        
        return feature_importance_results
