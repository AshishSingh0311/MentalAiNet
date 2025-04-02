"""
Helper functions for the MH-Net framework.
This file contains utility functions that support the MH-Net framework's operations
including data conversion, file handling, and text processing.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import re
import json
import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any

def create_directory_if_not_exists(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory to create
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def get_timestamp() -> str:
    """
    Get a timestamp string for file naming.
    
    Returns:
        String timestamp in format YYYYMMDD-HHMMSS
    """
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def save_dict_to_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save a dictionary to a JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save the JSON file
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    create_directory_if_not_exists(directory)
    
    # Convert numpy arrays to lists
    serializable_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            serializable_data[key] = value.tolist()
        elif isinstance(value, dict):
            serializable_data[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in value.items()}
        else:
            serializable_data[key] = value
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=4)
    
    print(f"Saved data to {filepath}")

def load_json_to_dict(filepath: str) -> Dict[str, Any]:
    """
    Load a dictionary from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary loaded from the JSON file
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data

def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{seconds:.2f}s"

def prepare_input_for_model(data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Prepare data dictionary for model input.
    
    Args:
        data: Dictionary containing multimodal data
        
    Returns:
        Dictionary formatted for model input
    """
    input_data = {}
    
    # Map modality keys to model input keys
    modality_mapping = {
        "text": "text_input",
        "audio": "audio_input",
        "physiological": "physiological_input",
        "imaging": "imaging_input"
    }
    
    # Prepare each modality
    for modality, input_key in modality_mapping.items():
        if modality in data:
            input_data[input_key] = data[modality]
    
    return input_data

def prepare_labels_for_model(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert labels to one-hot encoding for model training.
    
    Args:
        labels: Array of class labels
        num_classes: Number of classes
        
    Returns:
        One-hot encoded labels
    """
    # Ensure labels are integers
    labels = labels.astype(int)
    
    # Convert to one-hot encoding
    return to_categorical(labels, num_classes=num_classes)

def save_model_summary(model: tf.keras.Model, filepath: str) -> None:
    """
    Save a model's summary to a text file.
    
    Args:
        model: Keras model
        filepath: Path to save the summary
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    create_directory_if_not_exists(directory)
    
    # Get model summary
    model_summary = []
    
    def get_summary_as_list(s):
        model_summary.append(s)
    
    # Redirect summary to our list
    model.summary(print_fn=get_summary_as_list)
    
    # Save to file
    with open(filepath, 'w') as f:
        f.write('\n'.join(model_summary))
    
    print(f"Saved model summary to {filepath}")

def save_figure(fig: plt.Figure, filepath: str, dpi: int = 300) -> None:
    """
    Save a matplotlib figure to a file.
    
    Args:
        fig: Matplotlib figure
        filepath: Path to save the figure
        dpi: DPI for the saved figure
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    create_directory_if_not_exists(directory)
    
    # Save figure
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {filepath}")

def preprocess_text_sample(text: str, options: Dict[str, bool] = None) -> str:
    """
    Preprocess a single text sample.
    
    Args:
        text: Text to preprocess
        options: Dictionary of preprocessing options
        
    Returns:
        Preprocessed text
    """
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    import nltk
    
    # Download necessary NLTK data if not already downloaded
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    # Set default options if not provided
    if options is None:
        options = {
            "lowercase": True,
            "remove_stopwords": True,
            "stemming": False,
            "lemmatization": True
        }
    
    # Apply preprocessing steps
    if options.get("lowercase", True):
        text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords if specified
    if options.get("remove_stopwords", True):
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Apply stemming if specified
    if options.get("stemming", False):
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    
    # Apply lemmatization if specified
    if options.get("lemmatization", True):
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Reconstruct text
    preprocessed_text = " ".join(tokens)
    
    return preprocessed_text

def split_dataset_proportionally(
    data: Dict[str, Any],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    random_state: int = 42
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Split a dataset into train, validation, and test sets while maintaining proportions in each modality.
    
    Args:
        data: Dictionary containing multimodal data
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        stratify: Whether to stratify the split based on labels
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    from sklearn.model_selection import train_test_split
    
    # Check if labels are available
    if "labels" not in data:
        raise ValueError("Labels are required for splitting the dataset")
    
    # Extract labels for stratified splitting
    labels = data["labels"]
    
    # First, split into train and temp (val + test)
    train_indices, temp_indices = train_test_split(
        np.arange(len(labels)),
        test_size=(1 - train_ratio),
        stratify=labels if stratify else None,
        random_state=random_state
    )
    
    # Then split temp into val and test
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_ratio_adjusted),
        stratify=labels[temp_indices] if stratify else None,
        random_state=random_state
    )
    
    # Create the split datasets
    train_data = {}
    val_data = {}
    test_data = {}
    
    # Split each modality
    for key, value in data.items():
        if isinstance(value, (np.ndarray, list)):
            value = np.array(value)
            train_data[key] = value[train_indices]
            val_data[key] = value[val_indices]
            test_data[key] = value[test_indices]
    
    return train_data, val_data, test_data

def clean_html(text: str) -> str:
    """
    Clean HTML tags from text.
    
    Args:
        text: Text containing HTML tags
        
    Returns:
        Cleaned text
    """
    # Remove HTML tags
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def normalize_array(array: np.ndarray) -> np.ndarray:
    """
    Normalize an array to [0, 1] range.
    
    Args:
        array: Input array
        
    Returns:
        Normalized array
    """
    min_val = np.min(array)
    max_val = np.max(array)
    
    if max_val - min_val > 0:
        return (array - min_val) / (max_val - min_val)
    else:
        return array  # Return the original array if it's constant

def get_class_balance_ratio(labels: np.ndarray) -> float:
    """
    Calculate the class balance ratio (max count / min count).
    
    Args:
        labels: Array of class labels
        
    Returns:
        Class balance ratio
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    if len(unique_labels) <= 1:
        return 1.0
    
    min_count = np.min(counts)
    max_count = np.max(counts)
    
    if min_count == 0:
        return float('inf')
    
    return max_count / min_count

def get_metrics_from_history(history: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Extract the final metrics from training history.
    
    Args:
        history: Dictionary containing training history
        
    Returns:
        Dictionary of final metric values
    """
    final_metrics = {}
    
    for metric, values in history.items():
        if values:  # If the list is not empty
            final_metrics[metric] = values[-1]
    
    return final_metrics

def generate_report_summary(
    model_config: Dict[str, Any],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    dataset_info: Dict[str, Any],
    execution_time: float,
    timestamp: str
) -> str:
    """
    Generate a summary report of the model training and evaluation.
    
    Args:
        model_config: Model configuration
        train_metrics: Training metrics
        val_metrics: Validation metrics
        dataset_info: Dataset information
        execution_time: Execution time in seconds
        timestamp: Timestamp string
        
    Returns:
        Formatted report string
    """
    report = []
    
    # Add header
    report.append("=" * 80)
    report.append(f"MH-Net MODEL REPORT - {timestamp}")
    report.append("=" * 80)
    
    # Add model configuration
    report.append("\nMODEL CONFIGURATION:")
    report.append("-" * 80)
    for key, value in model_config.items():
        report.append(f"{key}: {value}")
    
    # Add dataset information
    report.append("\nDATASET INFORMATION:")
    report.append("-" * 80)
    for key, value in dataset_info.items():
        report.append(f"{key}: {value}")
    
    # Add training metrics
    report.append("\nTRAINING METRICS:")
    report.append("-" * 80)
    for metric, value in train_metrics.items():
        report.append(f"{metric}: {value:.4f}")
    
    # Add validation metrics
    report.append("\nVALIDATION METRICS:")
    report.append("-" * 80)
    for metric, value in val_metrics.items():
        report.append(f"{metric}: {value:.4f}")
    
    # Add execution information
    report.append("\nEXECUTION INFORMATION:")
    report.append("-" * 80)
    report.append(f"Execution time: {format_time(execution_time)}")
    report.append(f"Timestamp: {timestamp}")
    
    # Add footer
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)
