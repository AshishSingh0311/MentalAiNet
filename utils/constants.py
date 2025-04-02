"""
Constants for the MH-Net framework.
This file contains constants used throughout the application including dataset configurations,
model parameters, and visualization settings.
"""

# Dataset constants
DATASET_OPTIONS = {
    "AVEC": "Audio/Visual Emotion Challenge dataset",
    "eRisk": "Early Risk Detection dataset",
    "DAIC-WOZ": "Distress Analysis Interview Corpus",
    "ADNI": "Alzheimer's Disease Neuroimaging Initiative",
    "Custom": "Custom dataset upload"
}

# Mental health conditions supported by the framework
MENTAL_HEALTH_CONDITIONS = [
    "Major Depressive Disorder (MDD)",
    "Generalized Anxiety Disorder (GAD)",
    "Bipolar Disorder",
    "Post-Traumatic Stress Disorder (PTSD)",
    "Schizophrenia",
    "Control (Healthy)"
]

# Default model parameters
DEFAULT_MODEL_PARAMS = {
    "embed_dim": 256,
    "num_heads": 8,
    "num_layers": 4,
    "dropout_rate": 0.1,
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 30,
    "early_stopping": True,
    "patience": 10
}

# Model architecture options
MODEL_ARCHITECTURES = [
    "MH-Net (Multimodal Transformer)",
    "Unimodal (Text Only)",
    "Unimodal (Audio Only)",
    "Unimodal (Physiological Only)",
    "Unimodal (Imaging Only)"
]

# Optimizer options
OPTIMIZER_OPTIONS = [
    "Adam",
    "SGD",
    "RMSprop",
    "AdamW"
]

# Loss function options
LOSS_FUNCTION_OPTIONS = [
    "Categorical Cross Entropy",
    "Binary Cross Entropy",
    "Focal Loss"
]

# Explainability methods
EXPLAINABILITY_METHODS = [
    "LIME",
    "SHAP",
    "Attention Visualization",
    "Integrated Gradients",
    "All"
]

# Audio feature extraction options
AUDIO_FEATURE_TYPES = [
    "MFCC",
    "Mel Spectrogram",
    "Chroma",
    "Spectral Contrast"
]

# Text preprocessing options
TEXT_PREPROCESSING_OPTIONS = {
    "lowercase": True,
    "remove_stopwords": True,
    "stemming": False,
    "lemmatization": True
}

# Audio preprocessing options
AUDIO_PREPROCESSING_OPTIONS = {
    "normalize": True,
    "noise_reduction": True,
    "feature_extraction": "MFCC"
}

# Physiological signal preprocessing options
PHYSIO_PREPROCESSING_OPTIONS = {
    "filter": True,
    "normalize": True,
    "artifact_removal": True
}

# Imaging preprocessing options
IMAGING_PREPROCESSING_OPTIONS = {
    "normalize": True,
    "registration": False,
    "skull_strip": False
}

# Evaluation metrics
EVALUATION_METRICS = {
    "accuracy": True,
    "precision": True,
    "recall": True,
    "f1_score": True,
    "roc_auc": True,
    "confusion_matrix": True
}

# Visualization settings
VISUALIZATION_COLORS = {
    "positive": "darkblue",
    "negative": "darkred",
    "neutral": "gray",
    "highlight": "red",
    "primary": "blue",
    "secondary": "green"
}

# File paths
DEFAULT_CHECKPOINT_DIR = "./checkpoints"
DEFAULT_LOGS_DIR = "./logs"
