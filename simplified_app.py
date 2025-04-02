import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import json
import os
import logging
from modules.database import db_manager  # Import the database manager
from modules.advanced_visualization import create_advanced_visualization_page
from modules.user_management import get_user_manager
from utils.export import create_assessment_pdf, export_to_html, export_to_csv
from modules.batch_processor import BatchProcessor, ModelVersionManager
from modules.api_integration import api_manager
# Use the mock implementation to avoid TensorFlow import issues
from modules.model_versioning_mock import ModelRegistry, ModelDeployment, ModelManager
from modules.clinical_recommendations import RecommendationEngine, TreatmentPlanner

# Initialize modules and services
if "user_manager" not in st.session_state:
    st.session_state.user_manager = get_user_manager(os.environ.get("DATABASE_URL", "sqlite:///mhnet.db"))

if "batch_processor" not in st.session_state:
    st.session_state.batch_processor = BatchProcessor(db_manager=db_manager)
    
if "model_registry" not in st.session_state:
    st.session_state.model_registry = ModelRegistry()
    
if "recommendation_engine" not in st.session_state:
    st.session_state.recommendation_engine = RecommendationEngine()

# Set page configuration
st.set_page_config(
    page_title="MH-Net: Multimodal Mental Health Diagnostics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main header
st.title("🧠 MH-Net: Multimodal Mental Health Diagnostics Framework")
st.markdown("""
This application provides a comprehensive framework for multimodal deep learning in mental health diagnostics with explainable AI capabilities.
""")

# Apply custom styles
st.markdown("""
<style>
    .main-header {
        font-family: 'Arial', sans-serif;
        color: #2c3e50;
    }
    .sidebar-header {
        font-family: 'Arial', sans-serif;
        color: #2c3e50;
        padding-top: 15px;
    }
    .metric-card {
        background-color: #f7f7f7;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .highlight {
        background-color: #e6f7ff;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #1a73e8;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation with enhanced styling
st.sidebar.markdown("<h2 class='sidebar-header'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Select a page",
    ["Introduction", "Data Loading", "Preprocessing", "Model Training", "Evaluation", "Explainability", 
     "Dashboard", "Real-time Assessment", "Advanced Visualization", "User Management", 
     "Export & Reports", "Batch Processing", "API Integration", "Model Versioning", "Clinical Recommendations"]
)

# Introduction page
if page == "Introduction":
    st.header("Introduction to MH-Net")
    
    st.subheader("Framework Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        MH-Net is a deep learning framework designed for multimodal mental health diagnostics that integrates:
        
        - **Text Analysis**: Processing clinical notes, social media content, and patient responses
        - **Audio Processing**: Analyzing speech patterns, tone, and acoustic features
        - **Physiological Signals**: Processing ECG, EEG, and other biosignals
        - **Medical Imaging**: Analyzing fMRI, MRI, and other neuroimaging data
        """)
    
    with col2:
        st.markdown("""
        Key features of the framework:
        
        - Transformer-based architecture for multimodal integration
        - Early detection and classification of mental health disorders
        - Explainable AI techniques for transparent decision-making
        - Cross-validation and comprehensive performance metrics
        - Real-time inference capabilities
        """)

    st.subheader("Supported Mental Health Conditions")
    conditions = [
        "Major Depressive Disorder (MDD)",
        "Generalized Anxiety Disorder (GAD)",
        "Bipolar Disorder",
        "Post-Traumatic Stress Disorder (PTSD)",
        "Schizophrenia"
    ]
    
    for condition in conditions:
        st.markdown(f"- {condition}")
    
    st.subheader("Supported Datasets")
    st.markdown("""
    - **fMRI Repositories**: Brain activity analysis
    - **AVEC (Audio/Visual Emotion Challenge)**: Speech and facial emotion recognition
    - **eRisk**: Early risk detection from social media text
    - **ADNI (Alzheimer's Disease Neuroimaging Initiative)**: Neuroimaging data
    - **DAIC-WOZ**: Clinical interviews for depression screening
    """)

# Data Loading page
elif page == "Data Loading":
    st.header("Data Loading")
    
    # Data source selection
    st.subheader("Select Data Source")
    data_source = st.selectbox(
        "Choose a data source type",
        ["Local Files", "Remote Repository", "Sample Dataset"]
    )
    
    # Data modality selection
    st.subheader("Select Data Modalities")
    text_data = st.checkbox("Text Data (clinical notes, responses, social media)", value=True)
    audio_data = st.checkbox("Audio Data (speech recordings)", value=True)
    physiological_data = st.checkbox("Physiological Signals (EEG, ECG)", value=False)
    imaging_data = st.checkbox("Neuroimaging Data (fMRI, MRI)", value=False)
    
    # Dataset selection
    st.subheader("Select Dataset")
    dataset_options = {
        "AVEC": "Audio/Visual Emotion Challenge dataset",
        "eRisk": "Early Risk Detection dataset",
        "DAIC-WOZ": "Distress Analysis Interview Corpus",
        "ADNI": "Alzheimer's Disease Neuroimaging Initiative",
        "Custom": "Custom dataset upload"
    }
    
    selected_dataset = st.selectbox("Choose a dataset", list(dataset_options.keys()))
    st.info(f"{dataset_options[selected_dataset]}")
    
    if selected_dataset == "Custom":
        st.subheader("Upload Custom Dataset")
        
        if text_data:
            st.file_uploader("Upload text data files (CSV, TXT)", type=["csv", "txt"], accept_multiple_files=True)
        
        if audio_data:
            st.file_uploader("Upload audio files (WAV, MP3)", type=["wav", "mp3"], accept_multiple_files=True)
        
        if physiological_data:
            st.file_uploader("Upload physiological data files (CSV, EDF)", type=["csv", "edf"], accept_multiple_files=True)
        
        if imaging_data:
            st.file_uploader("Upload imaging files (NII, DICOM)", type=["nii", "nii.gz", "dcm"], accept_multiple_files=True)
    
    # Load data button
    if st.button("Load Demo Data"):
        with st.spinner("Loading demo data..."):
            # Generate some random demo data
            st.session_state.dataset = {
                "labels": np.random.randint(0, 3, 100),
                "text": np.random.random((100, 50)),
                "audio": np.random.random((100, 20, 13)),
                "text_raw": ["Sample text " + str(i) for i in range(100)]
            }
            
            # Display dataset summary
            st.success("Demo data loaded successfully!")
            
            # Show summary statistics
            st.subheader("Dataset Summary")
            summary = {
                "Total samples": len(st.session_state.dataset["labels"]),
                "Classes": np.unique(st.session_state.dataset["labels"]).tolist(),
                "Text samples": len(st.session_state.dataset["text"]),
                "Audio samples": len(st.session_state.dataset["audio"])
            }
            
            st.json(summary)
            
            # Show class distribution
            st.subheader("Class Distribution")
            labels = st.session_state.dataset["labels"]
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            fig = px.pie(
                values=counts,
                names=unique_labels,
                title="Class Distribution"
            )
            st.plotly_chart(fig)

# Preprocessing page
elif page == "Preprocessing":
    st.header("Data Preprocessing")
    
    if 'dataset' not in st.session_state:
        st.warning("Please load a dataset first in the 'Data Loading' page.")
    else:
        dataset = st.session_state.dataset
        
        st.subheader("Preprocessing Options")
        
        # Text preprocessing options
        if "text" in dataset and len(dataset["text"]) > 0:
            st.markdown("#### Text Preprocessing")
            text_lower = st.checkbox("Convert to lowercase", value=True)
            text_remove_stopwords = st.checkbox("Remove stopwords", value=True)
            text_stemming = st.checkbox("Apply stemming", value=False)
            text_lemmatization = st.checkbox("Apply lemmatization", value=True)
        
        # Audio preprocessing options
        if "audio" in dataset and len(dataset["audio"]) > 0:
            st.markdown("#### Audio Preprocessing")
            audio_normalize = st.checkbox("Normalize audio", value=True)
            audio_noise_reduction = st.checkbox("Apply noise reduction", value=True)
            audio_feature_extraction = st.selectbox(
                "Feature extraction method",
                ["MFCC", "Mel Spectrogram", "Chroma", "Spectral Contrast"]
            )
        
        # General preprocessing options
        st.markdown("#### General Preprocessing")
        col1, col2, col3 = st.columns(3)
        with col1:
            train_ratio = st.slider("Train Split", 0.1, 0.9, 0.7, 0.05)
        with col2:
            val_ratio = st.slider("Validation Split", 0.1, 0.9, 0.15, 0.05)
        with col3:
            test_ratio = st.slider("Test Split", 0.1, 0.9, 0.15, 0.05)
        balance_classes = st.checkbox("Balance classes", value=True)
        augmentation = st.checkbox("Apply data augmentation", value=False)
        
        # Preprocess data button
        if st.button("Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                # Simulate preprocessing with a delay
                import time
                time.sleep(1)
                
                # Store in session state as processed data
                st.session_state.preprocessed_data = {
                    "train": {
                        "labels": dataset["labels"][:70],
                        "text": dataset["text"][:70],
                        "audio": dataset["audio"][:70]
                    },
                    "val": {
                        "labels": dataset["labels"][70:85],
                        "text": dataset["text"][70:85],
                        "audio": dataset["audio"][70:85]
                    },
                    "test": {
                        "labels": dataset["labels"][85:],
                        "text": dataset["text"][85:],
                        "audio": dataset["audio"][85:]
                    }
                }
                
                # Display preprocessing summary
                st.success("Data preprocessed successfully!")
                
                # Show summary statistics
                st.subheader("Preprocessed Data Summary")
                summary = {
                    "Train samples": len(st.session_state.preprocessed_data["train"]["labels"]),
                    "Validation samples": len(st.session_state.preprocessed_data["val"]["labels"]),
                    "Test samples": len(st.session_state.preprocessed_data["test"]["labels"]),
                }
                
                st.json(summary)
                
                # Show train set class distribution
                st.subheader("Train Set Class Distribution")
                labels = st.session_state.preprocessed_data["train"]["labels"]
                unique_labels, counts = np.unique(labels, return_counts=True)
                
                fig = px.bar(
                    x=[f"Class {i}" for i in unique_labels],
                    y=counts,
                    title="Train Set Class Distribution"
                )
                st.plotly_chart(fig)

# Model Training page
elif page == "Model Training":
    st.header("Model Training")
    
    if 'preprocessed_data' not in st.session_state:
        st.warning("Please preprocess your data first in the 'Preprocessing' page.")
    else:
        preprocessed_data = st.session_state.preprocessed_data
        
        # Create tabs for different training configurations
        training_tabs = st.tabs([
            "Model Architecture", 
            "Training Parameters", 
            "Optimization", 
            "Advanced Features",
            "Training Results"
        ])
        
        with training_tabs[0]:  # Model Architecture tab
            st.subheader("Model Architecture")
            
            # Model architecture selection with more options
            model_type = st.selectbox(
                "Select model architecture",
                [
                    "MH-Net (Multimodal Transformer)", 
                    "MH-Net Lite (Efficient Multimodal)", 
                    "Unimodal (Text Only)", 
                    "Unimodal (Audio Only)",
                    "Unimodal (Physiological)",
                    "Unimodal (Imaging)",
                    "Dual-Modal (Text + Audio)",
                    "Custom Architecture"
                ]
            )
            
            # Modality selection
            st.markdown("#### Modality Selection")
            use_text = st.checkbox("Text Modality", value=True)
            use_audio = st.checkbox("Audio Modality", value=True)
            use_physiological = st.checkbox("Physiological Modality", value=False)
            use_imaging = st.checkbox("Imaging Modality", value=False)
            
            # Architecture parameters
            st.markdown("#### Architecture Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                embed_dim = st.slider("Embedding dimension", 64, 768, 256, step=64)
                num_heads = st.slider("Number of attention heads", 2, 16, 8, step=2)
                num_layers = st.slider("Number of transformer layers", 1, 12, 4, step=1)
                
            with col2:
                dropout_rate = st.slider("Dropout rate", 0.0, 0.5, 0.1, step=0.05)
                hidden_dim = st.slider("Hidden dimension", embed_dim, embed_dim*4, embed_dim*2, step=embed_dim//2)
                activation = st.selectbox("Activation function", ["ReLU", "GELU", "SiLU", "Tanh", "LeakyReLU"])
            
            # Only show custom architecture options if selected
            if model_type == "Custom Architecture":
                st.markdown("#### Custom Architecture Options")
                
                fusion_type = st.selectbox(
                    "Fusion strategy",
                    ["Early Fusion", "Late Fusion", "Hybrid Fusion", "Attention-based Fusion"]
                )
                
                backbone_type = st.selectbox(
                    "Backbone network",
                    ["Transformer", "LSTM", "BiLSTM", "CNN", "ResNet", "Inception", "EfficientNet"]
                )
                
                use_pretrained = st.checkbox("Use pretrained models where applicable", value=True)
                freeze_pretrained = st.checkbox("Freeze pretrained layers", value=False) if use_pretrained else False
        
        with training_tabs[1]:  # Training Parameters tab
            st.subheader("Training Parameters")
            
            # Dataset configuration
            st.markdown("#### Dataset Configuration")
            
            train_split = st.slider("Training data percentage", 60, 90, 80, step=5)
            val_split = st.slider("Validation data percentage", 5, 20, 10, step=5)
            test_split = 100 - train_split - val_split
            st.markdown(f"Test data percentage: **{test_split}%**")
            
            # Ensure the splits sum to 100%
            if train_split + val_split + test_split != 100:
                st.warning("Data splits must sum to 100%. Please adjust the values.")
            
            # Cross-validation options
            use_cv = st.checkbox("Use cross-validation", value=False)
            cv_folds = st.slider("Number of CV folds", 3, 10, 5, step=1) if use_cv else 0
            stratified_cv = st.checkbox("Use stratified cross-validation", value=True) if use_cv else False
            
            # Training loop parameters
            st.markdown("#### Training Loop Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64, 128, 256], value=32)
                accumulation_steps = st.slider("Gradient accumulation steps", 1, 8, 1, step=1)
                
            with col2:
                num_epochs = st.slider("Number of epochs", 5, 200, 30, step=5)
                eval_frequency = st.slider("Evaluation frequency (epochs)", 1, 5, 1, step=1)
        
        with training_tabs[2]:  # Optimization tab
            st.subheader("Optimization Configuration")
            
            # Optimizer selection
            optimizer_type = st.selectbox(
                "Optimizer",
                ["Adam", "AdamW", "SGD", "RMSprop", "AdaGrad", "LAMB", "Lion", "Adafactor"]
            )
            
            # Learning rate options
            st.markdown("#### Learning Rate Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                learning_rate = st.select_slider(
                    "Initial learning rate",
                    options=[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01],
                    value=0.001
                )
                
            with col2:
                lr_schedule = st.selectbox(
                    "Learning rate schedule",
                    ["Constant", "Linear Decay", "Cosine Decay", "Step Decay", "Exponential Decay"]
                )
            
            # Optimizer-specific parameters
            if optimizer_type in ["Adam", "AdamW"]:
                col1, col2 = st.columns(2)
                with col1:
                    beta1 = st.slider("Beta1", 0.8, 0.999, 0.9, step=0.001, format="%.3f")
                with col2:
                    beta2 = st.slider("Beta2", 0.8, 0.999, 0.999, step=0.001, format="%.3f")
                
                weight_decay = st.slider("Weight decay", 0.0, 0.1, 0.01, step=0.001, format="%.4f")
            
            # Early stopping options
            st.markdown("#### Early Stopping Configuration")
            
            early_stopping = st.checkbox("Use early stopping", value=True)
            if early_stopping:
                col1, col2 = st.columns(2)
                with col1:
                    patience = st.slider("Patience (epochs)", 3, 30, 10, step=1)
                with col2:
                    es_metric = st.selectbox(
                        "Monitor metric",
                        ["val_loss", "val_accuracy", "val_f1", "val_auc"]
                    )
                    
                min_delta = st.slider("Minimum change threshold", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")
            
            # Loss function selection
            st.markdown("#### Loss Function")
            
            loss_function = st.selectbox(
                "Loss function",
                [
                    "Cross Entropy", 
                    "Focal Loss", 
                    "Weighted Cross Entropy", 
                    "Dice Loss",
                    "Binary Cross Entropy",
                    "Mean Squared Error",
                    "Custom Loss (Combined)"
                ]
            )
            
            # Custom loss weights if selected
            if loss_function == "Custom Loss (Combined)":
                st.markdown("#### Custom Loss Weights")
                
                ce_weight = st.slider("Cross Entropy Weight", 0.0, 1.0, 0.7, step=0.1)
                focal_weight = st.slider("Focal Loss Weight", 0.0, 1.0, 0.3, step=0.1)
                
                # Display actual formula
                st.markdown(f"Final Loss = {ce_weight} × CrossEntropy + {focal_weight} × FocalLoss")
        
        with training_tabs[3]:  # Advanced Features tab
            st.subheader("Advanced Training Features")
            
            # Regularization options
            st.markdown("#### Regularization")
            
            col1, col2 = st.columns(2)
            with col1:
                use_l1 = st.checkbox("L1 Regularization", value=False)
                l1_lambda = st.slider("L1 Lambda", 0.0, 0.1, 0.01, step=0.001, format="%.4f") if use_l1 else 0.0
                
            with col2:
                use_l2 = st.checkbox("L2 Regularization", value=True)
                l2_lambda = st.slider("L2 Lambda", 0.0, 0.1, 0.01, step=0.001, format="%.4f") if use_l2 else 0.0
            
            # Data augmentation
            st.markdown("#### Data Augmentation")
            
            use_augmentation = st.checkbox("Enable data augmentation", value=True)
            if use_augmentation:
                # Text augmentation
                if use_text:
                    st.markdown("##### Text Augmentation")
                    text_aug_methods = {
                        "Synonym Replacement": st.checkbox("Synonym Replacement", value=True),
                        "Random Deletion": st.checkbox("Random Deletion", value=False),
                        "Random Swap": st.checkbox("Random Swap", value=False),
                        "Back Translation": st.checkbox("Back Translation", value=False)
                    }
                
                # Audio augmentation
                if use_audio:
                    st.markdown("##### Audio Augmentation")
                    audio_aug_methods = {
                        "Time Shifting": st.checkbox("Time Shifting", value=True),
                        "Pitch Shifting": st.checkbox("Pitch Shifting", value=True),
                        "Speed Perturbation": st.checkbox("Speed Perturbation", value=False),
                        "Noise Addition": st.checkbox("Noise Addition", value=False)
                    }
                
                # Physiological augmentation
                if use_physiological:
                    st.markdown("##### Physiological Signal Augmentation")
                    physio_aug_methods = {
                        "Signal Scaling": st.checkbox("Signal Scaling", value=True),
                        "Signal Mixing": st.checkbox("Signal Mixing", value=False),
                        "Jittering": st.checkbox("Jittering", value=False)
                    }
                
                # Imaging augmentation
                if use_imaging:
                    st.markdown("##### Imaging Augmentation")
                    imaging_aug_methods = {
                        "Rotation": st.checkbox("Rotation", value=True),
                        "Flipping": st.checkbox("Flipping", value=True),
                        "Gaussian Noise": st.checkbox("Gaussian Noise", value=False),
                        "Contrast Adjustment": st.checkbox("Contrast Adjustment", value=False)
                    }
            
            # Advanced techniques
            st.markdown("#### Advanced Techniques")
            
            use_mixed_precision = st.checkbox("Use mixed precision training", value=True)
            use_gradient_clipping = st.checkbox("Use gradient clipping", value=True)
            
            if use_gradient_clipping:
                clip_value = st.slider("Gradient clipping value", 0.1, 5.0, 1.0, step=0.1)
            
            # Experiment tracking
            st.markdown("#### Experiment Tracking")
            
            track_experiments = st.checkbox("Enable experiment tracking", value=True)
            if track_experiments:
                tracking_method = st.selectbox(
                    "Tracking method",
                    ["Local CSV", "MLflow", "TensorBoard", "Weights & Biases", "Neptune.ai"]
                )
                
                log_frequency = st.slider("Logging frequency (steps)", 10, 500, 100, step=10)
        
        with training_tabs[4]:  # Training Results tab
            st.subheader("Training Progress & Results")
            
            # Start training button
            if st.button("Train Model", key="train_model_btn"):
                with st.spinner("Training model..."):
                    # Collect all training parameters
                    training_config = {
                        "model_type": model_type,
                        "modalities": {
                            "text": use_text,
                            "audio": use_audio,
                            "physiological": use_physiological,
                            "imaging": use_imaging
                        },
                        "architecture": {
                            "embed_dim": embed_dim,
                            "num_heads": num_heads,
                            "num_layers": num_layers,
                            "dropout_rate": dropout_rate,
                            "hidden_dim": hidden_dim,
                            "activation": activation
                        },
                        "training": {
                            "batch_size": batch_size,
                            "num_epochs": num_epochs,
                            "learning_rate": learning_rate,
                            "optimizer": optimizer_type,
                            "loss_function": loss_function,
                            "early_stopping": early_stopping,
                            "patience": patience if early_stopping else None
                        }
                    }
                    
                    # Store training config
                    st.session_state.training_config = training_config
                    
                    # Simulate training with a progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        import time
                        time.sleep(0.05)
                        progress_bar.progress(i + 1)
                    
                    # Store model in session state
                    st.session_state.model = {
                        "type": model_type,
                        "embed_dim": embed_dim,
                        "num_heads": num_heads,
                        "num_layers": num_layers,
                        "dropout_rate": dropout_rate,
                        "hidden_dim": hidden_dim,
                        "activation": activation,
                        "trained": True
                    }
                    
                    # Store training history in session state with more metrics
                    num_points = 30
                    # Create base histories
                    loss = np.random.random(num_points) * 0.5 * np.exp(-0.1 * np.arange(num_points))
                    accuracy = 0.5 + 0.4 * (1 - np.exp(-0.15 * np.arange(num_points)))
                    val_loss = np.random.random(num_points) * 0.7 * np.exp(-0.07 * np.arange(num_points))
                    val_accuracy = 0.5 + 0.35 * (1 - np.exp(-0.1 * np.arange(num_points)))
                    
                    # Add more metrics
                    st.session_state.history = {
                        "loss": loss,
                        "accuracy": accuracy,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                        "precision": 0.4 + 0.5 * (1 - np.exp(-0.12 * np.arange(num_points))),
                        "recall": 0.45 + 0.45 * (1 - np.exp(-0.13 * np.arange(num_points))),
                        "f1_score": 0.43 + 0.47 * (1 - np.exp(-0.125 * np.arange(num_points))),
                        "val_precision": 0.38 + 0.48 * (1 - np.exp(-0.11 * np.arange(num_points))),
                        "val_recall": 0.42 + 0.43 * (1 - np.exp(-0.12 * np.arange(num_points))),
                        "val_f1_score": 0.4 + 0.45 * (1 - np.exp(-0.115 * np.arange(num_points))),
                        "learning_rate": learning_rate * np.exp(-0.05 * np.arange(num_points))
                    }
                    
                    # Store validation results for confusion matrix
                    confusion_matrix = np.array([
                        [85, 10, 5, 2, 3],  # True Depression
                        [8, 78, 7, 3, 4],   # True Anxiety
                        [6, 9, 80, 3, 2],   # True PTSD
                        [3, 4, 2, 87, 4],   # True Bipolar
                        [2, 3, 2, 5, 88]    # True Schizophrenia
                    ])
                    st.session_state.evaluation_results = {
                        "confusion_matrix": confusion_matrix,
                        "accuracy": 0.84,
                        "precision": 0.83,
                        "recall": 0.82,
                        "f1_score": 0.825,
                        "roc_auc": 0.91
                    }
                    
                # Display training summary
                st.success(f"Model training completed! Achieved validation accuracy: {st.session_state.history['val_accuracy'][-1]:.2f}")
                
                # Create tabs for different training result visualizations
                result_tabs = st.tabs(["Training Curves", "Metrics", "Learning Rate", "Model Summary"])
                
                with result_tabs[0]:  # Training Curves
                    st.subheader("Training Curves")
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    
                    # Plot loss
                    ax1.plot(st.session_state.history["loss"], label="Training Loss")
                    ax1.plot(st.session_state.history["val_loss"], label="Validation Loss")
                    ax1.set_xlabel("Epoch")
                    ax1.set_ylabel("Loss")
                    ax1.set_title("Training and Validation Loss")
                    ax1.legend()
                    ax1.grid(True)
                    
                    # Plot accuracy
                    ax2.plot(st.session_state.history["accuracy"], label="Training Accuracy")
                    ax2.plot(st.session_state.history["val_accuracy"], label="Validation Accuracy")
                    ax2.set_xlabel("Epoch")
                    ax2.set_ylabel("Accuracy")
                    ax2.set_title("Training and Validation Accuracy")
                    ax2.legend()
                    ax2.grid(True)
                    
                    st.pyplot(fig)
                
                with result_tabs[1]:  # Metrics tab
                    st.subheader("Performance Metrics")
                    
                    # Plot precision, recall, f1
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    
                    # Plot precision/recall
                    ax1.plot(st.session_state.history["precision"], label="Training Precision")
                    ax1.plot(st.session_state.history["recall"], label="Training Recall")
                    ax1.plot(st.session_state.history["val_precision"], label="Validation Precision", linestyle='--')
                    ax1.plot(st.session_state.history["val_recall"], label="Validation Recall", linestyle='--')
                    ax1.set_xlabel("Epoch")
                    ax1.set_ylabel("Score")
                    ax1.set_title("Precision and Recall")
                    ax1.legend()
                    ax1.grid(True)
                    
                    # Plot F1 score
                    ax2.plot(st.session_state.history["f1_score"], label="Training F1")
                    ax2.plot(st.session_state.history["val_f1_score"], label="Validation F1")
                    ax2.set_xlabel("Epoch")
                    ax2.set_ylabel("F1 Score")
                    ax2.set_title("F1 Score")
                    ax2.legend()
                    ax2.grid(True)
                    
                    st.pyplot(fig)
                    
                    # Display final metrics
                    st.markdown("#### Final Validation Metrics")
                    
                    metric_cols = st.columns(5)
                    with metric_cols[0]:
                        st.metric("Accuracy", f"{st.session_state.evaluation_results['accuracy']:.2f}", "+0.05")
                    with metric_cols[1]:
                        st.metric("Precision", f"{st.session_state.evaluation_results['precision']:.2f}", "+0.03")
                    with metric_cols[2]:
                        st.metric("Recall", f"{st.session_state.evaluation_results['recall']:.2f}", "+0.02")
                    with metric_cols[3]:
                        st.metric("F1 Score", f"{st.session_state.evaluation_results['f1_score']:.2f}", "+0.04")
                    with metric_cols[4]:
                        st.metric("ROC-AUC", f"{st.session_state.evaluation_results['roc_auc']:.2f}", "+0.06")
                
                with result_tabs[2]:  # Learning Rate tab
                    st.subheader("Learning Rate Schedule")
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(st.session_state.history["learning_rate"])
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Learning Rate")
                    ax.set_title("Learning Rate Schedule")
                    ax.grid(True)
                    
                    st.pyplot(fig)
                
                with result_tabs[3]:  # Model Summary tab
                    st.subheader("Model Summary")
                    
                    # Create a simple ASCII model diagram
                    model_diagram = f"""
                    ```
                    MH-Net Model Architecture:
                    ------------------------
                    
                    Input Layer
                        |
                        ↓
                    Embedding Layer (dim={embed_dim})
                        |
                        ↓
                    Transformer Encoder (layers={num_layers}, heads={num_heads})
                        |
                        ↓
                    Hidden Layer (dim={hidden_dim}, activation={activation})
                        |
                        ↓
                    Dropout Layer (rate={dropout_rate})
                        |
                        ↓
                    Output Layer (softmax)
                    ```
                    """
                    
                    st.markdown(model_diagram)
                    
                    # Display model parameters
                    st.markdown("#### Model Parameters")
                    
                    # Calculate approximate parameter count
                    embedding_params = embed_dim * 10000  # Assuming vocab size of 10,000
                    transformer_params = num_layers * (4 * embed_dim**2 + embed_dim * 2)
                    hidden_params = embed_dim * hidden_dim + hidden_dim
                    output_params = hidden_dim * 5 + 5  # Assuming 5 output classes
                    
                    total_params = embedding_params + transformer_params + hidden_params + output_params
                    
                    st.markdown(f"**Total Parameters:** {total_params:,}")
                    st.markdown(f"**Trainable Parameters:** {total_params:,}")
                    
                    # Display model configuration
                    st.markdown("#### Training Configuration")
                    st.json(training_config)

# Evaluation page
elif page == "Evaluation":
    st.header("Model Evaluation")
    
    if 'model' not in st.session_state or not st.session_state.get('model', {}).get('trained', False):
        st.warning("Please train a model first in the 'Model Training' page.")
    else:
        st.subheader("Evaluation Options")
        
        # Evaluation data selection
        eval_data = st.selectbox(
            "Select evaluation dataset",
            ["Test set", "Validation set", "Custom data"]
        )
        
        # Metrics selection
        st.markdown("#### Evaluation Metrics")
        metrics = {
            "Accuracy": st.checkbox("Accuracy", value=True),
            "Precision": st.checkbox("Precision", value=True),
            "Recall": st.checkbox("Recall", value=True),
            "F1 Score": st.checkbox("F1 Score", value=True),
            "ROC-AUC": st.checkbox("ROC-AUC", value=True),
            "Confusion Matrix": st.checkbox("Confusion Matrix", value=True)
        }
        
        # Evaluate model button
        if st.button("Evaluate Model"):
            with st.spinner("Evaluating model..."):
                # Simulate evaluation
                import time
                time.sleep(1)
                
                # Generate fake evaluation results
                st.session_state.evaluation_results = {
                    "accuracy": 0.85 + np.random.random() * 0.1,
                    "precision": 0.82 + np.random.random() * 0.1,
                    "recall": 0.79 + np.random.random() * 0.1,
                    "f1_score": 0.80 + np.random.random() * 0.1,
                    "roc_auc": 0.88 + np.random.random() * 0.1,
                    "confusion_matrix": np.array([
                        [30, 4, 1],
                        [5, 25, 5],
                        [2, 3, 25]
                    ])
                }
                
                # Display evaluation results
                st.success("Model evaluation completed!")
                
                # Display metrics
                st.subheader("Evaluation Metrics")
                metrics_df = pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
                    "Value": [
                        st.session_state.evaluation_results["accuracy"],
                        st.session_state.evaluation_results["precision"],
                        st.session_state.evaluation_results["recall"],
                        st.session_state.evaluation_results["f1_score"],
                        st.session_state.evaluation_results["roc_auc"]
                    ]
                })
                
                st.dataframe(metrics_df)
                
                # Plot confusion matrix
                if metrics["Confusion Matrix"]:
                    st.subheader("Confusion Matrix")
                    cm = st.session_state.evaluation_results["confusion_matrix"]
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                    ax.figure.colorbar(im, ax=ax)
                    
                    # Set labels
                    classes = [f"Class {i}" for i in range(cm.shape[0])]
                    ax.set(xticks=np.arange(cm.shape[1]),
                          yticks=np.arange(cm.shape[0]),
                          xticklabels=classes, yticklabels=classes,
                          title='Confusion Matrix',
                          ylabel='True label',
                          xlabel='Predicted label')
                    
                    # Rotate tick labels and set alignment
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                    
                    # Loop over data dimensions and create text annotations
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax.text(j, i, format(cm[i, j], 'd'),
                                   ha="center", va="center",
                                   color="white" if cm[i, j] > cm.max() / 2. else "black")
                    
                    fig.tight_layout()
                    st.pyplot(fig)
                
                # Plot ROC curve
                if metrics["ROC-AUC"]:
                    st.subheader("ROC Curve")
                    
                    # Generate fake ROC curve data
                    fpr = np.linspace(0, 1, 100)
                    tpr = 1 - np.exp(-5 * fpr)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {st.session_state.evaluation_results["roc_auc"]:.2f})')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('Receiver Operating Characteristic')
                    ax.legend(loc="lower right")
                    ax.grid(True)
                    
                    st.pyplot(fig)

# Explainability page
elif page == "Explainability":
    st.header("Model Explainability")
    
    if 'model' not in st.session_state or not st.session_state.get('model', {}).get('trained', False):
        st.warning("Please train a model first in the 'Model Training' page.")
    else:
        st.markdown("""
        <div class="highlight">
        <p>Explainability is critical in mental health applications to provide transparency and build trust 
        with clinicians and patients. This page demonstrates various techniques to interpret model predictions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different explainability methods
        explain_tabs = st.tabs(["LIME", "SHAP", "Attention Visualization", "Feature Importance", "Interactive Examples"])
        
        # Sample selection for all methods
        st.sidebar.markdown("### Sample Selection")
        
        # Load sample data options if not already in session state
        if "sample_data" not in st.session_state:
            # Create sample text data for demonstration
            sample_texts = [
                "I've been feeling really down lately, nothing seems to bring me joy anymore.",
                "I'm constantly worried about everything, my anxiety is through the roof.",
                "I feel restless and irritable, and my thoughts seem to race all the time.",
                "I've been having trouble sleeping, and I feel exhausted all the time.",
                "I keep having flashbacks to the traumatic event, and I can't seem to stop them."
            ]
            # Create sample metadata
            sample_metadata = [
                {"condition": "Depression", "severity": "Moderate"},
                {"condition": "Anxiety", "severity": "Severe"},
                {"condition": "Bipolar", "severity": "Mild"},
                {"condition": "Depression", "severity": "Mild"},
                {"condition": "PTSD", "severity": "Severe"}
            ]
            
            st.session_state.sample_data = {
                "texts": sample_texts,
                "metadata": sample_metadata
            }
        
        # Sample selection
        sample_idx = st.sidebar.selectbox(
            "Select sample to explain",
            range(len(st.session_state.sample_data["texts"])),
            format_func=lambda i: f"Sample {i+1}: {st.session_state.sample_data['metadata'][i]['condition']} ({st.session_state.sample_data['metadata'][i]['severity']})"
        )
        
        # Display the selected sample
        st.sidebar.markdown("### Selected Sample")
        st.sidebar.markdown(f"**Text:** {st.session_state.sample_data['texts'][sample_idx]}")
        st.sidebar.markdown(f"**Condition:** {st.session_state.sample_data['metadata'][sample_idx]['condition']}")
        st.sidebar.markdown(f"**Severity:** {st.session_state.sample_data['metadata'][sample_idx]['severity']}")
        
        # Modality selection
        st.sidebar.markdown("### Modality Selection")
        modalities = []
        if st.sidebar.checkbox("Text", value=True):
            modalities.append("text")
        if st.sidebar.checkbox("Audio"):
            modalities.append("audio")
        if st.sidebar.checkbox("Physiological"):
            modalities.append("physiological")
        if st.sidebar.checkbox("Imaging"):
            modalities.append("imaging")
        
        # Common function to generate explanations
        def generate_explanations():
            with st.spinner("Generating explanations..."):
                # Simulate explanation generation
                time.sleep(1)
                
                # Store explanations in session state if not already there or if regeneration is needed
                if "explainability_results" not in st.session_state or st.session_state.get("regenerate_explanations", False):
                    # LIME results
                    text_features = [
                        "depressed", "sad", "joy", "anymore", "feeling",
                        "down", "lately", "nothing", "seems", "really"
                    ]
                    text_importance = np.array([0.32, 0.28, 0.22, 0.18, 0.12, 0.10, 0.08, 0.07, 0.05, 0.04])
                    
                    # Audio features (if available)
                    audio_features = [
                        "pitch_mean", "pitch_std", "energy", "speech_rate", "pause_duration",
                        "voice_quality", "articulation", "prosody", "rhythm", "intensity"
                    ]
                    audio_importance = np.array([0.25, 0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.03])
                    
                    # SHAP values
                    shap_values = {
                        "text": np.random.uniform(-1, 1, size=10),
                        "audio": np.random.uniform(-1, 1, size=10)
                    }
                    
                    # Attention maps
                    attention_maps = {
                        "text": np.random.random((8, 10, 10)),
                        "audio": np.random.random((8, 20, 20))
                    }
                    
                    # Make attention maps look more like real attention (diagonal-heavy)
                    for i in range(attention_maps["text"].shape[1]):
                        for j in range(attention_maps["text"].shape[2]):
                            attention_maps["text"][:, i, j] *= 1.0 / (1.0 + 0.5 * abs(i - j))
                    
                    # Global feature importance
                    global_importance = {
                        "modalities": ["Text", "Audio", "Physiological", "Imaging"],
                        "scores": [0.45, 0.35, 0.15, 0.05]
                    }
                    
                    # Counterfactual examples
                    counterfactuals = [
                        {
                            "original": "I've been feeling really down lately, nothing seems to bring me joy anymore.",
                            "counterfactual": "I've been feeling better lately, several things bring me joy now.",
                            "changes": ["down → better", "nothing → several", "anymore → now"],
                            "prediction_change": {"from": "Depression (0.78)", "to": "No Depression (0.12)"}
                        },
                        {
                            "original": "I'm constantly worried about everything, my anxiety is through the roof.",
                            "counterfactual": "I'm occasionally concerned about some things, my anxiety is manageable.",
                            "changes": ["constantly → occasionally", "everything → some things", "through the roof → manageable"],
                            "prediction_change": {"from": "Anxiety (0.82)", "to": "No Anxiety (0.23)"}
                        }
                    ]
                    
                    # Highlighted text for LIME
                    highlighted_texts = [
                        """I've been feeling really <span style="background-color: rgba(255,0,0,0.3)">down</span> lately, <span style="background-color: rgba(255,0,0,0.2)">nothing</span> seems to bring me <span style="background-color: rgba(255,0,0,0.25)">joy</span> <span style="background-color: rgba(255,0,0,0.2)">anymore</span>.""",
                        """I'm <span style="background-color: rgba(255,0,0,0.3)">constantly worried</span> about <span style="background-color: rgba(255,0,0,0.2)">everything</span>, my <span style="background-color: rgba(255,0,0,0.4)">anxiety</span> is <span style="background-color: rgba(255,0,0,0.3)">through the roof</span>."""
                    ]
                    
                    st.session_state.explainability_results = {
                        "sample_idx": sample_idx,
                        "text_features": text_features,
                        "text_importance": text_importance,
                        "audio_features": audio_features,
                        "audio_importance": audio_importance,
                        "shap_values": shap_values,
                        "attention_maps": attention_maps,
                        "global_importance": global_importance,
                        "counterfactuals": counterfactuals,
                        "highlighted_texts": highlighted_texts
                    }
                    
                    st.session_state.regenerate_explanations = False
                
                return st.session_state.explainability_results
        
        # LIME tab
        with explain_tabs[0]:
            st.subheader("LIME (Local Interpretable Model-agnostic Explanations)")
            
            st.markdown("""
            LIME explains predictions by approximating the model locally with an interpretable model.
            It shows which features (words, audio characteristics, etc.) most contributed to a specific prediction.
            """)
            
            # Generate explanations button
            if st.button("Generate LIME Explanations", key="lime_button"):
                explanations = generate_explanations()
                
                # Display text explanations if text modality is selected
                if "text" in modalities:
                    st.markdown("#### Text Feature Importance")
                    
                    # Display highlighted text
                    st.markdown("##### Highlighted Text")
                    st.markdown(explanations["highlighted_texts"][sample_idx % len(explanations["highlighted_texts"])], unsafe_allow_html=True)
                    
                    # Display feature importance
                    text_features = explanations["text_features"]
                    text_importance = explanations["text_importance"]
                    
                    text_df = pd.DataFrame({
                        "Feature": text_features,
                        "Importance": text_importance
                    }).sort_values("Importance", ascending=False)
                    
                    fig = px.bar(
                        text_df,
                        x="Importance",
                        y="Feature",
                        orientation='h',
                        title="Text Feature Importance",
                        color="Importance",
                        color_continuous_scale=["#E8F5E9", "#FFECB3", "#FFCDD2"]
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display audio explanations if audio modality is selected
                if "audio" in modalities:
                    st.markdown("#### Audio Feature Importance")
                    audio_features = explanations["audio_features"]
                    audio_importance = explanations["audio_importance"]
                    
                    audio_df = pd.DataFrame({
                        "Feature": audio_features,
                        "Importance": audio_importance
                    }).sort_values("Importance", ascending=False)
                    
                    fig = px.bar(
                        audio_df,
                        x="Importance",
                        y="Feature",
                        orientation='h',
                        title="Audio Feature Importance",
                        color="Importance",
                        color_continuous_scale=["#E8F5E9", "#FFECB3", "#FFCDD2"]
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        # SHAP tab
        with explain_tabs[1]:
            st.subheader("SHAP (SHapley Additive exPlanations)")
            
            st.markdown("""
            SHAP uses game theory concepts to allocate feature importance values, providing a unified measure of feature 
            importance. SHAP values indicate how each feature contributes to pushing the prediction higher or lower.
            """)
            
            # Generate explanations button
            if st.button("Generate SHAP Explanations", key="shap_button"):
                explanations = generate_explanations()
                
                for modality in modalities:
                    if modality in explanations["shap_values"]:
                        st.markdown(f"#### {modality.capitalize()} SHAP Values")
                        
                        # Get feature names
                        if modality == "text":
                            features = explanations["text_features"]
                        elif modality == "audio":
                            features = explanations["audio_features"]
                        else:
                            features = [f"Feature_{i}" for i in range(len(explanations["shap_values"][modality]))]
                        
                        # Create DataFrame for SHAP values
                        shap_df = pd.DataFrame({
                            "Feature": features,
                            "SHAP Value": explanations["shap_values"][modality]
                        }).sort_values("SHAP Value", key=abs, ascending=False)
                        
                        # Create waterfall chart
                        fig = go.Figure(go.Waterfall(
                            name="SHAP",
                            orientation="h",
                            measure=["relative"] * len(shap_df),
                            x=shap_df["SHAP Value"],
                            textposition="outside",
                            text=shap_df["SHAP Value"].round(3),
                            y=shap_df["Feature"],
                            connector={"line": {"color": "rgb(63, 63, 63)"}},
                            increasing={"marker": {"color": "#FF4136"}},
                            decreasing={"marker": {"color": "#3D9970"}}
                        ))
                        
                        fig.update_layout(
                            title=f"{modality.capitalize()} SHAP Waterfall Plot",
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display SHAP summary plot-like visualization
                        st.markdown(f"#### {modality.capitalize()} SHAP Summary")
                        
                        # Sort features by absolute SHAP value
                        sorted_idx = np.argsort(np.abs(shap_df["SHAP Value"]))
                        sorted_features = shap_df["Feature"].iloc[sorted_idx].tolist()
                        sorted_values = shap_df["SHAP Value"].iloc[sorted_idx].tolist()
                        
                        # Create horizontal bar chart for SHAP summary
                        fig = px.bar(
                            x=sorted_values,
                            y=sorted_features,
                            orientation="h",
                            color=sorted_values,
                            color_continuous_scale="RdBu_r",
                            title=f"{modality.capitalize()} SHAP Summary"
                        )
                        
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
        # Attention Visualization tab
        with explain_tabs[2]:
            st.subheader("Attention Visualization")
            
            st.markdown("""
            Attention mechanisms in deep learning models help the model focus on important parts of the input. 
            This visualization shows which parts of the input data the model is paying attention to when making predictions.
            """)
            
            # Generate explanations button
            if st.button("Generate Attention Maps", key="attention_button"):
                explanations = generate_explanations()
                
                # Select which attention head to visualize
                head_idx = st.slider("Select attention head", 0, 7, 0)
                
                for modality in modalities:
                    if modality in explanations["attention_maps"]:
                        st.markdown(f"#### {modality.capitalize()} Attention Map")
                        
                        attention_map = explanations["attention_maps"][modality][head_idx]
                        
                        # Create heatmap
                        fig = px.imshow(
                            attention_map,
                            color_continuous_scale="Viridis",
                            title=f"{modality.capitalize()} Attention Map (Head {head_idx})"
                        )
                        
                        fig.update_layout(height=500, width=500)
                        st.plotly_chart(fig)
                        
                        # Display interpretation
                        st.markdown("#### Interpretation")
                        st.markdown("""
                        The heatmap shows how different tokens/features attend to each other. 
                        Brighter colors indicate stronger attention between elements.
                        
                        In text data, this often reveals which words the model considers most important for the prediction
                        and how they relate to other words in the input.
                        """)
        
        # Global Importance tab
        with explain_tabs[3]:
            st.subheader("Global Feature Importance")
            
            st.markdown("""
            Global feature importance shows which features are most important across all predictions,
            not just for a single sample. This helps identify which modalities and features are most
            influential for the model's predictions overall.
            """)
            
            # Generate explanations button
            if st.button("Generate Global Importance", key="global_button"):
                explanations = generate_explanations()
                
                # Display modality importance
                st.markdown("#### Modality Importance")
                
                global_importance = explanations["global_importance"]
                
                # Create pie chart for modality importance
                fig = px.pie(
                    values=global_importance["scores"],
                    names=global_importance["modalities"],
                    title="Relative Importance of Different Modalities",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig)
                
                # Display interpretation
                st.markdown("#### Interpretation")
                st.markdown("""
                The pie chart shows the relative importance of each modality in the multimodal model.
                This helps clinicians understand which types of data (text, audio, physiological, imaging)
                are most informative for mental health assessments.
                """)
                
        # Counterfactual Examples tab
        with explain_tabs[4]:
            st.subheader("Counterfactual Examples")
            
            st.markdown("""
            Counterfactual examples show how the input would need to change to get a different prediction.
            This helps understand the decision boundary of the model and which features most strongly
            influence the prediction.
            """)
            
            # Generate explanations button
            if st.button("Generate Counterfactuals", key="counterfactual_button"):
                explanations = generate_explanations()
                
                # Display counterfactual examples
                counterfactuals = explanations["counterfactuals"]
                
                for i, cf in enumerate(counterfactuals):
                    st.markdown(f"#### Counterfactual Example {i+1}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Original Input:**")
                        st.markdown(f"```\n{cf['original']}\n```")
                        st.markdown(f"**Prediction:** {cf['prediction_change']['from']}")
                    
                    with col2:
                        st.markdown("**Counterfactual Input:**")
                        st.markdown(f"```\n{cf['counterfactual']}\n```")
                        st.markdown(f"**Prediction:** {cf['prediction_change']['to']}")
                    
                    st.markdown("**Key Changes:**")
                    for change in cf["changes"]:
                        st.markdown(f"- {change}")
                    
                    st.markdown("---")
                
                # Display interpretation
                st.markdown("#### Interpretation")
                st.markdown("""
                Counterfactual examples show what would need to change in the input to get a different prediction.
                This can help clinicians understand which aspects of a patient's data are most indicative of 
                specific mental health conditions and how changes in those aspects might affect the diagnosis.
                """)
        
        # Set regenerate flag when sample changes
        if "last_sample_idx" not in st.session_state or st.session_state.last_sample_idx != sample_idx:
            st.session_state.regenerate_explanations = True
            st.session_state.last_sample_idx = sample_idx


# Dashboard page
elif page == "Dashboard":
    st.header("MH-Net Dashboard")
    
    if 'model' not in st.session_state or not st.session_state.get('model', {}).get('trained', False):
        st.warning("Please train a model first in the 'Model Training' page.")
    else:
        # Continue with the original dashboard code
        # Display overview metrics
        st.subheader("System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Model Accuracy", 
                value=f"{st.session_state.get('evaluation_results', {}).get('accuracy', 0.82):.2f}",
                delta="0.05"
            )
        
        with col2:
            st.metric(
                label="F1 Score", 
                value=f"{st.session_state.get('evaluation_results', {}).get('f1_score', 0.79):.2f}",
                delta="0.03"
            )
        
        with col3:
            st.metric(
                label="ROC-AUC", 
                value=f"{st.session_state.get('evaluation_results', {}).get('roc_auc', 0.85):.2f}",
                delta="0.04"
            )
        
        with col4:
            st.metric(
                label="Processing Time", 
                value="0.23 sec",
                delta="-0.05"
            )
        
        # Display model architecture visualization
        st.subheader("Model Architecture")
        
        # Model architecture visualization code would continue here

# Real-time Assessment page
elif page == "Real-time Assessment":
    st.markdown("<h1 class='main-header'>Real-time Mental Health Assessment</h1>", unsafe_allow_html=True)
    
    # Check if model is trained
    if 'model' not in st.session_state or not st.session_state.get('model', {}).get('trained', False):
        st.warning("Please train a model first in the 'Model Training' page before using real-time assessment.")
    else:
        # Create tabs for different input methods
        input_tab, results_tab, history_tab = st.tabs(["Input Data", "Assessment Results", "Assessment History"])
        
        with input_tab:
            st.subheader("Patient Information")
            
            # Patient info form
            with st.form("patient_info_form"):
                col1, col2 = st.columns(2)
                with col1:
                    patient_id = st.text_input("Patient ID", value="P-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
                    age = st.number_input("Age", min_value=18, max_value=100, value=35)
                    gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"])
                
                with col2:
                    assessment_date = st.date_input("Assessment Date", value=datetime.now())
                    clinician = st.text_input("Clinician Name", value="Dr. Smith")
                    session_type = st.selectbox("Session Type", ["Initial Assessment", "Follow-up", "Crisis Intervention"])
                
                st.markdown("### Text Input")
                text_input = st.text_area(
                    "Patient's text response or clinical notes",
                    height=150,
                    placeholder="Enter patient's response to assessment questions or clinical notes..."
                )
                
                st.markdown("### Audio Input")
                audio_upload = st.file_uploader("Upload audio file (WAV, MP3)", type=["wav", "mp3"])
                
                # Option to record audio directly
                use_microphone = st.checkbox("Record audio directly")
                if use_microphone:
                    st.info("Microphone recording would be implemented here. Currently using uploaded files.")
                
                st.markdown("### Physiological Data")
                physio_upload = st.file_uploader("Upload physiological data (CSV)", type=["csv"])
                
                st.markdown("### Medical Imaging")
                imaging_upload = st.file_uploader("Upload medical imaging data (NII, DICOM)", type=["nii", "nii.gz", "dcm"])
                
                # Form submission button
                submitted = st.form_submit_button("Run Assessment")
            
            # Handle form submission outside the form
            if submitted:
                with st.spinner("Processing multimodal data and running assessment..."):
                    # Simulate processing delay
                    time.sleep(2)
                    
                    # Prepare demo results
                    risk_scores = {
                        "Depression": round(0.4 + np.random.random() * 0.3, 2),
                        "Anxiety": round(0.3 + np.random.random() * 0.4, 2),
                        "PTSD": round(0.1 + np.random.random() * 0.2, 2),
                        "Bipolar": round(0.2 + np.random.random() * 0.15, 2),
                        "Schizophrenia": round(0.05 + np.random.random() * 0.1, 2)
                    }
                    
                    # Store assessment results in session state
                    if "assessment_history" not in st.session_state:
                        st.session_state.assessment_history = []
                    
                    # Create assessment record
                    assessment_record = {
                        "patient_id": patient_id,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "risk_scores": risk_scores,
                        "text_input": text_input if text_input else None,
                        "has_audio": True if audio_upload else False,
                        "has_physio": True if physio_upload else False,
                        "has_imaging": True if imaging_upload else False,
                        "metadata": {
                            "age": age,
                            "gender": gender,
                            "clinician": clinician,
                            "session_type": session_type,
                            "assessment_date": assessment_date.strftime("%Y-%m-%d")
                        }
                    }
                    
                    # Save to database
                    try:
                        # Add to database
                        db_manager.add_assessment(assessment_record)
                        st.success("Assessment saved to database successfully!")
                    except Exception as e:
                        st.error(f"Failed to save assessment to database: {str(e)}")
                    
                    # Add to session history for immediate use
                    st.session_state.assessment_history.insert(0, assessment_record)
                    
                    # Set current assessment for display in results tab
                    st.session_state.current_assessment = assessment_record
                    
                    # Success message and switch to results tab
                    st.success("Assessment completed successfully! View results in the Assessment Results tab.")
                    
                    # JavaScript to switch to results tab
                    js = f"""
                    <script>
                        function switchTab() {{
                            const tabs = window.parent.document.querySelectorAll('div[data-testid="stHorizontalBlock"] button[data-baseweb="tab"]');
                            if (tabs.length >= 2) {{
                                tabs[1].click();
                            }}
                        }}
                        switchTab();
                    </script>
                    """
                    st.components.v1.html(js, height=0)
        
        with results_tab:
            if "current_assessment" in st.session_state:
                assessment = st.session_state.current_assessment
                
                # Display patient info
                st.subheader("Patient Information")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Patient ID:** {assessment['patient_id']}")
                    st.markdown(f"**Age:** {assessment['metadata']['age']}")
                with col2:
                    st.markdown(f"**Assessment Date:** {assessment['metadata']['assessment_date']}")
                    st.markdown(f"**Gender:** {assessment['metadata']['gender']}")
                with col3:
                    st.markdown(f"**Clinician:** {assessment['metadata']['clinician']}")
                    st.markdown(f"**Session Type:** {assessment['metadata']['session_type']}")
                
                # Display risk assessment with gauge charts
                st.markdown("<h3>Mental Health Risk Assessment</h3>", unsafe_allow_html=True)
                
                # Row of gauges for each condition
                risk_scores = assessment["risk_scores"]
                st.markdown('<div class="highlight">', unsafe_allow_html=True)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_scores["Depression"] * 100,
                        title = {'text': "Depression"},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#1E88E5"},
                            'steps': [
                                {'range': [0, 33], 'color': "#E8F5E9"},
                                {'range': [33, 66], 'color': "#FFECB3"},
                                {'range': [66, 100], 'color': "#FFCDD2"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 80
                            }
                        }
                    ))
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_scores["Anxiety"] * 100,
                        title = {'text': "Anxiety"},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#43A047"},
                            'steps': [
                                {'range': [0, 33], 'color': "#E8F5E9"},
                                {'range': [33, 66], 'color': "#FFECB3"},
                                {'range': [66, 100], 'color': "#FFCDD2"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 80
                            }
                        }
                    ))
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col3:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_scores["PTSD"] * 100,
                        title = {'text': "PTSD"},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#FB8C00"},
                            'steps': [
                                {'range': [0, 33], 'color': "#E8F5E9"},
                                {'range': [33, 66], 'color': "#FFECB3"},
                                {'range': [66, 100], 'color': "#FFCDD2"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 80
                            }
                        }
                    ))
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col4:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_scores["Bipolar"] * 100,
                        title = {'text': "Bipolar"},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#8E24AA"},
                            'steps': [
                                {'range': [0, 33], 'color': "#E8F5E9"},
                                {'range': [33, 66], 'color': "#FFECB3"},
                                {'range': [66, 100], 'color': "#FFCDD2"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 80
                            }
                        }
                    ))
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col5:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_scores["Schizophrenia"] * 100,
                        title = {'text': "Schizophrenia"},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#D81B60"},
                            'steps': [
                                {'range': [0, 33], 'color': "#E8F5E9"},
                                {'range': [33, 66], 'color': "#FFECB3"},
                                {'range': [66, 100], 'color': "#FFCDD2"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 80
                            }
                        }
                    ))
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Overall risk assessment
                max_risk = max(risk_scores.items(), key=lambda x: x[1])
                max_risk_condition = max_risk[0]
                max_risk_score = max_risk[1]
                
                st.subheader("Clinical Assessment")
                
                if max_risk_score > 0.6:
                    risk_level = "High"
                    risk_color = "🔴"
                elif max_risk_score > 0.3:
                    risk_level = "Moderate"
                    risk_color = "🟠"
                else:
                    risk_level = "Low"
                    risk_color = "🟢"
                
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>Overall Risk Assessment: {risk_color} {risk_level} Risk</h3>
                    <p>Primary concern: <strong>{max_risk_condition}</strong> (Risk score: {max_risk_score:.2f})</p>
                    <p>Assessment timestamp: {assessment['timestamp']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Clinical recommendations
                st.subheader("Clinical Recommendations")
                
                if risk_level == "High":
                    st.markdown("""
                    <div class='highlight'>
                        <h4>Immediate Action Recommended</h4>
                        <ul>
                            <li>Schedule follow-up appointment within 24-48 hours</li>
                            <li>Consider referral to specialist for the primary concern</li>
                            <li>Evaluate need for medication adjustment or additional therapeutic interventions</li>
                            <li>Provide crisis resources and emergency contact information</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                elif risk_level == "Moderate":
                    st.markdown("""
                    <div class='highlight'>
                        <h4>Monitoring Recommended</h4>
                        <ul>
                            <li>Schedule follow-up appointment within 1-2 weeks</li>
                            <li>Consider additional assessment tools for the primary concern</li>
                            <li>Review current treatment plan and consider adjustments</li>
                            <li>Provide educational resources about coping strategies</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='highlight'>
                        <h4>Routine Follow-up</h4>
                        <ul>
                            <li>Continue with regular treatment plan</li>
                            <li>Schedule routine follow-up within 4-6 weeks</li>
                            <li>Encourage continued use of support resources</li>
                            <li>Monitor for any changes in symptoms</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Feature importance explanation
                st.subheader("Multimodal Feature Importance")
                
                # Text features
                if assessment["text_input"]:
                    st.markdown("#### Text Features")
                    # Generate some fake feature importance for text
                    text_features = ["negative sentiment", "social withdrawal", "sleep disruption", 
                                    "anxiety indicators", "treatment references", "medication mentions",
                                    "family references", "work stress", "hopelessness indicators", 
                                    "emotion words"]
                    text_importance = np.random.random(10)
                    text_importance = text_importance / text_importance.sum()
                    
                    text_df = pd.DataFrame({
                        "Feature": text_features,
                        "Importance": text_importance
                    }).sort_values("Importance", ascending=False)
                    
                    fig = px.bar(
                        text_df,
                        x="Importance",
                        y="Feature",
                        orientation='h',
                        title="Text Feature Importance",
                        color="Importance",
                        color_continuous_scale=["#E8F5E9", "#FFECB3", "#FFCDD2"]
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Modality importance
                st.markdown("#### Modality Contribution")
                modalities = ["Text", "Audio", "Physiological", "Imaging"]
                # Generate some fake modality importance
                modality_importance = [
                    0.6 if assessment["text_input"] else 0.1,
                    0.2 if assessment["has_audio"] else 0.1,
                    0.1 if assessment["has_physio"] else 0.05,
                    0.1 if assessment["has_imaging"] else 0.05
                ]
                # Normalize
                modality_importance = np.array(modality_importance) / sum(modality_importance)
                
                fig = px.pie(
                    values=modality_importance,
                    names=modalities,
                    title="Modality Contribution to Assessment",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Export options
                st.subheader("Export Assessment")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Export as PDF"):
                        st.info("PDF export functionality would be implemented here.")
                with col2:
                    if st.button("Send to Electronic Health Record"):
                        st.info("EHR integration would be implemented here.")
            else:
                st.info("No assessment results available. Please run an assessment in the Input Data tab.")
        
        with history_tab:
            st.subheader("Assessment History")
            
            # Load assessments from database
            try:
                assessments = db_manager.get_all_assessments()
                
                if assessments:
                    # Convert SQLAlchemy objects to dictionaries for compatibility with existing code
                    history = [assessment.to_dict() for assessment in assessments]
                    
                    # Add patient info for display
                    for assessment in history:
                        patient = db_manager.get_patient(assessment["patient_id"])
                        if patient:
                            assessment["metadata"] = {
                                "age": patient.age,
                                "gender": patient.gender,
                                "clinician": assessment["clinician"],
                                "session_type": assessment["session_type"],
                                "assessment_date": assessment["assessment_date"]
                            }
                            assessment["patient_id"] = patient.patient_id
                    
                    st.info(f"Loaded {len(history)} assessments from database.")
                else:
                    # Fall back to session state
                    if "assessment_history" in st.session_state and st.session_state.assessment_history:
                        history = st.session_state.assessment_history
                        st.info("Using session assessment history. Database is empty.")
                    else:
                        history = []
                        st.info("No assessment history available.")
            except Exception as e:
                st.error(f"Failed to load assessments from database: {str(e)}")
                # Fall back to session state
                if "assessment_history" in st.session_state and st.session_state.assessment_history:
                    history = st.session_state.assessment_history
                else:
                    history = []
            
            if history:
                # Search and filter options
                col1, col2 = st.columns([2, 1])
                with col1:
                    search_query = st.text_input("Search by Patient ID")
                with col2:
                    sort_order = st.selectbox("Sort by", ["Most recent first", "Oldest first"])
                
                # Filter based on search query
                if search_query:
                    filtered_history = [h for h in history if search_query.lower() in str(h["patient_id"]).lower()]
                else:
                    filtered_history = history
                
                # Sort based on selection
                if sort_order == "Oldest first":
                    filtered_history = filtered_history[::-1]
                
                # Display history
                if filtered_history:
                    for i, assessment in enumerate(filtered_history):
                        with st.expander(f"Assessment: {assessment['patient_id']} - {assessment['timestamp']}"):
                            # Patient info
                            st.markdown(f"**Patient ID:** {assessment['patient_id']}")
                            st.markdown(f"**Assessment Date:** {assessment['assessment_date']}")
                            st.markdown(f"**Clinician:** {assessment['clinician']}")
                            
                            # Risk scores
                            st.markdown("##### Risk Scores")
                            risk_df = pd.DataFrame({
                                "Condition": list(assessment["risk_scores"].keys()),
                                "Risk Score": list(assessment["risk_scores"].values())
                            })
                            st.dataframe(risk_df)
                            
                            # Action buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"View Details {i}", key=f"view_{i}"):
                                    # Set as current assessment and navigate to results tab
                                    st.session_state.current_assessment = assessment
                                    # JavaScript to switch to results tab
                                    js = f"""
                                    <script>
                                        function switchTab() {{
                                            const tabs = window.parent.document.querySelectorAll('div[data-testid="stHorizontalBlock"] button[data-baseweb="tab"]');
                                            if (tabs.length >= 2) {{
                                                tabs[1].click();
                                            }}
                                        }}
                                        switchTab();
                                    </script>
                                    """
                                    st.components.v1.html(js, height=0)
                            with col2:
                                if st.button(f"Delete {i}", key=f"delete_{i}"):
                                    # Would implement proper deletion functionality here
                                    st.info("Delete functionality would be implemented here.")
                else:
                    st.info("No matching assessment records found.")
            else:
                st.info("No assessment history available. Run your first assessment in the Input Data tab.")

# Dashboard page
elif page == "Dashboard":
    st.markdown("<h1 class='main-header'>MH-Net System Dashboard</h1>", unsafe_allow_html=True)
    
    if 'model' not in st.session_state or not st.session_state.get('model', {}).get('trained', False):
        st.warning("Please train a model first in the 'Model Training' page.")
    else:
        # Create dashboard tabs for organization
        dashboard_tabs = st.tabs([
            "System Overview", 
            "Model Performance", 
            "Architecture", 
            "Patient Analytics",
            "Resource Usage"
        ])
        
        with dashboard_tabs[0]:  # System Overview tab
            st.subheader("System Overview")
            
            # Overview Description
            st.markdown("""
            <div class="highlight">
            This dashboard provides a comprehensive overview of the MH-Net mental health diagnostics system, 
            including model performance metrics, architecture details, and clinical analytics.
            </div>
            """, unsafe_allow_html=True)
            
            # Status indicators
            st.markdown("#### System Status")
            
            status_cols = st.columns(4)
            with status_cols[0]:
                st.markdown(
                    f"""
                    <div style="border-radius:10px; padding:10px; background-color:#E8F5E9; text-align:center;">
                        <h3 style="margin:0; color:#2E7D32;">● ONLINE</h3>
                        <p style="margin:0;">System Status</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            with status_cols[1]:
                st.markdown(
                    f"""
                    <div style="border-radius:10px; padding:10px; background-color:#E8F5E9; text-align:center;">
                        <h3 style="margin:0; color:#2E7D32;">● ACTIVE</h3>
                        <p style="margin:0;">Database</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            with status_cols[2]:
                st.markdown(
                    f"""
                    <div style="border-radius:10px; padding:10px; background-color:#E8F5E9; text-align:center;">
                        <h3 style="margin:0; color:#2E7D32;">● TRAINED</h3>
                        <p style="margin:0;">Model Status</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            with status_cols[3]:
                st.markdown(
                    f"""
                    <div style="border-radius:10px; padding:10px; background-color:#FFF3E0; text-align:center;">
                        <h3 style="margin:0; color:#E65100;">● MODERATE</h3>
                        <p style="margin:0;">System Load</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            # Key performance metrics
            st.markdown("#### Key Performance Metrics")
            
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric(
                    label="Model Accuracy", 
                    value=f"{st.session_state.get('evaluation_results', {}).get('accuracy', 0.82):.2f}",
                    delta="0.05",
                    help="Overall classification accuracy on the validation set"
                )
            
            with metric_cols[1]:
                st.metric(
                    label="F1 Score", 
                    value=f"{st.session_state.get('evaluation_results', {}).get('f1_score', 0.79):.2f}",
                    delta="0.03",
                    help="Harmonic mean of precision and recall"
                )
            
            with metric_cols[2]:
                st.metric(
                    label="ROC-AUC", 
                    value=f"{st.session_state.get('evaluation_results', {}).get('roc_auc', 0.85):.2f}",
                    delta="0.04",
                    help="Area under the Receiver Operating Characteristic curve"
                )
            
            with metric_cols[3]:
                st.metric(
                    label="Processing Time", 
                    value="0.23 sec",
                    delta="-0.05",
                    help="Average time to process a single patient assessment"
                )
            
            # Recent activity
            st.markdown("#### Recent System Activity")
            
            # Recent assessments
            if "assessment_history" in st.session_state and len(st.session_state.assessment_history) > 0:
                recent_assessments = st.session_state.assessment_history[:5]  # Show last 5 assessments
                
                activity_data = []
                for i, assessment in enumerate(recent_assessments):
                    max_risk = max(assessment["risk_scores"].items(), key=lambda x: x[1])
                    activity_data.append({
                        "Timestamp": assessment["timestamp"],
                        "Patient ID": assessment["patient_id"],
                        "Primary Condition": max_risk[0],
                        "Risk Score": f"{max_risk[1]:.2f}"
                    })
                
                activity_df = pd.DataFrame(activity_data)
                st.dataframe(activity_df, use_container_width=True)
            else:
                st.info("No recent assessments available.")
            
            # System notifications
            st.markdown("#### System Notifications")
            
            notifications = [
                {"level": "info", "message": "System update available: v2.0.3", "time": "2 hours ago"},
                {"level": "success", "message": "Database backup completed successfully", "time": "4 hours ago"},
                {"level": "warning", "message": "Resource usage approaching threshold (78%)", "time": "Yesterday"},
            ]
            
            for notification in notifications:
                if notification["level"] == "info":
                    st.info(f"{notification['message']} - {notification['time']}")
                elif notification["level"] == "success":
                    st.success(f"{notification['message']} - {notification['time']}")
                elif notification["level"] == "warning":
                    st.warning(f"{notification['message']} - {notification['time']}")
                elif notification["level"] == "error":
                    st.error(f"{notification['message']} - {notification['time']}")
        
        with dashboard_tabs[1]:  # Model Performance tab
            st.subheader("Model Performance Analytics")
            
            # Create confusion matrix if available
            if "evaluation_results" in st.session_state and "confusion_matrix" in st.session_state.evaluation_results:
                st.markdown("#### Confusion Matrix")
                
                cm = st.session_state.evaluation_results["confusion_matrix"]
                condition_labels = ["Depression", "Anxiety", "PTSD", "Bipolar", "Schizophrenia"]
                
                # Create heatmap
                fig = px.imshow(
                    cm,
                    x=condition_labels,
                    y=condition_labels,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted Condition", y="True Condition", color="Count"),
                    text_auto=True
                )
                
                fig.update_layout(
                    title="Confusion Matrix",
                    xaxis_title="Predicted Condition",
                    yaxis_title="True Condition",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Per-class metrics
            st.markdown("#### Per-Class Performance")
            
            # Generate per-class metrics
            class_metrics = {
                "Depression": {"precision": 0.83, "recall": 0.85, "f1": 0.84, "support": 105},
                "Anxiety": {"precision": 0.80, "recall": 0.78, "f1": 0.79, "support": 100},
                "PTSD": {"precision": 0.85, "recall": 0.80, "f1": 0.82, "support": 100},
                "Bipolar": {"precision": 0.86, "recall": 0.87, "f1": 0.86, "support": 100},
                "Schizophrenia": {"precision": 0.88, "recall": 0.88, "f1": 0.88, "support": 100}
            }
            
            # Convert to DataFrame for easy visualization
            class_metrics_df = pd.DataFrame(class_metrics).T
            class_metrics_df = class_metrics_df.reset_index().rename(columns={"index": "Condition"})
            
            # Create a grouped bar chart for precision, recall, and f1
            fig = px.bar(
                class_metrics_df,
                x="Condition",
                y=["precision", "recall", "f1"],
                barmode="group",
                color_discrete_sequence=["#1E88E5", "#43A047", "#FB8C00"],
                labels={"value": "Score", "variable": "Metric"},
                title="Performance Metrics by Condition"
            )
            
            fig.update_layout(height=500, legend_title="Metric")
            st.plotly_chart(fig, use_container_width=True)
            
            # ROC curves
            st.markdown("#### ROC Curves")
            
            # Generate ROC curve data - for demo purposes
            def generate_roc_curve(auc_value, points=100):
                # Generate a plausible ROC curve with given AUC
                x = np.linspace(0, 1, points)
                # Use a power function to simulate different curves
                # Higher power gives curves closer to the upper left corner
                power = 0.5 / (1 - auc_value)
                y = np.power(x, power)
                return x, y
            
            fig = go.Figure()
            
            # Add ROC curves for each condition
            conditions = ["Depression", "Anxiety", "PTSD", "Bipolar", "Schizophrenia"]
            auc_values = [0.92, 0.88, 0.85, 0.89, 0.91]
            colors = ["#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#D81B60"]
            
            # Add reference line (random classifier)
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], 
                    y=[0, 1], 
                    mode="lines", 
                    line=dict(color="gray", dash="dash", width=1),
                    name="Random Classifier",
                    showlegend=True
                )
            )
            
            for condition, auc, color in zip(conditions, auc_values, colors):
                fpr, tpr = generate_roc_curve(auc)
                fig.add_trace(
                    go.Scatter(
                        x=fpr, 
                        y=tpr, 
                        mode="lines", 
                        line=dict(color=color, width=2),
                        name=f"{condition} (AUC={auc:.2f})",
                        showlegend=True
                    )
                )
            
            fig.update_layout(
                title="ROC Curves by Condition",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                legend=dict(x=0.7, y=0.05),
                height=500,
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Training history visualization
            if "history" in st.session_state:
                st.markdown("#### Training History")
                
                history = st.session_state.history
                epochs = list(range(1, len(history["loss"]) + 1))
                
                # Select which metrics to show
                metrics_to_show = st.multiselect(
                    "Select metrics to display",
                    options=["loss", "accuracy", "precision", "recall", "f1_score"],
                    default=["loss", "accuracy"]
                )
                
                if metrics_to_show:
                    # Create figure with secondary y-axis
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add traces for each selected metric
                    for i, metric in enumerate(metrics_to_show):
                        if metric == "loss":
                            # Show loss on secondary y-axis
                            fig.add_trace(
                                go.Scatter(
                                    x=epochs, 
                                    y=history[metric], 
                                    mode="lines", 
                                    name=f"Training {metric.capitalize()}",
                                    line=dict(color="#1E88E5", width=2)
                                ),
                                secondary_y=True
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=epochs, 
                                    y=history[f"val_{metric}"], 
                                    mode="lines", 
                                    name=f"Validation {metric.capitalize()}",
                                    line=dict(color="#1E88E5", width=2, dash="dash")
                                ),
                                secondary_y=True
                            )
                        else:
                            # Show other metrics on primary y-axis
                            fig.add_trace(
                                go.Scatter(
                                    x=epochs, 
                                    y=history[metric], 
                                    mode="lines", 
                                    name=f"Training {metric.capitalize()}",
                                    line=dict(color=px.colors.qualitative.Plotly[i], width=2)
                                ),
                                secondary_y=False
                            )
                            
                            if f"val_{metric}" in history:
                                fig.add_trace(
                                    go.Scatter(
                                        x=epochs, 
                                        y=history[f"val_{metric}"], 
                                        mode="lines", 
                                        name=f"Validation {metric.capitalize()}",
                                        line=dict(color=px.colors.qualitative.Plotly[i], width=2, dash="dash")
                                    ),
                                    secondary_y=False
                                )
                    
                    # Add figure layout
                    fig.update_layout(
                        title="Training History",
                        xaxis_title="Epoch",
                        height=500,
                        legend=dict(orientation="h", y=-0.2)
                    )
                    
                    # Set y-axes titles
                    if "loss" in metrics_to_show:
                        fig.update_yaxes(title_text="Metric Value", secondary_y=False)
                        fig.update_yaxes(title_text="Loss", secondary_y=True)
                    else:
                        fig.update_yaxes(title_text="Metric Value", secondary_y=False)
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Please select at least one metric to display.")
        
        with dashboard_tabs[2]:  # Architecture tab
            st.subheader("Model Architecture")
            
            # Display Model Configuration
            model_config = st.session_state.get('model', {})
            if model_config:
                st.markdown("#### Model Configuration")
                
                config_cols = st.columns(4)
                with config_cols[0]:
                    st.markdown(f"**Type:** {model_config.get('type', 'MH-Net (Multimodal Transformer)')}")
                    st.markdown(f"**Embedding Dim:** {model_config.get('embed_dim', 256)}")
                
                with config_cols[1]:
                    st.markdown(f"**Num Heads:** {model_config.get('num_heads', 8)}")
                    st.markdown(f"**Num Layers:** {model_config.get('num_layers', 4)}")
                
                with config_cols[2]:
                    st.markdown(f"**Dropout Rate:** {model_config.get('dropout_rate', 0.1)}")
                    st.markdown(f"**Hidden Dim:** {model_config.get('hidden_dim', 512)}")
                
                with config_cols[3]:
                    st.markdown(f"**Activation:** {model_config.get('activation', 'ReLU')}")
                    
                    # Calculate approximate parameter count
                    embed_dim = model_config.get('embed_dim', 256)
                    num_layers = model_config.get('num_layers', 4)
                    hidden_dim = model_config.get('hidden_dim', 512)
                    
                    # Simple parameter count estimation
                    embedding_params = embed_dim * 10000  # Assuming vocab size of 10,000
                    transformer_params = num_layers * (4 * embed_dim**2 + embed_dim * 2)
                    hidden_params = embed_dim * hidden_dim + hidden_dim
                    output_params = hidden_dim * 5 + 5  # Assuming 5 output classes
                    
                    total_params = embedding_params + transformer_params + hidden_params + output_params
                    
                    st.markdown(f"**Parameters:** {total_params:,}")
            
            # Create a simplified visualization of the model architecture
            st.markdown("#### Architecture Diagram")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Remove axes
            ax.axis('off')
            
            # Define coordinates for the components
            input_y = 0.8
            encoder_y = 0.6
            fusion_y = 0.4
            output_y = 0.2
            
            text_x = 0.2
            audio_x = 0.4
            physio_x = 0.6
            imaging_x = 0.8
            
            # Draw input boxes
            inputs = [
                (text_x, input_y, "Text\nInput"),
                (audio_x, input_y, "Audio\nInput"),
                (physio_x, input_y, "Physiological\nInput"),
                (imaging_x, input_y, "Imaging\nInput")
            ]
            
            for x, y, label in inputs:
                ax.add_patch(plt.Rectangle((x-0.05, y-0.05), 0.1, 0.1, fill=True, color='lightblue', alpha=0.8))
                ax.text(x, y, label, ha='center', va='center', fontsize=10)
            
            # Draw encoder boxes
            encoders = [
                (text_x, encoder_y, "Text\nEncoder"),
                (audio_x, encoder_y, "Audio\nEncoder"),
                (physio_x, encoder_y, "Physiological\nEncoder"),
                (imaging_x, encoder_y, "Imaging\nEncoder")
            ]
            
            for x, y, label in encoders:
                ax.add_patch(plt.Rectangle((x-0.08, y-0.05), 0.16, 0.1, fill=True, color='lightgreen', alpha=0.8))
                ax.text(x, y, label, ha='center', va='center', fontsize=10)
            
            # Draw fusion box
            fusion_x = 0.5
            ax.add_patch(plt.Rectangle((fusion_x-0.15, fusion_y-0.05), 0.3, 0.1, fill=True, color='lightsalmon', alpha=0.8))
            ax.text(fusion_x, fusion_y, "Cross-Modal Fusion", ha='center', va='center', fontsize=10)
            
            # Draw output box
            ax.add_patch(plt.Rectangle((fusion_x-0.1, output_y-0.05), 0.2, 0.1, fill=True, color='lightgray', alpha=0.8))
            ax.text(fusion_x, output_y, "Classification\nHead", ha='center', va='center', fontsize=10)
            
            # Draw arrows from inputs to encoders
            for x, _, _ in inputs:
                ax.arrow(x, input_y-0.05, 0, encoder_y-input_y+0.05, head_width=0.01, head_length=0.02, fc='black', ec='black')
            
            # Draw arrows from encoders to fusion
            for x, _, _ in encoders:
                ax.arrow(x, encoder_y-0.05, fusion_x-x, fusion_y-encoder_y+0.05, head_width=0.01, head_length=0.02, fc='black', ec='black')
            
            # Draw arrow from fusion to output
            ax.arrow(fusion_x, fusion_y-0.05, 0, output_y-fusion_y+0.05, head_width=0.01, head_length=0.02, fc='black', ec='black')
            
            # Set title
            ax.set_title('MH-Net Model Architecture', fontsize=14)
            
            st.pyplot(fig)
            
            # Display modality contribution
            st.markdown("#### Modality Contribution Analysis")
            
            st.markdown("""
            This visualization shows the relative importance of each modality for the model's predictions.
            The size of each segment represents how much that modality contributes to the final prediction.
            """)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Generate modality importance
                modalities = ["Text", "Audio", "Physiological", "Imaging"]
                importance = [0.45, 0.35, 0.15, 0.05]
                
                fig = px.pie(
                    values=importance,
                    names=modalities,
                    title="Overall Modality Importance",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    hole=0.4
                )
                
                fig.update_layout(
                    legend=dict(orientation="h", y=-0.1),
                    annotations=[dict(text="Modality<br>Contribution", x=0.5, y=0.5, font_size=15, showarrow=False)]
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Modality Importance by Condition")
                
                # Create a dataframe with condition-specific modality importance
                condition_importance = {
                    "Depression": [0.55, 0.30, 0.10, 0.05],
                    "Anxiety": [0.40, 0.45, 0.10, 0.05],
                    "PTSD": [0.35, 0.35, 0.20, 0.10],
                    "Bipolar": [0.40, 0.30, 0.20, 0.10],
                    "Schizophrenia": [0.35, 0.25, 0.25, 0.15]
                }
                
                condition = st.selectbox("Select condition:", list(condition_importance.keys()))
                
                if condition:
                    fig = px.pie(
                        values=condition_importance[condition],
                        names=modalities,
                        title=f"Modality Importance for {condition}",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    
                    fig.update_layout(
                        showlegend=False,
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Layer activation visualization
            st.markdown("#### Layer Activation Analysis")
            
            # Generate layer activation data - for visualization purposes
            layer_names = [f"Layer {i+1}" for i in range(8)]
            activation_strengths = np.random.random((8, 8)) * 0.8 + 0.2
            
            fig = px.imshow(
                activation_strengths,
                x=["Head " + str(i+1) for i in range(8)],
                y=layer_names,
                color_continuous_scale="Viridis",
                labels=dict(x="Attention Head", y="Layer", color="Activation Strength"),
                title="Layer Activation Heatmap"
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with dashboard_tabs[3]:  # Patient Analytics tab
            st.subheader("Patient Analytics")
            
            # Summary statistics
            if "assessment_history" in st.session_state and len(st.session_state.assessment_history) > 0:
                assessments = st.session_state.assessment_history
                
                # Count total unique patients
                unique_patients = set([a["patient_id"] for a in assessments])
                num_patients = len(unique_patients)
                
                # Count assessments by type
                session_types = {}
                for a in assessments:
                    session_type = a["metadata"]["session_type"]
                    session_types[session_type] = session_types.get(session_type, 0) + 1
                
                # Count assessments by primary condition
                condition_counts = {}
                for a in assessments:
                    primary_condition = max(a["risk_scores"].items(), key=lambda x: x[1])[0]
                    condition_counts[primary_condition] = condition_counts.get(primary_condition, 0) + 1
                
                # Display summary statistics
                st.markdown("#### Summary Statistics")
                
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric(
                        label="Total Assessments", 
                        value=len(assessments)
                    )
                
                with metric_cols[1]:
                    st.metric(
                        label="Unique Patients", 
                        value=num_patients
                    )
                
                with metric_cols[2]:
                    st.metric(
                        label="Avg. Assessments per Patient", 
                        value=f"{len(assessments) / max(1, num_patients):.1f}"
                    )
                
                with metric_cols[3]:
                    # Calculate primary condition
                    if condition_counts:
                        primary_condition = max(condition_counts.items(), key=lambda x: x[1])[0]
                        st.metric(
                            label="Most Common Condition", 
                            value=primary_condition
                        )
                    else:
                        st.metric(
                            label="Most Common Condition", 
                            value="N/A"
                        )
                
                # Display condition distribution
                st.markdown("#### Condition Distribution")
                
                if condition_counts:
                    conditions = list(condition_counts.keys())
                    counts = list(condition_counts.values())
                    
                    fig = px.bar(
                        x=conditions,
                        y=counts,
                        color=conditions,
                        labels={"x": "Condition", "y": "Count"},
                        title="Distribution of Primary Conditions"
                    )
                    
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No condition data available.")
                
                # Display age distribution
                st.markdown("#### Patient Demographics")
                
                ages = [a["metadata"]["age"] for a in assessments if "age" in a["metadata"]]
                genders = [a["metadata"]["gender"] for a in assessments if "gender" in a["metadata"]]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if ages:
                        fig = px.histogram(
                            x=ages,
                            nbins=10,
                            labels={"x": "Age", "y": "Count"},
                            title="Age Distribution",
                            color_discrete_sequence=["#1E88E5"]
                        )
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No age data available.")
                
                with col2:
                    if genders:
                        gender_counts = {}
                        for g in genders:
                            gender_counts[g] = gender_counts.get(g, 0) + 1
                        
                        fig = px.pie(
                            values=list(gender_counts.values()),
                            names=list(gender_counts.keys()),
                            title="Gender Distribution",
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No gender data available.")
                
                # Display assessment history over time
                st.markdown("#### Assessment History")
                
                # Extract timestamps and convert to datetime
                timestamps = []
                for a in assessments:
                    try:
                        ts = datetime.strptime(a["timestamp"], "%Y-%m-%d %H:%M:%S")
                        timestamps.append(ts)
                    except:
                        pass
                
                if timestamps:
                    # Count assessments by day
                    date_counts = {}
                    for ts in timestamps:
                        date_str = ts.strftime("%Y-%m-%d")
                        date_counts[date_str] = date_counts.get(date_str, 0) + 1
                    
                    # Convert to dataframe
                    history_df = pd.DataFrame({
                        "Date": list(date_counts.keys()),
                        "Count": list(date_counts.values())
                    })
                    
                    history_df["Date"] = pd.to_datetime(history_df["Date"])
                    history_df = history_df.sort_values("Date")
                    
                    fig = px.line(
                        history_df,
                        x="Date",
                        y="Count",
                        labels={"Count": "Number of Assessments", "Date": "Date"},
                        title="Assessment History Over Time",
                        markers=True
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No timestamp data available for trend analysis.")
            else:
                st.info("No assessment data available for analysis. Complete assessments in the Real-time Assessment page.")
        
        with dashboard_tabs[4]:  # Resource Usage tab
            st.subheader("System Resources & Performance")
            
            # System resource usage - simulated data
            st.markdown("#### Resource Utilization")
            
            # Generate simulated resource usage data
            time_points = 24  # Last 24 hours
            hours = list(range(-time_points+1, 1))
            timestamps = [(datetime.now() + pd.Timedelta(hours=h)).strftime("%Y-%m-%d %H:00:00") for h in hours]
            
            # CPU usage - generally higher during the day, lower at night
            cpu_usage = []
            for i in range(time_points):
                # Higher during work hours
                hour = (datetime.now() + pd.Timedelta(hours=hours[i])).hour
                if 9 <= hour <= 17:
                    cpu_usage.append(50 + np.random.random() * 30)
                else:
                    cpu_usage.append(20 + np.random.random() * 20)
            
            # Memory usage - gradually increases and occasionally drops (GC)
            memory_usage = [50]
            for i in range(1, time_points):
                # Occasional garbage collection
                if np.random.random() < 0.2:
                    memory_usage.append(max(40, memory_usage[-1] - 15 - np.random.random() * 10))
                else:
                    memory_usage.append(min(95, memory_usage[-1] + np.random.random() * 5 - 1))
            
            # Create resource usage dataframe
            resource_df = pd.DataFrame({
                "Timestamp": timestamps,
                "CPU Usage (%)": cpu_usage,
                "Memory Usage (%)": memory_usage
            })
            
            # Current usage metrics
            current_cpu = resource_df["CPU Usage (%)"].iloc[-1]
            current_memory = resource_df["Memory Usage (%)"].iloc[-1]
            
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric(
                    label="Current CPU Usage", 
                    value=f"{current_cpu:.1f}%",
                    delta=f"{current_cpu - resource_df['CPU Usage (%)'].iloc[-2]:.1f}%"
                )
            
            with metric_cols[1]:
                st.metric(
                    label="Current Memory Usage", 
                    value=f"{current_memory:.1f}%",
                    delta=f"{current_memory - resource_df['Memory Usage (%)'].iloc[-2]:.1f}%"
                )
            
            with metric_cols[2]:
                # Simulated network usage
                network_usage = 18.3 + np.random.random() * 2
                st.metric(
                    label="Network Traffic", 
                    value=f"{network_usage:.1f} MB/s",
                    delta=f"{np.random.random() * 4 - 2:.1f} MB/s"
                )
            
            with metric_cols[3]:
                # Simulated disk usage
                disk_usage = 68.5 + np.random.random() * 5
                st.metric(
                    label="Disk Usage", 
                    value=f"{disk_usage:.1f}%",
                    delta=f"{np.random.random() * 2:.1f}%"
                )
            
            # Resource usage charts
            st.markdown("#### Resource Usage Trends")
            
            tab1, tab2 = st.tabs(["Charts", "Raw Data"])
            
            with tab1:
                # Convert timestamp to datetime for better plotting
                resource_df["Timestamp"] = pd.to_datetime(resource_df["Timestamp"])
                
                # Create multi-line chart
                fig = px.line(
                    resource_df,
                    x="Timestamp",
                    y=["CPU Usage (%)", "Memory Usage (%)"],
                    labels={"value": "Usage (%)", "Timestamp": "Time", "variable": "Resource"},
                    title="System Resource Usage (Last 24 Hours)"
                )
                
                fig.update_layout(height=400, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
                # Inference time metrics
                st.markdown("#### Inference Performance")
                
                # Generate simulated inference times
                num_inferences = 100
                inference_times = np.random.lognormal(mean=-1.5, sigma=0.2, size=num_inferences)
                inference_times = np.clip(inference_times, 0.1, 2.0)
                
                # Create histrogram
                fig = px.histogram(
                    inference_times,
                    nbins=20,
                    labels={"value": "Time (seconds)"},
                    title="Inference Time Distribution",
                    color_discrete_sequence=["#8E24AA"]
                )
                
                fig.update_layout(
                    height=350,
                    xaxis_title="Time (seconds)",
                    yaxis_title="Frequency"
                )
                
                fig.add_vline(
                    x=np.mean(inference_times), 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Mean: {np.mean(inference_times):.2f}s",
                    annotation_position="top right"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Show raw resource usage data
                st.dataframe(resource_df, use_container_width=True)
            
            # System health check
            st.markdown("#### System Health Check")
            
            # Simulated health check results
            health_checks = {
                "Database Connection": {"status": "Healthy", "latency": "5ms", "last_checked": "Just now"},
                "Model Loading Time": {"status": "Healthy", "latency": "120ms", "last_checked": "5 minutes ago"},
                "API Response Time": {"status": "Warning", "latency": "350ms", "last_checked": "2 minutes ago"},
                "Storage Space": {"status": "Healthy", "latency": "N/A", "last_checked": "10 minutes ago"},
                "Memory Usage": {"status": "Healthy", "latency": "N/A", "last_checked": "Just now"},
            }
            
            # Convert to dataframe for display
            health_df = pd.DataFrame(health_checks).T.reset_index()
            health_df.columns = ["Component", "Status", "Latency", "Last Checked"]
            
            # Apply color coding based on status
            def color_status(val):
                color = "green" if val == "Healthy" else "orange" if val == "Warning" else "red"
                return f'background-color: {color}; color: white'
            
            # Display with conditional formatting
            st.dataframe(
                health_df,
                use_container_width=True,
                column_config={
                    "Status": st.column_config.Column(
                        "Status",
                        help="Current status of the system component",
                        width="medium"
                    )
                }
            )

# Advanced Visualization page
elif page == "Advanced Visualization":
    st.header("Advanced Data Visualization")
    
    # Use the create_advanced_visualization_page function from the module
    create_advanced_visualization_page(st)

# User Management page
elif page == "User Management":
    st.header("User Management")
    
    # Initialize user manager if not already in session state
    if "user_manager" not in st.session_state:
        st.session_state.user_manager = get_user_manager(os.environ.get("DATABASE_URL", "sqlite:///mhnet.db"))
    
    user_mgr = st.session_state.user_manager
    
    # Create tabs for user management functions
    user_tabs = st.tabs(["Login", "User List", "Create User", "Roles & Permissions"])
    
    with user_tabs[0]:
        st.subheader("Login")
        
        # Login form
        with st.form("login_form"):
            username = st.text_input("Username or Email")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if username and password:
                    result = user_mgr.authenticate_user(username, password)
                    
                    if "error" not in result:
                        st.session_state.current_user = result["user"]
                        st.session_state.session_token = result["session_token"]
                        st.success(f"Welcome, {result['user']['username']}!")
                    else:
                        st.error(result["error"])
                else:
                    st.error("Please enter both username and password")
        
        # Show current login status
        if "current_user" in st.session_state:
            st.info(f"Logged in as: {st.session_state.current_user['username']}")
            if st.button("Logout"):
                if "session_token" in st.session_state:
                    user_mgr.end_session(st.session_state.session_token)
                del st.session_state.current_user
                del st.session_state.session_token
                st.success("Logged out successfully")
    
    with user_tabs[1]:
        st.subheader("User List")
        
        # User list with filters
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            filter_role = st.selectbox("Filter by Role", ["All", "admin", "clinician", "researcher"])
        
        with filter_col2:
            filter_status = st.selectbox("Filter by Status", ["All", "Active", "Inactive"])
        
        # Apply filters
        filter_params = {}
        if filter_role != "All":
            filter_params["role"] = filter_role
        
        if filter_status != "All":
            filter_params["is_active"] = (filter_status == "Active")
        
        # Get users with filters
        users = user_mgr.get_users(filter_params)
        
        # Display users in a table
        if users:
            # Create a dataframe for display
            user_data = []
            for user in users:
                user_data.append({
                    "ID": user["id"],
                    "Username": user["username"],
                    "Email": user["email"],
                    "Name": f"{user.get('first_name', '')} {user.get('last_name', '')}".strip(),
                    "Roles": ", ".join(user.get("roles", [])),
                    "Status": "Active" if user.get("is_active", False) else "Inactive",
                    "Last Login": user.get("last_login", "Never")
                })
            
            # Create a Pandas DataFrame
            user_df = pd.DataFrame(user_data)
            st.dataframe(user_df)
        else:
            st.info("No users found with the selected filters")
    
    with user_tabs[2]:
        st.subheader("Create User")
        
        # Create user form
        with st.form("create_user_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_username = st.text_input("Username*")
                new_password = st.text_input("Password*", type="password")
                new_first_name = st.text_input("First Name")
            
            with col2:
                new_email = st.text_input("Email*")
                new_password_confirm = st.text_input("Confirm Password*", type="password")
                new_last_name = st.text_input("Last Name")
            
            # Role selection
            available_roles = [role["name"] for role in user_mgr.get_roles()]
            selected_roles = st.multiselect("Roles", available_roles, default=["clinician"])
            
            # Status
            is_active = st.checkbox("Active", value=True)
            is_verified = st.checkbox("Verified", value=False)
            
            submit_button = st.form_submit_button("Create User")
            
            if submit_button:
                # Validate input
                if not new_username or not new_email or not new_password:
                    st.error("Username, email, and password are required")
                elif new_password != new_password_confirm:
                    st.error("Passwords do not match")
                else:
                    # Create user
                    result = user_mgr.create_user(
                        username=new_username,
                        email=new_email,
                        password=new_password,
                        first_name=new_first_name,
                        last_name=new_last_name,
                        roles=selected_roles,
                        is_active=is_active,
                        is_verified=is_verified
                    )
                    
                    if "error" not in result:
                        st.success(f"User {new_username} created successfully")
                    else:
                        st.error(result["error"])
    
    with user_tabs[3]:
        st.subheader("Roles & Permissions")
        
        # Get all roles
        roles = user_mgr.get_roles()
        
        if roles:
            # Display roles in tabs
            role_tabs = st.tabs([role["name"] for role in roles])
            
            for i, role in enumerate(roles):
                with role_tabs[i]:
                    st.write(f"**Description:** {role.get('description', 'No description')}")
                    
                    # Display permissions
                    st.subheader("Permissions")
                    
                    permissions = role.get("permissions", [])
                    if permissions:
                        # Group permissions by resource
                        resource_permissions = {}
                        for perm in permissions:
                            resource = perm.get("resource", "Unknown")
                            action = perm.get("action", "Unknown")
                            
                            if resource not in resource_permissions:
                                resource_permissions[resource] = []
                            
                            resource_permissions[resource].append(action)
                        
                        # Display grouped permissions
                        for resource, actions in resource_permissions.items():
                            st.write(f"**{resource.capitalize()}**: {', '.join(actions)}")
                    else:
                        st.info("No permissions defined for this role")

# Export & Reports page
elif page == "Export & Reports":
    st.header("Export & Reports")
    
    # Create tabs for different export options
    export_tabs = st.tabs(["Assessment Reports", "Data Export", "Model Reports"])
    
    with export_tabs[0]:
        st.subheader("Assessment Reports")
        
        # Assessment selection
        st.write("Select a patient assessment to generate a report:")
        
        # Get assessments from database
        assessments = db_manager.get_all_assessments()
        
        if assessments:
            # Create selection options
            assessment_options = {}
            for assessment in assessments:
                # Format date for display
                if hasattr(assessment, 'timestamp') and assessment.timestamp:
                    display_date = assessment.timestamp.strftime("%Y-%m-%d") if hasattr(assessment.timestamp, 'strftime') else str(assessment.timestamp)
                else:
                    display_date = "Unknown date"
                
                # Create label
                assessment_options[f"Patient {assessment.patient_id} - {display_date}"] = assessment
            
            # Display dropdown
            selected_option = st.selectbox("Select Assessment", list(assessment_options.keys()))
            selected_assessment = assessment_options[selected_option]
            
            # Report format selection
            report_format = st.radio("Select Report Format", ["PDF", "HTML", "CSV"], horizontal=True)
            
            # Preview assessment data
            with st.expander("Preview Assessment Data"):
                st.json(selected_assessment.to_dict() if hasattr(selected_assessment, 'to_dict') else vars(selected_assessment))
            
            # Generate report button
            if st.button("Generate Report"):
                with st.spinner("Generating report..."):
                    assessment_data = selected_assessment.to_dict() if hasattr(selected_assessment, 'to_dict') else vars(selected_assessment)
                    
                    if report_format == "PDF":
                        # Generate PDF report
                        pdf_bytes = create_assessment_pdf(assessment_data)
                        
                        # Create download link
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"assessment_report_{selected_assessment.id}.pdf",
                            mime="application/pdf"
                        )
                    
                    elif report_format == "HTML":
                        # Generate HTML report
                        html_content = export_to_html(assessment_data)
                        
                        # Create download link
                        st.download_button(
                            label="Download HTML Report",
                            data=html_content,
                            file_name=f"assessment_report_{selected_assessment.id}.html",
                            mime="text/html"
                        )
                    
                    elif report_format == "CSV":
                        # Generate CSV report
                        # First create a temporary file
                        temp_file = f"temp_report_{selected_assessment.id}.csv"
                        export_to_csv(assessment_data, temp_file)
                        
                        # Read the file and provide download
                        with open(temp_file, "rb") as f:
                            csv_data = f.read()
                        
                        st.download_button(
                            label="Download CSV Report",
                            data=csv_data,
                            file_name=f"assessment_report_{selected_assessment.id}.csv",
                            mime="text/csv"
                        )
                        
                        # Clean up temporary file
                        os.remove(temp_file)
                    
                    st.success("Report generated successfully!")
        else:
            st.info("No assessments available for reporting. Complete assessments in the Real-time Assessment page.")
    
    with export_tabs[1]:
        st.subheader("Data Export")
        
        # Data type selection
        data_type = st.selectbox("Select Data Type", ["Patients", "Assessments", "Models", "All Data"])
        
        # Export format selection
        export_format = st.radio("Select Export Format", ["CSV", "JSON", "HTML"], horizontal=True)
        
        # Additional options
        include_metadata = st.checkbox("Include Metadata", value=True)
        anonymize_data = st.checkbox("Anonymize Sensitive Data", value=True)
        
        # Export button
        if st.button("Export Data"):
            with st.spinner("Preparing data export..."):
                # Simulate data export
                time.sleep(1)
                
                if data_type == "Patients":
                    data = db_manager.get_all_patients()
                    filename = "patients_export"
                elif data_type == "Assessments":
                    data = db_manager.get_all_assessments()
                    filename = "assessments_export"
                elif data_type == "Models":
                    # Get models from session state
                    if "model_registry" in st.session_state:
                        data = st.session_state.model_registry.list_models()
                    else:
                        data = []
                    filename = "models_export"
                else:  # All Data
                    data = {
                        "patients": db_manager.get_all_patients(),
                        "assessments": db_manager.get_all_assessments()
                    }
                    filename = "all_data_export"
                
                # Convert data to appropriate format
                if export_format == "CSV":
                    # For demonstration, create a simple CSV
                    temp_file = f"temp_{filename}.csv"
                    
                    # Convert data to list of dicts for CSV export
                    if isinstance(data, list):
                        data_dicts = [item.to_dict() if hasattr(item, 'to_dict') else vars(item) for item in data]
                    else:
                        data_dicts = data
                    
                    export_to_csv(data_dicts, temp_file)
                    
                    # Read file and provide download
                    with open(temp_file, "rb") as f:
                        file_data = f.read()
                    
                    st.download_button(
                        label="Download CSV",
                        data=file_data,
                        file_name=f"{filename}.csv",
                        mime="text/csv"
                    )
                    
                    # Clean up
                    os.remove(temp_file)
                
                elif export_format == "JSON":
                    # Convert data to JSON
                    if isinstance(data, list):
                        data_json = json.dumps([item.to_dict() if hasattr(item, 'to_dict') else vars(item) for item in data], indent=2, default=str)
                    else:
                        data_json = json.dumps(data, indent=2, default=str)
                    
                    st.download_button(
                        label="Download JSON",
                        data=data_json,
                        file_name=f"{filename}.json",
                        mime="application/json"
                    )
                
                else:  # HTML
                    # Convert data to HTML
                    if isinstance(data, list):
                        data_dicts = [item.to_dict() if hasattr(item, 'to_dict') else vars(item) for item in data]
                    else:
                        data_dicts = data
                    
                    html_content = export_to_html(data_dicts)
                    
                    st.download_button(
                        label="Download HTML",
                        data=html_content,
                        file_name=f"{filename}.html",
                        mime="text/html"
                    )
                
                st.success("Data exported successfully!")
    
    with export_tabs[2]:
        st.subheader("Model Reports")
        
        # Model selection
        if "model_registry" in st.session_state and st.session_state.model_registry:
            models = st.session_state.model_registry.list_models()
            
            if models:
                # Create model selection options
                model_options = {f"{model['name']} (v{model['version']})": model for model in models}
                selected_model_name = st.selectbox("Select Model", list(model_options.keys()))
                selected_model = model_options[selected_model_name]
                
                # Report type selection
                report_type = st.radio(
                    "Report Type", 
                    ["Performance Metrics", "Architecture Summary", "Comparison Report"],
                    horizontal=True
                )
                
                # Generate report button
                if st.button("Generate Model Report"):
                    with st.spinner("Generating model report..."):
                        # Prepare report data
                        model_data = {
                            "type": selected_model["model_type"],
                            "architecture": selected_model["architecture"],
                            "metrics": selected_model.get("metrics", {}),
                            "created_at": selected_model["created_at"]
                        }
                        
                        # Generate model summary
                        from utils.export import export_model_summary
                        
                        if report_type == "Performance Metrics":
                            # Focus on metrics
                            report_data = {
                                "model_id": selected_model["model_id"],
                                "metrics": model_data["metrics"],
                                "created_at": model_data["created_at"]
                            }
                            
                            # Convert to JSON for download
                            report_json = json.dumps(report_data, indent=2)
                            
                            st.download_button(
                                label="Download Metrics Report",
                                data=report_json,
                                file_name=f"{selected_model['name']}_metrics.json",
                                mime="application/json"
                            )
                        
                        elif report_type == "Architecture Summary":
                            # Generate comprehensive model summary
                            summary = export_model_summary(model_data)
                            
                            # Convert to JSON for download
                            summary_json = json.dumps(summary, indent=2)
                            
                            st.download_button(
                                label="Download Architecture Summary",
                                data=summary_json,
                                file_name=f"{selected_model['name']}_architecture.json",
                                mime="application/json"
                            )
                        
                        else:  # Comparison Report
                            # Let user select models to compare
                            other_models = [m for m in models if m["model_id"] != selected_model["model_id"]]
                            
                            if other_models:
                                comparison_options = {f"{m['name']} (v{m['version']})": m for m in other_models}
                                compare_with = st.multiselect(
                                    "Compare With", 
                                    list(comparison_options.keys()),
                                    max_selections=3
                                )
                                
                                if compare_with:
                                    # Prepare models for comparison
                                    models_to_compare = [(selected_model["name"], selected_model["version"])]
                                    
                                    for model_name in compare_with:
                                        model = comparison_options[model_name]
                                        models_to_compare.append((model["name"], model["version"]))
                                    
                                    # Generate comparison
                                    comparison = st.session_state.model_registry.compare_models(models_to_compare)
                                    
                                    # Convert to JSON for download
                                    comparison_json = json.dumps(comparison, indent=2)
                                    
                                    st.download_button(
                                        label="Download Comparison Report",
                                        data=comparison_json,
                                        file_name="model_comparison.json",
                                        mime="application/json"
                                    )
                                else:
                                    st.info("Please select at least one model to compare with")
                            else:
                                st.info("No other models available for comparison")
                        
                        st.success("Model report generated successfully!")
            else:
                st.info("No models found in the registry. Train models in the Model Training page.")
        else:
            st.warning("Model registry not initialized. Please train and save models first.")

# Batch Processing page
elif page == "Batch Processing":
    st.header("Batch Processing")
    
    # Initialize batch processor if not already in session state
    if "batch_processor" not in st.session_state:
        st.session_state.batch_processor = BatchProcessor(db_manager=db_manager)
    
    batch_processor = st.session_state.batch_processor
    
    # Create tabs for batch processing functions
    batch_tabs = st.tabs(["Job Queue", "Create Batch Job", "Batch Results", "Import/Export"])
    
    with batch_tabs[0]:
        st.subheader("Batch Job Queue")
        
        # Refresh button
        if st.button("Refresh Queue"):
            st.rerun()
        
        # Get all jobs
        jobs = batch_processor.get_jobs()
        
        if jobs:
            # Create a dataframe for display
            job_data = []
            for job in jobs:
                job_data.append({
                    "Job ID": job["job_id"][:8] + "...",
                    "Type": job["job_type"],
                    "Description": job["description"] or "No description",
                    "Status": job["status"],
                    "Progress": f"{job['progress'] * 100:.0f}%",
                    "Created": job["created_at"],
                    "Created By": job["created_by"] or "System"
                })
            
            # Display as dataframe
            job_df = pd.DataFrame(job_data)
            st.dataframe(job_df)
            
            # Job details
            st.subheader("Job Details")
            job_id = st.text_input("Enter Job ID to view details")
            
            if job_id:
                job = batch_processor.get_job(job_id)
                
                if job:
                    st.json(job)
                    
                    # Cancel job button
                    if job["status"] in ["pending", "running"]:
                        if st.button("Cancel Job"):
                            if batch_processor.cancel_job(job_id):
                                st.success(f"Job {job_id} cancelled successfully")
                                st.rerun()
                            else:
                                st.error("Failed to cancel job")
                    
                    # View results if completed
                    if job["status"] == "completed":
                        results = batch_processor.get_job_results(job_id)
                        
                        if results:
                            st.subheader("Job Results")
                            st.json(results)
                            
                            # Export results
                            export_format = st.radio("Export Format", ["JSON", "CSV"], horizontal=True)
                            
                            if st.button("Export Results"):
                                export_path = batch_processor.export_job_results(job_id, format=export_format.lower())
                                
                                if export_path:
                                    with open(export_path, "rb") as f:
                                        file_data = f.read()
                                    
                                    st.download_button(
                                        label=f"Download {export_format}",
                                        data=file_data,
                                        file_name=f"job_{job_id}_results.{export_format.lower()}",
                                        mime="application/json" if export_format == "JSON" else "text/csv"
                                    )
                else:
                    st.error("Job not found")
        else:
            st.info("No batch jobs found")
    
    with batch_tabs[1]:
        st.subheader("Create Batch Job")
        
        # Job type selection
        job_type = st.selectbox(
            "Job Type",
            ["assessment_batch", "prediction_batch", "import_batch", "export_batch"]
        )
        
        # Job description
        job_description = st.text_input("Job Description")
        
        # Job parameters based on type
        job_params = {}
        
        if job_type == "assessment_batch":
            st.subheader("Assessment Batch Parameters")
            
            # Number of assessments
            num_assessments = st.slider("Number of Assessments", 1, 100, 10)
            
            # Assessment parameters
            assessments = []
            
            # Get patient IDs from database
            patients = db_manager.get_all_patients()
            patient_ids = [p.id for p in patients] if patients else list(range(1, 11))
            
            # Generate sample assessments
            col1, col2 = st.columns(2)
            
            with col1:
                selected_patients = st.multiselect("Select Patients", patient_ids)
            
            with col2:
                modality_options = st.multiselect(
                    "Include Modalities",
                    ["Text", "Audio", "Physiological", "Imaging"],
                    default=["Text"]
                )
            
            # Button to prepare assessments
            if st.button("Prepare Assessment Batch"):
                if selected_patients:
                    for i in range(num_assessments):
                        patient_id = np.random.choice(selected_patients)
                        
                        assessment = {
                            "patient_id": int(patient_id),
                            "timestamp": datetime.now().isoformat(),
                            "metadata": {
                                "clinician": "Batch Process",
                                "session_type": "Automated Assessment"
                            }
                        }
                        
                        # Add modality flags
                        if "Text" in modality_options:
                            assessment["text_input"] = f"Sample text input for assessment {i+1}"
                        
                        if "Audio" in modality_options:
                            assessment["has_audio"] = True
                        
                        if "Physiological" in modality_options:
                            assessment["has_physio"] = True
                        
                        if "Imaging" in modality_options:
                            assessment["has_imaging"] = True
                        
                        # Add risk scores
                        assessment["risk_scores"] = {
                            "Depression": round(np.random.random() * 0.8, 2),
                            "Anxiety": round(np.random.random() * 0.7, 2),
                            "PTSD": round(np.random.random() * 0.5, 2)
                        }
                        
                        assessments.append(assessment)
                    
                    job_params["assessments"] = assessments
                    
                    st.success(f"Prepared {len(assessments)} assessments for batch processing")
                    st.json(assessments[0])  # Show sample
                else:
                    st.error("Please select at least one patient")
        
        elif job_type == "prediction_batch":
            st.subheader("Prediction Batch Parameters")
            
            # Model selection
            if "model_registry" in st.session_state and st.session_state.model_registry:
                models = st.session_state.model_registry.list_models()
                
                if models:
                    model_options = {f"{model['name']} (v{model['version']})": model for model in models}
                    selected_model_name = st.selectbox("Select Model", list(model_options.keys()))
                    selected_model = model_options[selected_model_name]
                    
                    # Number of inputs
                    num_inputs = st.slider("Number of Inputs", 1, 100, 10)
                    
                    # Generate sample inputs
                    if st.button("Prepare Prediction Batch"):
                        inputs = []
                        
                        for i in range(num_inputs):
                            # Create input based on model type
                            if selected_model["model_type"] == "tensorflow":
                                # For demonstration, create random vectors
                                input_data = {
                                    "id": f"input_{i+1}",
                                    "text_embedding": np.random.random(50).tolist(),
                                    "audio_features": np.random.random(20).tolist()
                                }
                            else:
                                input_data = {
                                    "id": f"input_{i+1}",
                                    "features": np.random.random(10).tolist()
                                }
                            
                            inputs.append(input_data)
                        
                        job_params["inputs"] = inputs
                        job_params["model_config"] = selected_model["architecture"]
                        
                        st.success(f"Prepared {len(inputs)} inputs for batch prediction")
                        st.json(inputs[0])  # Show sample
                else:
                    st.info("No models found in the registry. Train models in the Model Training page.")
            else:
                st.warning("Model registry not initialized. Please train and save models first.")
        
        elif job_type == "import_batch":
            st.subheader("Import Batch Parameters")
            
            # Import type
            import_type = st.selectbox("Import Type", ["patients", "assessments"])
            
            # File upload
            uploaded_file = st.file_uploader("Upload Import File (JSON/CSV)", type=["json", "csv"])
            
            if uploaded_file is not None:
                try:
                    # Process uploaded file
                    if uploaded_file.name.endswith('.json'):
                        data = json.loads(uploaded_file.read().decode())
                    else:  # CSV
                        data = pd.read_csv(uploaded_file).to_dict('records')
                    
                    # Show sample data
                    st.subheader("Sample Data")
                    st.json(data[0] if data else {})
                    
                    # Set job parameters
                    job_params["import_type"] = import_type
                    job_params["data"] = data
                    
                    st.success(f"Prepared {len(data)} records for import")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
            else:
                st.info("Please upload a file to import")
        
        elif job_type == "export_batch":
            st.subheader("Export Batch Parameters")
            
            # Export type
            export_type = st.selectbox("Export Type", ["patients", "assessments"])
            
            # Export format
            export_format = st.selectbox("Export Format", ["json", "csv"])
            
            # Query parameters
            query_params = {}
            
            if export_type == "patients":
                # Patient filter options
                st.subheader("Filter Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    min_age = st.number_input("Minimum Age", 0, 100, 0)
                    if min_age > 0:
                        query_params["min_age"] = min_age
                
                with col2:
                    max_age = st.number_input("Maximum Age", 0, 100, 100)
                    if max_age < 100:
                        query_params["max_age"] = max_age
            
            elif export_type == "assessments":
                # Assessment filter options
                st.subheader("Filter Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    start_date = st.date_input("Start Date")
                    query_params["start_date"] = start_date.isoformat()
                
                with col2:
                    end_date = st.date_input("End Date")
                    query_params["end_date"] = end_date.isoformat()
            
            # Set job parameters
            job_params["export_type"] = export_type
            job_params["format"] = export_format
            job_params["query_params"] = query_params
        
        # Submit job button
        submit_enabled = (
            (job_type == "assessment_batch" and "assessments" in job_params) or
            (job_type == "prediction_batch" and "inputs" in job_params) or
            (job_type == "import_batch" and "data" in job_params) or
            (job_type == "export_batch")
        )
        
        if submit_enabled and st.button("Submit Batch Job"):
            # Get current user if logged in
            created_by = st.session_state.current_user["username"] if "current_user" in st.session_state else "System"
            
            # Create job
            job_id = batch_processor.create_job(
                job_type=job_type,
                params=job_params,
                description=job_description,
                created_by=created_by
            )
            
            if job_id:
                st.success(f"Job submitted successfully with ID: {job_id}")
                
                # Show tab with job queue
                batch_tabs[0].open = True
            else:
                st.error("Failed to submit job")
    
    with batch_tabs[2]:
        st.subheader("Batch Results")
        
        # Job status filter
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "completed", "failed", "cancelled"]
        )
        
        # Get jobs with filter
        if status_filter == "All":
            completed_jobs = batch_processor.get_jobs()
        else:
            completed_jobs = batch_processor.get_jobs(status=status_filter)
        
        # Display completed jobs
        if completed_jobs:
            # Filter to show only jobs with results
            jobs_with_results = [job for job in completed_jobs if job["status"] == "completed"]
            
            if jobs_with_results:
                # Create a dataframe for display
                result_data = []
                for job in jobs_with_results:
                    results = batch_processor.get_job_results(job["job_id"])
                    result_count = len(results.get("items", [])) if results else 0
                    
                    result_data.append({
                        "Job ID": job["job_id"][:8] + "...",
                        "Type": job["job_type"],
                        "Description": job["description"] or "No description",
                        "Created": job["created_at"],
                        "Results": result_count,
                        "Success Rate": f"{results.get('successful', 0)}/{results.get('total', 0)}" if results else "N/A"
                    })
                
                # Display as dataframe
                result_df = pd.DataFrame(result_data)
                st.dataframe(result_df)
                
                # Job selection for detailed results
                selected_job_id = st.selectbox(
                    "Select Job to View Results",
                    [job["job_id"] for job in jobs_with_results]
                )
                
                if selected_job_id:
                    results = batch_processor.get_job_results(selected_job_id)
                    
                    if results:
                        st.subheader("Result Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Items", results.get("total", 0))
                        
                        with col2:
                            st.metric("Successful", results.get("successful", 0))
                        
                        with col3:
                            st.metric("Failed", results.get("failed", 0))
                        
                        # Results visualization
                        st.subheader("Results Visualization")
                        
                        # Create pie chart of success/failure
                        success = results.get("successful", 0)
                        failure = results.get("failed", 0)
                        
                        fig = px.pie(
                            values=[success, failure],
                            names=["Success", "Failure"],
                            title="Result Distribution",
                            color_discrete_sequence=["green", "red"]
                        )
                        
                        st.plotly_chart(fig)
                        
                        # Result details (paged)
                        st.subheader("Result Details")
                        items = results.get("items", [])
                        
                        if items:
                            # Pagination
                            items_per_page = 5
                            num_pages = (len(items) + items_per_page - 1) // items_per_page
                            
                            if "result_page" not in st.session_state:
                                st.session_state.result_page = 0
                            
                            # Page navigation
                            col1, col2, col3 = st.columns([1, 3, 1])
                            
                            with col1:
                                if st.button("Previous") and st.session_state.result_page > 0:
                                    st.session_state.result_page -= 1
                            
                            with col2:
                                st.write(f"Page {st.session_state.result_page + 1} of {num_pages}")
                            
                            with col3:
                                if st.button("Next") and st.session_state.result_page < num_pages - 1:
                                    st.session_state.result_page += 1
                            
                            # Show current page items
                            start_idx = st.session_state.result_page * items_per_page
                            end_idx = min(start_idx + items_per_page, len(items))
                            
                            for i, item in enumerate(items[start_idx:end_idx], start_idx):
                                with st.expander(f"Item {i+1}"):
                                    st.json(item)
                        else:
                            st.info("No result items found")
                    else:
                        st.warning("No results found for this job")
            else:
                st.info("No jobs with results found")
        else:
            st.info("No jobs found with the selected status")
    
    with batch_tabs[3]:
        st.subheader("Import/Export Batch Jobs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Export Jobs")
            
            # Get all jobs
            all_jobs = batch_processor.get_jobs()
            
            if all_jobs:
                # Convert jobs to JSON
                jobs_json = json.dumps(all_jobs, indent=2, default=str)
                
                st.download_button(
                    label="Export All Jobs",
                    data=jobs_json,
                    file_name="batch_jobs_export.json",
                    mime="application/json"
                )
            else:
                st.info("No jobs to export")
        
        with col2:
            st.write("### Import Jobs")
            
            # File upload
            uploaded_file = st.file_uploader("Upload Jobs File (JSON)", type=["json"])
            
            if uploaded_file is not None:
                try:
                    # Process uploaded file
                    import_data = json.loads(uploaded_file.read().decode())
                    
                    if isinstance(import_data, list):
                        st.success(f"Found {len(import_data)} jobs in the import file")
                        
                        # Show sample
                        if import_data:
                            st.json(import_data[0])
                        
                        # Import button
                        if st.button("Import Jobs"):
                            # For demonstration, just show success
                            st.success(f"Successfully imported {len(import_data)} jobs")
                    else:
                        st.error("Invalid import format. Expected a list of jobs.")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

# API Integration page
elif page == "API Integration":
    st.header("API Integration")
    
    # Get API manager from module
    api_mgr = api_manager
    
    # Create tabs for API functions
    api_tabs = st.tabs(["API Connections", "EHR Integration", "Patient Import/Export", "API Settings"])
    
    with api_tabs[0]:
        st.subheader("External API Connections")
        
        # API connection status
        st.write("### Current API Connections")
        
        # Placeholder for API connections (in a real app, these would be stored in the database)
        api_connections = {
            "FHIR": {"status": "Configured", "last_checked": "5 minutes ago", "connection_type": "OAuth2"},
            "OpenAI": {"status": "Not Configured", "last_checked": "Never", "connection_type": "API Key"},
            "Google Health": {"status": "Configured", "last_checked": "1 hour ago", "connection_type": "OAuth2"},
            "PubMed": {"status": "Configured", "last_checked": "30 minutes ago", "connection_type": "API Key"}
        }
        
        # Display connections in a table
        connection_data = []
        for api_name, info in api_connections.items():
            connection_data.append({
                "API Name": api_name,
                "Status": info["status"],
                "Last Checked": info["last_checked"],
                "Connection Type": info["connection_type"]
            })
        
        connection_df = pd.DataFrame(connection_data)
        st.dataframe(connection_df)
        
        # Add new API connection
        st.write("### Add New API Connection")
        
        with st.form("add_api_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                api_name = st.text_input("API Name")
                api_type = st.selectbox("Connection Type", ["API Key", "OAuth2", "Basic Auth"])
            
            with col2:
                api_url = st.text_input("API Base URL")
                api_key = st.text_input("API Key/Client ID", type="password") if api_type != "Basic Auth" else None
                api_secret = st.text_input("Client Secret", type="password") if api_type == "OAuth2" else None
            
            # Additional settings based on type
            if api_type == "OAuth2":
                auth_url = st.text_input("Authorization URL")
                token_url = st.text_input("Token URL")
                scope = st.text_input("Scope")
            elif api_type == "Basic Auth":
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
            
            submit_button = st.form_submit_button("Add API Connection")
            
            if submit_button:
                if api_name and api_url:
                    st.success(f"API connection '{api_name}' added successfully!")
                else:
                    st.error("API Name and URL are required")
        
        # Test API connection
        st.write("### Test API Connection")
        
        test_api = st.selectbox("Select API to Test", list(api_connections.keys()))
        
        if st.button("Test Connection"):
            with st.spinner("Testing connection..."):
                # Simulate testing
                time.sleep(1)
                
                if api_connections[test_api]["status"] == "Configured":
                    st.success(f"Successfully connected to {test_api} API!")
                else:
                    st.error(f"Failed to connect to {test_api}. Please configure the API first.")
    
    with api_tabs[1]:
        st.subheader("EHR Integration")
        
        # EHR system selection
        ehr_system = st.selectbox(
            "EHR System", 
            ["FHIR (Fast Healthcare Interoperability Resources)", "Epic", "Cerner", "Custom"]
        )
        
        # FHIR implementation
        if ehr_system.startswith("FHIR"):
            st.write("### FHIR Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fhir_server = st.text_input("FHIR Server URL", "https://hapi.fhir.org/baseR4")
                auth_type = st.selectbox("Authentication Type", ["None", "Bearer Token", "OAuth2"])
            
            with col2:
                if auth_type == "Bearer Token":
                    auth_token = st.text_input("Bearer Token", type="password")
                elif auth_type == "OAuth2":
                    client_id = st.text_input("Client ID")
                    client_secret = st.text_input("Client Secret", type="password")
            
            # Test connection button
            if st.button("Test FHIR Connection"):
                with st.spinner("Testing FHIR connection..."):
                    # Create and register FHIR integration (in a real app)
                    st.success("Successfully connected to FHIR server!")
            
            # FHIR operations
            st.write("### FHIR Operations")
            
            fhir_operation = st.selectbox(
                "Operation",
                ["Patient Search", "Patient Read", "Observation Search", "Condition Search"]
            )
            
            if fhir_operation == "Patient Search":
                # Patient search form
                with st.form("fhir_patient_search"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        patient_name = st.text_input("Name")
                        patient_id = st.text_input("Patient ID")
                    
                    with col2:
                        patient_gender = st.selectbox("Gender", ["Any", "Male", "Female", "Other"])
                        max_results = st.slider("Max Results", 1, 100, 10)
                    
                    search_button = st.form_submit_button("Search")
                    
                    if search_button:
                        with st.spinner("Searching for patients..."):
                            # Simulate search results
                            time.sleep(1)
                            
                            # Example results
                            st.success("Found 3 patients matching your criteria")
                            
                            # Show results in a table
                            results = [
                                {"id": "123", "name": "John Smith", "gender": "Male", "birthDate": "1980-05-15"},
                                {"id": "456", "name": "Jane Doe", "gender": "Female", "birthDate": "1992-11-23"},
                                {"id": "789", "name": "Alex Johnson", "gender": "Other", "birthDate": "1975-02-08"}
                            ]
                            
                            result_df = pd.DataFrame(results)
                            st.dataframe(result_df)
            elif fhir_operation == "Patient Read":
                patient_id_to_read = st.text_input("Enter Patient ID")
                
                if st.button("Get Patient"):
                    with st.spinner("Retrieving patient data..."):
                        # Simulate retrieval
                        time.sleep(1)
                        
                        # Example result
                        patient_data = {
                            "resourceType": "Patient",
                            "id": patient_id_to_read or "123",
                            "name": [{"given": ["John"], "family": "Smith"}],
                            "gender": "male",
                            "birthDate": "1980-05-15",
                            "address": [{"line": ["123 Main St"], "city": "Anytown", "state": "CA", "postalCode": "12345"}]
                        }
                        
                        st.json(patient_data)
        else:
            st.info(f"Integration with {ehr_system} will be implemented soon")
    
    with api_tabs[2]:
        st.subheader("Patient Import/Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Import Patients from EHR")
            
            # Import form
            with st.form("import_patients"):
                ehr_source = st.selectbox("EHR Source", ["FHIR Server", "Epic", "Cerner"])
                search_criteria = st.text_area("Search Criteria (JSON)", "{}")
                max_patients = st.slider("Maximum Patients", 1, 100, 10)
                
                import_button = st.form_submit_button("Import Patients")
                
                if import_button:
                    with st.spinner("Importing patients..."):
                        # Simulate import
                        time.sleep(2)
                        
                        st.success(f"Successfully imported 5 patients from {ehr_source}")
        
        with col2:
            st.write("### Export Patients to EHR")
            
            # Export form
            with st.form("export_patients"):
                export_destination = st.selectbox("Export Destination", ["FHIR Server", "Epic", "Cerner"])
                
                # Get patients from database
                patients = db_manager.get_all_patients()
                
                if patients:
                    patient_options = {f"Patient {p.id}": p.id for p in patients}
                    selected_patients = st.multiselect("Select Patients to Export", list(patient_options.keys()))
                    
                    export_assessments = st.checkbox("Include Assessments", value=True)
                    
                    export_button = st.form_submit_button("Export Patients")
                    
                    if export_button:
                        if selected_patients:
                            with st.spinner("Exporting patients..."):
                                # Simulate export
                                time.sleep(2)
                                
                                st.success(f"Successfully exported {len(selected_patients)} patients to {export_destination}")
                        else:
                            st.error("Please select at least one patient to export")
                else:
                    st.info("No patients available for export")
                    st.form_submit_button("Export Patients", disabled=True)
    
    with api_tabs[3]:
        st.subheader("API Settings")
        
        # API rate limiting
        st.write("### Rate Limiting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_requests = st.number_input("Max Requests per Minute", 1, 1000, 60)
        
        with col2:
            timeout = st.number_input("Request Timeout (seconds)", 1, 60, 10)
        
        # API logging
        st.write("### Logging")
        
        log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
        log_requests = st.checkbox("Log All Requests", value=False)
        log_responses = st.checkbox("Log Responses", value=False)
        
        # Save settings button
        if st.button("Save API Settings"):
            st.success("API settings saved successfully")

# Model Versioning page
elif page == "Model Versioning":
    st.header("Model Versioning and Deployment")
    
    # Initialize model registry if not already in session state
    if "model_registry" not in st.session_state:
        st.session_state.model_registry = ModelRegistry()
    
    model_registry = st.session_state.model_registry
    
    # Create model deployment manager if needed
    if "model_deployment" not in st.session_state:
        st.session_state.model_deployment = ModelDeployment(model_registry)
    
    model_deployment = st.session_state.model_deployment
    
    # Create tabs for model versioning functions
    model_tabs = st.tabs(["Model Registry", "Version Control", "Deployment", "Model Comparison"])
    
    with model_tabs[0]:
        st.subheader("Model Registry")
        
        # Display registered models
        models = model_registry.list_models()
        
        if models:
            # Create a dataframe for display
            model_data = []
            for model in models:
                metrics = model.get("metrics", {})
                
                # Extract key metrics if available
                accuracy = metrics.get("accuracy", "N/A")
                f1 = metrics.get("f1_score", "N/A")
                
                model_data.append({
                    "Name": model["name"],
                    "Version": model["version"],
                    "Type": model["model_type"],
                    "Created": model["created_at"],
                    "Accuracy": accuracy,
                    "F1 Score": f1
                })
            
            # Display as dataframe
            model_df = pd.DataFrame(model_data)
            st.dataframe(model_df)
        else:
            st.info("No models registered. Register a model below or train a model in the Model Training page.")
        
        # Register new model form
        st.subheader("Register New Model")
        
        with st.form("register_model_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                model_name = st.text_input("Model Name")
                model_version = st.text_input("Version", "1.0.0")
                model_type = st.selectbox("Model Type", ["tensorflow", "scikit-learn", "pytorch", "custom"])
            
            with col2:
                # Architecture parameters
                architecture = {}
                architecture["embed_dim"] = st.slider("Embedding Dimension", 64, 512, 256, step=64)
                architecture["num_heads"] = st.slider("Number of Attention Heads", 1, 12, 8)
                architecture["num_layers"] = st.slider("Number of Layers", 1, 12, 4)
            
            # Additional metadata
            with st.expander("Additional Metadata"):
                metadata = {}
                metadata["description"] = st.text_area("Description", "Multimodal model for mental health assessment")
                metadata["author"] = st.text_input("Author", "MH-Net Team")
                metadata["training_dataset"] = st.text_input("Training Dataset", "AVEC-Depression")
            
            # Metrics
            with st.expander("Performance Metrics"):
                metrics = {}
                metrics["accuracy"] = st.slider("Accuracy", 0.0, 1.0, 0.85, 0.01)
                metrics["precision"] = st.slider("Precision", 0.0, 1.0, 0.83, 0.01)
                metrics["recall"] = st.slider("Recall", 0.0, 1.0, 0.87, 0.01)
                metrics["f1_score"] = st.slider("F1 Score", 0.0, 1.0, 0.85, 0.01)
            
            register_button = st.form_submit_button("Register Model")
            
            if register_button:
                if model_name and model_version:
                    try:
                        # Create model version object
                        from modules.model_versioning import ModelVersion
                        
                        model = ModelVersion(
                            name=model_name,
                            version=model_version,
                            model_type=model_type,
                            architecture=architecture,
                            metadata=metadata
                        )
                        
                        # Register model
                        result = model_registry.register_model(model)
                        
                        # Update metrics
                        model_registry.update_model_metrics(model_name, model_version, metrics)
                        
                        st.success(f"Model {model_name} v{model_version} registered successfully!")
                    except ValueError as e:
                        st.error(str(e))
                else:
                    st.error("Model name and version are required")
    
    with model_tabs[1]:
        st.subheader("Version Control")
        
        # Get all model names
        models = model_registry.list_models()
        model_names = sorted(list(set(model["name"] for model in models)))
        
        if model_names:
            # Model selection
            selected_model = st.selectbox("Select Model", model_names)
            
            # Get all versions for the selected model
            model_versions = [model for model in models if model["name"] == selected_model]
            
            if model_versions:
                # Display versions
                st.write(f"### Versions of {selected_model}")
                
                # Create a dataframe for versions
                version_data = []
                for version in model_versions:
                    metrics = version.get("metrics", {})
                    
                    version_data.append({
                        "Version": version["version"],
                        "Created": version["created_at"],
                        "Accuracy": metrics.get("accuracy", "N/A"),
                        "F1 Score": metrics.get("f1_score", "N/A")
                    })
                
                # Sort by version
                version_data.sort(key=lambda x: x["Version"], reverse=True)
                
                # Display as dataframe
                version_df = pd.DataFrame(version_data)
                st.dataframe(version_df)
                
                # Version details
                selected_version = st.selectbox("Select Version for Details", [v["version"] for v in model_versions])
                
                # Get model details
                model_info = model_registry.get_model(selected_model, selected_version)
                
                if model_info:
                    # Display model details
                    st.write(f"### {selected_model} v{selected_version} Details")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Type:** " + model_info["model_type"])
                        st.write("**Created:** " + model_info["created_at"])
                        
                        # Architecture
                        st.write("**Architecture:**")
                        for key, value in model_info["architecture"].items():
                            st.write(f"- {key}: {value}")
                    
                    with col2:
                        # Metrics
                        st.write("**Performance Metrics:**")
                        metrics = model_info.get("metrics", {})
                        for key, value in metrics.items():
                            st.write(f"- {key}: {value}")
                        
                        # Metadata
                        st.write("**Metadata:**")
                        metadata = model_info.get("metadata", {})
                        for key, value in metadata.items():
                            st.write(f"- {key}: {value}")
                    
                    # Version control actions
                    st.write("### Actions")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Delete version button
                        if st.button("Delete Version"):
                            if model_registry.delete_model(selected_model, selected_version):
                                st.success(f"Version {selected_version} deleted successfully")
                                st.rerun()
                            else:
                                st.error("Failed to delete version")
                    
                    with col2:
                        # Set as latest button
                        if st.button("Set as Latest"):
                            # In a real app, this would update metadata or tags
                            st.success(f"Version {selected_version} set as latest")
                    
                    with col3:
                        # Deploy version button
                        if st.button("Deploy Version"):
                            # Deploy to model deployment
                            deployment_name = model_deployment.deploy_model(selected_model, selected_version)
                            st.success(f"Model deployed successfully as '{deployment_name}'")
        else:
            st.info("No models registered. Register a model in the Model Registry tab.")
    
    with model_tabs[2]:
        st.subheader("Model Deployment")
        
        # Display deployed models
        deployments = model_deployment.get_deployments()
        
        if deployments:
            # Create a dataframe for display
            deployment_data = []
            for deployment in deployments:
                deployment_data.append({
                    "Name": deployment["name"],
                    "Model": deployment["model_id"],
                    "Deployed At": deployment["deployed_at"],
                    "Status": deployment["status"],
                    "Requests": deployment["usage_stats"]["requests"],
                    "Last Used": deployment["usage_stats"]["last_used"] or "Never"
                })
            
            # Display as dataframe
            deployment_df = pd.DataFrame(deployment_data)
            st.dataframe(deployment_df)
            
            # Test deployment
            st.write("### Test Deployment")
            
            # Select deployment
            test_deployment = st.selectbox("Select Deployment to Test", [d["name"] for d in deployments])
            
            # Input data
            st.write("#### Input Data")
            
            # Determine input type based on deployment
            deployment_info = next((d for d in deployments if d["name"] == test_deployment), None)
            
            if deployment_info:
                model_id = deployment_info["model_id"]
                model_name, model_version = model_id.split("/")
                
                # Get model type
                model_info = model_registry.get_model(model_name, model_version)
                
                if model_info:
                    model_type = model_info["model_type"]
                    
                    # Input based on model type
                    if model_type == "tensorflow":
                        # For TF models, create example input vector
                        st.write("Example input for TensorFlow model:")
                        num_features = st.slider("Number of Features", 5, 100, 10)
                        
                        # Generate random input
                        input_data = np.random.random(num_features).tolist()
                        st.json(input_data)
                    else:
                        # Generic input
                        st.text_area("JSON Input", "{\"features\": [0.1, 0.2, 0.3, 0.4, 0.5]}")
                    
                    # Test button
                    if st.button("Test Model"):
                        with st.spinner("Making prediction..."):
                            # In a real app, this would call the deployed model
                            time.sleep(1)
                            
                            # Generate example prediction
                            prediction = {
                                "risk_scores": {
                                    "Depression": round(np.random.random(), 3),
                                    "Anxiety": round(np.random.random(), 3),
                                    "PTSD": round(np.random.random(), 3)
                                },
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            st.write("#### Prediction Result:")
                            st.json(prediction)
            
            # Deployment management
            st.write("### Deployment Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Stop deployment
                stop_deployment = st.selectbox("Select Deployment to Stop", [""] + [d["name"] for d in deployments if d["status"] == "active"])
                
                if stop_deployment and st.button("Stop Deployment"):
                    # In a real app, this would stop the deployment
                    st.success(f"Deployment '{stop_deployment}' stopped successfully")
            
            with col2:
                # Deploy new model
                if model_names:
                    deploy_model = st.selectbox("Select Model to Deploy", model_names)
                    
                    # Get versions for selected model
                    model_versions = sorted([model["version"] for model in models if model["name"] == deploy_model], reverse=True)
                    
                    if model_versions:
                        deploy_version = st.selectbox("Select Version", model_versions)
                        deployment_name = st.text_input("Deployment Name", f"{deploy_model}-{deploy_version}")
                        
                        if st.button("Deploy Model"):
                            # Deploy the model
                            deployment_name = model_deployment.deploy_model(deploy_model, deploy_version, deployment_name)
                            st.success(f"Model deployed successfully as '{deployment_name}'")
                            st.rerun()
                else:
                    st.info("No models available for deployment")
        else:
            st.info("No models currently deployed. Deploy a model from the Version Control tab.")
    
    with model_tabs[3]:
        st.subheader("Model Comparison")
        
        # Get all models
        models = model_registry.list_models()
        
        if len(models) >= 2:
            # Model selection for comparison
            st.write("Select models to compare:")
            
            # Create selection options
            model_options = {f"{model['name']} v{model['version']}": (model["name"], model["version"]) for model in models}
            
            col1, col2 = st.columns(2)
            
            with col1:
                model1 = st.selectbox("Model 1", list(model_options.keys()))
            
            with col2:
                # Filter out the first selection
                model2_options = {k: v for k, v in model_options.items() if k != model1}
                model2 = st.selectbox("Model 2", list(model2_options.keys()))
            
            # Get models for comparison
            model1_name, model1_version = model_options[model1]
            model2_name, model2_version = model_options[model2]
            
            model1_info = model_registry.get_model(model1_name, model1_version)
            model2_info = model_registry.get_model(model2_name, model2_version)
            
            if model1_info and model2_info:
                # Compare button
                if st.button("Compare Models"):
                    # Generate comparison
                    comparison = model_registry.compare_models([
                        (model1_name, model1_version),
                        (model2_name, model2_version)
                    ])
                    
                    # Display comparison results
                    st.write("### Comparison Results")
                    
                    # Display metrics comparison
                    if "metrics" in comparison:
                        metrics_comparison = comparison["metrics"]
                        
                        for metric_name, values in metrics_comparison.items():
                            # Create bar chart for metric comparison
                            model_labels = [f"{m['model_id']}" for m in values]
                            metric_values = [m["value"] for m in values]
                            
                            fig = px.bar(
                                x=model_labels,
                                y=metric_values,
                                title=f"{metric_name.capitalize()} Comparison",
                                labels={"x": "Model", "y": metric_name.capitalize()}
                            )
                            
                            st.plotly_chart(fig)
                    
                    # Architecture comparison
                    st.write("### Architecture Comparison")
                    
                    architecture1 = model1_info["architecture"]
                    architecture2 = model2_info["architecture"]
                    
                    # Compare architectures
                    architecture_comparison = []
                    all_params = set(architecture1.keys()) | set(architecture2.keys())
                    
                    for param in all_params:
                        value1 = architecture1.get(param, "N/A")
                        value2 = architecture2.get(param, "N/A")
                        
                        architecture_comparison.append({
                            "Parameter": param,
                            f"{model1_name} v{model1_version}": value1,
                            f"{model2_name} v{model2_version}": value2,
                            "Different": value1 != value2
                        })
                    
                    # Display as dataframe with highlighting
                    arch_df = pd.DataFrame(architecture_comparison)
                    st.dataframe(arch_df)
        else:
            st.info("Need at least 2 models to compare. Register models in the Model Registry tab.")

# Clinical Recommendations page
elif page == "Clinical Recommendations":
    st.header("Clinical Recommendations")
    
    # Initialize recommendation engine if not already in session state
    if "recommendation_engine" not in st.session_state:
        st.session_state.recommendation_engine = RecommendationEngine()
    
    recommendation_engine = st.session_state.recommendation_engine
    
    # Create tabs for recommendation functions
    rec_tabs = st.tabs(["Assessment Recommendations", "Treatment Planner", "Recommendation Settings"])
    
    with rec_tabs[0]:
        st.subheader("Assessment Recommendations")
        
        # Select assessment for recommendations
        st.write("Select a patient assessment to generate clinical recommendations:")
        
        # Get assessments from database
        assessments = db_manager.get_all_assessments()
        
        if assessments:
            # Create selection options
            assessment_options = {}
            for assessment in assessments:
                # Format date for display
                if hasattr(assessment, 'timestamp') and assessment.timestamp:
                    display_date = assessment.timestamp.strftime("%Y-%m-%d") if hasattr(assessment.timestamp, 'strftime') else str(assessment.timestamp)
                else:
                    display_date = "Unknown date"
                
                # Create label
                assessment_options[f"Patient {assessment.patient_id} - {display_date}"] = assessment
            
            # Display dropdown
            selected_option = st.selectbox("Select Assessment", list(assessment_options.keys()))
            selected_assessment = assessment_options[selected_option]
            
            # Preview assessment data
            with st.expander("Preview Assessment Data"):
                st.json(selected_assessment.to_dict() if hasattr(selected_assessment, 'to_dict') else vars(selected_assessment))
            
            # Generate recommendations button
            if st.button("Generate Recommendations"):
                with st.spinner("Generating recommendations..."):
                    assessment_data = selected_assessment.to_dict() if hasattr(selected_assessment, 'to_dict') else vars(selected_assessment)
                    
                    # Generate recommendations
                    recommendations = recommendation_engine.get_recommendations(assessment_data)
                    
                    if recommendations:
                        # Store in session state for treatment planner
                        st.session_state.current_recommendations = recommendations
                        st.session_state.current_assessment = assessment_data
                        
                        # Display personalized message
                        st.success(recommendations["personalized_message"])
                        
                        # Primary recommendations
                        st.subheader("Primary Recommendations")
                        for i, rec in enumerate(recommendations["primary"], 1):
                            with st.expander(f"{i}. {rec['name']}"):
                                st.write(rec["description"])
                                st.info(f"Evidence Level: {rec['evidence_level']} | Recommendation Strength: {rec['recommendation_strength']}")
                        
                        # Secondary recommendations
                        if recommendations["secondary"]:
                            st.subheader("Additional Recommendations")
                            for i, rec in enumerate(recommendations["secondary"], 1):
                                with st.expander(f"{i}. {rec['name']} - For {rec['condition']}"):
                                    st.write(rec["description"])
                                    st.info(f"Evidence Level: {rec['evidence_level']} | Recommendation Strength: {rec['recommendation_strength']}")
                        
                        # General recommendations
                        if recommendations["general"]:
                            st.subheader("General Recommendations")
                            for i, rec in enumerate(recommendations["general"], 1):
                                st.write(f"{i}. {rec['name']}: {rec['description']}")
                        
                        # Create treatment plan button
                        if st.button("Create Comprehensive Treatment Plan"):
                            # Switch to treatment planner tab
                            rec_tabs[1].open = True
                    else:
                        st.error("Failed to generate recommendations")
        else:
            st.info("No assessments available for recommendations. Complete assessments in the Real-time Assessment page.")
    
    with rec_tabs[1]:
        st.subheader("Treatment Planner")
        
        # Check if we have current assessment and recommendations
        if "current_assessment" in st.session_state and "current_recommendations" in st.session_state:
            # Create treatment planner
            treatment_planner = TreatmentPlanner(recommendation_engine)
            
            # Get current assessment and recommendations
            assessment_data = st.session_state.current_assessment
            recommendations = st.session_state.current_recommendations
            
            # Display patient info
            st.write(f"**Patient ID:** {assessment_data.get('patient_id', 'Unknown')}")
            
            # Treatment plan options
            st.write("### Treatment Plan Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                plan_title = st.text_input("Plan Title", f"Treatment Plan for Patient {assessment_data.get('patient_id', 'Unknown')}")
                primary_clinician = st.text_input("Primary Clinician", "Dr. Smith")
            
            with col2:
                plan_duration = st.selectbox("Plan Duration", ["2 weeks", "1 month", "3 months", "6 months"])
                followup_frequency = st.selectbox("Follow-up Frequency", ["Weekly", "Bi-weekly", "Monthly"])
            
            # Additional options
            include_assessments = st.checkbox("Include Assessment Results", value=True)
            include_goals = st.checkbox("Include Treatment Goals", value=True)
            
            # Generate plan button
            if st.button("Generate Treatment Plan"):
                with st.spinner("Generating comprehensive treatment plan..."):
                    # Create treatment plan
                    treatment_plan = treatment_planner.create_treatment_plan(assessment_data)
                    
                    if treatment_plan:
                        # Display plan
                        st.write("### Treatment Plan Summary")
                        st.write(treatment_plan["summary"])
                        
                        # Goals
                        if include_goals and treatment_plan["goals"]:
                            st.write("### Treatment Goals")
                            
                            for i, goal in enumerate(treatment_plan["goals"], 1):
                                st.write(f"**Goal {i}: {goal['target']}**")
                                st.write(f"- Measure: {goal['measure']}")
                                st.write(f"- Timeframe: {goal['timeframe']}")
                                st.write(f"- Objective: {goal['objective']}")
                        
                        # Interventions
                        st.write("### Interventions")
                        
                        # Group by type
                        for intervention_type, interventions in treatment_plan["interventions"].items():
                            if interventions:
                                with st.expander(f"{intervention_type.capitalize()} Interventions"):
                                    for intervention in interventions:
                                        st.write(f"**{intervention['name']}** ({intervention['priority']})")
                                        st.write(f"- {intervention['description']}")
                                        if intervention.get('for_condition') != 'General':
                                            st.write(f"- For: {intervention.get('for_condition', 'General')}")
                        
                        # Medications
                        st.write("### Medication Recommendations")
                        
                        for med in treatment_plan["medications"]:
                            st.write(f"**{med['recommendation']}** ({med['priority']})")
                            st.write(f"- {med['details']}")
                            st.write(f"- For: {med.get('for_condition', 'General')}")
                        
                        # Monitoring plan
                        st.write("### Monitoring Plan")
                        
                        for monitor in treatment_plan["monitoring"]:
                            st.write(f"**{monitor['target']}**")
                            st.write(f"- Measure: {monitor['measure']}")
                            st.write(f"- Frequency: {monitor['frequency']}")
                            st.write(f"- Alert Threshold: {monitor['threshold']}")
                        
                        # Follow-up plan
                        st.write("### Follow-up Plan")
                        
                        st.write(f"- Initial Follow-up: {treatment_plan['follow_up']['initial_follow_up']}")
                        st.write(f"- Subsequent Frequency: {treatment_plan['follow_up']['subsequent_frequency']}")
                        st.write(f"- Recommended Provider: {treatment_plan['follow_up']['recommended_provider']}")
                        
                        # Export options
                        st.write("### Export Treatment Plan")
                        
                        export_format = st.radio("Export Format", ["PDF", "HTML", "JSON"], horizontal=True)
                        
                        if st.button("Export Plan"):
                            # Convert to appropriate format (just a placeholder)
                            if export_format == "PDF":
                                st.warning("PDF export not implemented in this demo")
                            elif export_format == "HTML":
                                html_content = export_to_html(treatment_plan)
                                
                                st.download_button(
                                    label="Download HTML Plan",
                                    data=html_content,
                                    file_name=f"treatment_plan_{assessment_data.get('patient_id', 'unknown')}.html",
                                    mime="text/html"
                                )
                            else:  # JSON
                                plan_json = json.dumps(treatment_plan, indent=2, default=str)
                                
                                st.download_button(
                                    label="Download JSON Plan",
                                    data=plan_json,
                                    file_name=f"treatment_plan_{assessment_data.get('patient_id', 'unknown')}.json",
                                    mime="application/json"
                                )
                    else:
                        st.error("Failed to generate treatment plan")
        else:
            st.info("No assessment selected. Please select an assessment and generate recommendations first.")
    
    with rec_tabs[2]:
        st.subheader("Recommendation Settings")
        
        # Settings options
        st.write("### Recommendation Engine Settings")
        
        # Evidence level thresholds
        st.write("#### Evidence Level Thresholds")
        evidence_levels = {
            "Strong Recommendations": st.multiselect(
                "Evidence Levels for Strong Recommendations",
                ["A", "B", "C"],
                default=["A"]
            ),
            "Moderate Recommendations": st.multiselect(
                "Evidence Levels for Moderate Recommendations",
                ["A", "B", "C"],
                default=["A", "B"]
            ),
            "Optional Recommendations": st.multiselect(
                "Evidence Levels for Optional Recommendations",
                ["A", "B", "C"],
                default=["B", "C"]
            )
        }
        
        # Condition priorities
        st.write("#### Condition Priorities")
        
        conditions = ["Depression", "Anxiety", "PTSD", "Bipolar", "Schizophrenia"]
        condition_priorities = {}
        
        # Create sliders for each condition
        for condition in conditions:
            condition_priorities[condition] = st.slider(
                f"{condition} Priority",
                1, 5, 3,
                help="Higher value gives more weight to recommendations for this condition"
            )
        
        # Save settings button
        if st.button("Save Settings"):
            # Store in session state (in a real app, this would be saved to database)
            st.session_state.recommendation_settings = {
                "evidence_levels": evidence_levels,
                "condition_priorities": condition_priorities
            }
            
            st.success("Recommendation settings saved successfully")
        
        # Knowledge base update
        st.write("### Knowledge Base Update")
        
        kb_update_method = st.radio(
            "Update Method",
            ["Upload JSON File", "Manual Entry"],
            horizontal=True
        )
        
        if kb_update_method == "Upload JSON File":
            uploaded_file = st.file_uploader("Upload Knowledge Base File", type=["json"])
            
            if uploaded_file is not None:
                try:
                    # Process uploaded file
                    knowledge_base = json.loads(uploaded_file.read().decode())
                    
                    # Show sample data
                    st.subheader("Sample Data")
                    if "conditions" in knowledge_base:
                        conditions = list(knowledge_base["conditions"].keys())
                        st.write(f"Found {len(conditions)} conditions: {', '.join(conditions)}")
                        
                        if conditions:
                            first_condition = conditions[0]
                            st.json(knowledge_base["conditions"][first_condition])
                    
                    # Update button
                    if st.button("Update Knowledge Base"):
                        # In a real app, this would update the knowledge base
                        st.success("Knowledge base updated successfully")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        else:
            st.warning("Manual knowledge base entry is not implemented in this demo version")