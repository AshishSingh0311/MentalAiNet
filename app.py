import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from modules.data_loader import MultimodalDataLoader
from modules.preprocessor import MultimodalPreprocessor
from modules.models import MHNetModel
from modules.training import ModelTrainer
from modules.evaluation import ModelEvaluator
from modules.explainability import ExplainabilityEngine
from modules.visualization import (
    plot_model_performance,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_attention_maps,
    plot_embedding_visualization
)

# Set page configuration
st.set_page_config(
    page_title="MH-Net: Multimodal Mental Health Diagnostics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for model tracking
if 'model' not in st.session_state:
    st.session_state.model = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'explainability_results' not in st.session_state:
    st.session_state.explainability_results = None

# Main header
st.title("🧠 MH-Net: Multimodal Mental Health Diagnostics Framework")
st.markdown("""
This application provides a comprehensive framework for multimodal deep learning in mental health diagnostics with explainable AI capabilities.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Introduction", "Data Loading", "Preprocessing", "Model Training", "Evaluation", "Explainability", "Dashboard"]
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
    if st.button("Load Data"):
        try:
            with st.spinner("Loading data..."):
                # Initialize data loader
                data_loader = MultimodalDataLoader(
                    dataset_name=selected_dataset,
                    include_text=text_data,
                    include_audio=audio_data,
                    include_physiological=physiological_data,
                    include_imaging=imaging_data
                )
                
                # Load data
                dataset = data_loader.load_dataset()
                
                # Store in session state
                st.session_state.dataset = dataset
                
                # Display dataset summary
                st.success("Data loaded successfully!")
                
                # Show summary statistics
                st.subheader("Dataset Summary")
                summary = {
                    "Total samples": len(dataset["labels"]) if "labels" in dataset else 0,
                    "Classes": np.unique(dataset["labels"]).tolist() if "labels" in dataset else [],
                    "Text samples": len(dataset["text"]) if "text" in dataset else 0,
                    "Audio samples": len(dataset["audio"]) if "audio" in dataset else 0,
                    "Physiological samples": len(dataset["physiological"]) if "physiological" in dataset else 0,
                    "Imaging samples": len(dataset["imaging"]) if "imaging" in dataset else 0
                }
                
                st.json(summary)
                
                # Show class distribution if labels are available
                if "labels" in dataset and len(dataset["labels"]) > 0:
                    st.subheader("Class Distribution")
                    labels = dataset["labels"]
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    
                    fig = px.pie(
                        values=counts,
                        names=unique_labels,
                        title="Class Distribution"
                    )
                    st.plotly_chart(fig)
        
        except Exception as e:
            st.error(f"Error loading data: {e}")

# Preprocessing page
elif page == "Preprocessing":
    st.header("Data Preprocessing")
    
    if st.session_state.dataset is None:
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
        
        # Physiological signal preprocessing options
        if "physiological" in dataset and len(dataset["physiological"]) > 0:
            st.markdown("#### Physiological Signal Preprocessing")
            physio_filter = st.checkbox("Apply bandpass filtering", value=True)
            physio_normalize = st.checkbox("Normalize signals", value=True)
            physio_artifact_removal = st.checkbox("Remove artifacts", value=True)
        
        # Imaging preprocessing options
        if "imaging" in dataset and len(dataset["imaging"]) > 0:
            st.markdown("#### Imaging Preprocessing")
            image_normalize = st.checkbox("Normalize voxel intensities", value=True)
            image_registration = st.checkbox("Apply registration", value=True)
            image_skull_strip = st.checkbox("Apply skull stripping", value=True)
        
        # General preprocessing options
        st.markdown("#### General Preprocessing")
        split_ratio = st.slider("Train-Validation-Test Split Ratio", 0.1, 0.9, (0.7, 0.15, 0.15))
        balance_classes = st.checkbox("Balance classes", value=True)
        augmentation = st.checkbox("Apply data augmentation", value=False)
        
        # Preprocess data button
        if st.button("Preprocess Data"):
            try:
                with st.spinner("Preprocessing data..."):
                    # Initialize preprocessor
                    preprocessor = MultimodalPreprocessor(
                        text_options={
                            "lowercase": text_lower if "text" in dataset else False,
                            "remove_stopwords": text_remove_stopwords if "text" in dataset else False,
                            "stemming": text_stemming if "text" in dataset else False,
                            "lemmatization": text_lemmatization if "text" in dataset else False
                        },
                        audio_options={
                            "normalize": audio_normalize if "audio" in dataset else False,
                            "noise_reduction": audio_noise_reduction if "audio" in dataset else False,
                            "feature_extraction": audio_feature_extraction if "audio" in dataset else "MFCC"
                        },
                        physiological_options={
                            "filter": physio_filter if "physiological" in dataset else False,
                            "normalize": physio_normalize if "physiological" in dataset else False,
                            "artifact_removal": physio_artifact_removal if "physiological" in dataset else False
                        },
                        imaging_options={
                            "normalize": image_normalize if "imaging" in dataset else False,
                            "registration": image_registration if "imaging" in dataset else False,
                            "skull_strip": image_skull_strip if "imaging" in dataset else False
                        },
                        general_options={
                            "train_ratio": split_ratio[0],
                            "val_ratio": split_ratio[1],
                            "test_ratio": split_ratio[2],
                            "balance_classes": balance_classes,
                            "augmentation": augmentation
                        }
                    )
                    
                    # Preprocess data
                    preprocessed_data = preprocessor.preprocess(dataset)
                    
                    # Store in session state
                    st.session_state.preprocessed_data = preprocessed_data
                    
                    # Display preprocessing summary
                    st.success("Data preprocessed successfully!")
                    
                    # Show summary statistics
                    st.subheader("Preprocessed Data Summary")
                    summary = {
                        "Train samples": len(preprocessed_data["train"]["labels"]) if "labels" in preprocessed_data["train"] else 0,
                        "Validation samples": len(preprocessed_data["val"]["labels"]) if "labels" in preprocessed_data["val"] else 0,
                        "Test samples": len(preprocessed_data["test"]["labels"]) if "labels" in preprocessed_data["test"] else 0,
                    }
                    
                    st.json(summary)
                    
                    # Show train set class distribution
                    if "labels" in preprocessed_data["train"] and len(preprocessed_data["train"]["labels"]) > 0:
                        st.subheader("Train Set Class Distribution")
                        labels = preprocessed_data["train"]["labels"]
                        unique_labels, counts = np.unique(labels, return_counts=True)
                        
                        fig = px.bar(
                            x=unique_labels,
                            y=counts,
                            title="Train Set Class Distribution"
                        )
                        st.plotly_chart(fig)
            
            except Exception as e:
                st.error(f"Error preprocessing data: {e}")

# Model Training page
elif page == "Model Training":
    st.header("Model Training")
    
    if st.session_state.preprocessed_data is None:
        st.warning("Please preprocess your data first in the 'Preprocessing' page.")
    else:
        preprocessed_data = st.session_state.preprocessed_data
        
        st.subheader("Model Configuration")
        
        # Model architecture selection
        model_type = st.selectbox(
            "Select model architecture",
            ["MH-Net (Multimodal Transformer)", "Unimodal (Text Only)", "Unimodal (Audio Only)", 
             "Unimodal (Physiological Only)", "Unimodal (Imaging Only)"]
        )
        
        # Architecture parameters
        st.markdown("#### Architecture Parameters")
        embed_dim = st.slider("Embedding dimension", 64, 512, 256, step=64)
        num_heads = st.slider("Number of attention heads", 2, 16, 8, step=2)
        num_layers = st.slider("Number of transformer layers", 1, 12, 4, step=1)
        dropout_rate = st.slider("Dropout rate", 0.0, 0.5, 0.1, step=0.1)
        
        # Training parameters
        st.markdown("#### Training Parameters")
        batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64, 128], value=32)
        learning_rate = st.select_slider(
            "Learning rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001
        )
        num_epochs = st.slider("Number of epochs", 5, 100, 30, step=5)
        early_stopping = st.checkbox("Use early stopping", value=True)
        patience = st.slider("Patience for early stopping", 3, 20, 10, step=1) if early_stopping else 0
        
        # Loss function and optimizer
        loss_function = st.selectbox(
            "Loss function",
            ["Categorical Cross Entropy", "Binary Cross Entropy", "Focal Loss"]
        )
        optimizer = st.selectbox(
            "Optimizer",
            ["Adam", "SGD", "RMSprop", "AdamW"]
        )
        
        # Train model button
        if st.button("Train Model"):
            try:
                with st.spinner("Training model... This may take some time."):
                    # Create model instance
                    model = MHNetModel(
                        model_type=model_type,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        dropout_rate=dropout_rate,
                        input_shapes={
                            "text": preprocessed_data["train"]["text"].shape[1:] if "text" in preprocessed_data["train"] else None,
                            "audio": preprocessed_data["train"]["audio"].shape[1:] if "audio" in preprocessed_data["train"] else None,
                            "physiological": preprocessed_data["train"]["physiological"].shape[1:] if "physiological" in preprocessed_data["train"] else None,
                            "imaging": preprocessed_data["train"]["imaging"].shape[1:] if "imaging" in preprocessed_data["train"] else None
                        },
                        num_classes=len(np.unique(preprocessed_data["train"]["labels"]))
                    )
                    
                    # Initialize trainer
                    trainer = ModelTrainer(
                        model=model,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        num_epochs=num_epochs,
                        early_stopping=early_stopping,
                        patience=patience,
                        loss=loss_function,
                        optimizer=optimizer
                    )
                    
                    # Train model
                    training_history = trainer.train(
                        train_data=preprocessed_data["train"],
                        val_data=preprocessed_data["val"]
                    )
                    
                    # Store model and training history in session state
                    st.session_state.model = model
                    st.session_state.training_history = training_history
                    
                    # Display training summary
                    st.success("Model trained successfully!")
                    
                    # Plot training history
                    st.subheader("Training History")
                    
                    # Plot training and validation loss
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(training_history.history['loss'], label='Training Loss')
                    ax.plot(training_history.history['val_loss'], label='Validation Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Plot training and validation accuracy
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(training_history.history['accuracy'], label='Training Accuracy')
                    ax.plot(training_history.history['val_accuracy'], label='Validation Accuracy')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Accuracy')
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Display final metrics
                    final_train_loss = training_history.history['loss'][-1]
                    final_train_acc = training_history.history['accuracy'][-1]
                    final_val_loss = training_history.history['val_loss'][-1]
                    final_val_acc = training_history.history['val_accuracy'][-1]
                    
                    metrics = {
                        "Final Training Loss": round(final_train_loss, 4),
                        "Final Training Accuracy": round(final_train_acc, 4),
                        "Final Validation Loss": round(final_val_loss, 4),
                        "Final Validation Accuracy": round(final_val_acc, 4)
                    }
                    
                    st.json(metrics)
            
            except Exception as e:
                st.error(f"Error training model: {e}")

# Evaluation page
elif page == "Evaluation":
    st.header("Model Evaluation")
    
    if st.session_state.model is None or st.session_state.preprocessed_data is None:
        st.warning("Please train a model first in the 'Model Training' page.")
    else:
        model = st.session_state.model
        preprocessed_data = st.session_state.preprocessed_data
        
        st.subheader("Evaluation Options")
        
        # Evaluation dataset selection
        eval_dataset = st.selectbox(
            "Select evaluation dataset",
            ["Test set", "Validation set", "Training set"]
        )
        
        dataset_mapping = {
            "Test set": "test",
            "Validation set": "val",
            "Training set": "train"
        }
        
        # Evaluation metrics selection
        st.markdown("#### Evaluation Metrics")
        accuracy = st.checkbox("Accuracy", value=True)
        precision = st.checkbox("Precision", value=True)
        recall = st.checkbox("Recall", value=True)
        f1_score = st.checkbox("F1 Score", value=True)
        roc_auc = st.checkbox("ROC AUC", value=True)
        confusion_matrix = st.checkbox("Confusion Matrix", value=True)
        
        # Cross-validation options
        use_cv = st.checkbox("Use cross-validation", value=False)
        if use_cv:
            n_folds = st.slider("Number of folds", 3, 10, 5, step=1)
        
        # Evaluate model button
        if st.button("Evaluate Model"):
            try:
                with st.spinner("Evaluating model..."):
                    # Initialize evaluator
                    evaluator = ModelEvaluator(
                        model=model,
                        metrics={
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1_score,
                            "roc_auc": roc_auc,
                            "confusion_matrix": confusion_matrix
                        }
                    )
                    
                    # Select evaluation dataset
                    eval_data = preprocessed_data[dataset_mapping[eval_dataset]]
                    
                    # Evaluate model
                    if use_cv:
                        evaluation_results = evaluator.cross_validate(
                            data=eval_data,
                            n_folds=n_folds
                        )
                    else:
                        evaluation_results = evaluator.evaluate(
                            data=eval_data
                        )
                    
                    # Store results in session state
                    st.session_state.evaluation_results = evaluation_results
                    
                    # Display evaluation summary
                    st.success("Model evaluated successfully!")
                    
                    # Display metrics
                    st.subheader("Evaluation Metrics")
                    
                    # Create metrics display
                    metrics_to_display = {}
                    for metric, value in evaluation_results.items():
                        if metric != "confusion_matrix" and metric != "predictions" and metric != "true_labels":
                            if use_cv:
                                metrics_to_display[metric] = f"{np.mean(value):.4f} ± {np.std(value):.4f}"
                            else:
                                metrics_to_display[metric] = f"{value:.4f}"
                    
                    # Display metrics in columns
                    cols = st.columns(3)
                    for i, (metric, value) in enumerate(metrics_to_display.items()):
                        cols[i % 3].metric(metric.capitalize(), value)
                    
                    # Display confusion matrix if available
                    if confusion_matrix and "confusion_matrix" in evaluation_results:
                        st.subheader("Confusion Matrix")
                        fig = plot_confusion_matrix(evaluation_results["confusion_matrix"])
                        st.pyplot(fig)
                    
                    # Display ROC curve if available
                    if roc_auc and "predictions" in evaluation_results and "true_labels" in evaluation_results:
                        st.subheader("ROC Curve")
                        fig = plot_model_performance(
                            evaluation_results["predictions"],
                            evaluation_results["true_labels"],
                            metric="roc"
                        )
                        st.pyplot(fig)
                    
                    # Display precision-recall curve if available
                    if precision and recall and "predictions" in evaluation_results and "true_labels" in evaluation_results:
                        st.subheader("Precision-Recall Curve")
                        fig = plot_model_performance(
                            evaluation_results["predictions"],
                            evaluation_results["true_labels"],
                            metric="precision_recall"
                        )
                        st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Error evaluating model: {e}")

# Explainability page
elif page == "Explainability":
    st.header("Model Explainability")
    
    if st.session_state.model is None or st.session_state.preprocessed_data is None:
        st.warning("Please train a model first in the 'Model Training' page.")
    else:
        model = st.session_state.model
        preprocessed_data = st.session_state.preprocessed_data
        
        st.subheader("Explainability Options")
        
        # Dataset selection for explanation
        dataset_selection = st.selectbox(
            "Select dataset for explanation",
            ["Test set", "Validation set", "Training set"]
        )
        
        dataset_mapping = {
            "Test set": "test",
            "Validation set": "val",
            "Training set": "train"
        }
        
        selected_data = preprocessed_data[dataset_mapping[dataset_selection]]
        
        # Sample selection
        sample_idx = st.number_input(
            "Select sample index for explanation",
            min_value=0,
            max_value=len(selected_data["labels"]) - 1 if "labels" in selected_data else 0,
            value=0,
            step=1
        )
        
        # Explainability method selection
        explain_method = st.selectbox(
            "Select explainability method",
            ["LIME", "SHAP", "Attention Visualization", "Integrated Gradients", "All"]
        )
        
        # Modality selection for explanation
        available_modalities = []
        if "text" in selected_data:
            available_modalities.append("Text")
        if "audio" in selected_data:
            available_modalities.append("Audio")
        if "physiological" in selected_data:
            available_modalities.append("Physiological")
        if "imaging" in selected_data:
            available_modalities.append("Imaging")
        
        explain_modality = st.selectbox(
            "Select modality for explanation",
            available_modalities + ["All"],
            index=len(available_modalities)
        )
        
        # Generate explanation button
        if st.button("Generate Explanation"):
            try:
                with st.spinner("Generating explanation..."):
                    # Initialize explainability engine
                    explainer = ExplainabilityEngine(
                        model=model,
                        method=explain_method
                    )
                    
                    # Prepare sample for explanation
                    sample = {}
                    for modality in ["text", "audio", "physiological", "imaging"]:
                        if modality in selected_data:
                            sample[modality] = selected_data[modality][sample_idx:sample_idx+1]
                    
                    true_label = selected_data["labels"][sample_idx]
                    
                    # Generate explanation
                    modalities_to_explain = []
                    if explain_modality == "All":
                        modalities_to_explain = [m.lower() for m in available_modalities]
                    else:
                        modalities_to_explain = [explain_modality.lower()]
                    
                    explanation = explainer.explain(
                        sample=sample,
                        modalities=modalities_to_explain
                    )
                    
                    # Store explanation in session state
                    st.session_state.explainability_results = explanation
                    
                    # Display explanation
                    st.success("Explanation generated successfully!")
                    
                    # Display sample information
                    st.subheader("Sample Information")
                    st.write(f"True label: {true_label}")
                    
                    # Get model prediction for this sample
                    prediction = model.predict(sample)
                    predicted_label = np.argmax(prediction[0])
                    confidence = prediction[0][predicted_label]
                    
                    st.write(f"Predicted label: {predicted_label}")
                    st.write(f"Confidence: {confidence:.4f}")
                    
                    # Display explanations based on method and modality
                    if "text" in modalities_to_explain and "text" in explanation:
                        st.subheader("Text Explanation")
                        text_explanation = explanation["text"]
                        
                        if explain_method in ["LIME", "SHAP", "All"]:
                            # Display text feature importance
                            st.markdown("#### Feature Importance")
                            fig = plot_feature_importance(text_explanation["feature_importance"])
                            st.pyplot(fig)
                            
                            # Display highlighted text
                            st.markdown("#### Highlighted Text")
                            st.markdown(text_explanation["highlighted_text"], unsafe_allow_html=True)
                        
                        if explain_method in ["Attention Visualization", "All"]:
                            # Display attention maps
                            st.markdown("#### Attention Visualization")
                            fig = plot_attention_maps(text_explanation["attention_weights"])
                            st.pyplot(fig)
                    
                    if "audio" in modalities_to_explain and "audio" in explanation:
                        st.subheader("Audio Explanation")
                        audio_explanation = explanation["audio"]
                        
                        if explain_method in ["LIME", "SHAP", "All"]:
                            # Display audio feature importance
                            st.markdown("#### Feature Importance")
                            fig = plot_feature_importance(audio_explanation["feature_importance"])
                            st.pyplot(fig)
                            
                            # Display spectrogram with highlights
                            st.markdown("#### Spectrogram Highlights")
                            fig = audio_explanation["spectrogram_highlight"]
                            st.pyplot(fig)
                        
                        if explain_method in ["Attention Visualization", "All"]:
                            # Display attention maps
                            st.markdown("#### Attention Visualization")
                            fig = plot_attention_maps(audio_explanation["attention_weights"])
                            st.pyplot(fig)
                    
                    if "physiological" in modalities_to_explain and "physiological" in explanation:
                        st.subheader("Physiological Signal Explanation")
                        physio_explanation = explanation["physiological"]
                        
                        if explain_method in ["LIME", "SHAP", "All"]:
                            # Display physiological feature importance
                            st.markdown("#### Feature Importance")
                            fig = plot_feature_importance(physio_explanation["feature_importance"])
                            st.pyplot(fig)
                            
                            # Display signal with highlights
                            st.markdown("#### Signal Highlights")
                            fig = physio_explanation["signal_highlight"]
                            st.pyplot(fig)
                    
                    if "imaging" in modalities_to_explain and "imaging" in explanation:
                        st.subheader("Imaging Explanation")
                        imaging_explanation = explanation["imaging"]
                        
                        if explain_method in ["LIME", "SHAP", "All"]:
                            # Display saliency map
                            st.markdown("#### Saliency Map")
                            fig = imaging_explanation["saliency_map"]
                            st.pyplot(fig)
                            
                            # Display feature importance
                            st.markdown("#### Region Importance")
                            fig = imaging_explanation["region_importance"]
                            st.pyplot(fig)
                        
                        if explain_method in ["Attention Visualization", "All"]:
                            # Display attention maps
                            st.markdown("#### Attention Visualization")
                            fig = plot_attention_maps(imaging_explanation["attention_weights"])
                            st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Error generating explanation: {e}")

# Dashboard page
elif page == "Dashboard":
    st.header("Dashboard")
    
    if st.session_state.preprocessed_data is None:
        st.warning("Please preprocess data first in the 'Preprocessing' page.")
    elif st.session_state.model is None:
        st.warning("Please train a model first in the 'Model Training' page.")
    else:
        preprocessed_data = st.session_state.preprocessed_data
        model = st.session_state.model
        
        # Dashboard tabs
        tabs = st.tabs(["Model Performance", "Data Exploration", "Explanations", "Clinical Insights"])
        
        # Model Performance tab
        with tabs[0]:
            st.subheader("Model Performance")
            
            if st.session_state.evaluation_results is not None:
                evaluation_results = st.session_state.evaluation_results
                
                # Performance metrics
                if isinstance(evaluation_results.get("accuracy"), float):
                    # Display metrics in columns if they exist
                    col1, col2, col3 = st.columns(3)
                    
                    if "accuracy" in evaluation_results:
                        col1.metric("Accuracy", f"{evaluation_results['accuracy']:.4f}")
                    
                    if "precision" in evaluation_results:
                        col2.metric("Precision", f"{evaluation_results['precision']:.4f}")
                    
                    if "recall" in evaluation_results:
                        col3.metric("Recall", f"{evaluation_results['recall']:.4f}")
                    
                    col1, col2 = st.columns(2)
                    
                    if "f1_score" in evaluation_results:
                        col1.metric("F1 Score", f"{evaluation_results['f1_score']:.4f}")
                    
                    if "roc_auc" in evaluation_results:
                        col2.metric("ROC AUC", f"{evaluation_results['roc_auc']:.4f}")
                
                # Display confusion matrix if available
                if "confusion_matrix" in evaluation_results:
                    st.subheader("Confusion Matrix")
                    fig = plot_confusion_matrix(evaluation_results["confusion_matrix"])
                    st.pyplot(fig)
                
                # Display ROC curve if available
                if "predictions" in evaluation_results and "true_labels" in evaluation_results:
                    st.subheader("ROC Curve")
                    fig = plot_model_performance(
                        evaluation_results["predictions"],
                        evaluation_results["true_labels"],
                        metric="roc"
                    )
                    st.pyplot(fig)
            else:
                st.info("No evaluation results available. Please evaluate the model in the 'Evaluation' page.")
        
        # Data Exploration tab
        with tabs[1]:
            st.subheader("Data Exploration")
            
            # Dataset selection
            dataset_selection = st.selectbox(
                "Select dataset",
                ["Training set", "Validation set", "Test set"]
            )
            
            dataset_mapping = {
                "Training set": "train",
                "Validation set": "val",
                "Test set": "test"
            }
            
            selected_data = preprocessed_data[dataset_mapping[dataset_selection]]
            
            # Class distribution
            if "labels" in selected_data:
                st.markdown("#### Class Distribution")
                labels = selected_data["labels"]
                unique_labels, counts = np.unique(labels, return_counts=True)
                
                fig = px.bar(
                    x=unique_labels,
                    y=counts,
                    title=f"Class Distribution in {dataset_selection}"
                )
                st.plotly_chart(fig)
            
            # Feature exploration
            st.markdown("#### Feature Exploration")
            
            available_modalities = []
            if "text" in selected_data:
                available_modalities.append("Text")
            if "audio" in selected_data:
                available_modalities.append("Audio")
            if "physiological" in selected_data:
                available_modalities.append("Physiological")
            if "imaging" in selected_data:
                available_modalities.append("Imaging")
            
            modality = st.selectbox("Select modality", available_modalities)
            
            if modality == "Text" and "text" in selected_data:
                # Text length distribution
                st.markdown("##### Text Length Distribution")
                text_lengths = [len(text) for text in selected_data["text_raw"]] if "text_raw" in selected_data else []
                
                if text_lengths:
                    fig = px.histogram(
                        x=text_lengths,
                        nbins=30,
                        title="Text Length Distribution"
                    )
                    st.plotly_chart(fig)
                else:
                    st.info("Raw text data not available.")
            
            elif modality == "Audio" and "audio" in selected_data:
                # Audio feature visualization
                st.markdown("##### Audio Feature Visualization")
                
                if "audio_features" in selected_data:
                    # Display sample audio features
                    sample_idx = st.slider("Select sample index", 0, len(selected_data["audio"]) - 1, 0)
                    
                    fig = px.imshow(
                        selected_data["audio_features"][sample_idx].T,
                        aspect="auto",
                        title="Audio Features (e.g., MFCC or Mel Spectrogram)"
                    )
                    st.plotly_chart(fig)
                else:
                    st.info("Processed audio features not available.")
            
            elif modality == "Physiological" and "physiological" in selected_data:
                # Physiological signal visualization
                st.markdown("##### Physiological Signal Visualization")
                
                sample_idx = st.slider("Select sample index", 0, len(selected_data["physiological"]) - 1, 0)
                
                if selected_data["physiological"].ndim > 2:
                    # Multiple channels
                    num_channels = selected_data["physiological"].shape[1]
                    channel_idx = st.slider("Select channel", 0, num_channels - 1, 0)
                    
                    signal = selected_data["physiological"][sample_idx, channel_idx]
                    
                    fig = px.line(
                        x=np.arange(len(signal)),
                        y=signal,
                        title=f"Physiological Signal (Channel {channel_idx})"
                    )
                    st.plotly_chart(fig)
                else:
                    # Single channel
                    signal = selected_data["physiological"][sample_idx]
                    
                    fig = px.line(
                        x=np.arange(len(signal)),
                        y=signal,
                        title="Physiological Signal"
                    )
                    st.plotly_chart(fig)
            
            elif modality == "Imaging" and "imaging" in selected_data:
                # Imaging visualization
                st.markdown("##### Imaging Visualization")
                
                sample_idx = st.slider("Select sample index", 0, len(selected_data["imaging"]) - 1, 0)
                
                if selected_data["imaging"].ndim > 3:
                    # 3D image (e.g., fMRI)
                    image = selected_data["imaging"][sample_idx]
                    
                    # Select slice
                    slice_axis = st.selectbox("Select slice axis", ["x", "y", "z"])
                    axis_idx = {"x": 0, "y": 1, "z": 2}[slice_axis]
                    
                    slice_idx = st.slider(
                        f"Select {slice_axis}-axis slice",
                        0,
                        image.shape[axis_idx] - 1,
                        image.shape[axis_idx] // 2
                    )
                    
                    # Extract slice
                    if slice_axis == "x":
                        slice_data = image[slice_idx, :, :]
                    elif slice_axis == "y":
                        slice_data = image[:, slice_idx, :]
                    else:
                        slice_data = image[:, :, slice_idx]
                    
                    fig = px.imshow(
                        slice_data,
                        title=f"Brain Image ({slice_axis}-axis, slice {slice_idx})"
                    )
                    st.plotly_chart(fig)
                else:
                    # 2D image
                    image = selected_data["imaging"][sample_idx]
                    
                    fig = px.imshow(
                        image,
                        title="Brain Image"
                    )
                    st.plotly_chart(fig)
        
        # Explanations tab
        with tabs[2]:
            st.subheader("Explanation Visualization")
            
            if st.session_state.explainability_results is not None:
                explanation = st.session_state.explainability_results
                
                # Get available modalities in the explanation
                available_modalities = [m for m in ["text", "audio", "physiological", "imaging"] if m in explanation]
                
                if available_modalities:
                    # Select modality for visualization
                    modality = st.selectbox(
                        "Select modality for explanation visualization",
                        [m.capitalize() for m in available_modalities]
                    ).lower()
                    
                    if modality in explanation:
                        modality_explanation = explanation[modality]
                        
                        # Display explanation components
                        if "feature_importance" in modality_explanation:
                            st.markdown("#### Feature Importance")
                            fig = plot_feature_importance(modality_explanation["feature_importance"])
                            st.pyplot(fig)
                        
                        if "attention_weights" in modality_explanation:
                            st.markdown("#### Attention Visualization")
                            fig = plot_attention_maps(modality_explanation["attention_weights"])
                            st.pyplot(fig)
                        
                        # Modality-specific visualizations
                        if modality == "text" and "highlighted_text" in modality_explanation:
                            st.markdown("#### Highlighted Text")
                            st.markdown(modality_explanation["highlighted_text"], unsafe_allow_html=True)
                        
                        elif modality == "audio" and "spectrogram_highlight" in modality_explanation:
                            st.markdown("#### Spectrogram Highlights")
                            fig = modality_explanation["spectrogram_highlight"]
                            st.pyplot(fig)
                        
                        elif modality == "physiological" and "signal_highlight" in modality_explanation:
                            st.markdown("#### Signal Highlights")
                            fig = modality_explanation["signal_highlight"]
                            st.pyplot(fig)
                        
                        elif modality == "imaging" and "saliency_map" in modality_explanation:
                            st.markdown("#### Saliency Map")
                            fig = modality_explanation["saliency_map"]
                            st.pyplot(fig)
                            
                            if "region_importance" in modality_explanation:
                                st.markdown("#### Region Importance")
                                fig = modality_explanation["region_importance"]
                                st.pyplot(fig)
                else:
                    st.info("No explanations available for visualization.")
            else:
                st.info("No explanation results available. Please generate explanations in the 'Explainability' page.")
        
        # Clinical Insights tab
        with tabs[3]:
            st.subheader("Clinical Insights")
            
            if st.session_state.model is not None and st.session_state.evaluation_results is not None:
                # Model insights
                st.markdown("#### Model Performance by Condition")
                
                # Example plot showing condition-specific performance
                conditions = ["MDD", "GAD", "Bipolar", "PTSD", "Schizophrenia"]
                if "class_metrics" in st.session_state.evaluation_results:
                    class_metrics = st.session_state.evaluation_results["class_metrics"]
                    
                    metric_type = st.selectbox(
                        "Select metric",
                        ["Precision", "Recall", "F1 Score", "ROC AUC"]
                    )
                    
                    metric_mapping = {
                        "Precision": "precision",
                        "Recall": "recall",
                        "F1 Score": "f1_score",
                        "ROC AUC": "roc_auc"
                    }
                    
                    metric_key = metric_mapping[metric_type]
                    
                    if metric_key in class_metrics:
                        values = class_metrics[metric_key]
                        
                        fig = px.bar(
                            x=conditions[:len(values)],
                            y=values,
                            title=f"{metric_type} by Mental Health Condition",
                            labels={"x": "Condition", "y": metric_type}
                        )
                        st.plotly_chart(fig)
                else:
                    # Generate sample data for visualization
                    values = np.random.uniform(0.7, 0.95, len(conditions))
                    
                    fig = px.bar(
                        x=conditions,
                        y=values,
                        title="Example: F1 Score by Mental Health Condition",
                        labels={"x": "Condition", "y": "F1 Score"}
                    )
                    st.plotly_chart(fig)
                    
                    st.info("This is an example visualization. To see actual metrics by condition, run a model evaluation with class-specific metrics.")
                
                # Feature importance across conditions
                st.markdown("#### Key Features by Condition")
                
                if "feature_importance_by_class" in st.session_state.evaluation_results:
                    feature_importance = st.session_state.evaluation_results["feature_importance_by_class"]
                    
                    selected_condition = st.selectbox(
                        "Select condition",
                        conditions
                    )
                    
                    condition_idx = conditions.index(selected_condition)
                    
                    if condition_idx < len(feature_importance):
                        cond_importance = feature_importance[condition_idx]
                        
                        fig = px.bar(
                            x=cond_importance["features"],
                            y=cond_importance["importance"],
                            title=f"Key Features for {selected_condition}"
                        )
                        st.plotly_chart(fig)
                else:
                    st.info("Feature importance by condition not available. Generate explainability results with class-specific feature importance to see this visualization.")
                
                # Clinical decision support
                st.markdown("#### Clinical Decision Support")
                
                st.write("""
                **Potential Clinical Applications:**
                
                1. **Early Screening**: The model can assist in early screening for mental health conditions, particularly in resource-limited settings.
                
                2. **Differential Diagnosis**: The model can help differentiate between conditions with similar presentations, such as distinguishing bipolar disorder from major depression.
                
                3. **Treatment Response Prediction**: With additional data, the model could be extended to predict response to different treatment modalities.
                
                4. **Relapse Prediction**: By analyzing patterns in longitudinal data, the model could potentially identify early warning signs of relapse.
                
                **Important Clinical Considerations:**
                
                - This model is intended as a clinical decision support tool, not a replacement for clinical judgment.
                - Explainability features are essential for clinician trust and understanding of model recommendations.
                - Longitudinal validation is necessary before widespread clinical implementation.
                - Patient privacy and data security must be prioritized in any real-world deployment.
                """)
                
                # Risk stratification visualization
                st.markdown("#### Risk Stratification Example")
                
                risk_levels = ["Low", "Moderate", "High", "Severe"]
                risk_counts = [25, 40, 20, 15]
                
                fig = px.pie(
                    values=risk_counts,
                    names=risk_levels,
                    title="Example: Patient Risk Stratification",
                    color_discrete_sequence=px.colors.sequential.RdBu[::-1]
                )
                st.plotly_chart(fig)
                
                st.info("This is an example visualization. In a clinical implementation, this would show the distribution of patients by risk level as determined by the model.")
