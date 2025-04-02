import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Optional, Union, Any
import plotly.express as px
import plotly.graph_objects as go

def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None) -> plt.Figure:
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        
    Returns:
        Matplotlib figure containing the confusion matrix visualization
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    # Set labels
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_model_performance(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    metric: str = "roc",
    class_names: Optional[List[str]] = None
) -> plt.Figure:
    """
    Plot model performance curves (ROC or Precision-Recall).
    
    Args:
        predictions: Predicted probabilities or scores
        true_labels: True class labels
        metric: Type of curve to plot ('roc' or 'precision_recall')
        class_names: List of class names
        
    Returns:
        Matplotlib figure containing the performance curve
    """
    # One-hot encode true labels if needed
    if len(true_labels.shape) == 1:
        n_classes = predictions.shape[1] if len(predictions.shape) > 1 else 2
        true_labels_one_hot = np.eye(n_classes)[true_labels.astype(int)]
    else:
        true_labels_one_hot = true_labels
        n_classes = true_labels_one_hot.shape[1]
    
    # If predictions is 1D, convert to 2D
    if len(predictions.shape) == 1:
        predictions = np.column_stack([1 - predictions, predictions])
    
    # Initialize figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # Plot ROC curve
    if metric == "roc":
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        
        # Plot ROC curve for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(true_labels_one_hot[:, i], predictions[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    # Plot Precision-Recall curve
    elif metric == "precision_recall":
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        
        # Plot PR curve for each class
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(true_labels_one_hot[:, i], predictions[:, i])
            ap = average_precision_score(true_labels_one_hot[:, i], predictions[:, i])
            ax.plot(recall, precision, lw=2, label=f'{class_names[i]} (AP = {ap:.2f})')
    
    # Add legend
    ax.legend(loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_feature_importance(feature_importance: List[Tuple[str, float]], title: str = "Feature Importance") -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        feature_importance: List of (feature_name, importance_score) tuples
        title: Plot title
        
    Returns:
        Matplotlib figure containing the feature importance visualization
    """
    # Sort features by absolute importance
    sorted_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)
    
    # Extract names and scores
    feature_names = [x[0] for x in sorted_features]
    importance_scores = [x[1] for x in sorted_features]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))
    
    # Set colors based on sign of importance
    colors = ['darkred' if score < 0 else 'darkblue' for score in importance_scores]
    
    # Plot horizontal bar chart
    ax.barh(feature_names, importance_scores, color=colors)
    
    # Add zero line
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_attention_maps(attention_weights: np.ndarray, layer_idx: int = 0, head_idx: int = 0) -> plt.Figure:
    """
    Plot attention maps from transformer layers.
    
    Args:
        attention_weights: Attention weights tensor [batch, num_heads, seq_len, seq_len]
        layer_idx: Index of the layer to visualize
        head_idx: Index of the attention head to visualize
        
    Returns:
        Matplotlib figure containing the attention map visualization
    """
    # Extract attention weights for the specified head
    attn = attention_weights[0, head_idx]  # [seq_len, seq_len]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot attention heatmap
    im = ax.imshow(attn, cmap='viridis')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight')
    
    # Set labels and title
    ax.set_xlabel('Attention Target')
    ax.set_ylabel('Attention Source')
    ax.set_title(f'Attention Map (Head {head_idx})')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_embedding_visualization(embeddings: np.ndarray, labels: np.ndarray, class_names: Optional[List[str]] = None) -> plt.Figure:
    """
    Visualize embeddings using t-SNE.
    
    Args:
        embeddings: Embedding vectors
        labels: Class labels
        class_names: List of class names
        
    Returns:
        Matplotlib figure containing the embedding visualization
    """
    # Reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Set class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in np.unique(labels)]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot for each class
    for i, class_name in enumerate(class_names):
        mask = labels == i
        ax.scatter(
            reduced_embeddings[mask, 0],
            reduced_embeddings[mask, 1],
            label=class_name,
            alpha=0.7
        )
    
    # Set labels and title
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_title('Embedding Visualization (t-SNE)')
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_training_curves(history: Dict[str, List[float]]) -> Tuple[plt.Figure, plt.Figure]:
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary containing training history
        
    Returns:
        Tuple of (loss_figure, accuracy_figure)
    """
    # Create loss figure
    loss_fig, loss_ax = plt.subplots(figsize=(10, 6))
    
    # Plot training loss
    if 'loss' in history:
        loss_ax.plot(history['loss'], label='Training Loss')
    
    # Plot validation loss
    if 'val_loss' in history:
        loss_ax.plot(history['val_loss'], label='Validation Loss')
    
    # Set labels and title
    loss_ax.set_xlabel('Epoch')
    loss_ax.set_ylabel('Loss')
    loss_ax.set_title('Training and Validation Loss')
    
    # Add legend
    loss_ax.legend()
    
    # Add grid
    loss_ax.grid(True, linestyle='--', alpha=0.6)
    
    # Create accuracy figure
    acc_fig, acc_ax = plt.subplots(figsize=(10, 6))
    
    # Plot training accuracy
    if 'accuracy' in history:
        acc_ax.plot(history['accuracy'], label='Training Accuracy')
    
    # Plot validation accuracy
    if 'val_accuracy' in history:
        acc_ax.plot(history['val_accuracy'], label='Validation Accuracy')
    
    # Set labels and title
    acc_ax.set_xlabel('Epoch')
    acc_ax.set_ylabel('Accuracy')
    acc_ax.set_title('Training and Validation Accuracy')
    
    # Add legend
    acc_ax.legend()
    
    # Add grid
    acc_ax.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust layout
    plt.tight_layout()
    
    return loss_fig, acc_fig

def plot_multimodal_fusion_visualization(
    modality_weights: np.ndarray,
    modality_names: List[str],
    sample_indices: Optional[List[int]] = None
) -> plt.Figure:
    """
    Visualize multimodal fusion weights.
    
    Args:
        modality_weights: Weights for each modality [num_samples, num_modalities]
        modality_names: Names of the modalities
        sample_indices: Indices of samples to visualize
        
    Returns:
        Matplotlib figure containing the fusion weights visualization
    """
    # Select samples to visualize
    if sample_indices is None:
        if modality_weights.shape[0] > 10:
            sample_indices = list(range(10))
        else:
            sample_indices = list(range(modality_weights.shape[0]))
    
    selected_weights = modality_weights[sample_indices]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    im = ax.imshow(selected_weights, cmap='YlOrRd')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Modality Weight')
    
    # Set labels and title
    ax.set_xlabel('Modality')
    ax.set_ylabel('Sample')
    ax.set_title('Multimodal Fusion Weights')
    
    # Set ticks and labels
    ax.set_xticks(range(len(modality_names)))
    ax.set_xticklabels(modality_names)
    ax.set_yticks(range(len(sample_indices)))
    ax.set_yticklabels([f"Sample {idx}" for idx in sample_indices])
    
    # Add text annotations
    for i in range(len(sample_indices)):
        for j in range(len(modality_names)):
            text = ax.text(j, i, f"{selected_weights[i, j]:.2f}",
                          ha="center", va="center", color="black" if selected_weights[i, j] < 0.7 else "white")
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_class_distribution(labels: np.ndarray, class_names: Optional[List[str]] = None) -> plt.Figure:
    """
    Plot class distribution.
    
    Args:
        labels: Class labels
        class_names: List of class names
        
    Returns:
        Matplotlib figure containing the class distribution visualization
    """
    # Count class occurrences
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Set class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in unique_labels]
    else:
        class_names = [class_names[i] for i in unique_labels]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    ax.bar(class_names, counts)
    
    # Set labels and title
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution')
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        ax.text(i, count + 0.1, str(count), ha='center')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_model_architecture_diagram() -> plt.Figure:
    """
    Create a diagram of the MH-Net model architecture.
    
    Returns:
        Matplotlib figure containing the model architecture diagram
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Remove axes
    ax.axis('off')
    
    # Define coordinates for the boxes
    input_y = 0.8
    encoder_y = 0.6
    fusion_y = 0.4
    output_y = 0.2
    
    text_x = 0.2
    audio_x = 0.4
    physio_x = 0.6
    imaging_x = 0.8
    
    fusion_x = 0.5
    
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
    
    # Adjust layout
    plt.tight_layout()
    
    return fig
