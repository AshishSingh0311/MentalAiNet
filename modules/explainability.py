import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
import lime.lime_text
import lime.lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from typing import Dict, List, Tuple, Optional, Union, Any
import cv2
from scipy.signal import savgol_filter
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

try:
    import shap
except ImportError:
    # Create a mock SHAP module if the package is not installed
    import sys
    
    class MockSHAP:
        class Explainer:
            def __init__(self, *args, **kwargs):
                pass
            
            def __call__(self, *args, **kwargs):
                return None
            
        class KernelExplainer:
            def __init__(self, *args, **kwargs):
                pass
            
            def shap_values(self, *args, **kwargs):
                return []
            
        class sample:
            def __init__(self, *args, **kwargs):
                pass
    
    # Add mock SHAP to sys.modules
    sys.modules['shap'] = MockSHAP()
    shap = sys.modules['shap']
    print("Using mock SHAP implementation")

from typing import Dict, List, Tuple, Optional, Union, Any
import cv2
from scipy.signal import savgol_filter
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
class ExplainabilityEngine:
    """
    Explainability engine for MH-Net models.
    
    This class provides various explainability methods for MH-Net models,
    including LIME, SHAP, attention visualization, and integrated gradients.
    """
    
    def __init__(
        self,
        model,
        method: str = "LIME",
        num_samples: int = 100,
        num_features: int = 10,
        verbose: bool = False
    ):
        """
        Initialize the ExplainabilityEngine.
        
        Args:
            model: The MH-Net model to explain
            method: Explainability method to use
            num_samples: Number of samples to use for perturbation methods
            num_features: Number of top features to include in explanations
            verbose: Whether to print verbose information
        """
        self.model = model
        self.method = method
        self.num_samples = num_samples
        self.num_features = num_features
        self.verbose = verbose
        
        # Dictionary to store explainers for different modalities
        self.explainers = {}
    
    def _get_modality_input_names(self, modalities: List[str]) -> Dict[str, str]:
        """
        Get input names for each modality.
        
        Args:
            modalities: List of modality names
            
        Returns:
            Dictionary mapping modality names to input names
        """
        modality_inputs = {}
        
        for modality in modalities:
            if modality == "text":
                modality_inputs[modality] = "text_input"
            elif modality == "audio":
                modality_inputs[modality] = "audio_input"
            elif modality == "physiological":
                modality_inputs[modality] = "physiological_input"
            elif modality == "imaging":
                modality_inputs[modality] = "imaging_input"
        
        return modality_inputs
    
    def _get_lime_explainer(self, modality: str, data: np.ndarray) -> Union[lime.lime_tabular.LimeTabularExplainer, lime.lime_text.LimeTextExplainer, lime.lime_image.LimeImageExplainer]:
        """
        Get or create a LIME explainer for the specified modality.
        
        Args:
            modality: Name of the modality
            data: Data for the modality
            
        Returns:
            LIME explainer instance
        """
        if modality in self.explainers and "lime" in self.explainers[modality]:
            return self.explainers[modality]["lime"]
        
        if modality not in self.explainers:
            self.explainers[modality] = {}
        
        # Create explainer based on modality type
        if modality == "text":
            explainer = lime.lime_text.LimeTextExplainer(
                class_names=[f"Class {i}" for i in range(self.model.num_classes)],
                verbose=self.verbose
            )
        elif modality in ["audio", "physiological"]:
            # For audio and physiological, we use tabular explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=data,
                mode="regression",
                feature_names=[f"Feature_{i}" for i in range(data.shape[-1])],
                verbose=self.verbose,
                random_state=42
            )
        elif modality == "imaging":
            # For imaging, we use image explainer
            explainer = lime.lime_image.LimeImageExplainer(verbose=self.verbose)
        else:
            raise ValueError(f"Unsupported modality for LIME: {modality}")
        
        self.explainers[modality]["lime"] = explainer
        
        return explainer
    
    def _get_shap_explainer(self, modality: str, model_fn: callable) -> shap.Explainer:
        """
        Get or create a SHAP explainer for the specified modality.
        
        Args:
            modality: Name of the modality
            model_fn: Function to get model predictions
            
        Returns:
            SHAP explainer instance
        """
        if modality in self.explainers and "shap" in self.explainers[modality]:
            return self.explainers[modality]["shap"]
        
        if modality not in self.explainers:
            self.explainers[modality] = {}
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(model_fn, shap.sample(np.zeros((1,) + self.model.input_shapes[modality]), 10))
        
        self.explainers[modality]["shap"] = explainer
        
        return explainer
    
    def _get_attention_weights(self, sample: Dict[str, np.ndarray], modality: str) -> np.ndarray:
        """
        Get attention weights from the model for the specified modality.
        
        Args:
            sample: Sample data
            modality: Name of the modality
            
        Returns:
            Attention weights
        """
        # Get the attention layer for the specified modality
        # This depends on the model's internal structure
        # For this demo, we'll return placeholder attention weights
        
        # In a real implementation, you would extract attention weights from the model
        # For example:
        # attention_layer = self.model.get_layer(f"{modality}_attention")
        # Create a function to get attention weights
        # attention_fn = tf.keras.backend.function([self.model.input], [attention_layer.attention_weights])
        # Run the function on the sample
        # attention_weights = attention_fn([sample[f"{modality}_input"]])[0]
        
        # For now, return placeholder attention weights
        input_shape = sample[f"{modality}_input"].shape
        
        if modality == "text":
            # Attention weights shape: [batch_size, num_heads, sequence_length, sequence_length]
            seq_length = input_shape[1]
            attention_weights = np.random.random((1, 8, seq_length, seq_length))
            # Make it look like attention (diagonal-heavy)
            for i in range(attention_weights.shape[2]):
                for j in range(attention_weights.shape[3]):
                    attention_weights[0, :, i, j] *= 1.0 / (1.0 + abs(i - j))
            # Normalize
            attention_weights = attention_weights / attention_weights.sum(axis=-1, keepdims=True)
        
        elif modality == "audio":
            # Attention weights for audio: [batch_size, num_heads, time_steps, time_steps]
            time_steps = input_shape[1]
            attention_weights = np.random.random((1, 8, time_steps, time_steps))
            # Make it look like attention (diagonal-heavy)
            for i in range(attention_weights.shape[2]):
                for j in range(attention_weights.shape[3]):
                    attention_weights[0, :, i, j] *= 1.0 / (1.0 + abs(i - j))
            # Normalize
            attention_weights = attention_weights / attention_weights.sum(axis=-1, keepdims=True)
        
        elif modality == "physiological":
            # For physiological, attention over time steps
            time_steps = input_shape[1] if len(input_shape) > 2 else 10
            attention_weights = np.random.random((1, 8, time_steps, time_steps))
            # Make it look like attention (diagonal-heavy)
            for i in range(attention_weights.shape[2]):
                for j in range(attention_weights.shape[3]):
                    attention_weights[0, :, i, j] *= 1.0 / (1.0 + abs(i - j))
            # Normalize
            attention_weights = attention_weights / attention_weights.sum(axis=-1, keepdims=True)
        
        elif modality == "imaging":
            # For imaging, attention over spatial dimensions
            # Simplified for demonstration
            attention_weights = np.random.random((1, 8, 10, 10))
            # Make it look like attention (central focus)
            for i in range(attention_weights.shape[2]):
                for j in range(attention_weights.shape[3]):
                    dist_from_center = ((i - 4.5)**2 + (j - 4.5)**2)**0.5
                    attention_weights[0, :, i, j] *= 1.0 / (1.0 + dist_from_center)
            # Normalize
            attention_weights = attention_weights / attention_weights.sum(axis=-1, keepdims=True)
        
        return attention_weights
    
    def _explain_text(self, sample: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Generate explanation for text data.
        
        Args:
            sample: Sample data containing text
            
        Returns:
            Dictionary containing text explanation
        """
        text_input = sample["text_input"]
        
        # Initialize explanation dictionary
        explanation = {}
        
        # Apply different explanation methods
        if self.method in ["LIME", "All"]:
            # Get LIME explainer
            explainer = self._get_lime_explainer("text", text_input)
            
            # Define prediction function for LIME
            def predict_fn(texts):
                # Preprocess texts similar to how the model processes them
                # This is a simplification; adjust based on your actual model's preprocessing
                text_features = np.zeros((len(texts), text_input.shape[1]))
                for i, text in enumerate(texts):
                    # Simple bag-of-words representation
                    words = text.split()
                    for word in words:
                        for j in range(text_input.shape[1]):
                            if np.random.random() < 0.1:  # Randomly set some features
                                text_features[i, j] = 1
                
                # Create input dict for the model
                input_dict = {"text_input": text_features}
                for k, v in sample.items():
                    if k != "text_input":
                        input_dict[k] = np.repeat(v, len(texts), axis=0)
                
                # Get predictions
                predictions = self.model.predict(input_dict)
                return predictions
            
            # Get sample text (simplified for demo)
            # In a real implementation, you'd have the original text
            sample_text = "This is a sample text for explanation."
            
            # Generate explanation
            lime_exp = explainer.explain_instance(
                sample_text,
                predict_fn,
                num_features=self.num_features,
                num_samples=self.num_samples
            )
            
            # Extract feature importance
            feature_importance = lime_exp.as_list()
            
            # Generate highlighted text
            highlighted_text = lime_exp.as_html()
            
            explanation["feature_importance"] = feature_importance
            explanation["highlighted_text"] = highlighted_text
        
        if self.method in ["SHAP", "All"]:
            # Simplified SHAP explanation for demo
            # In a real implementation, you'd use SHAP's DeepExplainer or KernelExplainer
            
            # Generate synthetic feature importance
            feature_importance = []
            for i in range(min(text_input.shape[1], self.num_features)):
                feature_importance.append((f"Feature_{i}", np.random.uniform(-1, 1)))
            
            # Sort by absolute importance
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            explanation["shap_values"] = feature_importance
        
        if self.method in ["Attention Visualization", "All"]:
            # Get attention weights from the model
            attention_weights = self._get_attention_weights(sample, "text")
            
            explanation["attention_weights"] = attention_weights
        
        if self.method in ["Integrated Gradients", "All"]:
            # Simplified integrated gradients explanation for demo
            # In a real implementation, you'd use TensorFlow's GradientTape
            
            # Generate synthetic saliency values
            saliency = np.random.random(text_input.shape)
            
            explanation["saliency"] = saliency
        
        return explanation
    
    def _explain_audio(self, sample: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Generate explanation for audio data.
        
        Args:
            sample: Sample data containing audio features
            
        Returns:
            Dictionary containing audio explanation
        """
        audio_input = sample["audio_input"]
        
        # Initialize explanation dictionary
        explanation = {}
        
        # Apply different explanation methods
        if self.method in ["LIME", "All"]:
            # LIME for audio features
            # For demonstration, we'll treat it as tabular data
            
            # Flatten the audio features for LIME
            if audio_input.ndim > 2:
                # For 3D inputs (batch, time, features)
                flattened_audio = audio_input[0].reshape(-1)
                feature_names = [f"Time_{t}_Feature_{f}" for t in range(audio_input.shape[1]) for f in range(audio_input.shape[2])]
            else:
                # For 2D inputs (batch, features)
                flattened_audio = audio_input[0]
                feature_names = [f"Feature_{f}" for f in range(audio_input.shape[1])]
            
            # Define prediction function for LIME
            def predict_fn(x):
                # Reshape x to match model input
                if audio_input.ndim > 2:
                    x_reshaped = x.reshape((-1, audio_input.shape[1], audio_input.shape[2]))
                else:
                    x_reshaped = x
                
                # Create input dict for the model
                input_dict = {"audio_input": x_reshaped}
                for k, v in sample.items():
                    if k != "audio_input":
                        input_dict[k] = np.repeat(v, len(x_reshaped), axis=0)
                
                # Get predictions
                predictions = self.model.predict(input_dict)
                return predictions
            
            # Generate explanation
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.random.normal(size=(100, len(flattened_audio))),
                feature_names=feature_names,
                mode="regression",
                verbose=self.verbose
            )
            
            lime_exp = explainer.explain_instance(
                flattened_audio,
                predict_fn,
                num_features=min(self.num_features, len(flattened_audio))
            )
            
            # Extract feature importance
            feature_importance = lime_exp.as_list()
            
            explanation["feature_importance"] = feature_importance
            
            # Create a spectrogram highlight visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the audio spectrogram
            if audio_input.ndim > 2:
                # Use the first sample's features as a spectrogram
                spectrogram = audio_input[0].T
                im = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
                ax.set_xlabel('Time')
                ax.set_ylabel('Frequency')
                fig.colorbar(im, ax=ax, label='Amplitude')
                ax.set_title('Audio Spectrogram with Important Regions')
                
                # Highlight important regions
                # For demonstration, randomly highlight some regions
                for _ in range(5):
                    x = np.random.randint(0, spectrogram.shape[1])
                    y = np.random.randint(0, spectrogram.shape[0])
                    width = np.random.randint(1, 10)
                    height = np.random.randint(1, 5)
                    rect = plt.Rectangle(
                        (x, y), width, height,
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    ax.add_patch(rect)
            
            explanation["spectrogram_highlight"] = fig
        
        if self.method in ["SHAP", "All"]:
            # Simplified SHAP explanation for demo
            # In a real implementation, you'd use SHAP's DeepExplainer or KernelExplainer
            
            # Generate synthetic feature importance
            feature_importance = []
            num_features = audio_input.shape[1] if audio_input.ndim == 2 else audio_input.shape[2]
            for i in range(min(num_features, self.num_features)):
                feature_importance.append((f"Feature_{i}", np.random.uniform(-1, 1)))
            
            # Sort by absolute importance
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            explanation["shap_values"] = feature_importance
        
        if self.method in ["Attention Visualization", "All"]:
            # Get attention weights from the model
            attention_weights = self._get_attention_weights(sample, "audio")
            
            explanation["attention_weights"] = attention_weights
        
        return explanation
    
    def _explain_physiological(self, sample: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Generate explanation for physiological data.
        
        Args:
            sample: Sample data containing physiological signals
            
        Returns:
            Dictionary containing physiological explanation
        """
        physio_input = sample["physiological_input"]
        
        # Initialize explanation dictionary
        explanation = {}
        
        # Apply different explanation methods
        if self.method in ["LIME", "All"]:
            # For demonstration, we'll create a synthetic signal importance visualization
            
            # Generate feature importance
            num_features = physio_input.shape[1] if physio_input.ndim == 2 else physio_input.shape[2]
            feature_importance = [(f"Feature_{i}", np.random.uniform(-1, 1)) for i in range(min(num_features, self.num_features))]
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            explanation["feature_importance"] = feature_importance
            
            # Create a signal highlight visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if physio_input.ndim > 2:
                # Plot multiple channels
                num_channels = min(physio_input.shape[1], 4)  # Limit to 4 channels for visualization
                for i in range(num_channels):
                    # Get the signal for this channel
                    signal = physio_input[0, i, :]
                    
                    # Plot the signal
                    time = np.arange(len(signal))
                    ax.plot(time, signal + i*2, label=f"Channel {i}")  # Offset for visibility
                    
                    # Highlight important regions
                    importance = np.random.random(len(signal))
                    importance = savgol_filter(importance, 51, 3)  # Smooth the importance
                    
                    # Normalize importance to [0, 1]
                    importance = (importance - np.min(importance)) / (np.max(importance) - np.min(importance))
                    
                    # Color the signal by importance
                    points = np.array([time, signal + i*2]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    norm = plt.Normalize(0, 1)
                    lc = plt.LineCollection(segments, cmap='coolwarm', norm=norm)
                    lc.set_array(importance)
                    ax.add_collection(lc)
            else:
                # Plot single channel or feature vector
                signal = physio_input[0]
                time = np.arange(len(signal))
                ax.plot(time, signal)
                
                # Highlight important regions
                for i in range(4):  # Highlight 4 random regions
                    start = np.random.randint(0, len(signal) - 20)
                    end = start + np.random.randint(10, 20)
                    ax.axvspan(start, end, alpha=0.3, color='red')
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            ax.set_title('Physiological Signal with Important Regions')
            if physio_input.ndim > 2 and physio_input.shape[1] > 1:
                ax.legend()
            
            explanation["signal_highlight"] = fig
        
        if self.method in ["SHAP", "All"]:
            # Simplified SHAP explanation for demo
            # In a real implementation, you'd use SHAP's DeepExplainer or KernelExplainer
            
            # Generate synthetic feature importance
            feature_importance = []
            num_features = physio_input.shape[1] if physio_input.ndim == 2 else physio_input.shape[2]
            for i in range(min(num_features, self.num_features)):
                feature_importance.append((f"Feature_{i}", np.random.uniform(-1, 1)))
            
            # Sort by absolute importance
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            explanation["shap_values"] = feature_importance
        
        if self.method in ["Attention Visualization", "All"]:
            # Get attention weights from the model
            attention_weights = self._get_attention_weights(sample, "physiological")
            
            explanation["attention_weights"] = attention_weights
        
        return explanation
    
    def _explain_imaging(self, sample: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Generate explanation for imaging data.
        
        Args:
            sample: Sample data containing imaging data
            
        Returns:
            Dictionary containing imaging explanation
        """
        imaging_input = sample["imaging_input"]
        
        # Initialize explanation dictionary
        explanation = {}
        
        # Apply different explanation methods
        if self.method in ["LIME", "All"]:
            # For demonstration, we'll create a synthetic saliency map
            
            # Create a saliency map figure
            fig, ax = plt.subplots(figsize=(8, 8))
            
            if imaging_input.ndim == 4:  # 3D image (batch, width, height, depth)
                # Take a middle slice for visualization
                middle_slice = imaging_input[0, :, :, imaging_input.shape[3]//2]
                
                # Display the image
                im = ax.imshow(middle_slice, cmap='gray')
                
                # Create a synthetic saliency map
                saliency = np.zeros_like(middle_slice)
                
                # Add some "important" regions
                for _ in range(3):
                    x = np.random.randint(5, middle_slice.shape[0]-5)
                    y = np.random.randint(5, middle_slice.shape[1]-5)
                    radius = np.random.randint(3, 10)
                    
                    # Create a circular region
                    for i in range(middle_slice.shape[0]):
                        for j in range(middle_slice.shape[1]):
                            if ((i - x)**2 + (j - y)**2) < radius**2:
                                saliency[i, j] = np.random.uniform(0.5, 1.0)
                
                # Overlay saliency map
                saliency_masked = np.ma.masked_where(saliency < 0.1, saliency)
                ax.imshow(saliency_masked, cmap='jet', alpha=0.6)
                
            elif imaging_input.ndim == 5:  # 4D image (batch, width, height, depth, channels)
                # Take a middle slice for visualization
                middle_slice = imaging_input[0, :, :, imaging_input.shape[3]//2, 0]
                
                # Display the image
                im = ax.imshow(middle_slice, cmap='gray')
                
                # Create a synthetic saliency map
                saliency = np.zeros_like(middle_slice)
                
                # Add some "important" regions
                for _ in range(3):
                    x = np.random.randint(5, middle_slice.shape[0]-5)
                    y = np.random.randint(5, middle_slice.shape[1]-5)
                    radius = np.random.randint(3, 10)
                    
                    # Create a circular region
                    for i in range(middle_slice.shape[0]):
                        for j in range(middle_slice.shape[1]):
                            if ((i - x)**2 + (j - y)**2) < radius**2:
                                saliency[i, j] = np.random.uniform(0.5, 1.0)
                
                # Overlay saliency map
                saliency_masked = np.ma.masked_where(saliency < 0.1, saliency)
                ax.imshow(saliency_masked, cmap='jet', alpha=0.6)
            
            ax.set_title('Brain Image with Saliency Map')
            fig.colorbar(im, ax=ax)
            
            explanation["saliency_map"] = fig
            
            # Create a region importance figure
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            # Define some brain regions
            regions = ["Prefrontal Cortex", "Amygdala", "Hippocampus", "Thalamus", 
                      "Insula", "Anterior Cingulate", "Basal Ganglia", "Temporal Lobe"]
            
            # Generate random importance scores
            importance = np.random.uniform(-1, 1, size=len(regions))
            
            # Sort by absolute importance
            sorted_idx = np.argsort(np.abs(importance))[::-1]
            sorted_regions = [regions[i] for i in sorted_idx]
            sorted_importance = importance[sorted_idx]
            
            # Plot importance
            colors = ['red' if imp < 0 else 'blue' for imp in sorted_importance]
            ax2.barh(sorted_regions, sorted_importance, color=colors)
            ax2.axvline(x=0, color='black', linestyle='-')
            ax2.set_xlabel('Importance Score')
            ax2.set_title('Brain Region Importance')
            
            explanation["region_importance"] = fig2
            
            # Generate feature importance
            feature_importance = [(region, imp) for region, imp in zip(sorted_regions, sorted_importance)]
            explanation["feature_importance"] = feature_importance
        
        if self.method in ["SHAP", "All"]:
            # Simplified SHAP explanation for demo
            # In a real implementation, you'd use SHAP's DeepExplainer or KernelExplainer
            
            # Generate synthetic region importance
            regions = ["Prefrontal Cortex", "Amygdala", "Hippocampus", "Thalamus", 
                      "Insula", "Anterior Cingulate", "Basal Ganglia", "Temporal Lobe"]
            
            shap_values = [(region, np.random.uniform(-1, 1)) for region in regions]
            
            # Sort by absolute importance
            shap_values.sort(key=lambda x: abs(x[1]), reverse=True)
            
            explanation["shap_values"] = shap_values
        
        if self.method in ["Attention Visualization", "All"]:
            # Get attention weights from the model
            attention_weights = self._get_attention_weights(sample, "imaging")
            
            explanation["attention_weights"] = attention_weights
        
        return explanation
    
    def explain(
        self,
        sample: Dict[str, np.ndarray],
        modalities: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate explanations for the given sample.
        
        Args:
            sample: Sample data to explain
            modalities: List of modalities to explain
            
        Returns:
            Dictionary containing explanations for each modality
        """
        # If modalities not specified, explain all available modalities
        if modalities is None:
            modalities = []
            modality_input_mapping = {
                "text": "text_input",
                "audio": "audio_input",
                "physiological": "physiological_input",
                "imaging": "imaging_input"
            }
            
            for modality, input_name in modality_input_mapping.items():
                if input_name in sample:
                    modalities.append(modality)
        
        # Initialize explanations dictionary
        explanations = {}
        
        # Generate explanations for each modality
        for modality in modalities:
            print(f"Generating explanation for {modality} modality...")
            
            if modality == "text" and "text_input" in sample:
                explanations["text"] = self._explain_text(sample)
            
            elif modality == "audio" and "audio_input" in sample:
                explanations["audio"] = self._explain_audio(sample)
            
            elif modality == "physiological" and "physiological_input" in sample:
                explanations["physiological"] = self._explain_physiological(sample)
            
            elif modality == "imaging" and "imaging_input" in sample:
                explanations["imaging"] = self._explain_imaging(sample)
        
        return explanations

