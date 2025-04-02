import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
import librosa
import nltk
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, TFBertModel
from typing import Dict, List, Tuple, Optional, Union, Any

class FeatureExtractor:
    """
    Feature extractor for multimodal mental health data.
    Extracts features from different modalities for input to deep learning models.
    """
    
    def __init__(
        self,
        text_model: str = "bert-base-uncased",
        audio_feature_type: str = "mfcc",
        physiological_feature_type: str = "statistical",
        imaging_model: str = "resnet50"
    ):
        """
        Initialize the FeatureExtractor.
        
        Args:
            text_model: Name of the pre-trained text model to use
            audio_feature_type: Type of audio features to extract
            physiological_feature_type: Type of physiological features to extract
            imaging_model: Name of the pre-trained imaging model to use
        """
        self.text_model_name = text_model
        self.audio_feature_type = audio_feature_type
        self.physiological_feature_type = physiological_feature_type
        self.imaging_model_name = imaging_model
        
        # Initialize text model if needed
        if text_model.startswith("bert"):
            try:
                self.text_tokenizer = BertTokenizer.from_pretrained(text_model)
                self.text_model = TFBertModel.from_pretrained(text_model)
                print(f"Loaded {text_model} for text feature extraction")
            except Exception as e:
                print(f"Error loading BERT model: {e}")
                print("Using fallback text feature extraction")
                self.text_model = None
        else:
            self.text_model = None
        
        # Initialize imaging model if needed
        if imaging_model == "resnet50":
            try:
                base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                self.imaging_model = Model(
                    inputs=base_model.input,
                    outputs=GlobalAveragePooling2D()(base_model.output)
                )
                print("Loaded ResNet50 for imaging feature extraction")
            except Exception as e:
                print(f"Error loading imaging model: {e}")
                print("Using fallback imaging feature extraction")
                self.imaging_model = None
        else:
            self.imaging_model = None
    
    def extract_text_features(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """
        Extract features from text data.
        
        Args:
            texts: List of text samples
            max_length: Maximum sequence length
            
        Returns:
            Array of text features
        """
        if self.text_model is not None:
            # Use pre-trained BERT model for feature extraction
            try:
                # Tokenize texts
                encoded_texts = self.text_tokenizer(
                    texts,
                    padding='max_length',
                    truncation=True,
                    max_length=max_length,
                    return_tensors='tf'
                )
                
                # Extract features
                outputs = self.text_model(encoded_texts)
                
                # Use the [CLS] token embedding as the sentence representation
                return outputs.last_hidden_state[:, 0, :].numpy()
            
            except Exception as e:
                print(f"Error extracting BERT features: {e}")
                print("Falling back to basic text features")
        
        # Fallback: Basic bag-of-words representation
        return self._extract_basic_text_features(texts)
    
    def _extract_basic_text_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract basic features from text data as a fallback.
        
        Args:
            texts: List of text samples
            
        Returns:
            Array of text features
        """
        # Create a simple bag-of-words representation
        # First, create a vocabulary from all words in the samples
        all_words = set()
        for text in texts:
            tokens = word_tokenize(text.lower())
            all_words.update(tokens)
        
        vocabulary = list(all_words)
        vocab_size = len(vocabulary)
        word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
        
        # Create feature vectors (simplified bag-of-words)
        max_features = min(vocab_size, 300)  # Limit feature vector size
        features = np.zeros((len(texts), max_features))
        
        for i, text in enumerate(texts):
            tokens = word_tokenize(text.lower())
            for token in tokens:
                if token in word_to_idx:
                    idx = word_to_idx[token]
                    if idx < max_features:
                        features[i, idx] = 1
        
        return features
    
    def extract_audio_features(self, audio_data: np.ndarray, sr: int = 22050) -> np.ndarray:
        """
        Extract features from audio data.
        
        Args:
            audio_data: Array of audio samples or pre-computed features
            sr: Sample rate (for raw audio)
            
        Returns:
            Array of audio features
        """
        # Check if the input is already processed features
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            # Assume these are already extracted features
            # Just normalize and return
            features = audio_data.copy()
            
            # Normalize each sample
            for i in range(features.shape[0]):
                mean = np.mean(features[i])
                std = np.std(features[i])
                features[i] = (features[i] - mean) / (std + 1e-8)
            
            return features
        
        # If we have raw audio, extract features
        features = []
        
        for audio in audio_data:
            if self.audio_feature_type == "mfcc":
                # Extract MFCCs
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                features.append(mfccs.T)  # Transpose to (time, features)
            
            elif self.audio_feature_type == "mel_spectrogram":
                # Extract Mel spectrogram
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
                log_mel_spec = librosa.power_to_db(mel_spec)
                features.append(log_mel_spec.T)  # Transpose to (time, features)
            
            elif self.audio_feature_type == "spectral_contrast":
                # Extract spectral contrast
                contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
                features.append(contrast.T)  # Transpose to (time, features)
            
            else:
                # Default to MFCCs
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                features.append(mfccs.T)  # Transpose to (time, features)
        
        # Pad or truncate sequences to the same length
        max_length = max(f.shape[0] for f in features)
        padded_features = np.zeros((len(features), max_length, features[0].shape[1]))
        
        for i, feat in enumerate(features):
            if feat.shape[0] >= max_length:
                padded_features[i] = feat[:max_length, :]
            else:
                padded_features[i, :feat.shape[0], :] = feat
        
        return padded_features
    
    def extract_physiological_features(self, physio_data: np.ndarray) -> np.ndarray:
        """
        Extract features from physiological data.
        
        Args:
            physio_data: Array of physiological signals
            
        Returns:
            Array of physiological features
        """
        if self.physiological_feature_type == "raw":
            # Just use the raw data as features
            return physio_data
        
        elif self.physiological_feature_type == "statistical":
            # Extract statistical features from each channel
            n_samples = physio_data.shape[0]
            n_channels = physio_data.shape[1]
            
            # Define the statistical features to extract
            n_features = 5  # mean, std, min, max, median
            features = np.zeros((n_samples, n_channels, n_features))
            
            for i in range(n_samples):
                for c in range(n_channels):
                    signal = physio_data[i, c, :]
                    
                    features[i, c, 0] = np.mean(signal)
                    features[i, c, 1] = np.std(signal)
                    features[i, c, 2] = np.min(signal)
                    features[i, c, 3] = np.max(signal)
                    features[i, c, 4] = np.median(signal)
            
            # Flatten channel and feature dimensions
            return features.reshape(n_samples, -1)
        
        elif self.physiological_feature_type == "spectral":
            # Extract spectral features from each channel
            n_samples = physio_data.shape[0]
            n_channels = physio_data.shape[1]
            
            # Define the spectral features to extract
            features_list = []
            
            for i in range(n_samples):
                sample_features = []
                
                for c in range(n_channels):
                    signal = physio_data[i, c, :]
                    
                    # Compute power spectral density
                    psd = np.abs(np.fft.rfft(signal))**2
                    
                    # Extract features from PSD
                    mean_psd = np.mean(psd)
                    std_psd = np.std(psd)
                    max_psd = np.max(psd)
                    
                    # Dominant frequency
                    freqs = np.fft.rfftfreq(len(signal))
                    dominant_freq = freqs[np.argmax(psd)]
                    
                    # Power in different frequency bands (simplified)
                    # Assuming normalized frequency
                    low_freq_power = np.sum(psd[freqs < 0.1])
                    mid_freq_power = np.sum(psd[(freqs >= 0.1) & (freqs < 0.3)])
                    high_freq_power = np.sum(psd[freqs >= 0.3])
                    
                    channel_features = [
                        mean_psd, std_psd, max_psd, dominant_freq,
                        low_freq_power, mid_freq_power, high_freq_power
                    ]
                    
                    sample_features.extend(channel_features)
                
                features_list.append(sample_features)
            
            return np.array(features_list)
        
        else:
            # Default to raw data
            return physio_data
    
    def extract_imaging_features(self, imaging_data: np.ndarray) -> np.ndarray:
        """
        Extract features from imaging data.
        
        Args:
            imaging_data: Array of imaging data
            
        Returns:
            Array of imaging features
        """
        if self.imaging_model is not None:
            # Use pre-trained CNN for feature extraction
            try:
                # Prepare the data for the model
                n_samples = imaging_data.shape[0]
                features = np.zeros((n_samples, 2048))  # ResNet50 features
                
                for i in range(n_samples):
                    # For 3D images, take the middle slice from each dimension
                    if len(imaging_data.shape) == 4:  # 3D images
                        # Take the middle slice from each dimension
                        x_mid = imaging_data.shape[1] // 2
                        y_mid = imaging_data.shape[2] // 2
                        z_mid = imaging_data.shape[3] // 2
                        
                        slice_x = imaging_data[i, x_mid, :, :]
                        slice_y = imaging_data[i, :, y_mid, :]
                        slice_z = imaging_data[i, :, :, z_mid]
                        
                        # Stack the slices into an RGB-like image
                        img = np.stack([slice_x, slice_y, slice_z], axis=-1)
                    else:  # 2D images
                        img = imaging_data[i]
                        
                    # Resize to the model's input shape
                    img_resized = tf.image.resize(img, (224, 224))
                    
                    # Ensure 3 channels
                    if img_resized.shape[-1] == 1:
                        img_resized = tf.repeat(img_resized, 3, axis=-1)
                    elif img_resized.shape[-1] == 2:
                        # Add a third channel
                        zero_channel = tf.zeros_like(img_resized[:, :, :1])
                        img_resized = tf.concat([img_resized, zero_channel], axis=-1)
                    
                    # Preprocess for ResNet
                    img_preprocessed = preprocess_input(img_resized)
                    
                    # Extract features
                    features[i] = self.imaging_model.predict(tf.expand_dims(img_preprocessed, 0), verbose=0)
                
                return features
            
            except Exception as e:
                print(f"Error extracting imaging features: {e}")
                print("Falling back to basic imaging features")
        
        # Fallback: Extract basic statistical features
        return self._extract_basic_imaging_features(imaging_data)
    
    def _extract_basic_imaging_features(self, imaging_data: np.ndarray) -> np.ndarray:
        """
        Extract basic features from imaging data as a fallback.
        
        Args:
            imaging_data: Array of imaging data
            
        Returns:
            Array of imaging features
        """
        n_samples = imaging_data.shape[0]
        features_list = []
        
        for i in range(n_samples):
            img = imaging_data[i]
            
            # Extract basic statistical features
            features = [
                np.mean(img),
                np.std(img),
                np.median(img),
                np.min(img),
                np.max(img),
                np.percentile(img, 25),
                np.percentile(img, 75)
            ]
            
            # Extract histogram features
            hist, _ = np.histogram(img, bins=10, range=(0, 1))
            features.extend(hist / np.sum(hist))  # Normalized histogram
            
            # For 3D images, add features for each axis
            if len(img.shape) == 3:
                # Mean along each axis
                features.extend([
                    np.mean(np.mean(img, axis=0)),
                    np.mean(np.mean(img, axis=1)),
                    np.mean(np.mean(img, axis=2))
                ])
                
                # Standard deviation along each axis
                features.extend([
                    np.mean(np.std(img, axis=0)),
                    np.mean(np.std(img, axis=1)),
                    np.mean(np.std(img, axis=2))
                ])
            
            features_list.append(features)
        
        return np.array(features_list)
    
    def extract_multimodal_features(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Extract features from all modalities in the data.
        
        Args:
            data: Dictionary containing multimodal data
            
        Returns:
            Dictionary containing extracted features for each modality
        """
        features = {}
        
        if "text_raw" in data and len(data["text_raw"]) > 0:
            print("Extracting text features...")
            features["text"] = self.extract_text_features(data["text_raw"])
        elif "text" in data:
            features["text"] = data["text"]
        
        if "audio" in data:
            print("Extracting audio features...")
            features["audio"] = self.extract_audio_features(data["audio"])
        
        if "physiological" in data:
            print("Extracting physiological features...")
            features["physiological"] = self.extract_physiological_features(data["physiological"])
        
        if "imaging" in data:
            print("Extracting imaging features...")
            features["imaging"] = self.extract_imaging_features(data["imaging"])
        
        if "labels" in data:
            features["labels"] = data["labels"]
        
        return features
