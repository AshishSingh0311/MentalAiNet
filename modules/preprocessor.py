import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import librosa
import SimpleITK as sitk
import re
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

class MultimodalPreprocessor:
    """
    Preprocessor for multimodal mental health data.
    Handles preprocessing of text, audio, physiological signals, and imaging data.
    """
    
    def __init__(
        self,
        text_options: Dict[str, bool] = None,
        audio_options: Dict[str, Union[bool, str]] = None,
        physiological_options: Dict[str, bool] = None,
        imaging_options: Dict[str, bool] = None,
        general_options: Dict[str, Union[float, bool]] = None
    ):
        """
        Initialize the MultimodalPreprocessor.
        
        Args:
            text_options: Options for text preprocessing
            audio_options: Options for audio preprocessing
            physiological_options: Options for physiological signal preprocessing
            imaging_options: Options for imaging preprocessing
            general_options: General preprocessing options
        """
        # Set default options if not provided
        self.text_options = text_options or {
            "lowercase": True,
            "remove_stopwords": True,
            "stemming": False,
            "lemmatization": True
        }
        
        self.audio_options = audio_options or {
            "normalize": True,
            "noise_reduction": True,
            "feature_extraction": "MFCC"
        }
        
        self.physiological_options = physiological_options or {
            "filter": True,
            "normalize": True,
            "artifact_removal": True
        }
        
        self.imaging_options = imaging_options or {
            "normalize": True,
            "registration": False,
            "skull_strip": False
        }
        
        self.general_options = general_options or {
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "balance_classes": True,
            "augmentation": False
        }
        
        # Initialize NLP tools if needed
        if self.text_options["remove_stopwords"]:
            self.stop_words = set(stopwords.words('english'))
        
        if self.text_options["stemming"]:
            self.stemmer = PorterStemmer()
        
        if self.text_options["lemmatization"]:
            self.lemmatizer = WordNetLemmatizer()
    
    def preprocess(self, dataset: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Preprocess the dataset based on the specified options.
        
        Args:
            dataset: Dictionary containing the multimodal dataset
            
        Returns:
            Dictionary containing preprocessed train, validation, and test sets
        """
        preprocessed_data = {
            "train": {},
            "val": {},
            "test": {}
        }
        
        # Preprocess each modality if present in the dataset
        if "text" in dataset:
            preprocessed_text, preprocessed_text_raw = self._preprocess_text(dataset["text"], dataset.get("text_raw", []))
            dataset["text"] = preprocessed_text
            if preprocessed_text_raw:
                dataset["text_raw"] = preprocessed_text_raw
        
        if "audio" in dataset:
            dataset["audio"] = self._preprocess_audio(dataset["audio"])
        
        if "physiological" in dataset:
            dataset["physiological"] = self._preprocess_physiological(dataset["physiological"])
        
        if "imaging" in dataset:
            dataset["imaging"] = self._preprocess_imaging(dataset["imaging"])
        
        # Split the dataset into train, validation, and test sets
        train_data, val_data, test_data = self._split_dataset(dataset)
        
        # Apply class balancing if specified
        if self.general_options["balance_classes"]:
            train_data = self._balance_classes(train_data)
        
        # Apply data augmentation if specified
        if self.general_options["augmentation"]:
            train_data = self._augment_data(train_data)
        
        # Store the split datasets
        preprocessed_data["train"] = train_data
        preprocessed_data["val"] = val_data
        preprocessed_data["test"] = test_data
        
        return preprocessed_data
    
    def _preprocess_text(self, text_features: np.ndarray, raw_texts: List[str] = None) -> Tuple[np.ndarray, Optional[List[str]]]:
        """
        Preprocess text data.
        
        Args:
            text_features: Array of text features
            raw_texts: List of raw text strings (optional)
            
        Returns:
            Tuple of (preprocessed_features, preprocessed_raw_texts)
        """
        # If raw texts are available, preprocess them
        preprocessed_raw_texts = None
        if raw_texts:
            preprocessed_raw_texts = []
            for text in raw_texts:
                # Apply text preprocessing steps
                if self.text_options["lowercase"]:
                    text = text.lower()
                
                # Tokenize
                tokens = word_tokenize(text)
                
                # Remove stopwords
                if self.text_options["remove_stopwords"]:
                    tokens = [token for token in tokens if token not in self.stop_words]
                
                # Apply stemming
                if self.text_options["stemming"]:
                    tokens = [self.stemmer.stem(token) for token in tokens]
                
                # Apply lemmatization
                if self.text_options["lemmatization"]:
                    tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
                
                # Reconstruct text
                preprocessed_text = " ".join(tokens)
                preprocessed_raw_texts.append(preprocessed_text)
        
        # For text features, we'll just return them as is
        # In a real implementation, you might recompute features from the preprocessed raw texts
        return text_features, preprocessed_raw_texts
    
    def _preprocess_audio(self, audio_features: np.ndarray) -> np.ndarray:
        """
        Preprocess audio features.
        
        Args:
            audio_features: Array of audio features
            
        Returns:
            Preprocessed audio features
        """
        # In a real implementation, this would apply various preprocessing techniques
        # For this demo, we'll just simulate preprocessing
        
        processed_features = audio_features.copy()
        
        # Apply normalization if specified
        if self.audio_options["normalize"]:
            # Normalize each sample's features to zero mean and unit variance
            for i in range(processed_features.shape[0]):
                mean = np.mean(processed_features[i])
                std = np.std(processed_features[i])
                processed_features[i] = (processed_features[i] - mean) / (std + 1e-8)
        
        # Apply noise reduction if specified
        if self.audio_options["noise_reduction"]:
            # Simulate noise reduction by applying a simple smoothing
            for i in range(processed_features.shape[0]):
                # Apply a simple moving average filter for demonstration
                kernel_size = 3
                for j in range(processed_features.shape[2]):
                    processed_features[i, :, j] = np.convolve(
                        processed_features[i, :, j],
                        np.ones(kernel_size)/kernel_size,
                        mode='same'
                    )
        
        return processed_features
    
    def _preprocess_physiological(self, physio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess physiological signals.
        
        Args:
            physio_data: Array of physiological data
            
        Returns:
            Preprocessed physiological data
        """
        # In a real implementation, this would apply various preprocessing techniques
        # For this demo, we'll just simulate preprocessing
        
        processed_data = physio_data.copy()
        
        # Apply filtering if specified
        if self.physiological_options["filter"]:
            # Simulate bandpass filtering by applying a simple smoothing
            for i in range(processed_data.shape[0]):
                for c in range(processed_data.shape[1]):
                    # Apply a simple moving average filter for demonstration
                    kernel_size = 5
                    processed_data[i, c, :] = np.convolve(
                        processed_data[i, c, :],
                        np.ones(kernel_size)/kernel_size,
                        mode='same'
                    )
        
        # Apply normalization if specified
        if self.physiological_options["normalize"]:
            # Normalize each channel of each sample to zero mean and unit variance
            for i in range(processed_data.shape[0]):
                for c in range(processed_data.shape[1]):
                    mean = np.mean(processed_data[i, c, :])
                    std = np.std(processed_data[i, c, :])
                    processed_data[i, c, :] = (processed_data[i, c, :] - mean) / (std + 1e-8)
        
        # Apply artifact removal if specified
        if self.physiological_options["artifact_removal"]:
            # Simulate artifact removal by clipping extreme values
            for i in range(processed_data.shape[0]):
                for c in range(processed_data.shape[1]):
                    # Clip values outside of 3 standard deviations
                    mean = np.mean(processed_data[i, c, :])
                    std = np.std(processed_data[i, c, :])
                    threshold = 3 * std
                    processed_data[i, c, :] = np.clip(
                        processed_data[i, c, :],
                        mean - threshold,
                        mean + threshold
                    )
        
        return processed_data
    
    def _preprocess_imaging(self, imaging_data: np.ndarray) -> np.ndarray:
        """
        Preprocess imaging data.
        
        Args:
            imaging_data: Array of imaging data
            
        Returns:
            Preprocessed imaging data
        """
        # In a real implementation, this would apply various preprocessing techniques
        # For this demo, we'll just simulate preprocessing
        
        processed_data = imaging_data.copy()
        
        # Apply normalization if specified
        if self.imaging_options["normalize"]:
            # Normalize each sample's image to [0, 1] range
            for i in range(processed_data.shape[0]):
                min_val = np.min(processed_data[i])
                max_val = np.max(processed_data[i])
                processed_data[i] = (processed_data[i] - min_val) / (max_val - min_val + 1e-8)
        
        # Registration and skull stripping would typically require specialized libraries
        # In a real implementation, you would use SimpleITK, ANTs, or other tools
        # Here we just simulate these operations
        
        if self.imaging_options["registration"]:
            # Simulate registration by slightly shifting the images
            for i in range(processed_data.shape[0]):
                # Shift by 1 pixel in a random direction
                shift = np.random.randint(-1, 2, 3)
                for axis, shift_val in enumerate(shift):
                    if shift_val != 0:
                        slices = [slice(None)] * processed_data[i].ndim
                        if shift_val > 0:
                            slices[axis] = slice(0, -shift_val)
                            target_slices = [slice(None)] * processed_data[i].ndim
                            target_slices[axis] = slice(shift_val, None)
                            processed_data[i][tuple(target_slices)] = processed_data[i][tuple(slices)]
                            target_slices[axis] = slice(0, shift_val)
                            processed_data[i][tuple(target_slices)] = 0
                        else:
                            slices[axis] = slice(-shift_val, None)
                            target_slices = [slice(None)] * processed_data[i].ndim
                            target_slices[axis] = slice(0, shift_val)
                            processed_data[i][tuple(target_slices)] = processed_data[i][tuple(slices)]
                            target_slices[axis] = slice(shift_val, None)
                            processed_data[i][tuple(target_slices)] = 0
        
        if self.imaging_options["skull_strip"]:
            # Simulate skull stripping by creating a brain mask and applying it
            for i in range(processed_data.shape[0]):
                # Create a simple ellipsoid mask for the "brain"
                x, y, z = np.indices(processed_data[i].shape)
                center = np.array(processed_data[i].shape) // 2
                x = x - center[0]
                y = y - center[1]
                z = z - center[2]
                
                # Ellipsoid equation: (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1
                a, b, c = np.array(processed_data[i].shape) // 3
                brain_mask = (x/a)**2 + (y/b)**2 + (z/c)**2 <= 1
                
                # Apply the mask
                processed_data[i] = processed_data[i] * brain_mask
        
        return processed_data
    
    def _split_dataset(self, dataset: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Split the dataset into train, validation, and test sets.
        
        Args:
            dataset: Dictionary containing the multimodal dataset
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Extract labels for stratified splitting
        labels = dataset["labels"]
        
        # Calculate split sizes
        train_ratio = self.general_options["train_ratio"]
        val_ratio = self.general_options["val_ratio"]
        
        # First, split into train and temp (val + test)
        train_indices, temp_indices = train_test_split(
            np.arange(len(labels)),
            test_size=(1 - train_ratio),
            stratify=labels,
            random_state=42
        )
        
        # Then split temp into val and test
        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=(1 - val_ratio_adjusted),
            stratify=labels[temp_indices],
            random_state=42
        )
        
        # Create the split datasets
        train_data = {}
        val_data = {}
        test_data = {}
        
        # Split each modality
        for key, data in dataset.items():
            if isinstance(data, (np.ndarray, list)):
                data = np.array(data)
                train_data[key] = data[train_indices]
                val_data[key] = data[val_indices]
                test_data[key] = data[test_indices]
        
        return train_data, val_data, test_data
    
    def _balance_classes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Balance classes in the dataset.
        
        Args:
            data: Dictionary containing the dataset to balance
            
        Returns:
            Balanced dataset
        """
        if "labels" not in data:
            return data
        
        labels = data["labels"]
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # If there's only one class, return the original data
        if len(unique_labels) <= 1:
            return data
        
        # Find the class with the most samples
        max_count = np.max(counts)
        
        # Balance the dataset by upsampling minority classes
        balanced_data = {}
        for key in data.keys():
            balanced_modality = []
            
            for label in unique_labels:
                # Get indices for this class
                label_indices = np.where(labels == label)[0]
                
                # Get data for this class
                label_data = data[key][label_indices]
                
                # Upsample if needed
                if len(label_indices) < max_count:
                    if isinstance(label_data, np.ndarray):
                        # For NumPy arrays, use resample
                        upsampled_data = resample(
                            label_data,
                            replace=True,
                            n_samples=max_count,
                            random_state=42
                        )
                    else:
                        # For lists, use random sampling with replacement
                        upsampled_data = np.random.choice(
                            label_data,
                            size=max_count,
                            replace=True
                        )
                    
                    balanced_modality.append(upsampled_data)
                else:
                    balanced_modality.append(label_data)
            
            # Combine all classes
            balanced_data[key] = np.vstack(balanced_modality) if isinstance(balanced_modality[0], np.ndarray) else np.concatenate(balanced_modality)
        
        return balanced_data
    
    def _augment_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply data augmentation to the dataset.
        
        Args:
            data: Dictionary containing the dataset to augment
            
        Returns:
            Augmented dataset
        """
        # In a real implementation, this would apply various augmentation techniques
        # For this demo, we'll just return the original data
        # Augmentation would typically include:
        # - For text: synonym replacement, random insertion/deletion/swap
        # - For audio: time stretching, pitch shifting, adding noise
        # - For physiological: adding noise, scaling, warping
        # - For imaging: rotation, flipping, scaling, elastic deformation
        
        return data
