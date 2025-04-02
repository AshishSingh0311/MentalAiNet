import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import datetime

class ModelTrainer:
    """
    Trainer for MH-Net models.
    
    This class handles the training process for MH-Net models, including
    callbacks, learning rate scheduling, and data handling.
    """
    
    def __init__(
        self,
        model,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_epochs: int = 30,
        early_stopping: bool = True,
        patience: int = 10,
        loss: str = "Categorical Cross Entropy",
        optimizer: str = "Adam",
        class_weights: bool = True,
        checkpoint_dir: str = "./checkpoints"
    ):
        """
        Initialize the ModelTrainer.
        
        Args:
            model: The MH-Net model to train
            batch_size: Batch size for training
            learning_rate: Learning rate for the optimizer
            num_epochs: Number of epochs to train
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            loss: Loss function to use
            optimizer: Optimizer to use
            class_weights: Whether to use class weights for imbalanced data
            checkpoint_dir: Directory to save model checkpoints
        """
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.loss = loss
        self.optimizer = optimizer
        self.class_weights = class_weights
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Compile the model
        self.model.compile(
            optimizer=optimizer,
            learning_rate=learning_rate,
            loss=loss,
            metrics=["accuracy"]
        )
    
    def _prepare_callbacks(self, model_name: str) -> List[tf.keras.callbacks.Callback]:
        """
        Prepare training callbacks.
        
        Args:
            model_name: Name of the model for checkpoint and log files
            
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Create timestamp for unique directory names
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Model checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}_{timestamp}.h5")
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        if self.early_stopping:
            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stop)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=self.patience // 2,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard logging
        log_dir = os.path.join("logs", f"{model_name}_{timestamp}")
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch"
        )
        callbacks.append(tensorboard)
        
        return callbacks
    
    def _prepare_data(
        self,
        data: Dict[str, Any],
        is_train: bool = True
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Prepare data for training or evaluation.
        
        Args:
            data: Dictionary containing the multimodal data
            is_train: Whether this is training data
            
        Returns:
            Tuple of (input_data, targets)
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
        
        # Convert labels to one-hot encoding
        if "labels" in data:
            labels = data["labels"]
            num_classes = len(np.unique(labels))
            targets = to_categorical(labels, num_classes=num_classes)
        else:
            raise ValueError("Labels are required for training or evaluation")
        
        return input_data, targets
    
    def _compute_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for imbalanced data.
        
        Args:
            labels: Array of class labels
            
        Returns:
            Dictionary mapping class indices to weights
        """
        unique_classes = np.unique(labels)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=labels
        )
        
        return {i: weight for i, weight in zip(unique_classes, class_weights)}
    
    def train(
        self,
        train_data: Dict[str, Any],
        val_data: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None
    ) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            train_data: Dictionary containing the training data
            val_data: Dictionary containing the validation data
            model_name: Name of the model for checkpoint and log files
            
        Returns:
            A History object containing the training history
        """
        # Prepare model name
        if model_name is None:
            model_name = f"MHNet_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare callbacks
        callbacks = self._prepare_callbacks(model_name)
        
        # Prepare training data
        train_inputs, train_targets = self._prepare_data(train_data, is_train=True)
        
        # Prepare validation data if provided
        validation_data = None
        if val_data is not None:
            val_inputs, val_targets = self._prepare_data(val_data, is_train=False)
            validation_data = (val_inputs, val_targets)
        
        # Compute class weights if needed
        class_weights_dict = None
        if self.class_weights and "labels" in train_data:
            class_weights_dict = self._compute_class_weights(train_data["labels"])
            print("Using class weights:", class_weights_dict)
        
        # Train the model
        print(f"Starting training for {self.num_epochs} epochs...")
        history = self.model.fit(
            x=train_inputs,
            y=train_targets,
            validation_data=validation_data,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            callbacks=callbacks,
            class_weight=class_weights_dict,
            verbose=1
        )
        
        print("Training completed.")
        
        return history
    
    def evaluate(
        self,
        test_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Dictionary containing the test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Prepare test data
        test_inputs, test_targets = self._prepare_data(test_data, is_train=False)
        
        # Evaluate the model
        print("Evaluating model...")
        results = self.model.evaluate(
            x=test_inputs,
            y=test_targets,
            batch_size=self.batch_size,
            verbose=1
        )
        
        # Get metric names
        metric_names = self.model.model.metrics_names
        
        # Create results dictionary
        evaluation_results = {name: value for name, value in zip(metric_names, results)}
        
        print("Evaluation results:", evaluation_results)
        
        return evaluation_results
    
    def predict(
        self,
        data: Dict[str, Any]
    ) -> np.ndarray:
        """
        Generate predictions using the model.
        
        Args:
            data: Dictionary containing the input data
            
        Returns:
            Array of predictions
        """
        # Prepare input data
        inputs, _ = self._prepare_data(data, is_train=False)
        
        # Generate predictions
        predictions = self.model.predict(x=inputs, batch_size=self.batch_size)
        
        return predictions
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def plot_training_history(self, history: tf.keras.callbacks.History) -> Tuple[plt.Figure, plt.Figure]:
        """
        Plot the training history.
        
        Args:
            history: History object returned by model.fit()
            
        Returns:
            Tuple of (loss_figure, accuracy_figure)
        """
        # Plot training & validation loss
        loss_fig, loss_ax = plt.subplots(figsize=(10, 6))
        loss_ax.plot(history.history["loss"], label="Training Loss")
        if "val_loss" in history.history:
            loss_ax.plot(history.history["val_loss"], label="Validation Loss")
        loss_ax.set_title("Loss vs. Epochs")
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Loss")
        loss_ax.legend()
        loss_ax.grid(True)
        
        # Plot training & validation accuracy
        acc_fig, acc_ax = plt.subplots(figsize=(10, 6))
        acc_ax.plot(history.history["accuracy"], label="Training Accuracy")
        if "val_accuracy" in history.history:
            acc_ax.plot(history.history["val_accuracy"], label="Validation Accuracy")
        acc_ax.set_title("Accuracy vs. Epochs")
        acc_ax.set_xlabel("Epoch")
        acc_ax.set_ylabel("Accuracy")
        acc_ax.legend()
        acc_ax.grid(True)
        
        return loss_fig, acc_fig
