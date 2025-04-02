import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import ResNet50
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

class MultiheadSelfAttention(layers.Layer):
    """
    Multi-head self-attention layer implementation.
    
    This layer implements the multi-head self-attention mechanism described in 
    "Attention Is All You Need" paper by Vaswani et al.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float = 0.1, **kwargs):
        """
        Initialize the multi-head self-attention layer.
        
        Args:
            embed_dim: The embedding dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout rate for regularization
        """
        super(MultiheadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension {embed_dim} should be divisible by number of heads {num_heads}")
        
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)
        self.dropout = layers.Dropout(dropout_rate)
        self.attention_weights = None  # Store attention weights for explainability
    
    def attention(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Calculate scaled dot-product attention.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_score, axis=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to value
        attention_output = tf.matmul(attention_weights, value)
        
        return attention_output, attention_weights
    
    def separate_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """
        Separate the heads from the input tensor.
        
        Args:
            x: Input tensor
            batch_size: Batch size
            
        Returns:
            Tensor with separated heads
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Forward pass for the multi-head self-attention layer.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Output tensor after multi-head self-attention
        """
        batch_size = tf.shape(inputs)[0]
        
        # Linear projections
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        # Separate heads
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        
        # Compute attention
        attention_output, attention_weights = self.attention(query, key, value)
        self.attention_weights = attention_weights  # Store for explainability
        
        # Reshape and combine heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.embed_dim))
        
        # Final linear projection
        outputs = self.combine_heads(concat_attention)
        
        return outputs
    
    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer.
        
        Returns:
            Dictionary containing the layer configuration
        """
        config = super(MultiheadSelfAttention, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate
        })
        return config


class TransformerBlock(layers.Layer):
    """
    Transformer block implementation.
    
    This layer implements a transformer block consisting of multi-head self-attention
    followed by a feed-forward neural network, with residual connections and layer
    normalization.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        """
        Initialize the transformer block.
        
        Args:
            embed_dim: The embedding dimension
            num_heads: Number of attention heads
            ff_dim: Hidden dimension of the feed-forward network
            dropout_rate: Dropout rate for regularization
        """
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        self.attention = MultiheadSelfAttention(embed_dim, num_heads, dropout_rate)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass for the transformer block.
        
        Args:
            inputs: Input tensor
            training: Whether the layer should behave in training mode
            
        Returns:
            Output tensor after transformer block processing
        """
        # Self-attention with residual connection and layer normalization
        attention_output = self.attention(inputs)
        attention_output = self.dropout1(attention_output, training=training)
        output1 = self.layernorm1(inputs + attention_output)
        
        # Feed-forward network with residual connection and layer normalization
        ffn_output = self.ffn(output1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(output1 + ffn_output)
    
    def get_attention_weights(self) -> tf.Tensor:
        """
        Get the attention weights for explainability.
        
        Returns:
            Attention weights tensor
        """
        return self.attention.attention_weights
    
    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer.
        
        Returns:
            Dictionary containing the layer configuration
        """
        config = super(TransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate
        })
        return config


class ModalityEncoder(layers.Layer):
    """
    Encoder for a specific data modality.
    
    This layer processes a specific modality (text, audio, physiological, imaging)
    and encodes it into a fixed-size representation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        num_transformer_layers: int = 4,
        modality_type: str = "text",
        **kwargs
    ):
        """
        Initialize the modality encoder.
        
        Args:
            embed_dim: The embedding dimension
            num_heads: Number of attention heads
            ff_dim: Hidden dimension of the feed-forward network
            dropout_rate: Dropout rate for regularization
            num_transformer_layers: Number of transformer layers
            modality_type: Type of modality to encode ('text', 'audio', 'physiological', 'imaging')
        """
        super(ModalityEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.num_transformer_layers = num_transformer_layers
        self.modality_type = modality_type
        
        # Initial projection to embed_dim
        self.projection = layers.Dense(embed_dim, activation="relu")
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_transformer_layers)
        ]
        
        # Modality-specific processing layers
        if modality_type == "text":
            self.modality_projection = layers.Dense(embed_dim, activation="relu")
        elif modality_type == "audio":
            self.modality_projection = tf.keras.Sequential([
                layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu"),
                layers.MaxPool1D(pool_size=2),
                layers.Conv1D(filters=128, kernel_size=5, padding="same", activation="relu"),
                layers.MaxPool1D(pool_size=2),
                layers.Dense(embed_dim, activation="relu")
            ])
        elif modality_type == "physiological":
            if kwargs.get("physiological_input_shape", None) is not None:
                # If the physiological data is already in feature form (flattened)
                self.modality_projection = layers.Dense(embed_dim, activation="relu")
            else:
                # If the physiological data is in raw signal form
                self.modality_projection = tf.keras.Sequential([
                    layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu"),
                    layers.MaxPool1D(pool_size=2),
                    layers.Conv1D(filters=128, kernel_size=5, padding="same", activation="relu"),
                    layers.MaxPool1D(pool_size=2),
                    layers.Dense(embed_dim, activation="relu")
                ])
        elif modality_type == "imaging":
            if kwargs.get("imaging_input_shape", None) is not None:
                # If the imaging data is already in feature form
                self.modality_projection = layers.Dense(embed_dim, activation="relu")
            else:
                # If the imaging data is in raw form (2D or 3D)
                self.modality_projection = tf.keras.Sequential([
                    layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"),
                    layers.MaxPool2D(pool_size=2),
                    layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
                    layers.MaxPool2D(pool_size=2),
                    layers.GlobalAveragePooling2D(),
                    layers.Dense(embed_dim, activation="relu")
                ])
        else:
            raise ValueError(f"Unsupported modality type: {modality_type}")
        
        # Global pooling to get fixed-size representation
        self.global_pooling = layers.GlobalAveragePooling1D()
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass for the modality encoder.
        
        Args:
            inputs: Input tensor for the specific modality
            training: Whether the layer should behave in training mode
            
        Returns:
            Encoded representation of the modality
        """
        # Apply modality-specific projection
        x = self.modality_projection(inputs)
        
        # Apply initial projection to embed_dim if needed
        if self.modality_type not in ["text", "audio"]:
            x = self.projection(x)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        
        # Apply global pooling to get fixed-size representation
        # Only if the output is still a sequence
        if len(x.shape) > 2:
            encoded = self.global_pooling(x)
        else:
            encoded = x
        
        return encoded
    
    def get_attention_weights(self) -> List[tf.Tensor]:
        """
        Get attention weights from all transformer blocks for explainability.
        
        Returns:
            List of attention weight tensors
        """
        return [block.get_attention_weights() for block in self.transformer_blocks]
    
    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer.
        
        Returns:
            Dictionary containing the layer configuration
        """
        config = super(ModalityEncoder, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
            "num_transformer_layers": self.num_transformer_layers,
            "modality_type": self.modality_type
        })
        return config


class CrossModalFusion(layers.Layer):
    """
    Cross-modal fusion layer.
    
    This layer fuses representations from different modalities using 
    cross-attention and modality-specific weights.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float = 0.1, **kwargs):
        """
        Initialize the cross-modal fusion layer.
        
        Args:
            embed_dim: The embedding dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout rate for regularization
        """
        super(CrossModalFusion, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # Self-attention and modality attention
        self.self_attention = MultiheadSelfAttention(embed_dim, num_heads, dropout_rate)
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim * 4, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.dropout = layers.Dropout(dropout_rate)
        
        # Modality importance weights
        self.modality_weights = layers.Dense(1, activation="softmax", use_bias=False)
    
    def call(self, inputs: List[tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Forward pass for the cross-modal fusion layer.
        
        Args:
            inputs: List of input tensors from different modalities
            training: Whether the layer should behave in training mode
            
        Returns:
            Fused representation
        """
        # Stack the modality representations
        stacked_inputs = tf.stack(inputs, axis=1)  # [batch_size, num_modalities, embed_dim]
        
        # Apply self-attention to learn relationships between modalities
        attended = self.self_attention(stacked_inputs)
        attended = self.dropout(attended, training=training)
        attended = self.layer_norm1(stacked_inputs + attended)
        
        # Apply feed-forward network
        outputs = self.ffn(attended)
        outputs = self.dropout(outputs, training=training)
        outputs = self.layer_norm2(attended + outputs)
        
        # Learn importance weights for each modality
        modality_scores = self.modality_weights(outputs)  # [batch_size, num_modalities, 1]
        weighted_outputs = outputs * modality_scores
        
        # Sum over modalities to get final representation
        fused = tf.reduce_sum(weighted_outputs, axis=1)  # [batch_size, embed_dim]
        
        return fused
    
    def get_modality_weights(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        """
        Get the learned importance weights for each modality.
        
        Args:
            inputs: List of input tensors from different modalities
            
        Returns:
            Modality importance weights
        """
        stacked_inputs = tf.stack(inputs, axis=1)
        attended = self.self_attention(stacked_inputs)
        attended = self.layer_norm1(stacked_inputs + attended)
        
        # Get the importance weights for each modality
        modality_scores = self.modality_weights(attended)  # [batch_size, num_modalities, 1]
        return modality_scores
    
    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer.
        
        Returns:
            Dictionary containing the layer configuration
        """
        config = super(CrossModalFusion, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate
        })
        return config


class MHNetModel:
    """
    MH-Net model for multimodal mental health diagnostics.
    
    This class implements the MH-Net model architecture, which uses transformers
    to process and fuse multimodal data for mental health diagnostics.
    """
    
    def __init__(
        self,
        model_type: str = "MH-Net (Multimodal Transformer)",
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout_rate: float = 0.1,
        input_shapes: Dict[str, Tuple[int, ...]] = None,
        num_classes: int = 5
    ):
        """
        Initialize the MH-Net model.
        
        Args:
            model_type: Type of model to create
            embed_dim: The embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout_rate: Dropout rate for regularization
            input_shapes: Dictionary mapping modality names to their input shapes
            num_classes: Number of output classes
        """
        self.model_type = model_type
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.input_shapes = input_shapes or {}
        self.num_classes = num_classes
        
        # Create the model
        self.model = self._build_model()
    
    def _build_model(self) -> Model:
        """
        Build the MH-Net model.
        
        Returns:
            Compiled Keras model
        """
        inputs = {}
        encoded_modalities = []
        
        # Create inputs and encoders for each modality
        if self.model_type == "MH-Net (Multimodal Transformer)":
            # Text modality
            if "text" in self.input_shapes and self.input_shapes["text"] is not None:
                text_shape = self.input_shapes["text"]
                inputs["text"] = Input(shape=text_shape, name="text_input")
                text_encoder = ModalityEncoder(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    ff_dim=self.embed_dim * 4,
                    dropout_rate=self.dropout_rate,
                    num_transformer_layers=self.num_layers,
                    modality_type="text"
                )
                encoded_text = text_encoder(inputs["text"])
                encoded_modalities.append(encoded_text)
            
            # Audio modality
            if "audio" in self.input_shapes and self.input_shapes["audio"] is not None:
                audio_shape = self.input_shapes["audio"]
                inputs["audio"] = Input(shape=audio_shape, name="audio_input")
                audio_encoder = ModalityEncoder(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    ff_dim=self.embed_dim * 4,
                    dropout_rate=self.dropout_rate,
                    num_transformer_layers=self.num_layers,
                    modality_type="audio"
                )
                encoded_audio = audio_encoder(inputs["audio"])
                encoded_modalities.append(encoded_audio)
            
            # Physiological modality
            if "physiological" in self.input_shapes and self.input_shapes["physiological"] is not None:
                physio_shape = self.input_shapes["physiological"]
                inputs["physiological"] = Input(shape=physio_shape, name="physiological_input")
                physio_encoder = ModalityEncoder(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    ff_dim=self.embed_dim * 4,
                    dropout_rate=self.dropout_rate,
                    num_transformer_layers=self.num_layers,
                    modality_type="physiological",
                    physiological_input_shape=physio_shape
                )
                encoded_physio = physio_encoder(inputs["physiological"])
                encoded_modalities.append(encoded_physio)
            
            # Imaging modality
            if "imaging" in self.input_shapes and self.input_shapes["imaging"] is not None:
                imaging_shape = self.input_shapes["imaging"]
                inputs["imaging"] = Input(shape=imaging_shape, name="imaging_input")
                imaging_encoder = ModalityEncoder(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    ff_dim=self.embed_dim * 4,
                    dropout_rate=self.dropout_rate,
                    num_transformer_layers=self.num_layers,
                    modality_type="imaging",
                    imaging_input_shape=imaging_shape
                )
                encoded_imaging = imaging_encoder(inputs["imaging"])
                encoded_modalities.append(encoded_imaging)
            
            # If we have multiple modalities, use cross-modal fusion
            if len(encoded_modalities) > 1:
                fusion = CrossModalFusion(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate
                )
                fused_representation = fusion(encoded_modalities)
            else:
                fused_representation = encoded_modalities[0]
            
            # Add classification head
            x = layers.Dropout(self.dropout_rate)(fused_representation)
            x = layers.Dense(256, activation="relu", kernel_regularizer=l2(0.001))(x)
            x = layers.Dropout(self.dropout_rate)(x)
            outputs = layers.Dense(self.num_classes, activation="softmax", name="predictions")(x)
            
            # Create the model
            model = Model(inputs=list(inputs.values()), outputs=outputs, name="MH-Net")
        
        # Unimodal models
        else:
            modality = self.model_type.split("(")[1].split(")")[0].strip().lower()
            
            if modality == "text only" and "text" in self.input_shapes:
                text_shape = self.input_shapes["text"]
                inputs["text"] = Input(shape=text_shape, name="text_input")
                text_encoder = ModalityEncoder(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    ff_dim=self.embed_dim * 4,
                    dropout_rate=self.dropout_rate,
                    num_transformer_layers=self.num_layers,
                    modality_type="text"
                )
                encoded_text = text_encoder(inputs["text"])
                x = layers.Dropout(self.dropout_rate)(encoded_text)
            
            elif modality == "audio only" and "audio" in self.input_shapes:
                audio_shape = self.input_shapes["audio"]
                inputs["audio"] = Input(shape=audio_shape, name="audio_input")
                audio_encoder = ModalityEncoder(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    ff_dim=self.embed_dim * 4,
                    dropout_rate=self.dropout_rate,
                    num_transformer_layers=self.num_layers,
                    modality_type="audio"
                )
                encoded_audio = audio_encoder(inputs["audio"])
                x = layers.Dropout(self.dropout_rate)(encoded_audio)
            
            elif modality == "physiological only" and "physiological" in self.input_shapes:
                physio_shape = self.input_shapes["physiological"]
                inputs["physiological"] = Input(shape=physio_shape, name="physiological_input")
                physio_encoder = ModalityEncoder(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    ff_dim=self.embed_dim * 4,
                    dropout_rate=self.dropout_rate,
                    num_transformer_layers=self.num_layers,
                    modality_type="physiological",
                    physiological_input_shape=physio_shape
                )
                encoded_physio = physio_encoder(inputs["physiological"])
                x = layers.Dropout(self.dropout_rate)(encoded_physio)
            
            elif modality == "imaging only" and "imaging" in self.input_shapes:
                imaging_shape = self.input_shapes["imaging"]
                inputs["imaging"] = Input(shape=imaging_shape, name="imaging_input")
                imaging_encoder = ModalityEncoder(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    ff_dim=self.embed_dim * 4,
                    dropout_rate=self.dropout_rate,
                    num_transformer_layers=self.num_layers,
                    modality_type="imaging",
                    imaging_input_shape=imaging_shape
                )
                encoded_imaging = imaging_encoder(inputs["imaging"])
                x = layers.Dropout(self.dropout_rate)(encoded_imaging)
            
            else:
                raise ValueError(f"Unsupported model type or missing input shapes for {self.model_type}")
            
            # Add classification head
            x = layers.Dense(256, activation="relu", kernel_regularizer=l2(0.001))(x)
            x = layers.Dropout(self.dropout_rate)(x)
            outputs = layers.Dense(self.num_classes, activation="softmax", name="predictions")(x)
            
            # Create the model
            model = Model(inputs=list(inputs.values()), outputs=outputs, name=f"MH-Net-{modality}")
        
        return model
    
    def compile(
        self,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        loss: str = "categorical_crossentropy",
        metrics: List[str] = None
    ) -> None:
        """
        Compile the model.
        
        Args:
            optimizer: Name of the optimizer or optimizer instance
            learning_rate: Learning rate for the optimizer
            loss: Loss function to use
            metrics: List of metrics to track
        """
        metrics = metrics or ["accuracy"]
        
        # Create optimizer
        if optimizer == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "sgd":
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == "rmsprop":
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer == "adamw":
            opt = tf.keras.optimizers.experimental.AdamW(learning_rate=learning_rate, weight_decay=0.001)
        else:
            opt = optimizer
        
        # Create loss
        if loss == "Categorical Cross Entropy":
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
        elif loss == "Binary Cross Entropy":
            loss_fn = tf.keras.losses.BinaryCrossentropy()
        elif loss == "Focal Loss":
            def focal_loss(gamma=2., alpha=0.25):
                def focal_loss_fn(y_true, y_pred):
                    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
                    cross_entropy = -y_true * tf.math.log(y_pred)
                    
                    # Calculate the focal weight
                    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1-y_pred)
                    focal_weight = tf.pow(1-p_t, gamma)
                    
                    # Add alpha for class imbalance
                    alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1-alpha)
                    focal_weight = alpha_factor * focal_weight
                    
                    return tf.reduce_sum(focal_weight * cross_entropy, axis=-1)
                return focal_loss_fn
            
            loss_fn = focal_loss()
        else:
            loss_fn = loss
        
        # Compile the model
        self.model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)
    
    def fit(
        self,
        x: Dict[str, np.ndarray],
        y: np.ndarray,
        validation_data: Tuple[Dict[str, np.ndarray], np.ndarray] = None,
        batch_size: int = 32,
        epochs: int = 10,
        callbacks: List[tf.keras.callbacks.Callback] = None,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            x: Dictionary mapping input names to input data
            y: Target data
            validation_data: Tuple of (validation_inputs, validation_targets)
            batch_size: Number of samples per gradient update
            epochs: Number of epochs to train the model
            callbacks: List of callbacks to apply during training
            verbose: Verbosity mode
            
        Returns:
            A History object containing the training history
        """
        return self.model.fit(
            x=x,
            y=y,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
    
    def evaluate(
        self,
        x: Dict[str, np.ndarray],
        y: np.ndarray,
        batch_size: int = 32,
        verbose: int = 1
    ) -> List[float]:
        """
        Evaluate the model.
        
        Args:
            x: Dictionary mapping input names to input data
            y: Target data
            batch_size: Number of samples per batch
            verbose: Verbosity mode
            
        Returns:
            List of loss and metrics values
        """
        return self.model.evaluate(x=x, y=y, batch_size=batch_size, verbose=verbose)
    
    def predict(
        self,
        x: Dict[str, np.ndarray],
        batch_size: int = 32,
        verbose: int = 0
    ) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            x: Dictionary mapping input names to input data
            batch_size: Number of samples per batch
            verbose: Verbosity mode
            
        Returns:
            Array of predictions
        """
        return self.model.predict(x=x, batch_size=batch_size, verbose=verbose)
    
    def save(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
    
    def load_weights(self, filepath: str) -> None:
        """
        Load weights from a file.
        
        Args:
            filepath: Path to the weights file
        """
        self.model.load_weights(filepath)
    
    def summary(self) -> None:
        """
        Print a summary of the model.
        """
        self.model.summary()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the model configuration.
        
        Returns:
            Dictionary containing the model configuration
        """
        return {
            "model_type": self.model_type,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "input_shapes": self.input_shapes,
            "num_classes": self.num_classes
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MHNetModel':
        """
        Create a model from a configuration.
        
        Args:
            config: Dictionary containing the model configuration
            
        Returns:
            Instantiated model
        """
        return cls(**config)
