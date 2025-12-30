"""
This module defines the CNN-Transformer model architecture for the ECG Apnea project.
It combines the strengths of Convolutional Neural Networks (CNNs) for local feature
extraction with Transformer Encoders for capturing long-range dependencies in
physiological time-series data.

Functions:
    - positional_encoding(): Generates positional encodings for transformer inputs.
    - transformer_encoder_block(): Constructs a single block of the Transformer Encoder.
    - create_model(): Builds the complete CNN-Transformer model.
"""

# Deep learning framework and layers
import tensorflow as tf
from tensorflow.keras.layers import (LayerNormalization, MultiHeadAttention, Add,
                                     Dense, Dropout, Flatten, Input, Conv1D,
                                     MaxPooling1D)
from tensorflow.keras.models import Model


def positional_encoding(seq_length: int, d_model: int) -> tf.Tensor:
    """
    Generates positional encodings to inject information about the relative or
    absolute position of the tokens in the sequence. This is crucial for
    Transformers as they do not inherently process sequential data.

    The formula used is from the "Attention Is All You Need" paper:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_length (int): The maximum length of the input sequence (e.g., number of timesteps).
        d_model (int): The dimensionality of the model's embeddings (feature dimension).

    Returns:
        tf.Tensor: A tensor of shape (seq_length, d_model) containing the positional encodings.
    """
    # Create a tensor for positions (0 to seq_length-1)
    position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
    
    # Create a tensor for the division term (1 / 10000^(2i/d_model))
    # This term decreases with increasing 'i' (feature dimension index)
    div_term = tf.pow(10000.0, 2.0 * tf.range(d_model // 2, dtype=tf.float32) / d_model)
    
    # Calculate the argument for sin and cos functions
    # Using tf.matmul for broadcasting 'position' across 'div_term'
    angle = tf.matmul(position, div_term[tf.newaxis, :])
    
    # Apply sin to even indices and cos to odd indices of the angle
    sin = tf.sin(angle)
    cos = tf.cos(angle)
    
    # Concatenate sin and cos results to form the full positional encoding
    # This results in a tensor of shape (seq_length, d_model)
    pos_encoding = tf.concat([sin, cos], axis=-1)
    return pos_encoding


def transformer_encoder_block(inputs: tf.Tensor, num_heads: int, key_dim: int, dropout_rate: float) -> tf.Tensor:
    """
    Constructs a single Transformer Encoder block as described in "Attention Is All You Need".
    This block is responsible for processing sequential data and capturing dependencies
    between different parts of the sequence.

    The block consists of:
    1. Layer Normalization
    2. Multi-Head Self-Attention
    3. Add & Normalize (Residual connection + Layer Norm)
    4. Position-wise Feed-Forward Network
    5. Add & Normalize (Residual connection + Layer Norm)
    6. Dropout

    Args:
        inputs (tf.Tensor): The input tensor to the encoder block (typically from CNN output).
                            Expected shape: (batch_size, sequence_length, feature_dimension).
        num_heads (int): The number of attention heads in the MultiHeadAttention layer.
                         More heads allow the model to focus on different parts of the input.
        key_dim (int): The dimensionality of the key, query, and value vectors in the
                       MultiHeadAttention layer.
        dropout_rate (float): The dropout rate to apply for regularization, helping
                              to prevent overfitting.

    Returns:
        tf.Tensor: The output tensor of the Transformer Encoder block.
                   Shape: (batch_size, sequence_length, feature_dimension).
    """
    # --- Layer Normalization (Pre-Attention) ---
    # Normalizes the input across the feature dimension, stabilizing training.
    normalized_input = LayerNormalization()(inputs)

    # --- Positional Encoding ---
    # Get the static sequence length and feature dimension from the input shape.
    # These are used to generate positional encodings.
    seq_length = inputs.shape[1]
    d_model = inputs.shape[2]
    pos_enc = positional_encoding(seq_length, d_model=d_model)
    
    # Add positional encodings to the input. This provides the model with
    # information about the order of the elements in the sequence.
    transformer_input_with_pos = normalized_input + pos_enc

    # --- Multi-Head Attention ---
    # Allows the model to jointly attend to information from different representation
    # subspaces at different positions. Here, it's a self-attention mechanism
    # (query, key, value all come from the same input).
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim
    )(transformer_input_with_pos, transformer_input_with_pos)

    # --- Add & Normalize (Post-Attention) ---
    # Residual connection: adds the input of the block to its output,
    # helping gradients flow through the network.
    # Then, applies Layer Normalization.
    attention_output = Add()([transformer_input_with_pos, attention_output])
    normalized_output = LayerNormalization()(attention_output)

    # --- Feed-Forward Network (FFN) ---
    # A simple, position-wise fully connected feed-forward network applied independently
    # to each position (timestep) in the sequence. It typically consists of two linear
    # transformations with a ReLU activation in between.
    ff_output = Dense(128, activation='relu')(normalized_output) # First linear transformation + ReLU
    ff_output = Dense(128)(ff_output)                             # Second linear transformation

    # --- Add & Normalize (Post-FFN) ---
    # Another residual connection followed by Layer Normalization.
    encoder_output = Add()([normalized_output, ff_output])
    normalized_encoder_output = LayerNormalization()(encoder_output)

    # --- Dropout for Regularization ---
    # Randomly sets a fraction of input units to 0 at each update during training
    # to prevent overfitting.
    dropout_output = Dropout(dropout_rate)(normalized_encoder_output)

    return dropout_output


def create_model(input_shape: tuple, num_heads: int = 2, key_dim: int = 32, dropout_rate: float = 0.5) -> Model:
    """
    Constructs and returns the complete CNN-Transformer model.

    The model architecture consists of:
    1. A series of 1D Convolutional layers (CNN block) for local feature extraction.
    2. A Transformer Encoder block for capturing long-range dependencies.
    3. Fully connected layers for classification.

    Args:
        input_shape (tuple): The shape of the input data, excluding the batch dimension.
                             Expected format: (sequence_length, num_features).
                             For this project: (900, 2) corresponding to (timesteps, [RRI, Amplitude]).
        num_heads (int): Number of attention heads for the TransformerEncoderBlock.
        key_dim (int): Dimension of the key/query/value for the TransformerEncoderBlock.
        dropout_rate (float): Dropout rate for the TransformerEncoderBlock.

    Returns:
        tensorflow.keras.models.Model: The compiled Keras model ready for training.
    """
    # --- Input Layer ---
    # Defines the shape of the input data to the model.
    inputs = Input(shape=input_shape)

    # --- CNN Block ---
    # This block processes the input sequence with 1D convolutions and pooling layers.
    # CNNs are effective at automatically learning local patterns (e.g., specific ECG morphologies).
    x = Conv1D(64, kernel_size=7, strides=1, padding="same", activation="relu", kernel_initializer="he_normal")(inputs)
    x = MaxPooling1D(pool_size=4)(x) # Reduces sequence length, summarizes features
    x = Conv1D(128, kernel_size=7, strides=1, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = MaxPooling1D(pool_size=4)(x)
    x = Conv1D(128, kernel_size=7, strides=1, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = MaxPooling1D(pool_size=4)(x)
    cnn_output = Dropout(0.5)(x) # Dropout for regularization after CNN features are extracted

    # --- Transformer Encoder Block ---
    # The output of the CNN is fed into the Transformer Encoder.
    # This layer helps the model understand the contextual relationships between
    # the features extracted by the CNN across the sequence.
    transformer_output = transformer_encoder_block(cnn_output, num_heads=num_heads, key_dim=key_dim, dropout_rate=dropout_rate)

    # --- Fully Connected Layers (Classifier Head) ---
    # Flattens the output of the Transformer to prepare it for a standard feed-forward classifier.
    fc_output = Flatten()(transformer_output)
    # A dense layer with ReLU activation for learning complex patterns from the combined features.
    fc_output = Dense(128, activation='relu')(fc_output)
    # Output layer with softmax activation for multi-class classification (Apnea vs. Non-Apnea).
    # It produces probabilities for each class.
    outputs = Dense(2, activation="softmax")(fc_output)

    # --- Model Definition ---
    # Create the Keras Model by specifying its inputs and outputs.
    model = Model(inputs=inputs, outputs=outputs)
    return model
