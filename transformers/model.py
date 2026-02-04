"""
Transformer Model Architecture for Process Prediction
Includes timestep-preserving variants for explainability

File: transformers/model.py
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(max_len, d_model)
    
    def positional_encoding(self, max_len, d_model):
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)
    
    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False):
        attn_output = self.att(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# ============================================================================
# ACTIVITY PREDICTION MODEL (Unchanged - uses GlobalAveragePooling)
# ============================================================================
def build_next_activity_model(vocab_size, max_len, d_model=64, num_heads=4, 
                               num_blocks=2, dropout_rate=0.1):
    """
    Build model for next activity prediction.
    Uses GlobalAveragePooling1D (appropriate for classification).
    """
    inputs = layers.Input(shape=(max_len,))
    x = layers.Embedding(vocab_size, d_model, mask_zero=True)(inputs)
    x = PositionalEncoding(max_len, d_model)(x)
    
    for _ in range(num_blocks):
        x = TransformerBlock(d_model, num_heads, dff=128, dropout_rate=dropout_rate)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(vocab_size, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# ============================================================================
# TIME PREDICTION MODEL - TIMESTEP-PRESERVING VERSION (RECOMMENDED)
# ============================================================================
def build_time_prediction_model(vocab_size, max_len, d_model=64, num_heads=4, 
                                num_blocks=2, dropout_rate=0.1, 
                                use_timestep_explainability=True):
    """
    Build model for time prediction (event time or remaining time).
    
    Args:
        vocab_size: Size of activity vocabulary
        max_len: Maximum sequence length
        d_model: Transformer embedding dimension
        num_heads: Number of attention heads
        num_blocks: Number of transformer blocks
        dropout_rate: Dropout rate
        use_timestep_explainability: If True, returns per-timestep predictions
                                     for explainability (RECOMMENDED)
    
    Returns:
        If use_timestep_explainability=True:
            Model with outputs: [final_output, timestep_predictions]
        If use_timestep_explainability=False:
            Model with output: final_output only (original behavior)
    """
    seq_input = layers.Input(shape=(max_len,), name='sequence_input')
    temp_input = layers.Input(shape=(3,), name='temporal_input')
    
    # Embedding and positional encoding
    x = layers.Embedding(vocab_size, d_model, mask_zero=True)(seq_input)
    x = PositionalEncoding(max_len, d_model)(x)
    
    # Transformer blocks
    for _ in range(num_blocks):
        x = TransformerBlock(d_model, num_heads, dff=128, dropout_rate=dropout_rate)(x)
    
    if use_timestep_explainability:
        # ===================================================================
        # TIMESTEP-PRESERVING VERSION (for explainability)
        # ===================================================================
        # Keep timestep dimension: x has shape (batch_size, max_len, d_model)
        
        # Expand temporal features to match each timestep
        # Use Lambda layer to wrap TensorFlow operations
        temp_expanded = layers.Lambda(
            lambda t: tf.tile(tf.expand_dims(t, axis=1), [1, max_len, 1]),
            name='temporal_expansion'
        )(temp_input)
        
        # Concatenate sequence features with temporal features at each timestep
        x = layers.Concatenate(axis=-1, name='feature_concat')([x, temp_expanded])
        
        # Generate prediction for EACH timestep
        x = layers.Dense(64, activation='relu', name='timestep_dense1')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(32, activation='relu', name='timestep_dense2')(x)
        timestep_predictions = layers.Dense(1, activation='linear', name='timestep_output')(x)
        
        # Squeeze last dimension - no name to avoid Keras treating it as a loss output
        timestep_predictions_squeezed = layers.Lambda(
            lambda t: tf.squeeze(t, axis=-1)
        )(timestep_predictions)
        
        # Create mask using Lambda layer (KerasTensor-safe)
        mask = layers.Lambda(
            lambda s: tf.cast(tf.not_equal(s, 0), tf.float32)
        )(seq_input)
        
        # Apply mask to zero out padded positions
        timestep_predictions_masked = layers.Multiply()(
            [timestep_predictions_squeezed, mask]
        )
        
        # Aggregate timestep predictions for final output
        # Sum and count non-padded timesteps
        sum_preds = layers.Lambda(
            lambda t: tf.reduce_sum(t, axis=1, keepdims=True)
        )(timestep_predictions_masked)
        
        seq_lengths = layers.Lambda(
            lambda m: tf.reduce_sum(m, axis=1, keepdims=True)
        )(mask)
        
        # Divide to get average
        final_output = layers.Lambda(
            lambda inputs: inputs[0] / (inputs[1] + 1e-7),  # Add epsilon to avoid division by zero
            name='time_output'
        )([sum_preds, seq_lengths])
        
        # Return both final output and timestep predictions (squeezed version)
        model = keras.Model(
            inputs=[seq_input, temp_input],
            outputs=[final_output, timestep_predictions_squeezed],
            name='time_prediction_timestep_explainable'
        )
        
    else:
        # ORIGINAL VERSION (without timestep preservation)
        # Use GlobalAveragePooling1D (collapses timestep dimension)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Combine with temporal features
        combined = layers.Concatenate()([x, temp_input])
        x = layers.Dense(128, activation='relu')(combined)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(32, activation='relu')(x)
        final_output = layers.Dense(1, activation='linear', name='time_output')(x)
        
        # Return only final output
        model = keras.Model(
            inputs=[seq_input, temp_input],
            outputs=final_output,
            name='time_prediction_original'
        )
    
    return model



def compile_time_model(model, learning_rate=0.001):
    
    if len(model.outputs) > 1:
        # Compile with loss only on final output
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={'time_output': 'mse'},
            metrics={'time_output': ['mae', 'mse']}
        )
    else:
        # Original single-output version
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
    
    return model


def predict_time(model, X_seq, X_temp):
    """
    Make predictions with time model, handling both output types.
    
    Args:
        model: Time prediction model
        X_seq: Sequence input
        X_temp: Temporal input
    
    Returns:
        final_predictions: (num_samples,) array of time predictions
        timestep_predictions: (num_samples, max_len) array or None
    """
    outputs = model.predict([X_seq, X_temp], verbose=0)
    
    if isinstance(outputs, list) and len(outputs) > 1:
        # Timestep-explainable version
        final_predictions = outputs[0].flatten()
        timestep_predictions = outputs[1]
        return final_predictions, timestep_predictions
    else:
        # Original version
        final_predictions = outputs.flatten() if hasattr(outputs, 'flatten') else outputs
        return final_predictions, None



def build_time_prediction_model_original(vocab_size, max_len, d_model=64, 
                                        num_heads=4, num_blocks=2, dropout_rate=0.1):
    """
    Original time prediction model (for backward compatibility).
    Uses GlobalAveragePooling1D.
    """
    return build_time_prediction_model(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=d_model,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
        use_timestep_explainability=False
    )