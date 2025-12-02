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


def build_next_activity_model(vocab_size, max_len, d_model=64, num_heads=4, 
                               num_blocks=2, dropout_rate=0.1):
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


def build_time_prediction_model(vocab_size, max_len, d_model=64, num_heads=4, 
                                 num_blocks=2, dropout_rate=0.1):
    seq_input = layers.Input(shape=(max_len,), name='sequence_input')
    temp_input = layers.Input(shape=(3,), name='temporal_input')
    x = layers.Embedding(vocab_size, d_model, mask_zero=True)(seq_input)
    x = PositionalEncoding(max_len, d_model)(x)
    for _ in range(num_blocks):
        x = TransformerBlock(d_model, num_heads, dff=128, dropout_rate=dropout_rate)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    combined = layers.Concatenate()([x, temp_input])
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1, activation='linear', name='time_output')(x)
    model = keras.Model(inputs=[seq_input, temp_input], outputs=output)
    return model
