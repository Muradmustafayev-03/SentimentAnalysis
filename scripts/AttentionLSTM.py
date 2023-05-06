from typing import List
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Lambda, Multiply, Permute
from keras.layers import Input, LSTM, Bidirectional, Dense
from keras.layers import Embedding, Dropout, BatchNormalization
from .constants import MAX_WORDS, MAX_TEXT_LEN, OUTPUT_SIZE


def attention(input_tensor: tf.Tensor) -> tf.Tensor:
    """Compute Attention weights and apply them to the input tensor.

    Args:
        input_tensor: A 3D tensor of shape (batch_size, time_steps, input_dim).

    Returns:
        A 2D tensor of shape (batch_size, input_dim), where each row is a weighted
        sum of the input tensor along the time axis.
    """
    batch_size, time_steps, input_dim = input_tensor.shape

    # Compute Attention scores
    attention_scores = Permute((2, 1))(input_tensor)
    attention_scores = Dense(time_steps, activation='sigmoid')(attention_scores)
    attention_scores = Dense(time_steps, activation='softmax')(attention_scores)

    # Apply Attention weights
    attention_scores = Permute((2, 1))(attention_scores)
    weighted_input = Multiply()([input_tensor, attention_scores])
    output = Lambda(lambda x: K.sum(x, axis=1))(weighted_input)

    return output


def build_unidirectional_lstm_attention_model(
        vocab_size: int,
        max_length: int,
        layers_sizes: List[int],
        output_size: int,
        dropout: float = 0.5,
        batch_normalization: bool = True,
        activation='sigmoid'
) -> Model:
    """
    Builds a unidirectional LSTM model with the specified architecture and Attention mechanism.

    Args:
        vocab_size: The size of the vocabulary.
        max_length: The maximum length of the input sequence.
        layers_sizes: A tuple containing the embedding size, a list of sizes.
        output_size: The size of the output layer.
        dropout: The dropout rate to use.
        batch_normalization: Whether to use batch normalization.
        activation: The activation function on the output layer

    Returns:
        A Keras model object.
    """

    EMBEDDING_SIZE, LSTM_SIZES = layers_sizes[0], layers_sizes[1:]

    input_layer = Input(shape=(max_length,))
    embedding_layer = Embedding(vocab_size, EMBEDDING_SIZE, input_length=max_length)(input_layer)
    lstm_layer = embedding_layer

    for layer_size in LSTM_SIZES:
        lstm_layer = LSTM(layer_size, return_sequences=True)(lstm_layer)
        lstm_layer = Dropout(dropout)(lstm_layer)
        if batch_normalization:
            lstm_layer = BatchNormalization()(lstm_layer)

    attention_layer = attention(lstm_layer)
    attention_layer = Dropout(dropout)(attention_layer)

    output_layer = Dense(output_size, activation=activation)(attention_layer)
    model = Model(inputs=[input_layer], outputs=[output_layer])

    return model


def build_bidirectional_lstm_attention_model(
        vocab_size: int,
        max_length: int,
        layers_sizes: List[int],
        output_size: int,
        dropout: float = 0.5,
        batch_normalization: bool = True,
        activation='sigmoid'
) -> Model:
    """
    Builds a bidirectional LSTM model with the specified architecture and Attention mechanism.

    Args:
        vocab_size: The size of the vocabulary.
        max_length: The maximum length of the input sequence.
        layers_sizes: A tuple containing the embedding size, a list of sizes.
        output_size: The size of the output layer.
        dropout: The dropout rate to use.
        batch_normalization: Whether to use batch normalization.
        activation: The activation function on the output layer

    Returns:
        A Keras model object.
    """

    EMBEDDING_SIZE, LSTM_SIZES = layers_sizes[0], layers_sizes[1:]

    input_layer = Input(shape=(max_length,))
    embedding_layer = Embedding(vocab_size, EMBEDDING_SIZE, input_length=max_length)(input_layer)
    lstm_layer = embedding_layer

    for layer_size in LSTM_SIZES:
        lstm_layer = Bidirectional(LSTM(layer_size, return_sequences=True))(lstm_layer)
        lstm_layer = Dropout(dropout)(lstm_layer)
        if batch_normalization:
            lstm_layer = BatchNormalization()(lstm_layer)

    attention_layer = attention(lstm_layer)
    attention_layer = Dropout(dropout)(attention_layer)

    output_layer = Dense(output_size, activation=activation)(attention_layer)
    model = Model(inputs=[input_layer], outputs=[output_layer])

    return model


UnidirectionalLSTMAttention = build_unidirectional_lstm_attention_model(MAX_WORDS, MAX_TEXT_LEN, [64, 32, 16], OUTPUT_SIZE)
BidirectionalLSTMAttention = build_bidirectional_lstm_attention_model(MAX_WORDS, MAX_TEXT_LEN, [64, 32, 16], OUTPUT_SIZE)
