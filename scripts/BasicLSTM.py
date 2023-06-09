from typing import List
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Bidirectional, Dense
from keras.layers import Embedding, Dropout, BatchNormalization
from .constants import MAX_WORDS, MAX_TEXT_LEN, OUTPUT_SIZE


def build_unidirectional_lstm_model(
        vocab_size: int,
        max_length: int,
        layers_sizes: List[int],
        output_size: int,
        dropout: float = 0.5,
        batch_normalization: bool = True,
        activation='softmax',
) -> Model:
    """
    Builds a unidirectional LSTM model with the specified architecture.

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

    model = Sequential()
    model.add(Input(shape=(MAX_TEXT_LEN,)))
    model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=max_length))

    for layer_size in LSTM_SIZES[:-1]:
        model.add(LSTM(layer_size, return_sequences=True))
        model.add(Dropout(dropout))
        if batch_normalization:
            model.add(BatchNormalization())

    model.add(LSTM(LSTM_SIZES[-1]))
    model.add(Dropout(dropout))
    if batch_normalization:
        model.add(BatchNormalization())

    model.add(Dense(output_size, activation=activation))

    return model


def build_bidirectional_lstm_model(
        vocab_size: int,
        max_length: int,
        layers_sizes: List[int],
        output_size: int,
        dropout: float = 0.5,
        batch_normalization: bool = True,
        activation='softmax'
) -> Model:
    """
    Builds a bidirectional LSTM model with the specified architecture.

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

    EMBEDDING_SIZE, BIDIRECTIONAL_SIZES, UNIDIRECTIONAL_SIZE = layers_sizes[0], layers_sizes[1:-1], layers_sizes[-1]

    model = Sequential()
    model.add(Input(shape=(MAX_TEXT_LEN,)))
    model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=max_length))

    for layer_size in BIDIRECTIONAL_SIZES:
        model.add(Bidirectional(LSTM(layer_size, return_sequences=True)))
        model.add(Dropout(dropout))
        if batch_normalization:
            model.add(BatchNormalization())

    model.add(LSTM(UNIDIRECTIONAL_SIZE))
    model.add(Dropout(dropout))
    if batch_normalization:
        model.add(BatchNormalization())

    model.add(Dense(output_size, activation=activation))

    return model


UnidirectionalLSTM = build_unidirectional_lstm_model(MAX_WORDS, MAX_TEXT_LEN, [64, 32, 16], OUTPUT_SIZE)
BidirectionalLSTM = build_bidirectional_lstm_model(MAX_WORDS, MAX_TEXT_LEN, [64, 32, 16], OUTPUT_SIZE)
