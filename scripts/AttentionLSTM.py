import tensorflow as tf
from keras.layers import Dense, Lambda, Multiply, Permute
from keras import backend as K


@tf.function
def attention(input_tensor: tf.Tensor) -> tf.Tensor:
    """Compute attention weights and apply them to the input tensor.

    Args:
        input_tensor: A 3D tensor of shape (batch_size, time_steps, input_dim).

    Returns:
        A 2D tensor of shape (batch_size, input_dim), where each row is a weighted
        sum of the input tensor along the time axis.
    """
    batch_size, time_steps, input_dim = input_tensor.shape

    # Compute attention scores
    attention_scores = Permute((2, 1))(input_tensor)
    attention_scores = Dense(time_steps, activation='sigmoid')(attention_scores)
    attention_scores = Dense(time_steps, activation='softmax')(attention_scores)

    # Apply attention weights
    weighted_input = Multiply()([input_tensor, attention_scores])
    output = Lambda(lambda x: K.sum(x, axis=1))(weighted_input)

    return output
