from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from .constants import *
import numpy as np


def get_reverse_word_map(tokenizer: Tokenizer):
    return dict(map(reversed, tokenizer.word_index.items()))


def sequence_to_text(reverse_word_map: dict, list_of_indices):
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return words


def output_interpretation(output: np.array):
    return EMOTIONS[np.argmax(output)]


def test_on_examples(model: Model, tokenizer: Tokenizer):
    reverse_word_map = get_reverse_word_map(tokenizer)
    correct = 0

    for example, emotion in TEST_EXAMPLES.items():
        tokens = tokenizer.texts_to_sequences([example])[0]
        restored_sentence = sequence_to_text(reverse_word_map, tokens)
        pad_tokens = pad_sequences([tokens], maxlen=MAX_TEXT_LEN)
        prediction = output_interpretation(model.predict(pad_tokens))

        if prediction == emotion:
            correct += 1

        print('Input Sentence:')
        print(example)
        print('Restored Sentence:')
        print(restored_sentence)
        print(f'Expected output: {emotion}')
        print(f'Model output: {prediction}')

    return correct
