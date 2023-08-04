from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import spacy

model_name = "en_core_web_sm"
if model_name not in spacy.util.get_installed_models():
    spacy.cli.download(model_name)
nlp = spacy.load(model_name)


class ExtendedTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def texts_to_padded_sequences(self, texts, maxlen=None):
        return pad_sequences(self.texts_to_sequences(texts), maxlen=maxlen)

    def padded_sequences_to_text(self, sequences):
        cleared_sequences = [list(filter(lambda x: x != 0, row)) for row in sequences]
        return self.sequences_to_texts(cleared_sequences)


def filter_tokens(text: str, join: bool = False) -> str or list:
    """
    Filter out punctuation and stop tokens.
    :param text: input text (sentence)
    :param join: if True, return string, else return list of tokens
    :return: text without punctuation and stop words
    """
    doc = nlp(text)
    filtered = [token.lower_ for token in doc if not token.is_stop and not token.is_punct]
    if join:
        return ' '.join(filtered)
    return filtered


def extract_lemma(text: str, join: bool = False) -> str or list:
    """
    Extract lemma(base) from each word in the text.
    Example: feeling -> feel; pencils -> pencil; exhausted -> exhaust
    :param text: input text (sentence)
    :param join: if True, return string, else return list of tokens
    :return: text with each word replaced to its base word
    """
    doc = nlp(text)
    ret = [token.lemma_ for token in doc]
    if join:
        return ' '.join(ret)
    return ret


def filter_and_extract_lemma(text: str, join: bool = False) -> str or list:
    """
    Filter out punctuation and stop tokens and extract lemma(base) from each word in the text.
    :param text: input text (sentence)
    :param join: if True, return string, else return list of tokens
    :return: text without punctuation and stop words and each word replaced to its base word
    """
    doc = nlp(text)
    filtered = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    if join:
        return ' '.join(filtered)
    return filtered


def one_hot_encode(column: pd.Series) -> np.array:
    """
    Converts a data series of classes into one-hot encoded array
    :param column: the data Series or DataFrame column to encode
    :return: 2D array with encoded data
    """
    return np.array(pd.get_dummies(column))


def one_hot_decode(encoded: np.array, classes: pd.Series) -> np.array:
    """
    Decodes one-hot encoded values according to existing classes
    :param encoded: one-hot encoded numpy array, either from one_hot_encode function or from the model output
    :param classes: classes column of the source dataset
    :return: numpy array with the name of the class corresponding to each one-hot subarray
    """
    classes = sorted(classes.unique().tolist())
    return np.array([classes[np.argmax(one_hot)] for one_hot in encoded])
