from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import spacy

model_name = "en_core_web_sm"
if model_name not in spacy.util.get_installed_models():
    spacy.cli.download(model_name)
nlp = spacy.load(model_name)


class ExtendedTokenizer(Tokenizer):
    """
    Extended tokenizer with additional methods for converting texts to padded sequences and back.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def texts_to_padded_sequences(self, texts, maxlen=None):
        """
        Convert texts to padded sequences.
        :param texts: texts to vectorize
        :param maxlen: maximum length of the sequences
        :return: padded sequences
        """
        return pad_sequences(self.texts_to_sequences(texts), maxlen=maxlen)

    def padded_sequences_to_text(self, sequences):
        """
        Convert padded sequences to texts.
        :param sequences: sequences to convert
        :return: texts
        """
        cleared_sequences = [list(filter(lambda x: x != 0, row)) for row in sequences]
        return self.sequences_to_texts(cleared_sequences)


class Word2VecVectorizer(Word2Vec):
    """
    Extended Word2Vec model with additional methods.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def vectorize_sentence(self, sentence: str) -> np.ndarray:
        vectorized = [self.wv[word] for word in sentence.split(' ') if word in self.wv]
        if not vectorized:
            # If none of the words are in the vocabulary, return zeros
            vectorized = [np.zeros(self.vector_size)]
        return np.mean(vectorized, axis=0)

    def vectorize_sentences(self, sentences: pd.Series) -> np.ndarray:
        return np.array([self.vectorize_sentence(sentence) for sentence in sentences.values])


def filter_tokens(text: str, join: bool = True) -> str or list:
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


def extract_lemma(text: str, join: bool = True) -> str or list:
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


def filter_and_extract_lemma(text: str, join: bool = True) -> str or list:
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


def enumerate_encode(column: pd.Series) -> np.array:
    """
    Converts a data series of classes into enumerated array
    :param column: the data Series or DataFrame column to encode
    :return: 2D array with encoded data
    """
    classes = sorted(column.unique().tolist())
    return np.array([classes.index(value) for value in column])


def enumerate_decode(encoded: np.array, classes: pd.Series) -> np.array:
    """
    Decodes enumerated values according to existing classes
    :param encoded: enumerated numpy array, either from enumerate_encode function or from the model output
    :param classes: classes column of the source dataset
    :return: numpy array with the name of the class corresponding to each enumerated value
    """
    classes = sorted(classes.unique().tolist())
    return np.array([classes[value] for value in encoded])


def one_hot_to_enumerate(encoded: np.array) -> np.array:
    """
    Converts one-hot encoded array into enumerated array
    :param encoded: one-hot encoded numpy array, either from one_hot_encode function or from the model output
    :return: enumerated numpy array
    """
    return np.array([np.argmax(one_hot) for one_hot in encoded])


def enumerate_to_one_hot(encoded: np.array, classes: pd.Series) -> np.array:
    """
    Converts enumerated array into one-hot encoded array
    :param encoded: encoded 1-D numpy array, either from enumerate_encode function or from the model output
    :param classes: classes column of the source dataset
    :return: one-hot encoded numpy array
    """
    N = len(classes.unique().tolist())
    ret = np.zeros((len(encoded), N))
    for i, value in enumerate(encoded):
        ret[i, value] = 1
    return ret

