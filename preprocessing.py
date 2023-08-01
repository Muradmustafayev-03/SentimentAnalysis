import spacy

# run 'python -m spacy download en_core_web_sm' in the terminal before running this line
nlp = spacy.load("en_core_web_sm")


def filter_tokens(text: str) -> str:
    """
    Filters out punctuation and stop tokens.
    :param text: input text (sentence)
    :return: text without punctuation and stop words
    """
    doc = nlp(text)
    filtered = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(filtered)


def extract_lemma(text: str) -> str:
    """
    Extracts lemma(base) from each word in the text
    Example: feeling -> feel; pencils -> pencil; exhausted -> exhaust
    :param text: input text (sentence)
    :return: text with each word replaced to its base word
    """
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])
