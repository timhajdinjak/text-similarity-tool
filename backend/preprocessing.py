import re

from model_cache import get_spacy_nlp


def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def preprocess(text, *, lowercase=True, remove_punct=True, remove_stopwords=False, spacy_model="en_core_web_sm"):
    if lowercase:
        text = text.lower()

    if remove_punct:
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"_", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

    if remove_stopwords:
        nlp = get_spacy_nlp(spacy_model)
        doc = nlp(text)
        tokens = [t.text for t in doc if not t.is_stop and t.text.strip()]
        text = " ".join(tokens)

    return text


def parse_preprocess_opts(data):
    return {
        "lowercase": bool(data.get("lowercase", True)),
        "remove_punct": bool(data.get("remove_punct", True)),
        "remove_stopwords": bool(data.get("remove_stopwords", False)),
        "spacy_model": data.get("spacy_model", "en_core_web_sm"),
    }
