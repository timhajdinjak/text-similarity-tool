import gc

import numpy as np

_loaded_model_id = None
_model = None


def get_loaded_model_id():
    return _loaded_model_id


def get_model(model_id):
    global _model, _loaded_model_id
    if _loaded_model_id == model_id and _model is not None:
        return _model

    if _model is not None:
        print(f"[model] evicting '{_loaded_model_id}' from RAM")
        del _model
        _model = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    print(f"[model] loading '{model_id}'")
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(model_id)
    _loaded_model_id = model_id
    return _model


def get_embeddings(texts, model_id):
    model = get_model(model_id)
    return model.encode(texts, convert_to_numpy=True)


def compute_anisotropy(embs):
    if len(embs) < 2:
        return 0.0
    normed = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    sim_matrix = normed @ normed.T
    n = len(embs)
    total = float(sim_matrix.sum()) - n
    pairs = n * (n - 1)
    return round(total / pairs, 4) if pairs > 0 else 0.0


_spacy_model_name = None
_spacy_nlp = None


def get_spacy_nlp(model_name):
    global _spacy_nlp, _spacy_model_name
    if _spacy_model_name == model_name and _spacy_nlp is not None:
        return _spacy_nlp
    if _spacy_nlp is not None:
        print(f"[spacy] evicting '{_spacy_model_name}'")
        del _spacy_nlp
        _spacy_nlp = None
        gc.collect()
    try:
        import spacy
        print(f"[spacy] loading '{model_name}'")
        _spacy_nlp = spacy.load(model_name, disable=["parser", "ner", "lemmatizer"])
        _spacy_model_name = model_name
        return _spacy_nlp
    except OSError:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Run: python -m spacy download {model_name}"
        )


def check_spacy_installed(model_name):
    try:
        import spacy
        return spacy.util.is_package(model_name)
    except ImportError:
        return False
