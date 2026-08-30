import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from config import DEFAULT_MODEL_ID, MODEL_GROUPS, MODEL_ID_TO_GROUP, SPACY_LANGUAGES
from heatmaps import HEATMAP_STRATEGIES, HEATMAP_STRATEGY_LABELS
from model_cache import check_spacy_installed, get_embeddings, get_loaded_model_id
from preprocessing import parse_preprocess_opts, preprocess, tokenize
from similarity import (
    METHODS_META,
    bert_score_approx,
    bigram_overlap,
    bleu_score,
    containment_similarity,
    cosine_embedding,
    cosine_tfidf,
    dice_coefficient,
    hamming_similarity,
    jaccard,
    jaro_winkler,
    jensen_shannon_sim,
    kl_divergence_sim,
    lcs_similarity,
    levenshtein,
    meteor_score,
    normalized_compression_distance,
    overlap_coefficient,
    rouge_l,
    rouge_n,
    soft_cosine,
    trigram_overlap,
    word_mover_distance_approx,
)

app = Flask(__name__)
CORS(app)

def _known_model_ids():
    return {m["id"] for g in MODEL_GROUPS for m in g["models"]}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "loaded_model": get_loaded_model_id()})


@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({
        "groups": MODEL_GROUPS,
        "loaded_model": get_loaded_model_id(),
        "default_model": DEFAULT_MODEL_ID,
    })


@app.route("/spacy-models", methods=["GET"])
def spacy_models():
    result = [
        {**lang, "installed": check_spacy_installed(lang["model"])}
        for lang in SPACY_LANGUAGES
    ]
    return jsonify({"languages": result})


@app.route("/similarity", methods=["POST"])
def similarity():
    data = request.json
    t1_raw = data.get("text1", "").strip()
    t2_raw = data.get("text2", "").strip()
    model_id = data.get("model_id", DEFAULT_MODEL_ID).strip()
    pp_opts = parse_preprocess_opts(data)

    if not t1_raw or not t2_raw:
        return jsonify({"error": "Both texts are required"}), 400
    if model_id not in _known_model_ids():
        return jsonify({"error": f"Unknown model_id: {model_id}"}), 400

    try:
        t1 = preprocess(t1_raw, **pp_opts)
        t2 = preprocess(t2_raw, **pp_opts)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    if not t1.strip() or not t2.strip():
        return jsonify({"error": "After preprocessing, one or both texts are empty. Try relaxing the filters."}), 400

    results = {}

    static_computations = {
        "jaccard": jaccard,
        "dice": dice_coefficient,
        "overlap": overlap_coefficient,
        "tfidf_cosine": cosine_tfidf,
        "levenshtein": levenshtein,
        "jaro_winkler": jaro_winkler,
        "bigram_overlap": bigram_overlap,
        "lcs": lcs_similarity,
        "bleu": bleu_score,
        "rouge_l": rouge_l,
        "rouge_2": lambda a, b: rouge_n(a, b, n=2),
        "meteor": meteor_score,
        "kl_divergence": kl_divergence_sim,
        "jensen_shannon": jensen_shannon_sim,
        "ncd": normalized_compression_distance,
        "trigram": trigram_overlap,
        "containment": containment_similarity,
        "hamming": hamming_similarity,
    }
    for key, fn in static_computations.items():
        try:
            score, h1, h2 = fn(t1, t2)
            results[key] = {"score": score, "highlights_text1": h1, "highlights_text2": h2, **METHODS_META[key]}
        except Exception as e:
            results[key] = {"score": 0.0, "highlights_text1": [], "highlights_text2": [], "error": str(e), **METHODS_META[key]}

    try:
        emb_score, emb_h1, emb_h2, emb_meta = cosine_embedding(t1, t2, model_id)
        results["embedding_cosine"] = {
            "score": emb_score,
            "highlights_text1": emb_h1,
            "highlights_text2": emb_h2,
            "anisotropy_before": emb_meta.get("anisotropy_before"),
            **METHODS_META["embedding_cosine"],
        }
    except Exception as e:
        results["embedding_cosine"] = {
            "score": 0.0, "highlights_text1": [], "highlights_text2": [],
            "anisotropy_before": None,
            "error": str(e), **METHODS_META["embedding_cosine"],
        }

    for key, fn in [("wmd", word_mover_distance_approx),
                    ("soft_cosine", soft_cosine),
                    ("bert_score", bert_score_approx)]:
        try:
            score, h1, h2 = fn(t1, t2, model_id)
            results[key] = {"score": score, "highlights_text1": h1, "highlights_text2": h2, **METHODS_META[key]}
        except Exception as e:
            results[key] = {"score": 0.0, "highlights_text1": [], "highlights_text2": [], "error": str(e), **METHODS_META[key]}

    return jsonify({
        "results": results,
        "text1": t1,
        "text2": t2,
        "text1_raw": t1_raw,
        "text2_raw": t2_raw,
        "preprocessed": t1 != t1_raw or t2 != t2_raw,
        "model_id": model_id,
        "model_group": MODEL_ID_TO_GROUP.get(model_id, "Unknown"),
        "loaded_model": get_loaded_model_id(),
        "preprocess_opts": pp_opts,
    })


@app.route("/heatmap", methods=["POST"])
def heatmap():
    data = request.json
    t1_raw = data.get("text1", "").strip()
    t2_raw = data.get("text2", "").strip()
    model_id = data.get("model_id", DEFAULT_MODEL_ID).strip()
    method_key = data.get("method_key", "embedding_cosine").strip()
    pp_opts = parse_preprocess_opts(data)

    if not t1_raw or not t2_raw:
        return jsonify({"error": "Both texts are required"}), 400
    if model_id not in _known_model_ids():
        return jsonify({"error": f"Unknown model_id: {model_id}"}), 400
    if method_key not in HEATMAP_STRATEGIES:
        return jsonify({"error": f"Unknown method_key: {method_key}"}), 400

    try:
        t1 = preprocess(t1_raw, **pp_opts)
        t2 = preprocess(t2_raw, **pp_opts)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    if not t1.strip() or not t2.strip():
        return jsonify({"error": "After preprocessing, one or both texts are empty."}), 400

    MAX_TOKENS = 40

    def ordered_unique(tokens):
        seen, out = set(), []
        for t in tokens:
            if t not in seen:
                seen.add(t); out.append(t)
        return out

    tok1 = ordered_unique(tokenize(t1))[:MAX_TOKENS]
    tok2 = ordered_unique(tokenize(t2))[:MAX_TOKENS]

    if not tok1 or not tok2:
        return jsonify({"error": "Could not tokenize texts"}), 400

    matrix = HEATMAP_STRATEGIES[method_key](tok1, tok2, model_id)
    label, desc = HEATMAP_STRATEGY_LABELS.get(method_key, ("Similarity", "Token-level similarity"))

    return jsonify({
        "tokens1": tok1,
        "tokens2": tok2,
        "matrix": matrix,
        "model_id": model_id,
        "method_key": method_key,
        "strategy_label": label,
        "strategy_description": desc,
    })


@app.route("/embeddings", methods=["POST"])
def embeddings():
    data = request.json
    t1_raw = data.get("text1", "").strip()
    t2_raw = data.get("text2", "").strip()
    model_id = data.get("model_id", DEFAULT_MODEL_ID).strip()
    method = data.get("method", "pca").lower()
    dims = int(data.get("dims", 2))
    pp_opts = parse_preprocess_opts(data)

    if not t1_raw or not t2_raw:
        return jsonify({"error": "Both texts are required"}), 400
    if method not in ("pca", "tsne", "umap"):
        return jsonify({"error": "method must be pca, tsne or umap"}), 400
    if dims not in (2, 3):
        return jsonify({"error": "dims must be 2 or 3"}), 400
    if model_id not in _known_model_ids():
        return jsonify({"error": f"Unknown model_id: {model_id}"}), 400

    try:
        t1 = preprocess(t1_raw, **pp_opts)
        t2 = preprocess(t2_raw, **pp_opts)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    if not t1.strip() or not t2.strip():
        return jsonify({"error": "After preprocessing, one or both texts are empty."}), 400

    MAX_TOKENS = 60
    tok1 = tokenize(t1)[:MAX_TOKENS]
    tok2 = tokenize(t2)[:MAX_TOKENS]
    if not tok1 or not tok2:
        return jsonify({"error": "Could not tokenize texts"}), 400

    all_tokens = tok1 + tok2
    all_embs = get_embeddings(all_tokens, model_id)
    n1 = len(tok1)
    projection_quality = {}

    if method == "pca":
        from numpy.linalg import svd
        X = all_embs.astype(np.float64)
        X -= X.mean(axis=0)
        _, s, Vt = svd(X, full_matrices=False)
        coords = X @ Vt[:dims].T
        total_var = float(np.sum(s**2))
        explained_variance = float(np.sum(s[:dims]**2) / total_var) if total_var > 0 else None
        projection_quality["explained_variance"] = explained_variance
        projection_quality["metric_label"] = "Explained variance"
        projection_quality["metric_description"] = (
            f"Fraction of total variance retained after projecting to {dims}D. Higher = less geometric distortion."
        )

    elif method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            return jsonify({"error": "scikit-learn is required for t-SNE. Run: pip install scikit-learn"}), 500
        import sklearn
        perplexity = min(30, max(5, len(all_tokens) // 3))
        tsne_kwargs = dict(n_components=dims, perplexity=perplexity, random_state=42,
                           learning_rate="auto", init="pca")
        sk_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        tsne_kwargs["max_iter" if sk_version >= (1, 5) else "n_iter"] = 500
        tsne = TSNE(**tsne_kwargs)
        coords = tsne.fit_transform(all_embs.astype(np.float64))
        kl = float(tsne.kl_divergence_)
        projection_quality["kl_divergence"] = kl
        projection_quality["metric_label"] = "KL divergence"
        projection_quality["metric_description"] = (
            "Optimisation loss between high-D and low-D neighbourhood distributions. "
            "Lower = neighbourhoods better preserved. Typical range 0.1-2.0."
        )

    elif method == "umap":
        try:
            import umap as umap_lib
        except ImportError:
            return jsonify({"error": "umap-learn is required for UMAP. Run: pip install umap-learn"}), 500
        n_neighbors = min(15, max(2, len(all_tokens) // 2 - 1))
        reducer = umap_lib.UMAP(n_components=dims, n_neighbors=n_neighbors,
                                min_dist=0.1, random_state=42)
        coords = reducer.fit_transform(all_embs.astype(np.float64))
        try:
            from sklearn.manifold import trustworthiness
            trust = float(trustworthiness(all_embs.astype(np.float64), coords, n_neighbors=n_neighbors))
        except Exception:
            trust = None
        projection_quality["trustworthiness"] = trust
        projection_quality["metric_label"] = "Trustworthiness"
        projection_quality["metric_description"] = (
            f"Fraction of each point's true nearest neighbours (in high-D) still nearest in the projection. "
            f"n_neighbors={n_neighbors}. Range 0→1; higher = less local distortion."
        )

    points = []
    for i, (tok, coord) in enumerate(zip(all_tokens, coords)):
        pt = {"token": tok, "source": 1 if i < n1 else 2,
              "x": round(float(coord[0]), 5), "y": round(float(coord[1]), 5)}
        if dims == 3:
            pt["z"] = round(float(coord[2]), 5)
        points.append(pt)

    return jsonify({
        "points": points,
        "method": method,
        "dims": dims,
        "model_id": model_id,
        "explained_variance": projection_quality.get("explained_variance"),
        "projection_quality": projection_quality,
        "n_tokens_text1": n1,
        "n_tokens_text2": len(tok2),
    })


@app.route("/corpus", methods=["POST"])
def corpus():
    data = request.json
    query = data.get("query", "").strip()
    docs_raw = data.get("documents", [])
    model_id = data.get("model_id", DEFAULT_MODEL_ID).strip()
    pp_opts = parse_preprocess_opts(data)

    if not query:
        return jsonify({"error": "query is required"}), 400
    if not docs_raw or len(docs_raw) < 1:
        return jsonify({"error": "at least one document is required"}), 400
    if len(docs_raw) > 50:
        return jsonify({"error": "maximum 50 documents"}), 400
    if model_id not in _known_model_ids():
        return jsonify({"error": f"Unknown model_id: {model_id}"}), 400

    try:
        q = preprocess(query, **pp_opts)
        docs = [preprocess(d, **pp_opts) for d in docs_raw]
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    if not q.strip():
        return jsonify({"error": "Query is empty after preprocessing."}), 400
    docs = [d if d.strip() else "(empty after preprocessing)" for d in docs]

    static_computations = {
        "jaccard": jaccard,
        "dice": dice_coefficient,
        "overlap": overlap_coefficient,
        "tfidf_cosine": cosine_tfidf,
        "levenshtein": levenshtein,
        "jaro_winkler": jaro_winkler,
        "bigram_overlap": bigram_overlap,
        "lcs": lcs_similarity,
        "bleu": bleu_score,
        "rouge_l": rouge_l,
        "rouge_2": lambda a, b, _n=2: rouge_n(a, b, n=_n),
        "meteor": meteor_score,
        "kl_divergence": kl_divergence_sim,
        "jensen_shannon": jensen_shannon_sim,
        "ncd": normalized_compression_distance,
        "trigram": trigram_overlap,
        "containment": containment_similarity,
        "hamming": hamming_similarity,
    }
    embedding_computations = {
        "embedding_cosine": cosine_embedding,
        "wmd": word_mover_distance_approx,
        "soft_cosine": soft_cosine,
        "bert_score": bert_score_approx,
    }

    scores = {k: [] for k in list(static_computations) + list(embedding_computations)}

    for key, fn in static_computations.items():
        for doc in docs:
            try:
                s, _, _ = fn(q, doc)
            except Exception:
                s = 0.0
            scores[key].append(round(float(s), 4))

    all_texts = [q] + docs

    try:
        all_embs = get_embeddings(all_texts, model_id)
        q_emb = all_embs[0:1]
        d_embs = all_embs[1:]
        q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8)
        d_norm = d_embs / (np.linalg.norm(d_embs, axis=1, keepdims=True) + 1e-8)
        cosine_scores = (q_norm @ d_norm.T)[0].tolist()
        scores["embedding_cosine"] = [round(float(s), 4) for s in cosine_scores]
    except Exception:
        scores["embedding_cosine"] = [0.0] * len(docs)

    for key, fn in [("wmd", word_mover_distance_approx),
                    ("soft_cosine", soft_cosine),
                    ("bert_score", bert_score_approx)]:
        for doc in docs:
            try:
                s, _, _ = fn(q, doc, model_id)
            except Exception:
                s = 0.0
            scores[key].append(round(float(s), 4))

    doc_previews = [d[:120] + ("…" if len(d) > 120 else "") for d in docs]
    rankings = {}
    for key, sc_list in scores.items():
        ranked = sorted(
            [{"doc_index": i, "score": sc_list[i], "doc_preview": doc_previews[i]}
             for i in range(len(docs))],
            key=lambda x: x["score"], reverse=True
        )
        for pos, entry in enumerate(ranked):
            entry["rank"] = pos + 1
        rankings[key] = ranked

    return jsonify({
        "rankings": rankings,
        "query_preprocessed": q,
        "docs_preprocessed": docs,
        "doc_previews": doc_previews,
        "model_id": model_id,
        "n_docs": len(docs),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
