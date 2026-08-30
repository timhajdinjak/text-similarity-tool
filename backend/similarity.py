import math
import zlib
from collections import Counter

import numpy as np

from config import DEFAULT_MODEL_ID
from model_cache import get_embeddings, compute_anisotropy
from preprocessing import tokenize, ngrams


def jaccard(t1, t2):
    s1, s2 = set(tokenize(t1)), set(tokenize(t2))
    if not s1 and not s2:
        return 0.0, [], []
    inter = s1 & s2
    union = s1 | s2
    score = len(inter) / len(union) if union else 0.0
    h1 = [w for w in tokenize(t1) if w in inter]
    h2 = [w for w in tokenize(t2) if w in inter]
    return round(score, 4), list(set(h1)), list(set(h2))


def dice_coefficient(t1, t2):
    s1, s2 = set(tokenize(t1)), set(tokenize(t2))
    if not s1 and not s2:
        return 0.0, [], []
    inter = s1 & s2
    score = 2 * len(inter) / (len(s1) + len(s2)) if (s1 or s2) else 0.0
    return round(score, 4), list(inter), list(inter)


def overlap_coefficient(t1, t2):
    s1, s2 = set(tokenize(t1)), set(tokenize(t2))
    if not s1 or not s2:
        return 0.0, [], []
    inter = s1 & s2
    score = len(inter) / min(len(s1), len(s2))
    return round(score, 4), list(inter), list(inter)


def containment_similarity(t1, t2):
    s1 = set(tokenize(t1))
    s2 = set(tokenize(t2))
    if not s1:
        return 0.0, [], []
    inter = s1 & s2
    score = len(inter) / len(s1)
    return round(score, 4), list(inter), list(inter)


def cosine_tfidf(t1, t2):
    def tfidf_vec(texts):
        tokenized = [tokenize(t) for t in texts]
        vocab = set(w for doc in tokenized for w in doc)
        N = len(texts)
        idf = {w: math.log((N + 1) / (sum(1 for doc in tokenized if w in doc) + 1)) + 1 for w in vocab}
        vecs = []
        for doc in tokenized:
            tf = Counter(doc)
            total = len(doc) or 1
            vecs.append({w: (tf[w] / total) * idf[w] for w in vocab})
        return vecs

    vecs = tfidf_vec([t1, t2])
    v1, v2 = vecs[0], vecs[1]
    dot = sum(v1[w] * v2[w] for w in v1)
    n1 = math.sqrt(sum(x**2 for x in v1.values()))
    n2 = math.sqrt(sum(x**2 for x in v2.values()))
    score = dot / (n1 * n2) if n1 and n2 else 0.0
    contributions = {w: v1[w] * v2[w] for w in v1 if v2[w] > 0}
    top = sorted(contributions, key=contributions.get, reverse=True)[:5]
    return round(score, 4), top, top


def levenshtein(t1, t2):
    if not t1 and not t2:
        return 1.0, [], []
    a, b = t1.lower(), t2.lower()
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    dist = dp[m][n]
    max_len = max(m, n) or 1
    score = 1.0 - dist / max_len
    common_chars = set(a) & set(b)
    return round(score, 4), list(common_chars), list(common_chars)


def jaro_winkler(t1, t2):
    def jaro(s1, s2):
        if s1 == s2: return 1.0
        l1, l2 = len(s1), len(s2)
        if not l1 or not l2: return 0.0
        match_dist = max(l1, l2) // 2 - 1
        s1_matches = [False] * l1
        s2_matches = [False] * l2
        matches = 0
        for i in range(l1):
            start = max(0, i - match_dist)
            end = min(i + match_dist + 1, l2)
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]: continue
                s1_matches[i] = s2_matches[j] = True
                matches += 1
                break
        if not matches: return 0.0
        transpositions = 0
        k = 0
        for i in range(l1):
            if not s1_matches[i]: continue
            while not s2_matches[k]: k += 1
            if s1[i] != s2[k]: transpositions += 1
            k += 1
        return (matches/l1 + matches/l2 + (matches - transpositions/2)/matches) / 3

    s1, s2 = t1.lower(), t2.lower()
    j = jaro(s1, s2)
    prefix = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]: prefix += 1
        else: break
    score = j + prefix * 0.1 * (1 - j)
    common = [c for c in set(s1) if c in set(s2) and c.strip()]
    return round(score, 4), common[:5], common[:5]


def trigram_overlap(t1, t2):
    def char_trigrams(text):
        t = text.lower().replace(" ", "_")
        return set(t[i:i+3] for i in range(len(t)-2))
    s1 = char_trigrams(t1)
    s2 = char_trigrams(t2)
    if not s1 and not s2:
        return 0.0, [], []
    inter = s1 & s2
    score = len(inter) / len(s1 | s2)
    h1 = [w for w in tokenize(t1) if any(tg in w for tg in inter)]
    h2 = [w for w in tokenize(t2) if any(tg in w for tg in inter)]
    return round(score, 4), list(set(h1))[:6], list(set(h2))[:6]


def hamming_similarity(t1, t2):
    tok1, tok2 = tokenize(t1), tokenize(t2)
    max_len = max(len(tok1), len(tok2), 1)
    tok1 += [""] * (max_len - len(tok1))
    tok2 += [""] * (max_len - len(tok2))
    mismatches = sum(1 for a, b in zip(tok1, tok2) if a != b)
    score = 1.0 - mismatches / max_len
    matches = [tok1[i] for i in range(max_len) if tok1[i] == tok2[i] and tok1[i]]
    return round(score, 4), list(set(matches)), list(set(matches))


def bigram_overlap(t1, t2):
    tokens1 = tokenize(t1)
    tokens2 = tokenize(t2)
    bg1 = set(ngrams(tokens1, 2))
    bg2 = set(ngrams(tokens2, 2))
    if not bg1 or not bg2:
        return 0.0, [], []
    inter = bg1 & bg2
    score = len(inter) / max(len(bg1), len(bg2))
    shared_words = list(set(w for bg in inter for w in bg))
    return round(score, 4), shared_words, shared_words


def lcs_similarity(t1, t2):
    tokens1 = tokenize(t1)
    tokens2 = tokenize(t2)
    m, n = len(tokens1), len(tokens2)
    if not m or not n:
        return 0.0, [], []
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i-1] == tokens2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[m][n]
    score = 2 * lcs_len / (m + n)
    lcs_words = []
    i, j = m, n
    while i > 0 and j > 0:
        if tokens1[i-1] == tokens2[j-1]:
            lcs_words.append(tokens1[i-1]); i -= 1; j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    return round(score, 4), list(set(lcs_words)), list(set(lcs_words))


def bleu_score(t1, t2):
    ref = tokenize(t1)
    hyp = tokenize(t2)
    if not ref or not hyp:
        return 0.0, [], []
    bp = min(1.0, math.exp(1 - len(ref) / len(hyp))) if len(hyp) < len(ref) else 1.0
    precisions = []
    highlight_words = set()
    for n in range(1, 5):
        ref_ng = Counter(ngrams(ref, n))
        hyp_ng = Counter(ngrams(hyp, n))
        clipped = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
        total = max(sum(hyp_ng.values()), 1)
        p = clipped / total
        precisions.append(p)
        if n == 1:
            for g, c in hyp_ng.items():
                if ref_ng[g] > 0:
                    highlight_words.update(g)
    if any(p == 0 for p in precisions):
        score = 0.0
    else:
        score = bp * math.exp(sum(math.log(p) for p in precisions) / 4)
    shared = list(highlight_words)
    return round(score, 4), shared, shared


def rouge_l(t1, t2):
    r = tokenize(t1)
    h = tokenize(t2)
    if not r or not h:
        return 0.0, [], []
    m, n = len(r), len(h)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1]+1 if r[i-1]==h[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[m][n]
    prec = lcs_len / n
    rec = lcs_len / m
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    lcs_words, i, j = [], m, n
    while i > 0 and j > 0:
        if r[i-1] == h[j-1]:
            lcs_words.append(r[i-1]); i -= 1; j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    shared = list(set(lcs_words))
    return round(f1, 4), shared, shared


def rouge_n(t1, t2, n=2):
    ref = tokenize(t1)
    hyp = tokenize(t2)
    if not ref or not hyp:
        return 0.0, [], []
    ref_ng = Counter(ngrams(ref, n))
    hyp_ng = Counter(ngrams(hyp, n))
    overlap = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
    prec = overlap / max(sum(hyp_ng.values()), 1)
    rec = overlap / max(sum(ref_ng.values()), 1)
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    shared_words = list(set(w for g, c in hyp_ng.items() if ref_ng[g] > 0 for w in g))
    return round(f1, 4), shared_words, shared_words


def meteor_score(t1, t2):
    ref = tokenize(t1)
    hyp = tokenize(t2)
    if not ref or not hyp:
        return 0.0, [], []
    ref_c = Counter(ref)
    hyp_c = Counter(hyp)
    matches = sum(min(hyp_c[w], ref_c[w]) for w in hyp_c)
    if matches == 0:
        return 0.0, [], []
    prec = matches / len(hyp)
    rec = matches / len(ref)
    f = 10*prec*rec / (9*prec + rec)
    matched_hyp = {w for w in hyp_c if ref_c[w] > 0}
    chunks = 1
    prev_match = False
    for w in hyp:
        cur = w in matched_hyp
        if cur and not prev_match: chunks += 1
        prev_match = cur
    penalty = 0.5 * (chunks / matches) ** 3
    score = f * (1 - penalty)
    return round(max(0, score), 4), list(matched_hyp), list(matched_hyp)


def _term_distribution(tokens, vocab, alpha=0.01):
    c = Counter(tokens)
    total = sum(c.values()) + alpha * len(vocab)
    return {w: (c.get(w, 0) + alpha) / total for w in vocab}


def kl_divergence_sim(t1, t2):
    tok1, tok2 = tokenize(t1), tokenize(t2)
    if not tok1 or not tok2:
        return 0.0, [], []
    vocab = set(tok1) | set(tok2)
    p = _term_distribution(tok1, vocab)
    q = _term_distribution(tok2, vocab)
    kl_pq = sum(p[w] * math.log(p[w]/q[w]) for w in vocab)
    kl_qp = sum(q[w] * math.log(q[w]/p[w]) for w in vocab)
    sym_kl = (kl_pq + kl_qp) / 2
    score = 1.0 / (1.0 + sym_kl)
    contrib = {w: abs(p[w] - q[w]) for w in vocab}
    low_div = sorted(contrib, key=contrib.get)[:6]
    return round(score, 4), low_div, low_div


def jensen_shannon_sim(t1, t2):
    tok1, tok2 = tokenize(t1), tokenize(t2)
    if not tok1 or not tok2:
        return 0.0, [], []
    vocab = set(tok1) | set(tok2)
    p = _term_distribution(tok1, vocab)
    q = _term_distribution(tok2, vocab)
    m = {w: (p[w]+q[w])/2 for w in vocab}
    jsd = (sum(p[w]*math.log(p[w]/m[w]) for w in vocab) +
           sum(q[w]*math.log(q[w]/m[w]) for w in vocab)) / 2
    score = 1.0 - min(jsd, 1.0)
    shared = list(set(tok1) & set(tok2))
    return round(score, 4), shared, shared


def normalized_compression_distance(t1, t2):
    if not t1 or not t2:
        return 0.0, [], []
    b1 = t1.encode()
    b2 = t2.encode()
    b12 = b1 + b2
    cx = len(zlib.compress(b1, level=9))
    cy = len(zlib.compress(b2, level=9))
    cxy = len(zlib.compress(b12, level=9))
    ncd = (cxy - min(cx, cy)) / max(max(cx, cy), 1)
    score = max(0.0, 1.0 - ncd)
    shared = list(set(tokenize(t1)) & set(tokenize(t2)))
    return round(score, 4), shared, shared


def cosine_embedding(t1, t2, model_id=DEFAULT_MODEL_ID):
    tokens1 = tokenize(t1)
    tokens2 = tokenize(t2)
    all_tokens = list(dict.fromkeys(tokens1 + tokens2))

    if not all_tokens:
        return 0.0, [], [], {"anisotropy_before": None}

    raw_embs = get_embeddings(all_tokens, model_id)
    anisotropy_before = compute_anisotropy(raw_embs)

    tok_embs = raw_embs / (np.linalg.norm(raw_embs, axis=1, keepdims=True) + 1e-8)
    tok_map = {tok: tok_embs[i] for i, tok in enumerate(all_tokens)}

    def mean_emb(toks):
        vecs = [tok_map[t] for t in toks if t in tok_map]
        if not vecs:
            return np.zeros(tok_embs.shape[1])
        m = np.mean(vecs, axis=0)
        n = np.linalg.norm(m)
        return m / n if n > 1e-10 else m

    e1 = mean_emb(tokens1)
    e2 = mean_emb(tokens2)
    score = float(np.dot(e1, e2))
    score = max(-1.0, min(1.0, score))

    highlights1, highlights2 = [], []
    if tokens1 and tokens2:
        threshold = 0.6
        e1_mat = np.array([tok_map[t] for t in tokens1 if t in tok_map])
        e2_mat = np.array([tok_map[t] for t in tokens2 if t in tok_map])
        if e1_mat.size and e2_mat.size:
            scores1 = (e1_mat @ e2_mat.T).max(axis=1)
            scores2 = (e2_mat @ e1_mat.T).max(axis=1)
            highlights1 = [tokens1[i] for i, s in enumerate(scores1) if s > threshold]
            highlights2 = [tokens2[i] for i, s in enumerate(scores2) if s > threshold]

    meta = {"anisotropy_before": anisotropy_before}
    return round(score, 4), list(set(highlights1)), list(set(highlights2)), meta


def word_mover_distance_approx(t1, t2, model_id=DEFAULT_MODEL_ID):
    tokens1 = list(set(tokenize(t1)))
    tokens2 = list(set(tokenize(t2)))
    if not tokens1 or not tokens2:
        return 0.0, [], []
    all_tokens = tokens1 + tokens2
    embs = get_embeddings(all_tokens, model_id)
    embs1 = embs[:len(tokens1)]
    embs2 = embs[len(tokens1):]
    total_dist = 0.0
    matched2 = []
    for e1 in embs1:
        sims = [float(np.dot(e1, e2)/(np.linalg.norm(e1)*np.linalg.norm(e2)+1e-8)) for e2 in embs2]
        best_idx = int(np.argmax(sims))
        total_dist += sims[best_idx]
        matched2.append(tokens2[best_idx])
    score = total_dist / len(embs1)
    return round(max(0, score), 4), tokens1[:5], list(set(matched2))[:5]


def soft_cosine(t1, t2, model_id=DEFAULT_MODEL_ID):
    tok1, tok2 = tokenize(t1), tokenize(t2)
    if not tok1 or not tok2:
        return 0.0, [], []
    vocab = list(set(tok1) | set(tok2))
    embs = get_embeddings(vocab, model_id)
    emb_map = {w: embs[i] for i, w in enumerate(vocab)}
    v1 = np.array([tok1.count(w) for w in vocab], dtype=float)
    v2 = np.array([tok2.count(w) for w in vocab], dtype=float)
    E = np.stack([emb_map[w] for w in vocab])
    S = (E @ E.T)
    np.fill_diagonal(S, 1.0)
    S = np.clip(S, 0, 1)
    num = float(v1 @ S @ v2)
    den = math.sqrt(max(float(v1 @ S @ v1), 1e-10)) * math.sqrt(max(float(v2 @ S @ v2), 1e-10))
    score = num / den if den else 0.0
    high = [w1 for i, w1 in enumerate(vocab)
            for j, w2 in enumerate(vocab)
            if i != j and S[i, j] > 0.75 and v1[i] > 0 and v2[j] > 0]
    return round(min(score, 1.0), 4), list(set(high)), list(set(high))


def bert_score_approx(t1, t2, model_id=DEFAULT_MODEL_ID):
    tok1, tok2 = tokenize(t1), tokenize(t2)
    if not tok1 or not tok2:
        return 0.0, [], []
    embs1 = get_embeddings(tok1, model_id)
    embs2 = get_embeddings(tok2, model_id)

    def max_cos(a, B):
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        return float(np.max(B_norm @ a_norm))

    P = np.mean([max_cos(e, embs1) for e in embs2])
    R = np.mean([max_cos(e, embs2) for e in embs1])
    F1 = 2*P*R/(P+R) if (P+R) else 0.0
    threshold = 0.75
    h1 = [tok1[i] for i, e in enumerate(embs1) if max_cos(e, embs2) > threshold]
    h2 = [tok2[i] for i, e in enumerate(embs2) if max_cos(e, embs1) > threshold]
    return round(max(0.0, F1), 4), list(set(h1)), list(set(h2))


METHODS_META = {
    "jaccard": {
        "name": "Jaccard Similarity",
        "description": "Measures overlap between two sets of words. Divides the size of the intersection by the size of the union of word sets.",
        "formula": "J(A,B) = |A ∩ B| / |A ∪ B|",
        "type": "Token / Set-Based"
    },
    "dice": {
        "name": "Sørensen-Dice Coefficient",
        "description": "Similar to Jaccard but gives more weight to the intersection. Double the shared words divided by total words in both texts.",
        "formula": "Dice(A,B) = 2|A ∩ B| / (|A| + |B|)",
        "type": "Token / Set-Based"
    },
    "overlap": {
        "name": "Overlap Coefficient",
        "description": "Measures how much the smaller set is contained within the larger. Good for detecting containment relationships.",
        "formula": "Overlap(A,B) = |A ∩ B| / min(|A|, |B|)",
        "type": "Token / Set-Based"
    },
    "tfidf_cosine": {
        "name": "TF-IDF Cosine",
        "description": "Weighs words by how unique they are (IDF) and how often they appear (TF), then measures the angle between the resulting vectors.",
        "formula": "cos(θ) = (v₁ · v₂) / (‖v₁‖ × ‖v₂‖), where v = TF-IDF vector",
        "type": "Vector / TF-IDF"
    },
    "levenshtein": {
        "name": "Levenshtein (Edit Distance)",
        "description": "Counts the minimum number of single-character edits to transform one text into another.",
        "formula": "score = 1 − edit_distance(s₁, s₂) / max(|s₁|, |s₂|)",
        "type": "String / Character-Based"
    },
    "jaro_winkler": {
        "name": "Jaro-Winkler",
        "description": "Character-level similarity that rewards matching characters and common prefixes. Originally designed for name matching.",
        "formula": "JW = Jaro + p × prefix_len × (1 − Jaro), where p = 0.1",
        "type": "String / Character-Based"
    },
    "embedding_cosine": {
        "name": "Embedding Cosine (SBERT)",
        "description": "Uses a pretrained Sentence-BERT model to encode both texts into dense vector embeddings, then measures their cosine similarity.",
        "formula": "cos(θ) = (e₁ · e₂) / (‖e₁‖ × ‖e₂‖), where e = sentence embedding",
        "type": "Embedding / Semantic"
    },
    "bigram_overlap": {
        "name": "Bigram Overlap",
        "description": "Extends Jaccard to pairs of consecutive words (bigrams). Captures local word order and phrase-level similarity.",
        "formula": "Bigram(A,B) = |bigrams(A) ∩ bigrams(B)| / max(|bigrams(A)|, |bigrams(B)|)",
        "type": "Token / N-Gram"
    },
    "lcs": {
        "name": "Longest Common Subsequence",
        "description": "Finds the longest sequence of words that appear in the same order in both texts. Measures structural similarity.",
        "formula": "score = 2 × LCS_length / (|tokens₁| + |tokens₂|)",
        "type": "Sequence-Based"
    },
    "wmd": {
        "name": "Word Mover's Distance (approx.)",
        "description": "Measures the minimum semantic 'travel' needed to transform one text's word embeddings into another's.",
        "formula": "WMD(d₁,d₂) = min flow cost between word embedding distributions",
        "type": "Embedding / Semantic"
    },
    "bleu": {
        "name": "BLEU",
        "description": "Bilingual Evaluation Understudy. Geometric mean of 1-4 gram precisions with a brevity penalty.",
        "formula": "BLEU = BP × exp(Σ wₙ log pₙ), pₙ = clipped n-gram precision",
        "type": "NLG Evaluation"
    },
    "rouge_l": {
        "name": "ROUGE-L",
        "description": "Recall-Oriented Understudy for Gisting Evaluation using Longest Common Subsequence.",
        "formula": "F₁ = 2 × P_lcs × R_lcs / (P_lcs + R_lcs)",
        "type": "NLG Evaluation"
    },
    "rouge_2": {
        "name": "ROUGE-2",
        "description": "ROUGE variant measuring bigram overlap between reference and hypothesis.",
        "formula": "F₁ = 2 × (bigram_overlap) / (|bigrams(ref)| + |bigrams(hyp)|)",
        "type": "NLG Evaluation"
    },
    "meteor": {
        "name": "METEOR",
        "description": "Metric for Evaluation of Translation with Explicit ORdering. Combines unigram precision/recall with a fragmentation penalty.",
        "formula": "METEOR = F_mean × (1 − penalty), penalty = 0.5 × (chunks/matches)³",
        "type": "NLG Evaluation"
    },
    "kl_divergence": {
        "name": "Symmetrised KL Divergence",
        "description": "Measures how different two term probability distributions are. Converted to similarity via 1/(1+KL).",
        "formula": "sim = 1/(1 + (KL(P‖Q) + KL(Q‖P))/2)",
        "type": "Information-Theoretic"
    },
    "jensen_shannon": {
        "name": "Jensen-Shannon Divergence",
        "description": "Smoothed, symmetric version of KL divergence bounded in [0,1]. Similarity = 1 − JSD.",
        "formula": "JSD(P‖Q) = ½KL(P‖M) + ½KL(Q‖M), sim = 1 − JSD",
        "type": "Information-Theoretic"
    },
    "ncd": {
        "name": "Normalized Compression Distance",
        "description": "Uses zlib compression to estimate shared information. If two texts share lots of patterns, compressing them together is much cheaper.",
        "formula": "NCD = (C(xy) − min(C(x),C(y))) / max(C(x),C(y)), sim = 1 − NCD",
        "type": "Information-Theoretic"
    },
    "soft_cosine": {
        "name": "Soft Cosine Measure",
        "description": "Like TF cosine but uses embedding similarity between words instead of binary match.",
        "formula": "SCM(d₁,d₂) = (d₁ᵀ S d₂) / √(d₁ᵀ S d₁ · d₂ᵀ S d₂)",
        "type": "Embedding / Semantic"
    },
    "bert_score": {
        "name": "BERTScore (approx.)",
        "description": "Token-level semantic matching: for each token in one text, find the most similar token in the other using embeddings.",
        "formula": "P = avg max cos(hᵢ, rⱼ), R = avg max cos(rᵢ, hⱼ), F₁ = 2PR/(P+R)",
        "type": "Embedding / Semantic"
    },
    "trigram": {
        "name": "Character Trigram Overlap",
        "description": "Jaccard similarity over character-level trigrams. Captures morphological similarity.",
        "formula": "Jaccard(trigrams(s₁), trigrams(s₂)) = |T₁ ∩ T₂| / |T₁ ∪ T₂|",
        "type": "String / Character-Based"
    },
    "containment": {
        "name": "Containment Similarity",
        "description": "Asymmetric measure: what fraction of text1's words appear in text2. Good for plagiarism/subset detection.",
        "formula": "C(A,B) = |A ∩ B| / |A|",
        "type": "Token / Set-Based"
    },
    "hamming": {
        "name": "Hamming Similarity (Token)",
        "description": "Counts positional token mismatches between the two texts (after padding to equal length).",
        "formula": "sim = 1 − #{mismatches} / max(|t₁|, |t₂|)",
        "type": "Sequence-Based"
    },
}
