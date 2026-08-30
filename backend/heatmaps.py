import math
from collections import Counter

import numpy as np

from model_cache import get_embeddings


def _clip01(matrix):
    return [[max(0.0, min(1.0, round(v, 4))) for v in row] for row in matrix]


def _hm_embedding_cosine(tok1, tok2, model_id):
    all_embs = get_embeddings(tok1 + tok2, model_id)
    e1 = all_embs[:len(tok1)]
    e2 = all_embs[len(tok1):]
    e1 /= (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-8)
    e2 /= (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-8)
    return _clip01((e1 @ e2.T).tolist())


def _hm_jaccard(tok1, tok2, model_id):
    return _clip01([[1.0 if a == b else 0.0 for b in tok2] for a in tok1])

def _hm_dice(tok1, tok2, model_id):
    return _clip01([[1.0 if a == b else 0.0 for b in tok2] for a in tok1])

def _hm_overlap(tok1, tok2, model_id):
    return _clip01([[1.0 if a == b else 0.0 for b in tok2] for a in tok1])

def _hm_containment(tok1, tok2, model_id):
    return _clip01([[1.0 if a == b else 0.0 for b in tok2] for a in tok1])


def _hm_levenshtein(tok1, tok2, model_id):
    def lev_sim(a, b):
        m, n = len(a), len(b)
        if m == 0 and n == 0: return 1.0
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1): dp[i][0] = i
        for j in range(n+1): dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1,
                               dp[i-1][j-1]+(0 if a[i-1]==b[j-1] else 1))
        return 1.0 - dp[m][n] / max(m, n)
    return _clip01([[lev_sim(a, b) for b in tok2] for a in tok1])


def _hm_jaro_winkler(tok1, tok2, model_id):
    def jw(s1, s2):
        if s1 == s2: return 1.0
        l1, l2 = len(s1), len(s2)
        if not l1 or not l2: return 0.0
        md = max(l1, l2)//2 - 1
        m1 = [False]*l1; m2 = [False]*l2
        matches = 0
        for i in range(l1):
            for j in range(max(0, i-md), min(i+md+1, l2)):
                if not m2[j] and s1[i]==s2[j]:
                    m1[i]=m2[j]=True; matches+=1; break
        if not matches: return 0.0
        t = sum(s1[i]!=s2[k] for i, (k, _) in
                zip([i for i in range(l1) if m1[i]],
                    enumerate([j for j in range(l2) if m2[j]]))) / 2
        j_score = (matches/l1+matches/l2+(matches-t)/matches)/3
        prefix = sum(1 for i in range(min(4, l1, l2)) if s1[i]==s2[i])
        return j_score + prefix*0.1*(1-j_score)
    return _clip01([[jw(a, b) for b in tok2] for a in tok1])


def _hm_trigram(tok1, tok2, model_id):
    def tg(w): return set(w[i:i+3] for i in range(max(1, len(w)-2)))
    def sim(a, b):
        sa, sb = tg(a), tg(b)
        return len(sa & sb)/len(sa | sb) if (sa | sb) else 0.0
    return _clip01([[sim(a, b) for b in tok2] for a in tok1])


def _hm_hamming(tok1, tok2, model_id):
    def ham(a, b):
        ml = max(len(a), len(b), 1)
        ap = a.ljust(ml); bp = b.ljust(ml)
        return 1.0 - sum(x!=y for x,y in zip(ap,bp)) / ml
    return _clip01([[ham(a, b) for b in tok2] for a in tok1])


def _hm_bigram(tok1, tok2, model_id):
    bg1 = set(zip(tok1, tok1[1:]))
    bg2 = set(zip(tok2, tok2[1:]))
    shared = bg1 & bg2
    return _clip01([[1.0 if (a, b) in shared else 0.0 for b in tok2] for a in tok1])


def _hm_lcs(tok1, tok2, model_id):
    def lcs_sim(a, b):
        m, n = len(a), len(b)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return 2*dp[m][n]/(m+n) if (m+n) else 0.0
    return _clip01([[lcs_sim(a, b) for b in tok2] for a in tok1])


def _hm_tfidf(tok1, tok2, model_id):
    vocab = set(tok1 + tok2)
    N = 2
    docs = [set(tok1), set(tok2)]
    idf = {w: math.log(1 + N / max(sum(1 for d in docs if w in d), 1)) for w in vocab}
    tf1 = {w: (1 / len(tok1) if w in tok1 else 0) for w in vocab}
    tf2 = {w: (1 / len(tok2) if w in tok2 else 0) for w in vocab}
    def cell(a, b):
        if a != b: return 0.0
        return tf1.get(a, 0) * idf.get(a, 0) * tf2.get(b, 0) * idf.get(b, 0)
    raw = [[cell(a, b) for b in tok2] for a in tok1]
    max_v = max((v for row in raw for v in row), default=1.0) or 1.0
    return _clip01([[v / max_v for v in row] for row in raw])


def _hm_bleu(tok1, tok2, model_id):
    return _clip01([[1.0 if a == b else 0.0 for b in tok2] for a in tok1])


def _hm_rouge_l(tok1, tok2, model_id):
    return _clip01([[1.0 if a == b else 0.0 for b in tok2] for a in tok1])


def _hm_rouge_2(tok1, tok2, model_id):
    bg1 = set(zip(tok1, tok1[1:]))
    bg2 = set(zip(tok2, tok2[1:]))
    shared = bg1 & bg2
    return _clip01([[1.0 if (a, b) in shared else 0.0 for b in tok2] for a in tok1])


def _hm_meteor(tok1, tok2, model_id):
    def bg(w): return Counter(w[i:i+2] for i in range(max(1, len(w)-1)))
    def cell(a, b):
        if a == b: return 1.0
        ba, bb = bg(a), bg(b)
        inter = sum(min(ba[k], bb[k]) for k in ba)
        denom = max(sum(ba.values()), sum(bb.values()), 1)
        return inter / denom * 0.75
    return _clip01([[cell(a, b) for b in tok2] for a in tok1])


def _char_dist(word):
    if not word: return {}
    counts = Counter(word)
    total = len(word)
    return {c: v / total for c, v in counts.items()}


def _hm_kl_divergence(tok1, tok2, model_id):
    def sym_kl(a, b):
        pa = _char_dist(a)
        pb = _char_dist(b)
        vocab = set(pa) | set(pb)
        eps = 1e-9
        kl_ab = sum(pa.get(g, eps) * math.log(pa.get(g, eps) / pb.get(g, eps)) for g in vocab)
        kl_ba = sum(pb.get(g, eps) * math.log(pb.get(g, eps) / pa.get(g, eps)) for g in vocab)
        return 1.0 / (1.0 + (kl_ab + kl_ba) / 2.0)
    return _clip01([[sym_kl(a, b) for b in tok2] for a in tok1])


def _hm_jensen_shannon(tok1, tok2, model_id):
    def jsd_sim(a, b):
        pa = _char_dist(a)
        pb = _char_dist(b)
        vocab = set(pa) | set(pb)
        eps = 1e-9
        m = {g: 0.5 * (pa.get(g, eps) + pb.get(g, eps)) for g in vocab}
        kl_am = sum(pa.get(g, eps) * math.log(pa.get(g, eps) / m[g]) for g in vocab)
        kl_bm = sum(pb.get(g, eps) * math.log(pb.get(g, eps) / m[g]) for g in vocab)
        jsd = 0.5 * kl_am + 0.5 * kl_bm
        return max(0.0, 1.0 - (jsd / math.log(2)))
    return _clip01([[jsd_sim(a, b) for b in tok2] for a in tok1])


def _hm_ncd(tok1, tok2, model_id):
    def lev_sim(a, b):
        m, n = len(a), len(b)
        if m == 0 and n == 0: return 1.0
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1): dp[i][0] = i
        for j in range(n+1): dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1,
                               dp[i-1][j-1]+(0 if a[i-1]==b[j-1] else 1))
        return 1.0 - dp[m][n] / max(m, n)
    return _clip01([[lev_sim(a, b) for b in tok2] for a in tok1])


HEATMAP_STRATEGIES = {
    "embedding_cosine": _hm_embedding_cosine,
    "wmd": _hm_embedding_cosine,
    "soft_cosine": _hm_embedding_cosine,
    "bert_score": _hm_embedding_cosine,
    "jaccard": _hm_jaccard,
    "dice": _hm_dice,
    "overlap": _hm_overlap,
    "containment": _hm_containment,
    "levenshtein": _hm_levenshtein,
    "jaro_winkler": _hm_jaro_winkler,
    "trigram": _hm_trigram,
    "bigram_overlap": _hm_bigram,
    "lcs": _hm_lcs,
    "hamming": _hm_hamming,
    "tfidf_cosine": _hm_tfidf,
    "bleu": _hm_bleu,
    "rouge_l": _hm_rouge_l,
    "rouge_2": _hm_rouge_2,
    "meteor": _hm_meteor,
    "kl_divergence": _hm_kl_divergence,
    "jensen_shannon": _hm_jensen_shannon,
    "ncd": _hm_ncd,
}

# ToDo -add labels
HEATMAP_STRATEGY_LABELS = {
    "embedding_cosine": ("", ""),
    "jaccard": ("", ""),
    "dice": ("", ""),
    "overlap": ("", ""),
    "containment": ("", ""),
    "levenshtein": ("", ""),
    "jaro_winkler": ("", ""),
    "trigram": ("", ""),
    "bigram_overlap": ("", ""),
    "lcs": ("", ""),
    "hamming": ("", ""),
    "tfidf_cosine": ("", ""),
    "bleu": ("", ""),
    "rouge_l": ("", ""),
    "rouge_2": ("", ""),
    "meteor": ("", ""),
    "kl_divergence": ("", ""),
    "jensen_shannon": ("", ""),
    "ncd": ("", ""),
    "wmd": ("", ""),
    "soft_cosine": ("", ""),
    "bert_score": ("", ""),
}
