"""Summary, keyword extraction and topic identification."""
import re

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .embedder import embed


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in parts if 8 <= len(s.split()) <= 60]


def summarize(model, full_text: str, n_sentences: int = 6) -> str:
    """Extractive summary: pick the sentences closest to the centre of the paper's meaning."""
    sents = _sentences(full_text)
    if len(sents) <= n_sentences:
        return " ".join(sents)
    vecs = embed(model, sents)
    centroid = vecs.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    scores = vecs @ centroid
    top = sorted(np.argsort(-scores)[:n_sentences])  # keep original order
    return " ".join(sents[i] for i in top)


def extract_keywords(model, full_text: str, top_n: int = 10) -> list[str]:
    """Keyword candidates ranked by similarity to the whole document (KeyBERT-style)."""
    vec = CountVectorizer(ngram_range=(1, 2), stop_words="english", max_features=300,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b")
    try:
        vec.fit([full_text])
    except ValueError:
        return []
    cands = list(vec.get_feature_names_out())
    doc_vec = embed(model, [full_text[:20000]])[0]
    cand_vecs = embed(model, cands)
    order = np.argsort(-(cand_vecs @ doc_vec))
    chosen = []
    for i in order:
        c = cands[i]
        if any(c in k or k in c for k in chosen):  # skip near-duplicates
            continue
        chosen.append(c)
        if len(chosen) == top_n:
            break
    return chosen


def identify_topics(index, n_topics: int = 4, terms_per_topic: int = 4) -> list[list[str]]:
    """Cluster chunks by meaning and describe each cluster by its most distinctive terms."""
    chunks, vectors = index.chunks, index.vectors
    k = min(n_topics, len(chunks))
    if k < 2:
        return []
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(vectors)
    docs = [" ".join(chunks[i].text for i in range(len(chunks)) if labels[i] == c) for c in range(k)]
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000,
                            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b")
    m = tfidf.fit_transform(docs)
    terms = np.array(tfidf.get_feature_names_out())
    topics = []
    for row in m.toarray():
        topics.append(list(terms[np.argsort(-row)[:terms_per_topic]]))
    return topics
