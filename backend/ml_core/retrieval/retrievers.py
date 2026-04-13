"""
Stage 1: Retrieval Models
- BM25 (sparse)
- TF-IDF + Cosine Similarity (sparse)
- Sentence-BERT (dense bi-encoder, optional)
"""

import numpy as np
import time
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import ml_core.config as config

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # optional dependency
    SentenceTransformer = None


class BM25Retriever:
    def __init__(self, corpus_texts):
        self.corpus_texts = corpus_texts
        tokenized = [doc.lower().split() for doc in corpus_texts]
        self.use_native = BM25Okapi is not None
        if self.use_native:
            self.bm25 = BM25Okapi(tokenized)
        else:
            # Approximate fallback so the app still runs without rank_bm25 installed.
            self.vectorizer = CountVectorizer(stop_words='english')
            self.term_matrix = self.vectorizer.fit_transform(corpus_texts).astype(float)
            self.doc_len = np.asarray(self.term_matrix.sum(axis=1)).ravel()
            self.avgdl = self.doc_len.mean() if len(self.doc_len) else 0.0
            df = np.bincount(self.term_matrix.indices, minlength=self.term_matrix.shape[1])
            n_docs = self.term_matrix.shape[0]
            self.idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def retrieve(self, query, top_k=20):
        tokenized_query = query.lower().split()
        if self.use_native:
            scores = self.bm25.get_scores(tokenized_query)
        else:
            q_vec = self.vectorizer.transform([query])
            q_terms = q_vec.indices
            scores = np.zeros(self.term_matrix.shape[0], dtype=float)
            k1 = 1.5
            b = 0.75
            for term_id in q_terms:
                col = self.term_matrix[:, term_id].toarray().ravel()
                denom = col + k1 * (1 - b + b * self.doc_len / (self.avgdl + 1e-9))
                scores += self.idf[term_id] * (col * (k1 + 1) / (denom + 1e-9))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def batch_retrieve(self, queries, top_k=20):
        return {q: self.retrieve(q, top_k) for q in tqdm(queries, desc='BM25 Retrieval')}


class TFIDFRetriever:
    def __init__(self, corpus_texts):
        self.corpus_texts = corpus_texts
        self.vectorizer = TfidfVectorizer(max_features=50000, stop_words='english')
        self.corpus_matrix = self.vectorizer.fit_transform(corpus_texts)

    def retrieve(self, query, top_k=20):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.corpus_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def batch_retrieve(self, queries, top_k=20):
        return {q: self.retrieve(q, top_k) for q in tqdm(queries, desc='TF-IDF Retrieval')}


class SBERTRetriever:
    def __init__(self, corpus_texts, model_name=None):
        if SentenceTransformer is None:
            raise RuntimeError('sentence-transformers is not installed. Install optional dependencies to use SBERT retrieval.')
        model_name = model_name or config.SBERT_MODEL
        self.model = SentenceTransformer(model_name, device=config.DEVICE)
        self.corpus_texts = corpus_texts
        self.corpus_embeddings = self.model.encode(
            corpus_texts,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def retrieve(self, query, top_k=20):
        query_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores = np.dot(self.corpus_embeddings, query_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def batch_retrieve(self, queries, top_k=20):
        query_embs = self.model.encode(
            queries,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        results = {}
        scores_matrix = np.dot(query_embs, self.corpus_embeddings.T)
        for i, q in enumerate(tqdm(queries, desc='SBERT Retrieval')):
            scores = scores_matrix[i]
            top_indices = np.argsort(scores)[::-1][:top_k]
            results[q] = [(int(idx), float(scores[idx])) for idx in top_indices]
        return results


def get_retriever(name, corpus_texts):
    name = name.lower()
    if name == 'bm25':
        return BM25Retriever(corpus_texts)
    if name == 'tfidf':
        return TFIDFRetriever(corpus_texts)
    if name == 'sbert':
        return SBERTRetriever(corpus_texts)
    raise ValueError(f'Unknown retriever: {name}')
