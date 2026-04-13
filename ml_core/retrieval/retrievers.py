"""
Stage 1: Retrieval Models
- BM25 (sparse)
- TF-IDF + Cosine Similarity (sparse)
- Sentence-BERT (dense bi-encoder)
"""

import numpy as np
import time
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config


class BM25Retriever:
    def __init__(self, corpus_texts):
        print("[BM25] Building index...")
        t0 = time.time()
        self.corpus_texts = corpus_texts
        tokenized = [doc.lower().split() for doc in corpus_texts]
        self.bm25 = BM25Okapi(tokenized)
        self.build_time = time.time() - t0
        print(f"[BM25] Index built in {self.build_time:.2f}s")

    def retrieve(self, query, top_k=20):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def batch_retrieve(self, queries, top_k=20):
        results = {}
        for q in tqdm(queries, desc="BM25 Retrieval"):
            results[q] = self.retrieve(q, top_k)
        return results


class TFIDFRetriever:
    def __init__(self, corpus_texts):
        print("[TF-IDF] Building index...")
        t0 = time.time()
        self.corpus_texts = corpus_texts
        self.vectorizer = TfidfVectorizer(max_features=50000, stop_words="english")
        self.corpus_matrix = self.vectorizer.fit_transform(corpus_texts)
        self.build_time = time.time() - t0
        print(f"[TF-IDF] Index built in {self.build_time:.2f}s")

    def retrieve(self, query, top_k=20):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.corpus_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def batch_retrieve(self, queries, top_k=20):
        results = {}
        for q in tqdm(queries, desc="TF-IDF Retrieval"):
            results[q] = self.retrieve(q, top_k)
        return results


class SBERTRetriever:
    def __init__(self, corpus_texts, model_name=None):
        model_name = model_name or config.SBERT_MODEL
        print(f"[SBERT] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=config.DEVICE)
        self.corpus_texts = corpus_texts
        print(f"[SBERT] Encoding corpus ({len(corpus_texts)} docs)...")
        t0 = time.time()
        self.corpus_embeddings = self.model.encode(
            corpus_texts, batch_size=128, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True
        )
        self.build_time = time.time() - t0
        print(f"[SBERT] Corpus encoded in {self.build_time:.2f}s")

    def retrieve(self, query, top_k=20):
        query_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores = np.dot(self.corpus_embeddings, query_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def batch_retrieve(self, queries, top_k=20):
        print(f"[SBERT] Encoding {len(queries)} queries...")
        query_embs = self.model.encode(
            queries, batch_size=128, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True
        )
        results = {}
        scores_matrix = np.dot(query_embs, self.corpus_embeddings.T)
        for i, q in enumerate(tqdm(queries, desc="SBERT Retrieval")):
            scores = scores_matrix[i]
            top_indices = np.argsort(scores)[::-1][:top_k]
            results[q] = [(int(idx), float(scores[idx])) for idx in top_indices]
        return results


def get_retriever(name, corpus_texts):
    """Factory function."""
    if name == "bm25":
        return BM25Retriever(corpus_texts)
    elif name == "tfidf":
        return TFIDFRetriever(corpus_texts)
    elif name == "sbert":
        return SBERTRetriever(corpus_texts)
    else:
        raise ValueError(f"Unknown retriever: {name}")
