"""
Evaluation Metrics for Information Retrieval
- nDCG@k, MRR@k, MAP@k, Recall@k, Precision@k
"""

import numpy as np
from collections import defaultdict


def dcg_at_k(relevance_scores, k):
    """Compute DCG@k."""
    relevance_scores = np.array(relevance_scores[:k], dtype=np.float64)
    if len(relevance_scores) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevance_scores) + 2))
    return np.sum(relevance_scores / discounts)


def ndcg_at_k(relevance_scores, k, n_relevant):
    """Compute nDCG@k."""
    actual_dcg = dcg_at_k(relevance_scores, k)
    # Ideal: all relevant docs at top
    ideal_relevance = [1.0] * min(n_relevant, k) + [0.0] * max(0, k - n_relevant)
    ideal_dcg = dcg_at_k(ideal_relevance, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def mrr_at_k(relevance_scores, k):
    """Compute MRR@k (reciprocal rank of first relevant doc)."""
    for i, rel in enumerate(relevance_scores[:k]):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(relevance_scores, k):
    """Compute Precision@k."""
    return sum(1 for r in relevance_scores[:k] if r > 0) / k


def recall_at_k(relevance_scores, k, n_relevant):
    """Compute Recall@k."""
    if n_relevant == 0:
        return 0.0
    return sum(1 for r in relevance_scores[:k] if r > 0) / n_relevant


def average_precision_at_k(relevance_scores, k, n_relevant):
    """Compute Average Precision@k."""
    if n_relevant == 0:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        if rel > 0:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / min(n_relevant, k)


def evaluate_ranking(ranked_doc_ids, relevant_doc_ids, k_values=[1, 5, 10, 20]):
    """
    Evaluate a single query's ranking.

    Args:
        ranked_doc_ids: list of doc IDs in ranked order
        relevant_doc_ids: set of truly relevant doc IDs
        k_values: list of k values to evaluate

    Returns:
        dict of metric_name -> value
    """
    # Build binary relevance vector
    relevance_scores = [1 if doc_id in relevant_doc_ids else 0 for doc_id in ranked_doc_ids]
    n_relevant = len(relevant_doc_ids)

    results = {}
    for k in k_values:
        results[f"nDCG@{k}"] = ndcg_at_k(relevance_scores, k, n_relevant)
        results[f"MRR@{k}"] = mrr_at_k(relevance_scores, k)
        results[f"P@{k}"] = precision_at_k(relevance_scores, k)
        results[f"Recall@{k}"] = recall_at_k(relevance_scores, k, n_relevant)
        results[f"MAP@{k}"] = average_precision_at_k(relevance_scores, k, n_relevant)

    return results


def aggregate_metrics(all_query_results):
    """Average metrics across all queries."""
    if not all_query_results:
        return {}
    agg = defaultdict(list)
    for qr in all_query_results:
        for k, v in qr.items():
            agg[k].append(v)
    return {k: np.mean(v) for k, v in agg.items()}
