"""
Visualize actual ranking examples — see what the model does query by query.
Saves output to results/qualitative_examples.txt
"""

import os, sys, torch, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_prep import load_data, split_data
from retrieval.retrievers import BM25Retriever, TFIDFRetriever
from reranker.models import StandardCrossEncoder, get_tokenizer
from reranker.trainer import rerank_candidates


def main():
    output_path = os.path.join(config.RESULTS_DIR, "qualitative_examples.txt")
    outfile = open(output_path, "w")

    def log(msg=""):
        print(msg)
        outfile.write(msg + "\n")

    # Load data
    df = load_data()
    train_df, val_df, test_df = split_data(df)

    all_q2 = pd.concat([train_df, val_df, test_df])["question2"].unique().tolist()
    corpus_texts = all_q2
    corpus_id_map = {text: idx for idx, text in enumerate(corpus_texts)}

    # Relevance
    relevance = defaultdict(set)
    for _, row in test_df.iterrows():
        if row["is_duplicate"] == 1 and row["question2"] in corpus_id_map:
            relevance[row["question1"]].add(corpus_id_map[row["question2"]])

    test_queries = [q for q in test_df["question1"].unique() if len(relevance[q]) > 0]

    # Build retriever
    print("Building TF-IDF index...")
    retriever = TFIDFRetriever(corpus_texts)

    # Load trained model
    model_path = os.path.join(config.MODEL_DIR, "Standard-CE-DistilBERT.pt")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Run run_transformers.py first.")
        return

    print("Loading Standard-CE-DistilBERT model...")
    model = StandardCrossEncoder(model_name=config.DISTILBERT_MODEL)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    model.eval()
    tokenizer = get_tokenizer(config.DISTILBERT_MODEL)

    # Pick 5 random queries that have relevant docs
    np.random.seed(42)
    sample_queries = list(np.random.choice(test_queries, min(5, len(test_queries)), replace=False))

    log("=" * 80)
    log("QUALITATIVE ANALYSIS - Two-Stage Passage Ranking Examples")
    log("=" * 80)

    for i, query in enumerate(sample_queries):
        log(f"\n{'=' * 80}")
        log(f"QUERY {i+1}: \"{query}\"")
        log(f"{'=' * 80}")

        relevant_ids = relevance[query]
        relevant_texts = [corpus_texts[rid] for rid in relevant_ids]
        log(f"\nGROUND TRUTH DUPLICATES ({len(relevant_ids)}):")
        for rt in relevant_texts:
            log(f"  [RELEVANT] \"{rt}\"")

        # Stage 1: Retrieval
        retrieved = retriever.retrieve(query, top_k=10)
        log(f"\n--- STAGE 1: TF-IDF Retrieval (Top 10) ---")
        for rank, (doc_id, score) in enumerate(retrieved):
            text = corpus_texts[doc_id]
            is_rel = "[DUPLICATE]" if doc_id in relevant_ids else "[x]"
            log(f"  Rank {rank+1}: [{score:.4f}] {is_rel} \"{text}\"")

        # Stage 2: Reranking
        cand_ids = [r[0] for r in retrieved]
        cand_texts = [corpus_texts[r[0]] for r in retrieved]
        reranked = rerank_candidates(
            model, tokenizer, query, cand_texts, cand_ids,
            model_type="standard", max_seq_len=config.DEFAULT_MAX_SEQ_LEN
        )

        log(f"\n--- STAGE 2: Cross-Encoder Reranking ---")
        for rank, (doc_id, score) in enumerate(reranked):
            text = corpus_texts[doc_id]
            is_rel = "[DUPLICATE]" if doc_id in relevant_ids else "[x]"
            log(f"  Rank {rank+1}: [{score:.4f}] {is_rel} \"{text}\"")

        # Show improvement
        retrieval_rank = None
        rerank_rank = None
        if relevant_ids:
            for rank, (doc_id, _) in enumerate(retrieved):
                if doc_id in relevant_ids:
                    retrieval_rank = rank + 1
                    break
            for rank, (doc_id, _) in enumerate(reranked):
                if doc_id in relevant_ids:
                    rerank_rank = rank + 1
                    break

        if retrieval_rank and rerank_rank:
            log(f"\n  >> Duplicate was at rank {retrieval_rank} after retrieval, "
                f"moved to rank {rerank_rank} after reranking")
            if rerank_rank < retrieval_rank:
                log(f"  >> IMPROVED by {retrieval_rank - rerank_rank} positions!")
            elif rerank_rank == retrieval_rank:
                log(f"  >> Same position (already correct)")
            else:
                log(f"  >> Dropped by {rerank_rank - retrieval_rank} positions")
        elif retrieval_rank is None:
            log(f"\n  >> Duplicate was NOT in top-10 retrieval results")

    log(f"\n{'=' * 80}")
    log("DONE - These are real examples from your test set!")
    log(f"{'=' * 80}")

    outfile.close()
    print(f"\n>>> Results saved to: {output_path}")


if __name__ == "__main__":
    main()
