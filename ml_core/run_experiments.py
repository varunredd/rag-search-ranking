"""
Main Experiment Runner
Runs the complete two-stage passage ranking pipeline.
"""

import os, sys, json, time, numpy as np, pandas as pd
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_prep import load_data, split_data, build_training_triplets
from retrieval.retrievers import BM25Retriever, TFIDFRetriever
from reranker.models import SklearnCrossEncoder
from evaluation.metrics import evaluate_ranking, aggregate_metrics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def prepare_data():
    print("=" * 70)
    print("STEP 1: DATA PREPARATION")
    print("=" * 70)
    df = load_data()
    train_df, val_df, test_df = split_data(df)

    all_q2 = pd.concat([train_df, val_df, test_df])["question2"].unique().tolist()
    corpus_texts = all_q2
    corpus_id_map = {text: idx for idx, text in enumerate(corpus_texts)}

    relevance = defaultdict(set)
    for _, row in test_df.iterrows():
        if row["is_duplicate"] == 1 and row["question2"] in corpus_id_map:
            relevance[row["question1"]].add(corpus_id_map[row["question2"]])

    test_queries = [q for q in test_df["question1"].unique() if len(relevance[q]) > 0]
    if len(test_queries) > config.NUM_EVAL_QUERIES:
        np.random.seed(config.RANDOM_SEED)
        test_queries = list(np.random.choice(test_queries, config.NUM_EVAL_QUERIES, replace=False))

    train_rel = defaultdict(set)
    for _, row in train_df.iterrows():
        if row["is_duplicate"] == 1 and row["question2"] in corpus_id_map:
            train_rel[row["question1"]].add(corpus_id_map[row["question2"]])

    triplets = build_training_triplets(train_df, corpus_texts, corpus_id_map, train_rel, n_hard_negatives=config.DEFAULT_HARD_NEGS)

    print(f"\n  Corpus: {len(corpus_texts)} | Test queries: {len(test_queries)} | Triplets: {len(triplets)}")
    return {"corpus_texts": corpus_texts, "corpus_id_map": corpus_id_map,
            "relevance": relevance, "test_queries": test_queries,
            "train_triplets": triplets, "train_df": train_df, "val_df": val_df, "test_df": test_df}


def run_retrieval(data, top_k=None):
    top_k = top_k or config.DEFAULT_TOP_K
    print(f"\n{'='*70}\nSTEP 2: STAGE 1 RETRIEVAL (Top-{top_k})\n{'='*70}")
    corpus = data["corpus_texts"]
    queries = data["test_queries"]
    relevance = data["relevance"]

    retriever_classes = {"BM25": BM25Retriever, "TF-IDF": TFIDFRetriever}
    retrieval_results, retrieval_metrics, retrieval_times = {}, {}, {}

    for name, cls in retriever_classes.items():
        print(f"\n--- {name} ---")
        retriever = cls(corpus)
        t0 = time.time()
        results = retriever.batch_retrieve(queries, top_k=top_k)
        retrieval_times[name] = time.time() - t0
        retrieval_results[name] = results

        all_m = []
        for q in queries:
            ranked = [r[0] for r in results[q]]
            all_m.append(evaluate_ranking(ranked, relevance[q], k_values=config.EVAL_K_VALUES))
        agg = aggregate_metrics(all_m)
        retrieval_metrics[name] = agg
        print(f"  nDCG@10={agg.get('nDCG@10',0):.4f}  MRR@10={agg.get('MRR@10',0):.4f}  "
              f"Recall@20={agg.get('Recall@20',0):.4f}  Time={retrieval_times[name]:.1f}s")

    return retrieval_results, retrieval_metrics, retrieval_times


def train_and_evaluate_rerankers(data, retrieval_results, retriever_name="TF-IDF"):
    print(f"\n{'='*70}\nSTEP 3: RERANKER TRAINING & EVALUATION\n{'='*70}")
    corpus = data["corpus_texts"]
    queries = data["test_queries"]
    relevance = data["relevance"]
    triplets = data["train_triplets"]
    candidates = retrieval_results[retriever_name]

    model_configs = [
        {"name": "MLP-Small(128)", "hidden": 128, "type": "standard"},
        {"name": "MLP-Medium(256)", "hidden": 256, "type": "standard"},
        {"name": "MLP-Large(512)", "hidden": 512, "type": "standard"},
        {"name": "Modified-MLP(256)", "hidden": 256, "type": "modified"},
        {"name": "Modified-MLP(512)", "hidden": 512, "type": "modified"},
    ]

    reranker_results, reranker_times = {}, {}
    for cfg in model_configs:
        print(f"\n--- {cfg['name']} ---")
        model = SklearnCrossEncoder(hidden_size=cfg["hidden"], model_type=cfg["type"])
        t0 = time.time()
        model.fit(triplets, epochs=config.DEFAULT_EPOCHS)
        train_time = time.time() - t0

        t0 = time.time()
        all_m = []
        for q in tqdm(queries, desc="Reranking"):
            cands = candidates[q]
            cand_ids = [c[0] for c in cands]
            cand_texts = [corpus[c[0]] for c in cands]
            reranked = model.rerank(q, cand_texts, cand_ids)
            ranked_ids = [r[0] for r in reranked]
            all_m.append(evaluate_ranking(ranked_ids, relevance[q], k_values=config.EVAL_K_VALUES))
        rerank_time = time.time() - t0

        agg = aggregate_metrics(all_m)
        reranker_results[cfg["name"]] = agg
        reranker_times[cfg["name"]] = {"train": train_time, "rerank": rerank_time}
        print(f"  nDCG@10={agg.get('nDCG@10',0):.4f}  MRR@10={agg.get('MRR@10',0):.4f}  "
              f"P@1={agg.get('P@1',0):.4f}  Train={train_time:.1f}s")

    return reranker_results, reranker_times


def run_ablations(data, retrieval_results, retriever_name="TF-IDF"):
    print(f"\n{'='*70}\nSTEP 4: ABLATION STUDIES\n{'='*70}")
    corpus = data["corpus_texts"]
    queries = data["test_queries"]
    relevance = data["relevance"]
    triplets = data["train_triplets"]
    candidates = retrieval_results[retriever_name]

    def eval_model(model):
        all_m = []
        for q in queries:
            cands = candidates[q]
            cand_ids = [c[0] for c in cands]
            cand_texts = [corpus[c[0]] for c in cands]
            reranked = model.rerank(q, cand_texts, cand_ids)
            ranked_ids = [r[0] for r in reranked]
            all_m.append(evaluate_ranking(ranked_ids, relevance[q], k_values=config.EVAL_K_VALUES))
        return aggregate_metrics(all_m)

    ablation_results = {}

    # Ablation 1: Hidden layer size
    print("\n[ABLATION 1] Hidden layer size")
    hidden_results = {}
    for h in [64, 128, 256, 512]:
        model = SklearnCrossEncoder(hidden_size=h, model_type="modified")
        model.fit(triplets, epochs=5)
        hidden_results[h] = eval_model(model)
        print(f"  hidden={h}: nDCG@10={hidden_results[h].get('nDCG@10',0):.4f}")
    ablation_results["hidden_size"] = hidden_results

    # Ablation 2: Training data size
    print("\n[ABLATION 2] Training data size")
    size_results = {}
    for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
        n = max(100, int(len(triplets) * frac))
        model = SklearnCrossEncoder(hidden_size=256, model_type="modified")
        model.fit(triplets[:n], epochs=5)
        size_results[frac] = eval_model(model)
        print(f"  frac={frac} ({n}): nDCG@10={size_results[frac].get('nDCG@10',0):.4f}")
    ablation_results["training_size"] = size_results

    # Ablation 3: Hard negatives
    print("\n[ABLATION 3] Hard negatives per query")
    neg_results = {}
    for n_neg in [1, 3, 5, 7]:
        neg_triplets = build_training_triplets(
            data["train_df"], corpus, data["corpus_id_map"],
            defaultdict(set), n_hard_negatives=n_neg
        )
        if len(neg_triplets) < 50:
            neg_triplets = triplets[:500]
        model = SklearnCrossEncoder(hidden_size=256, model_type="modified")
        model.fit(neg_triplets, epochs=5)
        neg_results[n_neg] = eval_model(model)
        print(f"  neg={n_neg}: nDCG@10={neg_results[n_neg].get('nDCG@10',0):.4f}")
    ablation_results["hard_negatives"] = neg_results

    # Ablation 4: Top-K sweep
    print("\n[ABLATION 4] Top-K candidate pool size")
    topk_results = {}
    retriever = BM25Retriever(corpus)
    for k in [5, 10, 20, 50, 100]:
        results_k = retriever.batch_retrieve(queries, top_k=k)
        all_m = []
        for q in queries:
            ranked = [r[0] for r in results_k[q]]
            all_m.append(evaluate_ranking(ranked, relevance[q], k_values=[1, 5, 10, min(k, 20)]))
        topk_results[k] = aggregate_metrics(all_m)
        print(f"  K={k}: nDCG@10={topk_results[k].get('nDCG@10',0):.4f}")
    ablation_results["top_k"] = topk_results

    return ablation_results


def generate_all_plots(retrieval_metrics, reranker_results, ablation_results,
                       retrieval_times, reranker_times):
    print(f"\n{'='*70}\nGENERATING PLOTS\n{'='*70}")
    out = config.RESULTS_DIR
    plt.rcParams.update({"font.size": 11, "figure.facecolor": "white"})

    # Combine all models
    all_models = {}
    for k, v in retrieval_metrics.items():
        all_models[f"Retrieval-{k}"] = v
    all_models.update(reranker_results)

    # Plot 1: Retrieval comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    models = list(retrieval_metrics.keys())
    metrics_show = ["nDCG@10", "MRR@10", "Recall@20", "P@1"]
    x = np.arange(len(models)); w = 0.2
    for i, m in enumerate(metrics_show):
        vals = [retrieval_metrics[model].get(m, 0) for model in models]
        axes[0].bar(x + i * w, vals, w, label=m)
    axes[0].set_xticks(x + 1.5 * w); axes[0].set_xticklabels(models)
    axes[0].set_title("Stage 1: Retrieval Model Comparison"); axes[0].set_ylabel("Score")
    axes[0].legend(fontsize=9); axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(models, [retrieval_times[m] for m in models], color=["#4C72B0", "#DD8452"])
    axes[1].set_title("Retrieval Time (seconds)"); axes[1].set_ylabel("Time (s)")
    axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, "01_retrieval_comparison.png"), dpi=150); plt.close()

    # Plot 2: Full pipeline comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    model_names = list(all_models.keys())
    metrics_show = ["nDCG@10", "MRR@10", "P@1", "MAP@10"]
    x = np.arange(len(model_names)); w = 0.2
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for i, m in enumerate(metrics_show):
        vals = [all_models[mn].get(m, 0) for mn in model_names]
        ax.bar(x + i * w, vals, w, label=m, color=colors[i])
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=9)
    ax.set_title("Full Pipeline: Retrieval vs Retrieval + Reranking")
    ax.set_ylabel("Score"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, "02_full_comparison.png"), dpi=150); plt.close()

    # Plot 3: Hidden size ablation
    if "hidden_size" in ablation_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        hs = sorted(ablation_results["hidden_size"].keys())
        for metric in ["nDCG@10", "MRR@10", "P@1"]:
            ax.plot(hs, [ablation_results["hidden_size"][h].get(metric, 0) for h in hs], "o-", label=metric, linewidth=2)
        ax.set_xlabel("Hidden Layer Size"); ax.set_ylabel("Score")
        ax.set_title("Ablation: Model Capacity"); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out, "03_ablation_hidden_size.png"), dpi=150); plt.close()

    # Plot 4: Training size ablation
    if "training_size" in ablation_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        fracs = sorted(ablation_results["training_size"].keys())
        for metric in ["nDCG@10", "MRR@10"]:
            ax.plot([f*100 for f in fracs], [ablation_results["training_size"][f].get(metric, 0) for f in fracs], "o-", label=metric, linewidth=2)
        ax.set_xlabel("Training Data (%)"); ax.set_ylabel("Score")
        ax.set_title("Ablation: Training Data Size"); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out, "04_ablation_training_size.png"), dpi=150); plt.close()

    # Plot 5: Hard negatives ablation
    if "hard_negatives" in ablation_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        negs = sorted(ablation_results["hard_negatives"].keys())
        for metric in ["nDCG@10", "MRR@10"]:
            ax.plot(negs, [ablation_results["hard_negatives"][n].get(metric, 0) for n in negs], "o-", label=metric, linewidth=2)
        ax.set_xlabel("Hard Negatives per Query"); ax.set_ylabel("Score")
        ax.set_title("Ablation: Hard Negative Mining"); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out, "05_ablation_hard_negatives.png"), dpi=150); plt.close()

    # Plot 6: Top-K sweep
    if "top_k" in ablation_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        ks = sorted(ablation_results["top_k"].keys())
        for metric in ["nDCG@10", "MRR@10"]:
            ax.plot(ks, [ablation_results["top_k"][k].get(metric, 0) for k in ks], "o-", label=metric, linewidth=2)
        ax.set_xlabel("Top-K Candidates"); ax.set_ylabel("Score")
        ax.set_title("Ablation: Top-K Candidate Pool Size"); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out, "06_ablation_topk.png"), dpi=150); plt.close()

    # Plot 7: Heatmap
    fig, ax = plt.subplots(figsize=(14, 7))
    model_names = list(all_models.keys())
    all_metric_names = ["nDCG@1","nDCG@5","nDCG@10","nDCG@20","MRR@5","MRR@10","P@1","P@5","Recall@10","Recall@20","MAP@10","MAP@20"]
    matrix = np.array([[all_models[mn].get(m, 0) for m in all_metric_names] for mn in model_names])
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(len(all_metric_names))); ax.set_xticklabels(all_metric_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(model_names))); ax.set_yticklabels(model_names, fontsize=9)
    for i in range(len(model_names)):
        for j in range(len(all_metric_names)):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im); ax.set_title("Comprehensive Metrics Heatmap")
    plt.tight_layout(); plt.savefig(os.path.join(out, "07_metrics_heatmap.png"), dpi=150); plt.close()

    print(f"  7 plots saved to {out}/")


def save_results(retrieval_metrics, reranker_results, ablation_results):
    out = config.RESULTS_DIR
    all_models = {}
    for k, v in retrieval_metrics.items():
        all_models[f"Retrieval-{k}"] = v
    all_models.update(reranker_results)

    df = pd.DataFrame(all_models).T.round(4)
    df.to_csv(os.path.join(out, "main_results.csv"))

    print(f"\n{'='*70}\nRESULTS SUMMARY\n{'='*70}")
    key_cols = [c for c in df.columns if any(c.startswith(p) for p in ["nDCG@1", "nDCG@5", "nDCG@10", "MRR@10", "P@1"])]
    if not key_cols:
        key_cols = df.columns[:8].tolist()
    print(df[key_cols].to_string())

    for name, results in ablation_results.items():
        adf = pd.DataFrame(results).T.round(4)
        adf.to_csv(os.path.join(out, f"ablation_{name}.csv"))
    print(f"\nCSVs saved to {out}/")


def main():
    t0 = time.time()
    print("="*70)
    print("TWO-STAGE PASSAGE RANKING — FULL EXPERIMENT PIPELINE")
    print(f"Device: {config.DEVICE}")
    print("="*70)

    data = prepare_data()
    retrieval_results, retrieval_metrics, retrieval_times = run_retrieval(data)
    reranker_results, reranker_times = train_and_evaluate_rerankers(data, retrieval_results)
    ablation_results = run_ablations(data, retrieval_results)
    save_results(retrieval_metrics, reranker_results, ablation_results)
    generate_all_plots(retrieval_metrics, reranker_results, ablation_results, retrieval_times, reranker_times)

    print(f"\n{'='*70}\nCOMPLETE — {(time.time()-t0)/60:.1f} minutes\n{'='*70}")

if __name__ == "__main__":
    main()
