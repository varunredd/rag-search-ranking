"""
Main Experiment Runner — Transformer Mode
Uses real DistilBERT/BERT cross-encoders for reranking.
"""

import os, sys, json, time, numpy as np, pandas as pd
from collections import defaultdict
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_prep import load_data, split_data, build_training_triplets
from retrieval.retrievers import BM25Retriever, TFIDFRetriever, SBERTRetriever
from reranker.models import StandardCrossEncoder, ModifiedCrossEncoder, get_tokenizer
from reranker.trainer import train_model, batch_rerank
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

    triplets = build_training_triplets(train_df, corpus_texts, corpus_id_map, train_rel,
                                       n_hard_negatives=config.DEFAULT_HARD_NEGS)

    val_rel = defaultdict(set)
    for _, row in val_df.iterrows():
        if row["is_duplicate"] == 1 and row["question2"] in corpus_id_map:
            val_rel[row["question1"]].add(corpus_id_map[row["question2"]])

    val_triplets = build_training_triplets(val_df, corpus_texts, corpus_id_map, val_rel,
                                           n_hard_negatives=config.DEFAULT_HARD_NEGS)

    print(f"\n  Corpus: {len(corpus_texts)} | Test queries: {len(test_queries)}")
    print(f"  Train triplets: {len(triplets)} | Val triplets: {len(val_triplets)}")

    return {"corpus_texts": corpus_texts, "corpus_id_map": corpus_id_map,
            "relevance": relevance, "test_queries": test_queries,
            "train_triplets": triplets, "val_triplets": val_triplets,
            "train_df": train_df, "val_df": val_df, "test_df": test_df}


def run_retrieval(data, top_k=None):
    top_k = top_k or config.DEFAULT_TOP_K
    print(f"\n{'='*70}\nSTEP 2: STAGE 1 RETRIEVAL (Top-{top_k})\n{'='*70}")
    corpus = data["corpus_texts"]
    queries = data["test_queries"]
    relevance = data["relevance"]

    retriever_classes = {
        "BM25": BM25Retriever,
        "TF-IDF": TFIDFRetriever,
        "SBERT": SBERTRetriever,
    }
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
    print(f"\n{'='*70}\nSTEP 3: TRANSFORMER RERANKER TRAINING & EVALUATION\n{'='*70}")
    corpus = data["corpus_texts"]
    queries = data["test_queries"]
    relevance = data["relevance"]
    triplets = data["train_triplets"]
    val_triplets = data["val_triplets"]
    candidates = retrieval_results[retriever_name]

    # ---- Transformer model configs ----
    model_configs = [
        {
            "name": "Standard-CE-DistilBERT",
            "model_class": StandardCrossEncoder,
            "model_type": "standard",
            "backbone": config.DISTILBERT_MODEL,
        },
        {
            "name": "Modified-CE-DistilBERT",
            "model_class": ModifiedCrossEncoder,
            "model_type": "modified",
            "backbone": config.DISTILBERT_MODEL,
        },
        {
            "name": "Standard-CE-BERT-base",
            "model_class": StandardCrossEncoder,
            "model_type": "standard",
            "backbone": config.BERT_MODEL,
        },
        {
            "name": "Modified-CE-BERT-base",
            "model_class": ModifiedCrossEncoder,
            "model_type": "modified",
            "backbone": config.BERT_MODEL,
        },
    ]

    reranker_results = {}
    reranker_times = {}
    all_histories = {}

    for cfg in model_configs:
        print(f"\n{'='*50}")
        print(f"  Training: {cfg['name']}")
        print(f"  Backbone: {cfg['backbone']}")
        print(f"{'='*50}")

        # Create model
        model = cfg["model_class"](
            model_name=cfg["backbone"],
            dropout=config.DEFAULT_DROPOUT
        )

        # Train
        t0 = time.time()
        trained_model, history = train_model(
            model,
            triplets,
            val_triplets,
            model_name_str=cfg["backbone"],
            model_type=cfg["model_type"],
            lr=config.DEFAULT_LR,
            batch_size=config.DEFAULT_BATCH_SIZE,
            epochs=config.DEFAULT_EPOCHS,
            alpha=config.DEFAULT_LOSS_WEIGHT,
            max_seq_len=config.DEFAULT_MAX_SEQ_LEN,
        )
        train_time = time.time() - t0
        all_histories[cfg["name"]] = history

        # Rerank
        tokenizer = get_tokenizer(cfg["backbone"])
        t0 = time.time()
        reranked = batch_rerank(
            trained_model, tokenizer, candidates, corpus,
            model_type=cfg["model_type"],
            max_seq_len=config.DEFAULT_MAX_SEQ_LEN,
        )
        rerank_time = time.time() - t0

        # Evaluate
        all_m = []
        for q in queries:
            ranked_ids = [r[0] for r in reranked[q]]
            all_m.append(evaluate_ranking(ranked_ids, relevance[q], k_values=config.EVAL_K_VALUES))
        agg = aggregate_metrics(all_m)
        reranker_results[cfg["name"]] = agg
        reranker_times[cfg["name"]] = {"train": train_time, "rerank": rerank_time}

        print(f"\n  RESULTS for {cfg['name']}:")
        print(f"    nDCG@10={agg.get('nDCG@10',0):.4f}  MRR@10={agg.get('MRR@10',0):.4f}  "
              f"P@1={agg.get('P@1',0):.4f}")
        print(f"    Train={train_time:.1f}s  Rerank={rerank_time:.1f}s")

        # Save model checkpoint
        save_path = os.path.join(config.MODEL_DIR, f"{cfg['name']}.pt")
        torch.save(trained_model.state_dict(), save_path)
        print(f"    Model saved to {save_path}")

        # Free GPU memory
        del trained_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return reranker_results, reranker_times, all_histories


def run_ablations(data, retrieval_results, retriever_name="TF-IDF"):
    print(f"\n{'='*70}\nSTEP 4: ABLATION STUDIES (Transformer)\n{'='*70}")
    corpus = data["corpus_texts"]
    queries = data["test_queries"]
    relevance = data["relevance"]
    triplets = data["train_triplets"]
    val_triplets = data["val_triplets"]
    candidates = retrieval_results[retriever_name]

    ablation_results = {}

    def train_and_eval(model, model_type, backbone, epochs=3, lr=None, alpha=None, triplets_in=None):
        """Helper: train a model and evaluate on test queries."""
        lr = lr or config.DEFAULT_LR
        alpha = alpha or config.DEFAULT_LOSS_WEIGHT
        triplets_in = triplets_in or triplets

        trained, _ = train_model(
            model, triplets_in, None,
            model_name_str=backbone, model_type=model_type,
            epochs=epochs, lr=lr, alpha=alpha,
            batch_size=config.DEFAULT_BATCH_SIZE,
            max_seq_len=config.DEFAULT_MAX_SEQ_LEN,
        )
        tokenizer = get_tokenizer(backbone)
        reranked = batch_rerank(trained, tokenizer, candidates, corpus,
                                model_type=model_type, max_seq_len=config.DEFAULT_MAX_SEQ_LEN)
        all_m = []
        for q in queries:
            ranked_ids = [r[0] for r in reranked[q]]
            all_m.append(evaluate_ranking(ranked_ids, relevance[q], k_values=config.EVAL_K_VALUES))

        del trained
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return aggregate_metrics(all_m)

    # ---- Ablation 1: Loss weight (alpha) ----
    # alpha=0 means CE only, alpha=1 means cosine only
    print("\n[ABLATION 1] Loss weight alpha (Modified-CE-DistilBERT)")
    alpha_results = {}
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        print(f"  alpha={alpha}")
        model = ModifiedCrossEncoder(model_name=config.DISTILBERT_MODEL)
        agg = train_and_eval(model, "modified", config.DISTILBERT_MODEL, epochs=3, alpha=alpha)
        alpha_results[alpha] = agg
        print(f"    nDCG@10={agg.get('nDCG@10',0):.4f}  MRR@10={agg.get('MRR@10',0):.4f}")
    ablation_results["loss_weight"] = alpha_results

    # ---- Ablation 2: Learning rate ----
    print("\n[ABLATION 2] Learning rate (Modified-CE-DistilBERT)")
    lr_results = {}
    for lr in [1e-5, 2e-5, 3e-5, 5e-5]:
        print(f"  lr={lr}")
        model = ModifiedCrossEncoder(model_name=config.DISTILBERT_MODEL)
        agg = train_and_eval(model, "modified", config.DISTILBERT_MODEL, epochs=3, lr=lr)
        lr_results[lr] = agg
        print(f"    nDCG@10={agg.get('nDCG@10',0):.4f}  MRR@10={agg.get('MRR@10',0):.4f}")
    ablation_results["learning_rate"] = lr_results

    # ---- Ablation 3: Training data size ----
    print("\n[ABLATION 3] Training data size (Modified-CE-DistilBERT)")
    size_results = {}
    for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
        n = max(100, int(len(triplets) * frac))
        print(f"  frac={frac} ({n} triplets)")
        model = ModifiedCrossEncoder(model_name=config.DISTILBERT_MODEL)
        agg = train_and_eval(model, "modified", config.DISTILBERT_MODEL, epochs=3,
                             triplets_in=triplets[:n])
        size_results[frac] = agg
        print(f"    nDCG@10={agg.get('nDCG@10',0):.4f}")
    ablation_results["training_size"] = size_results

    # ---- Ablation 4: Hard negatives ----
    print("\n[ABLATION 4] Hard negatives per query")
    neg_results = {}
    for n_neg in [1, 3, 5, 7]:
        print(f"  negatives={n_neg}")
        neg_triplets = build_training_triplets(
            data["train_df"], corpus, data["corpus_id_map"],
            defaultdict(set), n_hard_negatives=n_neg
        )
        if len(neg_triplets) < 50:
            neg_triplets = triplets[:500]
        model = ModifiedCrossEncoder(model_name=config.DISTILBERT_MODEL)
        agg = train_and_eval(model, "modified", config.DISTILBERT_MODEL, epochs=3,
                             triplets_in=neg_triplets)
        neg_results[n_neg] = agg
        print(f"    nDCG@10={agg.get('nDCG@10',0):.4f}")
    ablation_results["hard_negatives"] = neg_results

    # ---- Ablation 5: Top-K sweep (retrieval only, no reranker needed) ----
    print("\n[ABLATION 5] Top-K candidate pool size (BM25)")
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
                       retrieval_times, reranker_times, histories):
    print(f"\n{'='*70}\nGENERATING PLOTS\n{'='*70}")
    out = config.RESULTS_DIR
    plt.rcParams.update({"font.size": 11, "figure.facecolor": "white"})

    all_models = {}
    for k, v in retrieval_metrics.items():
        all_models[f"Retrieval-{k}"] = v
    all_models.update(reranker_results)

    # ---- Plot 1: Retrieval comparison ----
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
    axes[1].bar(models, [retrieval_times[m] for m in models],
                color=["#4C72B0", "#DD8452", "#55A868"][:len(models)])
    axes[1].set_title("Retrieval Time (seconds)"); axes[1].set_ylabel("Time (s)")
    axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, "01_retrieval_comparison.png"), dpi=150); plt.close()

    # ---- Plot 2: Full pipeline comparison ----
    fig, ax = plt.subplots(figsize=(16, 6))
    model_names = list(all_models.keys())
    metrics_show = ["nDCG@10", "MRR@10", "P@1", "MAP@10"]
    x = np.arange(len(model_names)); w = 0.2
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for i, m in enumerate(metrics_show):
        vals = [all_models[mn].get(m, 0) for mn in model_names]
        ax.bar(x + i * w, vals, w, label=m, color=colors[i])
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
    ax.set_title("Full Pipeline: Retrieval vs Retrieval + Transformer Reranking")
    ax.set_ylabel("Score"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, "02_full_comparison.png"), dpi=150); plt.close()

    # ---- Plot 3: Training loss curves ----
    if histories:
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, hist in histories.items():
            epochs = range(1, len(hist["train_loss"]) + 1)
            ax.plot(epochs, hist["train_loss"], "o-", label=f"{name} (train)", linewidth=2)
            if hist.get("val_loss"):
                ax.plot(epochs, hist["val_loss"], "s--", label=f"{name} (val)", linewidth=1.5)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss Curves")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out, "03_training_loss.png"), dpi=150); plt.close()

    # ---- Plot 4: Loss weight ablation ----
    if "loss_weight" in ablation_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        alphas = sorted(ablation_results["loss_weight"].keys())
        for metric in ["nDCG@10", "MRR@10", "P@1"]:
            ax.plot(alphas, [ablation_results["loss_weight"][a].get(metric, 0) for a in alphas],
                    "o-", label=metric, linewidth=2)
        ax.set_xlabel("Alpha (0 = CE only, 1 = Cosine only)")
        ax.set_ylabel("Score"); ax.set_title("Ablation: Joint Loss Weight (α)")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out, "04_ablation_loss_weight.png"), dpi=150); plt.close()

    # ---- Plot 5: Learning rate ablation ----
    if "learning_rate" in ablation_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        lrs = sorted(ablation_results["learning_rate"].keys())
        for metric in ["nDCG@10", "MRR@10"]:
            ax.plot([str(lr) for lr in lrs],
                    [ablation_results["learning_rate"][lr].get(metric, 0) for lr in lrs],
                    "o-", label=metric, linewidth=2)
        ax.set_xlabel("Learning Rate"); ax.set_ylabel("Score")
        ax.set_title("Ablation: Learning Rate"); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out, "05_ablation_learning_rate.png"), dpi=150); plt.close()

    # ---- Plot 6: Training size ablation ----
    if "training_size" in ablation_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        fracs = sorted(ablation_results["training_size"].keys())
        for metric in ["nDCG@10", "MRR@10"]:
            ax.plot([f*100 for f in fracs],
                    [ablation_results["training_size"][f].get(metric, 0) for f in fracs],
                    "o-", label=metric, linewidth=2)
        ax.set_xlabel("Training Data (%)"); ax.set_ylabel("Score")
        ax.set_title("Ablation: Training Data Size"); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out, "06_ablation_training_size.png"), dpi=150); plt.close()

    # ---- Plot 7: Hard negatives ablation ----
    if "hard_negatives" in ablation_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        negs = sorted(ablation_results["hard_negatives"].keys())
        for metric in ["nDCG@10", "MRR@10"]:
            ax.plot(negs, [ablation_results["hard_negatives"][n].get(metric, 0) for n in negs],
                    "o-", label=metric, linewidth=2)
        ax.set_xlabel("Hard Negatives per Query"); ax.set_ylabel("Score")
        ax.set_title("Ablation: Hard Negative Mining"); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out, "07_ablation_hard_negatives.png"), dpi=150); plt.close()

    # ---- Plot 8: Top-K sweep ----
    if "top_k" in ablation_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        ks = sorted(ablation_results["top_k"].keys())
        for metric in ["nDCG@10", "MRR@10"]:
            ax.plot(ks, [ablation_results["top_k"][k].get(metric, 0) for k in ks],
                    "o-", label=metric, linewidth=2)
        ax.set_xlabel("Top-K Candidates"); ax.set_ylabel("Score")
        ax.set_title("Ablation: Top-K Candidate Pool Size"); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out, "08_ablation_topk.png"), dpi=150); plt.close()

    # ---- Plot 9: Heatmap ----
    fig, ax = plt.subplots(figsize=(16, 8))
    model_names = list(all_models.keys())
    all_metric_names = ["nDCG@1","nDCG@5","nDCG@10","nDCG@20",
                        "MRR@5","MRR@10","P@1","P@5",
                        "Recall@10","Recall@20","MAP@10","MAP@20"]
    matrix = np.array([[all_models[mn].get(m, 0) for m in all_metric_names] for mn in model_names])
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(len(all_metric_names)))
    ax.set_xticklabels(all_metric_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)
    for i in range(len(model_names)):
        for j in range(len(all_metric_names)):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im); ax.set_title("Comprehensive Metrics Heatmap")
    plt.tight_layout(); plt.savefig(os.path.join(out, "09_metrics_heatmap.png"), dpi=150); plt.close()

    # ---- Plot 10: Inference time comparison ----
    if reranker_times:
        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(reranker_times.keys())
        train_t = [reranker_times[n]["train"] for n in names]
        rerank_t = [reranker_times[n]["rerank"] for n in names]
        x = np.arange(len(names)); w = 0.35
        ax.bar(x - w/2, train_t, w, label="Training Time", color="#4C72B0")
        ax.bar(x + w/2, rerank_t, w, label="Reranking Time", color="#DD8452")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Time (seconds)"); ax.set_title("Latency: Training vs Inference Time")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out, "10_latency_comparison.png"), dpi=150); plt.close()

    print(f"  10 plots saved to {out}/")


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
    print("=" * 70)
    print("TWO-STAGE PASSAGE RANKING — TRANSFORMER EXPERIMENTS")
    print(f"Device: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # Clear old results
    for f in os.listdir(config.RESULTS_DIR):
        os.remove(os.path.join(config.RESULTS_DIR, f))
    print("[CLEANUP] Old results cleared.\n")

    data = prepare_data()
    retrieval_results, retrieval_metrics, retrieval_times = run_retrieval(data)
    reranker_results, reranker_times, histories = train_and_evaluate_rerankers(
        data, retrieval_results, retriever_name="TF-IDF"
    )
    ablation_results = run_ablations(data, retrieval_results, retriever_name="TF-IDF")
    save_results(retrieval_metrics, reranker_results, ablation_results)
    generate_all_plots(retrieval_metrics, reranker_results, ablation_results,
                       retrieval_times, reranker_times, histories)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"COMPLETE — {elapsed/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
