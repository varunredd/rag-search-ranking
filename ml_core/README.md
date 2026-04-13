# Modified Cross Encoder for Two-Stage Passage Ranking

**ITCS 5154 — Applied Machine Learning, Spring 2026**  
**Varun Reddy Gutha (801448125)**

## Overview

This project implements and evaluates a two-stage passage ranking system based on the paper *"Modified Cross Encoder for Two-Stage Passage Ranking"* (ICTIS 2025). The pipeline combines fast first-stage retrieval with accurate cross-encoder re-ranking.

## Architecture

### Stage 1: Candidate Retrieval
- **BM25** — sparse lexical retrieval
- **TF-IDF + Cosine Similarity** — sparse vector retrieval  
- **Sentence-BERT (all-MiniLM-L6-v2)** — dense bi-encoder retrieval

### Stage 2: Cross-Encoder Re-ranking
- **Standard Cross-Encoder (DistilBERT)** — classification head only, BCE loss
- **Modified Cross-Encoder (DistilBERT)** — dual scoring (logits + CLS cosine similarity), joint loss
- **Standard Cross-Encoder (BERT-base)** — larger backbone comparison

### Ablation Studies
1. **Loss weight (α)** — CE-only vs cosine-only vs joint loss
2. **Hard negatives count** — 1 vs 3 vs 5 negatives per query
3. **Learning rate sweep** — 1e-5, 2e-5, 5e-5
4. **Top-K sweep** — effect of candidate pool size on ranking quality

## Dataset

**Quora Question Pairs (QQP)** — Kaggle  
- question1 = query, question2 = candidate document  
- is_duplicate = relevance label  
- 70/15/15 train/val/test split

### Using Real Kaggle Data
1. Download `train.csv` from https://www.kaggle.com/c/quora-question-pairs
2. Place it in `data/train.csv`
3. Set `SYNTHETIC_DATA = False` in `config.py`

## Evaluation Metrics
- nDCG@k (k = 1, 5, 10, 20)
- MRR@k, MAP@k, Precision@k, Recall@k

## Running

```bash
python run_experiments.py
```

Results and plots are saved to `results/`.

## Project Structure

```
project/
├── config.py                    # All hyperparameters
├── data_prep.py                 # Data loading, splitting, triplet construction
├── retrieval/
│   └── retrievers.py            # BM25, TF-IDF, SBERT retrievers
├── reranker/
│   ├── models.py                # Standard & Modified Cross-Encoder
│   └── trainer.py               # Training & inference
├── evaluation/
│   └── metrics.py               # nDCG, MRR, MAP, Recall, Precision
├── run_experiments.py           # Main experiment runner
├── results/                     # Output plots & CSV tables
└── README.md
```

## References

1. Panchumarthi et al., "Modified Cross Encoder for Two-Stage Passage Ranking", ICTIS 2025
2. Nogueira & Cho, "Passage Re-ranking with BERT", arXiv 2019
3. Reimers & Gurevych, "Sentence-BERT", arXiv 2019
