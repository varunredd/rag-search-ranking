"""
Training and Inference for Cross-Encoder Rerankers
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
from tqdm import tqdm

import config
from reranker.models import (
    StandardCrossEncoder, ModifiedCrossEncoder,
    JointLoss, get_tokenizer
)


class PairwiseDataset(Dataset):
    """Dataset for cross-encoder training with query-doc pairs."""

    def __init__(self, triplets, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length or config.DEFAULT_MAX_SEQ_LEN
        # Flatten triplets into (query, doc, label) pairs
        self.pairs = []
        for t in triplets:
            self.pairs.append((t["query"], t["positive"], 1))
            self.pairs.append((t["query"], t["negative"], 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query, doc, label = self.pairs[idx]
        # Encode pair
        encoded = self.tokenizer(
            query, doc,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # Encode query alone (for modified cross-encoder cosine sim)
        query_encoded = self.tokenizer(
            query,
            max_length=self.max_length // 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "query_input_ids": query_encoded["input_ids"].squeeze(0),
            "query_attention_mask": query_encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def train_model(model, train_triplets, val_triplets=None,
                model_name_str="distilbert", model_type="standard",
                lr=None, batch_size=None, epochs=None, alpha=None,
                max_seq_len=None):
    """
    Train a cross-encoder model.

    Args:
        model: StandardCrossEncoder or ModifiedCrossEncoder
        train_triplets: list of {query, positive, negative}
        val_triplets: optional validation triplets
        model_name_str: backbone name for tokenizer
        model_type: "standard" or "modified"
        lr, batch_size, epochs, alpha, max_seq_len: hyperparameters
    """
    lr = lr or config.DEFAULT_LR
    batch_size = batch_size or config.DEFAULT_BATCH_SIZE
    epochs = epochs or config.DEFAULT_EPOCHS
    alpha = alpha or config.DEFAULT_LOSS_WEIGHT
    max_seq_len = max_seq_len or config.DEFAULT_MAX_SEQ_LEN

    tokenizer = get_tokenizer(model_name_str)
    train_dataset = PairwiseDataset(train_triplets, tokenizer, max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_triplets:
        val_dataset = PairwiseDataset(val_triplets, tokenizer, max_seq_len)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = model.to(config.DEVICE)

    if model_type == "modified":
        criterion = JointLoss(alpha=alpha)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    history = {"train_loss": [], "val_loss": []}

    print(f"\n[TRAIN] Model: {model_type}, LR: {lr}, BS: {batch_size}, "
          f"Epochs: {epochs}, Alpha: {alpha}")
    print(f"[TRAIN] Training on {len(train_dataset)} pairs, {len(train_loader)} batches/epoch")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            labels = batch["label"].to(config.DEVICE)

            optimizer.zero_grad()

            if model_type == "modified":
                query_ids = batch["query_input_ids"].to(config.DEVICE)
                query_mask = batch["query_attention_mask"].to(config.DEVICE)
                logits, cosine_sim, proj_pair = model(
                    input_ids, attention_mask, query_ids, query_mask
                )
                # Get query projection for cosine loss
                with torch.no_grad():
                    q_out = model.encoder(input_ids=query_ids, attention_mask=query_mask)
                    proj_query = model.projection(q_out.last_hidden_state[:, 0, :])
                loss = criterion(logits, cosine_sim, labels, proj_pair, proj_query)
            else:
                logits, _ = model(input_ids, attention_mask)
                loss = criterion(logits, labels.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # Validation
        val_loss_avg = 0
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(config.DEVICE)
                    attention_mask = batch["attention_mask"].to(config.DEVICE)
                    labels = batch["label"].to(config.DEVICE)

                    if model_type == "modified":
                        query_ids = batch["query_input_ids"].to(config.DEVICE)
                        query_mask = batch["query_attention_mask"].to(config.DEVICE)
                        logits, cosine_sim, proj_pair = model(
                            input_ids, attention_mask, query_ids, query_mask
                        )
                        q_out = model.encoder(input_ids=query_ids, attention_mask=query_mask)
                        proj_query = model.projection(q_out.last_hidden_state[:, 0, :])
                        loss = criterion(logits, cosine_sim, labels, proj_pair, proj_query)
                    else:
                        logits, _ = model(input_ids, attention_mask)
                        loss = criterion(logits, labels.float())

                    val_loss += loss.item()

            val_loss_avg = val_loss / len(val_loader)
            history["val_loss"].append(val_loss_avg)

        print(f"  Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, "
              f"Val Loss = {val_loss_avg:.4f}")

    return model, history


def rerank_candidates(model, tokenizer, query, candidate_texts, candidate_ids,
                      model_type="standard", max_seq_len=None, alpha=None):
    """
    Rerank candidate documents for a single query.

    Returns: list of (doc_id, score) sorted by score descending.
    """
    max_seq_len = max_seq_len or config.DEFAULT_MAX_SEQ_LEN
    alpha = alpha or config.DEFAULT_LOSS_WEIGHT
    model.eval()

    scores = []
    with torch.no_grad():
        for doc_text, doc_id in zip(candidate_texts, candidate_ids):
            encoded = tokenizer(
                query, doc_text,
                max_length=max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(config.DEVICE)

            if model_type == "modified":
                logits, _, _ = model(encoded["input_ids"], encoded["attention_mask"])
                score = torch.sigmoid(logits).item()
            else:
                logits, _ = model(encoded["input_ids"], encoded["attention_mask"])
                score = torch.sigmoid(logits).item()

            scores.append((doc_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def batch_rerank(model, tokenizer, retrieval_results, corpus_texts,
                 model_type="standard", max_seq_len=None):
    """
    Rerank retrieval results for multiple queries.

    Args:
        retrieval_results: dict {query: [(doc_id, retrieval_score), ...]}
        corpus_texts: list of all corpus texts

    Returns: dict {query: [(doc_id, rerank_score), ...]}
    """
    reranked = {}
    queries = list(retrieval_results.keys())
    for q in tqdm(queries, desc=f"Reranking ({model_type})"):
        candidates = retrieval_results[q]
        cand_ids = [c[0] for c in candidates]
        cand_texts = [corpus_texts[c[0]] for c in candidates]
        reranked[q] = rerank_candidates(
            model, tokenizer, q, cand_texts, cand_ids,
            model_type=model_type, max_seq_len=max_seq_len
        )
    return reranked
