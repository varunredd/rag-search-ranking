"""
Stage 2: Cross-Encoder Reranker Models
Transformer classes are optional at runtime.
A lightweight sklearn proxy is included for self-contained demo mode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_core.config as config

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:  # optional dependency
    AutoModel = None
    AutoTokenizer = None


class StandardCrossEncoder(nn.Module):
    def __init__(self, model_name=None, dropout=None):
        super().__init__()
        if AutoModel is None:
            raise RuntimeError('transformers is not installed.')
        model_name = model_name or config.DISTILBERT_MODEL
        dropout = dropout if dropout is not None else config.DEFAULT_DROPOUT
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output).squeeze(-1)
        return logits, cls_output

    def get_score(self, input_ids, attention_mask):
        logits, _ = self.forward(input_ids, attention_mask)
        return torch.sigmoid(logits)


class ModifiedCrossEncoder(nn.Module):
    def __init__(self, model_name=None, dropout=None):
        super().__init__()
        if AutoModel is None:
            raise RuntimeError('transformers is not installed.')
        model_name = model_name or config.DISTILBERT_MODEL
        dropout = dropout if dropout is not None else config.DEFAULT_DROPOUT
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask, input_ids_query=None, attention_mask_query=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_pair = outputs.last_hidden_state[:, 0, :]
        cls_pair = self.dropout(cls_pair)
        logits = self.classifier(cls_pair).squeeze(-1)
        proj_pair = self.projection(cls_pair)
        if input_ids_query is not None:
            query_outputs = self.encoder(input_ids=input_ids_query, attention_mask=attention_mask_query)
            cls_query = query_outputs.last_hidden_state[:, 0, :]
            proj_query = self.projection(cls_query)
            cosine_sim = F.cosine_similarity(proj_pair, proj_query, dim=-1)
            return logits, cosine_sim, proj_pair
        return logits, None, proj_pair

    def get_score(self, input_ids, attention_mask, alpha=None):
        logits, _, _ = self.forward(input_ids, attention_mask)
        return torch.sigmoid(logits)


class JointLoss(nn.Module):
    def __init__(self, alpha=None):
        super().__init__()
        self.alpha = alpha if alpha is not None else config.DEFAULT_LOSS_WEIGHT
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=0.0)

    def forward(self, logits, cosine_sim, labels, proj_pair=None, proj_query=None):
        labels_float = labels.float()
        loss_bce = self.bce_loss(logits, labels_float)
        if cosine_sim is not None:
            cosine_targets = 2 * labels_float - 1
            loss_cosine = self.cosine_loss(proj_pair, proj_query, cosine_targets) if proj_pair is not None and proj_query is not None else F.mse_loss(cosine_sim, labels_float)
            return (1 - self.alpha) * loss_bce + self.alpha * loss_cosine
        return loss_bce


class SklearnCrossEncoder:
    """Lightweight demo proxy using TF-IDF pair features + LogisticRegression."""

    def __init__(self, hidden_size=256, dropout=0.1, model_type='standard'):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=6000, stop_words='english', ngram_range=(1, 2))
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.model = None
        self._fitted = False

    def fit(self, triplets, epochs=5):
        from sklearn.linear_model import LogisticRegression
        texts = []
        labels = []
        for t in triplets:
            texts.append(t['query'] + ' [SEP] ' + t['positive'])
            labels.append(1)
            texts.append(t['query'] + ' [SEP] ' + t['negative'])
            labels.append(0)
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(max_iter=400, random_state=config.RANDOM_SEED, n_jobs=None)
        self.model.fit(X, labels)
        self._fitted = True
        return self

    def predict_score(self, query, doc):
        text = query + ' [SEP] ' + doc
        X = self.vectorizer.transform([text])
        return float(self.model.predict_proba(X)[0][1])

    def rerank(self, query, candidate_texts, candidate_ids):
        scores = []
        for doc_text, doc_id in zip(candidate_texts, candidate_ids):
            scores.append((doc_id, self.predict_score(query, doc_text)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

def get_tokenizer(model_name=None):
    if AutoTokenizer is None:
        raise RuntimeError('transformers is not installed.')
    model_name = model_name or config.DISTILBERT_MODEL
    return AutoTokenizer.from_pretrained(model_name)
