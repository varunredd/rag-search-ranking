"""
Stage 2: Cross-Encoder Reranker Models
- StandardCrossEncoder: classification head only (sigmoid CE loss)
- ModifiedCrossEncoder: dual scoring (logits + CLS cosine similarity), joint loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import config


class StandardCrossEncoder(nn.Module):
    """
    Standard cross-encoder: encodes query-doc pair, uses [CLS] for binary classification.
    Loss: Sigmoid Cross-Entropy only.
    """

    def __init__(self, model_name=None, dropout=None):
        super().__init__()
        model_name = model_name or config.DISTILBERT_MODEL
        dropout = dropout if dropout is not None else config.DEFAULT_DROPOUT

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] token (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output).squeeze(-1)
        return logits, cls_output

    def get_score(self, input_ids, attention_mask):
        """Get relevance score for inference."""
        logits, _ = self.forward(input_ids, attention_mask)
        return torch.sigmoid(logits)


class ModifiedCrossEncoder(nn.Module):
    """
    Modified cross-encoder from the primary paper:
    - Encodes query-doc pair jointly
    - Uses DUAL scoring: classification logits + cosine similarity of [CLS] embeddings
    - Joint loss: sigmoid CE + cosine similarity loss
    """

    def __init__(self, model_name=None, dropout=None):
        super().__init__()
        model_name = model_name or config.DISTILBERT_MODEL
        dropout = dropout if dropout is not None else config.DEFAULT_DROPOUT

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

        # Projection head for cosine similarity scoring
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask,
                input_ids_query=None, attention_mask_query=None):
        """
        Forward pass.
        For training: also pass query-only inputs to compute cosine similarity.
        For inference: only pass concatenated query-doc pair.
        """
        # Encode concatenated query-doc pair
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_pair = outputs.last_hidden_state[:, 0, :]
        cls_pair = self.dropout(cls_pair)

        # Classification logits
        logits = self.classifier(cls_pair).squeeze(-1)

        # Projected embedding for cosine similarity
        proj_pair = self.projection(cls_pair)

        if input_ids_query is not None:
            # Encode query alone for cosine similarity
            query_outputs = self.encoder(
                input_ids=input_ids_query, attention_mask=attention_mask_query
            )
            cls_query = query_outputs.last_hidden_state[:, 0, :]
            proj_query = self.projection(cls_query)
            cosine_sim = F.cosine_similarity(proj_pair, proj_query, dim=-1)
            return logits, cosine_sim, proj_pair
        else:
            return logits, None, proj_pair

    def get_score(self, input_ids, attention_mask, alpha=None):
        """
        Get combined relevance score for inference.
        score = alpha * sigmoid(logit) + (1 - alpha) * normalized_cosine
        """
        alpha = alpha if alpha is not None else config.DEFAULT_LOSS_WEIGHT
        logits, _, proj = self.forward(input_ids, attention_mask)
        sigmoid_score = torch.sigmoid(logits)
        # For inference without separate query encoding, use sigmoid score only
        # (cosine sim requires separate query encoding)
        return sigmoid_score


class JointLoss(nn.Module):
    """
    Joint loss = alpha * CosineEmbeddingLoss + (1 - alpha) * BCEWithLogitsLoss
    """

    def __init__(self, alpha=None):
        super().__init__()
        self.alpha = alpha if alpha is not None else config.DEFAULT_LOSS_WEIGHT
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=0.0)

    def forward(self, logits, cosine_sim, labels, proj_pair=None, proj_query=None):
        labels_float = labels.float()

        # BCE loss on classification logits
        loss_bce = self.bce_loss(logits, labels_float)

        if cosine_sim is not None:
            # Cosine embedding loss: +1 for relevant, -1 for irrelevant
            cosine_targets = 2 * labels_float - 1  # map 0->-1, 1->+1
            loss_cosine = self.cosine_loss(
                proj_pair, proj_query, cosine_targets
            ) if proj_pair is not None and proj_query is not None else \
                F.mse_loss(cosine_sim, labels_float)

            return (1 - self.alpha) * loss_bce + self.alpha * loss_cosine
        else:
            return loss_bce


class SklearnCrossEncoder:
    """
    Sklearn-based cross-encoder proxy for environments without HuggingFace access.
    Uses TF-IDF features + MLP classifier to simulate cross-encoder behavior.
    Replace with transformer-based models when running locally.
    """

    def __init__(self, hidden_size=256, dropout=0.1, model_type="standard"):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neural_network import MLPClassifier
        import scipy.sparse as sp

        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.mlp = None
        self._fitted = False

    def fit(self, triplets, epochs=5):
        from sklearn.neural_network import MLPClassifier
        import scipy.sparse as sp

        # Build features from query-doc pairs
        texts = []
        labels = []
        for t in triplets:
            texts.append(t["query"] + " [SEP] " + t["positive"])
            labels.append(1)
            texts.append(t["query"] + " [SEP] " + t["negative"])
            labels.append(0)

        X = self.vectorizer.fit_transform(texts)
        y = labels

        self.mlp = MLPClassifier(
            hidden_layer_sizes=(self.hidden_size, 128),
            max_iter=epochs * 10,
            random_state=config.RANDOM_SEED,
            early_stopping=True,
            validation_fraction=0.1,
            alpha=0.001,
            learning_rate_init=0.001,
        )
        self.mlp.fit(X, y)
        self._fitted = True
        train_acc = self.mlp.score(X, y)
        print(f"  [{self.model_type}] Train accuracy: {train_acc:.4f}")
        return self

    def predict_score(self, query, doc):
        text = query + " [SEP] " + doc
        X = self.vectorizer.transform([text])
        prob = self.mlp.predict_proba(X)[0][1]  # P(relevant)
        return float(prob)

    def rerank(self, query, candidate_texts, candidate_ids):
        scores = []
        for doc_text, doc_id in zip(candidate_texts, candidate_ids):
            score = self.predict_score(query, doc_text)
            scores.append((doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


def get_tokenizer(model_name=None):
    model_name = model_name or config.DISTILBERT_MODEL
    return AutoTokenizer.from_pretrained(model_name)
