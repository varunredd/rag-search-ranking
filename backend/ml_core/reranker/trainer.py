"""Training and inference helpers for optional transformer rerankers."""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import ml_core.config as config
from ml_core.reranker.models import JointLoss, get_tokenizer

try:
    from transformers import get_linear_schedule_with_warmup
except Exception:
    get_linear_schedule_with_warmup = None


class PairwiseDataset(Dataset):
    def __init__(self, triplets, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length or config.DEFAULT_MAX_SEQ_LEN
        self.pairs = []
        for t in triplets:
            self.pairs.append((t['query'], t['positive'], 1))
            self.pairs.append((t['query'], t['negative'], 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query, doc, label = self.pairs[idx]
        encoded = self.tokenizer(query, doc, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        query_encoded = self.tokenizer(query, max_length=self.max_length // 2, padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'query_input_ids': query_encoded['input_ids'].squeeze(0),
            'query_attention_mask': query_encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
        }


def train_model(model, train_triplets, val_triplets=None, model_name_str='distilbert', model_type='standard', lr=None, batch_size=None, epochs=None, alpha=None, max_seq_len=None):
    if get_linear_schedule_with_warmup is None:
        raise RuntimeError('transformers is not installed, so transformer training is unavailable in demo mode.')
    lr = lr or config.DEFAULT_LR
    batch_size = batch_size or config.DEFAULT_BATCH_SIZE
    epochs = epochs or config.DEFAULT_EPOCHS
    alpha = alpha or config.DEFAULT_LOSS_WEIGHT
    max_seq_len = max_seq_len or config.DEFAULT_MAX_SEQ_LEN

    tokenizer = get_tokenizer(model_name_str)
    train_loader = DataLoader(PairwiseDataset(train_triplets, tokenizer, max_seq_len), batch_size=batch_size, shuffle=True)
    model = model.to(config.DEVICE)
    criterion = JointLoss(alpha=alpha) if model_type == 'modified' else torch.nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    history = {'train_loss': [], 'val_loss': []}

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc='Training'):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['label'].to(config.DEVICE)
            optimizer.zero_grad()
            if model_type == 'modified':
                query_ids = batch['query_input_ids'].to(config.DEVICE)
                query_mask = batch['query_attention_mask'].to(config.DEVICE)
                logits, cosine_sim, proj_pair = model(input_ids, attention_mask, query_ids, query_mask)
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
        history['train_loss'].append(total_loss / max(1, len(train_loader)))
    return model, history


def rerank_candidates(model, tokenizer, query, candidate_texts, candidate_ids, model_type='standard', max_seq_len=None):
    max_seq_len = max_seq_len or config.DEFAULT_MAX_SEQ_LEN
    model.eval()
    scores = []
    with torch.no_grad():
        for doc_text, doc_id in zip(candidate_texts, candidate_ids):
            encoded = tokenizer(query, doc_text, max_length=max_seq_len, padding='max_length', truncation=True, return_tensors='pt').to(config.DEVICE)
            logits, *_ = model(encoded['input_ids'], encoded['attention_mask'])
            score = torch.sigmoid(logits).item()
            scores.append((doc_id, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
