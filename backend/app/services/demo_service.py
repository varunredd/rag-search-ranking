from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

import ml_core.config as core_config
from ml_core.data_prep import load_data, split_data, build_training_triplets
from ml_core.retrieval.retrievers import get_retriever
from ml_core.reranker.models import StandardCrossEncoder, ModifiedCrossEncoder, get_tokenizer
from ml_core.reranker.trainer import rerank_candidates

from app.core_paths import ML_MODELS_DIR


class DemoEngine:
    CHECKPOINTS = {
        'standard-ce-distilbert': {
            'label': 'Standard CE (DistilBERT checkpoint)',
            'filename': 'Standard-CE-DistilBERT.pt',
            'model_class': StandardCrossEncoder,
            'model_type': 'standard',
            'backbone': core_config.DISTILBERT_MODEL,
        },
        'modified-ce-distilbert': {
            'label': 'Modified CE (DistilBERT checkpoint)',
            'filename': 'Modified-CE-DistilBERT.pt',
            'model_class': ModifiedCrossEncoder,
            'model_type': 'modified',
            'backbone': core_config.DISTILBERT_MODEL,
        },
        'standard-ce-bert-base': {
            'label': 'Standard CE (BERT-base checkpoint)',
            'filename': 'Standard-CE-BERT-base.pt',
            'model_class': StandardCrossEncoder,
            'model_type': 'standard',
            'backbone': core_config.BERT_MODEL,
        },
        'modified-ce-bert-base': {
            'label': 'Modified CE (BERT-base checkpoint)',
            'filename': 'Modified-CE-BERT-base.pt',
            'model_class': ModifiedCrossEncoder,
            'model_type': 'modified',
            'backbone': core_config.BERT_MODEL,
        },
    }

    def __init__(self):
        self.initialized = False
        self.dataset_mode = 'synthetic'
        self.df = None
        self.corpus_texts: List[str] = []
        self.corpus_id_map: Dict[str, int] = {}
        self.duplicates_lookup = defaultdict(list)
        self.retrievers = {}
        self.rerankers = {}

    def initialize(self):
        if self.initialized:
            return
        self.df = load_data()
        self.dataset_mode = 'real-kaggle' if (not core_config.SYNTHETIC_DATA and Path(core_config.KAGGLE_CSV_PATH).exists()) else 'synthetic'
        train_df, _, _ = split_data(self.df)
        combined = self.df['question2'].astype(str).unique().tolist()
        self.corpus_texts = combined
        self.corpus_id_map = {text: idx for idx, text in enumerate(self.corpus_texts)}

        relevance = defaultdict(set)
        for _, row in train_df.iterrows():
            q1 = str(row['question1'])
            q2 = str(row['question2'])
            if int(row['is_duplicate']) == 1 and q2 in self.corpus_id_map:
                relevance[q1].add(self.corpus_id_map[q2])
                self.duplicates_lookup[q1].append(q2)

        for _, row in self.df.iterrows():
            q1 = str(row['question1'])
            q2 = str(row['question2'])
            if int(row['is_duplicate']) == 1:
                self.duplicates_lookup[q1].append(q2)

        self.initialized = True

    def has_any_checkpoint(self) -> bool:
        return any((ML_MODELS_DIR / cfg['filename']).exists() for cfg in self.CHECKPOINTS.values())

    def available_retrievers(self):
        return [
            {'label': 'BM25', 'value': 'bm25'},
            {'label': 'TF-IDF', 'value': 'tfidf'},
            {'label': 'SBERT', 'value': 'sbert'},
        ]

    def available_rerankers(self):
        items = [{'label': 'No reranker (retrieval only)', 'value': 'none'}]
        for value, cfg in self.CHECKPOINTS.items():
            if (ML_MODELS_DIR / cfg['filename']).exists():
                items.append({'label': cfg['label'], 'value': value})
        return items

    def get_retriever(self, name: str):
        self.initialize()
        key = name.lower()
        if key not in self.retrievers:
            self.retrievers[key] = get_retriever(key, self.corpus_texts)
        return self.retrievers[key]

    def get_reranker(self, name: str):
        self.initialize()
        key = name.lower()
        if key == 'none':
            return None
        if key in self.rerankers:
            return self.rerankers[key]
        if key not in self.CHECKPOINTS:
            raise ValueError(f'Unknown reranker: {name}')

        cfg = self.CHECKPOINTS[key]
        checkpoint_path = ML_MODELS_DIR / cfg['filename']
        if not checkpoint_path.exists():
            raise ValueError(
                f"Checkpoint '{cfg['filename']}' not found in ml_core/models. Copy your trained .pt file into that folder to enable this reranker."
            )

        model = cfg['model_class'](model_name=cfg['backbone'], dropout=core_config.DEFAULT_DROPOUT)
        state = torch.load(checkpoint_path, map_location=core_config.DEVICE)
        model.load_state_dict(state)
        model = model.to(core_config.DEVICE)
        model.eval()
        tokenizer = get_tokenizer(cfg['backbone'])
        payload = {'model': model, 'tokenizer': tokenizer, **cfg}
        self.rerankers[key] = payload
        return payload

    def search(self, query: str, retriever_name: str, reranker_name: str, top_k: int):
        self.initialize()
        retriever = self.get_retriever(retriever_name)
        retrieval = retriever.retrieve(query, top_k=top_k)
        truth_set = set(self.duplicates_lookup.get(query, []))

        retrieval_results = []
        for rank, (doc_id, score) in enumerate(retrieval, start=1):
            text = self.corpus_texts[doc_id]
            retrieval_results.append({
                'doc_id': doc_id,
                'text': text,
                'score': float(score),
                'rank': rank,
                'is_ground_truth': text in truth_set,
            })

        reranked_results = [dict(item, previous_rank=item['rank'], rank_delta=0) for item in retrieval_results]
        note = 'Showing retrieval-only results. Add a checkpoint to ml_core/models to enable real reranking in the live demo.'

        if reranker_name.lower() != 'none':
            reranker = self.get_reranker(reranker_name)
            candidate_texts = [self.corpus_texts[item['doc_id']] for item in retrieval_results]
            candidate_ids = [item['doc_id'] for item in retrieval_results]
            reranked = rerank_candidates(
                reranker['model'],
                reranker['tokenizer'],
                query,
                candidate_texts,
                candidate_ids,
                model_type=reranker['model_type'],
                max_seq_len=core_config.DEFAULT_MAX_SEQ_LEN,
                alpha=core_config.DEFAULT_LOSS_WEIGHT,
            )
            previous_ranks = {item['doc_id']: item['rank'] for item in retrieval_results}
            reranked_results = []
            for rank, (doc_id, score) in enumerate(reranked, start=1):
                text = self.corpus_texts[doc_id]
                previous_rank = previous_ranks.get(doc_id, rank)
                reranked_results.append({
                    'doc_id': doc_id,
                    'text': text,
                    'score': float(score),
                    'rank': rank,
                    'previous_rank': previous_rank,
                    'rank_delta': previous_rank - rank,
                    'is_ground_truth': text in truth_set,
                })
            note = f"Live reranking loaded from checkpoint: {reranker['filename']}."

        return {
            'query': query,
            'retriever': retriever_name,
            'reranker': reranker_name,
            'dataset_mode': self.dataset_mode,
            'note': note,
            'ground_truths': list(dict.fromkeys(self.duplicates_lookup.get(query, [])))[:5],
            'retrieval_results': retrieval_results,
            'reranked_results': reranked_results,
        }


engine = DemoEngine()
