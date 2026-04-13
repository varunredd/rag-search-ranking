"""
Configuration for Two-Stage Passage Ranking Project
ITCS 5154 — Applied Machine Learning, Spring 2026
"""

import os
import torch

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Dataset
KAGGLE_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
SYNTHETIC_DATA = False
NUM_SYNTHETIC_PAIRS = 15000
RANDOM_SEED = 42

# Data splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Number of queries to evaluate (subset for speed)
NUM_EVAL_QUERIES = 100

# Retrieval
TOP_K_VALUES = [10, 20, 50, 100]
DEFAULT_TOP_K = 20

# Reranker Training
LEARNING_RATES = [1e-5, 2e-5, 3e-5, 5e-5]
BATCH_SIZES = [16, 32, 64]
EPOCHS_LIST = [3, 5, 10]
DROPOUT_RATES = [0.1, 0.2, 0.3]
MAX_SEQ_LENGTHS = [64, 128]
HARD_NEGATIVES_PER_QUERY = [1, 3, 5]
LOSS_WEIGHTS = [0.0, 0.3, 0.5, 0.7, 1.0]

# Default training config
DEFAULT_LR = 2e-5
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 3
DEFAULT_DROPOUT = 0.1
DEFAULT_MAX_SEQ_LEN = 64
DEFAULT_HARD_NEGS = 1
DEFAULT_LOSS_WEIGHT = 0.5

# Model backbones
DISTILBERT_MODEL = "distilbert-base-uncased"
BERT_MODEL = "bert-base-uncased"
MINILM_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SBERT_MODEL = "all-MiniLM-L6-v2"

# Evaluation
EVAL_K_VALUES = [1, 5, 10, 20]

# Device — uses Mac GPU if available, then CUDA, then CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"