"""
Data Preparation — Generates realistic QQP-format data with lexical overlap
so BM25/TF-IDF can retrieve duplicates, and rerankers have signal to work with.
"""

import os, random, re
import pandas as pd
import numpy as np
from collections import defaultdict
import ml_core.config as config


def generate_synthetic_qqp(n_pairs=15000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    topics = [
        "Python", "Java", "JavaScript", "machine learning", "deep learning",
        "data science", "web development", "databases", "SQL", "cloud computing",
        "AWS", "Docker", "Kubernetes", "React", "Angular", "Node.js", "Django",
        "TensorFlow", "PyTorch", "neural networks", "NLP", "computer vision",
        "blockchain", "cybersecurity", "data engineering", "API design",
        "microservices", "DevOps", "Linux", "Git", "agile", "Scrum",
        "statistics", "probability", "calculus", "linear algebra",
        "physics", "chemistry", "biology", "economics", "finance",
        "investing", "stock market", "real estate", "marketing", "SEO",
        "photography", "cooking", "fitness", "yoga", "meditation",
        "travel", "hiking", "gaming", "music", "guitar", "piano",
    ]

    # Templates where duplicates SHARE key words (realistic for BM25/TF-IDF)
    dup_templates = [
        ("What is the best way to learn {t}?", "What is the best method to learn {t}?"),
        ("How do I learn {t} quickly?", "How can I learn {t} fast?"),
        ("What are the best {t} tutorials?", "Where can I find good {t} tutorials?"),
        ("How long does it take to learn {t}?", "How much time to learn {t}?"),
        ("Is {t} worth learning in 2025?", "Should I learn {t} in 2025?"),
        ("What are the best books for {t}?", "What books should I read for {t}?"),
        ("How do I get a job in {t}?", "How to find a job in {t}?"),
        ("What skills do I need for {t}?", "What are the required skills for {t}?"),
        ("{t} vs {t2}: which is better?", "Which one is better: {t} or {t2}?"),
        ("What salary can I expect with {t} skills?", "How much does a {t} professional earn?"),
        ("What are common {t} interview questions?", "What {t} questions are asked in interviews?"),
        ("How to prepare for a {t} interview?", "What is the best way to prepare for {t} interviews?"),
        ("What are the advantages of {t}?", "What are the benefits of using {t}?"),
        ("What are the disadvantages of {t}?", "What are the drawbacks of {t}?"),
        ("How to start a career in {t}?", "How do I begin a career in {t}?"),
        ("What certifications are available for {t}?", "Which {t} certifications should I get?"),
        ("What tools are commonly used in {t}?", "What are the most popular {t} tools?"),
        ("Is {t} difficult for beginners?", "Is {t} hard for someone just starting out?"),
        ("What projects can I build with {t}?", "What {t} projects should I work on?"),
        ("How to become an expert in {t}?", "How do I master {t}?"),
    ]

    non_dup_templates = [
        "What is the best way to learn {t}?",
        "How do I get a job in {t}?",
        "What are the best {t} books?",
        "Is {t} worth learning?",
        "What tools do {t} professionals use?",
        "How much time does it take to learn {t}?",
        "What are the career prospects in {t}?",
        "Can I learn {t} without a degree?",
        "What is the future of {t}?",
        "How is {t} used in industry?",
        "What are the top {t} frameworks?",
        "What mistakes should I avoid when learning {t}?",
        "How do I practice {t} effectively?",
        "What are some {t} beginner projects?",
        "Where can I find free {t} courses?",
    ]

    rows = []
    qid = 0

    # Duplicate pairs (40%)
    n_dup = int(n_pairs * 0.4)
    for _ in range(n_dup):
        t = random.choice(topics)
        t2 = random.choice([x for x in topics if x != t])
        tmpl = random.choice(dup_templates)
        q1 = tmpl[0].format(t=t, t2=t2)
        q2 = tmpl[1].format(t=t, t2=t2)
        rows.append({"id": len(rows), "qid1": qid, "qid2": qid+1,
                      "question1": q1, "question2": q2, "is_duplicate": 1})
        qid += 2

    # Non-duplicate pairs (60%) — different topics
    n_non = n_pairs - n_dup
    for _ in range(n_non):
        t1, t2 = random.sample(topics, 2)
        tmpl1 = random.choice(non_dup_templates)
        tmpl2 = random.choice(non_dup_templates)
        q1 = tmpl1.format(t=t1)
        q2 = tmpl2.format(t=t2)
        rows.append({"id": len(rows), "qid1": qid, "qid2": qid+1,
                      "question1": q1, "question2": q2, "is_duplicate": 0})
        qid += 2

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df["id"] = range(len(df))
    return df


def load_data():
    if not config.SYNTHETIC_DATA and os.path.exists(config.KAGGLE_CSV_PATH):
        print("[DATA] Loading real Kaggle QQP data...")
        df = pd.read_csv(config.KAGGLE_CSV_PATH)
        df = df.dropna(subset=["question1", "question2", "is_duplicate"])
        df["question1"] = df["question1"].astype(str)
        df["question2"] = df["question2"].astype(str)
        df["is_duplicate"] = df["is_duplicate"].astype(int)
    else:
        print("[DATA] Generating synthetic QQP data...")
        df = generate_synthetic_qqp(n_pairs=config.NUM_SYNTHETIC_PAIRS, seed=config.RANDOM_SEED)
    print(f"[DATA] Total pairs: {len(df)}, Dup ratio: {df['is_duplicate'].mean():.2%}")
    return df


def split_data(df):
    np.random.seed(config.RANDOM_SEED)
    n = len(df)
    idx = np.random.permutation(n)
    t_end = int(n * config.TRAIN_RATIO)
    v_end = t_end + int(n * config.VAL_RATIO)
    train = df.iloc[idx[:t_end]].reset_index(drop=True)
    val = df.iloc[idx[t_end:v_end]].reset_index(drop=True)
    test = df.iloc[idx[v_end:]].reset_index(drop=True)
    print(f"[DATA] Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def build_training_triplets(train_df, corpus_texts, corpus_id_map, relevance,
                            n_hard_negatives=3, use_hard_negatives=True):
    triplets = []
    all_ids = set(range(len(corpus_texts)))
    for _, row in train_df.iterrows():
        if row["is_duplicate"] == 1:
            q, p = row["question1"], row["question2"]
            rel_ids = relevance.get(q, set())
            neg_pool = list(all_ids - rel_ids)
            if not neg_pool:
                continue
            n = min(n_hard_negatives, len(neg_pool))
            for nid in random.sample(neg_pool, n):
                triplets.append({"query": q, "positive": p, "negative": corpus_texts[nid]})
    random.shuffle(triplets)
    print(f"[DATA] Training triplets: {len(triplets)}")
    return triplets
