import os
from pathlib import Path
import pandas as pd

from app.core_paths import ML_RESULTS_DIR

RESULTS_DIR = ML_RESULTS_DIR
SUMMARY_CSV = RESULTS_DIR / 'main_results.csv'


def load_results_summary():
    df = pd.read_csv(SUMMARY_CSV)
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'model'})
    records = df.to_dict(orient='records')
    best_by_ndcg = max(records, key=lambda x: x.get('nDCG@10', 0)) if records else None
    best_by_mrr = max(records, key=lambda x: x.get('MRR@10', 0)) if records else None
    best_by_p1 = max(records, key=lambda x: x.get('P@1', 0)) if records else None
    return {
        'rows': records,
        'highlights': {
            'best_ndcg10': best_by_ndcg,
            'best_mrr10': best_by_mrr,
            'best_p1': best_by_p1,
        },
    }


def chart_title(filename: str) -> str:
    stem = filename.replace('.png', '')
    stem = stem.split('_', 1)[-1] if '_' in stem else stem
    return stem.replace('_', ' ').title()


def list_charts(base_url: str = ''):
    charts = []
    for filename in sorted(os.listdir(RESULTS_DIR)):
        if not filename.endswith('.png'):
            continue
        charts.append({
            'filename': filename,
            'title': chart_title(filename),
            'image_url': f'{base_url}/static/results/{filename}',
        })
    return charts
