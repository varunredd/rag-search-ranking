from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core_paths import ML_RESULTS_DIR
from app.schemas import SearchRequest
from app.services.results_service import load_results_summary, list_charts
from app.services.parser_service import parse_qualitative_examples
from app.services.demo_service import engine

app = FastAPI(title='Two-Stage Passage Ranking API', version='2.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Serve charts directly from ml_core/results so backend is not a second copy of the ML artifacts.
app.mount('/static/results', StaticFiles(directory=str(ML_RESULTS_DIR)), name='results-static')


@app.get('/api/health')
def health():
    return {'status': 'ok'}


@app.get('/api/config')
def get_config():
    return {
        'retrievers': engine.available_retrievers(),
        'rerankers': engine.available_rerankers(),
        'topKOptions': [5, 10, 20, 50],
        'projectName': 'Modified Cross Encoder for Two-Stage Passage Ranking',
        'usesCheckpoints': engine.has_any_checkpoint(),
    }


@app.get('/api/overview')
def get_overview():
    return {
        'title': 'Modified Cross Encoder for Two-Stage Passage Ranking',
        'subtitle': 'A research demo dashboard for comparing first-stage retrieval against cross-encoder reranking.',
        'dataset': 'Quora Question Pairs (QQP)',
        'summary': 'This app uses the original ml_core project as the single source of truth for retrieval, reranking, metrics, and saved charts.',
        'pipeline': [
            'User query',
            'Stage 1 retrieval (BM25 / TF-IDF / SBERT)',
            'Top-K candidate set',
            'Checkpoint-backed cross-encoder reranking when available',
            'Final ranked output',
        ],
        'stack': {
            'frontend': ['React', 'Vite', 'Custom CSS'],
            'backend': ['FastAPI', 'Python'],
            'ml': ['BM25', 'TF-IDF', 'Sentence-BERT', 'Cross-Encoders'],
        },
        'demoMode': 'Results pages always load from ml_core/results. Live reranking is enabled only when trained checkpoints exist in ml_core/models.',
    }


@app.get('/api/results/summary')
def results_summary():
    return load_results_summary()


@app.get('/api/results/charts')
def results_charts(request: Request):
    base_url = str(request.base_url).rstrip('/')
    return {'charts': list_charts(base_url)}


@app.get('/api/results/examples')
def results_examples():
    return {'examples': parse_qualitative_examples()}


@app.post('/api/demo/search')
def demo_search(payload: SearchRequest):
    try:
        return engine.search(payload.query.strip(), payload.retriever, payload.reranker, payload.top_k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
