# Backend

FastAPI service for the Two-Stage Passage Ranking dashboard.

## Design
The backend is intentionally thin:
- API routes live in `backend/app`
- ML logic stays in `ml_core`
- Results pages read directly from `ml_core/results`
- Live reranking uses real checkpoints only when `.pt` files exist in `ml_core/models`

## Run locally
```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Endpoints
- `GET /api/health`
- `GET /api/config`
- `GET /api/overview`
- `GET /api/results/summary`
- `GET /api/results/charts`
- `GET /api/results/examples`
- `POST /api/demo/search`

## Notes
- Without checkpoints, the live demo falls back to retrieval-only mode.
- Saved charts and qualitative examples still work because they are read from `ml_core/results`.
- Python 3.11 is recommended for the ML stack.
