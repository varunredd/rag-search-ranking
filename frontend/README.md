# Frontend

React + Vite UI for the ranking dashboard.

## Run locally
```bash
cd frontend
npm install
npm run dev
```

By default Vite proxies `/api` and `/static` to `http://localhost:8000`.
If you deploy the backend elsewhere, set `VITE_API_BASE_URL`.


The live reranker dropdown only shows checkpoint-backed rerankers when matching `.pt` files are present in `ml_core/models/`.
