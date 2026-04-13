import PageHeader from "../components/PageHeader";

export default function ArchitecturePage() {
  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Architecture"
        title="How the full-stack app is organized"
        description="The UI, API, and ranking engine are cleanly separated so you can evolve the demo without touching the experiment scripts."
      />

      <section className="grid-3">
        <div className="panel architecture-card">
          <p className="chip">Frontend</p>
          <h2>React + Vite</h2>
          <p>Multi-page dashboard with Overview, Live Demo, Experiments, Analysis, and Architecture pages. Fetches JSON from FastAPI and renders saved charts plus live ranking results.</p>
        </div>
        <div className="panel architecture-card">
          <p className="chip">Backend</p>
          <h2>FastAPI service</h2>
          <p>Serves project metadata, reads charts/examples directly from ml_core/results, and exposes a live search route that wraps retrieval plus optional checkpoint-backed reranking.</p>
        </div>
        <div className="panel architecture-card">
          <p className="chip">ML core</p>
          <h2>Reusable ranking modules</h2>
          <p>Includes your config, data prep, retrievers, reranker models, and training helpers so the web layer stays thin and the original project structure remains recognizable.</p>
        </div>
      </section>

      <section className="panel">
        <div className="section-head compact">
          <div>
            <h2>API endpoints</h2>
            <p>These are the routes the frontend uses.</p>
          </div>
        </div>
        <div className="endpoint-list">
          <div className="endpoint-row"><code>GET /api/overview</code><span>Project title, dataset, stack, and pipeline summary.</span></div>
          <div className="endpoint-row"><code>GET /api/config</code><span>Supported retrievers, rerankers, and Top-K options.</span></div>
          <div className="endpoint-row"><code>GET /api/results/summary</code><span>Reads the main results CSV and computes highlights.</span></div>
          <div className="endpoint-row"><code>GET /api/results/charts</code><span>Lists saved experiment chart images from the results folder.</span></div>
          <div className="endpoint-row"><code>GET /api/results/examples</code><span>Parses qualitative examples into structured JSON.</span></div>
          <div className="endpoint-row"><code>POST /api/demo/search</code><span>Runs retrieval and optional reranking for an input query.</span></div>
        </div>
      </section>

      <section className="panel">
        <div className="section-head compact">
          <div>
            <h2>Suggested next upgrades</h2>
            <p>This version is production-looking, but you can push it further.</p>
          </div>
        </div>
        <ul className="bullet-list">
          <li>Swap the proxy reranker for a saved transformer checkpoint when you are ready.</li>
          <li>Add authentication if you want to host private experiments online.</li>
          <li>Persist user queries and demo runs for analytics.</li>
          <li>Add model cards and latency breakdowns from future experiments.</li>
        </ul>
      </section>
    </div>
  );
}
