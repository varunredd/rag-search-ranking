import { useEffect, useState } from "react";
import { ArrowRight, Database, Gauge, Layers3, Sparkles } from "lucide-react";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import { getJSON } from "../lib/api";

export default function HomePage() {
  const [overview, setOverview] = useState(null);
  const [summary, setSummary] = useState(null);

  useEffect(() => {
    Promise.all([getJSON("/api/overview"), getJSON("/api/results/summary")])
      .then(([overviewData, summaryData]) => {
        setOverview(overviewData);
        setSummary(summaryData);
      })
      .catch(console.error);
  }, []);

  const best = summary?.highlights?.best_ndcg10;

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Research demo"
        title="Modified Cross Encoder for Two-Stage Passage Ranking"
        description="A full-stack dashboard for exploring retrieval, reranking, experiment results, and qualitative ranking improvements in one place."
      >
        <div className="hero-card">
          <p className="hero-card-label">Best nDCG@10</p>
          <h3>{best?.model || "Loading..."}</h3>
          <p className="hero-card-value">{best ? Number(best["nDCG@10"]).toFixed(4) : "--"}</p>
        </div>
      </PageHeader>

      <section className="grid-4">
        <StatCard label="Dataset" value={overview?.dataset || "QQP"} caption="Query pair dataset used for ranking experiments" />
        <StatCard label="Best model" value={best?.model || "Loading"} caption="Top performer by nDCG@10 from saved experiment runs" />
        <StatCard label="nDCG@10" value={best ? Number(best["nDCG@10"]).toFixed(4) : "--"} caption="Higher means better ranking quality at top positions" />
        <StatCard label="MRR@10" value={best ? Number(best["MRR@10"]).toFixed(4) : "--"} caption="Measures how early the first relevant result appears" />
      </section>

      <section className="panel">
        <div className="section-head">
          <div>
            <h2>Why this interface matters</h2>
            <p>{overview?.summary}</p>
          </div>
        </div>
        <div className="pipeline-grid">
          {(overview?.pipeline || []).map((step, index) => (
            <div key={step} className="pipeline-step">
              <div className="step-index">0{index + 1}</div>
              <p>{step}</p>
              {index < (overview?.pipeline?.length || 0) - 1 ? <ArrowRight size={18} className="step-arrow" /> : null}
            </div>
          ))}
        </div>
      </section>

      <section className="grid-2">
        <div className="panel info-panel">
          <div className="section-head compact">
            <div>
              <h2>System highlights</h2>
              <p>Built to present both the engineering pipeline and the research outcomes clearly.</p>
            </div>
          </div>
          <div className="feature-list">
            <div className="feature-item"><Layers3 size={18} /><span>Compare stage-1 retrieval with stage-2 reranking side by side.</span></div>
            <div className="feature-item"><Database size={18} /><span>Serve saved charts, metrics, and qualitative examples from the original project outputs.</span></div>
            <div className="feature-item"><Gauge size={18} /><span>Run live search with cached retrievers and checkpoint-backed reranking when available.</span></div>
            <div className="feature-item"><Sparkles size={18} /><span>Use a polished UI that feels like a real research product, not a raw notebook.</span></div>
          </div>
        </div>

        <div className="panel info-panel gradient-panel">
          <div className="section-head compact">
            <div>
              <h2>Tech stack</h2>
              <p>The app is split into a clean frontend and a Python inference/data API.</p>
            </div>
          </div>
          <div className="stack-columns">
            {overview?.stack
              ? Object.entries(overview.stack).map(([group, tools]) => (
                  <div key={group}>
                    <p className="stack-group">{group}</p>
                    <ul>
                      {tools.map((tool) => (
                        <li key={tool}>{tool}</li>
                      ))}
                    </ul>
                  </div>
                ))
              : null}
          </div>
        </div>
      </section>
    </div>
  );
}
