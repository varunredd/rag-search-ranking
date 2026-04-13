import { useEffect, useMemo, useState } from "react";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import { getJSON } from "../lib/api";

export default function ExperimentsPage() {
  const [summary, setSummary] = useState(null);
  const [charts, setCharts] = useState([]);

  useEffect(() => {
    Promise.all([getJSON("/api/results/summary"), getJSON("/api/results/charts")])
      .then(([summaryData, chartsData]) => {
        setSummary(summaryData);
        setCharts(chartsData.charts || []);
      })
      .catch(console.error);
  }, []);

  const rows = summary?.rows || [];
  const best = summary?.highlights?.best_ndcg10;
  const runnerUp = useMemo(() => {
    if (!rows.length) return null;
    return [...rows].sort((a, b) => Number(b["nDCG@10"]) - Number(a["nDCG@10"]))[1] || null;
  }, [rows]);

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Experiment dashboard"
        title="Saved evaluation results and ablations"
        description="These panels surface your original comparison charts, summary metrics, and ablation outputs in a cleaner presentation layer."
      />

      <section className="grid-3">
        <StatCard label="Top model" value={best?.model || "Loading"} caption="Highest nDCG@10 among saved runs" />
        <StatCard label="nDCG@10" value={best ? Number(best["nDCG@10"]).toFixed(4) : "--"} caption={runnerUp ? `Runner-up: ${runnerUp.model}` : "Comparing retrievers and rerankers"} />
        <StatCard label="P@1" value={best ? Number(best["P@1"]).toFixed(4) : "--"} caption="Precision at the first result" />
      </section>

      <section className="panel">
        <div className="section-head compact">
          <div>
            <h2>Metric table</h2>
            <p>A compact view of the saved experiment CSV.</p>
          </div>
        </div>
        <div className="table-wrap">
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>nDCG@10</th>
                <th>MRR@10</th>
                <th>P@1</th>
                <th>Recall@20</th>
                <th>MAP@10</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.model}>
                  <td>{row.model}</td>
                  <td>{Number(row["nDCG@10"]).toFixed(4)}</td>
                  <td>{Number(row["MRR@10"]).toFixed(4)}</td>
                  <td>{Number(row["P@1"]).toFixed(4)}</td>
                  <td>{Number(row["Recall@20"]).toFixed(4)}</td>
                  <td>{Number(row["MAP@10"]).toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="chart-grid">
        {charts.map((chart) => (
          <article key={chart.filename} className="panel chart-card">
            <div className="section-head compact">
              <div>
                <h2>{chart.title}</h2>
                <p>{chart.filename}</p>
              </div>
            </div>
            <img src={chart.image_url} alt={chart.title} className="chart-image" />
          </article>
        ))}
      </section>
    </div>
  );
}
