import { useEffect, useState } from "react";
import PageHeader from "../components/PageHeader";
import { getJSON } from "../lib/api";

export default function AnalysisPage() {
  const [examples, setExamples] = useState([]);

  useEffect(() => {
    getJSON("/api/results/examples")
      .then((data) => setExamples(data.examples || []))
      .catch(console.error);
  }, []);

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Qualitative analysis"
        title="Where reranking helps most"
        description="These examples explain the value of the reranker in plain ranking terms: which relevant candidate moved upward and by how much."
      />

      <section className="analysis-grid">
        {examples.map((example) => (
          <article key={example.index} className="panel analysis-card">
            <div className="analysis-topline">
              <span className="chip">Query {example.index}</span>
              <span className={`rank-badge ${example.improvement > 0 ? "rank-up" : "rank-neutral"}`}>
                {example.improvement > 0 ? `Improved by ${example.improvement}` : "Already optimal"}
              </span>
            </div>
            <h2>{example.query}</h2>
            <p className="ground-truth-line">Ground truth: {example.ground_truth}</p>
            <p className="analysis-summary">{example.summary}</p>
            <div className="comparison-columns">
              <div>
                <p className="comparison-heading">Stage 1</p>
                <ul className="mini-rank-list">
                  {example.stage1.slice(0, 5).map((item) => (
                    <li key={`s1-${example.index}-${item.rank}`} className={item.is_duplicate ? "highlight-line" : ""}>
                      <span>#{item.rank}</span>
                      <span>{item.text}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <p className="comparison-heading">Stage 2</p>
                <ul className="mini-rank-list">
                  {example.stage2.slice(0, 5).map((item) => (
                    <li key={`s2-${example.index}-${item.rank}`} className={item.is_duplicate ? "highlight-line" : ""}>
                      <span>#{item.rank}</span>
                      <span>{item.text}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </article>
        ))}
      </section>
    </div>
  );
}
