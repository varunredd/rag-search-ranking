import { useEffect, useState } from "react";
import PageHeader from "../components/PageHeader";
import ResultColumn from "../components/ResultColumn";
import { getJSON, postJSON } from "../lib/api";

const initialForm = {
  query: "How to start a career in data engineering?",
  retriever: "tfidf",
  reranker: "none",
  top_k: 10,
};

export default function DemoPage() {
  const [config, setConfig] = useState(null);
  const [form, setForm] = useState(initialForm);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  useEffect(() => {
    getJSON("/api/config").then(setConfig).catch(console.error);
  }, []);

  async function runSearch(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const data = await postJSON("/api/demo/search", form);
      setResult(data);
    } catch (err) {
      setError(err.message || "Unable to run the demo.");
    } finally {
      setLoading(false);
    }
  }

  function onChange(event) {
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: name === "top_k" ? Number(value) : value }));
  }

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Interactive demo"
        title="Run retrieval and reranking on demand"
        description="Choose a retriever, then optionally apply a real checkpoint-backed reranker when one is available in ml_core/models."
      />

      <section className="grid-2 form-layout">
        <form className="panel form-panel" onSubmit={runSearch}>
          <div className="section-head compact">
            <div>
              <h2>Search controls</h2>
              <p>The live demo uses the real ml_core retrieval pipeline. Reranking is enabled only when trained checkpoints are present in ml_core/models.</p>
            </div>
          </div>

          <label className="field-block">
            <span>Query</span>
            <textarea name="query" rows="4" value={form.query} onChange={onChange} />
          </label>

          <div className="form-row">
            <label className="field-block">
              <span>Retriever</span>
              <select name="retriever" value={form.retriever} onChange={onChange}>
                {config?.retrievers?.map((item) => (
                  <option value={item.value} key={item.value}>{item.label}</option>
                ))}
              </select>
            </label>

            <label className="field-block">
              <span>Reranker</span>
              <select name="reranker" value={form.reranker} onChange={onChange}>
                {config?.rerankers?.map((item) => (
                  <option value={item.value} key={item.value}>{item.label}</option>
                ))}
              </select>
            </label>

            <label className="field-block">
              <span>Top K</span>
              <select name="top_k" value={form.top_k} onChange={onChange}>
                {config?.topKOptions?.map((value) => (
                  <option value={value} key={value}>{value}</option>
                ))}
              </select>
            </label>
          </div>

          <button className="primary-button" type="submit" disabled={loading}>
            {loading ? "Running..." : "Run demo"}
          </button>

          {error ? <p className="error-text">{error}</p> : null}
        </form>

        <div className="panel helper-panel">
          <div className="section-head compact">
            <div>
              <h2>How to read the result</h2>
              <p>Focus on whether relevant candidates move closer to rank 1 after reranking.</p>
            </div>
          </div>
          <ul className="bullet-list">
            <li>Stage 1 retrieval returns the initial candidate list and scores.</li>
            <li>Stage 2 reranking reorders the same candidates using a stronger pairwise scorer.</li>
            <li>Positive rank delta means the item moved up after reranking.</li>
            <li>Ground truth badges show exact duplicates when they exist in the underlying dataset.</li>
          </ul>
          {result ? <p className="callout-note">{result.note}</p> : null}
          {result?.ground_truths?.length ? (
            <div className="truth-box">
              <p className="truth-title">Known ground-truth duplicates for this exact query</p>
              <ul>
                {result.ground_truths.map((item) => <li key={item}>{item}</li>)}
              </ul>
            </div>
          ) : null}
        </div>
      </section>

      {result ? (
        <section className="grid-2 results-grid">
          <ResultColumn
            title="Stage 1 retrieval"
            subtitle={`${result.retriever.toUpperCase()} candidate ranking`}
            items={result.retrieval_results}
          />
          <ResultColumn
            title="Stage 2 reranking"
            subtitle={`${result.reranker} reordered ranking`}
            items={result.reranked_results}
            showMovement
          />
        </section>
      ) : null}
    </div>
  );
}
