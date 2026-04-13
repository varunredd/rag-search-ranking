import RankBadge from "./RankBadge";

export default function ResultColumn({ title, subtitle, items, showMovement = false }) {
  return (
    <section className="result-column panel">
      <div className="section-head compact">
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
        <span className="chip">{items?.length || 0} results</span>
      </div>
      <div className="result-list">
        {items?.map((item) => (
          <article key={`${title}-${item.doc_id}`} className={`result-card ${item.is_ground_truth ? "is-ground-truth" : ""}`}>
            <div className="result-topline">
              <div className="rank-pill">#{item.rank}</div>
              <div className="result-meta">
                <span>Score {Number(item.score).toFixed(4)}</span>
                {showMovement ? <RankBadge delta={item.rank_delta} /> : null}
                {item.is_ground_truth ? <span className="truth-chip">Ground truth</span> : null}
              </div>
            </div>
            <p className="result-text">{item.text}</p>
            {showMovement && item.previous_rank ? (
              <p className="muted-line">Previously ranked #{item.previous_rank}</p>
            ) : null}
          </article>
        ))}
      </div>
    </section>
  );
}
