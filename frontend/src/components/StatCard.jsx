export default function StatCard({ label, value, caption }) {
  return (
    <div className="stat-card">
      <p className="stat-label">{label}</p>
      <h3>{value}</h3>
      {caption ? <p className="stat-caption">{caption}</p> : null}
    </div>
  );
}
